from __future__ import annotations

from dataclasses import dataclass, asdict
from math import log
from typing import Callable, Iterable, Sequence
import numpy as np

from rehab_ai.control.stochastic_mpc import RehabAction, default_actions
from rehab_ai.ie.human_performance import HumanPerformanceState, safe_load_limit, transition_human_performance


@dataclass(frozen=True)
class APACEConfig:
    horizon: int = 4
    beam_width: int = 18
    rollouts: int = 96
    alpha_cvar: float = 0.90
    discount: float = 0.96
    base_information_weight: float = 0.20
    risk_weight: float = 0.75
    burden_weight: float = 0.18
    fatigue_limit: float = 0.82
    pain_limit: float = 0.78
    min_stability_margin_m: float = 0.005
    seed: int = 42
    enable_dominance_pruning: bool = True


@dataclass(frozen=True)
class BeliefState:
    performance: HumanPerformanceState
    stability_margin_m: float
    parameter_mean: float = 1.0
    parameter_std: float = 0.20
    state_uncertainty: float = 0.12
    adherence: float = 0.90
    recent_overload: float = 0.0
    history: tuple[tuple[float, float, float, float, float], ...] = tuple()


@dataclass(frozen=True)
class APACEStep:
    week: int
    action: str
    expected_gain: float
    information_gain: float
    cvar_loss: float
    burden: float
    score: float
    capacity: float
    fatigue: float
    pain: float
    parameter_std: float


@dataclass(frozen=True)
class APACESolution:
    status: str
    sequence: tuple[str, ...]
    score: float
    trajectory: tuple[APACEStep, ...]
    expanded_nodes: int
    pruned_unsafe: int
    pruned_dominated: int
    beam_width: int
    method: str = "APACE uncertainty-aware safe dual-control beam search"

    def to_dict(self) -> dict:
        return asdict(self)


def _normal_entropy(std: float) -> float:
    s = max(1e-9, float(std))
    return 0.5 * log(2 * np.pi * np.e * s * s)


def _posterior_std(prior_std: float, observation_std: float) -> float:
    p = max(1e-9, prior_std) ** 2
    r = max(1e-9, observation_std) ** 2
    return float(np.sqrt(1.0 / (1.0 / p + 1.0 / r)))


def _action_burden(action: RehabAction) -> float:
    return float(0.45 * action.intensity + 0.35 * action.duration_min / 60.0 + 0.20 * action.instability_demand)


def _safe_action(belief: BeliefState, action: RehabAction, config: APACEConfig, safe_limit_predictor=None) -> bool:
    load = action.as_load().functional_load
    limit = float(safe_limit_predictor(belief, action)) if safe_limit_predictor is not None else safe_load_limit(belief.performance.capacity, belief.performance.pain, belief.stability_margin_m)
    uncertainty_buffer = 1.0 + 1.15 * belief.state_uncertainty + 0.80 * belief.parameter_std
    return (
        load * uncertainty_buffer <= limit
        and belief.stability_margin_m >= config.min_stability_margin_m
    )


def _rollout_action(
    belief: BeliefState,
    action: RehabAction,
    response_predictor: Callable[[HumanPerformanceState, RehabAction], tuple[float, float]],
    config: APACEConfig,
    rng: np.random.Generator,
    temporal_capacity_predictor=None,
) -> tuple[BeliefState, dict]:
    mean_gain, model_std = response_predictor(belief.performance, action)
    model_std = max(0.004, float(model_std))
    n = max(16, config.rollouts)
    patient_response = rng.normal(belief.parameter_mean, belief.parameter_std, n)
    stochastic_gain = rng.normal(mean_gain * patient_response, model_std, n)

    base_next = transition_human_performance(
        belief.performance,
        action.as_load(),
        belief.stability_margin_m,
    )
    cap = np.clip(base_next.capacity + stochastic_gain, 0.0, 1.0)
    temporal_capacity = None
    temporal_std = None
    if temporal_capacity_predictor is not None:
        temporal_capacity, temporal_std = temporal_capacity_predictor(belief, action, base_next)
        temporal_std = max(0.004, float(temporal_std))
        # Precision-weighted fusion of action-response rollout and longitudinal forecast.
        gp_var = max(1e-8, float(np.var(cap)) + model_std**2)
        temp_var = temporal_std**2
        fused_mean = (float(np.mean(cap))/gp_var + float(temporal_capacity)/temp_var) / (1.0/gp_var + 1.0/temp_var)
        cap = np.clip(cap + (fused_mean - float(np.mean(cap))), 0.0, 1.0)
    fatigue = np.clip(rng.normal(base_next.fatigue, 0.025 + 0.04 * belief.state_uncertainty, n), 0.0, 1.0)
    pain = np.clip(rng.normal(base_next.pain, 0.020 + 0.03 * belief.state_uncertainty, n), 0.0, 1.0)

    loss = (1.0 - cap) + 0.65 * fatigue + 0.55 * pain
    tail_n = max(1, int(np.ceil((1.0 - config.alpha_cvar) * n)))
    cvar = float(np.mean(np.sort(loss)[-tail_n:]))
    expected_gain = float(np.mean(cap) - belief.performance.capacity)

    # Approximate expected information gain about patient-specific response gain.
    # Higher model precision and larger informative dose produce stronger posterior contraction.
    observation_std = model_std / max(0.25, action.intensity + 0.35 * action.instability_demand)
    post_std = _posterior_std(belief.parameter_std, observation_std)
    information_gain = max(0.0, _normal_entropy(belief.parameter_std) - _normal_entropy(post_std))

    next_belief = BeliefState(
        performance=HumanPerformanceState(
            capacity=float(np.mean(cap)),
            fatigue=float(np.mean(fatigue)),
            pain=float(np.mean(pain)),
            cumulative_dose=base_next.cumulative_dose,
        ),
        stability_margin_m=max(0.0, belief.stability_margin_m + 0.010 * expected_gain - 0.006 * float(np.mean(fatigue))),
        parameter_mean=belief.parameter_mean,
        parameter_std=max(0.015, post_std),
        state_uncertainty=max(0.025, 0.86 * belief.state_uncertainty + 0.22 * model_std),
        adherence=belief.adherence,
        recent_overload=float(np.clip(0.72*belief.recent_overload + 0.28*max(0.0, action.as_load().functional_load / max(0.05, safe_load_limit(belief.performance.capacity, belief.performance.pain, belief.stability_margin_m)) - 1.0), 0, 1)),
        history=(belief.history + ((belief.performance.capacity, belief.performance.fatigue, belief.performance.pain, belief.adherence, action.as_load().functional_load),))[-4:],
    )
    return next_belief, {
        "expected_gain": expected_gain,
        "information_gain": information_gain,
        "cvar_loss": cvar,
        "burden": _action_burden(action),
        "temporal_capacity": temporal_capacity,
        "temporal_std": temporal_std,
    }


def _dominates(a: dict, b: dict) -> bool:
    # a dominates b if no worse on all relevant dimensions and strictly better on at least one.
    av = (a["expected_gain"], a["information_gain"], -a["cvar_loss"], -a["burden"])
    bv = (b["expected_gain"], b["information_gain"], -b["cvar_loss"], -b["burden"])
    return all(x >= y - 1e-12 for x, y in zip(av, bv)) and any(x > y + 1e-12 for x, y in zip(av, bv))


def solve_apace(
    initial: BeliefState,
    response_predictor: Callable[[HumanPerformanceState, RehabAction], tuple[float, float]],
    config: APACEConfig | None = None,
    actions: Sequence[RehabAction] | None = None,
    safe_limit_predictor=None,
    temporal_capacity_predictor=None,
) -> APACESolution:
    cfg = config or APACEConfig()
    if not 1 <= cfg.horizon <= 8:
        raise ValueError("APACE horizon must be between 1 and 8")
    acts = tuple(actions or default_actions())
    # Branch-level seeds make a candidate rollout invariant to beam ordering.

    # Each beam node: cumulative score, belief, steps, sequence.
    beam: list[tuple[float, BeliefState, tuple[APACEStep, ...], tuple[str, ...]]] = [(0.0, initial, tuple(), tuple())]
    expanded = unsafe = dominated = 0

    initial_unc = max(1e-9, initial.parameter_std + initial.state_uncertainty)
    for week in range(1, cfg.horizon + 1):
        candidates: list[tuple[float, BeliefState, tuple[APACEStep, ...], tuple[str, ...], dict]] = []
        for cumulative, belief, steps, seq in beam:
            local: list[tuple[RehabAction, BeliefState, dict]] = []
            for action in acts:
                expanded += 1
                if not _safe_action(belief, action, cfg, safe_limit_predictor):
                    unsafe += 1
                    continue
                import zlib
                path_key = ">".join(seq + (action.name,)).encode("utf-8")
                branch_seed = (cfg.seed + zlib.crc32(path_key)) % (2**32 - 1)
                branch_rng = np.random.default_rng(branch_seed)
                nxt, metrics = _rollout_action(belief, action, response_predictor, cfg, branch_rng, temporal_capacity_predictor)
                if nxt.performance.fatigue > cfg.fatigue_limit or nxt.performance.pain > cfg.pain_limit:
                    unsafe += 1
                    continue
                local.append((action, nxt, metrics))

            keep = [True] * len(local)
            if cfg.enable_dominance_pruning:
                for i in range(len(local)):
                    for j in range(len(local)):
                        if i != j and _dominates(local[j][2], local[i][2]):
                            keep[i] = False
                            dominated += 1
                            break

            for flag, (action, nxt, metrics) in zip(keep, local):
                if not flag:
                    continue
                remaining_unc = belief.parameter_std + belief.state_uncertainty
                exploration_weight = cfg.base_information_weight * min(1.5, remaining_unc / initial_unc)
                stage_score = (
                    metrics["expected_gain"]
                    + exploration_weight * metrics["information_gain"]
                    - cfg.risk_weight * metrics["cvar_loss"]
                    - cfg.burden_weight * metrics["burden"]
                )
                discounted = (cfg.discount ** (week - 1)) * stage_score
                total = cumulative + discounted
                step = APACEStep(
                    week=week,
                    action=action.name,
                    expected_gain=metrics["expected_gain"],
                    information_gain=metrics["information_gain"],
                    cvar_loss=metrics["cvar_loss"],
                    burden=metrics["burden"],
                    score=stage_score,
                    capacity=nxt.performance.capacity,
                    fatigue=nxt.performance.fatigue,
                    pain=nxt.performance.pain,
                    parameter_std=nxt.parameter_std,
                )
                candidates.append((total, nxt, steps + (step,), seq + (action.name,), metrics))

        if not candidates:
            return APACESolution("INFEASIBLE", tuple(), float("-inf"), tuple(), expanded, unsafe, dominated, cfg.beam_width)

        # Diversity-aware beam: top score plus representatives for gain, low risk, and information.
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected: list[tuple[float, BeliefState, tuple[APACEStep, ...], tuple[str, ...], dict]] = candidates[: max(1, cfg.beam_width - 3)]
        specials = [
            max(candidates, key=lambda x: x[4]["expected_gain"]),
            min(candidates, key=lambda x: x[4]["cvar_loss"]),
            max(candidates, key=lambda x: x[4]["information_gain"]),
        ]
        seen = {x[3] for x in selected}
        for item in specials:
            if item[3] not in seen and len(selected) < cfg.beam_width:
                selected.append(item); seen.add(item[3])
        beam = [(x[0], x[1], x[2], x[3]) for x in selected[: cfg.beam_width]]

    best = max(beam, key=lambda x: x[0])
    return APACESolution("OPTIMAL_BEAM_APPROXIMATION", best[3], float(best[0]), best[2], expanded, unsafe, dominated, cfg.beam_width)


def exact_apace_small(
    initial: BeliefState,
    response_predictor: Callable[[HumanPerformanceState, RehabAction], tuple[float, float]],
    config: APACEConfig,
    actions: Sequence[RehabAction] | None = None,
    safe_limit_predictor=None,
    temporal_capacity_predictor=None,
) -> APACESolution:
    """Exact oracle for tiny deterministic-seed instances by using a beam wider than full tree."""
    acts = tuple(actions or default_actions())
    full_width = len(acts) ** config.horizon
    cfg = APACEConfig(**{**asdict(config), "beam_width": full_width, "enable_dominance_pruning": False})
    return solve_apace(initial, response_predictor, cfg, acts, safe_limit_predictor=safe_limit_predictor, temporal_capacity_predictor=temporal_capacity_predictor)


def evaluate_apace_sequence(
    initial: BeliefState,
    response_predictor: Callable[[HumanPerformanceState, RehabAction], tuple[float, float]],
    sequence: Sequence[RehabAction],
    config: APACEConfig,
    safe_limit_predictor=None,
    temporal_capacity_predictor=None,
) -> APACESolution:
    """Evaluate a fixed action sequence under one common APACE objective.

    This is used to compare externally selected baseline policies without comparing
    scores produced under different objective weights.
    """
    belief=initial; cumulative=0.0; steps=[]; unsafe=0
    initial_unc=max(1e-9,initial.parameter_std+initial.state_uncertainty)
    names=tuple(a.name for a in sequence)
    import zlib
    for week,action in enumerate(sequence,1):
        if not _safe_action(belief,action,config,safe_limit_predictor):
            return APACESolution('INFEASIBLE',names,float('-inf'),tuple(steps),week,unsafe+1,0,1,'APACE fixed-sequence evaluation')
        path_key='>'.join(names[:week]).encode('utf-8')
        branch_seed=(config.seed+zlib.crc32(path_key))%(2**32-1)
        nxt,metrics=_rollout_action(belief,action,response_predictor,config,np.random.default_rng(branch_seed),temporal_capacity_predictor)
        if nxt.performance.fatigue>config.fatigue_limit or nxt.performance.pain>config.pain_limit:
            return APACESolution('INFEASIBLE',names,float('-inf'),tuple(steps),week,unsafe+1,0,1,'APACE fixed-sequence evaluation')
        remaining_unc=belief.parameter_std+belief.state_uncertainty
        exploration_weight=config.base_information_weight*min(1.5,remaining_unc/initial_unc)
        stage=(metrics['expected_gain']+exploration_weight*metrics['information_gain']-config.risk_weight*metrics['cvar_loss']-config.burden_weight*metrics['burden'])
        cumulative+=(config.discount**(week-1))*stage
        steps.append(APACEStep(week,action.name,metrics['expected_gain'],metrics['information_gain'],metrics['cvar_loss'],metrics['burden'],stage,nxt.performance.capacity,nxt.performance.fatigue,nxt.performance.pain,nxt.parameter_std))
        belief=nxt
    return APACESolution('EVALUATED',names,float(cumulative),tuple(steps),len(sequence),unsafe,0,1,'APACE fixed-sequence evaluation')
