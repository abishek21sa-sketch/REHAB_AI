"""SAFE-MPC bounded progression reference contract."""


def choose_safe_load(state, target, loads, *, lower_state, upper_state, max_step, burden_weight=0.2):
    if lower_state > upper_state or not loads:
        raise ValueError("invalid state envelope or empty action set")
    feasible = []
    for load in loads:
        load = float(load)
        if load < 0 or load > max_step:
            continue
        next_state = float(state) + 0.6 * load - 0.1 * float(state)
        if lower_state <= next_state <= upper_state:
            objective = (next_state - target) ** 2 + burden_weight * load ** 2
            feasible.append((objective, load, next_state))
    return {"load": min(feasible)[1], "next_state": min(feasible)[2], "objective": min(feasible)[0]} if feasible else None


def ablation(state, target, loads, **kwargs):
    """Remove the envelope gate for a declared nominal-gain ablation."""
    kwargs["lower_state"] = float("-inf")
    kwargs["upper_state"] = float("inf")
    return choose_safe_load(state, target, loads, **kwargs)


def sensitivity(state, target, loads, margin_delta, **kwargs):
    kwargs["lower_state"] = kwargs["lower_state"] + margin_delta
    kwargs["upper_state"] = kwargs["upper_state"] - margin_delta
    if kwargs["lower_state"] > kwargs["upper_state"]:
        return None
    return choose_safe_load(state, target, loads, **kwargs)
