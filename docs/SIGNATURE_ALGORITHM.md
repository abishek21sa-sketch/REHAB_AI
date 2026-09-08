# SAFE-MPC — Signature Algorithm Contract

This document is the project-native mathematical center required by the portfolio governance pack. The implementation alias is **MOTION-GUARD/APACE bounded progression**.

## Operational decision

The module makes one operational decision: **choose a rehabilitation load that progresses toward a target while respecting a state safety envelope**.

## Mathematical center

- **Decision variables:** continuous state estimate and bounded candidate load; next state follows the declared transition equation.
- **Objective:** minimize target-tracking error plus a burden penalty.
- **Constraints and release gates:** lower_state <= next_state <= upper_state; 0 <= load <= max_step.
- **Determinism:** the reference contract is deterministic for a fixed candidate set, scenario, and seed.
- **Solver status:** the current reference is an executable enumerative/closed-form contract; production solver integration remains downstream of this gate.

## Baseline and counterfactual

The named baseline is **maximum nominal gain or fixed progression without an explicit envelope**. The counterfactual is evaluated on the same inputs and scenario so that a claimed improvement cannot be caused by a changed data slice.

## Ablation

The declared ablation is to **remove the state envelope while retaining load bounds**. It is executable through the module's `ablation(...)` function and is covered by the signature tests.

## Sensitivity

The sensitivity sweep is: **tighten the envelope margin and report selected load or controlled no-feasible-action outcome**. Sensitivity output is evidence about robustness, not a claim of causal production impact.

## Evidence classes and authority

Evidence is kept separate as observed, simulated, optimized, shadow-mode, and realized. **observed human-activity data for context; simulated state-transition scenarios; clinician authority and shadow-mode boundary remain required** A human authority remains required before any operational action; autonomous execution is disabled.

## Implementation and acceptance

- Implementation: `src/rehab_ai/control/signature_algorithm.py`
- Windows acceptance test: `tests/test_signature_algorithm.py`
- Required acceptance result: `4 tests, OK`, with invalid inputs and no-feasible cases controlled explicitly.

## Release boundary

This signature is release-ready only when this contract, the research-validation protocol, the machine-readable governance artifact, the existing Airlines 1.5x gates, and the final integrity/hash checks all pass together.
