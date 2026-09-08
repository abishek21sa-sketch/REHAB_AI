# Model Cards — REHAB AI v0.3.0

All current prediction functions are transparent engineering baselines. They are deliberately versioned and bounded but have **not** been fitted, calibrated or externally validated on an adjudicated clinical cohort.

## Fall risk — `transparent-fall-risk-v1`
Inputs include gait speed/variability, sway, knee asymmetry, trunk instability, fatigue and functional reserve. Output is a six-week risk index with heuristic uncertainty interval. Intended use: engineering demonstration and integration testing only.

## Recovery — `transparent-recovery-v1`
Inputs include functional reserve, therapy dose/intensity/balance share, adherence, fatigue and pain. Output is expected six-week functional-capacity gain plus interval. Intended use: scenario-ranking baseline only.

## Temporal risk — `transparent-trajectory-v2`
Produces gait deterioration, nonadherence and fatigue-exacerbation risk indices from current state and, for deterioration, longitudinal twin deltas. Intended use: demonstrate multi-horizon risk-to-decision integration.

## Prohibited claims
No clinical sensitivity, specificity, AUROC, calibration, causal treatment effect, outcome improvement, safety, fairness, subgroup performance or regulatory claim may be inferred from synthetic test results. Those metrics become meaningful only after a governed real-data validation work package is executed.
