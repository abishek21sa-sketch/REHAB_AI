# Validation Strategy — v0.3.0

Validation is separated into engineering evidence and future clinical evidence.

## Automated engineering evidence

The release suite verifies:

- input contract and temporal monotonicity;
- synthetic sample-rate and physical feature bounds;
- impairment-to-functional-capacity and impairment-to-risk monotonicity;
- multimodal fusion boundedness and confidence;
- fall, deterioration, fatigue and adherence risk bounds;
- deterministic seeded simulation;
- therapy-plan clinical feasibility constraints;
- Pareto dominance behavior;
- intervention-conditioned digital-twin trajectory bounds;
- state persistence and longitudinal deltas;
- rehabilitation scheduling capacity and prioritization;
- decision-audit digest determinism;
- clinician-feedback and observability paths;
- API health, decision, scheduling and expanded v0.2 payload integration;
- required governance/deployment assets.

## Release gates

A portfolio release requires all automated tests to pass, Python compilation to succeed, the validation report to regenerate, package build to succeed and an in-process API smoke test to execute the real decision chain.

## Clinical-validation work package

When an adjudicated dataset is supplied, the repository must add dataset lineage, cohort definitions, leakage controls, training/validation/test separation, calibration curves, discrimination metrics where appropriate, MAE/RMSE for continuous outcomes, subgroup analyses, robustness and missingness tests, temporal/external validation, clinician review protocol, sensitivity analysis and explicit failure-mode analysis.

Synthetic evidence must never be substituted for this work package.
