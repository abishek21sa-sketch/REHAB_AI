# REHAB AI v0.3.0 — Portfolio Demonstration Guide

## Demonstration objective

Show a complete rehabilitation decision-intelligence chain, not a pose demo or dashboard. Start the API/UI, choose a synthetic impairment level and run the case. Explain that the same service validates sensing, reconstructs a biomechanical patient state, updates a persistent digital twin, estimates multiple risks, simulates interventions, applies clinical constraints, evaluates trade-offs, forecasts the selected intervention and emits an auditable clinician-review recommendation.

## Suggested walkthrough

1. Show `/health` and OpenAPI to establish the production service boundary.
2. Run the synthetic patient from the frontend.
3. Point to gait speed and functional capacity as reconstructed patient state, not direct labels.
4. Show multimodal fusion confidence and explain cross-modal consistency between wearable and pose-derived stability signals.
5. Show fall, gait-deterioration, fatigue and adherence risk as distinct time-horizon decision inputs.
6. Show the selected therapy dose and scenario outcome, then explain the feasibility constraints that prevent unsafe intensity/session choices.
7. Show the digital-twin trajectory for the selected plan and explain that intervention alternatives are evaluated before recommendation.
8. Mention Pareto-efficient alternatives to demonstrate multi-objective trade-off reasoning rather than one opaque scalar score.
9. Show the decision ID and audit record to demonstrate reproducibility/provenance.
10. Call the scheduling endpoint to show the Industrial Engineering/Operations Research layer for therapist capacity and patient priority.

## Claims you can make

- The repository implements an end-to-end, production-oriented rehabilitation decision-intelligence architecture.
- Predictions are connected to simulation, clinical constraints, optimization and an explainable recommendation.
- The platform has a longitudinal patient digital twin, multimodal sensing/fusion, therapy optimization and resource scheduling.
- Automated tests validate engineering invariants and synthetic behavior.

## Claims you must not make

Do not say the system is clinically validated, FDA cleared, safe for autonomous treatment, proven to reduce falls, proven to accelerate recovery, or trained on a real patient cohort. The current models are transparent engineering baselines intentionally kept separate from future dataset-backed clinical validation.
