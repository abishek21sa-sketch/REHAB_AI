# Frontend Demo Flow — REHAB AI

This product keeps its own visual language: **human-motion biomechanics studio**. The shared contract is behavioral evidence, not a shared layout or theme.

## Native entrypoint

`src/rehab_ai/web/frontend/index.html`

## Project-specific demo sequence

1. load observed motion session
2. estimate latent recovery state
3. select safe bounded load
4. review counterfactual progression
5. clinician approves or holds

## Evidence requirements

The screen must show the project-native inputs, objective, constraints, baseline/counterfactual, evidence class, signature decision, and human approval/hold state. The product must not imply autonomous actuation.

## API evidence surface

The read-only signature evidence endpoint is `/api/governance/signature`. Its response is linked to `artifacts/fortune50_capability_benchmark.json` and exposes the current decision, baseline, sensitivity/counterfactual evidence, and human-gated status.
