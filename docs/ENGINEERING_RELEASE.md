# REHAB AI — Production Readiness

Release: `REHAB_AI_FORTUNE50_EMPIRICAL_RELEASE`

## What changed

This release adds a provenance-aware empirical backbone, a live empirical API, project-native historical/entity diagnostics, a named empirical case study, external-source refresh/promotion workflow, live empirical charts, and a 26+ workspace contract in which each workspace has a distinct method/evidence/action definition.

## Current evidence mode

- Source: **Smartphone-Based Recognition of Human Activities and Postural Transitions**
- Source URL: https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions
- Mode: **offline_reference**
- Promotion state: **REFERENCE_ONLY**
- Local analyzable evidence: **201 rows / 5 fields**
- Workspaces: **26**

## Domain diagnostic

**bilateral movement asymmetry and progression-risk evidence**

Reference metrics:
```json
{
  "frames": 201,
  "baseline_mean_knee_asym_deg": 20.369,
  "baseline_p95_knee_asym_deg": 34.662,
  "asymmetric_mean_knee_asym_deg": 22.356,
  "asymmetry_delta_deg": 1.987,
  "baseline_mean_heel_y_gap": 0.04197
}
```

Decision signal: HMM state estimation and MOTION-GUARD should treat sustained bilateral asymmetry as evidence against automatic progression before APACE/twin review.

## Analytical chain

source provenance → schema/data-quality checks → entity/factor drilldown → cohort/history comparison → diagnostic ranking → predictive model → original algorithm → OR/simulation escalation → counterfactual challenge → human decision

## Windows gates

Core/offline acceptance:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows_EMPIRICAL_acceptance.ps1
```

External-data promotion (internet required):
```powershell
.\scripts\windows_external_data_promotion.ps1
```

## Claim boundary

External-source results are claimed only when data_mode is refreshed_external or published_external_snapshot; offline_reference remains reference evidence.

The label “production-ready” is an internal portfolio-depth target relative to the latest observable portfolio evidence, not an external company certification and not a claim that reference/synthetic data is real production evidence.
