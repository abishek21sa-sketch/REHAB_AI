# Rehab AI

## Production readiness

Rehab AI includes a live empirical and historical analysis layer, project-native domain diagnostics, external-source provenance, and AI decisions grounded in explicit evidence. See `docs/ENGINEERING_RELEASE.md`.

- Repository-authored algorithm: **MOTION-GUARD-v1**
- Unique predictive-learning family: **Gaussian hidden-state sequence learning (HMM)**
- Analytical AI role: **AI Rehabilitation Planner**
- TENX workspaces: **20**
- Operational authority: **human-gated; autonomous execution blocked**

### Test the TENX layer on Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows_tenx_acceptance.ps1
.\scripts\start_tenx_workstation.ps1
```

The first command validates prediction → decision → counterfactual → OR escalation → user-aid behavior and a five-seed originality stress suite. The second opens the dedicated analytical workstation.

> **Evidence boundary:** TENX bundled metrics are synthetic/reference validation, not field deployment validation. Existing native Windows, Julia/Go/Rust/frontend, external-data, clinical, or production gates remain applicable where documented.

---


## Portfolio release — APACE safety and policy validation

The release candidate adds a portfolio validation gate that reports APACE objective regret, unsafe rate, final capacity, cumulative tail loss and remaining uncertainty separately. The validator explicitly refuses a universal-dominance claim and labels all current policy evidence as synthetic with external clinical validation pending.

# REHAB AI — V0.95 Full Rehabilitation Intelligence Platform — Adaptive Rehabilitation Decision Intelligence

REHAB AI is a research-grade rehabilitation decision-intelligence system built around a patient-specific digital twin. It reconstructs movement state from multimodal evidence, estimates hidden motor capacity under uncertainty, learns treatment-response behavior, and computes a safe adaptive rehabilitation policy.

> **Evidence boundary:** the included ML benchmarks and patient scenarios are synthetic validation. This repository is not clinically validated and must not be used for autonomous clinical care.

## Distinctive decision problem

The operational decision is not “what dashboard metric is high?” It is: **which intervention should be selected next when therapy both changes the patient and teaches us how that particular patient responds?**

The V0.6 computational chain is:

`IMU / pose / assessment → biomechanics → UKF latent state → phenotype ML → GP treatment-response distribution → human-performance load model → APACE dual-control search → counterfactual trajectory → clinician review`

## Signature capability — Recovery Policy Studio

The Vue 3 interface is organized as a recovery-state workspace rather than a KPI dashboard. It shows a patient recovery topology, biomechanical functional envelope, counterfactual recovery trajectory, and the APACE-selected intervention sequence with expected gain, information value, CVaR tail loss, and uncertainty contraction.

## Artificial intelligence

- **Unscented Kalman Filter:** nonlinear latent state estimation with covariance.
- **Gaussian-process treatment-response model:** predicts capacity response and predictive uncertainty; validated against a mean-response baseline on a deterministic synthetic holdout.
- **Rehabilitation phenotype model:** K-means latent-response archetypes with silhouette diagnostics on synthetic populations.
- Existing calibrated fall-risk modeling remains a secondary safety capability rather than the project centerpiece.

## Industrial Engineering / Rehabilitation Engineering

The human-performance subsystem models patient-specific capacity, therapeutic load, fatigue carry-over, recovery and pain. Biomechanics include bilateral symmetry and dynamic stability using extrapolated center of mass and Margin of Stability.

## Operations Research / Algorithmic contribution

### APACE — Adaptive Patient-Aware Control and Exploration

APACE is a rehabilitation-specific risk-sensitive dual-control search algorithm. A candidate intervention receives value for expected recovery and information gained about the patient's response dynamics, while CVaR tail risk and treatment burden are penalized. Unsafe candidates are rejected before exploration value is considered.

For action `a` at belief `b`:

`score(a|b) = E[gain] + λ(b)·IG(a) − β·CVaRα(loss) − η·burden(a)`

`λ(b)` decreases as patient-response and state uncertainty contract. Search uses biomechanical safety pruning, Pareto-style dominance pruning, diversity-preserving beam expansion, seeded digital-twin rollouts and receding-horizon re-optimization. Tiny instances have an exact exhaustive oracle for verification.

## Data engineering

High-frequency sensing sessions use **HDF5** with compressed channel datasets and metadata. This is deliberately different from treating dense IMU/pose streams as ordinary application rows. Existing canonical validation contracts remain available for batch/session input.

## Product architecture

- **Scientific / decision core:** Python 3.11+
- **Web service:** Starlette ASGI (thin transport layer, no scientific logic in endpoints)
- **Frontend:** embedded Vue 3.5 runtime + custom SVG clinical visualization
- **Sensor archive:** HDF5 via h5py
- **Numerical/ML:** NumPy, SciPy, scikit-learn
- **Local server:** Uvicorn
- **Primary local target:** Windows + PowerShell

The embedded Vue runtime means ordinary use does not require npm/Node after extraction.

## Run on Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_windows.ps1
```

The script creates `.venv`, installs Python dependencies, runs the full tests, runs APACE diagnostics, opens the browser, and starts the application at `http://127.0.0.1:8010`.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\phase_apace_diagnostics.py
```

## Validation status

- Computational biomechanics: **TESTED on reference/synthetic cases**
- UKF state estimator: **TESTED on synthetic/reference cases**
- GP treatment response: **VALIDATED ON SYNTHETIC HOLDOUT**
- Phenotype clustering: **VALIDATED ON SYNTHETIC POPULATION**
- Human-performance dynamics: **TESTED against mathematical invariants**
- APACE safety/search logic: **TESTED; small-instance exact oracle included**
- Vue frontend source/runtime integration: **LOCAL RUNTIME INCLUDED**
- External clinical validity: **PENDING**

## Repository status

V0.6 is the **computational-identity build**, not public V1.0. Remaining work focuses on richer movement reconstruction/video ingestion, deeper patient-dynamics learning, expanded algorithm benchmarking, product workflow hardening, real/public data adapters where defensible, and final public-release validation.

## V0.8 computational depth
V0.8 adds two new ML layers and deeper biomechanics while preserving the accepted Digital Patient Bay visual identity. A time-aware longitudinal recovery model is benchmarked against persistence, and a split-conformal patient-specific safe-envelope learner estimates a conservative therapy-load boundary. Joint power/work and IMU periodicity/smoothness analytics deepen the biomechanics/sensing layer. All reported validation remains explicitly synthetic until external rehabilitation data are supplied.

## V0.9 — Integrated Adaptive Rehabilitation System

V0.9 couples the previously validated ML components directly into the decision loop. The split-conformal safe-envelope learner now defines APACE feasibility, the longitudinal temporal model participates in future capacity rollouts, and a longitudinal patient episode engine replans after each observed therapy response. A phenotype stress laboratory exercises fast-response, slow-response, fatigue-sensitive, pain-sensitive, and balanced synthetic patient twins.

Evidence language remains strict: these results are **synthetic validation / simulation**, not clinical efficacy claims. External clinical validation remains pending.


## V0.95 — Full Rehabilitation Intelligence Platform

V0.95 expands the validated adaptive-control core into a rehabilitation lifecycle platform. The application now exposes eight workspaces: Digital Patient Bay, Motion Capture Studio, Assessment Laboratory, Longitudinal Patient Record, APACE Decision Chamber, Population Research Lab, Algorithm Observatory, and Twin Theater.

The data architecture uses a transactional longitudinal evidence store with deterministic evidence fingerprints and supports DuckDB-backed Parquet analytical export. Motion evidence can enter through an explicit landmark CSV contract; derived assessment findings remain traceable to observed/computed evidence. All clinical outputs remain research decision support and require clinician review; external clinical validation is pending.

## V1.0 acceptance fixture

The release includes `sample_data/observed_landmark_session.csv`, a deterministic **non-patient validation fixture** for the real browser upload path. Open **Motion Capture**, upload that file (or click **RUN BUNDLED ACCEPTANCE FIXTURE**) and verify computed frame count, duration, knee ROM, symmetry, and cadence. Removing `right_knee_deg` must produce a readiness/contract error; missing signals are never fabricated.

## V0.99 — Full clinical workflow release candidate

V0.99 deliberately steps back from the premature V1.0 label and closes the missing product workflow around the computational core. The application now connects a full rehabilitation episode rather than presenting independent research screens:

`evidence acquisition → readiness gate → multi-protocol assessment → latent-state estimation → safe-envelope inference → APACE control → delivered load → observed response → prediction-error ledger → uncertainty update → re-plan → cohort/algorithm evaluation → exportable engineering report`

### What changed

- **Motion Capture Studio:** uploaded landmark sessions now return downsampled bilateral heel and knee traces for visual reconstruction, plus an explicit readiness contract. Signals that are absent remain `INSUFFICIENT_SIGNALS`; they are not synthesized.
- **Assessment Laboratory:** gait, balance, mobility and endurance protocols are independently computed and can be compared side by side.
- **Longitudinal Patient Record:** the application preserves predicted vs observed response, safe limit vs delivered load, capacity/fatigue/pain state, uncertainty contraction and the weekly re-plan.
- **Evidence ledger:** outputs are explicitly labelled CALCULATED, ESTIMATED, PREDICTED, SIMULATED and OPTIMIZED.
- **Population / Observatory:** cohort stress results are exposed as quantitative phenotype rows rather than title-only cards.
- **Episode export:** a deterministic Markdown engineering report can be downloaded from the running application.

The V0.99 reference episode remains **synthetic validation**. The uploaded landmark fixtures are non-patient acceptance data and are used only to validate the real file-ingestion path.

## Enterprise operability gate

This source release includes a governed decision-assurance layer, negative-path operability tests, hash-verifiable evidence, and a Windows enterprise acceptance gate. See `docs/ENTERPRISE_OPERABILITY.md`.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\scripts\\windows_enterprise_acceptance.ps1
```


## Public Data Backbone
This release contains a structured public-data layer under `data/raw`, `data/processed`, `data/contracts`, `data/dictionaries`, `data/provenance`, and `data/snapshots`. Run `scripts\fetch_public_data_windows.ps1` when the primary public dataset is not bundled, then run `scripts\windows_real_data_acceptance.ps1`. `artifacts/data_backbone_status.json` records source state, row/feature counts, missingness, SHA-256, validation status, case-study state, claim boundary, model version, and the human decision authority.

The public-data case is `Postural-Transition State-Confidence Progression Guard` and is wired into `MOTION-GUARD-v1` review. Missing external raw data never silently falls back to a real-data claim; the dossier explicitly enters `REFERENCE_MODE_HOLD_FOR_REAL_DATA_CLAIM`.
## Deployment

Deploy `src/rehab_ai/web/` as the Vercel project root. Its rewrite map serves
the static Vue lab from `frontend/`, while `config.js` sends `/api/*` and
`/health` calls to the Render service declared in the root `render.yaml`.
Deploy the repository root as a Render Blueprint and verify `/health` before
opening the Vercel lab.
