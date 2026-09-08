# REHAB AI — Technical Methods

This document describes the implemented computational methods and evidence boundaries for the V0.6 APACE computational-identity build.

## A. Artificial Intelligence

### A1. Latent rehabilitation-state estimation

**Task.** Estimate a partially observed patient state from biomechanical, wearable and assessment evidence.

**Model.** Unscented Kalman Filter (UKF).

**State.** `[motor_capacity, fatigue_burden, pain_burden, stability_reserve]`.

**Observation.** Reconstructed functional capacity, reported/derived fatigue, pain and normalized dynamic-stability evidence.

**Uncertainty.** The UKF maintains a covariance matrix `P_t`; standard deviations are exposed to downstream algorithms rather than collapsed into an arbitrary confidence score.

**Decision use.** State mean and covariance initialize the patient belief consumed by APACE.

**Implementation.** `src/rehab_ai/state_estimation/ukf.py`, integrated by `src/rehab_ai/services/apace_studio.py`.

**Validation.** Reference tests verify state/covariance behavior. External patient validation is pending.

### A2. Treatment-response machine learning

**Task.** Predict change in functional capacity for a candidate therapy dose while representing predictive uncertainty.

**Model.** Gaussian Process Regressor with Matern kernel and observation-noise component.

**Features.** Current capacity, fatigue, pain, functional load and instability demand.

**Target.** Capacity gain after intervention.

**Training data.** Deterministic synthetic treatment-response population in V0.6.

**Validation.** Held-out synthetic data. The declared baseline is constant mean response from the training partition.

**Metrics.** MAE, RMSE and empirical 95% predictive-interval coverage.

**Decision use.** Mean response contributes expected improvement; model uncertainty contributes APACE safety/learning behavior.

**Limitations.** This is not an estimate of real clinical treatment effect. Real longitudinal rehabilitation data are required before clinical interpretation.

**Implementation.** `src/rehab_ai/ml/treatment_response.py`.

### A3. Rehabilitation phenotype discovery

**Task.** Identify patient-state archetypes that can eventually inform priors and cohort diagnostics.

**Model.** K-means clustering over normalized `[capacity, fatigue, pain, stability, symmetry]` features.

**Validation.** Synthetic archetype population using silhouette coefficient. Results are explicitly labeled synthetic validation.

**Decision use.** V0.6 exposes phenotype context in the Recovery Policy Studio. Future validated work can use phenotype-conditioned response priors.

**Implementation.** `src/rehab_ai/ml/phenotypes.py`.

### A4. Secondary safety prediction

The earlier calibrated/logistic fall-risk machinery remains in the repository as a secondary safety capability. It is deliberately not the flagship AI method after the V0.6 architecture reset.

---

## B. Industrial Engineering / Rehabilitation Engineering

### B1. Bilateral symmetry

For left/right quantity `L,R`:

`SI = |L-R| / ((|L|+|R|)/2)`

**Units.** Dimensionless.

**Interpretation.** Zero indicates bilateral equality; increasing values indicate larger asymmetry.

**Implementation.** `src/rehab_ai/biomechanics/stability.py`.

### B2. Extrapolated center of mass and Margin of Stability

For anterior-posterior COM position `x`, COM velocity `xdot`, gravitational acceleration `g` and effective pendulum length `l`:

`XCoM = x + xdot / sqrt(g/l)`

`MoS = BoS_boundary - XCoM`

**Units.** meters.

**Interpretation.** Positive MoS indicates the extrapolated COM remains inside the modeled base-of-support boundary. V0.6 uses an engineering approximation and does not claim force-plate-grade kinetics.

**Implementation.** `src/rehab_ai/biomechanics/stability.py` and service integration.

### B3. Therapeutic functional load

`L = intensity × (duration_min/60) × (1 + instability_demand) × repetition_factor`

**Units.** normalized functional-load units.

The patient-specific safe envelope is a function of current capacity, pain and dynamic stability. The precise implemented mapping is documented in code because it is a research engineering model rather than a validated clinical scale.

### B4. Capacity–load–fatigue–recovery dynamics

Fatigue carries over with exponential recovery and increases with load/overload. Capacity adaptation is hormetic: under-load produces limited adaptation, an intermediate load region maximizes modeled gain, and excessive load creates an overload penalty. Pain also carries over and responds to overload.

**Implementation.** `src/rehab_ai/ie/human_performance.py`.

**Mathematical tests.** Tests verify overload increases fatigue burden, safe-load bounds remain finite, capacity/pain/fatigue states remain bounded and policy trajectories respect configured constraints.

---

## C. Operations Research / Adaptive Control

### C1. Decision problem

At each reassessment, select a sequence of rehabilitation actions that improves expected functional state while respecting biomechanical/human-performance safety and accounting for uncertainty about how this patient responds.

### C2. APACE algorithm

**Name.** Adaptive Patient-Aware Control and Exploration.

**Decision variables.** Discrete therapy action at each planning stage. Current action set includes recovery, balance precision, gait progression, strength adaptation and integrated function. Each action has modality, intensity, duration and instability demand.

**State.** Belief over human-performance state and patient-response parameter uncertainty.

**Stage objective.**

`A(a|b) = E[functional gain] + λ(b)·IG(a) - β·CVaRα(loss) - η·burden(a)`

where:

- `E[functional gain]` is estimated by stochastic digital-twin rollout;
- `IG(a)` is entropy reduction in patient-response uncertainty;
- `CVaRα(loss)` measures tail loss over simulated response outcomes;
- `burden(a)` combines intensity, duration and instability demand;
- `λ(b)` contracts as state/response uncertainty decreases.

**Safety constraints.** Candidate branches are rejected when patient-specific robust functional load exceeds the safe envelope, modeled Margin of Stability is below the configured minimum, or predicted fatigue/pain exceed limits.

**Search.** APACE uses:

1. safety pruning;
2. action-level dominance pruning;
3. uncertainty-aware diversity-preserving beam expansion;
4. seeded Monte Carlo patient-twin rollout;
5. CVaR tail evaluation;
6. receding-horizon re-optimization after new evidence.

**Domains.** Therapy decisions are discrete; state and uncertainty variables are continuous.

**Solver.** Custom Python algorithm using NumPy-based stochastic rollouts. APACE is not described as an exact MILP solver.

**Solution status.** `OPTIMAL_BEAM_APPROXIMATION` means best policy in the retained beam search; it is not a claim of global optimality. `INFEASIBLE` means no safe branch survived.

**Oracle validation.** Tiny problems can disable dominance pruning and use a beam equal to the full action tree. `exact_apace_small` provides an exhaustive reference oracle. Unit tests compare the tested search configuration to that oracle on a tractable case.

**Complexity.** Naive enumeration is `|A|^H`. Beam search reduces retained width to `K`, making expansion approximately `O(H·K·|A|·N)` where `N` is rollout count, excluding ML inference cost.

**Limitations.** V0.6 transition and treatment-response models are synthetic/research models. APACE is an algorithmic research contribution in this repository, not a clinically validated treatment policy.

### C3. Legacy/secondary optimization

Earlier finite-horizon MPC and therapy optimization implementations are retained only where they provide baselines or regression evidence. They are no longer the project's signature OR capability.


## D. COMPUTATIONAL BIOMECHANICS EXTENSION — V0.6

### D1. Segmental center of mass
For segment COM positions r_i and segment masses m_i, the whole-body planar COM is

COM = sum(m_i r_i) / sum(m_i).

Units: meters for position, kilograms for mass. `rehab_ai.biomechanics.dynamics` implements segment COM interpolation and mass-weighted whole-body COM. Reference unit tests verify midpoint and weighted-average cases independently.

### D2. Simplified planar inverse dynamics
For a single segment with generalized angle q, the implemented educational/research planar moment is

M_net = I alpha + m g r sin(q),

where I = m L^2 / 12, alpha is angular acceleration, r is the COM radius, and moments are in N·m. This is not a substitute for laboratory inverse dynamics using force plates; it is explicitly a simplified computational biomechanics component for controlled validation and policy features.

### D3. Reliability-weighted multimodal gait events
`fuse_gait_events` combines pose-derived heel height, heel velocity, IMU gyroscope magnitude and pose confidence. Visual evidence is attenuated when pose confidence falls while the IMU channel remains available. Local maxima plus a refractory separation rule produce candidate heel-strike/toe-off events. This is an implemented signal-fusion algorithm, validated on deterministic synthetic cycles; external gait-laboratory validation is pending.

## E. APACE ALGORITHM BENCHMARKING — V0.6

APACE is now benchmarked against four policy references on deterministic synthetic patient twins: greedy response, risk-only control, standard MPC without information value, and a small-horizon exact oracle. Each baseline first selects its policy under its own rule, then the selected sequence is replayed under one common APACE objective before regret is computed. This prevents invalid comparison of raw objective values generated with different weights.

Reported benchmark quantities are mean final capacity, cumulative CVaR tail loss, final response-parameter uncertainty, regret to the exact small-horizon APACE oracle, and unsafe-policy rate. Results are labeled **SYNTHETIC POLICY BENCHMARK** and are not clinical outcomes.

## V0.8 Computational-depth additions

### Artificial Intelligence — temporal recovery model
A time-aware Extra Trees regression model predicts next-assessment motor capacity from four lagged weekly states. Each state contains capacity, fatigue, pain, adherence and therapy dose. Entire patient episodes, rather than randomly shuffled observations, are split into development and held-out cohorts to avoid future-state leakage. The declared baseline is persistence: next capacity equals current capacity. Reported metrics are MAE/RMSE and are explicitly synthetic longitudinal validation.

### Artificial Intelligence + safety — patient-specific safe-envelope learner
A HistGradientBoosting regression model learns maximum tolerated functional load from capacity, fatigue, pain, dynamic stability margin, recent overload and adherence. A split-conformal calibration set produces a conservative lower bound on tolerated load. The operational decision use is safety pruning before adaptive therapy search; the model is never presented as clinically validated.

### Biomechanics — joint energetics
Joint mechanical power is implemented as `P(t)=M(t)ω(t)`. Positive, negative and net joint work are numerically integrated over time, with mechanical cost normalized by body mass. This adds an energetic interpretation to joint loading rather than relying only on angles and moments.

### Signal engineering — movement periodicity and smoothness
Wearable acceleration signals are characterized with RMS acceleration, RMS jerk, normalized spectral entropy and dominant-periodicity index. These quantities provide movement-quality and signal-readiness evidence for gait reconstruction and are mathematical preprocessing outputs, not AI predictions.

---

## V0.99 Clinical Workflow Integration

### Evidence readiness gate

Observed landmark files are accepted only when the required kinematic channels are present. The motion adapter computes bilateral traces and kinematic features from the provided evidence, but does **not** fabricate gait speed, margin of stability, force or pressure measurements that are absent from the file. Downstream readiness is therefore explicit (`READY_FOR_KINEMATIC_FEATURES` versus `INSUFFICIENT_SIGNALS`).

### Multi-protocol rehabilitation assessment

The assessment layer contains four executable protocols: gait, balance, mobility and endurance. Each protocol normalizes only domain-relevant deficits and returns a score, severity, primary limiting mechanism, evidence vector and deterministic rationale. These are engineering assessments on the available/synthetic signals; they are not replacements for validated clinical instruments.

### Closed-loop learning evidence

For each treatment week the longitudinal episode preserves the policy selected before treatment, safe-load limit, delivered load, predicted response, observed synthetic response, functional state and parameter uncertainty. The learning diagnostic measures mean absolute prediction error and uncertainty contraction before the next APACE solve. This prevents the digital twin from claiming personalization without retaining evidence that its belief changed.
