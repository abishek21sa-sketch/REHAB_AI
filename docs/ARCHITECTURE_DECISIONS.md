# Architecture Decisions — V0.6 Identity Reset

| Component | Chosen technology | Why it fits REHAB AI | Why it is preferable here to portfolio defaults |
|---|---|---|---|
| Scientific language | Python 3.11+ | Biomechanics, filtering, ML and stochastic control share one numerical ecosystem | Keeps mathematical core inspectable and testable |
| Web transport | Starlette ASGI | Thin service layer with minimal framework behavior | Avoids another FastAPI-centric architecture while preserving Python ASGI reliability |
| Frontend | Vue 3.5 global runtime + custom SVG | Highly reactive recovery-state workspace without a generic dashboard component system | Materially different interaction/visual stack; no React/Svelte/Shiny reuse |
| Sensor archive | HDF5/h5py | Dense synchronized numerical channels, compression and hierarchical patient/session organization | Better fit for IMU/pose arrays than ordinary row-oriented CRUD storage |
| AI | UKF + Gaussian Process + K-means | Explicit uncertainty, patient-state estimation and response learning | Different from ordinary classifiers/forecast models |
| IE | Biomechanics + human-performance dynamics | Rehabilitation-specific capacity/load/fatigue/stability mechanics | Avoids generic staffing/utilization IE |
| OR | APACE custom adaptive dual-control search | Sequential therapy changes both patient state and knowledge | Avoids another MILP scheduler; directly exploits partial observability/learning |
| Simulation | Seeded stochastic patient-twin rollouts | Required for CVaR and counterfactual policy stress tests | Directly coupled to optimization rather than isolated scenario charts |
| Deployment | Windows local-first / Docker optional | Clinical/research demonstration with local computation and no PHI cloud assumption | Reproducible portfolio demo without pretending production clinical deployment |
