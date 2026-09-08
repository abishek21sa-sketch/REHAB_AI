# Architecture — Recovery Policy Studio

```text
Wearable / pose / assessment evidence
              │
              ▼
 canonical validation + sensing features
              │
              ▼
 computational biomechanics
 COM / XCoM / MoS / symmetry / gait features
              │
              ▼
 UKF latent patient belief + covariance
              │
      ┌───────┴────────┐
      ▼                ▼
 phenotype ML      GP response ML
      │                │
      └───────┬────────┘
              ▼
 Capacity–Load–Fatigue–Recovery twin
              │
              ▼
 APACE adaptive dual-control algorithm
 info gain + CVaR + burden + safety pruning
              │
              ▼
 Counterfactual policy trajectory
              │
              ▼
 Starlette transport → Vue Recovery Policy Studio
```

High-frequency sensor sessions can be archived as compressed HDF5 groups by patient/session. Application transport is deliberately thin; scientific models are ordinary Python modules and are directly testable without HTTP.
