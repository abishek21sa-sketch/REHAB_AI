# Session Data Contract

`SessionInput` is the canonical v0.1 multimodal contract. It contains a patient pseudonym, age, raw IMU samples, pose-derived frames, six-minute-walk distance, pain, fatigue, and adherence.

Wearable samples contain monotonic timestamps, acceleration (`ax`, `ay`, `az`) and angular velocity (`gx`, `gy`, `gz`). Pose frames contain timestamped bilateral hip vertical position, bilateral knee angle, and trunk lean. The quality gate rejects unsupported sample/frame rates and non-monotonic time.

The current contract is intentionally de-identified and omits names, addresses, dates of birth, contact data, and free-text clinical notes. Future EHR adapters should map into a separate governed clinical-data model rather than placing PHI into this demonstration contract.

## External clinical and streaming adapters

`rehab_ai.data.ingestion` defines a strict canonical `ClinicalObservation`, a minimal FHIR Observation normalizer, and bounded wearable/pose streaming buffers. These are integration contracts rather than claims of a live hospital EHR connection. Incomplete or semantically ambiguous FHIR-like payloads are rejected rather than silently mapped.
