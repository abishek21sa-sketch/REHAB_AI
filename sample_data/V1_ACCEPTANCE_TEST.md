# REHAB AI V1.0 — concrete acceptance test

Use these files in **Motion Capture Studio** after the app starts. They are deterministic validation fixtures, **not patient data**.

1. Upload `observed_landmark_session.csv`. Expected: accepted; ~201 frames; ~4.0 s; knee ROM > 20°; cadence > 60 spm.
2. Upload `observed_landmark_session_asymmetric.csv`. Expected: accepted; computed ROM/symmetry differ visibly from the first fixture.
3. Upload `invalid_missing_right_knee.csv`. Expected: **REJECTED** with a message naming `right_knee_deg`; the app must not invent the missing field.
4. Open **Population** and **Observatory**. Expected: real non-zero synthetic-cohort metrics, not 0/0/NaN placeholders.
5. Open **Patient Record**. Expected: weekly action, capacity, observed gain, safe limit/load, and uncertainty values are populated.
