# V1.0 motion acceptance fixture

`observed_landmark_session.csv` is a deterministic validation fixture shipped with the repository. It is **not real patient data**. Use it to exercise the real file-ingestion and computation path in Motion Capture Studio.

Expected behavior:
- upload is accepted;
- source is labelled `uploaded landmark CSV`;
- frames > 100;
- duration approximately 4 s;
- knee ROM > 20 deg;
- cadence is computed from bilateral heel crossings;
- removing `right_knee_deg` must return a validation error instead of fabricating a value.
