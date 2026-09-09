# Rehab AI deployment

The rehabilitation decision API runs on Render and the patient-lab frontend runs on Vercel. The safety copilot remains useful without a Gemini key.

1. Create a Render Blueprint from this repository. Keep the service name `rehab-ai-api`; Render starts `rehab_ai.web.app:app` and checks `/health`.
2. Confirm `https://rehab-ai-api.onrender.com/health` is healthy.
3. Import the repository in Vercel and set Root Directory to `src/rehab_ai/web`. The local `vercel.json` publishes the frontend assets.
4. Open the Vercel URL, run the patient demo, load a bundled motion session, and ask the copilot for a safety review.
5. Optional: add `GEMINI_API_KEY` only as a Render secret.

The API bridge is in `src/rehab_ai/web/frontend/config.js`; update it if the Render service is renamed.
