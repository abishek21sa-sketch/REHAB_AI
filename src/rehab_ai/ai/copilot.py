"""REHAB//AI copilot: deterministic safety readout with optional Gemini narration."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

def status() -> dict[str, Any]:
    return {"provider": "Google Gemini", "model": DEFAULT_MODEL, "key_present": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")), "deterministic_fallback": True, "temperature": 0, "claim_boundary": "REHAB//AI is research decision support; clinical validation and clinician authority remain required."}

def _deterministic(message: str, context: dict[str, Any]) -> str:
    q = message.casefold()
    if any(k in q for k in ("pain", "fatigue", "safe", "safety", "load", "risk")):
        next_step = "Inspect the patient-specific safe envelope, fatigue/pain state, and the APACE CVaR path before considering any intervention."
    elif any(k in q for k in ("gait", "motion", "symmetry", "assessment")):
        next_step = "Open Motion Capture or Assessment and verify signal readiness, bilateral evidence, and missing inputs."
    elif any(k in q for k in ("learn", "prediction", "response", "record", "uncertainty")):
        next_step = "Open the Longitudinal Patient Record and compare predicted versus observed response before replanning."
    else:
        next_step = "Start at Patient Bay, inspect the evidence ledger, then review APACE's selected path and its safety constraints."
    return ("Deterministic REHAB//AI safety readout\n\n" f"Question: {message.strip()}\n\n" f"Recommended path: {next_step}\n" f"Evidence: {context.get('evidence', 'synthetic motion, biomechanical, twin, and safe-control evidence')}\n" "Boundary: no diagnosis, prescription, or autonomous patient action is produced; clinician review is mandatory.")

def _gemini(message: str, context: dict[str, Any]) -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY is not configured")
    body = {"system_instruction": {"parts": [{"text": "You are a rehabilitation engineering copilot. Deterministic biomechanics, patient-safe-envelope, APACE, and longitudinal evidence are authoritative. Do not diagnose, prescribe, or invent clinical validation."}]}, "contents": [{"role": "user", "parts": [{"text": f"Context: {json.dumps(context, sort_keys=True)}\nQuestion: {message.strip()}"}]}], "generationConfig": {"temperature": 0, "seed": 42, "maxOutputTokens": 700}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    with urllib.request.urlopen(req, timeout=25) as response: data = json.loads(response.read().decode())
    text = "\n".join(p.get("text", "") for p in data.get("candidates", [{}])[0].get("content", {}).get("parts", []) if p.get("text"))
    if not text.strip(): raise RuntimeError("Gemini returned no visible text")
    return text.strip()

def build_response(message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}; fallback = _deterministic(message, context)
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return {**status(), "answer": fallback, "provider": "deterministic", "model": "rule-based", "fallback": True}
    try: return {"answer": _gemini(message, context), "provider": "Google Gemini", "model": DEFAULT_MODEL, "fallback": False, **status()}
    except (OSError, urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
        return {**status(), "answer": fallback, "provider": "deterministic-fallback", "model": "rule-based", "fallback": True, "error": type(exc).__name__}
