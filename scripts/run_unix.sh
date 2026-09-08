#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
python scripts/phase_apace_diagnostics.py
PYTHONPATH=src python -m uvicorn rehab_ai.web.app:app --host 127.0.0.1 --port 8010
