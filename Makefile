.PHONY: install test run demo validate
install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest

run:
	uvicorn rehab_ai.api.main:app --reload --host 0.0.0.0 --port 8000

demo:
	python scripts/run_demo.py

validate:
	python scripts/generate_validation_report.py
