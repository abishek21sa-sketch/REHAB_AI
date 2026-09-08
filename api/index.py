"""Vercel adapter for the src-layout Starlette application."""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rehab_ai.web.app import app

__all__ = ["app"]
