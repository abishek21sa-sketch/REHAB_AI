from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class JsonlExperimentTracker:
    def __init__(self, path: str = "artifacts/runtime/experiments.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def log(self, event: str, payload: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
