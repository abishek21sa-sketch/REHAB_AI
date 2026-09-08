from __future__ import annotations

from collections import Counter
from threading import Lock


class MetricsRegistry:
    def __init__(self):
        self._counts: Counter[str] = Counter()
        self._lock = Lock()

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counts[name] += value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


metrics = MetricsRegistry()
