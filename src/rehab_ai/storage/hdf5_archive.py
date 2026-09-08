from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import h5py
import numpy as np


@dataclass(frozen=True)
class ArchiveReceipt:
    patient_id: str
    session_id: str
    samples: int
    channels: tuple[str, ...]
    path: str


def write_sensor_session(path: str | Path, patient_id: str, session_id: str, timestamps: np.ndarray, channels: dict[str, np.ndarray], metadata: dict | None = None) -> ArchiveReceipt:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ts = np.asarray(timestamps, dtype=float)
    if ts.ndim != 1 or len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be a strictly increasing 1-D array")
    if not channels:
        raise ValueError("at least one sensor channel is required")
    for name, values in channels.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape != ts.shape:
            raise ValueError(f"channel {name} length does not match timestamps")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"channel {name} contains non-finite values")
    with h5py.File(target, "a") as h5:
        grp = h5.require_group(f"patients/{patient_id}/sessions/{session_id}")
        for key in list(grp.keys()):
            del grp[key]
        grp.create_dataset("timestamp_s", data=ts, compression="gzip", shuffle=True)
        cgrp = grp.create_group("channels")
        for name, values in channels.items():
            cgrp.create_dataset(name, data=np.asarray(values, dtype=float), compression="gzip", shuffle=True)
        grp.attrs["metadata_json"] = json.dumps(metadata or {}, sort_keys=True)
    return ArchiveReceipt(patient_id, session_id, len(ts), tuple(sorted(channels)), str(target))


def read_sensor_session(path: str | Path, patient_id: str, session_id: str) -> dict:
    with h5py.File(path, "r") as h5:
        grp = h5[f"patients/{patient_id}/sessions/{session_id}"]
        return {
            "timestamps": grp["timestamp_s"][:],
            "channels": {k: grp["channels"][k][:] for k in grp["channels"]},
            "metadata": json.loads(grp.attrs.get("metadata_json", "{}")),
        }
