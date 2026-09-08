from __future__ import annotations

from rehab_ai.domain import SessionInput


class DataQualityError(ValueError):
    pass


def validate_session(session: SessionInput) -> dict[str, float | int]:
    wearable_times = [s.t for s in session.wearable]
    pose_times = [p.t for p in session.pose]
    if any(b <= a for a, b in zip(wearable_times, wearable_times[1:])):
        raise DataQualityError("Wearable timestamps must be strictly increasing")
    if any(b <= a for a, b in zip(pose_times, pose_times[1:])):
        raise DataQualityError("Pose timestamps must be strictly increasing")
    duration = wearable_times[-1] - wearable_times[0]
    if duration < 2.0:
        raise DataQualityError("Wearable window must be at least 2 seconds")
    sample_rate = (len(wearable_times) - 1) / duration
    if not 10 <= sample_rate <= 250:
        raise DataQualityError(f"Wearable sample rate {sample_rate:.1f} Hz outside supported 10-250 Hz")
    pose_duration = pose_times[-1] - pose_times[0]
    pose_rate = (len(pose_times) - 1) / pose_duration
    if not 5 <= pose_rate <= 120:
        raise DataQualityError(f"Pose frame rate {pose_rate:.1f} Hz outside supported 5-120 Hz")
    return {
        "wearable_samples": len(session.wearable),
        "pose_frames": len(session.pose),
        "wearable_sample_rate_hz": round(sample_rate, 3),
        "pose_frame_rate_hz": round(pose_rate, 3),
    }
