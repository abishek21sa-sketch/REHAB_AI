from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GaitEvent:
    index: int
    event: str
    confidence: float
    evidence: dict[str, float]


def _normalize_score(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    lo, hi = np.percentile(a, [5,95])
    if hi <= lo + 1e-12:
        return np.zeros_like(a)
    return np.clip((a-lo)/(hi-lo), 0, 1)


def fuse_gait_events(
    heel_height: np.ndarray,
    heel_velocity: np.ndarray,
    imu_gyro: np.ndarray,
    pose_confidence: np.ndarray,
    min_separation: int = 8,
    threshold: float = 0.58,
) -> tuple[GaitEvent, ...]:
    """Reliability-weighted multimodal heel-strike/toe-off detector.

    Heel strike favors low heel height, downward-to-upward velocity transition and a gyro impulse.
    Toe off favors rising heel height, positive heel velocity and gyro activity.
    Pose confidence modulates the visual modalities rather than zeroing the IMU channel.
    """
    h = np.asarray(heel_height, float)
    v = np.asarray(heel_velocity, float)
    g = np.asarray(imu_gyro, float)
    c = np.clip(np.asarray(pose_confidence, float),0,1)
    if not (h.shape == v.shape == g.shape == c.shape) or h.ndim != 1:
        raise ValueError('all gait signals must be aligned 1D arrays')
    if h.size < 20:
        raise ValueError('at least 20 samples required')

    low_h = 1-_normalize_score(h)
    high_h = _normalize_score(h)
    pos_v = _normalize_score(v)
    neg_v = _normalize_score(-v)
    gyro = _normalize_score(np.abs(g))
    vel_turn_hs = np.r_[0.0, ((v[:-1] < 0) & (v[1:] >= 0)).astype(float)]

    hs = 0.32*(low_h*c) + 0.28*(vel_turn_hs*c) + 0.40*gyro
    to = 0.33*(high_h*c) + 0.37*(pos_v*c) + 0.30*gyro

    candidates=[]
    for label, score in [('HEEL_STRIKE',hs),('TOE_OFF',to)]:
        for i in range(1,len(score)-1):
            if score[i] >= threshold and score[i] >= score[i-1] and score[i] >= score[i+1]:
                candidates.append((float(score[i]), i, label))
    candidates.sort(reverse=True)
    selected=[]
    for score,i,label in candidates:
        if any(abs(i-j) < min_separation for _,j,_ in selected):
            continue
        selected.append((score,i,label))
    selected.sort(key=lambda z:z[1])
    events=[]
    for score,i,label in selected:
        ev={
            'pose_confidence':float(c[i]),
            'gyro_evidence':float(gyro[i]),
            'heel_height_evidence':float(low_h[i] if label=='HEEL_STRIKE' else high_h[i]),
            'velocity_evidence':float(vel_turn_hs[i] if label=='HEEL_STRIKE' else pos_v[i]),
        }
        events.append(GaitEvent(i,label,score,ev))
    return tuple(events)
