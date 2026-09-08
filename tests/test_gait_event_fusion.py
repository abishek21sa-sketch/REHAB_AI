import numpy as np
from rehab_ai.biomechanics.gait_events import fuse_gait_events

def synthetic_cycle(n=120):
    t=np.linspace(0,4*np.pi,n)
    h=.05+.03*(1-np.cos(t))/2
    v=np.gradient(h)
    g=np.abs(np.sin(t))*1.4
    c=np.ones(n)
    return h,v,g,c

def test_fusion_returns_ordered_high_confidence_events():
    h,v,g,c=synthetic_cycle()
    ev=fuse_gait_events(h,v,g,c,min_separation=7,threshold=.45)
    assert len(ev)>=4
    assert [x.index for x in ev]==sorted(x.index for x in ev)
    assert all(0<=x.confidence<=1 for x in ev)

def test_low_pose_confidence_still_allows_imu_supported_detection():
    h,v,g,c=synthetic_cycle(); c[20:80]=.05
    ev=fuse_gait_events(h,v,g,c,min_separation=7,threshold=.40)
    assert any(x.evidence['pose_confidence']<.1 and x.evidence['gyro_evidence']>.5 for x in ev)
