from __future__ import annotations
import numpy as np
from rehab_ai.biomechanics.dynamics import Segment, planar_joint_inverse_dynamics
from rehab_ai.biomechanics.gait_events import fuse_gait_events
from rehab_ai.apace.benchmark import run_policy_benchmark

q=np.full(9,np.pi/6)
r=planar_joint_inverse_dynamics(q,.01,Segment(4,.5,.4))
expected=4*9.80665*.2*np.sin(np.pi/6)
assert np.allclose(r.net_joint_moment_nm,expected)
print('PASS inverse-dynamics static reference')

t=np.linspace(0,4*np.pi,120); h=.05+.03*(1-np.cos(t))/2; v=np.gradient(h); g=np.abs(np.sin(t))*1.4; c=np.ones_like(t)
ev=fuse_gait_events(h,v,g,c,min_separation=7,threshold=.45)
assert len(ev)>=4
print(f'PASS multimodal gait-event fusion ({len(ev)} events)')

suite=run_policy_benchmark(patients=4,horizon=2,seed=42)
rows={x.policy:x for x in suite.results}
assert rows['apace'].unsafe_rate==0 and rows['oracle'].mean_regret_to_oracle==0
print('PASS APACE synthetic policy benchmark')
for x in suite.results:
    print(f'  {x.policy:12s} capacity={x.mean_final_capacity:.3f} tail={x.mean_cumulative_tail_loss:.3f} uncertainty={x.mean_final_uncertainty:.3f} regret={x.mean_regret_to_oracle:.3f}')
print('VALIDATION_SCOPE=SYNTHETIC POLICY BENCHMARK')
