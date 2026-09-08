import numpy as np
from rehab_ai.biomechanics.dynamics import Segment, segment_com, whole_body_com, planar_joint_inverse_dynamics

def test_segment_com_midpoint_reference():
    p=np.array([[0.,0.],[1.,1.]])
    d=np.array([[2.,0.],[3.,1.]])
    assert np.allclose(segment_com(p,d,.5),np.array([[1.,0.],[2.,1.]]))

def test_whole_body_com_mass_weighted_reference():
    a=np.array([[0.,0.],[0.,1.]])
    b=np.array([[2.,0.],[2.,1.]])
    com=whole_body_com([a,b],[1.,3.])
    assert np.allclose(com,np.array([[1.5,0.],[1.5,1.]]))

def test_inverse_dynamics_static_reference_equals_gravity_moment():
    q=np.full(9,np.pi/6)
    s=Segment(4.,.5,.4)
    r=planar_joint_inverse_dynamics(q,.01,s)
    expected=4*9.80665*.2*np.sin(np.pi/6)
    assert np.allclose(r.angular_acceleration,0,atol=1e-10)
    assert np.allclose(r.net_joint_moment_nm,expected)
