import numpy as np
import pytest
from ._subject import load_subject
m=load_subject('shoulder_retarget')
core=load_subject('teleop_core')

def mapper():
    a=m.ShoulderReachMapper(side='left',position_scale=1.,forward_position_scale=1.,maximum_forward_displacement=0.,
                           filter_alpha=.5,maximum_position_step=.01,maximum_orientation_step=.1,maximum_anchor_displacement=1.,left_y_minimum=-2.,right_y_maximum=2.,
                           workspace_minimum=[-2,-2,-2],workspace_maximum=[2,2,2],forbidden_box_minimum=[-3,-3,-3],forbidden_box_maximum=[-2,-2,-2])
    a.set_geometry({'shoulder':[0,.22,1.],'reach':.53});a.human_length=.65
    return a

def sample(vector):
    shoulder=np.array([-.03,.19,-.22]);head=np.array([0,1.6,0.])
    return core.ControllerSample(head+core.VR_TO_ROBOT_ROT.T@(np.asarray(vector)+shoulder),np.array([0.,0.,0.,1.]),True,0.)

def test_clutch_no_jump_then_full_reach_and_raise_preserves_radius():
    a=mapper();h=np.array([0,1.6,0.]);start=sample([.35,0,0])
    pose=dict(position=np.array([.32,.22,1.]),orientation=np.array([0,0,0,1.]))
    a.latch(start,pose,h);p,_=a.map(start,h)
    np.testing.assert_allclose(p['position'],pose['position'],atol=1e-9)
    p,_=a.map(sample([.65,0,0]),h)
    assert np.linalg.norm(p['position']-[0,.22,1.])==pytest.approx(.53)
    p,_=a.map(sample([.65/2**.5,0,.65/2**.5]),h)
    assert np.linalg.norm(p['position']-[0,.22,1.])==pytest.approx(.53)
    assert p['position'][2]>1.3

def test_already_extended_grip_cannot_straighten_robot_by_itself():
    a=mapper();h=np.array([0,1.6,0.]);s=sample([.65,0,0]);pose=dict(position=np.array([.32,.22,1.]),orientation=np.array([0,0,0,1.]))
    a.latch(s,pose,h);p,_=a.map(s,h)
    assert not a.spherical
    np.testing.assert_allclose(p['position'],pose['position'])

def test_body_translation_does_not_move_target_and_missing_head_holds():
    a=mapper();h=np.array([0,1.6,0.]);s=sample([.35,0,0]);pose=dict(position=np.array([.32,.22,1.]),orientation=np.array([0,0,0,1.]))
    a.latch(s,pose,h)
    d=np.array([.2,0,.1]);s.position+=d
    p,_=a.map(s,h+d);np.testing.assert_allclose(p['position'],pose['position'])
    p,reason=a.map(s,None);assert p is None and 'HMD' in reason

def test_antipodal_direction_rotation_is_finite_and_maps_axis():
    r=m.rotation_between([1,0,0],[-1,0,0])
    np.testing.assert_allclose(r@[1,0,0],[-1,0,0]);np.testing.assert_allclose(r.T@r,np.eye(3))

def test_elbow_bend_reduces_with_reach_instead_of_locking_joint_posture():
    pytest.importorskip('pinocchio')
    g=load_subject('arm_geometry');s=np.zeros(3);e=np.array([.2,0,-.2]);w=np.array([.35,0,0])
    bent=g.elbow_target(s,[.3,0,0],s,e,w,.3,.25)
    straight=g.elbow_target(s,[.549,0,0],s,e,w,.3,.25)
    assert abs(straight[2])<abs(bent[2])*.2
