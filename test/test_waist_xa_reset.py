"""No hardware commands: verify waist reset targets and feedback boundaries."""
import numpy as np
import pytest
from types import SimpleNamespace as NS
from openarmx_teleop_vr_306_v5.controller_node import IndependentArmController as C
from openarmx_teleop_vr_306_v5.controller_node import _waist_assist_window_active
from openarmx_teleop_vr_306_v5.grip_session_ik import GripSessionIK
from openarmx_teleop_vr_306_v5.reset_path import feedback_bounded_reset_progress,smoothstep5
from openarmx_teleop_vr_306_v5.trajectory_limiter import TrajectoryLimiter


def fixture(waist=(0.,0.)):
    params={'quick_reset_left_arm_joints':[20.,0.,0.,-110.,0.,0.,0.],
            'quick_reset_right_arm_joints':[20.,0.,0.,-110.,0.,0.,0.],
            'quick_reset_neck_joints':[0.,0.,0.], 'quick_reset_waist_joints':waist,
            'quick_reset_open_grippers':False}
    groups={'left_arm':np.array([20.,0.,0.,-110.,0.,0.,0.]),
            'right_arm':np.array([20.,0.,0.,-110.,0.,0.,0.]),'neck':np.zeros(3),
            'leg_waist':np.array([12.,27.,20.,8.]),
            'left_gripper':np.array([80.]),'right_gripper':np.array([90.])}
    calls=[]
    subject=NS(get_parameter=lambda n:NS(value=params[n]),
        _configuration_guard_error=lambda *a,**kw:('',None),
        _kinematics=NS(groups_from_q_deg=lambda q,g:g),
        _reset_required_vector=lambda g:np.r_[g['left_arm'],g['right_arm'],g['neck'],g['leg_waist']],
        _build_motion_controller_state=lambda g:{},
        _prepare_quick_reset_path=lambda a,b:(calls.append((a,b)) or '',{'start':np.zeros(21),'goal':np.ones(21),'duration_sec':2}),
        _grip_ik=GripSessionIK(),_latest_target_mailbox=NS(reset=lambda:None),
        _commit_motion_controller_state=lambda s:None,_lock_head_follow_for_control_boundary_locked=lambda:None,
        _deactivate_body_height_locked=lambda:None,_velocity_active_sides=set(),_waist_follow_enabled=False,
        _gripper_targets={'left':270.,'right':250.})
    return subject,groups,calls

def test_xa_restores_pitch_yaw_and_keeps_ankle_knee():
    obj,g,calls=fixture();msg=C._start_quick_reset_locked(obj,1.,NS(as_dict=lambda:g))
    assert msg==''
    np.testing.assert_allclose(obj._target_groups['leg_waist'],[12.,27.,81.*(12.*76.+27.*155.)/(76.**2+155.**2),0.])
    np.testing.assert_allclose(g['leg_waist'],[12.,27.,20.,8.])
    assert len(calls)==1

def test_other_reset_does_not_move_waist():
    obj,g,_=fixture();msg=C._start_quick_reset_locked(obj,1.,NS(as_dict=lambda:g),source='hold_gripper_reset',preserve_grippers=True)
    assert msg==''
    np.testing.assert_allclose(obj._target_groups['leg_waist'],g['leg_waist'])

@pytest.mark.parametrize('value',[(0,), (0,float('nan')), (1,2,3)])
def test_invalid_waist_reference_is_rejected_before_motion(value):
    obj,g,calls=fixture(value);msg=C._start_quick_reset_locked(obj,1.,NS(as_dict=lambda:g))
    assert 'waist configuration' in msg and not calls and not hasattr(obj,'_reset_active')

def test_waist_yaw_cannot_run_ahead_of_physical_feedback():
    start=np.zeros(21);goal=np.zeros(21);measured=np.zeros(21)
    start[3]=10.;measured[3]=10.
    p=feedback_bounded_reset_progress(start,goal,measured,0.,1.,2.)
    assert smoothstep5(p)*10 <= 2.0001
    assert p<1.

def test_reset_completion_vector_contains_pitch_and_yaw():
    obj,g,_=fixture();v=C._reset_required_vector(g)
    assert len(v)==21 and v[-2:]==pytest.approx([20.,8.])

def test_waist_speedup_is_bounded_and_returns_as_fast_as_it_leans():
    def run(speed,direction):
        initial=0. if direction>0 else 10.
        target=10. if direction>0 else 0.
        limiter=TrajectoryLimiter([-30.],[30.],[speed],[12.]);limiter.reset([initial]);last=initial
        for i in range(1000):
            value=float(limiter.step([target],.01)[0]);assert abs(value-last)<=speed*.01+1e-8;last=value
            if abs(value-target)<.5:return i*.01
        raise AssertionError('did not arrive')
    slow=run(3.,1);fast=run(4.,1);back=run(4.,-1)
    assert fast<slow and abs(fast-back)<.03

def test_follow_window_survives_hand_return_only_while_grip_and_feedback_fresh():
    args=(True,'forward_pitch_only',0.,0.,'cartesian_ik')
    assert _waist_assist_window_active(*args,{'left'},4.,True)
    assert not _waist_assist_window_active(*args,{'left','right'},4.,True)
    assert not _waist_assist_window_active(*args,{'left'},4.,False)
    assert not _waist_assist_window_active(*args,{'left'},0.,True)
    assert not _waist_assist_window_active(True,'forward_pitch_only',0.,0.,'grip_release_hold',{'left'},4.,True)
