from pathlib import Path
import numpy as np
import pytest
from ._subject import load_subject,source_directory

m=load_subject('grip_session_ik')

def test_bounded_lead_still_detects_a_real_stall():
    state=m.GripSessionIK();q=np.zeros(14)
    names=list(m.ARM_NAMES['left'])+list(m.ARM_NAMES['right'])
    joint_q={n:i for i,n in enumerate(names)}
    state.observe('stall',dict(left=True,right=False),q,1.)
    command=q.copy();command[0]=np.deg2rad(9)
    with pytest.raises(m.SessionTrackingError,match='stalled'):
        for i in range(200):
            t=state.prepare(('left',),q,joint_q,1.01+i*.01)
            assert state.commit(t,command)
    assert 'left' in state.blocked

def test_reset_launch_does_not_override_robot_neutral_with_s2_head_pitch():
    root=source_directory().parent
    text=(root/'launch/full_vr_teleop.launch.py').read_text()
    assert "'quick_reset_neck_joints', default_value='[0.0,0.0,0.0]'" in text

def test_one_second_reset_request_and_failure_survive_regular_status():
    from types import SimpleNamespace as NS
    from concurrent.futures import Future
    from unittest.mock import Mock
    from openarmx_teleop_vr_306_v5.vr_mapper_node import IndependentVrMapper as Mapper
    from openarmx_teleop_vr_306_v5.reset_state import QuickResetState
    params={'quick_reset_enabled':True,'quick_reset_input_fresh_sec':.3,'quick_reset_hold_seconds':1.}
    state=QuickResetState();future=Future();client=NS(service_is_ready=lambda:True,call_async=Mock(return_value=future))
    import threading
    c=NS(_lock=threading.RLock(),get_parameter=lambda k:NS(value=params[k]),_quick_future=None,
         _hardware_enabled=True,_backend_status={},_dry_run=False,_release_all=Mock(),
         _quick_result={'sequence':0,'success':None,'message':''},_quick_state=state,
         _backend_reset_active=False,get_logger=lambda:NS(info=Mock()))
    c._record_quick_result=lambda ok,msg:Mapper._record_quick_result(c,ok,msg)
    for i in range(120):
        now=1+i*.01;state.update_full_snapshot(True,True,now)
        Mapper._check_reset_gesture(c,now,state,client)
    assert client.call_async.call_count==1
    future.set_result(NS(success=False,message='direct quick-reset path is in collision'))
    c._last_reason='waiting for grip clutch'
    status=Mapper._quick_reset_status(c,2.2)
    assert status['last_result']['success'] is False
    assert 'collision' in status['last_result']['message']
    assert status['last_result']['sequence']==1

def test_fast_target_keeps_continuous_state_bounded_without_regrip():
    pytest.importorskip('placo')
    k=load_subject('quest_placo_kinematics')
    root=source_directory().parent
    ik=k.QuestPlacoKinematics(root/'urdf/robot_v3_1_simplified.urdf',collision_check=False)
    groups=dict(leg_waist=np.zeros(4),left_arm=np.array([20,0,0,-110,0,0,0.]),
                right_arm=np.array([20,0,0,-110,0,0,0.]),neck=np.zeros(3))
    actual=ik.q_from_feedback(groups);q0=actual.copy();goal=actual.copy()
    idx=[ik.joint_q[n] for n in k.LEFT_ARM_JOINTS]
    goal[idx[0]]+=np.deg2rad(40);goal=np.clip(goal,ik.lower,ik.upper)
    pose=ik.eef_poses(goal)['left'];targets={'left':(pose['position'],pose['orientation'])}
    state=m.GripSessionIK();state.observe('fast',dict(left=True,right=False),actual,1.)
    limited=False
    for i in range(500):
        ticket=state.prepare(targets,actual,ik.joint_q,1.01+i*.01)
        result=ik.solve(targets,ticket.seed,posture_reference=ticket.reference,
                        integration_dt=ticket.dt,feedback_q=ticket.measured,
                        maximum_arm_lead_rad=state.maximum_lead)
        assert result.success,result.reason
        assert np.max(np.abs(result.q[idx]-actual[idx]))<=np.deg2rad(10)+1e-10
        limited |= ik.last_lead_scale['left']<.999
        assert state.commit(ticket,result.q)
        actual[idx]+=np.clip(result.q[idx]-actual[idx],-np.deg2rad(.2),np.deg2rad(.2))
    assert limited
    assert not state.blocked
    assert result.position_error_m['left']<.006
    np.testing.assert_array_equal(state.reference['left'],q0)
