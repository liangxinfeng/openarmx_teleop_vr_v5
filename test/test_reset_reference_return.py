"""Native model/QP regression; no ROS node or motor publishers."""
import numpy as np
import pytest
pytest.importorskip('mink')
from .test_mink_armshape import make_ik, seed, target
from ._subject import load_subject
module=load_subject('mink_kinematics')


@pytest.mark.parametrize('side',['left','right'])
def test_new_grip_at_arbitrary_shape_does_not_jump_toward_home(side):
    ik=make_ik();q=seed(ik,-65);home=seed(ik,-110)
    result=ik.solve(target(ik,q,side),q,reset_reference=home,session_epoch=1)
    assert result.success,result.reason
    np.testing.assert_allclose(result.q,q,atol=1e-9)
    assert ik.last_retarget_status[side]['return_reference']=='vr_quick_reset'
    assert ik.last_retarget_status[side]['return_blend']==0


@pytest.mark.parametrize('side',['left','right'])
def test_actual_inward_hand_path_biases_only_proximal_posture(side):
    ik=make_ik();q=seed(ik,-25);grip=q.copy();home=seed(ik,-110)
    names=module.ARM_JOINTS[side]
    peak=0
    peak_speed=0
    for angle in np.linspace(-25,-110,250):
        desired=seed(ik,angle)
        result=ik.solve(target(ik,desired,side),q,feedback_q=q,posture_reference=grip,
                        reset_reference=home,session_epoch=1)
        assert result.success,result.reason
        peak_speed=max(peak_speed,float(np.max(np.abs(np.rad2deg(result.q-q))))/.01)
        q=result.q
        b=ik.last_retarget_status[side]['return_blend'];peak=max(peak,b)
        actual=ik.posture_tasks[side].target_q
        for j,name in enumerate(names):
            g=grip[ik.joint_q[name]];h=home[ik.joint_q[name]]
            expected=g+b*(h-g) if j<4 else g
            assert actual[ik._mj_q[name]]==pytest.approx(expected)
        # Uncommanded other arm, waist and neck never move.
        inactive=set(module.ALL_JOINTS)-set(names)
        for n in inactive: assert q[ik.joint_q[n]]==grip[ik.joint_q[n]]
    assert peak>.9
    assert result.position_error_m[side]<.015
    # The representative smooth inward motion must not hit the 90 deg/s
    # shoulder cap because of an abrupt recovery-weight onset.
    assert peak_speed<60


def test_failed_or_superseded_solve_does_not_advance_return_history(monkeypatch):
    ik=make_ik();q=seed(ik,-25);home=seed(ik,-110)
    result=ik.solve(target(ik,q),q,reset_reference=home,session_epoch=1,transactional=True)
    assert result.success and ik._return_shape_history=={}
    ik.commit_retarget_state()
    before=dict(ik._return_shape_history['left'])
    r=ik.solve(target(ik,home),q,reset_reference=home,session_epoch=1,transactional=True)
    assert r.success and ik._return_shape_history['left']==before
    def fail(*args,**kwargs): raise RuntimeError('fixture rejection')
    monkeypatch.setattr(module.mink,'solve_ik',fail)
    r=ik.solve(target(ik,home),q,reset_reference=home,session_epoch=1)
    assert not r.success and ik._return_shape_history['left']==before


def test_changing_live_reset_reference_rebases_return_without_jump():
    ik=make_ik();q=seed(ik,-25);home=seed(ik,-110)
    assert ik.solve(target(ik,q),q,reset_reference=home,session_epoch=1).success
    r=ik.solve(target(ik,home),q,reset_reference=home,session_epoch=1)
    assert r.success and ik.last_retarget_status['left']['return_blend']>0
    newhome=seed(ik,-100)
    r=ik.solve(target(ik,home),r.q,reset_reference=newhome,session_epoch=1)
    assert r.success and ik.last_retarget_status['left']['return_blend']==0
