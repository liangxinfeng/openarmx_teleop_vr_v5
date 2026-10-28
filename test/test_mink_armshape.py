"""Arm-shape regressions with native FK/QP, never motor publishers."""
import numpy as np
import pytest

pytest.importorskip('mink')
pytest.importorskip('quadprog')
from ._subject import load_subject, source_directory

module = load_subject('mink_kinematics')


def make_ik():
    root = source_directory().parent
    return module.MinkKinematics(root/'urdf/robot_v3_1_simplified.urdf', collision_check=False)


def seed(ik, elbow=-80.):
    return ik.q_from_feedback(dict(leg_waist=np.zeros(4), neck=np.zeros(3),
        left_arm=np.array([20., 0., 0., elbow, 0., 0., 0.]),
        right_arm=np.array([20., 0., 0., elbow, 0., 0., 0.])))


def target(ik, q, side='left'):
    p=ik.eef_poses(q)[side]
    return {side: (p['position'], p['orientation'])}


@pytest.mark.parametrize('side', ['left', 'right'])
def test_captured_pose_is_a_fixed_point_with_no_new_arm_jump(side):
    ik=make_ik(); q=seed(ik)
    result=ik.solve(target(ik,q,side),q,feedback_q=q,session_epoch=1)
    assert result.success,result.reason
    np.testing.assert_allclose(result.q,q,atol=1.e-10)
    assert ik.last_retarget_status[side]['return_blend'] == pytest.approx(1.)
    assert abs(ik.last_retarget_status[side]['elbow_plane_error_m']) < 1.e-10


@pytest.mark.parametrize('side', ['left', 'right'])
def test_redundant_shape_perturbation_returns_without_a_cartesian_tail(side):
    ik=make_ik(); original=seed(ik)
    targets=target(ik,original,side)
    assert ik.solve(targets,original,feedback_q=original,session_epoch=1).success
    names=module.ARM_JOINTS[side]
    qi=np.array([ik.joint_q[n] for n in names]); vi=np.array([ik._mj_v[n] for n in names])
    frame=ik.tasks[side]
    jac=frame.compute_jacobian(ik.configuration)[:,vi]
    _,_,vt=np.linalg.svd(jac,full_matrices=True)
    perturbed=original.copy(); perturbed[qi]+=0.15*vt[-1]
    perturbed=np.clip(perturbed,ik.lower,ik.upper)
    # Start a fresh history from this physically displaced arm, preserving its
    # original shape reference. This tests actual recovery, not a hard reset.
    ik._reference_keys.clear()
    q=perturbed.copy()
    for _ in range(100):
        r=ik.solve(targets,q,feedback_q=q,posture_reference=original,
                   session_epoch=2,integration_dt=.01)
        assert r.success,r.reason
        q=r.q
    assert np.linalg.norm(q[qi]-original[qi]) < .15*np.linalg.norm(perturbed[qi]-original[qi])
    assert r.position_error_m[side] < .004
    assert r.orientation_error_rad[side] < np.deg2rad(.19)


def test_straight_arm_keeps_rotational_reference_and_finite_nonzero_orientation_cost():
    ik=make_ik(); q=seed(ik,-3.)
    result=ik.solve(target(ik,q),q,feedback_q=q,session_epoch=1)
    assert result.success,result.reason
    status=ik.last_retarget_status['left']
    assert 0. < status['orientation_cost'] <= ik._orientation_weight
    assert np.isfinite(ik.elbow_plane_tasks['left'].compute_jacobian(ik.configuration)).all()
    costs=ik.posture_tasks['left'].cost
    assert costs[ik._mj_v[module.LEFT_ARM_JOINTS[2]]] >= ik._posture_weight


def test_orientation_only_request_is_still_followed():
    ik=make_ik(); original=seed(ik)
    desired=original.copy()
    desired[ik.joint_q[module.LEFT_ARM_JOINTS[-1]]]+=np.deg2rad(15.)
    targets=target(ik,desired)
    q=original.copy()
    for _ in range(120):
        r=ik.solve(targets,q,feedback_q=q,posture_reference=original,session_epoch=1)
        assert r.success,r.reason
        q=r.q
    assert r.position_error_m['left'] < .004
    assert r.orientation_error_rad['left'] < np.deg2rad(.5)


def test_projected_outward_request_never_reports_the_raw_goal_reached():
    ik=make_ik(); q=seed(ik,-4.)
    targets=target(ik,q)
    position=np.asarray(targets['left'][0]).copy()
    shoulder=ik.configuration.data.xanchor[ik._shoulder_joint_ids['left']].copy()
    # Use a native configuration before reading its joint centres.
    ik.configuration.update(ik._mujoco_from_guard(q))
    shoulder=ik.configuration.data.xanchor[ik._shoulder_joint_ids['left']].copy()
    axis=position-shoulder;axis/=np.linalg.norm(axis)
    targets['left']=(position+.05*axis,targets['left'][1])
    result=ik.solve(targets,q,feedback_q=q,session_epoch=1)
    assert result.success,result.reason
    assert ik.last_retarget_status['left']['reach_projection_m'] > .01
    assert result.position_error_m['left'] > .03
    assert not ik.last_retarget_status['left']['converged']


def test_inward_request_remains_available_at_straight_arm():
    ik=make_ik(); q=seed(ik,-4.)
    ik.configuration.update(ik._mujoco_from_guard(q))
    targets=target(ik,q)
    position=np.asarray(targets['left'][0]).copy()
    shoulder=ik.configuration.data.xanchor[ik._shoulder_joint_ids['left']].copy()
    axis=position-shoulder;axis/=np.linalg.norm(axis)
    targets['left']=(position-.03*axis,targets['left'][1])
    result=ik.solve(targets,q,feedback_q=q,session_epoch=1)
    assert result.success,result.reason
    assert ik.last_retarget_status['left']['reach_projection_m'] == 0.
