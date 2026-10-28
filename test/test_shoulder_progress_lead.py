"""Bounded shoulder response and fail-closed behavior; no motor access."""
import ast
from pathlib import Path
import numpy as np
import pytest
from ._subject import load_subject

dynamic=load_subject('adaptive_position').DynamicCommandLead


def update(guard,vel,command=11.6,goal=12.,dt=.01):
    return guard.update([0.],[goal],[vel],[command],dt,
        minimum=12.,nominal=12.,maximum=14.,speed_low=5.,speed_high=40.,
        target_error_low=2.,target_error_high=10.,following_error_soft=12.,
        following_error_hard=16.,expansion_filter_tau=.08,require_progress=True)[0][0]


def test_extra_lead_requires_real_motion_toward_goal_and_healthy_feedback():
    guard=dynamic(1,[0]);guard.reset(12.)
    for _ in range(100):lead=update(guard,34.)
    assert 13.5<lead<=14.
    assert update(guard,0.)==pytest.approx(12.)
    for _ in range(100):update(guard,34.)
    assert update(guard,-34.)==pytest.approx(12.)
    for _ in range(100):update(guard,34.)
    assert update(guard,34.,command=16.5)==pytest.approx(12.)


def test_no_extra_lead_when_goal_is_reached_even_if_joint_moving():
    guard=dynamic(1,[0]);guard.reset(12.)
    for _ in range(100):assert update(guard,34.,goal=0.)==pytest.approx(12.)


def test_extra_shoulder_envelope_never_expands_other_joints():
    pytest.importorskip('mink')
    from .test_mink_armshape import make_ik,seed,target
    module=load_subject('mink_kinematics');ik=make_ik();measured=seed(ik,-110);q=measured.copy()
    names=module.ARM_JOINTS['right'];idx=np.array([ik.joint_q[n] for n in names]);goal=measured.copy()
    goal[idx[0]]+=np.deg2rad(32.);goal[idx[4]]+=np.deg2rad(25.)
    targets=target(ik,goal,'right')
    for _ in range(120):
        r=ik.solve(targets,q,feedback_q=measured,posture_reference=measured,
                   maximum_arm_lead_rad=np.deg2rad(12.),
                   maximum_shoulder_lead_rad=np.deg2rad(14.),session_epoch=1)
        assert r.success,r.reason
        q=r.q
        diff=np.abs(np.rad2deg(q[idx]-measured[idx]))
        assert np.max(diff[:2])<=14.00001
        assert np.max(diff[2:])<=12.00001
    assert max(diff[:2])>12.


def test_rejects_shoulder_limit_above_fixed_cap():
    pytest.importorskip('mink')
    from .test_mink_armshape import make_ik,seed,target
    ik=make_ik();q=seed(ik,-110)
    r=ik.solve(target(ik,q),q,maximum_arm_lead_rad=np.deg2rad(12.),
               maximum_shoulder_lead_rad=np.deg2rad(14.01))
    assert not r.success and 'shoulder feedback lead' in r.reason


def test_extra_allowance_only_during_live_cartesian_grip():
    src=(Path(__file__).parents[1]/'openarmx_teleop_vr_306_v5/controller_node.py').read_text()
    ast.parse(src)
    assert "not target_stale and not self._reset_active" in src
    assert "self._target_source == 'cartesian_ik'" in src
    assert 'side in self._grip_ik.reference and side in self._velocity_active_sides' in src
    assert "maximum_lead[4:18] = fixed_arm_lead" in src
