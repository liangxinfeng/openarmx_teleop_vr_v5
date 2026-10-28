"""Native kinematics/QP checks.  These tests never create motor publishers."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mink')
pytest.importorskip('quadprog')

from ._subject import load_subject, source_directory


module = load_subject('mink_kinematics')
MinkKinematics = module.MinkKinematics


@pytest.fixture
def ik():
    source = source_directory().parent
    return MinkKinematics(
        source/'urdf'/'robot_v3_1_simplified.urdf',
        source/'urdf'/'robot_v3_1.srdf', collision_check=False)


def seed_for(ik):
    return ik.q_from_feedback({
        'leg_waist': np.zeros(4), 'left_arm': np.array([0, 20, 0, -50, 0, 0, 0]),
        'right_arm': np.array([0, -20, 0, -50, 0, 0, 0]), 'neck': np.zeros(3),
    })


def targets_for(ik, q, sides=('left',), dz=0.01):
    poses = ik.eef_poses(q)
    output = {}
    for side in sides:
        pose = poses[side]
        position = np.asarray(pose['position']).copy()
        position[2] += dz
        output[side] = (position, np.asarray(pose['orientation']).copy())
    return output


def test_single_arm_never_moves_waist_head_or_other_arm(ik):
    seed = seed_for(ik)
    result = ik.solve(targets_for(ik, seed), seed, feedback_q=seed)
    assert result.success, result.reason
    assert result.iterations == 1
    for name in module.LEG_WAIST_JOINTS+module.NECK_JOINTS+module.RIGHT_ARM_JOINTS:
        assert result.q[ik.joint_q[name]] == seed[ik.joint_q[name]]
    indices = [ik.joint_q[name] for name in module.LEFT_ARM_JOINTS]
    assert np.linalg.norm(result.q[indices]-seed[indices]) > 1.e-8
    assert ik.last_retarget_status['left']['solver'] == 'mink/quadprog'


def test_constructor_and_setter_reject_waist_ik(ik):
    assert not ik.allow_waist
    ik.allow_waist = False
    with pytest.raises(ValueError, match='seven-DoF'):
        ik.allow_waist = True
    with pytest.raises(ValueError, match='seven-DoF'):
        MinkKinematics('unused.urdf', allow_waist=True)


def test_only_one_qp_even_when_legacy_iteration_budget_is_large(ik, monkeypatch):
    seed = seed_for(ik)
    original = module.mink.solve_ik
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs['solver'])
        return original(*args, **kwargs)

    monkeypatch.setattr(module.mink, 'solve_ik', record)
    result = ik.solve(targets_for(ik, seed), seed, max_iterations=100)
    assert result.success, result.reason
    assert calls == ['quadprog']


def test_transaction_does_not_advance_history_until_commit(ik):
    seed = seed_for(ik)
    result = ik.solve(targets_for(ik, seed), seed, transactional=True,
                      session_epoch={'left': (0, 1)}, feedback_q=seed)
    assert result.success, result.reason
    assert ik._previous_commands == {}
    assert ik._target_history == {}
    assert ik._pending_retarget is not None
    assert ik.last_retarget_status['left']['accepted'] is False
    assert ik.last_retarget_status['left']['attempt_status'] == 'pending'
    ik.commit_retarget_state()
    indices = [ik.joint_q[name] for name in module.LEFT_ARM_JOINTS]
    np.testing.assert_allclose(ik._previous_commands['left'], result.q[indices])
    assert ik._pending_retarget is None
    assert ik.last_retarget_status['left']['accepted'] is True
    assert ik.last_retarget_status['left']['attempt_status'] == 'accepted'


def test_superseded_transaction_cannot_be_committed_after_failed_solve(ik, monkeypatch):
    seed = seed_for(ik)
    assert ik.solve(targets_for(ik, seed), seed, transactional=True).success

    def fail(*args, **kwargs):
        raise RuntimeError('deliberate QP failure')

    monkeypatch.setattr(module.mink, 'solve_ik', fail)
    failed = ik.solve(targets_for(ik, seed), seed, transactional=True)
    assert not failed.success
    assert 'deliberate QP failure' in failed.reason
    assert ik.last_retarget_status['left']['accepted'] is False
    assert ik.last_retarget_status['left']['attempt_status'] == 'rejected'
    ik.commit_retarget_state()
    assert ik._previous_commands == {}
    assert ik._session_references == {}


def test_same_grip_token_freezes_posture_even_if_incoming_reference_changes(ik):
    seed = seed_for(ik)
    targets = targets_for(ik, seed)
    first = ik.solve(targets, seed, posture_reference=seed, session_epoch={'left': (0, 1)})
    assert first.success, first.reason
    reference = seed.copy()
    reference[ik.joint_q[module.LEFT_ARM_JOINTS[0]]] += 0.15
    second = ik.solve(targets, first.q, feedback_q=first.q,
                      posture_reference=reference, session_epoch={'left': (0, 1)})
    assert second.success, second.reason
    indices = [ik.joint_q[name] for name in module.LEFT_ARM_JOINTS]
    np.testing.assert_array_equal(ik._session_references['left'], seed[indices])
    for name in module.LEFT_ARM_JOINTS:
        assert ik.posture_tasks['left'].target_q[ik._mj_q[name]] == seed[ik.joint_q[name]]


def test_other_arm_grip_does_not_reset_existing_reference(ik):
    seed = seed_for(ik)
    targets = targets_for(ik, seed, ('left', 'right'))
    first = ik.solve(targets, seed, session_epoch={'left': (0, 1), 'right': (0, 1)})
    assert first.success, first.reason
    frozen = ik._session_references['left'].copy()
    left_key = copy.deepcopy(ik._reference_keys['left'])
    reference = first.q.copy()
    second = ik.solve(targets, first.q, feedback_q=first.q, posture_reference=reference,
                      session_epoch={'left': (0, 1), 'right': (0, 2)})
    assert second.success, second.reason
    assert ik._reference_keys['left'] == left_key
    np.testing.assert_array_equal(ik._session_references['left'], frozen)
    right_indices = [ik.joint_q[name] for name in module.RIGHT_ARM_JOINTS]
    np.testing.assert_array_equal(ik._session_references['right'], reference[right_indices])


def test_final_commands_obey_speed_acceleration_and_feedback_lead(ik):
    measured = seed_for(ik)
    seed = measured.copy()
    targets = targets_for(ik, seed, ('left', 'right'), dz=0.12)
    previous = seed.copy()
    previous_velocity = {side: np.zeros(7) for side in module.ARM_JOINTS}
    lead = np.deg2rad(2.0)
    for _ in range(35):
        result = ik.solve(targets, seed, feedback_q=measured, integration_dt=0.01,
                          maximum_arm_lead_rad=lead,
                          posture_reference=measured, session_epoch=1)
        assert result.success, result.reason
        for side, names in module.ARM_JOINTS.items():
            idx = [ik.joint_q[name] for name in names]
            delta = result.q[idx]-previous[idx]
            velocity = delta/0.01
            assert np.all(np.abs(velocity) <= ik._velocity+1.e-6)
            assert np.max(np.abs(result.q[idx]-measured[idx])) <= lead+1.e-7
            if not ik.last_retarget_status[side]['acceleration_recovery']:
                assert np.all(np.abs(velocity-previous_velocity[side]) <= ik._acceleration*0.01+1.e-5)
            assert np.all(result.q[idx] >= ik.lower[idx]-1.e-7)
            assert np.all(result.q[idx] <= ik.upper[idx]+1.e-7)
            previous_velocity[side] = velocity
        previous = result.q.copy()
        seed = result.q.copy()


def test_large_feedback_displacement_fails_without_advancing_history(ik):
    seed = seed_for(ik)
    targets = targets_for(ik, seed)
    first = ik.solve(targets, seed, session_epoch=1)
    assert first.success, first.reason
    previous = ik._previous_commands['left'].copy()
    measured = first.q.copy()
    measured[ik.joint_q[module.LEFT_ARM_JOINTS[0]]] += 0.5
    result = ik.solve(targets, first.q, feedback_q=measured,
                      maximum_arm_lead_rad=0.1, session_epoch=1)
    assert not result.success
    assert 'envelope infeasible' in result.reason
    np.testing.assert_array_equal(ik._previous_commands['left'], previous)


def test_collision_rejection_does_not_advance_history(ik, monkeypatch):
    seed = seed_for(ik)
    monkeypatch.setattr(ik, 'collision_pairs', lambda q: [('left_hand', 'torso')])
    result = ik.solve(targets_for(ik, seed), seed, session_epoch=1)
    assert not result.success
    assert result.collision_pairs == [('left_hand', 'torso')]
    assert ik._previous_commands == {}


def test_plateau_is_not_successful_convergence_and_target_motion_resets_it(ik, monkeypatch):
    seed = seed_for(ik)
    targets = targets_for(ik, seed, dz=0.05)
    monkeypatch.setattr(module.mink, 'solve_ik',
                        lambda configuration, *args, **kwargs: np.zeros(configuration.model.nv))
    for _ in range(35):
        result = ik.solve(targets, seed, session_epoch=1)
        assert result.success  # Accepted zero-motion step, not target reached.
    status = ik.last_retarget_status['left']
    assert status['plateau']
    assert not status['converged']
    assert status['state'] == 'plateau'
    assert 'not converged' in result.reason
    changed = targets_for(ik, seed, dz=0.06)
    assert ik.solve(changed, seed, session_epoch=1).success
    assert not ik.last_retarget_status['left']['plateau']


def test_convergence_reports_point19_degrees_without_blocking(ik):
    seed = seed_for(ik)
    result = ik.solve(targets_for(ik, seed, dz=0.0), seed)
    assert result.success
    status = ik.last_retarget_status['left']
    assert status['orientation_tolerance_deg'] == pytest.approx(0.19)
    assert status['converged']
    assert not status['plateau']
    assert result.iterations == 1


@pytest.mark.parametrize('bad_dt', [0.0, -0.01, float('nan')])
def test_invalid_timestep_fails_closed(ik, bad_dt):
    seed = seed_for(ik)
    result = ik.solve(targets_for(ik, seed), seed, integration_dt=bad_dt)
    assert not result.success
    assert ik._previous_commands == {}


def test_invalid_target_does_not_create_history(ik):
    seed = seed_for(ik)
    targets = {'left': (np.array([0.3, np.nan, 0.4]), np.array([0, 0, 0, 1]))}
    assert not ik.solve(targets, seed, session_epoch=1).success
    assert ik._previous_commands == {}


def test_effective_limits_and_task_types_are_configured(ik):
    assert isinstance(ik.damping_task, module.mink.DampingTask)
    assert isinstance(ik.configuration_limit, module.mink.ConfigurationLimit)
    assert isinstance(ik.velocity_limit, module.mink.VelocityLimit)
    for side in module.ARM_JOINTS:
        assert isinstance(ik.tasks[side], module.mink.FrameTask)
        assert isinstance(ik.posture_tasks[side], module.mink.PostureTask)
    assert not hasattr(ik, 'elbow_tasks')
    assert not hasattr(ik, 'manipulability_tasks')
    for name in module.ALL_JOINTS:
        jid = ik.mujoco_model.joint(name).id
        np.testing.assert_allclose(ik.mujoco_model.jnt_range[jid],
                                   [ik.lower[ik.joint_q[name]], ik.upper[ik.joint_q[name]]])


def test_scalar_intersection_preserves_tightest_bounds_and_freezes_by_equality():
    model = SimpleNamespace(nv=3)

    class Box:
        def __init__(self, lower, upper):
            self.lower, self.upper = np.asarray(lower), np.asarray(upper)

        def compute_qp_inequalities(self, configuration, dt):
            return module.Constraint(G=np.vstack((np.eye(3), -np.eye(3))),
                                     h=np.concatenate((self.upper, -self.lower)))

    limits = [Box([-1, -1, -1], [1, 1, 1]),
              Box([-.1, -.1, -.1], [.1, .1, .1]),
              Box([.02, 0, -.03], [.03, 0, .04])]
    limit = module._ScalarLimitIntersection(model, limits, {0, 2})
    result = limit.compute_qp_inequalities(None, .01)
    assert result.G.shape == (4, 3)
    np.testing.assert_array_equal(result.G, [[1, 0, 0], [0, 0, 1],
                                          [-1, 0, 0], [0, 0, -1]])
    np.testing.assert_allclose(result.h, [.03, .04, -.02, .03])
    frozen = module._FrozenDofsConstraint(model, {0, 2})
    np.testing.assert_array_equal(frozen.compute_jacobian(None), [[0, 1, 0]])
    np.testing.assert_array_equal(frozen.compute_error(None), [0])


def test_impossible_scalar_intersection_is_rejected():
    model = SimpleNamespace(nv=1)

    class Impossible:
        def compute_qp_inequalities(self, configuration, dt):
            return module.Constraint(G=np.array([[1.], [-1.]]), h=np.array([.1, -.2]))

    with pytest.raises(ValueError, match='infeasible'):
        module._ScalarLimitIntersection(model, [Impossible()], {0}).compute_qp_inequalities(None, .01)
