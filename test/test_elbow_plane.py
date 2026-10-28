"""Native kinematic checks for the single-coordinate elbow-plane task."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mink')
mujoco = pytest.importorskip('mujoco')
import mink

from ._subject import load_subject, source_directory


ElbowPlaneTask = load_subject('elbow_plane').ElbowPlaneTask
build_model = load_subject('mujoco_urdf_model').build_mujoco_model


@pytest.fixture
def model():
    package_root = source_directory().parent
    return build_model(package_root/'urdf'/'robot_v3_1_simplified.urdf', {
        'left': 'Link_Left_Wrist_Lower_to_Gripper',
        'right': 'Link_Right_Wrist_Lower_to_Gripper',
    })


def seed(model, side='left', elbow=-55.):
    configuration = mink.Configuration(model)
    q = model.qpos0.copy()
    angles = [12., 28. if side == 'left' else -28., 17., elbow, -14., 8., 4.]
    for part, angle in zip((
            'Shoulder_Inner', 'Shoulder_Outer', 'UpperArm', 'Elbow',
            'Forearm', 'Wrist_Upper', 'Wrist_Lower'), angles):
        jid = model.joint(f'Joint_{side.title()}_{part}').id
        q[model.jnt_qposadr[jid]] = np.deg2rad(angle)
    configuration.update(q)
    return configuration


@pytest.mark.parametrize('side', ['left', 'right'])
def test_reference_has_zero_error_and_one_coordinate(model, side):
    configuration = seed(model, side)
    task = ElbowPlaneTask(model, side, cost=.7)
    task.set_reference(configuration.data)
    assert isinstance(task, mink.Task)
    assert task.gain == pytest.approx(.6)
    np.testing.assert_array_equal(task.cost, [.7])
    assert task.compute_error(configuration).shape == (1,)
    assert abs(task.compute_error(configuration)[0]) < 1.e-12
    assert task.plane_error_m == pytest.approx(0., abs=1.e-12)
    assert task.compute_jacobian(configuration).shape == (1, model.nv)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_native_jacobian_matches_central_configuration_difference(model, side):
    configuration = seed(model, side)
    task = ElbowPlaneTask(model, side)
    task.set_reference(configuration.data)
    q = configuration.q.copy()
    for part, delta in [('Shoulder_Inner', .14), ('UpperArm', -.18), ('Elbow', .11)]:
        jid = model.joint(f'Joint_{side.title()}_{part}').id
        q[model.jnt_qposadr[jid]] += delta
    configuration.update(q)
    computed = task.compute_jacobian(configuration)[0]
    expected = np.zeros(model.nv)
    step = 2.e-6
    for dof in task._active_dofs:
        velocity = np.zeros(model.nv)
        velocity[dof] = 1.
        plus, minus = q.copy(), q.copy()
        mujoco.mj_integratePos(model, plus, velocity, step)
        mujoco.mj_integratePos(model, minus, velocity, -step)
        configuration.update(plus)
        positive = task.compute_error(configuration)[0]
        configuration.update(minus)
        negative = task.compute_error(configuration)[0]
        expected[dof] = (positive-negative)/(2.*step)
    configuration.update(q)
    np.testing.assert_allclose(computed, expected, atol=2.e-7, rtol=3.e-5)
    assert np.linalg.norm(computed) > 1.e-4


@pytest.mark.parametrize('side', ['left', 'right'])
def test_other_arm_waist_and_neck_columns_are_zero(model, side):
    configuration = seed(model, side)
    task = ElbowPlaneTask(model, side)
    task.set_reference(configuration.data)
    jacobian = task.compute_jacobian(configuration)
    inactive = np.setdiff1d(np.arange(model.nv), task._active_dofs)
    np.testing.assert_array_equal(jacobian[:, inactive], 0.)


@pytest.mark.parametrize('elbow', [-3., -.001, 0.])
def test_near_straight_reference_and_evaluation_remain_finite(model, elbow):
    configuration = seed(model, elbow=elbow)
    task = ElbowPlaneTask(model, 'left')
    task.set_reference(configuration.data)
    assert np.all(np.isfinite(task.compute_error(configuration)))
    assert np.all(np.isfinite(task.compute_jacobian(configuration)))
    assert abs(task.compute_error(configuration)[0]) < 1.e-11


def test_collinear_reference_uses_joint_axis_and_antipodal_is_finite(model):
    configuration = seed(model)
    task = ElbowPlaneTask(model, 'left')
    anchors = configuration.data.xanchor.copy()
    s_id, e_id, w_id = task._landmarks
    shoulder = np.array([.1, .2, .3])
    anchors[s_id] = shoulder
    anchors[e_id] = shoulder+np.array([.15, 0., 0.])
    anchors[w_id] = shoulder+np.array([.4, 0., 0.])
    task.set_reference(SimpleNamespace(
        xanchor=anchors, xaxis=configuration.data.xaxis.copy()))
    assert np.isfinite(task._reference_bend).all()
    assert np.linalg.norm(task._reference_bend) == pytest.approx(1.)
    for epsilon in [0., 1.e-10, -1.e-10, 1.e-4, -1.e-4]:
        positions = np.array([
            shoulder, shoulder+[-.15, .03, .02],
            shoulder+[-.4, epsilon, 0.],
        ])
        assert np.isfinite(task._geometric_error(positions))
        bend = task._transport_bend(
            (positions[2]-positions[0])/np.linalg.norm(positions[2]-positions[0]))
        assert np.isfinite(bend).all()
        assert np.linalg.norm(bend) == pytest.approx(1.)


def test_short_axis_and_unset_reference_produce_finite_zero_task(model):
    configuration = seed(model)
    task = ElbowPlaneTask(model, 'left')
    np.testing.assert_array_equal(task.compute_error(configuration), [0.])
    np.testing.assert_array_equal(task.compute_jacobian(configuration), 0.)
    task.set_reference(configuration.data)
    anchors = configuration.data.xanchor.copy()
    s_id, _, w_id = task._landmarks
    anchors[w_id] = anchors[s_id]
    synthetic = SimpleNamespace(data=SimpleNamespace(xanchor=anchors))
    np.testing.assert_array_equal(task.compute_error(synthetic), [0.])
    np.testing.assert_array_equal(task.compute_jacobian(synthetic), 0.)


def test_reference_is_not_mutated_by_task_evaluation(model):
    configuration = seed(model)
    task = ElbowPlaneTask(model, 'left')
    task.set_reference(configuration.data)
    reference_axis = task._reference_axis.copy()
    reference_bend = task._reference_bend.copy()
    q = configuration.q.copy()
    jid = model.joint('Joint_Left_UpperArm').id
    q[model.jnt_qposadr[jid]] += .2
    configuration.update(q)
    for _ in range(3):
        task.compute_error(configuration)
        task.compute_jacobian(configuration)
    np.testing.assert_array_equal(task._reference_axis, reference_axis)
    np.testing.assert_array_equal(task._reference_bend, reference_bend)
