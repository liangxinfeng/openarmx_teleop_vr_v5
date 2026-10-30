"""Execute production boundary methods without ROS or robot dependencies."""
import ast
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from ._subject import load_subject


ROOT = Path(__file__).resolve().parents[1] / 'openarmx_teleop_vr_306_v5'
SESSION = load_subject('grip_session_ik')
JOINT_Q = {name: index for index, name in enumerate(
    list(SESSION.ARM_NAMES['left']) + list(SESSION.ARM_NAMES['right']))}


def production_methods(filename, names):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == set(names)
    cls = ast.ClassDef(name='ProductionMethods', bases=[], keywords=[],
                       body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {'np': np, 'json': json, 'time': time, 'String': SimpleNamespace}
    exec(compile(module, filename, 'exec'), scope)
    return scope['ProductionMethods']


Controller = production_methods('controller_node.py', [
    '_on_release_hold', '_apply_clutch_envelope_locked', '_hold_sides_locked',
    '_hold_failed_ik_locked', '_ik_worker_loop',
])
Mapper = production_methods('vr_mapper_node.py', ['_publish_release_hold'])


class Limiter:
    def __init__(self):
        self.calls = []

    def reset_indices(self, indices, values):
        self.calls.append((list(indices), np.array(values)))


class VelocityEstimator(Limiter):
    def __init__(self):
        super().__init__()
        self.velocity = np.zeros(21)

    def reset_indices(self, indices, values):
        super().reset_indices(indices, values)
        self.velocity[list(indices)] = 0


class Mailbox:
    def __init__(self, sequence=7):
        self.sequence = sequence
        self.samples = []

    def is_latest(self, sequence):
        return sequence == self.sequence

    def take(self):
        return self.samples.pop(0) if self.samples else None


def make_controller():
    node = Controller()
    node._lock = threading.RLock()
    node._clutch_session = 'browser-a'
    node._clutch_sequence = 10
    node._reset_active = node._enable_pending = node._estop_latched = False
    node._hardware_enabled = True
    node._state = 'ARMED'
    node._reason = ''
    node._output_control_mode = 'position'
    node._grip_release_held = set()
    node._waist_follow_enabled = False
    node._waist_follow_anchor_x = {'left': None, 'right': None}
    node._waist_follow_extension_m = 0
    node._velocity_servo = None
    node._velocity_active_sides = {'left', 'right'}
    node._latest_target_mailbox = Mailbox()
    node._limiter = Limiter()
    node._arm_smoother = SimpleNamespace(rebase_side=lambda side, values: None)
    node._target_velocity_estimator = VelocityEstimator()
    node._target_velocity_estimate = np.zeros(21)
    measured = dict(left_arm=np.ones(7), right_arm=np.full(7, 2.),
                    leg_waist=np.zeros(4), neck=np.zeros(3))
    node._feedback = SimpleNamespace(as_dict=lambda: measured)
    node._feedback_time = time.monotonic()
    node._ik_kinematics = SimpleNamespace(q_from_feedback=lambda groups: np.deg2rad(
        np.concatenate([groups['left_arm'], groups['right_arm'],
                        groups['leg_waist'], groups['neck']])))
    node._grip_ik = SESSION.GripSessionIK()
    node._grip_ik.observe(node._clutch_session, {'left': True, 'right': True},
                         node._ik_kinematics.q_from_feedback(measured),
                         node._feedback_time)
    node._target_groups = {key: value.copy() for key, value in measured.items()}
    node._target_groups['left_arm'][:] = 5.
    node._target_groups['right_arm'][:] = 6.
    node.get_parameter = lambda name: SimpleNamespace(value=.3)
    node.faults = []
    node._set_hardware_fault = lambda reason: node.faults.append(reason)
    node.errors = []
    node.get_logger = lambda: SimpleNamespace(error=node.errors.append)
    return node


def envelope(sequence, left=True, right=True, session='browser-a'):
    return dict(clutch_session=session, clutch_sequence=sequence,
                clutch_state=dict(left=left, right=right))


def release(node, sequence, session='browser-a', state=None):
    payload = envelope(sequence, left=False, session=session)
    payload['sides'] = ['left']
    if state is not None:
        payload['clutch_state'] = state
    node._on_release_hold(SimpleNamespace(data=json.dumps(payload)))


def test_mapper_release_uses_same_clock_and_complete_other_arm_state():
    mapper = Mapper()
    mapper._hardware_enabled = True
    mapper._release_hold_sent = dict(left=False, right=False)
    mapper._clutch_session = 'browser-a'
    mapper._clutch_sequence = 10
    mapper._mappers = {side: SimpleNamespace(latched=True) for side in ('left', 'right')}
    sent = []
    mapper._release_pub = SimpleNamespace(publish=lambda msg: sent.append(json.loads(msg.data)))
    mapper._publish_release_hold('left')
    assert sent[0]['clutch_sequence'] == mapper._clutch_sequence == 11
    assert sent[0]['clutch_session'] == 'browser-a'
    assert sent[0]['clutch_state'] == dict(left=False, right=True)
    mapper._publish_release_hold('left')
    assert len(sent) == 1


def test_fresh_release_holds_only_released_arm_and_updates_authoritative_state():
    node = make_controller()
    right_reference = node._grip_ik.reference['right'].copy()
    release(node, 11)
    assert node._clutch_sequence == 11
    assert 'left' not in node._grip_ik.reference
    assert not node._grip_ik.down['left']
    np.testing.assert_array_equal(node._grip_ik.reference['right'], right_reference)
    np.testing.assert_array_equal(node._target_groups['left_arm'], np.ones(7))
    np.testing.assert_array_equal(node._target_groups['right_arm'], np.full(7, 6.))
    assert node._velocity_active_sides == {'right'}


def test_late_release_cannot_erase_new_grip_reference():
    node = make_controller()
    release(node, 11)
    node._apply_clutch_envelope_locked(envelope(12), time.monotonic())
    reference = node._grip_ik.reference['left'].copy()
    token = node._grip_ik.side_epochs['left']
    release(node, 11)
    release(node, 999, session='older-browser')
    assert node._clutch_sequence == 12
    assert node._grip_ik.side_epochs['left'] == token
    np.testing.assert_array_equal(node._grip_ik.reference['left'], reference)


def test_release_requires_declared_side_false_and_rejects_legacy_event():
    node = make_controller()
    release(node, 11, state=dict(left=True, right=True))
    node._on_release_hold(SimpleNamespace(data=json.dumps({'sides': ['left']})))
    assert node._clutch_sequence == 10
    assert 'left' in node._grip_ik.reference


def test_qp_failure_immediately_replaces_old_position_goal_and_requires_regrip():
    node = make_controller()
    measured = node._ik_kinematics.q_from_feedback(node._feedback.as_dict())
    ticket = node._grip_ik.prepare(['left'], measured, JOINT_Q, time.monotonic())
    assert node._hold_failed_ik_locked(['left'], 'IK rejected: infeasible',
        time.monotonic(), sequence=7, ticket=ticket)
    np.testing.assert_array_equal(node._target_groups['left_arm'], np.ones(7))
    np.testing.assert_array_equal(node._target_groups['right_arm'], np.full(7, 6.))
    assert node._state == 'HOLDING'
    assert 'left' in node._grip_ik.blocked
    assert 'release Grip' in node._reason
    node._apply_clutch_envelope_locked(envelope(11), time.monotonic())
    assert 'left' not in node._grip_ik.reference
    node._apply_clutch_envelope_locked(envelope(12, left=False), time.monotonic())
    node._apply_clutch_envelope_locked(envelope(13), time.monotonic())
    assert 'left' in node._grip_ik.reference
    assert 'left' not in node._grip_ik.blocked


def test_old_inflight_failure_cannot_hold_a_new_grip_or_new_frame():
    node = make_controller()
    measured = node._ik_kinematics.q_from_feedback(node._feedback.as_dict())
    ticket = node._grip_ik.prepare(['left'], measured, JOINT_Q, time.monotonic())
    release(node, 11)
    node._apply_clutch_envelope_locked(envelope(12), time.monotonic())
    node._target_groups['left_arm'][:] = 8.
    assert not node._hold_failed_ik_locked(['left'], 'old failure',
        time.monotonic(), sequence=7, ticket=ticket)
    assert not node._hold_failed_ik_locked(['left'], 'old envelope failure',
        time.monotonic(), sequence=7, payload=envelope(10))
    node._latest_target_mailbox.sequence = 8
    assert not node._hold_failed_ik_locked(['left'], 'old frame failure',
        time.monotonic(), sequence=7, payload=envelope(12))
    np.testing.assert_array_equal(node._target_groups['left_arm'], np.full(7, 8.))
    assert 'left' not in node._grip_ik.blocked


def test_failure_with_stale_feedback_faults_instead_of_reusing_old_pose():
    node = make_controller()
    node._feedback_time -= 1.
    assert not node._hold_failed_ik_locked(['left'], 'IK exception',
        time.monotonic(), sequence=7, payload=envelope(10))
    assert node.faults and 'fresh feedback hold unavailable' in node.faults[0]


def test_worker_survives_unexpected_commit_exception_and_holds_current_goal():
    node = make_controller()
    payload = envelope(10)
    payload['pos_left_in_robot'] = [0., 0., 0.]
    data = json.dumps(payload)
    node._latest_target_mailbox.samples = [(7, (data, time.monotonic()))]*2
    calls = []

    def process(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('conversion failed')

    node._process_latest_target = process
    node._ik_worker_loop()
    assert len(calls) == 2
    np.testing.assert_array_equal(node._target_groups['left_arm'], np.ones(7))
    assert node.errors and not node.faults


def test_reanchor_resets_only_selected_arm_limiter_indices():
    source = (ROOT / 'controller_node.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == '_process_latest_target')
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Attribute)
             and node.func.value.attr == '_limiter']
    assert any(call.func.attr == 'reset_indices' for call in calls)
    assert all(call.func.attr != 'reset' for call in calls)
