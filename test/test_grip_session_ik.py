import numpy as np
import pytest
from ._subject import load_subject

m = load_subject('grip_session_ik')
NAMES = sum((list(m.ARM_NAMES[s]) for s in ('left', 'right')), [])
J = {name: i for i, name in enumerate(NAMES)}


def fresh():
    state = m.GripSessionIK()
    q = np.zeros(16)
    state.observe('session', dict(left=True, right=True), q, 1.0)
    return state, q


def test_reference_and_seed_ignore_encoder_noise_for_long_hold():
    state, q = fresh()
    rng = np.random.default_rng(12)
    for i in range(1000):
        measured = q + rng.normal(0, np.deg2rad(0.05), q.shape)
        ticket = state.prepare(('left', 'right'), measured, J, 1.01+i*.01)
        np.testing.assert_array_equal(ticket.reference[:14], q[:14])
        np.testing.assert_array_equal(ticket.seed[:14], q[:14])
        assert state.commit(ticket, ticket.seed)


def test_only_accepted_result_advances_history_and_inactive_joints_follow_feedback():
    state, q = fresh()
    t = state.prepare(('left',), q, J, 1.01)
    solved = q.copy(); solved[0] = .01
    # Simulate a stale solve: no commit.
    newer = q.copy(); newer[14:] = .2
    t2 = state.prepare(('left',), newer, J, 1.02)
    assert t2.seed[0] == 0
    np.testing.assert_array_equal(t2.seed[14:], newer[14:])
    assert state.commit(t2, solved)
    assert not state.commit(t, q)  # a delayed older result cannot rewind history
    t3 = state.prepare(('left',), newer, J, 1.03)
    assert t3.seed[0] == .01


def test_release_and_reset_fence_inflight_and_other_arm_keeps_reference():
    state, q = fresh()
    t = state.prepare(('left', 'right'), q, J, 1.01)
    state.observe('session', dict(left=False, right=True), q, 1.02)
    assert not state.commit(t, q)
    measured = q.copy(); measured[:7] = .1
    state.observe('session', dict(left=True, right=True), measured, 1.03)
    t = state.prepare(('left', 'right'), measured, J, 1.04)
    np.testing.assert_array_equal(t.reference[:7], measured[:7])
    np.testing.assert_array_equal(t.reference[7:14], q[7:14])
    state.reset()
    assert not state.commit(t, measured)


def test_feedback_error_does_not_rewrite_accepted_seed_or_reference():
    state, q = fresh()
    t = state.prepare(('left',), q, J, 1.01)
    ahead=q.copy(); ahead[:7]=np.deg2rad(9)
    assert state.commit(t, ahead)
    measured=q.copy(); measured[:7]=np.deg2rad(3)
    t = state.prepare(('left',), measured, J, 1.02)
    # Six degrees exceeds the old correction deadband. Feedback belongs in
    # constraints, not in a repeated blend that changes the continuous seed.
    np.testing.assert_array_equal(t.seed[:7], ahead[:7])
    np.testing.assert_array_equal(t.measured, measured)
    np.testing.assert_array_equal(t.reference[:7], q[:7])
    assert state.status()['correction_step_deg']['left'] == 0
    assert state.status()['feedback_strategy'] == 'qp_envelope'
    assert state.status()['mode'] == 'mink_quadprog_session_servo'


def test_long_motion_session_keeps_latched_reference_and_last_accepted_seed():
    state, q = fresh()
    accepted = q.copy()
    for i in range(1000):
        now = 1.01+i*.01
        measured = accepted.copy()
        measured[:7] -= np.deg2rad(6)
        ticket = state.prepare(('left',), measured, J, now)
        np.testing.assert_array_equal(ticket.reference[:7], q[:7])
        np.testing.assert_array_equal(ticket.seed[:7], accepted[:7])
        accepted = ticket.seed.copy()
        accepted[:7] = .15*np.sin(i*.01)
        assert state.commit(ticket, accepted)


def test_other_arm_edges_preserve_inflight_result_and_solver_history_key():
    state, q = fresh()
    ticket = state.prepare(('left',), q, J, 1.01)
    original_key = ticket.solver_epochs['left']
    state.observe('session', dict(left=True, right=False), q, 1.02)
    measured = q.copy(); measured[7:14] = .1
    state.observe('session', dict(left=True, right=True), measured, 1.03)
    assert state.valid(ticket)
    assert state.commit(ticket, ticket.seed)
    next_ticket = state.prepare(('left', 'right'), measured, J, 1.04)
    assert next_ticket.solver_epochs['left'] == original_key
    np.testing.assert_array_equal(next_ticket.reference[:7], q[:7])
    np.testing.assert_array_equal(next_ticket.reference[7:14], measured[7:14])


def test_regrip_same_pose_changes_only_that_arm_solver_key_and_fences_old_work():
    state, q = fresh()
    before = state.prepare(('left', 'right'), q, J, 1.01)
    state.observe('session', dict(left=False, right=True), q, 1.02)
    state.observe('session', dict(left=True, right=True), q, 1.03)
    after = state.prepare(('left', 'right'), q, J, 1.04)
    assert not state.commit(before, q)
    assert before.solver_epochs['right'] == after.solver_epochs['right']
    assert before.solver_epochs['left'] != after.solver_epochs['left']


def test_non_grip_measured_motion_is_latched_at_next_grip():
    state, q = fresh()
    state.observe('session', dict(left=False, right=False), q, 1.01)
    moved = q.copy(); moved[:14] = .12
    state.observe('session', dict(left=False, right=False), moved, 2.0)
    assert state.reference == {}
    assert state.accepted == {}
    state.observe('session', dict(left=True, right=False), moved, 2.01)
    ticket = state.prepare(('left',), moved, J, 2.02)
    np.testing.assert_array_equal(ticket.seed, moved)
    np.testing.assert_array_equal(ticket.reference, moved)


def test_same_timestamp_and_duplicate_results_cannot_rewind_accepted_state():
    state, q = fresh()
    first = state.prepare(('left',), q, J, 1.01)
    parallel = state.prepare(('left',), q, J, 1.01)
    solved = q.copy(); solved[0] = .01
    assert state.commit(first, solved)
    assert not state.commit(first, q)
    assert not state.commit(parallel, q)
    assert state.prepare(('left',), q, J, 1.02).seed[0] == .01


def test_failed_solve_does_not_advance_reference_seed_or_revision():
    state, q = fresh()
    ticket = state.prepare(('left',), q, J, 1.01)
    invalid = q.copy(); invalid[0] = np.nan
    assert not state.commit(ticket, invalid)
    next_ticket = state.prepare(('left',), q, J, 1.02)
    assert next_ticket.accepted_versions == ticket.accepted_versions
    np.testing.assert_array_equal(next_ticket.seed, ticket.seed)
    np.testing.assert_array_equal(next_ticket.reference, ticket.reference)


def test_measured_stall_guard_still_requires_deliberate_release():
    state, q = fresh()
    for i in range(151):
        ticket = state.prepare(('left',), q, J, 1.01+i*.01)
        ahead = q.copy(); ahead[0] = np.deg2rad(9)
        assert state.commit(ticket, ahead)
    with pytest.raises(m.SessionTrackingError, match='measured motion stalled'):
        state.prepare(('left',), q, J, 2.53)
    assert 'left' in state.blocked


def test_hard_error_blocks_until_observed_release_not_internal_hold():
    state, q = fresh()
    t=state.prepare(('left',),q,J,1.01)
    ahead=q.copy();ahead[0]=np.deg2rad(24)
    state.commit(t,ahead)
    with pytest.raises(m.SessionTrackingError): state.prepare(('left',),q,J,1.02)
    state.release(('left',))
    state.observe('session',dict(left=True,right=True),q,1.03)
    with pytest.raises(m.SessionTrackingError): state.prepare(('left',),q,J,1.04)
    state.observe('session',dict(left=False,right=True),q,1.05)
    state.observe('session',dict(left=True,right=True),q,1.06)
    assert state.prepare(('left',),q,J,1.07).seed[0] == 0


def test_timeout_rejects_old_state_and_new_session_latches_fresh():
    state,q=fresh()
    with pytest.raises(m.SessionTrackingError):state.prepare(('left',),q,J,1.5)
    q[:7]=.1
    state.observe('new-session',dict(left=True,right=False),q,1.6)
    t=state.prepare(('left',),q,J,1.61)
    np.testing.assert_array_equal(t.seed,q)


def test_nonfinite_configuration_and_parameters_rejected():
    with pytest.raises(ValueError):m.GripSessionIK(hard_error_deg=2)
    state,q=fresh();t=state.prepare(('left',),q,J,1.01)
    q[0]=np.nan
    assert not state.commit(t,q)
    with pytest.raises(ValueError):state.observe('new',dict(left=True,right=True),q,1.02)
