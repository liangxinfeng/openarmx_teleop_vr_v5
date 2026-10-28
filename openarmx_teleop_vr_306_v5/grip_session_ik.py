"""Grip state and accepted IK history. All calls are under the controller lock.

No solver mutation occurs here. Tickets fence late results across release,
reset and faults; rejected solves never advance continuous state.
"""
from dataclasses import dataclass
import numpy as np


ARM_NAMES = {
    side: tuple(f'Joint_{side.title()}_{part}' for part in (
        'Shoulder_Inner', 'Shoulder_Outer', 'UpperArm', 'Elbow', 'Forearm',
        'Wrist_Upper', 'Wrist_Lower')) for side in ('left', 'right')
}


class SessionTrackingError(ValueError):
    pass


@dataclass(frozen=True)
class IKTicket:
    epoch: int
    sides: tuple
    seed: np.ndarray
    reference: np.ndarray
    dt: float
    stamp: float
    measured: np.ndarray
    side_epochs: tuple
    accepted_versions: tuple

    @property
    def solver_epochs(self):
        """Stable per-arm history keys; another arm's Grip does not reset IK."""
        return {side: (self.epoch, generation)
                for side, generation in zip(self.sides, self.side_epochs)}


class GripSessionIK:
    def __init__(self, deadband_deg=4.0, correction_rate_deg_sec=20.0,
                 correction_gain=2.0, hard_error_deg=22.0, timeout=0.3, maximum_lead_deg=10.0):
        values = (deadband_deg, correction_rate_deg_sec, correction_gain,
                  hard_error_deg, timeout, maximum_lead_deg)
        if not all(np.isfinite(x) for x in values) or not (
                0 <= deadband_deg < maximum_lead_deg < hard_error_deg and
                correction_rate_deg_sec > 0 and correction_gain > 0 and timeout > 0):
            raise ValueError('invalid V5 session correction parameters')
        self.deadband = np.deg2rad(deadband_deg)
        self.rate = np.deg2rad(correction_rate_deg_sec)
        self.gain = correction_gain
        self.hard = np.deg2rad(hard_error_deg)
        self.timeout = timeout
        self.maximum_lead = np.deg2rad(maximum_lead_deg)
        self.epoch = 0
        self.reset()

    def reset(self):
        # Only a whole-session change/reset invalidates both arms. Grip edges
        # have their own generations below so independent arms stay continuous.
        self.epoch += 1
        self.session = None
        self.down = dict(left=False, right=False)
        self.reference = {}
        self.accepted = {}
        self.stamp = {}
        self.blocked = set()
        self.error_deg = dict(left=0.0, right=0.0)
        self.corrected_deg = dict(left=0.0, right=0.0)
        self.stall_watch = {}
        self.side_epochs = dict(left=0, right=0)
        self.accepted_versions = dict(left=0, right=0)

    def release(self, sides):
        # This can also be called by an internal hold: do not invent a physical
        # release edge or clear blocked state. Only observe(False) can do that.
        for side in sides:
            if side in self.reference or side in self.accepted:
                self.side_epochs[side] += 1
            self.reference.pop(side, None)
            self.accepted.pop(side, None)
            self.stamp.pop(side, None)
            self.stall_watch.pop(side, None)

    def observe(self, session, state, measured, now):
        measured = np.asarray(measured, dtype=float)
        if not np.all(np.isfinite(measured)):
            raise ValueError('non-finite measured configuration')
        if session != self.session:
            self.reset()
            self.session = session
        for side in ('left', 'right'):
            pressed = bool(state[side])
            if not pressed:
                self.release((side,))
                self.blocked.discard(side)
            elif not self.down[side] and side not in self.blocked:
                self.side_epochs[side] += 1
                # Non-Grip motion is never integrated into an old session.
                # Capture current measured posture at the new physical edge;
                # keep this reference fixed until that arm releases Grip.
                self.reference[side] = measured.copy()
                self.accepted[side] = measured.copy()
                self.stamp[side] = now
                self.accepted_versions[side] = 0
            self.down[side] = pressed

    def expire(self, now):
        stale = [side for side, stamp in self.stamp.items()
                 if now - stamp > self.timeout]
        if stale:
            self.blocked.update(stale)
            self.release(stale)

    def prepare(self, targets, measured, joint_q, now):
        self.expire(now)
        sides = tuple(sorted(targets))
        if not sides or any(s not in self.reference or s in self.blocked for s in sides):
            raise SessionTrackingError('Grip session missing or expired')
        seed = np.asarray(measured, dtype=float).copy()
        reference = seed.copy()
        if not np.all(np.isfinite(seed)):
            raise SessionTrackingError('invalid measured configuration')
        dt = float(np.clip(min(now - self.stamp[s] for s in sides), 0.002, 0.03))
        for side in sides:
            idx = [joint_q[name] for name in ARM_NAMES[side]]
            previous = self.accepted[side][idx]
            error = seed[idx] - previous
            self.error_deg[side] = float(np.rad2deg(np.max(np.abs(error))))
            if np.max(np.abs(error)) >= self.hard:
                self.blocked.add(side)
                self.release((side,))
                raise SessionTrackingError(f'{side} tracking error {self.error_deg[side]:.1f} deg')
            # Bounded IK history must not conceal a genuinely stalled motor.
            # Normal fast tracking makes measured progress; a persistent large
            # lead with no measured motion still requires deliberate re-grip.
            if np.max(np.abs(error)) >= .8 * self.maximum_lead:
                anchor, started = self.stall_watch.get(side, (seed[idx].copy(), now))
                if np.max(np.abs(seed[idx] - anchor)) >= np.deg2rad(.2):
                    anchor, started = seed[idx].copy(), now
                self.stall_watch[side] = (anchor, started)
                if now - started >= 1.5:
                    self.blocked.add(side)
                    self.release((side,))
                    raise SessionTrackingError(f'{side} measured motion stalled')
            else:
                self.stall_watch.pop(side, None)
            # q_ik is exactly the last accepted result. Feedback constrains the
            # Mink QP's feasible envelope instead of changing its initial state.
            # The hard-error/stall checks above remain independent safeguards.
            seed[idx] = previous
            reference[idx] = self.reference[side][idx]
            self.corrected_deg[side] = 0.0
        return IKTicket(self.epoch, sides, seed, reference, dt, now,
                        np.asarray(measured, dtype=float).copy(),
                        tuple(self.side_epochs[s] for s in sides),
                        tuple(self.accepted_versions[s] for s in sides))

    def valid(self, ticket):
        return (ticket.epoch == self.epoch and
                all(s in self.reference and s not in self.blocked and
                    generation == self.side_epochs[s] and
                    version == self.accepted_versions[s] and
                    ticket.stamp >= self.stamp[s]
                    for s, generation, version in zip(
                        ticket.sides, ticket.side_epochs, ticket.accepted_versions)))

    def commit(self, ticket, solved):
        solved = np.asarray(solved, dtype=float)
        if not self.valid(ticket) or solved.shape != ticket.seed.shape or not np.all(np.isfinite(solved)):
            return False
        for side in ticket.sides:
            self.accepted[side] = solved.copy()
            self.stamp[side] = ticket.stamp
            self.accepted_versions[side] += 1
        return True

    def status(self):
        return dict(mode='mink_quadprog_session_servo',
                    active=sorted(self.reference), require_regrip=sorted(self.blocked),
                    feedback_strategy='qp_envelope',
                    maximum_ik_lead_deg=float(np.rad2deg(self.maximum_lead)),
                    tracking_error_deg=dict(self.error_deg),
                    correction_step_deg=dict(self.corrected_deg))
