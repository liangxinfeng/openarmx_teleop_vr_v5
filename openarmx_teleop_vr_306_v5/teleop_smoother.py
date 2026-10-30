"""Offline candidate for independently bounded dual-arm position targets.

Position/velocity/acceleration bounds limit the reference only; they are not
verified physical motor limits. Emergency holds bypass the smoother. A
feedback-bound rebase can create a discontinuity to preserve the guard.
"""
import numpy as np

from .trajectory_limiter import TrajectoryLimiter, bounded_target_by_feedback


class ArmTargetSmoother:
    def __init__(self, lower, upper, max_velocity, max_acceleration):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if lower.shape != (14,) or upper.shape != (14,):
            raise ValueError('expected exactly two seven-joint arms')
        self._limiter = TrajectoryLimiter(lower, upper, max_velocity, max_acceleration)
        self._active = set()

    def reset(self, measured):
        self._limiter.reset(measured)
        self._active.clear()

    def rebase_side(self, side, measured):
        if side not in ('left', 'right'):
            raise ValueError('unknown arm')
        measured = np.asarray(measured, dtype=float)
        if measured.shape != (7,):
            raise ValueError('expected seven measured joints')
        indices = list(range(0, 7) if side == 'left' else range(7, 14))
        self._limiter.reset_indices(indices, measured)
        self._active.discard(side)

    def step(self, measured, desired, lead, dt, active_sides, target_velocity=None):
        measured = np.asarray(measured, dtype=float)
        desired = np.asarray(desired, dtype=float)
        lead = np.asarray(lead, dtype=float)
        if measured.shape != (14,) or desired.shape != (14,) or lead.shape != (14,):
            raise ValueError('invalid arm target shape')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('invalid controller period')
        active = set(active_sides)
        if not active <= {'left', 'right'}:
            raise ValueError('unknown arm')
        if self._limiter.position is None:
            self.reset(measured)
        for side, idx in (('left', range(0, 7)), ('right', range(7, 14))):
            # Releases must go directly to the actual measured pose. A new
            # clutch starts at measured feedback and must not inherit history.
            if side not in active or side not in self._active:
                indices = list(idx)
                self._limiter.reset_indices(indices, measured[indices])
        safe_target = bounded_target_by_feedback(measured, desired, lead)
        if target_velocity is not None:
            velocity = np.asarray(target_velocity, dtype=float)
            if velocity.shape != (14,) or not np.all(np.isfinite(velocity)):
                raise ValueError('invalid target velocity')
            velocity = np.clip(velocity, -self._limiter.max_velocity, self._limiter.max_velocity)
            # Compensate at most one output cycle of smoothing lag. No new
            # speed, acceleration or feedback-lead allowance is introduced.
            # Only continue an already advancing reference; never predict a
            # first clutch, reversal, stopped or feedback-clamped target.
            advancing = ((safe_target-self._limiter.position)*velocity > 0.)
            advancing &= np.abs(safe_target-desired) < 1.e-9
            for side, idx in (('left', range(7)), ('right', range(7,14))):
                if side not in active or side not in self._active:
                    advancing[list(idx)] = False
            preview = np.where(advancing, velocity*min(dt,.02), 0.)
            safe_target = bounded_target_by_feedback(measured, safe_target+preview, lead)
        out = self._limiter.step(safe_target, min(dt, .02))
        bounded = bounded_target_by_feedback(measured, out, lead)
        changed = np.flatnonzero(np.abs(bounded-out) > 1e-9)
        if changed.size:
            # Feedback-relative protection takes precedence over smoothness.
            self._limiter.reset_indices(changed, bounded[changed])
        for side, idx in (('left', range(0, 7)), ('right', range(7, 14))):
            if side not in active:
                indices = list(idx)
                bounded[indices] = measured[indices]
                self._limiter.reset_indices(indices, measured[indices])
        self._active = active
        return bounded
