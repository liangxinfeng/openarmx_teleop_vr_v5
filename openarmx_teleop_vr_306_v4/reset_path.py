"""Continuous X+A reset path from measured joints to the system pose.

The path deliberately contains no intermediate shoulder detour.  It is one
quintic sweep whose complete sampled path is validated by the controller
before any command is committed.
"""

import numpy as np


def smoothstep5(progress):
    """Quintic zero-velocity/zero-acceleration interpolation in ``[0, 1]``."""
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress ** 3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def continuous_direct_reset(start, goal, progress):
    """Return one sample of the direct, zero-end-velocity reset sweep."""
    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if start.shape != goal.shape or start.size != 21:
        raise ValueError('reset start and goal must both contain 21 joints')
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError('reset path contains NaN or Inf')
    blend = smoothstep5(progress)
    return start + (goal - start) * blend


def conservative_reset_duration(
    start,
    goal,
    arm_velocity_deg_sec,
    neck_velocity_deg_sec,
    minimum_duration_sec=1.0,
):
    """Choose a duration that keeps the moving reference below speed limits.

    The quintic blend peaks at 1.875 times the average rate.  Runtime
    trajectory limiting remains the final acceleration/velocity guard.
    """
    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if start.shape != goal.shape or start.size != 21:
        raise ValueError('reset start and goal must both contain 21 joints')
    arm_velocity = float(arm_velocity_deg_sec)
    neck_velocity = float(neck_velocity_deg_sec)
    minimum = float(minimum_duration_sec)
    if not all(np.isfinite(value) and value > 0.0 for value in (
            arm_velocity, neck_velocity, minimum)):
        raise ValueError('reset duration limits must be finite and positive')

    arm_delta = np.abs(goal[4:18] - start[4:18])
    arm_path_rate = 1.875 * arm_delta
    neck_path_rate = 1.875 * np.abs(goal[18:21] - start[18:21])
    return max(
        minimum,
        float(np.max(arm_path_rate)) / arm_velocity,
        float(np.max(neck_path_rate)) / neck_velocity,
    )
