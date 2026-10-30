"""Single torso convention shared by clutch FK and independent arm IK."""
import numpy as np


def arm_reference_groups(groups, neutral_pitch=None, locked_yaw=None):
    reference = {k: np.asarray(v, dtype=float).copy() for k, v in groups.items()}
    for index, value in ((2, neutral_pitch), (3, locked_yaw)):
        if value is not None:
            if not np.isfinite(value):
                raise ValueError('invalid arm reference torso angle')
            reference['leg_waist'][index] = float(value)
    return reference
