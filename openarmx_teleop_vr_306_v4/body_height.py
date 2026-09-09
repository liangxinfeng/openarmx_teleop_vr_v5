"""Coordinated S2 waist/leg profile for vertical body-height control."""

import math


MAX_BODY_LOWERING_M = 0.56
BODY_HEIGHT_JOINT_NAMES = (
    'Joint_Ankle',
    'Joint_Knee',
    'Joint_Waist_Pitch',
)
BODY_HEIGHT_MAXIMUM_DEG = (76.0, 155.0, 81.0)


def body_height_joint_targets(lowering_m):
    """Return ankle, knee and waist-pitch targets for a lowering distance."""
    lowering = min(MAX_BODY_LOWERING_M, max(0.0, float(lowering_m)))
    scale = lowering / MAX_BODY_LOWERING_M
    return tuple(maximum * scale for maximum in BODY_HEIGHT_MAXIMUM_DEG)


def estimate_body_lowering(leg_waist_deg):
    """Project a four-joint leg/waist pose onto the coordinated height curve."""
    values = tuple(float(value) for value in leg_waist_deg)
    if len(values) != 4:
        raise ValueError('leg_waist_deg must contain four joints')
    denominator = sum(value * value for value in BODY_HEIGHT_MAXIMUM_DEG)
    scale = sum(
        values[index] * maximum
        for index, maximum in enumerate(BODY_HEIGHT_MAXIMUM_DEG)
    ) / denominator
    return min(1.0, max(0.0, scale)) * MAX_BODY_LOWERING_M


def coordinated_body_height_targets(
    lowering_m,
    measured_deg,
    previous_progress,
    command_lead_deg,
):
    """Advance ankle, knee and waist pitch with one feedback-bounded progress.

    The vendor position loops do not track the three joints at exactly the same
    rate. Sending a moving target independently lets a fast leg joint get ahead
    of waist pitch. A single normalized progress value keeps the body-height
    geometry intact and pauses all three targets when one joint is catching up.

    ``previous_progress`` also makes the command monotonic in the requested
    direction, so small encoder noise cannot make the body rock back and forth.
    """
    measured = tuple(float(value) for value in measured_deg)
    if len(measured) != len(BODY_HEIGHT_MAXIMUM_DEG):
        raise ValueError('measured_deg must contain ankle, knee and waist pitch')
    if not all(math.isfinite(value) for value in measured):
        raise ValueError('measured_deg must contain only finite values')

    desired = min(
        1.0,
        max(0.0, float(lowering_m) / MAX_BODY_LOWERING_M),
    )
    previous = min(1.0, max(0.0, float(previous_progress)))
    lead = max(0.1, float(command_lead_deg))
    if not all(math.isfinite(value) for value in (desired, previous, lead)):
        raise ValueError('body-height progress inputs must be finite')

    epsilon = 1.0e-9
    if desired > previous + epsilon:
        feedback_limit = min(
            value / maximum + lead / maximum
            for value, maximum in zip(measured, BODY_HEIGHT_MAXIMUM_DEG)
        )
        progress = max(previous, min(desired, feedback_limit))
    elif desired < previous - epsilon:
        feedback_limit = max(
            value / maximum - lead / maximum
            for value, maximum in zip(measured, BODY_HEIGHT_MAXIMUM_DEG)
        )
        progress = min(previous, max(desired, feedback_limit))
    else:
        progress = previous

    progress = min(1.0, max(0.0, progress))
    return (
        tuple(maximum * progress for maximum in BODY_HEIGHT_MAXIMUM_DEG),
        progress,
    )
