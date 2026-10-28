"""Smooth shoulder/elbow return assistance; no motor or ROS access."""
import math


def return_shape_step(state, distance, dt, reference_key, *, start=.45, full=.08,
                      transition=1.0):
    """Pure candidate state, committed only with the associated IK command.

    Require inward hand travel before assistance. This prevents a new Grip or
    an orientation-only gesture from initiating home recovery. Hysteresis
    rejects millimetre input noise; outward travel smoothly releases the bias.
    """
    if not all(math.isfinite(x) for x in (distance, dt, start, full, transition)):
        raise ValueError('return shape parameters must be finite')
    if not (distance >= 0 and dt > 0 and start > full >= 0 and transition > 0):
        raise ValueError('invalid return shape distances/timing')
    if state is None or state['reference_key'] != reference_key:
        return dict(reference_key=reference_key, peak=distance, valley=distance,
                    active=False, blend=0.0)
    result = dict(state)
    if result['active']:
        result['valley'] = min(result['valley'], distance)
        if distance-result['valley'] >= .025:
            result.update(active=False, peak=distance)
    else:
        result['peak'] = max(result['peak'], distance)
        if result['peak']-distance >= .015:
            result.update(active=True, valley=distance)
    proximity = min(1., max(0., (start-distance)/(start-full)))
    desired = proximity*proximity*(3.-2.*proximity) if result['active'] else 0.
    # Bound per-second weight change as well as smoothing the spatial gate.
    step = dt/transition
    result['blend'] += min(step, max(-step, desired-result['blend']))
    return result
