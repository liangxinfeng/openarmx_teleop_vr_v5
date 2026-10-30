"""Grip re-anchoring preserves one waist baseline and one assist budget."""
import math
from .body_height import BODY_HEIGHT_MAXIMUM_DEG


def held_waist_assist(measured_pitch, neutral, target):
    """Separate attained assist from a height stop; never use future lead."""
    if neutral is None or target is None:
        return 0.0
    return min(12.0, max(0.0, float(target)-float(neutral)),
               max(0.0, float(measured_pitch)-float(neutral)))


def waist_reset_reference(configured, measured_leg):
    """Reset assist and yaw without undoing the current leg-height geometry."""
    pitch, yaw = (float(x) for x in configured)
    ankle, knee = (float(x) for x in measured_leg[:2])
    if not all(math.isfinite(x) for x in (pitch,yaw,ankle,knee)):
        raise ValueError('waist reset requires finite reference and leg feedback')
    a,k,w=BODY_HEIGHT_MAXIMUM_DEG
    height_pitch=w*(ankle*a+knee*k)/(a*a+k*k)
    return [height_pitch+pitch,yaw]


def waist_reanchor_x(target_x, measured_pitch, neutral, start, full, maximum):
    values=tuple(float(x) for x in (target_x,measured_pitch,neutral,start,full,maximum))
    if not all(math.isfinite(x) for x in values) or full<=start or maximum<0:
        raise ValueError('invalid waist re-anchor parameters')
    target_x,measured_pitch,neutral,start,full,maximum=values
    lean=min(maximum,max(0.,measured_pitch-neutral))
    if lean<=1.e-6:
        extension=0.
    else:
        # Invert the same smoothstep used by forward_reach_to_waist_pitch.
        ratio=min(1.,max(0.,lean/maximum))
        u=.5-math.sin(math.asin(1.-2.*ratio)/3.)
        extension=start+(full-start)*u
    return target_x-extension
