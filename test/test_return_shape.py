import numpy as np
import pytest
from ._subject import load_subject

step = load_subject('return_shape').return_shape_step


def test_static_grip_and_orientation_only_do_not_start_return():
    state = None
    for _ in range(300):
        state = step(state, .12, .01, 'poseA')
    assert state['blend'] == 0 and not state['active']


def test_early_return_before_thirty_cm_and_bounded_weight_changes():
    state = None
    for d in np.linspace(.55, .32, 150):
        previous = 0 if state is None else state['blend']
        state = step(state, float(d), .01, 'poseA')
        assert abs(state['blend']-previous) <= .04+1e-12
    assert state['active'] and state['blend'] > .25
    for d in np.linspace(.32, .06, 150):
        state = step(state, float(d), .01, 'poseA')
    assert state['blend'] == pytest.approx(1)
    for d in np.linspace(.06, .55, 200):
        state = step(state, float(d), .01, 'poseA')
    assert state['blend'] == 0 and not state['active']


def test_new_reset_pose_and_new_grip_clear_old_return_state():
    state = None
    for d in np.linspace(.55, .06, 200):
        state = step(state, float(d), .01, 'poseA')
    assert state['blend'] > .9
    changed = step(state, .06, .01, 'poseB')
    assert changed['blend'] == 0 and not changed['active']
    assert step(None, .06, .01, 'poseA')['blend'] == 0


def test_millimetre_jitter_never_initiates_return_and_state_is_pure():
    state = step(None, .2, .01, 'poseA')
    original = dict(state)
    for d in [.199, .201]*100:
        state = step(state, d, .01, 'poseA')
    assert state['blend'] == 0
    step(original, .1, .01, 'poseA')
    assert original['peak'] == .2 and original['blend'] == 0


@pytest.mark.parametrize('kwargs', [dict(start=.05), dict(dt=0), dict(distance=float('nan'))])
def test_invalid_config_fails(kwargs):
    args=dict(state=None,distance=.1,dt=.01,reference_key='poseA');args.update(kwargs)
    with pytest.raises(ValueError): step(**args)
