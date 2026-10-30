import numpy as np
import pytest
from ._subject import load_subject
from .test_mink_session_boundaries import make_controller

state=load_subject('waist_follow_state')
core=load_subject('teleop_core')

@pytest.mark.parametrize('neutral', [0.,-18.])
@pytest.mark.parametrize('lean',[0.,.01,2.,6.,10.,12.])
def test_reclutch_holds_angle_and_never_allocates_a_second_lean_budget(neutral,lean):
    measured=neutral+lean
    for i in range(30):
        x=.3+i*.02
        anchor=state.waist_reanchor_x(x,measured,neutral,.06,.24,12.)
        held=core.forward_reach_to_waist_pitch(x-anchor,neutral,.06,.24,12.)
        assert held==pytest.approx(measured,abs=1.e-6)
        measured=core.forward_reach_to_waist_pitch(x+.01-anchor,neutral,.06,.24,12.)
        assert neutral<=measured<=neutral+12.

def test_return_after_regrip_releases_assistance_without_reverse_lean():
    anchor=state.waist_reanchor_x(.65,12.,0.,.06,.24,12.)
    angles=[core.forward_reach_to_waist_pitch(x-anchor,0.,.06,.24,12.) for x in np.linspace(.65,.1,100)]
    assert np.all(np.diff(angles)<=1.e-12)
    assert angles[-1]==0.

@pytest.mark.parametrize('height_active',[False,True])
def test_real_release_handler_keeps_original_baseline_and_holds_measured_waist(height_active):
    c=make_controller();c._waist_follow_enabled=True;c._waist_follow_profile='forward_pitch_only'
    c._waist_follow_neutral_pitch_deg=0.;c._body_height_active=height_active;c._body_height_joint_targets=np.zeros(3)
    c._feedback.as_dict()['leg_waist'][2]=12.
    c._hold_sides_locked(['left','right'],1.)
    assert c._waist_follow_neutral_pitch_deg==0.
    assert c._target_groups['leg_waist'][2]==12.

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_invalid_configuration_cannot_generate_waist_target(bad):
    with pytest.raises(ValueError):state.waist_reanchor_x(.5,10.,0.,.06,.24,bad)

@pytest.mark.parametrize('budget',[0.,3.,6.,10.,12.])
def test_reclutch_uses_the_same_urdf_limited_budget_as_forward_mapping(budget):
    neutral=30.;measured=neutral+budget
    anchor=state.waist_reanchor_x(.5,measured,neutral,.06,.24,budget)
    held=core.forward_reach_to_waist_pitch(.5-anchor,neutral,.06,.24,budget)
    assert held==pytest.approx(measured)

@pytest.mark.parametrize('height',[-.53,-.3,0.,.3,.5])
def test_reset_removes_assist_but_preserves_current_height(height):
    legs=np.array([76.,155.,81.,0.])*height/.56
    legs[2]+=12.;legs[3]=10.
    pitch,yaw=state.waist_reset_reference([0.,0.],legs)
    assert pitch==pytest.approx(81.*height/.56)
    assert yaw==0.

@pytest.mark.parametrize('lean',[4.,12.])
@pytest.mark.parametrize('boundary',['timeout','disable'])
def test_height_stop_does_not_turn_attained_assist_into_new_neutral(lean,boundary):
    from .test_height_watchdog_regression import Harness
    from unittest.mock import patch
    from types import SimpleNamespace as NS
    import time
    c=Harness();c.measured=np.array([0.,0.,lean,0.])
    c._waist_follow_neutral_pitch_deg=0.;c._waist_follow_pitch_target_deg=lean
    c._body_height_joint_targets=np.zeros(3)
    if boundary=='timeout':
        held=c._body_height_output_target_locked(10.,c._feedback)
        np.testing.assert_allclose(held,c.measured[:3])
        with patch.object(time,'monotonic',return_value=10.02):
            c._on_body_height_command(NS(data='{"enabled":true,"direction":0}'))
    else:
        c._on_body_height_command(NS(data='{"enabled":false}'))
    assert c._waist_follow_neutral_pitch_deg==pytest.approx(0.)
    assert c._target_groups['leg_waist'][2]==pytest.approx(lean)
    anchor=state.waist_reanchor_x(.5,lean,c._waist_follow_neutral_pitch_deg,.06,.24,12.)
    requested=core.forward_reach_to_waist_pitch(1.-anchor,0.,.06,.24,12.)
    assert requested==12.
