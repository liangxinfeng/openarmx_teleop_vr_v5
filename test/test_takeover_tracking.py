"""No hardware publishers: clutch continuity and bounded moving references."""
import numpy as np
import pytest
from ._subject import load_subject
from .test_quest_incremental_mapper import _mapper, _sample
from .test_shoulder_retarget import mapper as calibrated_mapper, sample as human_sample


def test_grip_tracking_noise_does_not_project_or_accumulate_motion():
    a=_mapper();origin=dict(position=[2.01,.3,1.],orientation=[0.,0.,0.,1.])
    a.latch(_sample([0.,0.,0.]),origin)
    for sign in (-1.,1.)*100:
        pose,_=a.map(_sample([.001*sign,0,0],[0.,0.,0.,sign]))
        np.testing.assert_array_equal(pose['position'],origin['position'])
    # Input gate only delays intent recognition; it never discards that delta.
    a.latch(_sample([0.,0.,0.]),dict(position=[.4,.3,1.],orientation=[0.,0.,0.,1.]))
    p,_=a.map(_sample([.003,0,0]))
    assert p['position'][1]==pytest.approx(.3-.003*.7)
    p,_=a.map(_sample([0.,0.,0.]))
    np.testing.assert_allclose(p['position'],[.4,.3,1.])


def test_calibrated_origin_outside_sphere_remains_relative_and_still():
    a=calibrated_mapper();head=np.array([0.,1.6,0.]);s=human_sample([.35,0,0])
    origin=dict(position=[.56,.22,1.],orientation=[0.,0.,0.,1.])
    a.latch(s,origin,head)
    assert not a.spherical
    pose,_=a.map(s,head)
    np.testing.assert_array_equal(pose['position'],origin['position'])


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('lean',[0.,6.,12.])
def test_neutral_torso_clutch_is_fixed_point_with_production_home_reference(side,lean):
    from .test_mink_armshape import make_ik,seed,module
    ik=make_ik();physical=seed(ik,-70.)
    physical[ik.joint_q['Joint_Waist_Pitch']]=np.deg2rad(lean)
    groups=dict(leg_waist=[0.,0.,lean,0.],neck=[0.,0.,0.],
                left_arm=[20.,0.,0.,-70.,0.,0.,0.],right_arm=[20.,0.,0.,-70.,0.,0.,0.])
    reference=load_subject('arm_reference_frame').arm_reference_groups(groups,0.,0.)
    q=ik.q_from_feedback(reference);start=q.copy();home=seed(ik,-110.)
    poses=ik.eef_poses(q);pose=poses[side];target={side:(pose['position'],pose['orientation'])}
    for _ in range(40):
        result=ik.solve(target,q,feedback_q=q,posture_reference=start,
                        reset_reference=home,session_epoch=1,integration_dt=.01)
        assert result.success,result.reason
        q=result.q
    np.testing.assert_allclose(q,start,atol=1.e-8)
    assert groups['leg_waist'][2]==pytest.approx(lean)


def trace(preview):
    smoother=load_subject('teleop_smoother').ArmTargetSmoother(np.full(14,-180.),np.full(14,180.),np.full(14,90.),np.full(14,1500.))
    measured=np.zeros(14);smoother.reset(measured);samples=[]
    for i in range(250):
        t=i*.01;goal=np.zeros(14);v=np.zeros(14)
        goal[7]=min(max(t-.2,0.)*60.,45.);v[7]=60. if .2<t<.95 else 0.
        output=smoother.step(measured,goal,np.full(14,12.),.01,{'right'},target_velocity=v if preview else None)
        samples.append((goal.copy(),output.copy()))
        measured=output.copy() # ideal following isolates software overhead
    return np.array(samples)


def test_preview_reduces_software_ramp_lag_without_increasing_limits():
    old=trace(False);new=trace(True)
    lag_old=np.mean(abs(old[30:85,0,7]-old[30:85,1,7]))
    lag_new=np.mean(abs(new[30:85,0,7]-new[30:85,1,7]))
    assert lag_new < lag_old*.85,(lag_old,lag_new)
    velocity=np.diff(new[:,1,:],axis=0)/.01
    assert np.max(abs(velocity))<=90.+1.e-7
    assert np.max(abs(np.diff(velocity,axis=0)/.01))<=1500.+1.e-6
    np.testing.assert_array_equal(new[:,1,:7],0.)
    assert np.max(abs(new[-1,1,:]-new[-1,0,:]))<1.e-6


def test_preview_cannot_cross_feedback_envelope_or_inherit_regrip_velocity():
    m=load_subject('teleop_smoother').ArmTargetSmoother(np.full(14,-180.),np.full(14,180.),np.full(14,90.),np.full(14,1500.))
    measured=np.zeros(14);m.reset(measured)
    for _ in range(250):
        out=m.step(measured,np.full(14,80.),np.full(14,12.),.01,{'right'},np.full(14,180.))
        assert np.max(abs(out))<=12.+1.e-9
        np.testing.assert_array_equal(out[:7],0.)
    np.testing.assert_array_equal(m.step(measured,measured,np.full(14,12.),.01,set(),np.full(14,180.)),measured)
    np.testing.assert_array_equal(m.step(measured,measured,np.full(14,12.),.01,{'right'},np.full(14,180.)),measured)


@pytest.mark.parametrize('side,roll',[('right',40.),('left',-40.)])
def test_straight_axial_preference_releases_for_inward_return(side,roll):
    from .test_mink_armshape import make_ik,seed,module
    from scipy.spatial.transform import Rotation,Slerp
    ik=make_ik();home=seed(ik,-110.);start=home.copy()
    idx=np.array([ik.joint_q[n] for n in module.ARM_JOINTS[side]])
    start[idx[3]]=np.deg2rad(-5.);start[idx[2]]=np.deg2rad(roll);start[idx[4]]=np.deg2rad(-roll)
    p0=ik.eef_poses(start)[side];p1=ik.eef_poses(home)[side]
    slerp=Slerp([0.,1.],Rotation.from_quat([p0['orientation'],p1['orientation']]))
    q=start.copy()
    for i in range(650):
        u=np.clip((i-30)/250.,0.,1.);u=u*u*(3.-2.*u)
        target={side:((1-u)*np.array(p0['position'])+u*np.array(p1['position']),slerp(u).as_quat())}
        result=ik.solve(target,q,feedback_q=q,posture_reference=start,reset_reference=home,
                        session_epoch=1,integration_dt=.01)
        assert result.success,result.reason
        q=result.q
    assert result.position_error_m[side]<.004
    assert result.orientation_error_rad[side]<np.deg2rad(.19)
    assert abs(np.rad2deg(q[idx[3]]))>80.
