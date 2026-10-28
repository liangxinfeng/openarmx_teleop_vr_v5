"""Fixed-wrist side sweeps across the shoulder side-joint limit."""
import numpy as np
import pytest
pytest.importorskip('mink')
from .test_mink_armshape import make_ik,seed
from ._subject import load_subject
m=load_subject('mink_kinematics')


@pytest.mark.parametrize('side',['left','right'])
def test_lateral_sweep_decelerates_before_limit_without_envelope_recovery(side):
    ik=make_ik();home=seed(ik,-110);q=home.copy()
    pose=ik.eef_poses(home)[side];base=np.asarray(pose['position'])
    names=m.ARM_JOINTS[side];idx=np.array([ik.joint_q[n] for n in names])
    previous_velocity=np.zeros(7);near_acc=[];peak_error=0
    for frame in range(500):
        goal=base+np.array([0,.1*np.sin(2*np.pi*frame/400),0])
        r=ik.solve({side:(goal,pose['orientation'])},q,feedback_q=q,
                   posture_reference=home,reset_reference=home,session_epoch=1)
        assert r.success,r.reason
        v=np.rad2deg(r.q[idx]-q[idx])/.01
        a=(v-previous_velocity)/.01
        assert not ik.last_retarget_status[side]['acceleration_recovery']
        assert np.max(np.abs(v[:3]))<=90+1e-6
        if frame>5:
            assert np.max(np.abs(a[:3]))<=1200+1e-5
            assert np.max(np.abs(a[3:]))<=2400+1e-5
        distance=min(r.q[idx[1]]-ik.lower[idx[1]],ik.upper[idx[1]]-r.q[idx[1]])
        approaching = v[1]<0 if side=='left' else v[1]>0
        if frame>5 and distance<np.deg2rad(2) and approaching:
            near_acc.append(abs(a[1]))
        peak_error=max(peak_error,r.position_error_m[side])
        q=r.q;previous_velocity=v
    assert near_acc and max(near_acc)<300
    assert peak_error<.007
