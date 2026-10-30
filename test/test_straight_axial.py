import numpy as np
import pytest
from .test_mink_armshape import make_ik,seed,target,module

@pytest.mark.parametrize('side',['left','right'])
def test_axis_reference_is_transactional_and_reclutch_uses_current_pose(side):
    ik=make_ik();q=seed(ik,-5.)
    r=ik.solve(target(ik,q,side),q,feedback_q=q,session_epoch=1,transactional=True)
    assert r.success,r.reason
    assert not ik._straight_history
    ik.commit_retarget_state()
    reference=ik._straight_history[side]['reference'].copy()
    q2=q.copy();q2[ik.joint_q[module.ARM_JOINTS[side][2]]]+=np.deg2rad(4.)
    r=ik.solve(target(ik,q2,side),q2,feedback_q=q2,session_epoch=2,transactional=True)
    assert r.success,r.reason
    np.testing.assert_equal(ik._straight_history[side]['reference'],reference)
    ik.commit_retarget_state()
    assert ik._straight_history[side]['reference'][2]==pytest.approx(reference[2]+np.deg2rad(4.))

@pytest.mark.parametrize('side',['left','right'])
def test_straight_hold_does_not_self_rotate_and_other_arm_is_unchanged(side):
    ik=make_ik();q=seed(ik,-5.);start=q.copy();pose=target(ik,q,side)
    for i in range(200):
        r=ik.solve(pose,q,feedback_q=q,session_epoch=1)
        assert r.success,r.reason
        q=r.q
    np.testing.assert_allclose(q,start,atol=1.e-9)
