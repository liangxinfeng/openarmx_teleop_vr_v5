import numpy as np
import pytest
pytest.importorskip('placo')
from ._subject import load_subject,source_directory
m=load_subject('quest_placo_kinematics')

def test_unaccepted_solve_cannot_advance_servo_or_preview_history():
    ik=m.QuestPlacoKinematics(source_directory().parent/'urdf/robot_v3_1_simplified.urdf',collision_check=False)
    groups=dict(leg_waist=np.zeros(4),neck=np.zeros(3),left_arm=np.array([20,0,0,-110,0,0,0.]),right_arm=np.array([20,0,0,-110,0,0,0.]))
    q=ik.q_from_feedback(groups);pose=ik.eef_poses(q)['left'];p=np.asarray(pose['position'])+np.array([.02,0,.01])
    targets={'left':(p,pose['orientation'])}
    first=ik.solve(targets,q,posture_reference=q,feedback_q=q,transactional=True,session_epoch=1)
    assert first.success
    assert not ik._previous_commands and not ik._target_history
    again=ik.solve(targets,q,posture_reference=q,feedback_q=q,transactional=True,session_epoch=1)
    np.testing.assert_allclose(again.q,first.q,atol=1e-8)
    ik.commit_retarget_state()
    assert ik._previous_commands and ik._target_history
    # Re-grip at the same measured angles starts a new epoch, not old motion.
    fresh={'left':(pose['position'],pose['orientation'])}
    result=ik.solve(fresh,q,posture_reference=q,feedback_q=q,transactional=True,session_epoch=2)
    assert result.success
    idx=[ik.joint_q[n] for n in m.LEFT_ARM_JOINTS]
    assert np.max(np.abs(result.q[idx]-q[idx]))<np.deg2rad(.3)
