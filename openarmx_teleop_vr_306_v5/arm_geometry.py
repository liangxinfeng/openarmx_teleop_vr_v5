"""URDF-derived shoulder/elbow landmarks for display and retargeting."""
import numpy as np
import pinocchio as pin


def arm_geometry(model, q):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    result = {}
    straight = np.asarray(q).copy()
    for side in ('Left','Right'):
        for part in ('Shoulder_Inner','Shoulder_Outer','UpperArm','Elbow','Forearm','Wrist_Upper','Wrist_Lower'):
            j = model.joints[model.getJointId(f'Joint_{side}_{part}')]
            straight[j.idx_q] = np.deg2rad(-5.) if part == 'Elbow' else 0.
    straight_data = model.createData();pin.forwardKinematics(model,straight_data,straight)
    for side in ('left','right'):
        ids = [model.getJointId(f'Joint_{side.title()}_{p}') for p in ('Shoulder_Outer','Elbow','Wrist_Lower')]
        s,e,w = [data.oMi[i].translation.copy() for i in ids]
        ss,_,ww = [straight_data.oMi[i].translation.copy() for i in ids]
        result[side] = dict(shoulder=s.tolist(),elbow=e.tolist(),wrist=w.tolist(),
                            reach=float(np.linalg.norm(ww-ss)-.005),
                            upper_length=float(np.linalg.norm(e-s)),lower_length=float(np.linalg.norm(w-e)))
    return result


def elbow_target(shoulder, wrist, reference_shoulder, reference_elbow, reference_wrist, upper, lower):
    """Choose elbow swivel from clutch; derive bend from current wrist reach."""
    from .shoulder_retarget import rotation_between
    s=np.asarray(shoulder);w=np.asarray(wrist);axis=w-s;d=np.linalg.norm(axis)
    if d<1e-6:return np.asarray(reference_elbow).copy()
    axis/=d
    ref_axis=np.asarray(reference_wrist)-reference_shoulder
    ref_axis/=max(np.linalg.norm(ref_axis),1e-9)
    bend=np.asarray(reference_elbow)-reference_shoulder
    bend-=np.dot(bend,ref_axis)*ref_axis
    if np.linalg.norm(bend)<1e-4:
        bend=np.cross(ref_axis,np.eye(3)[np.argmin(np.abs(ref_axis))])
    bend=rotation_between(ref_axis,axis) @ (bend/np.linalg.norm(bend))
    distance=np.clip(d,abs(upper-lower)+.001,upper+lower-.001)
    along=(upper*upper-lower*lower+distance*distance)/(2*distance)
    height=np.sqrt(max(upper*upper-along*along,0.))
    return s+along*axis+height*bend
