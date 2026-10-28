"""Controller-only shoulder/reach retargeting; no claim of measured human elbows.

Grip captures a no-jump correspondence. Radial motion maps that correspondence
to the calibrated full reach; angular motion parallel-transports the robot arm.
Uncalibrated/desktop use remains isotropic relative control.
"""
import numpy as np
from .quest_incremental import QuestIncrementalArmMapper
from .teleop_core import VR_TO_ROBOT_ROT, quaternion_to_matrix, matrix_to_quaternion


def rotation_between(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / max(np.linalg.norm(a), 1e-9); b = b / max(np.linalg.norm(b), 1e-9)
    v = np.cross(a, b); c = float(np.clip(a @ b, -1., 1.))
    if c < -.9999:
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))]); axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + k + k @ k / max(1 + c, 1e-9)


def mapped_radius(h, h0, r0, human_length, robot_length):
    # The calibrated maximum remains a geometric endpoint, not a gain applied
    # differently to XYZ. Near-full clutching is gated before this function.
    if h >= h0:
        return float(np.clip(r0 + (robot_length-r0)*(h-h0)/max(human_length-h0,.03), .04, robot_length))
    return float(np.clip(r0*h/max(h0,.04), .04, robot_length))


class ShoulderReachMapper(QuestIncrementalArmMapper):
    def __init__(self, *args, **kwargs):
        self.geometry = None
        self.human_length = None
        self.retarget_status = {'mode': 'isotropic_relative', 'calibrated': False}
        super().__init__(*args, **kwargs)

    def set_geometry(self, geometry):
        shoulder = np.asarray(geometry['shoulder'], float)
        length = float(geometry['reach'])
        if shoulder.shape != (3,) or not np.all(np.isfinite(shoulder)) or not .25 < length < 1.:
            raise ValueError('invalid robot shoulder geometry')
        self.geometry = {'shoulder': shoulder, 'reach': length}

    def human_vector(self, sample, body_position):
        # HMD-to-shoulder is an estimate. Calibration measures reach relative to
        # this estimate, not a motion-capture measurement of the shoulder.
        shoulder_offset = np.array([-.03, .19 if self.side == 'left' else -.19, -.22])
        return VR_TO_ROBOT_ROT @ (np.asarray(sample.position)-np.asarray(body_position)) - shoulder_offset

    def release(self):
        super().release()
        self.spherical = False

    def latch(self, sample, current_pose, body_position=None):
        super().latch(sample, current_pose, body_position)
        self.retarget_status['reason']=''
        self.spherical = bool(self.human_length is not None and self.geometry is not None and body_position is not None)
        if self.spherical:
            self.h0 = self.human_vector(sample, body_position)
            self.shoulder0 = self.geometry['shoulder'].copy()
            self.r0 = self.robot_origin_position-self.shoulder0
            if np.linalg.norm(self.h0) > .93*self.human_length and np.linalg.norm(self.r0) < .9*self.geometry['reach']:
                # Never straighten a bent physical arm merely because Grip was
                # pressed with an already extended human arm.
                self.spherical = False
                self.retarget_status['reason'] = 'bend arm before Grip for calibrated reach mapping'
        self.retarget_status.update(mode='shoulder_reach' if self.spherical else 'isotropic_relative', calibrated=self.human_length is not None)

    def map(self, sample, body_position=None):
        if not self.spherical:
            target,reason=super().map(sample, body_position)
            return target,reason or self.retarget_status.get('reason','')
        if body_position is None:
            return None, 'fresh HMD position required for shoulder mapping'
        h = self.human_vector(sample, body_position)
        if np.linalg.norm(h) < .08:
            return None, 'controller too close to estimated shoulder'
        delta = quaternion_to_matrix(sample.orientation) @ quaternion_to_matrix(self.vr_origin_orientation).T
        rotation = VR_TO_ROBOT_ROT @ delta @ VR_TO_ROBOT_ROT.T
        orientation = matrix_to_quaternion(rotation @ quaternion_to_matrix(self.robot_origin_orientation))
        radius = mapped_radius(np.linalg.norm(h),np.linalg.norm(self.h0),np.linalg.norm(self.r0),self.human_length,self.geometry['reach'])
        direction = rotation_between(self.h0,h) @ (self.r0/max(np.linalg.norm(self.r0),1e-9))
        position = self.shoulder0 + radius*direction
        position, reason = self._constrain_position(position)
        self.target_position = position; self.target_orientation = orientation
        self.retarget_status.update(human_extension=round(float(np.linalg.norm(h)/self.human_length),3), robot_extension=round(float(radius/self.geometry['reach']),3), reason=reason)
        return {'position':position.copy(),'orientation':orientation.copy()}, reason
