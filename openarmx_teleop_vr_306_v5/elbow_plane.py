"""One soft elbow-swivel coordinate; no arm-length or joint-angle lock.

The fixed Grip reference supplies a bend direction.  Its shortest-arc transport
to the current shoulder/wrist axis defines a plane, and the signed elbow
distance from that plane is the single task coordinate.  Joint centres, rather
than tool frames or idealised two-link lengths, come from the native model.
"""

import mink
import mujoco
import numpy as np


_PARTS = (
    'Shoulder_Inner', 'Shoulder_Outer', 'UpperArm', 'Elbow', 'Forearm',
    'Wrist_Upper', 'Wrist_Lower',
)


class ElbowPlaneTask(mink.Task):
    """Regulate the one redundant arm-plane coordinate, in metres.

    The reference is immutable between explicit ``set_reference`` calls.  The
    calculation has no previous-solution state: rejected QPs cannot gradually
    rotate the plane, and evaluating a task never changes its reference.
    """

    _AXIS_EPS = 1.e-6
    _BEND_EPS = 1.e-5
    _DIFFERENCE_STEP = 1.e-6
    _DIFFERENCE_OFFSETS = np.concatenate((np.eye(9), -np.eye(9))).reshape(
        18, 3, 3)*_DIFFERENCE_STEP

    def __init__(self, model, side, cost=1.0):
        if side not in ('left', 'right'):
            raise ValueError('elbow plane side must be left or right')
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError('elbow plane cost must be finite and nonnegative')
        super().__init__(cost=np.array([float(cost)]), gain=0.6)
        self.model = model
        self.side = side
        self._joints = {}
        for part in _PARTS:
            name = f'Joint_{side.title()}_{part}'
            joint_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                raise ValueError(f'elbow plane model is missing {name}')
            self._joints[part] = joint_id
        self._landmarks = tuple(self._joints[part] for part in (
            'Shoulder_Outer', 'Elbow', 'Wrist_Upper'))
        self._active_dofs = np.array(
            [model.jnt_dofadr[self._joints[part]] for part in _PARTS], dtype=int)
        self._reference_axis = None
        self._reference_bend = None
        self.plane_error_m = 0.0

    def set_reference(self, reference_data):
        """Capture actual shoulder/elbow/wrist joint centres at the Grip edge."""
        shoulder, elbow, wrist = self._positions(reference_data)
        axis = wrist-shoulder
        length = float(np.linalg.norm(axis))
        self._reference_axis = None
        self._reference_bend = None
        if not np.all(np.isfinite([shoulder, elbow, wrist])):
            raise ValueError('elbow plane reference contains non-finite centres')
        if length < self._AXIS_EPS:
            return
        axis /= length
        bend = elbow-shoulder
        bend -= np.dot(bend, axis)*axis
        if np.linalg.norm(bend) < self._BEND_EPS:
            # At a straight arm, the elbow rotation axis retains the intended
            # bending plane even though the three centres are nearly collinear.
            # Choose deterministically, with no dependency on a previous solve.
            elbow_axis = np.asarray(
                reference_data.xaxis[self._joints['Elbow']], dtype=float)
            upper_axis = np.asarray(
                reference_data.xaxis[self._joints['UpperArm']], dtype=float)
            candidates = (
                np.cross(elbow_axis, axis),
                upper_axis-np.dot(upper_axis, axis)*axis,
                np.cross(upper_axis, axis),
                np.eye(3)[int(np.argmin(np.abs(axis)))],
            )
            for candidate in candidates:
                candidate = candidate-np.dot(candidate, axis)*axis
                if np.linalg.norm(candidate) >= self._BEND_EPS:
                    bend = candidate
                    break
        self._reference_axis = axis.copy()
        self._reference_bend = bend/np.linalg.norm(bend)

    def _positions(self, data):
        return np.asarray(data.xanchor[list(self._landmarks)], dtype=float).copy()

    @staticmethod
    def _skew(vector):
        x, y, z = vector
        return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

    def _transport_bend(self, axis):
        reference_axis = self._reference_axis
        reference_bend = self._reference_bend
        cosine = float(np.clip(np.dot(reference_axis, axis), -1., 1.))
        if cosine < -0.9999:
            # The shortest rotation is undefined at antipodal axes.  Resolve
            # it about the *fixed reference bend*, then apply the tiny remaining
            # correction from -reference_axis to axis.  This is finite at 180
            # degrees and does not select an axis from numerical noise.
            half_turn = 2.*np.outer(reference_bend, reference_bend)-np.eye(3)
            source = -reference_axis
            cross = np.cross(source, axis)
            skew = self._skew(cross)
            correction = (np.eye(3)+skew+skew@skew/
                          max(1.+float(np.dot(source, axis)), 1.e-12))
            bend = correction@half_turn@reference_bend
        else:
            skew = self._skew(np.cross(reference_axis, axis))
            rotation = np.eye(3)+skew+skew@skew/(1.+cosine)
            bend = rotation@reference_bend
        bend -= np.dot(bend, axis)*axis
        length = float(np.linalg.norm(bend))
        return bend/length if length >= self._BEND_EPS else None

    def _geometric_error(self, positions):
        if self._reference_axis is None:
            return 0.0
        shoulder, elbow, wrist = np.asarray(positions).reshape(3, 3)
        axis = wrist-shoulder
        length = float(np.linalg.norm(axis))
        if not np.isfinite(length) or length < self._AXIS_EPS:
            return 0.0
        axis /= length
        # A unique shortest-arc reference cannot be continued through the
        # antipode without a branch.  Fade this *soft* task to zero before the
        # ambiguous direction, rather than commanding a sudden elbow flip.
        cosine = float(np.clip(np.dot(self._reference_axis, axis), -1., 1.))
        if cosine <= -0.9999:
            return 0.0
        antipodal = float(np.clip((cosine+0.9999)/0.0199, 0., 1.))
        antipodal = antipodal*antipodal*(3.-2.*antipodal)
        short_axis = float(np.clip((length-0.02)/0.04, 0., 1.))
        short_axis = short_axis*short_axis*(3.-2.*short_axis)
        bend = self._transport_bend(axis)
        if bend is None:
            return 0.0
        normal = np.cross(axis, bend)
        return float(antipodal*short_axis*np.dot(normal, elbow-shoulder))

    def compute_error(self, configuration):
        self.plane_error_m = self._geometric_error(
            self._positions(configuration.data))
        return np.array([self.plane_error_m])

    def _geometric_errors(self, positions):
        """Vectorised scalar error for the 18 Cartesian difference samples.

        For a reference tangent b0 perpendicular to a0, shortest-arc parallel
        transport has the identity b = b0 - (b0.a)/(1+a0.a)*(a0+a).
        This avoids constructing 18 Rodrigues matrices.  The same normalisation
        and both fades as the scalar error remain part of differentiation.
        """
        shoulder, elbow, wrist = (positions[:, index, :] for index in range(3))
        axes = wrist-shoulder
        lengths = np.linalg.norm(axes, axis=1)
        axes = axes/np.maximum(lengths[:, None], self._AXIS_EPS)
        cosines = np.clip(axes@self._reference_axis, -1., 1.)
        valid = np.isfinite(lengths) & (lengths >= self._AXIS_EPS) & (cosines > -0.9999)
        bends = (self._reference_bend-
                 ((axes@self._reference_bend)/np.maximum(1.+cosines, 1.e-12))[:, None]
                 *(self._reference_axis+axes))
        bends -= np.einsum('ij,ij->i', bends, axes)[:, None]*axes
        bend_lengths = np.linalg.norm(bends, axis=1)
        valid &= bend_lengths >= self._BEND_EPS
        bends = bends/np.maximum(bend_lengths[:, None], self._BEND_EPS)
        antipodal = np.clip((cosines+0.9999)/0.0199, 0., 1.)
        antipodal = antipodal*antipodal*(3.-2.*antipodal)
        short_axis = np.clip((lengths-0.02)/0.04, 0., 1.)
        short_axis = short_axis*short_axis*(3.-2.*short_axis)
        errors = antipodal*short_axis*np.einsum(
            'ij,ij->i', np.cross(axes, bends), elbow-shoulder)
        return np.where(valid, errors, 0.)

    def compute_jacobian(self, configuration):
        positions = self._positions(configuration.data)
        jacobian = np.zeros((1, self.model.nv))
        if (self._reference_axis is None or
                np.linalg.norm(positions[2]-positions[0]) < self._AXIS_EPS):
            return jacobian
        # Differentiate only nine Cartesian coordinates.  In particular this
        # includes the plane-normal change when S or W moves; differentiating
        # only E would be an inconsistent task Jacobian.  No extra FK is run.
        samples = positions[None, :, :]+self._DIFFERENCE_OFFSETS
        errors = self._geometric_errors(samples)
        gradient = (errors[:9]-errors[9:])/(2.*self._DIFFERENCE_STEP)
        for landmark, joint_id in enumerate(self._landmarks):
            point_jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jac(
                self.model, configuration.data, point_jacobian, None,
                positions[landmark], int(self.model.jnt_bodyid[joint_id]))
            jacobian[0] += gradient[3*landmark:3*landmark+3]@point_jacobian
        # The whole-body controller owns all other joints.  This arm task must
        # never invite the QP to borrow the waist, neck or opposite arm.
        active = jacobian[:, self._active_dofs].copy()
        jacobian[:] = 0.0
        jacobian[:, self._active_dofs] = active
        return jacobian
