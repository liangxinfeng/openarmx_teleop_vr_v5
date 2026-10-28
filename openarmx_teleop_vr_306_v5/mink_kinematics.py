"""One-step Mink/quadprog arm IK with transactional command continuity.

MuJoCo supplies kinematics only.  This module does not step physics, create a
viewer, or communicate with motors.  Pinocchio remains the independent FK and
collision guard used by the surrounding controller.
"""

import copy
import time

import mink
import mujoco
import numpy as np
from mink.limits.limit import Constraint, Limit

from .kinematics import (
    EEF_FRAMES, LEG_WAIST_JOINTS, LEFT_ARM_JOINTS, NECK_JOINTS,
    RIGHT_ARM_JOINTS, IkResult, IndependentArmKinematics,
    quaternion_xyzw_to_rotation,
)
from .mujoco_urdf_model import build_mujoco_model
from .elbow_plane import ElbowPlaneTask
from .return_shape import return_shape_step


ALL_JOINTS = LEG_WAIST_JOINTS + LEFT_ARM_JOINTS + NECK_JOINTS + RIGHT_ARM_JOINTS
ARM_JOINTS = {'left': LEFT_ARM_JOINTS, 'right': RIGHT_ARM_JOINTS}


class _CommandEnvelopeLimit(Limit):
    """Absolute final-command bounds expressed in Mink's delta-q variable.

All imported joints are scalar hinges/slides.  Uncommanded DOFs have equal
lower/upper bounds at the seed, so the QP cannot borrow waist or other-arm
motion to solve an active hand's task.
    """

    def __init__(self, model, lower, upper):
        self.lower = np.asarray(lower, dtype=float).copy()
        self.upper = np.asarray(upper, dtype=float).copy()
        if self.lower.shape != (model.nq,) or self.upper.shape != (model.nq,):
            raise ValueError('command envelope shape mismatch')
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError('command envelope must be finite')
        if np.any(self.lower > self.upper):
            raise ValueError('command envelope is infeasible')
        self.projection = np.zeros((model.nq, model.nv))
        for jid in range(model.njnt):
            if int(model.jnt_type[jid]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
            ):
                raise ValueError('Mink arm adapter requires fixed-base scalar joints')
            self.projection[model.jnt_qposadr[jid], model.jnt_dofadr[jid]] = 1.0

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        return Constraint(
            G=np.vstack((self.projection, -self.projection)),
            h=np.concatenate((self.upper-configuration.q, configuration.q-self.lower)),
        )


class _ScalarLimitIntersection(Limit):
    """Canonical box intersection; do not feed duplicate rows to quadprog.

The three limits remain independently computed.  For this scalar-joint model
their intersection is exactly one lower/upper bound per active DOF.  Inactive
DOFs use hard equalities instead of the degenerate pair +dq<=0, -dq<=0.
    """

    def __init__(self, model, limits, active_dofs):
        self.nv = model.nv
        self.limits = limits
        self.active_dofs = np.asarray(sorted(active_dofs), dtype=int)
        self.projection = np.eye(model.nv)[self.active_dofs]

    def compute_qp_inequalities(self, configuration, dt):
        lower = np.full(self.nv, -np.inf)
        upper = np.full(self.nv, np.inf)
        for limit in self.limits:
            constraint = limit.compute_qp_inequalities(configuration, dt)
            if constraint.inactive:
                continue
            rows, rhs = constraint.G, constraint.h
            if np.any(np.count_nonzero(rows, axis=1) != 1):
                raise ValueError('non-scalar constraint in scalar joint envelope')
            indices = np.argmax(np.abs(rows), axis=1)
            coefficients = rows[np.arange(len(rows)), indices]
            values = rhs/coefficients
            positive = coefficients > 0
            np.minimum.at(upper, indices[positive], values[positive])
            np.maximum.at(lower, indices[~positive], values[~positive])
        lower, upper = lower[self.active_dofs], upper[self.active_dofs]
        if np.any(lower > upper) or not np.all(np.isfinite(lower+upper)):
            raise ValueError('intersected QP command envelope is infeasible')
        return Constraint(G=np.vstack((self.projection, -self.projection)),
                          h=np.concatenate((upper, -lower)))


class _FrozenDofsConstraint:
    """Mink hard-constraint protocol: inactive generalized velocities are zero."""

    gain = 1.0

    def __init__(self, model, active_dofs):
        inactive = [i for i in range(model.nv) if i not in active_dofs]
        self.projection = np.eye(model.nv)[inactive]

    def compute_error(self, configuration):
        del configuration
        return np.zeros(len(self.projection))

    def compute_jacobian(self, configuration):
        del configuration
        return self.projection


class MinkKinematics:
    """Controller-compatible, fixed-seven-DoF-per-arm differential IK.

FrameTask, elbow-plane, PostureTask and DampingTask are weighted costs, not strict
hierarchical priorities.  A successful result is a valid bounded control step;
it does not claim that the Cartesian target has already been reached.
    """

    _HISTORY_FIELDS = (
        '_previous_commands', '_previous_velocities', '_reference_keys',
        '_session_references', '_target_history', '_return_shape_history',
    )
    algorithm_revision = 'shoulder-progress-lead-20261028'

    def __init__(
        self, urdf_path, srdf_path='', allow_waist=False, collision_check=True,
        limit_margin_rad=np.deg2rad(2.0), dt=0.01, frame_weight=1.0,
        orientation_weight=0.25, posture_weight=0.003, regularization=0.01,
        shoulder_velocity_deg_s=90.0, other_velocity_deg_s=180.0,
        shoulder_acceleration_deg_s2=1200.0, other_acceleration_deg_s2=2400.0,
    ):
        if allow_waist:
            raise ValueError('Mink V5 is seven-DoF arm IK; waist IK is unsupported')
        for name, value in (
            ('dt', dt), ('frame_weight', frame_weight),
            ('posture_weight', posture_weight), ('regularization', regularization),
            ('shoulder_velocity_deg_s', shoulder_velocity_deg_s),
            ('other_velocity_deg_s', other_velocity_deg_s),
            ('shoulder_acceleration_deg_s2', shoulder_acceleration_deg_s2),
            ('other_acceleration_deg_s2', other_acceleration_deg_s2),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be positive and finite')
        if not np.isfinite(orientation_weight) or orientation_weight < 0.0:
            raise ValueError('orientation_weight must be finite and nonnegative')
        self._dt = float(np.clip(dt, 0.002, 0.03))
        self._frame_weight = float(frame_weight)
        self._orientation_weight = float(orientation_weight)
        self._posture_weight = float(posture_weight)
        self._regularization = float(regularization)
        self._velocity = np.deg2rad(
            [shoulder_velocity_deg_s]*3 + [other_velocity_deg_s]*4)
        self._acceleration = np.deg2rad(
            [shoulder_acceleration_deg_s2]*3 + [other_acceleration_deg_s2]*4)
        self._guard = IndependentArmKinematics(
            urdf_path, srdf_path, allow_waist=False,
            collision_check=collision_check, limit_margin_rad=limit_margin_rad)
        for name in ('model', 'lower', 'upper', 'joint_q', 'joint_v',
                     'collision_available', 'collision_error'):
            setattr(self, name, getattr(self._guard, name))
        self.mujoco_model = build_mujoco_model(urdf_path, EEF_FRAMES)
        self._mj_q = {}
        self._mj_v = {}
        for name in ALL_JOINTS:
            jid = int(mujoco.mj_name2id(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if jid < 0:
                raise ValueError(f'MuJoCo model is missing {name}')
            if int(self.mujoco_model.jnt_type[jid]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
            ):
                raise ValueError(f'{name} must be a scalar joint')
            self._mj_q[name] = int(self.mujoco_model.jnt_qposadr[jid])
            self._mj_v[name] = int(self.mujoco_model.jnt_dofadr[jid])
            # The native effective motor limits can be narrower than URDF.
            self.mujoco_model.jnt_limited[jid] = True
            self.mujoco_model.jnt_range[jid] = (
                self.lower[self.joint_q[name]], self.upper[self.joint_q[name]])
        self.configuration = mink.Configuration(self.mujoco_model)
        self._reference_configuration = mink.Configuration(self.mujoco_model)
        self.tasks = {
            side: mink.FrameTask(
                frame_name=frame, frame_type='site',
                position_cost=self._frame_weight,
                orientation_cost=self._orientation_weight, gain=1.0)
            for side, frame in EEF_FRAMES.items()
        }
        self.posture_tasks = {
            side: mink.PostureTask(self.mujoco_model, cost=np.zeros(self.mujoco_model.nv))
            for side in ARM_JOINTS
        }
        # Only the redundant elbow plane is regulated.  No ideal two-link
        # lengths or 3D elbow position are imposed on this offset-joint URDF.
        self.elbow_plane_tasks = {
            side: ElbowPlaneTask(self.mujoco_model, side, cost=1.0)
            for side in ARM_JOINTS
        }
        self._site_ids = {side: int(mujoco.mj_name2id(
            self.mujoco_model, mujoco.mjtObj.mjOBJ_SITE, frame))
            for side, frame in EEF_FRAMES.items()}
        self._shoulder_joint_ids = {side: int(mujoco.mj_name2id(
            self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, names[1]))
            for side, names in ARM_JOINTS.items()}
        self.damping_task = mink.DampingTask(
            self.mujoco_model, cost=self._regularization)
        self.configuration_limit = mink.ConfigurationLimit(self.mujoco_model, gain=1.0)
        self.velocity_limit = mink.VelocityLimit(
            self.mujoco_model,
            {name: self._velocity[j] for names in ARM_JOINTS.values()
             for j, name in enumerate(names)})
        for field in self._HISTORY_FIELDS:
            setattr(self, field, {})
        self._pending_retarget = None
        self.last_retarget_status = {}
        self.last_lead_scale = {}

    @property
    def allow_waist(self):
        return False

    @allow_waist.setter
    def allow_waist(self, enabled):
        if enabled:
            raise ValueError('Mink V5 is seven-DoF arm IK; waist IK is unsupported')

    def q_from_feedback(self, groups_deg, *, clip=True):
        return self._guard.q_from_feedback(groups_deg, clip=clip)

    def groups_from_q_deg(self, q, base_groups_deg):
        return self._guard.groups_from_q_deg(q, base_groups_deg)

    def eef_poses(self, q):
        return self._guard.eef_poses(q)

    def forward_poses(self, q):
        return self._guard.forward_poses(q)

    def collision_pairs(self, q):
        return self._guard.collision_pairs(q)

    def accept_forward_reach_boundary(self, *args, **kwargs):
        return self._guard.accept_forward_reach_boundary(*args, **kwargs)

    def _mujoco_from_guard(self, q):
        q = self._guard._configuration(q)
        result = self.mujoco_model.qpos0.copy()
        for name in ALL_JOINTS:
            result[self._mj_q[name]] = q[self.joint_q[name]]
        return result

    def solve(self, *args, transactional=False, session_epoch=None, **kwargs):
        # Failed or superseded solves never become the next command's history.
        saved = {field: copy.deepcopy(getattr(self, field))
                 for field in self._HISTORY_FIELDS}
        self._pending_retarget = None
        result = None
        try:
            result = self._solve(*args, session_epoch=session_epoch, **kwargs)
            if transactional and result.success:
                self._pending_retarget = {
                    field: copy.deepcopy(getattr(self, field))
                    for field in self._HISTORY_FIELDS}
            return result
        finally:
            if transactional or result is None or not result.success:
                for field, value in saved.items():
                    setattr(self, field, value)
            # Diagnostic errors describe the latest attempted solution.  Mark
            # acceptance explicitly so a discarded solve cannot look like an
            # actually issued motor command in the UI.
            accepted = result is not None and result.success and not transactional
            for status in self.last_retarget_status.values():
                status['accepted'] = bool(accepted)
                status['attempt_status'] = (
                    'accepted' if accepted else 'pending' if result is not None
                    and result.success else 'rejected')
                if result is not None and not result.success:
                    status['attempt_reason'] = result.reason

    def commit_retarget_state(self):
        if self._pending_retarget is not None:
            for field, value in self._pending_retarget.items():
                setattr(self, field, value)
            self._pending_retarget = None
            for status in self.last_retarget_status.values():
                status['accepted'] = True
                status['attempt_status'] = 'accepted'

    def _solve(
        self, targets, q_seed, max_iterations=1, position_tolerance=0.004,
        orientation_tolerance=np.deg2rad(0.19), step_limit=0.10,
        damping=0.025, centering_gain=0.025, orientation_weight=1.0,
        posture_reference=None, integration_dt=None, feedback_q=None,
        maximum_arm_lead_rad=None, maximum_shoulder_lead_rad=None, session_epoch=None, reset_reference=None,
        return_start_m=.45, return_full_m=.08, return_transition_sec=1.0,
    ):
        # Old Pinocchio call arguments are accepted for the controller contract;
        # this backend always performs one QP and owns its task weights.
        del max_iterations, step_limit, damping, centering_gain, orientation_weight
        start = time.perf_counter()
        q_seed = self._guard._configuration(q_seed).copy()
        self.last_retarget_status = {}
        self.last_lead_scale = {}

        def failed(reason):
            return IkResult(False, q_seed, 0,
                            (time.perf_counter()-start)*1000.0, reason=reason)

        if not targets:
            return failed('no active joints')
        if set(targets)-set(ARM_JOINTS):
            return failed(f'unknown arm target(s): {sorted(set(targets)-set(ARM_JOINTS))}')
        if np.any(q_seed < self.lower-1.e-7) or np.any(q_seed > self.upper+1.e-7):
            return failed('IK seed outside effective joint limits')
        dt = self._dt if integration_dt is None else float(integration_dt)
        if not np.isfinite(dt) or dt <= 0.0:
            return failed('invalid integration timestep')
        dt = float(np.clip(dt, 0.002, 0.03))
        reference = q_seed if posture_reference is None else self._guard._configuration(posture_reference)
        home = None if reset_reference is None else self._guard._configuration(reset_reference)
        measured = q_seed if feedback_q is None else self._guard._configuration(feedback_q)
        lead = np.deg2rad(12.0) if maximum_arm_lead_rad is None else float(maximum_arm_lead_rad)
        if not np.isfinite(lead) or lead <= 0.0:
            return failed('invalid measured IK lead envelope')
        if maximum_shoulder_lead_rad is None:
            shoulder_lead = lead
        else:
            shoulder_lead = float(maximum_shoulder_lead_rad)
            if not np.isfinite(shoulder_lead) or shoulder_lead < lead or shoulder_lead > np.deg2rad(14.0)+1.e-12:
                return failed('invalid shoulder feedback lead envelope')
        # These are convergence labels only, never a blocking inner-loop target.
        pos_tol = float(position_tolerance)
        rot_tol = float(orientation_tolerance)
        if not (np.isfinite(pos_tol) and pos_tol > 0 and np.isfinite(rot_tol) and rot_tol > 0):
            return failed('invalid convergence tolerances')
        seed_mj = self._mujoco_from_guard(q_seed)
        self.configuration.update(seed_mj)
        lower, upper = seed_mj.copy(), seed_mj.copy()
        task_stack = [self.damping_task]
        raw_targets = {}
        bounds = {}

        for side, names in ARM_JOINTS.items():
            if side not in targets:
                continue
            idx = np.asarray([self.joint_q[name] for name in names])
            epoch = session_epoch.get(side) if isinstance(session_epoch, dict) else session_epoch
            # In production the per-arm token is provided.  The reference is
            # only a backwards-compatible identity when there is no token.
            key = ('epoch', epoch) if epoch is not None else ('reference', reference[idx].tobytes())
            if self._reference_keys.get(side) != key:
                self._reference_keys[side] = key
                self._session_references[side] = reference[idx].copy()
                self._previous_commands[side] = measured[idx].copy()
                self._previous_velocities[side] = np.zeros(7)
                self._target_history.pop(side, None)
                self._return_shape_history.pop(side, None)
            previous = self._previous_commands[side]
            previous_velocity = self._previous_velocities[side]
            lead_vector = np.full(7, lead)
            lead_vector[:2] = shoulder_lead
            lo = np.maximum.reduce((self.lower[idx], measured[idx]-lead_vector,
                                    previous-self._velocity*dt,
                                    q_seed[idx]-self._velocity*dt))
            hi = np.minimum.reduce((self.upper[idx], measured[idx]+lead_vector,
                                    previous+self._velocity*dt,
                                    q_seed[idx]+self._velocity*dt))
            if np.any(lo > hi+1.e-12):
                return failed(f'{side} feedback/command envelope infeasible; release Grip and re-engage')
            # Anticipate static joint limits instead of arriving with a
            # nonzero speed and dropping the acceleration envelope next frame.
            # A local 120ms velocity damper lets the redundant joints take over
            # smoothly. It does not change the Cartesian target, motor limits,
            # normal velocity caps, feedback lead, or reset trajectory.
            approach_gain = -np.expm1(-dt/.12)/dt
            lo = np.maximum(lo, previous-approach_gain*np.maximum(0., previous-self.lower[idx])*dt)
            hi = np.minimum(hi, previous+approach_gain*np.maximum(0., self.upper[idx]-previous)*dt)
            # Acceleration is subordinate to physical position/speed/lead
            # bounds.  Report any recovery instead of widening those bounds.
            alo = np.maximum(lo, previous+(previous_velocity-self._acceleration*dt)*dt)
            ahi = np.minimum(hi, previous+(previous_velocity+self._acceleration*dt)*dt)
            recover = alo > ahi
            lo = np.where(recover, lo, alo)
            hi = np.where(recover, hi, ahi)
            bounds[side] = (lo, hi)
            for j, name in enumerate(names):
                lower[self._mj_q[name]], upper[self._mj_q[name]] = lo[j], hi[j]

            position, quaternion = targets[side]
            position = np.asarray(position, dtype=float)
            quaternion = np.asarray(quaternion, dtype=float)
            if position.shape != (3,) or quaternion.shape != (4,) or not (
                np.all(np.isfinite(position)) and np.all(np.isfinite(quaternion))
            ):
                return failed(f'invalid {side} Cartesian target')
            try:
                transform = np.eye(4)
                transform[:3, :3] = quaternion_xyzw_to_rotation(quaternion)
                transform[:3, 3] = position
            except ValueError as exc:
                return failed(f'invalid {side} Cartesian target: {exc}')
            raw_targets[side] = transform
            # Near extension, following every orientation residual can spend
            # the remaining motion on axial arm rotation. Ease this cost
            # smoothly, while retaining a nonzero orientation objective.
            elbow = abs(float(q_seed[idx[3]]))
            extension = float(np.clip((np.deg2rad(30.0)-elbow)/np.deg2rad(25.0), 0., 1.))
            extension = extension*extension*(3.-2.*extension)
            # Estimate this frame's remaining outward reach using native FK
            # with only the elbow extended to its effective limit. Do not
            # force an ideal arm length or alter inward/tangential requests.
            # Raw requested errors below remain measured against the actual
            # operator target, so a projected request is never called reached.
            effective_transform = transform.copy()
            reach_projection = 0.0
            if extension > 0.0:
                site = self._site_ids[side]
                shoulder_id = self._shoulder_joint_ids[side]
                shoulder = self.configuration.data.xanchor[shoulder_id].copy()
                current_position = self.configuration.data.site_xpos[site].copy()
                radial = current_position-shoulder
                radius = float(np.linalg.norm(radial))
                if radius > 0.05:
                    radial /= radius
                    probe = seed_mj.copy()
                    probe[self._mj_q[names[3]]] = float(np.clip(0., self.lower[idx[3]], self.upper[idx[3]]))
                    self._reference_configuration.update(probe)
                    probe_radius = float(np.linalg.norm(
                        self._reference_configuration.data.site_xpos[site]
                        - self._reference_configuration.data.xanchor[shoulder_id]))
                    outward = float(np.dot(position-current_position, radial))
                    available = max(0., probe_radius-radius)
                    reach_projection = extension*max(0., outward-available)
                    effective_transform[:3, 3] -= reach_projection*radial
            self.tasks[side].set_target(mink.SE3.from_matrix(effective_transform))
            orientation_cost = self._orientation_weight*(1.-0.5*extension)
            self.tasks[side].set_orientation_cost(orientation_cost)
            task_stack.append(self.tasks[side])

            # Retain the frozen Grip reference for normal teleoperation.
            # Home is read from the very same parameters as X+A, never cached
            # as a separate pose. Reconstruct it at the current torso pose.
            shape = self._session_references[side].copy()
            site = self._site_ids[side]
            reference_mj = seed_mj.copy()
            for j, name in enumerate(names):
                reference_mj[self._mj_q[name]] = shape[j]
            self._reference_configuration.update(reference_mj)
            ref_position = self._reference_configuration.data.site_xpos[site].copy()
            ref_rotation = self._reference_configuration.data.site_xmat[site].reshape(3, 3).copy()
            if home is None:
                # Legacy direct solver callers retain their original contract.
                return_distance = float(np.linalg.norm(position-ref_position))
                cosine = np.clip((np.trace(ref_rotation.T@transform[:3, :3])-1.)/2., -1., 1.)
                p = float(np.clip(return_distance/.10, 0., 1.))
                r = float(np.clip(np.arccos(cosine)/.60, 0., 1.))
                return_blend = (1.-p*p*(3.-2.*p))*(1.-r*r*(3.-2.*r))
            else:
                home_arm = home[idx]
                if np.any(home_arm < self.lower[idx]-1.e-7) or np.any(home_arm > self.upper[idx]+1.e-7):
                    return failed(f'{side} reset arm reference outside effective limits')
                for j, name in enumerate(names):
                    reference_mj[self._mj_q[name]] = home_arm[j]
                self._reference_configuration.update(reference_mj)
                home_position = self._reference_configuration.data.site_xpos[site].copy()
                return_distance = float(np.linalg.norm(position-home_position))
                state = return_shape_step(
                    self._return_shape_history.get(side), return_distance, dt,
                    home_arm.tobytes(), start=float(return_start_m), full=float(return_full_m),
                    transition=float(return_transition_sec))
                self._return_shape_history[side] = state
                return_blend = state['blend']
                # Only shoulders, upper arm and elbow receive a home bias.
                # Keep forearm/wrist references and costs unchanged.
                shape[:4] += return_blend*(home_arm[:4]-shape[:4])
            for j, name in enumerate(names):
                reference_mj[self._mj_q[name]] = shape[j]
            self._reference_configuration.update(reference_mj)
            plane = self.elbow_plane_tasks[side]
            plane.set_reference(self._reference_configuration.data)
            task_stack.append(plane)
            posture_cost = self._posture_weight*(1.+3.*return_blend)
            costs = np.zeros(self.mujoco_model.nv)
            posture_q = seed_mj.copy()
            for j, name in enumerate(names):
                axial_multiplier = 2.0 if j in (2, 4) else 1.0
                joint_cost = posture_cost if home is None or j < 4 else self._posture_weight
                costs[self._mj_v[name]] = joint_cost*axial_multiplier
                posture_q[self._mj_q[name]] = shape[j]
            self.posture_tasks[side].set_cost(costs)
            self.posture_tasks[side].set_target(posture_q)
            task_stack.append(self.posture_tasks[side])
            self.last_retarget_status[side] = {
                'solver': 'mink/quadprog', 'reference_locked': True,
                'algorithm_revision': self.algorithm_revision,
                'posture_cost': posture_cost,
                'damping_cost': self._regularization,
                'orientation_cost': orientation_cost,
                'elbow_plane_cost': float(plane.cost[0]),
                'return_blend': return_blend,
                'return_reference': 'vr_quick_reset' if home is not None else 'grip',
                'return_distance_m': return_distance,
                'return_start_m': float(return_start_m) if home is not None else .10,
                'extension_blend': extension,
                'reach_projection_m': reach_projection,
                'lead_limit_deg': float(np.rad2deg(lead)),
                'shoulder_lead_limit_deg': float(np.rad2deg(shoulder_lead)),
                'acceleration_recovery': bool(np.any(recover)),
                'position_tolerance_m': pos_tol,
                'orientation_tolerance_deg': float(np.rad2deg(rot_tol)),
            }
        try:
            envelope = _CommandEnvelopeLimit(self.mujoco_model, lower, upper)
            active_dofs = {self._mj_v[name] for side in targets for name in ARM_JOINTS[side]}
            limits = _ScalarLimitIntersection(
                self.mujoco_model,
                [self.configuration_limit, self.velocity_limit, envelope], active_dofs)
            frozen = _FrozenDofsConstraint(self.mujoco_model, active_dofs)
            velocity = mink.solve_ik(
                self.configuration, task_stack, dt, solver='quadprog',
                damping=1.e-8, safety_break=True,
                limits=[limits], constraints=[frozen])
            if not np.all(np.isfinite(velocity)):
                return failed('Mink QP produced non-finite velocity')
            self.configuration.integrate_inplace(velocity, dt)
        except Exception as exc:
            return failed(f'Mink quadprog failed: {exc}')

        q = q_seed.copy()
        for side, names in ARM_JOINTS.items():
            if side not in targets:
                continue
            idx = np.asarray([self.joint_q[name] for name in names])
            output = np.asarray([self.configuration.q[self._mj_q[name]] for name in names])
            lo, hi = bounds[side]
            if not np.all(np.isfinite(output)) or np.any(output < lo-1.e-7) or np.any(output > hi+1.e-7):
                return failed(f'{side} QP final command bound violation')
            q[idx] = output
            actual_lead = float(np.max(np.abs(output-measured[idx])))
            lead_vector = np.full(7, lead)
            lead_vector[:2] = shoulder_lead
            lead_ratio = float(np.max(np.abs(output-measured[idx])/lead_vector))
            self.last_lead_scale[side] = 0.0 if lead_ratio >= 0.98 else 1.0
            self.last_retarget_status[side]['actual_lead_deg'] = float(np.rad2deg(actual_lead))
            self.last_retarget_status[side]['elbow_plane_error_m'] = float(
                self.elbow_plane_tasks[side].compute_error(self.configuration)[0])
        errors = self._guard._pose_errors(q, targets)
        position_errors = {side: float(np.linalg.norm(error[:3])) for side, error in errors.items()}
        orientation_errors = {side: float(np.linalg.norm(error[3:])) for side, error in errors.items()}
        collisions = self.collision_pairs(q)
        if collisions:
            return IkResult(False, q, 1, (time.perf_counter()-start)*1000.0,
                            position_errors, orientation_errors, 0.0, collisions,
                            'self collision')
        any_plateau = False
        for side in targets:
            idx = np.asarray([self.joint_q[name] for name in ARM_JOINTS[side]])
            self._previous_velocities[side] = (q[idx]-self._previous_commands[side])/dt
            self._previous_commands[side] = q[idx].copy()
            converged = position_errors[side] <= pos_tol and orientation_errors[side] <= rot_tol
            previous_target = self._target_history.get(side)
            current_target = raw_targets[side]
            static = previous_target is not None and (
                np.linalg.norm(current_target[:3, 3]-previous_target['target'][:3, 3]) < 0.0005
                and np.linalg.norm(current_target[:3, :3]-previous_target['target'][:3, :3]) < 0.003)
            metric = position_errors[side]+0.1*orientation_errors[side]
            samples = (previous_target['samples']+[metric])[-30:] if static else [metric]
            duration = previous_target['duration']+dt if static else 0.0
            plateau = bool(not converged and duration >= 0.29 and len(samples) == 30
                           and max(samples)-min(samples) < 1.e-5)
            self._target_history[side] = {
                'target': previous_target['target'] if static else current_target.copy(),
                'samples': samples, 'duration': duration,
            }
            self.last_retarget_status[side].update(
                state='converged' if converged else 'plateau' if plateau else 'tracking',
                converged=bool(converged), plateau=plateau,
                position_error_m=position_errors[side],
                orientation_error_deg=float(np.rad2deg(orientation_errors[side])),
            )
            any_plateau |= plateau
        return IkResult(
            True, q, 1, (time.perf_counter()-start)*1000.0,
            position_errors, orientation_errors, 0.0, [],
            'Mink QP plateau (not converged)' if any_plateau else 'Mink QP step',
        )
