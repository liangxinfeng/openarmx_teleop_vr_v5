"""Quest3-style Placo QP IK adapter for the 306 controller interface."""

import time
import copy

import numpy as np
import placo
from .arm_geometry import arm_geometry, elbow_target

from .kinematics import (
    EEF_FRAMES,
    LEG_WAIST_JOINTS,
    LEFT_ARM_JOINTS,
    NECK_JOINTS,
    RIGHT_ARM_JOINTS,
    IkResult,
    IndependentArmKinematics,
    quaternion_xyzw_to_rotation,
)


ALL_JOINTS = (
    LEG_WAIST_JOINTS + LEFT_ARM_JOINTS + NECK_JOINTS + RIGHT_ARM_JOINTS
)


class QuestPlacoKinematics:
    """Expose Placo one-step QP IK through V3's kinematics contract.

    V5 refreshes the model from a feedback-corrected accepted seed, updates tasks,
    and executes one QP integration step per controller cycle.  It does not
    iterate a stale Cartesian goal to convergence before publishing.  The
    surrounding 306 controller still applies hard joint bounds, optional SRDF
    collision validation, feedback watchdogs and the vendor SYNC lease.
    """

    def __init__(
        self,
        urdf_path,
        srdf_path='',
        allow_waist=False,
        collision_check=True,
        limit_margin_rad=np.deg2rad(2.0),
        dt=0.01,
        frame_weight=1.0,
        manipulability_weight=0.0001,
        manipulability_min_weight=0.00002,
        manipulability_fade_start_elbow_deg=50.0,
        manipulability_fade_full_elbow_deg=25.0,
        kinetic_regularization=1.0e-6,
        posture_weight=0.0000001,
        regularization=0.0001,
    ):
        self._guard = IndependentArmKinematics(
            urdf_path,
            srdf_path,
            allow_waist=allow_waist,
            collision_check=collision_check,
            limit_margin_rad=limit_margin_rad,
        )
        # Public compatibility fields used by controller_node.py.
        self.model = self._guard.model
        self.lower = self._guard.lower
        self.upper = self._guard.upper
        self.joint_q = self._guard.joint_q
        self.joint_v = self._guard.joint_v
        self.collision_available = self._guard.collision_available
        self.collision_error = self._guard.collision_error
        self._allow_waist = bool(allow_waist)
        self._manipulability_weight_max = float(manipulability_weight)
        self._manipulability_weight_min = float(manipulability_min_weight)
        self._manipulability_fade_start_elbow_deg = float(
            manipulability_fade_start_elbow_deg
        )
        self._manipulability_fade_full_elbow_deg = float(
            manipulability_fade_full_elbow_deg
        )
        values = (
            self._manipulability_weight_max,
            self._manipulability_weight_min,
            self._manipulability_fade_start_elbow_deg,
            self._manipulability_fade_full_elbow_deg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError('Placo manipulability parameters must be finite')
        if not 0.0 <= self._manipulability_weight_min <= self._manipulability_weight_max:
            raise ValueError('Placo manipulability weight bounds are invalid')
        if not (
            0.0 <= self._manipulability_fade_full_elbow_deg
            < self._manipulability_fade_start_elbow_deg
        ):
            raise ValueError('Placo manipulability elbow envelope is invalid')

        # Quest3-Teleoperation uses Placo for differential QP IK, not for
        # geometry collision queries. Collision remains an outer Pinocchio /
        # SRDF guard, so avoid loading Placo's raw adjacent-link geometry.
        self.robot = placo.RobotWrapper(
            str(urdf_path), placo.Flags.ignore_collisions
        )
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.dt = float(dt)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.enable_velocity_limits(True)
        self._previous_commands = {}
        self._previous_velocities = {}
        self._reference_keys = {}
        self._target_history = {}
        self.last_retarget_status = {}
        self._frame_weight = float(frame_weight)
        if float(kinetic_regularization) > 0.0:
            self.solver.add_kinetic_energy_regularization_task(
                float(kinetic_regularization)
            )

        if not np.isfinite(posture_weight) or posture_weight <= 0:
            raise ValueError('posture_weight must be positive and finite')
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError('regularization must be positive and finite')
        self.solver.add_regularization_task(float(regularization))
        self._posture_weight = float(posture_weight)
        self.posture_tasks = {}
        for side in ('left', 'right'):
            task = self.solver.add_joints_task()
            task.configure(f'{side}_session_posture', 'soft', self._posture_weight)
            self.posture_tasks[side] = task

        self._placo_joint_q = {}
        for name in ALL_JOINTS:
            joint_id = self.robot.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f'Placo URDF is missing {name}')
            joint = self.robot.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f'Placo joint {name} is not one DoF')
            self._placo_joint_q[name] = int(joint.idx_q)
            # Start fail-closed. solve() opens only the joints belonging to
            # an actively commanded arm for the duration of that QP step.
            self.solver.mask_dof(name)

        self.tasks = {}
        self.manipulability_tasks = {}
        self.elbow_tasks = {}
        for side, frame in EEF_FRAMES.items():
            initial = self.robot.get_T_world_frame(frame)
            task = self.solver.add_frame_task(frame, initial)
            task.configure(f'{side}_eef', 'soft', float(frame_weight), float(frame_weight)*.25)
            elbow_frame=f'Link_{side.title()}_Elbow_to_Forearm'
            elbow=self.solver.add_position_task(elbow_frame,self.robot.get_T_world_frame(elbow_frame)[:3,3])
            elbow.configure(f'{side}_swivel', 'soft', .002)
            self.elbow_tasks[side]=elbow
            self.tasks[side] = task
            manipulability = self.solver.add_manipulability_task(
                frame, 'both', 1.0
            )
            manipulability.configure(
                f'{side}_manipulability',
                'soft',
                self._manipulability_weight_max,
            )
            self.manipulability_tasks[side] = manipulability

    @property
    def allow_waist(self):
        return self._allow_waist

    @allow_waist.setter
    def allow_waist(self, enabled):
        # The inherited UI may explicitly request V3's IK pitch/yaw profile.
        # It is opt-in; ankle, knee and neck are never opened by arm IK.
        self._allow_waist = bool(enabled)
        self._guard.allow_waist = bool(enabled)

    def q_from_feedback(self, groups_deg, *, clip=True):
        return self._guard.q_from_feedback(groups_deg, clip=clip)

    def groups_from_q_deg(self, q, base_groups_deg):
        return self._guard.groups_from_q_deg(q, base_groups_deg)

    def eef_poses(self, q):
        return self._guard.eef_poses(q)

    def collision_pairs(self, q):
        return self._guard.collision_pairs(q)

    def accept_forward_reach_boundary(self, *args, **kwargs):
        return self._guard.accept_forward_reach_boundary(*args, **kwargs)

    def _placo_from_guard(self, q_guard):
        q_guard = self._guard._configuration(q_guard)
        q_placo = self.robot.state.q.copy()
        q_placo[:7] = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
        )
        for name in ALL_JOINTS:
            q_placo[self._placo_joint_q[name]] = q_guard[self.joint_q[name]]
        return q_placo

    def _select_active_dofs(self, targets):
        active = set()
        if 'left' in targets:
            active.update(LEFT_ARM_JOINTS)
        if 'right' in targets:
            active.update(RIGHT_ARM_JOINTS)
        if self._allow_waist:
            active.update(('Joint_Waist_Pitch', 'Joint_Waist_Yaw'))
        for name in ALL_JOINTS:
            if name in active:
                self.solver.unmask_dof(name)
            else:
                self.solver.mask_dof(name)
        return active

    def _guard_from_placo(self, q_placo, q_seed, active_joints):
        output = np.asarray(q_seed, dtype=float).copy()
        # Never copy Placo values for masked joints. This is a second
        # fail-closed boundary in addition to the solver mask.
        for name in active_joints:
            output[self.joint_q[name]] = q_placo[
                self._placo_joint_q[name]
            ]
        return np.clip(output, self.lower, self.upper)

    def manipulability_weight_for(self, side, q_seed):
        """Reduce the singularity bias only as an arm approaches straight.

        The full Quest-style manipulability bias is useful through the normal
        workspace, but keeping it fixed near a straight elbow makes the QP
        fight an intentional forward reach.  A smooth lower bound retains a
        finite singularity reserve without turning the last part of the reach
        into a slow, spring-like motion.
        """
        if side not in ('left', 'right'):
            raise ValueError(f'unknown arm side: {side}')
        q_seed = self._guard._configuration(q_seed)
        elbow_name = (
            'Joint_Left_Elbow' if side == 'left' else 'Joint_Right_Elbow'
        )
        elbow_deg = abs(float(np.rad2deg(q_seed[self.joint_q[elbow_name]])))
        start = self._manipulability_fade_start_elbow_deg
        full = self._manipulability_fade_full_elbow_deg
        ratio = float(np.clip((elbow_deg - full) / (start - full), 0.0, 1.0))
        ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        return self._manipulability_weight_min + (
            self._manipulability_weight_max
            - self._manipulability_weight_min
        ) * ratio

    def solve(self, *args, transactional=False, session_epoch=None, **kwargs):
        fields=('_previous_commands','_previous_velocities','_reference_keys','_target_history')
        saved={k:copy.deepcopy(getattr(self,k)) for k in fields} if transactional else None
        self._session_epoch=session_epoch
        self._pending_retarget=None
        try:
            result=self._solve(*args,**kwargs)
            if transactional and result.success:
                self._pending_retarget={k:copy.deepcopy(getattr(self,k)) for k in fields}
            return result
        finally:
            if saved is not None:
                for k,v in saved.items():setattr(self,k,v)

    def commit_retarget_state(self):
        if self._pending_retarget is not None:
            for k,v in self._pending_retarget.items():setattr(self,k,v)
            self._pending_retarget=None

    def _solve(
        self,
        targets,
        q_seed,
        max_iterations=1,
        position_tolerance=0.004,
        orientation_tolerance=0.035,
        step_limit=0.10,
        damping=0.025,
        centering_gain=0.025,
        orientation_weight=1.0,
        posture_reference=None,
        integration_dt=None,
        feedback_q=None,
        maximum_arm_lead_rad=None,
    ):
        del max_iterations, position_tolerance, orientation_tolerance
        del step_limit, damping, centering_gain, orientation_weight
        start = time.perf_counter()
        q_seed = np.clip(
            self._guard._configuration(q_seed).copy(), self.lower, self.upper
        )
        if not targets:
            return IkResult(False, q_seed, 0, 0.0, reason='no active joints')

        unknown = set(targets) - {'left', 'right'}
        if unknown:
            return IkResult(
                False, q_seed, 0, 0.0,
                reason=f'unknown arm target(s): {sorted(unknown)}',
            )

        if integration_dt is not None:
            self.solver.dt = float(np.clip(integration_dt, 0.002, 0.03))
        reference = q_seed if posture_reference is None else self._guard._configuration(posture_reference)
        if not np.all(np.isfinite(reference)):
            raise ValueError('non-finite posture reference')
        for side, names in (('left', LEFT_ARM_JOINTS), ('right', RIGHT_ARM_JOINTS)):
            self.posture_tasks[side].set_joints({
                name: float(reference[self.joint_q[name]]) for name in names})
        measured = q_seed if feedback_q is None else self._guard._configuration(feedback_q)
        lead = np.deg2rad(12.) if maximum_arm_lead_rad is None else float(maximum_arm_lead_rad)
        if not np.isfinite(lead) or lead<=0 or not np.all(np.isfinite(measured)):
            raise ValueError('invalid measured IK lead envelope')
        self.robot.state.q = self._placo_from_guard(q_seed)
        self.robot.update_kinematics()
        active_joints = self._select_active_dofs(targets)

        geometry = arm_geometry(self.model,q_seed)
        reference_geometry = arm_geometry(self.model,reference)
        dt=self.solver.dt
        self.last_retarget_status = {}
        for side,names in (('left',LEFT_ARM_JOINTS),('right',RIGHT_ARM_JOINTS)):
            idx=np.array([self.joint_q[n] for n in names])
            key=(getattr(self,'_session_epoch',None),reference[idx].tobytes())
            if side not in targets:
                self._reference_keys.pop(side,None);self._target_history.pop(side,None)
            if self._reference_keys.get(side)!=key:
                self._previous_commands[side]=measured[idx].copy()
                self._previous_velocities[side]=np.zeros(7)
                self._target_history.pop(side,None)
                self._reference_keys[side]=key
            previous=self._previous_commands[side]
            previous_velocity=self._previous_velocities[side]
            relaxed=False
            for j,name in enumerate(names):
                qi=idx[j]
                vmax=np.deg2rad(90. if j<3 else 180.)
                self.robot.set_velocity_limit(name,vmax)
                lo=max(self.lower[qi],measured[qi]-lead,previous[j]-vmax*dt)
                hi=min(self.upper[qi],measured[qi]+lead,previous[j]+vmax*dt)
                if lo>hi:
                    # Reset only the controller-local continuity estimate after
                    # feedback displacement; never enlarge physical limits.
                    lo=max(self.lower[qi],measured[qi]-vmax*dt)
                    hi=min(self.upper[qi],measured[qi]+vmax*dt)
                accel=np.deg2rad(1200. if j<3 else 2400.)
                alo=max(lo,previous[j]+(previous_velocity[j]-accel*dt)*dt)
                ahi=min(hi,previous[j]+(previous_velocity[j]+accel*dt)*dt)
                if alo<=ahi:lo,hi=alo,ahi
                else:relaxed=True
                self.robot.set_joint_limits(name,float(lo),float(hi))
            self.last_retarget_status[side]={'acceleration_recovery':relaxed,'lead_limit_deg':float(np.rad2deg(lead))}

        # Configure before the one QP step so the newest measured elbow bend,
        # rather than an earlier command, controls the singularity reserve.
        for side in targets:
            self.manipulability_tasks[side].configure(
                f'{side}_manipulability',
                'soft',
                self.manipulability_weight_for(side, q_seed),
            )

        # Inactive arms are softly held at measured FK.  Active arms receive
        # the newest accumulated Quest target.
        seed_poses = self._guard.forward_poses(q_seed)
        for side, task in self.tasks.items():
            if side in targets:
                position, quaternion = targets[side]
                target = np.eye(4, dtype=float)
                target[:3, :3] = quaternion_xyzw_to_rotation(quaternion)
                target[:3, 3] = np.asarray(position, dtype=float)
            else:
                placement = seed_poses[side]
                target = np.eye(4, dtype=float)
                target[:3, :3] = placement.rotation
                target[:3, 3] = placement.translation
            if side in targets:
                # A short bounded Cartesian preview compensates sample latency.
                # It is applied once, before the constrained QP; there is no
                # extrapolation at the motor output and no target backlog.
                raw=target[:3,3].copy()
                prior=self._target_history.get(side)
                velocity=np.zeros(3) if prior is None else (raw-prior)/max(dt,.002)
                preview=velocity*.02
                norm=np.linalg.norm(preview)
                if norm>.008:preview*=.008/norm
                target[:3,3]+=preview
                self._target_history[side]=raw
                g=geometry[side];ref=reference_geometry[side]
                desired_elbow=elbow_target(g['shoulder'],target[:3,3],np.asarray(ref['shoulder']),np.asarray(ref['elbow']),np.asarray(ref['wrist']),g['upper_length'],g['lower_length'])
                self.elbow_tasks[side].target_world=desired_elbow
                # Position takes precedence by weight near full reach while
                # orientation remains tracked. This is weighted QP, not HQP.
                extension=np.linalg.norm(target[:3,3]-g['shoulder'])/g['reach']
                ow=.25 if extension<.9 else .08
                task.configure(f'{side}_eef','soft',self._frame_weight,self._frame_weight*ow)
                self.last_retarget_status[side].update(extension=float(extension),preview_m=float(np.linalg.norm(preview)),orientation_weight=ow)
            else:
                self.elbow_tasks[side].target_world=np.asarray(geometry[side]['elbow'])
            task.T_world_frame = target

        try:
            self.solver.solve(True)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            return IkResult(
                False, q_seed, 1, elapsed,
                reason=f'Placo QP failed: {exc}',
            )

        q = self._guard_from_placo(
            self.robot.state.q, q_seed, active_joints
        )
        if not np.all(np.isfinite(q)):
            elapsed = (time.perf_counter() - start) * 1000.0
            return IkResult(
                False, q_seed, 1, elapsed,
                reason='Placo QP produced NaN or Inf',
            )

        # Bounds were part of the QP. No per-arm scaling or post-solve
        # clipping: one saturated joint can be compensated by the other joints.
        self.last_lead_scale={}
        for side,names in (('left',LEFT_ARM_JOINTS),('right',RIGHT_ARM_JOINTS)):
            if side not in targets:continue
            idx=np.array([self.joint_q[n] for n in names])
            actual_lead=float(np.max(np.abs(q[idx]-measured[idx])))
            if actual_lead>lead+1.e-5:
                return IkResult(False,q_seed,1,0.,reason='QP feedback bound violation')
            self.last_lead_scale[side]=0. if actual_lead>=.98*lead else 1.
            self.last_retarget_status[side]['actual_lead_deg']=float(np.rad2deg(actual_lead))
        # Pose errors and collision checks must describe the bounded result.
        errors = self._guard._pose_errors(q, targets)
        position_errors = {
            side: float(np.linalg.norm(error[:3]))
            for side, error in errors.items()
        }
        orientation_errors = {
            side: float(np.linalg.norm(error[3:]))
            for side, error in errors.items()
        }
        collisions = self.collision_pairs(q)
        if not collisions:
            for side in targets:
                names=LEFT_ARM_JOINTS if side=='left' else RIGHT_ARM_JOINTS
                idx=np.array([self.joint_q[n] for n in names])
                self._previous_velocities[side]=(q[idx]-self._previous_commands[side])/max(dt,.002)
                self._previous_commands[side]=q[idx].copy()
        elapsed = (time.perf_counter() - start) * 1000.0
        return IkResult(
            not collisions,
            q,
            1,
            elapsed,
            position_errors,
            orientation_errors,
            0.0,
            collisions,
            'self collision' if collisions else 'Quest Placo QP step',
        )
