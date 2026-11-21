"""GenCoord representation extended with one rigid object.

Full state: robot ``[root 13, qj, vj]`` + object ``[pos 3, quat 4, lin_vel 3,
ang_vel 3]`` + the world's object variant (1, passed through unchanged).
Aux state: ``[root_pos, root_quat, obj_pos, obj_quat, variant]``.
Rep state: robot GenCoord rep + object in the root frame ``[pos 3, rot6d 6,
lin_vel 3, ang_vel 3]`` + object height (1) + hand-object features (24) +
object clearance (1). The features are recomputed from the integrated state, so
the network only predicts velocity deltas: robot ``[dv_b, dw_b, dvj]`` plus the
object's ``[dv, dw]`` in the current root frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul

from supermimic.hoi import terms
from supermimic.hoi.env_cfg import hoi_task_cfg
from supermimic.hoi.geometry import HAND_FEATURE_DIM, NUM_HANDS, NUM_PROBES, hand_object_features, hand_probes, object_clearance
from supermimic.hoi.goals import hoi_goal
from supermimic.hoi.mdp import WRIST_BODIES
from supermimic.representations import GenCoordRepresentation
from supermimic.utils.gencoord_helpers import compute_gencoord_state, decompose_rep_state
from supermimic.utils.helpers import quat_integrate, quat_to_rot6d
from supermimic.utils.tracking_reward import tracking_reward

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.sim_context import SimContext

OBJ_FULL_DIM = 14
OBJ_REP_DIM = 16 + HAND_FEATURE_DIM + 1
_LIN, _ANG, _FEAT = slice(9, 12), slice(12, 15), slice(16, 16 + HAND_FEATURE_DIM)


class HoiGenCoordRepresentation(GenCoordRepresentation):
  LOSS_WEIGHT_CONFIG = {
    **GenCoordRepresentation.LOSS_WEIGHT_CONFIG,
    "object_pos": "nominal_error_object_pos",
    "object_ori": "nominal_error_object_ori",
  }

  def __init__(self, num_joints: int, default_joint_pos: torch.Tensor):
    super().__init__(num_joints, default_joint_pos)
    self.robot_full_dim = 13 + 2 * num_joints
    self.robot_rep_dim = 10 + 2 * num_joints
    self.full_state_dim = self.robot_full_dim + OBJ_FULL_DIM
    self.rep_state_dim = self.robot_rep_dim + OBJ_REP_DIM
    self.prediction_dim = 6 + num_joints + 6

  @staticmethod
  def num_joints_from_full_dim(full_state_dim: int) -> int:
    return (full_state_dim - OBJ_FULL_DIM - 13) // 2

  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    super().bind(env, sim_context)
    cmd = env.command_manager.get_term("motion")
    self.lib = cmd.lib
    self.task = hoi_task_cfg(env.cfg)
    self.reward_weights = self.task.reward_weights()
    self.probes, self.radii = hand_probes(str(self._default_joint_pos.device))
    self._wrist_idx = [cmd.cfg.body_names.index(b) for b in WRIST_BODIES]
    self.context_columns = sim_context.find("object_identity")

  FULL_ACTOR_GOAL = True

  # ------------------------------------------------------------------ state

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    rf = self.robot_full_dim
    return torch.cat([full_state[..., :7], full_state[..., rf : rf + 7], full_state[..., rf + 13 : rf + 14]], -1)

  def _object_rep(self, robot_rep, aux, obj_lin_w, obj_ang_w) -> torch.Tensor:
    root_pos, root_quat = aux[:, :3], aux[:, 3:7]
    obj_pos, obj_quat = aux[:, 7:10], aux[:, 10:14]
    variant = aux[:, 14].round().long()
    _, _, _, _, qj, _ = decompose_rep_state(robot_rep)
    zeros3 = torch.zeros_like(root_pos)
    pos, quat, _, _ = self.fk(root_pos, root_quat, qj, zeros3, zeros3, torch.zeros_like(qj))
    _, feats = hand_object_features(
      self.lib, self.probes, self.radii, variant,
      pos[:, self._wrist_idx], quat[:, self._wrist_idx], obj_pos, obj_quat, root_quat,
    )
    return torch.cat(
      [
        robot_rep,
        quat_apply_inverse(root_quat, obj_pos - root_pos),
        quat_to_rot6d(quat_mul(quat_inv(root_quat), obj_quat)),
        quat_apply_inverse(root_quat, obj_lin_w),
        quat_apply_inverse(root_quat, obj_ang_w),
        obj_pos[:, 2:3],
        feats,
        object_clearance(self.lib, variant, obj_pos, obj_quat)[:, None],
      ],
      dim=-1,
    )

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    flat = full_state.reshape(-1, self.full_state_dim)
    rf = self.robot_full_dim
    rep = self._object_rep(
      compute_gencoord_state(flat[:, :rf]), self.extract_aux_state(flat), flat[:, rf + 7 : rf + 10], flat[:, rf + 10 : rf + 13]
    )
    return rep.reshape(*full_state.shape[:-1], -1)

  def apply_prediction(self, aux_state, curr_rep_state, prediction, dt):
    rr, n = self.robot_rep_dim, self.num_joints
    robot_aux, robot_rep = super().apply_prediction(
      aux_state[..., :7], curr_rep_state[..., :rr], prediction[..., : 6 + n], dt
    )
    c = self.substep_coeff
    root_quat = aux_state[..., 3:7]
    obj = curr_rep_state[..., rr:]
    dv, dw = prediction[..., 6 + n : 9 + n], prediction[..., 9 + n : 12 + n]
    lin_next = quat_apply(root_quat, obj[..., _LIN] + dv)
    ang_next = quat_apply(root_quat, obj[..., _ANG] + dw)
    obj_pos = aux_state[..., 7:10] + quat_apply(root_quat, obj[..., _LIN] + c * dv) * dt
    obj_quat = quat_integrate(aux_state[..., 10:14], quat_apply(root_quat, obj[..., _ANG] + c * dw), dt)
    next_aux = torch.cat([robot_aux, obj_pos, obj_quat, aux_state[..., 14:15]], dim=-1)
    shape = next_aux.shape[:-1]
    next_rep = self._object_rep(
      robot_rep.reshape(-1, rr), next_aux.reshape(-1, next_aux.shape[-1]), lin_next.reshape(-1, 3), ang_next.reshape(-1, 3)
    )
    return next_aux, next_rep.reshape(*shape, -1)

  def _object_world_vel(self, aux, rep):
    obj = rep[..., self.robot_rep_dim :]
    return quat_apply(aux[..., 3:7], obj[..., _LIN]), quat_apply(aux[..., 3:7], obj[..., _ANG])

  def reconstruct_full_state(self, aux_state, rep_state):
    robot = super().reconstruct_full_state(aux_state[..., :7], rep_state[..., : self.robot_rep_dim])
    lin, ang = self._object_world_vel(aux_state, rep_state)
    return torch.cat([robot, aux_state[..., 7:14], lin, ang, aux_state[..., 14:15]], dim=-1)

  def loss_blocks(self, rep_state, aux_state):
    blocks = super().loss_blocks(rep_state[..., : self.robot_rep_dim], aux_state[..., :7])
    rot = matrix_from_quat(aux_state[..., 10:14]).flatten(-2, -1)
    return blocks | {"object_pos": aux_state[..., 7:10], "object_ori": rot * math.sqrt(0.5)}

  def joint_state(self, rep_state):
    return super().joint_state(rep_state[..., : self.robot_rep_dim])

  def world_frame_bodies(self, aux_state, rep_state):
    return super().world_frame_bodies(aux_state[..., :7], rep_state[..., : self.robot_rep_dim])

  # ------------------------------------------------------- goal / reward / done

  def compute_wm_goal(self, motion, motion_times, aux_state):
    goal = hoi_goal(
      motion, self.lib, motion_times, aux_state[..., :3], aux_state[..., 3:7], aux_state[..., 7:10], aux_state[..., 10:14]
    )
    return goal, goal

  def _gap(self, rep_state: torch.Tensor) -> torch.Tensor:
    feats = rep_state[..., self.robot_rep_dim :][..., _FEAT]
    return feats.reshape(-1, NUM_HANDS, NUM_PROBES, 4)[..., 0]

  def compute_wm_reward_from_reference(
    self, next_rep_state, next_aux_state, next_joint_state, action, last_action,
    target_body_pos_w, target_body_quat_w, target_body_lin_vel_w, target_body_ang_vel_w,
    motion_times=None,
  ):
    bodies = self.world_frame_bodies(next_aux_state, next_rep_state)
    reward = tracking_reward(
      next_aux_state, *bodies, next_joint_state, action, last_action,
      target_body_pos_w, target_body_quat_w, target_body_lin_vel_w, target_body_ang_vel_w,
      self._soft_joint_pos_limits, self.dt,
    )
    lib, t = self.lib, motion_times
    lin, ang = self._object_world_vel(next_aux_state, next_rep_state)
    hoi = terms.hoi_reward_terms(
      self.task,
      next_aux_state[..., 7:10], next_aux_state[..., 10:14], lin, ang,
      lib.object_pos_w[t], lib.object_quat_w[t], lib.object_lin_vel_w[t], lib.object_ang_vel_w[t],
      bodies[0][:, self._wrist_idx], target_body_pos_w[:, self._wrist_idx],
      self._gap(next_rep_state), lib.hand_contact[t],
    )
    return reward + self.dt * sum(self.reward_weights[k] * v for k, v in hoi.items())

  def extra_termination_errors(self, aux_state, ref_anchor_pos_w, ref_anchor_quat_w, motion_times):
    lib, t = self.lib, motion_times
    return terms.object_termination_errors(
      self.task,
      aux_state[..., :3], aux_state[..., 3:7], aux_state[..., 7:10], aux_state[..., 10:14],
      ref_anchor_pos_w, ref_anchor_quat_w, lib.object_pos_w[t], lib.object_quat_w[t],
      lib.hand_contact[t],
    )
