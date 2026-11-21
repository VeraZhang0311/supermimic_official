"""HOI goal window, shared by the simulator observation and the world model.

Per frame of the official 10-frame, stride-5 window:
  joint_pos (J), joint_vel (J), anchor rel pos (3), anchor rel rot6d (6),
  object rel pos (3), object rel rot6d (6), object lin/ang vel in the root frame
  (6), hand contact label (2) and validity (2).
Plus, once, the reference object at the current frame relative to the actual
object (pos 3, rot6d 6). All positions are env-local (no env origins).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul

from supermimic.representations import NUM_GOAL_FRAMES, assemble_anchor_window, goal_window_indices
from supermimic.utils.helpers import quat_to_rot6d

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.hoi.library import HoiLibrary


def hoi_goal_dim(num_joints: int) -> int:
  return NUM_GOAL_FRAMES * (2 * num_joints + 3 + 6 + 3 + 6 + 6 + 4) + 9


def hoi_goal(
  motion: MotionLoader,
  lib: HoiLibrary,
  motion_times: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  obj_pos: torch.Tensor,
  obj_quat: torch.Tensor,
) -> torch.Tensor:
  n = motion_times.shape[0]
  idx = goal_window_indices(motion, motion_times)  # (N, F)
  f = idx.shape[1]
  joints = torch.cat([motion.joint_pos[idx], motion.joint_vel[idx]], dim=-1)
  a_pos, a_ori = assemble_anchor_window(root_pos, root_quat, motion.root_pos_w[idx], motion.root_quat_w[idx])
  o_pos, o_ori = assemble_anchor_window(root_pos, root_quat, lib.object_pos_w[idx], lib.object_quat_w[idx])
  rq = root_quat[:, None, :].expand(n, f, 4)
  o_vel = torch.cat(
    [quat_apply_inverse(rq, lib.object_lin_vel_w[idx]), quat_apply_inverse(rq, lib.object_ang_vel_w[idx])],
    dim=-1,
  )
  contact = lib.hand_contact[idx]
  labels = torch.cat([(contact == 1).float(), (contact >= 0).float()], dim=-1)
  per_frame = torch.cat(
    [joints, a_pos.view(n, f, 3), a_ori.view(n, f, 6), o_pos.view(n, f, 3), o_ori.view(n, f, 6), o_vel, labels],
    dim=-1,
  )
  ref_pos, ref_quat = lib.object_pos_w[idx[:, 0]], lib.object_quat_w[idx[:, 0]]
  err_pos = quat_apply_inverse(obj_quat, ref_pos - obj_pos)
  err_ori = quat_to_rot6d(quat_mul(quat_inv(obj_quat), ref_quat))
  return torch.cat([per_frame.reshape(n, -1), err_pos, err_ori], dim=-1)
