"""HOI reward and termination math shared by the simulator and the world model.

Every function takes world-frame tensors, so ``mdp.py`` feeds them simulator
state and ``representation.py`` feeds them predicted state. Weights, scales and
thresholds live only in ``HoiTaskCfg``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

from supermimic.utils.helpers import rotation_error_magnitude


@dataclass(frozen=True)
class HoiTaskCfg:
  # Rewards: weight * exp(-err^2 / std^2), every term in [0, 1] before weighting.
  object_pos_weight: float = 1.0
  object_pos_std: float = 0.15
  object_ori_weight: float = 1.0
  object_ori_std: float = 0.4
  object_lin_vel_weight: float = 0.5
  object_lin_vel_std: float = 0.5
  object_ang_vel_weight: float = 0.5
  object_ang_vel_std: float = 1.5
  hand_object_weight: float = 1.0
  hand_object_std: float = 0.1
  contact_weight: float = 1.0
  contact_std: float = 0.03
  penetration_weight: float = -20.0
  """Per metre of hand-probe penetration, summed over probes."""
  # Terminations (object pose relative to the anchor body, as in the reference).
  object_pos_threshold_contact: float = 0.25
  object_pos_threshold_free: float = 0.5
  object_ori_threshold: float = 0.8

  def reward_weights(self) -> dict[str, float]:
    return {
      "object_pos": self.object_pos_weight,
      "object_ori": self.object_ori_weight,
      "object_lin_vel": self.object_lin_vel_weight,
      "object_ang_vel": self.object_ang_vel_weight,
      "hand_object": self.hand_object_weight,
      "contact": self.contact_weight,
      "penetration": self.penetration_weight,
    }


def _kernel(err_sq: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-err_sq / std**2)


def object_pos_reward(cfg: HoiTaskCfg, pos, ref_pos) -> torch.Tensor:
  return _kernel(((pos - ref_pos) ** 2).sum(-1), cfg.object_pos_std)


def object_ori_reward(cfg: HoiTaskCfg, quat, ref_quat) -> torch.Tensor:
  err = rotation_error_magnitude(matrix_from_quat(quat), matrix_from_quat(ref_quat))
  return _kernel(err**2, cfg.object_ori_std)


def object_lin_vel_reward(cfg: HoiTaskCfg, vel, ref_vel) -> torch.Tensor:
  return _kernel(((vel - ref_vel) ** 2).sum(-1), cfg.object_lin_vel_std)


def object_ang_vel_reward(cfg: HoiTaskCfg, vel, ref_vel) -> torch.Tensor:
  return _kernel(((vel - ref_vel) ** 2).sum(-1), cfg.object_ang_vel_std)


def hand_object_reward(
  cfg: HoiTaskCfg, hand_pos, obj_pos, obj_quat, ref_hand_pos, ref_obj_pos, ref_obj_quat
) -> torch.Tensor:
  """Hands placed on the object as in the reference, measured in the object frame.

  ``hand_pos``: ``(N, 2, 3)`` wrist_yaw_link positions.
  """
  q = obj_quat[:, None, :].expand(-1, hand_pos.shape[1], 4)
  rq = ref_obj_quat[:, None, :].expand(-1, hand_pos.shape[1], 4)
  local = quat_apply_inverse(q, hand_pos - obj_pos[:, None, :])
  ref_local = quat_apply_inverse(rq, ref_hand_pos - ref_obj_pos[:, None, :])
  return _kernel(((local - ref_local) ** 2).sum(-1).mean(-1), cfg.hand_object_std)


def contact_reward(cfg: HoiTaskCfg, gap: torch.Tensor, hand_contact: torch.Tensor) -> torch.Tensor:
  """Hands the reference has in contact get their nearest probe to the surface.

  ``gap``: ``(N, 2, P)`` hand-surface-to-object gaps; ``hand_contact``: ``(N, 2)``
  labels in {-1, 0, 1}. Zero when no hand is labelled in contact.
  """
  near = gap.clamp(min=0.0).min(dim=-1).values
  active = (hand_contact == 1).to(gap.dtype)
  per_hand = _kernel(near**2, cfg.contact_std) * active
  return per_hand.sum(-1) / active.sum(-1).clamp(min=1.0)


def penetration(gap: torch.Tensor) -> torch.Tensor:
  """Summed probe penetration in metres, ``(N,)``; weighted negatively."""
  return (-gap).clamp(min=0.0).sum(dim=(-2, -1))


def hoi_reward_terms(
  cfg: HoiTaskCfg,
  obj_pos,
  obj_quat,
  obj_lin_vel,
  obj_ang_vel,
  ref_obj_pos,
  ref_obj_quat,
  ref_obj_lin_vel,
  ref_obj_ang_vel,
  hand_pos,
  ref_hand_pos,
  gap,
  hand_contact,
) -> dict[str, torch.Tensor]:
  """Unweighted per-term rewards keyed like ``HoiTaskCfg.reward_weights``."""
  return {
    "object_pos": object_pos_reward(cfg, obj_pos, ref_obj_pos),
    "object_ori": object_ori_reward(cfg, obj_quat, ref_obj_quat),
    "object_lin_vel": object_lin_vel_reward(cfg, obj_lin_vel, ref_obj_lin_vel),
    "object_ang_vel": object_ang_vel_reward(cfg, obj_ang_vel, ref_obj_ang_vel),
    "hand_object": hand_object_reward(
      cfg, hand_pos, obj_pos, obj_quat, ref_hand_pos, ref_obj_pos, ref_obj_quat
    ),
    "contact": contact_reward(cfg, gap, hand_contact),
    "penetration": penetration(gap),
  }


def object_termination_errors(
  cfg: HoiTaskCfg,
  anchor_pos,
  anchor_quat,
  obj_pos,
  obj_quat,
  ref_anchor_pos,
  ref_anchor_quat,
  ref_obj_pos,
  ref_obj_quat,
  hand_contact,
) -> torch.Tensor:
  """Object errors divided by their thresholds, ``(N, 2)``; > 1 terminates.

  Position: the object in the anchor frame against the reference object in the
  reference anchor frame, so global drift alone does not end an episode. The
  threshold is tighter while the reference holds the object; the orientation
  term only applies then.
  """
  rel = quat_apply_inverse(anchor_quat, obj_pos - anchor_pos)
  ref_rel = quat_apply_inverse(ref_anchor_quat, ref_obj_pos - ref_anchor_pos)
  pos_err = torch.linalg.vector_norm(rel - ref_rel, dim=-1)
  held = (hand_contact == 1).any(dim=-1)
  pos_thr = torch.where(
    held,
    torch.full_like(pos_err, cfg.object_pos_threshold_contact),
    torch.full_like(pos_err, cfg.object_pos_threshold_free),
  )
  ori_err = rotation_error_magnitude(matrix_from_quat(obj_quat), matrix_from_quat(ref_obj_quat))
  ori = torch.where(held, ori_err / cfg.object_ori_threshold, torch.zeros_like(ori_err))
  return torch.stack([pos_err / pos_thr, ori], dim=-1)
