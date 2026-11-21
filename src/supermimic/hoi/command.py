"""Motion command for HOI: the robot reference plus one object per world.

Every world hosts one fixed object variant (``world_variants``). Resets draw a
start frame from that variant's clips only, from per-clip bins of
``bin_seconds`` laid over the frames where the reference has no robot body
inside the object, and place the object at its reference pose (lifted out of
the floor if the reference sinks it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, sample_uniform

from supermimic.hoi import OBJECT_TYPES
from supermimic.hoi.assets import OBJECT, OBJECT_FLOOR
from supermimic.hoi.geometry import object_clearance, sdf_query
from supermimic.hoi.library import HoiLibrary, assign_world_variants, read_variants
from supermimic.hoi.stats import ObjectStats

FLOOR_CLEARANCE = 0.001
"""Gap left under the object at reset when its reference pose sinks into the floor."""

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def library_root(env: ManagerBasedRlEnv, command_name: str = "motion") -> str:
  root = env.cfg.commands[command_name].motion_file
  if not isinstance(root, str):
    raise ValueError("HOI needs one packed motion directory as motion_file")
  return root


def world_variants(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """The fixed object variant of every world, ``(num_envs,)``; computed once."""
  cached = getattr(env, "_hoi_world_variant", None)
  if cached is None:
    cfg = env.cfg.commands[command_name]
    variants = read_variants(library_root(env, command_name))
    assignment = assign_world_variants(variants, env.num_envs, cfg.min_worlds_per_variant)
    cached = torch.as_tensor(assignment, dtype=torch.long, device=env.device)
    env._hoi_world_variant = cached
  return cached


def robot_object_points(
  m: mujoco.MjModel, robot_body_ids: list[int], robot_geom_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Sample points (body frame), radii and entity-local bodies of every robot
  collision geom that can touch the object: capsule axes, sphere centres and
  hull vertices (subsampled)."""
  local_of = {g: i for i, g in enumerate(robot_body_ids)}
  pts, rad, body = [], [], []
  for g in robot_geom_ids:
    if not (int(m.geom_conaffinity[g]) & OBJECT or int(m.geom_contype[g]) & OBJECT_FLOOR):
      continue
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.geom_quat[g])
    t, size = int(m.geom_type[g]), m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
      local, r = np.stack([np.zeros(5), np.zeros(5), np.linspace(-size[1], size[1], 5)], 1), size[0]
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
      local, r = np.zeros((1, 3)), size[0]
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
      mesh = m.geom_dataid[g]
      verts = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
      local, r = verts[:: max(1, len(verts) // 12)], 0.0
    else:
      continue
    p = m.geom_pos[g] + local @ rot.reshape(3, 3).T
    pts.append(p)
    rad += [r] * len(p)
    body += [local_of[int(m.geom_bodyid[g])]] * len(p)
  return np.concatenate(pts), np.asarray(rad), np.asarray(body)


def reference_penetration(
  m: mujoco.MjModel,
  robot_body_ids: list[int],
  robot_joint_q_adr: list[int],
  robot_geom_ids: list[int],
  motion,
  lib: HoiLibrary,
  device: str,
  chunk: int = 4096,
) -> torch.Tensor:
  """How deep the robot's collision geometry sits inside the object in each
  reference frame (metres, <= 0 when clear), measured on the object SDF."""
  pts, rad, body = robot_object_points(m, robot_body_ids, robot_geom_ids)
  bodies = sorted(set(body.tolist()))
  fk = ForwardKinematics(
    m,
    torch.tensor(robot_body_ids),
    torch.tensor(robot_joint_q_adr),
    torch.tensor(bodies, dtype=torch.long),
    device=device,
  )
  slot = torch.tensor([bodies.index(b) for b in body], device=device)
  pts = torch.tensor(pts, dtype=torch.float32, device=device)
  rad = torch.tensor(rad, dtype=torch.float32, device=device)
  total = motion.time_step_total
  depth = torch.empty(total, device=device)
  for lo in range(0, total, chunk):
    f = torch.arange(lo, min(lo + chunk, total), device=device)
    z3, zj = torch.zeros(len(f), 3, device=device), torch.zeros_like(motion.joint_pos[f])
    pos, quat, _, _ = fk(motion.root_pos_w[f], motion.root_quat_w[f], motion.joint_pos[f], z3, z3, zj)
    k = len(slot)
    world = pos[:, slot] + quat_apply(quat[:, slot], pts.expand(len(f), k, 3))
    oq = lib.object_quat_w[f][:, None, :].expand(len(f), k, 4)
    local = quat_apply_inverse(oq, world - lib.object_pos_w[f][:, None, :])
    phi, _ = sdf_query(lib, lib.frame_variant[f], local)
    depth[f] = (rad - phi).max(dim=-1).values
  return depth


class HoiMotionCommand(MotionCommand):
  cfg: HoiMotionCommandCfg

  def __init__(self, cfg: HoiMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.lib = HoiLibrary(cfg.motion_file, self.device, cfg.bin_seconds, env.step_dt)
    if self.lib.frame_variant.numel() != self.motion.time_step_total:
      raise ValueError("HOI arrays and the motion library disagree on frame count")
    self.reference_penetration = self._reference_penetration()
    self._restrict_starts()
    self.object: Entity = env.scene[cfg.object_entity_name]
    self.world_variant = world_variants(env)
    self.world_type = self.lib.variant_type[self.world_variant]
    # Replace the global-timeline bins with per-clip bins.
    self.bin_count = self.lib.bin_count
    self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
    self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)
    self._external_probs = torch.full((self.bin_count,), 1.0 / self.bin_count, device=self.device)
    self.metrics["object_pos_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["object_ori_error"] = torch.zeros(self.num_envs, device=self.device)
    # Per-object statistics: one row per object type present in the library.
    present = sorted(set(self.lib.variant_type.tolist()))
    slot = torch.full((len(OBJECT_TYPES),), -1, dtype=torch.long, device=self.device)
    slot[present] = torch.arange(len(present), device=self.device)
    self._world_slot = slot[self.world_type]
    self.stats = ObjectStats([OBJECT_TYPES[t] for t in present], self.device)

  # -- reset frames ---------------------------------------------------------------

  def _reference_penetration(self) -> torch.Tensor:
    ix = self.robot.indexing
    return reference_penetration(
      self._env.sim.mj_model, ix.body_ids.tolist(), ix.joint_q_adr.tolist(),
      ix.geom_ids.tolist(), self.motion, self.lib, self.device,
    )

  def _restrict_starts(self) -> None:
    """Reset only where the reference keeps the robot out of the object (within
    ``start_penetration``); a clip with no such frame keeps its least-penetrated
    tenth."""
    depth = self.reference_penetration.cpu().numpy()
    ok = depth <= self.cfg.start_penetration
    ends = self.motion.motion_ends.cpu().numpy()
    relaxed = 0
    for s, e in zip(np.r_[0, ends[:-1]], ends, strict=True):
      if not ok[s : e - 1].any():
        ok[s:e] = depth[s:e] <= np.quantile(depth[s : e - 1], 0.1)
        relaxed += 1
    self.lib.restrict_starts(ok)
    print(
      f"[INFO] HOI reset frames: {ok.mean():.1%} of frames keep the robot within "
      f"{self.cfg.start_penetration * 100:.1f} cm of the object surface "
      f"(reference penetration p50/p90/max = {np.quantile(depth, 0.5):.3f}/"
      f"{np.quantile(depth, 0.9):.3f}/{depth.max():.3f} m); {relaxed} clips relaxed"
    )

  # -- references at the current frame (world frame, env origins added) -------

  @property
  def ref_object_pos_w(self) -> torch.Tensor:
    return self.lib.object_pos_w[self.time_steps] + self._env.scene.env_origins

  @property
  def ref_object_quat_w(self) -> torch.Tensor:
    return self.lib.object_quat_w[self.time_steps]

  @property
  def ref_object_lin_vel_w(self) -> torch.Tensor:
    return self.lib.object_lin_vel_w[self.time_steps]

  @property
  def ref_object_ang_vel_w(self) -> torch.Tensor:
    return self.lib.object_ang_vel_w[self.time_steps]

  @property
  def ref_hand_contact(self) -> torch.Tensor:
    return self.lib.hand_contact[self.time_steps]

  # -- sampling ------------------------------------------------------------------

  def _sample_bins(self, env_ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Draw start frames for each env from its own variant's bins."""
    n = len(env_ids)
    mask = self.lib.variant_start_mask if self.cfg.start_of_clip else self.lib.variant_bin_mask
    weights = probs[None, :] * mask
    weights = torch.where(weights.sum(-1, keepdim=True) > 0, weights, mask)
    draws = torch.multinomial(weights, n, replacement=True)  # (V, n)
    bins = draws[self.world_variant[env_ids], torch.arange(n, device=self.device)]
    lo = self.lib.bin_start[bins]
    if self.cfg.start_of_clip:
      frames = lo
    else:
      span = (self.lib.bin_end[bins] - lo).clamp(min=1)
      frames = lo + (sample_uniform(0.0, 1.0, (n,), device=self.device) * span).long()
    entropy = -(probs * (probs + 1e-12).log()).sum()
    self.metrics["sampling_entropy"][:] = entropy / math.log(max(self.bin_count, 2))
    pmax, imax = probs.max(dim=0)
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
    return frames

  def _uniform_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self._sample_bins(env_ids, torch.full((self.bin_count,), 1.0 / self.bin_count, device=self.device))

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("HOI uses sampling_mode 'external' (return) or 'uniform'")

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      raise ValueError("HOI starts clips with sampling_mode='uniform', start_of_clip=True")
    super()._resample_command(env_ids)
    frames = self.time_steps[env_ids]
    pos = self.lib.object_pos_w[frames]
    quat = self.lib.object_quat_w[frames]
    # Retargeted references sometimes sink the object into the floor; start it
    # resting on the floor instead of letting the contact solver eject it.
    clearance = object_clearance(self.lib, self.world_variant[env_ids], pos, quat)
    lift = (FLOOR_CLEARANCE - clearance).clamp(min=0.0)
    pos = pos + torch.stack([torch.zeros_like(lift), torch.zeros_like(lift), lift], dim=-1)
    state = torch.cat(
      [
        pos + self._env.scene.env_origins[env_ids],
        quat,
        self.lib.object_lin_vel_w[frames],
        self.lib.object_ang_vel_w[frames],
      ],
      dim=-1,
    )
    self.object.write_root_state_to_sim(state, env_ids=env_ids)

  # -- statistics ----------------------------------------------------------------

  def compute(self, dt, env_ids=None):
    super().compute(dt, env_ids)
    if env_ids is None:  # the once-per-env-step call, after that step's resets
      m = self.metrics
      self.stats.add_steps(
        self._world_slot, torch.stack([m["object_pos_error"], m["object_ori_error"], m["error_body_pos"]], dim=-1)
      )

  def reset(self, env_ids):
    """Count how each finished episode ended (before the base reset clears it)."""
    if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
      length = self._env.episode_length_buf[env_ids]
      ended = length > 0  # the first reset of a run finishes no episode
      if bool(ended.any()):
        ids = env_ids[ended]
        tm = self._env.termination_manager
        none = torch.zeros(len(ids), dtype=torch.bool, device=self.device)

        def term(name: str) -> torch.Tensor:
          return tm.get_term(name)[ids].bool() if name in tm.active_terms else none

        complete = term("motion_end") | term("time_out")
        bad = term("bad_object") & ~complete
        fall = (term("anchor_pos") | term("anchor_ori")) & ~complete & ~bad
        self.stats.add_episodes(self._world_slot[ids], length[ended], torch.stack([complete, bad, fall], dim=-1))
    return super().reset(env_ids)

  def log_stats(self) -> dict[str, float]:
    """``HOI/<object>/...`` since the last call; the runner logs it to W&B."""
    return self.stats.summary(self._env.step_dt)

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics["object_pos_error"] = torch.linalg.vector_norm(
      self.ref_object_pos_w - self.object.data.root_link_pos_w, dim=-1
    )
    q1, q2 = self.ref_object_quat_w, self.object.data.root_link_quat_w
    self.metrics["object_ori_error"] = 2 * torch.acos((q1 * q2).sum(-1).abs().clamp(max=1.0))


@dataclass(kw_only=True)
class HoiMotionCommandCfg(MotionCommandCfg):
  object_entity_name: str = "object"
  bin_seconds: float = 1.0
  min_worlds_per_variant: int = 16
  start_of_clip: bool = False
  """Start every reset at the first frame of a clip (evaluation)."""
  start_penetration: float = 0.01
  """Resets start only at frames where the reference sinks the robot's collision
  geometry at most this deep (m) into the object."""

  def build(self, env: ManagerBasedRlEnv) -> HoiMotionCommand:
    return HoiMotionCommand(self, env)
