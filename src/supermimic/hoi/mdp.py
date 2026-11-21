"""Simulator-side HOI terms: observations, rewards, terminations, startup event.

Rewards and terminations evaluate ``terms.py`` on simulator state; the world
model evaluates the same functions on predicted state.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.event_manager import requires_model_fields

from supermimic.hoi import terms
from supermimic.hoi.assets import PARK_DISTANCE
from supermimic.hoi.command import HoiMotionCommand, library_root, world_variants
from supermimic.hoi.geometry import hand_object_features, hand_probes
from supermimic.hoi.goals import hoi_goal
from supermimic.hoi.library import read_variants

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _command(env: ManagerBasedRlEnv, name: str = "motion") -> HoiMotionCommand:
  return cast(HoiMotionCommand, env.command_manager.get_term(name))


_COMPILED: dict[str, object] = {}


def _on_device(fn, device: str):
  """``fn`` compiled once for CUDA (eager, these terms launch hundreds of tiny
  kernels per env step); as is on CPU."""
  if not str(device).startswith("cuda"):
    return fn
  if fn.__name__ not in _COMPILED:
    _COMPILED[fn.__name__] = torch.compile(fn)
  return _COMPILED[fn.__name__]


# --------------------------------------------------------------------- startup


def quiet_linesearch_warnings(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Stop mjwarp printing a line per world whose Newton linesearch hits its cap.

  With elliptic friction ~1.5% of world-steps do; the solve still converges
  (``tools/hoi/solver_check.py``), but at 4096 worlds the printout grows the
  job log by GBs over a run. The solver iteration-cap warning stays on.
  """
  from mujoco_warp._src.types import OverflowType

  del env_ids
  opt = env.sim.wp_model.opt
  opt.warn_overflow = int(opt.warn_overflow) & ~int(OverflowType.LS_ITERATIONS)
  env.sim.create_graph()  # the flag is compiled into the solver kernels


@requires_model_fields("geom_pos", "body_mass", "body_inertia", "body_ipos", "body_iquat")
def activate_object_variants(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_name: str = "object",
  mass_scale_range: tuple[float, float] = (0.5, 1.5),
) -> None:
  """Unpark each world's variant hulls and give the body its (scaled) inertia."""
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  obj: Entity = env.scene[entity_name]
  variants = read_variants(library_root(env))
  wv = world_variants(env)[ids]
  geom_ids = obj.indexing.geom_ids
  geom_variant = torch.tensor(
    [int(re.search(r"obj_v(\d+)_h\d+", n).group(1)) for n in obj.geom_names], device=env.device
  )
  # MuJoCo re-centres each mesh on compile and stores that offset in geom_pos,
  # so start from the compiled values; the spec parks every variant but 0.
  compiled = env.sim.get_default_field("geom_pos")[geom_ids]
  active = compiled.clone()
  active[:, 2] -= PARK_DISTANCE * (geom_variant > 0).float()
  parked = (geom_variant[None, :] != wv[:, None]).float()
  pos = active[None].repeat(len(ids), 1, 1)
  pos[..., 2] += PARK_DISTANCE * parked
  model = env.sim.model
  model.geom_pos[ids[:, None], geom_ids[None, :]] = pos
  lo, hi = mass_scale_range
  scale = lo + (hi - lo) * torch.rand(len(ids), device=env.device)

  def table(attr: str) -> torch.Tensor:
    return torch.tensor([getattr(v, attr) for v in variants], dtype=torch.float32, device=env.device)

  body = obj.indexing.body_ids[0]
  model.body_mass[ids, body] = table("mass")[wv] * scale
  model.body_inertia[ids, body] = table("inertia")[wv] * scale[:, None]
  model.body_ipos[ids, body] = table("ipos")[wv]
  model.body_iquat[ids, body] = table("iquat")[wv]


# ---------------------------------------------------------------- observations


def object_state_w(env: ManagerBasedRlEnv, entity_name: str = "object") -> torch.Tensor:
  """``[pos - env origin, quat, lin_vel, ang_vel]`` of the object, ``(N, 13)``."""
  d = env.scene[entity_name].data
  return torch.cat(
    [d.root_link_pos_w - env.scene.env_origins, d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w],
    dim=-1,
  )


def object_variant(env: ManagerBasedRlEnv) -> torch.Tensor:
  return world_variants(env).float()[:, None]


def hoi_goal_obs(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  cmd = _command(env, command_name)
  origins = env.scene.env_origins
  d = cmd.object.data
  return _on_device(hoi_goal, env.device)(
    cmd.motion,
    cmd.lib,
    cmd.time_steps,
    cmd.robot_anchor_pos_w - origins,
    cmd.robot_anchor_quat_w,
    d.root_link_pos_w - origins,
    d.root_link_quat_w,
  )


# ------------------------------------------------------- rewards / terminations


def _step_values(
  task, lib, probes, radii, variant, wrist_pos, wrist_quat, obj_pos, obj_quat, obj_lin, obj_ang,
  anchor_pos, anchor_quat, ref_obj_pos, ref_obj_quat, ref_obj_lin, ref_obj_ang, ref_wrist_pos,
  ref_anchor_pos, ref_anchor_quat, contact,
) -> dict[str, torch.Tensor]:
  gap, _ = hand_object_features(lib, probes, radii, variant, wrist_pos, wrist_quat, obj_pos, obj_quat, anchor_quat)
  values = terms.hoi_reward_terms(
    task, obj_pos, obj_quat, obj_lin, obj_ang, ref_obj_pos, ref_obj_quat, ref_obj_lin, ref_obj_ang,
    wrist_pos, ref_wrist_pos, gap, contact,
  )
  values["termination"] = terms.object_termination_errors(
    task, anchor_pos, anchor_quat, obj_pos, obj_quat, ref_anchor_pos, ref_anchor_quat,
    ref_obj_pos, ref_obj_quat, contact,
  )
  return values


class _StepTerms:
  """HOI rewards and object errors for the current step, computed once."""

  def __init__(self, env: ManagerBasedRlEnv, task: terms.HoiTaskCfg, command_name: str):
    self.env, self.task, self.command_name = env, task, command_name
    cmd = _command(env, command_name)
    robot = cmd.robot
    self.wrist_ids = torch.tensor(robot.find_bodies(WRIST_BODIES, preserve_order=True)[0], device=env.device)
    self.ref_wrist_idx = torch.tensor([cmd.cfg.body_names.index(b) for b in WRIST_BODIES], device=env.device)
    self.probes, self.radii = hand_probes(env.device)
    self.step = -1

  def get(self) -> dict[str, torch.Tensor]:
    if self.step == self.env.common_step_counter:
      return self.values
    cmd = _command(self.env, self.command_name)
    robot, d = cmd.robot, cmd.object.data
    values = _on_device(_step_values, self.env.device)(
      self.task, cmd.lib, self.probes, self.radii, cmd.world_variant,
      robot.data.body_link_pos_w[:, self.wrist_ids], robot.data.body_link_quat_w[:, self.wrist_ids],
      d.root_link_pos_w, d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w,
      cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w,
      cmd.ref_object_pos_w, cmd.ref_object_quat_w, cmd.ref_object_lin_vel_w, cmd.ref_object_ang_vel_w,
      cmd.body_pos_w[:, self.ref_wrist_idx], cmd.anchor_pos_w, cmd.anchor_quat_w, cmd.ref_hand_contact,
    )
    self.values, self.step = values, self.env.common_step_counter
    return values


def _step_terms(env: ManagerBasedRlEnv, task: terms.HoiTaskCfg, command_name: str) -> _StepTerms:
  cache = getattr(env, "_hoi_step_terms", None)
  if cache is None:
    cache = _StepTerms(env, task, command_name)
    env._hoi_step_terms = cache
  return cache


def hoi_reward(
  env: ManagerBasedRlEnv, term: str, task: terms.HoiTaskCfg, command_name: str = "motion"
) -> torch.Tensor:
  """One unweighted HOI reward term; the reward manager applies the weight."""
  return _step_terms(env, task, command_name).get()[term]


def bad_object(env: ManagerBasedRlEnv, task: terms.HoiTaskCfg, command_name: str = "motion") -> torch.Tensor:
  return (_step_terms(env, task, command_name).get()["termination"] > 1.0).any(dim=-1)


def nonfinite_state(env: ManagerBasedRlEnv, entity_name: str = "object") -> torch.Tensor:
  """End episodes whose robot or object state diverged, before they reach the replay."""
  robot = env.scene["robot"].data
  obj = env.scene[entity_name].data
  parts = [robot.root_link_pos_w, robot.root_link_lin_vel_w, robot.joint_pos, robot.joint_vel,
           obj.root_link_pos_w, obj.root_link_lin_vel_w, obj.root_link_ang_vel_w]
  return ~torch.stack([torch.isfinite(p).all(dim=-1) for p in parts], dim=-1).all(dim=-1)
