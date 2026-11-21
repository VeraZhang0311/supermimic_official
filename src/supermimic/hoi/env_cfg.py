"""HOI environment config: the GenCoord G1 tracking task plus one object per world."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from supermimic.hoi import mdp
from supermimic.hoi.assets import get_g1_hoi_robot_cfg, get_object_cfg, get_object_floor_cfg
from supermimic.hoi.command import HoiMotionCommandCfg
from supermimic.hoi.library import read_variants
from supermimic.hoi.terms import HoiTaskCfg

HOI_REWARD_PREFIX = "hoi_"


def hoi_env_cfg(play: bool = False, task: HoiTaskCfg | None = None) -> ManagerBasedRlEnvCfg:
  # Imported here: supermimic.tasks registers this task, so a module-level import
  # is circular when this module is imported first.
  from supermimic.tasks.env_cfgs import gencoord_env_cfg

  task = task or HoiTaskCfg()
  cfg = gencoord_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_g1_hoi_robot_cfg()}  # "object" added by configure_hoi_env

  base = cfg.commands["motion"]
  cmd = HoiMotionCommandCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
  # Small reset noise: mid-carry resets must not push the hands through the object.
  cmd.pose_range = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.005, 0.005),
                    "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)}
  cmd.velocity_range = {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
  cmd.joint_position_range = (-0.03, 0.03)
  if play:
    cmd.pose_range, cmd.velocity_range, cmd.joint_position_range = {}, {}, (0.0, 0.0)
    cmd.sampling_mode, cmd.start_of_clip = "uniform", True
  cfg.commands["motion"] = cmd

  # Elliptic friction at impratio 10 (grip without slip) needs more Newton steps:
  # at the official 10, ~4% of substeps stop unconverged, mostly while holding the
  # object; at 20 none do, for ~2% more step time (tools/hoi/solver_check.py,
  # reference replay). Off-reference states in training (an untrained policy)
  # still hit the cap now and then: a few warnings per 10 steps across 4096 worlds.
  cfg.sim.mujoco = dataclasses.replace(cfg.sim.mujoco, cone="elliptic", impratio=10, iterations=20)
  # Measured at 4096 worlds: ~17 contacts per world (pooled across worlds, peak
  # step mean 22) and <= 465 constraint rows in any world (mean 53). Larger
  # buffers only add idle threads to the solver kernels.
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 800
  # SuperMimic drops the self-collision reward; its contact sensor only costs time.
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "self_collision")

  cfg.events["object_variant"] = EventTermCfg(
    mode="startup",
    func=mdp.activate_object_variants,
    params={"entity_name": "object", "mass_scale_range": (0.5, 1.5)},
  )
  cfg.events["quiet_linesearch_warnings"] = EventTermCfg(mode="startup", func=mdp.quiet_linesearch_warnings)

  full = cfg.observations["full_state"].terms
  full["object_state"] = ObservationTermCfg(func=mdp.object_state_w)
  full["object_variant"] = ObservationTermCfg(func=mdp.object_variant)
  cfg.observations["goal"] = ObservationGroupCfg(
    terms={"hoi_goal": ObservationTermCfg(func=mdp.hoi_goal_obs)},
    concatenate_terms=True,
    enable_corruption=False,
  )

  for term, weight in task.reward_weights().items():
    cfg.rewards[HOI_REWARD_PREFIX + term] = RewardTermCfg(
      func=mdp.hoi_reward, weight=weight, params={"term": term, "task": task}
    )
  cfg.terminations["bad_object"] = TerminationTermCfg(func=mdp.bad_object, params={"task": task})
  cfg.terminations["nonfinite_state"] = TerminationTermCfg(func=mdp.nonfinite_state)
  return cfg


def configure_hoi_env(cfg: ManagerBasedRlEnvCfg, motion_dir: str) -> None:
  """Point the task at a packed HOI library and add the matching object entity."""
  motion_dir = os.path.expanduser(motion_dir)
  cfg.commands["motion"].motion_file = motion_dir
  cfg.scene.entities["object"] = get_object_cfg(read_variants(motion_dir))
  cfg.scene.entities["object_floor"] = get_object_floor_cfg()


def hoi_task_cfg(cfg: ManagerBasedRlEnvCfg) -> HoiTaskCfg:
  return cfg.rewards[HOI_REWARD_PREFIX + "object_pos"].params["task"]


def is_hoi_env(cfg: ManagerBasedRlEnvCfg) -> bool:
  return isinstance(cfg.commands.get("motion"), HoiMotionCommandCfg)
