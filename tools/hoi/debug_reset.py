"""Print object and pelvis poses right after a reset and after one step.

  uv run python tools/hoi/debug_reset.py <library dir>
"""

from __future__ import annotations

import sys

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse

import supermimic.tasks  # noqa: F401
from supermimic.hoi.env_cfg import configure_hoi_env, hoi_env_cfg
from supermimic.utils.helpers import extract_action_scale_offset


def show(raw, cmd, tag: str, n: int = 4) -> None:
  o = raw.scene.env_origins[:n]
  obj = cmd.object.data.root_link_pos_w[:n] - o
  ref = cmd.ref_object_pos_w[:n] - o
  pel = cmd.robot_anchor_pos_w[:n] - o
  rpel = cmd.anchor_pos_w[:n] - o
  rel = quat_apply_inverse(cmd.robot_anchor_quat_w[:n], obj - pel)
  rrel = quat_apply_inverse(cmd.anchor_quat_w[:n], ref - rpel)
  print(f"--- {tag}: frames {cmd.time_steps[:n].tolist()} variants {cmd.world_variant[:n].tolist()}")
  for i in range(n):
    print(f"  env{i} obj {obj[i].tolist()} ref {ref[i].tolist()}")
    print(f"       pelvis {pel[i].tolist()} ref pelvis {rpel[i].tolist()}")
    print(f"       obj-in-pelvis {rel[i].tolist()} ref {rrel[i].tolist()}")


def main() -> None:
  cfg = hoi_env_cfg()
  configure_hoi_env(cfg, sys.argv[1])
  cfg.commands["motion"].sampling_mode = "uniform"
  cfg.scene.num_envs = 8
  # Keep every episode alive so the post-step state can be inspected.
  for name in ("bad_object", "anchor_pos", "anchor_ori", "nonfinite_state", "motion_end"):
    cfg.terminations.pop(name, None)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
  raw = env.unwrapped
  cmd = raw.command_manager.get_term("motion")
  scale, offset = extract_action_scale_offset(env)
  env.reset()
  show(raw, cmd, "after reset")
  cache = raw._hoi_step_terms if hasattr(raw, "_hoi_step_terms") else None
  if cache is not None:
    cache.step = -1
    print("termination errors at reset", cache.get()["termination"][:4].tolist())
  for k in range(3):
    env.step((cmd.joint_pos - offset) / scale)
    show(raw, cmd, f"after step {k + 1}")
    print("object lin vel", cmd.object.data.root_link_lin_vel_w[:4].tolist())
    print("termination errors", raw._hoi_step_terms.values["termination"][:4].tolist())
  print("origins", raw.scene.env_origins[:4].tolist())
  gids = cmd.object.indexing.geom_ids
  print("object geom names", cmd.object.geom_names[:3], "...")
  print("compiled geom_pos", raw.sim.get_default_field("geom_pos")[gids][:4].tolist())
  print("world0 geom_pos", raw.sim.model.geom_pos[0, gids][:4].tolist())
  print("world0 geom_pos (z) all", raw.sim.model.geom_pos[0, gids][:, 2].tolist())
  print("object body ipos / mass", raw.sim.model.body_ipos[:4, cmd.object.indexing.body_ids[0]].tolist(),
        raw.sim.model.body_mass[:4, cmd.object.indexing.body_ids[0]].tolist())


if __name__ == "__main__":
  torch.set_printoptions(precision=3)
  main()
