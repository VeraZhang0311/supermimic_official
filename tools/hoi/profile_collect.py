"""Where an env step's time goes: physics substeps vs the managers, plus the
top GPU kernels, at training scale.

Steps the env with the reference joint targets, then profiles ``--steps`` env
steps with CUDA graphs off so every kernel shows up.

  uv run python tools/hoi/profile_collect.py <library> [--num-envs 4096] [--robot-only]
"""

from __future__ import annotations

import argparse
import time

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from torch.profiler import ProfilerActivity, profile, record_function

import supermimic.tasks  # noqa: F401
from supermimic.hoi.env_cfg import configure_hoi_env, hoi_env_cfg
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from supermimic.utils.helpers import extract_action_scale_offset


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("library")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--warmup", type=int, default=30)
  ap.add_argument("--steps", type=int, default=10)
  ap.add_argument("--robot-only", action="store_true")
  args = ap.parse_args()

  cfg = gencoord_env_cfg() if args.robot_only else hoi_env_cfg()
  if args.robot_only:
    cfg.commands["motion"].motion_file = args.library
  else:
    configure_hoi_env(cfg, args.library)
  cfg.commands["motion"].sampling_mode = "uniform"
  cfg.scene.num_envs = args.num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
  raw = env.unwrapped
  sim = raw.sim
  scale, offset = extract_action_scale_offset(env)
  cmd = raw.command_manager.get_term("motion")
  env.reset()

  physics = [0.0]
  inner_step = sim.step

  def timed_step(*a, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with record_function("sim.step"):
      out = inner_step(*a, **kw)
    torch.cuda.synchronize()
    physics[0] += time.perf_counter() - t0
    return out

  sim.step = timed_step

  def run(n: int) -> float:
    physics[0] = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
      env.step((cmd.joint_pos - offset) / scale)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

  run(args.warmup)
  total = run(args.steps)
  label = "robot-only" if args.robot_only else "hoi"
  print(f"=== {label}, {args.num_envs} worlds, CUDA graphs on: {1000 * total / args.steps:.1f} ms per env step, "
        f"physics {1000 * physics[0] / args.steps:.1f} ms, managers and rest {1000 * (total - physics[0]) / args.steps:.1f} ms")

  sim.use_cuda_graph = False
  run(3)
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    total = run(args.steps)
  print(f"CUDA graphs off: {1000 * total / args.steps:.1f} ms per env step, physics {1000 * physics[0] / args.steps:.1f} ms")
  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))


if __name__ == "__main__":
  main()
