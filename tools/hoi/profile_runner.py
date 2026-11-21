"""Where the runner's collection step spends its time, at training scale.

Builds the env and runner as ``train.py`` does (no W&B), runs collection steps
with the initial policy, and times each part of ``SuperMimicRunner.step`` with a
device sync around it.

  uv run python tools/hoi/profile_runner.py <library> [--num-envs 4096] [--robot-only]
"""

from __future__ import annotations

import argparse
import collections
import time

import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.hoi.env_cfg import configure_hoi_env
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device

TIMES: dict[str, float] = collections.defaultdict(float)


def timed(name: str, fn):
  def wrapper(*a, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    torch.cuda.synchronize()
    TIMES[name] += time.perf_counter() - t0
    return out

  return wrapper


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("library")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--steps", type=int, default=30)
  ap.add_argument("--robot-only", action="store_true")
  args = ap.parse_args()

  task = "SuperMimic-GenCoord-G1" if args.robot_only else "SuperMimic-HOI-G1"
  env_cfg = load_env_cfg(task, play=False)
  cfg = SuperMimicRunnerCfg()
  if args.robot_only:
    env_cfg.commands["motion"].motion_file = args.library
  else:
    configure_hoi_env(env_cfg, args.library)
    cfg.representation = "hoi_gencoord"
  env_cfg.commands["motion"].sampling_mode = "external"
  device = setup_device(None)
  env = create_env(env_cfg, args.num_envs, device)
  runner = SuperMimicRunner(env=env, cfg=cfg, task_id=task, device=device)
  runner.env.reset()
  runner._seed_sim_history()

  r = runner
  rep = r._representation
  rep.represent = timed("represent", rep.represent)
  rep.extract_aux_state = timed("extract_aux_state", rep.extract_aux_state)
  r._sim_context.read = timed("sim_context.read", r._sim_context.read)
  r._build_sim_obs_td = timed("build_sim_obs_td (sensor + goals)", r._build_sim_obs_td)
  actor = r.policy_learner.actor
  actor.forward = timed("actor forward", actor.forward)
  r.env.step = timed("env.step (physics + managers)", r.env.step)
  r.replay_buffer.insert = timed("replay_buffer.insert", r.replay_buffer.insert)
  r.logger.process_env_step = timed("logger.process_env_step", r.logger.process_env_step)
  if r._sampler is not None:
    r._sampler.record = timed("return_sampler.record", r._sampler.record)
    r.motion_cmd.set_sampling_distribution = timed("set_sampling_distribution", r.motion_cmd.set_sampling_distribution)
  r._flush_normalization = timed("flush_normalization", r._flush_normalization)
  step = timed("runner.step total", r.step)

  for _ in range(args.warmup):
    step()
  TIMES.clear()
  for _ in range(args.steps):
    step()
  total = TIMES["runner.step total"]
  print(f"=== {task}, {args.num_envs} worlds: {1000 * total / args.steps:.1f} ms per collection step")
  for name, t in sorted(TIMES.items(), key=lambda kv: -kv[1]):
    print(f"  {name:38s} {1000 * t / args.steps:7.2f} ms  {t / total:6.1%}")


if __name__ == "__main__":
  main()
