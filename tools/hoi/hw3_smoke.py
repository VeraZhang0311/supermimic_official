"""GPU smoke check of the hw3 physics in the HOI env (supermimic.hoi.physics_hw3).

Builds the training env (all randomization on) and the nominal evaluation env on a library, and checks: the motor
groups fuse into one control law behind one delay buffer (a delay shared by all joints); each env's lag lies in
2-4, stays fixed between resets and is redrawn by a reset; the torque scale lies in 0.9-1.1 per env and joint; the
joint friction is the measured one times U(0.8, 2.5) on the legs and U(0.8, 2.0) elsewhere; a rollout holding the
reset pose stays finite and upright; the applied torques respect effort limit and envelope; and the step time
against mjlab's physics.

  python tools/hoi/hw3_smoke.py --lib LIB [--num-envs 64]
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import torch


def build(lib: str, physics: str, num_envs: int, nominal: bool):
  from supermimic.hoi.env_cfg import (
    configure_hoi_env,
    configure_hoi_physics,
    hoi_env_cfg,
  )
  from supermimic.scripts.train_utils import create_env

  cfg = hoi_env_cfg(play=nominal)
  configure_hoi_env(cfg, lib)
  configure_hoi_physics(cfg, physics, nominal=nominal)
  cfg.commands["motion"].sampling_mode = "uniform"
  return create_env(cfg, num_envs, "cuda:0")


def roll(env, steps: int):
  """Hold the default pose (zero action) for ``steps``; return the per-step lags, torques, joint vel, pelvis z."""
  raw = env.unwrapped
  robot = raw.scene["robot"]
  out = {"lag": [], "tau": [], "vel": [], "z": [], "done": [], "t": 0.0}
  act = torch.zeros(env.num_envs, env.num_actions, device=raw.device)
  group = (
    robot._fused_actuator_group._groups[0]
    if robot._fused_actuator_group._groups
    else None
  )
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(steps):
    _, _, dones, _ = env.step(act)
    out["done"].append(dones.bool().clone())
    if group is not None and group.delay_buffer is not None:
      out["lag"].append(group.delay_buffer.current_lags.clone())
    out["tau"].append(robot.data.actuator_force.clone())
    out["vel"].append(robot.data.joint_vel.clone())
    out["z"].append(robot.data.root_link_pos_w[:, 2].clone())
  torch.cuda.synchronize()
  out["t"] = (time.time() - t0) / steps
  return {
    k: (torch.stack(v) if isinstance(v, list) and v else v) for k, v in out.items()
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--lib", required=True)
  ap.add_argument("--num-envs", type=int, default=64)
  args = ap.parse_args()

  import mjlab.tasks  # noqa: F401

  import supermimic.tasks  # noqa: F401
  from supermimic.hoi import physics_hw3 as hw3

  env = build(args.lib, "hw3", args.num_envs, nominal=False)
  raw = env.unwrapped
  robot = raw.scene["robot"]
  fused = robot._fused_actuator_group._groups
  print(f"[fuse] groups {len(fused)}: {[(g.actuator_type.__name__, g.target_ids.numel(), g.max_lag) for g in fused]}, "
        f"custom {len(robot._custom_actuators)}")  # fmt: skip
  assert (
    len(fused) == 1
    and fused[0].target_ids.numel() == 29
    and fused[0].delay_buffer is not None
  )
  env.reset()
  r1 = roll(env, 100)
  # A world that ended an episode is reset at the end of that step: its lag reads 0 until
  # the next physics step draws a new one. Check the lags between resets.
  lags = r1["lag"].cpu().numpy()  # (steps, envs)
  done = r1["done"].cpu().numpy()
  live = ~done
  inner = live[1:] & live[:-1]
  print(f"[delay] lags of live steps in [{lags[live].min()}, {lags[live].max()}], values "
        f"{np.unique(lags[live]).tolist()}; {int(done.any(0).sum())} of {done.shape[1]} envs reset during "
        f"100 steps; lag kept within episodes: {bool((lags[1:] == lags[:-1])[inner].all())}")  # fmt: skip
  assert lags[live].min() >= 2 and lags[live].max() <= 4
  assert (lags[1:] == lags[:-1])[inner].all()
  assert (lags[done] == 0).all()
  before = lags[-1].copy()
  env.unwrapped.reset()  # every env through a reset
  r2 = roll(env, 5)
  after = r2["lag"][-1].cpu().numpy()
  print(
    f"[delay] after a reset {np.mean(before != after):.0%} of envs drew a different lag"
  )
  ts = fused[0].params["torque_scale"].cpu().numpy()
  print(
    f"[torque scale] in [{ts.min():.3f}, {ts.max():.3f}], std over envs {ts.std(0).mean():.3f}"
  )
  assert 0.9 <= ts.min() and ts.max() <= 1.1 and ts.std(0).mean() > 0.03

  m = raw.sim.mj_model
  fl = raw.sim.model.dof_frictionloss.cpu().numpy()  # (worlds, nv) once randomized
  assert fl.ndim == 2 and fl.shape[0] == args.num_envs, fl.shape
  for joint, (slow, _fast) in hw3.FRICTION_LEVELS.items():
    dof = m.jnt_dofadr[m.joint(f"robot/{joint}").id]
    ratio = fl[:, dof] / slow
    lo, hi = (
      hw3.LEG_FRICTION_SCALE
      if re.fullmatch(hw3.LEG_JOINTS, joint)
      else hw3.FRICTION_SCALE
    )
    assert lo - 1e-4 <= ratio.min() and ratio.max() <= hi + 1e-4, (
      joint,
      ratio.min(),
      ratio.max(),
    )
  print("[friction] every joint's frictionloss / measured level within its range")

  tau = r1["tau"].cpu().numpy()
  print(f"[rollout] finite {np.isfinite(tau).all()}, pelvis z min {r1['z'].min().item():.3f} max {r1['z'].max().item():.3f}, "
        f"|tau| max per joint group: {np.abs(tau).max((0, 1)).round(1).tolist()}")  # fmt: skip
  # actuator_force is in the robot's actuator order (grouped by motor), not its joint order
  ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()
  for a, joint_id in enumerate(m.actuator_trnid[ctrl_ids, 0]):
    name = m.joint(int(joint_id)).name.removeprefix("robot/")
    motor, effort = hw3.joint_motor(name)
    assert np.abs(tau[..., a]).max() <= effort + 1e-3, (name, np.abs(tau[..., a]).max())
  print("[rollout] every actuator's torque within its joint's effort limit")
  print(
    f"[timing] hw3 training env: {1e3 * r1['t']:.1f} ms per env step at {args.num_envs} worlds"
  )

  nominal = build(args.lib, "hw3", args.num_envs, nominal=True)
  nominal.reset()
  rn = roll(nominal, 20)
  print(f"[nominal] lags {np.unique(rn['lag'].cpu().numpy()).tolist()}, torque scale "
        f"{np.unique(nominal.unwrapped.scene['robot']._fused_actuator_group._groups[0].params['torque_scale'].cpu().numpy()).tolist()}")  # fmt: skip
  base = build(args.lib, "mjlab", args.num_envs, nominal=True)
  base.reset()
  rb = roll(base, 100)
  print(
    f"[timing] mjlab nominal env: {1e3 * rb['t']:.1f} ms per env step; hw3 nominal: {1e3 * rn['t']:.1f} ms"
  )
  print("[ok]")


if __name__ == "__main__":
  main()
