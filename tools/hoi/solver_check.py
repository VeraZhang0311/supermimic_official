"""How often the contact solver stops at its iteration cap, and how much that
changes the physics.

Replays the reference joint targets (as ``replay_reference.py``). Every step it
records which worlds hit the iteration cap in any substep, with step time and
outcomes. Every ``--probe`` steps it clones the simulator state and takes one
physics step twice, at the configured cap and with ``--ref-iterations`` (a
converged solve), and compares the resulting accelerations.

  uv run python tools/hoi/solver_check.py <library> [--iterations 10] [--cone pyramidal]
      [--impratio 1] [--robot-only]

``--robot-only`` runs the official GenCoord tracking task (no object, official
solver settings unless overridden) on the library's robot motions, as a baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import mujoco_warp as mjw
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mujoco_warp._src.types import OverflowType

import supermimic.tasks  # noqa: F401
from supermimic.hoi.env_cfg import configure_hoi_env, hoi_env_cfg
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from supermimic.utils.helpers import extract_action_scale_offset

ITER, LS = int(OverflowType.ITERATIONS), int(OverflowType.LS_ITERATIONS)


def clone(x):
  """Deep copy of a mjwarp struct: every warp array cloned, everything else shared."""
  if isinstance(x, wp.array):
    return wp.clone(x)
  if dataclasses.is_dataclass(x) and not isinstance(x, type):
    return dataclasses.replace(x, **{f.name: clone(getattr(x, f.name)) for f in dataclasses.fields(x) if f.init})
  return x


def q(x: torch.Tensor, ps=(0.5, 0.9, 0.99)) -> str:
  if x.numel() == 0:
    return "-"
  return "/".join(f"{v:.2g}" for v in torch.quantile(x.float(), torch.tensor(ps, device=x.device)).tolist())


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("library")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--probe", type=int, default=30, help="compare against a converged step every N steps")
  ap.add_argument("--iterations", type=int, default=None)
  ap.add_argument("--ls-iterations", type=int, default=None)
  ap.add_argument("--ref-iterations", type=int, default=200)
  ap.add_argument("--cone", choices=("pyramidal", "elliptic"), default=None)
  ap.add_argument("--impratio", type=float, default=None)
  ap.add_argument("--tolerance", type=float, default=None)
  ap.add_argument("--ccd-iterations", type=int, default=None)
  ap.add_argument("--njmax", type=int, default=None)
  ap.add_argument("--nconmax", type=int, default=None)
  ap.add_argument("--robot-only", action="store_true")
  args = ap.parse_args()

  cfg = gencoord_env_cfg() if args.robot_only else hoi_env_cfg()
  opts = {k: v for k, v in (("iterations", args.iterations), ("ls_iterations", args.ls_iterations),
                            ("cone", args.cone), ("impratio", args.impratio), ("tolerance", args.tolerance),
                            ("ccd_iterations", args.ccd_iterations)) if v is not None}
  cfg.sim.mujoco = dataclasses.replace(cfg.sim.mujoco, **opts)
  if args.njmax is not None:
    cfg.sim.njmax = args.njmax
  if args.nconmax is not None:
    cfg.sim.nconmax = args.nconmax
  if args.robot_only:
    cfg.commands["motion"].motion_file = args.library
  else:
    configure_hoi_env(cfg, args.library)
  cfg.commands["motion"].sampling_mode = "uniform"
  cfg.scene.num_envs = args.num_envs
  cfg.seed = 0
  mj = cfg.sim.mujoco
  label = "robot-only" if args.robot_only else "hoi"
  print(f"=== {label}: iterations {mj.iterations} ls_iterations {mj.ls_iterations} cone {mj.cone} "
        f"impratio {mj.impratio} tolerance {mj.tolerance} ccd_iterations {mj.ccd_iterations} "
        f"njmax {cfg.sim.njmax} nconmax {cfg.sim.nconmax} worlds {args.num_envs}", flush=True)

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
  raw = env.unwrapped
  sim = raw.sim
  m, d = sim.wp_model, sim.wp_data
  m.opt.warn_overflow = 0  # read from the flags below instead of device printf
  sim.create_graph()
  m_ref = dataclasses.replace(
    m, opt=dataclasses.replace(m.opt, iterations=args.ref_iterations, ls_iterations=max(m.opt.ls_iterations, 50))
  )
  scale, offset = extract_action_scale_offset(env)
  cmd = raw.command_manager.get_term("motion")
  hoi = not args.robot_only
  obj_v = raw.scene["object"].indexing.free_joint_v_adr.long() if hoi else None
  root_v = raw.scene["robot"].indexing.free_joint_v_adr.long()
  overflow = wp.to_torch(d.overflow)
  nefc = wp.to_torch(d.nefc)
  nacon = wp.to_torch(d.nacon)
  niter = wp.to_torch(d.solver_niter)
  dt = mj.timestep
  term_names = list(raw.termination_manager.active_terms)
  counts = {n: 0 for n in term_names}
  env.reset()

  hit_steps, ls_steps, nefc_all, held_all, variant_all = [], [], [], [], []
  ncon_all, niter_all, other_flags = [], [], 0
  probes = {"rel": [], "hit": [], "obj_dv": [], "root_dv": [], "ref_hit": 0, "n": 0}
  step_time = 0.0
  for t in range(1, args.steps + 1):
    if args.probe and t % args.probe == 0:
      da, db = clone(d), clone(d)
      for x in (da, db):
        wp.to_torch(x.overflow).zero_()
      mjw.step(m, da)
      mjw.step(m_ref, db)
      qa, qb = wp.to_torch(da.qacc), wp.to_torch(db.qacc)
      diff = qa - qb
      probes["rel"].append(diff.norm(dim=-1) / qb.norm(dim=-1).clamp(min=1.0))
      probes["hit"].append((wp.to_torch(da.overflow) & ITER) != 0)
      probes["ref_hit"] += int(((wp.to_torch(db.overflow) & ITER) != 0).sum())
      probes["n"] += args.num_envs
      probes["root_dv"].append(diff[:, root_v[:3]].norm(dim=-1) * dt)
      if hoi:
        probes["obj_dv"].append(diff[:, obj_v[:3]].norm(dim=-1) * dt)
    action = (cmd.joint_pos - offset) / scale
    held = (cmd.ref_hand_contact == 1).any(-1) if hoi else None
    overflow.zero_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    env.step(action)
    torch.cuda.synchronize()
    step_time += time.perf_counter() - t0
    flags = overflow.clone()
    for b in range(16):
      if bool(((flags >> b) & 1).any()):
        other_flags |= 1 << b
    ncon_all.append(int(nacon[0]))
    niter_all.append(niter.clone())
    hit_steps.append((flags & ITER) != 0)
    ls_steps.append((flags & LS) != 0)
    nefc_all.append(nefc.clone())
    for n in term_names:
      counts[n] += int(raw.termination_manager.get_term(n).sum())
    if hoi:
      held_all.append(held)
      variant_all.append(cmd.world_variant.clone())

  hit = torch.stack(hit_steps)
  ls = torch.stack(ls_steps)
  ne = torch.stack(nefc_all)
  print(f"step time {1000 * step_time / args.steps:.1f} ms per env step ({args.num_envs} worlds)")
  it = torch.stack(niter_all).float()
  print(f"nefc per world: max {int(ne.max())}, p99.9 {float(torch.quantile(ne.float().flatten()[::7], 0.999)):.0f}, mean {float(ne.float().mean()):.0f}; "
        f"contacts per world: mean {sum(ncon_all) / len(ncon_all) / args.num_envs:.1f}, max step {max(ncon_all) / args.num_envs:.1f}")
  print(f"solver iterations (last substep): mean {float(it.mean()):.1f}, batch max per step mean {float(it.max(1).values.mean()):.1f}")
  print(f"overflow bits seen (other than iteration caps): {hex(other_flags & ~(ITER | LS))}")
  print(f"world-steps hitting the iteration cap in any substep: {hit.float().mean():.1%}; "
        f"linesearch cap: {ls.float().mean():.1%}")
  print(f"nefc p50/p90/p99 when capped {q(ne[hit])}, otherwise {q(ne[~hit])}")
  if hoi:
    h = torch.stack(held_all)
    v = torch.stack(variant_all)
    print(f"capped rate when the reference holds the object {hit[h].float().mean():.1%}, otherwise {hit[~h].float().mean():.1%}")
    names = [x.name for x in cmd.lib.variants]
    print("capped rate per variant: " + ", ".join(f"{n} {hit[v == i].float().mean():.1%}" for i, n in enumerate(names)))
  print(f"terminations {counts}")
  if probes["n"]:
    rel = torch.cat(probes["rel"])
    ph = torch.cat(probes["hit"])
    root = torch.cat(probes["root_dv"])
    print(f"probe: {probes['n']} world-steps, capped {ph.float().mean():.1%}; reference solve still capped "
          f"{probes['ref_hit'] / probes['n']:.2%} at {args.ref_iterations} iterations")
    print(f"  |qacc - qacc_converged| / max(|qacc_converged|, 1) p50/p90/p99: capped {q(rel[ph])}, otherwise {q(rel[~ph])}")
    print(f"  root velocity error after one {dt * 1000:.0f} ms substep (m/s): capped {q(root[ph])}, otherwise {q(root[~ph])}")
    if hoi:
      obj = torch.cat(probes["obj_dv"])
      print(f"  object velocity error after one substep (m/s): capped {q(obj[ph])}, otherwise {q(obj[~ph])}")


if __name__ == "__main__":
  main()
