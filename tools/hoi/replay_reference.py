"""Drive the HOI env with the reference joint targets and report physics health.

The actions are the reference joint positions (the PD targets that reproduce the
motion if the robot could follow them), so episodes should mostly run to the
end of their clip. Prints contact and constraint counts, HOI reward terms,
terminations and object tracking every ``--every`` steps.

  uv run python tools/hoi/replay_reference.py <library dir> [--num-envs 64] [--steps 200]
"""

from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

import supermimic.tasks  # noqa: F401
from supermimic.hoi.env_cfg import configure_hoi_env, hoi_env_cfg
from supermimic.utils.helpers import extract_action_scale_offset


def _wp_to_torch(x):
  import warp as wp

  return wp.to_torch(x)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("library")
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--steps", type=int, default=200)
  ap.add_argument("--every", type=int, default=20)
  ap.add_argument("--object-solref", type=float, default=None, help="time constant for object contacts")
  ap.add_argument("--iterations", type=int, default=None)
  args = ap.parse_args()

  if args.object_solref is not None:
    from supermimic.hoi import assets

    assets.OBJECT_SOLREF = (args.object_solref, 1.0)
  cfg = hoi_env_cfg()
  if args.object_solref is not None:
    coll = cfg.scene.entities["robot"].collisions[0]
    coll.solref = {k: (args.object_solref, 1.0) for k in coll.solref}
  if args.iterations is not None:
    import dataclasses

    cfg.sim.mujoco = dataclasses.replace(cfg.sim.mujoco, iterations=args.iterations, ls_iterations=2 * args.iterations)
  configure_hoi_env(cfg, args.library)
  cfg.commands["motion"].sampling_mode = "uniform"
  cfg.scene.num_envs = args.num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
  raw = env.unwrapped
  scale, offset = extract_action_scale_offset(env)
  cmd = raw.command_manager.get_term("motion")
  env.reset()
  term_names = list(raw.termination_manager.active_terms)
  counts = {n: 0 for n in term_names}
  peak = {"ncon": 0, "nefc": 0}
  bad = {"steps": [], "pos": 0, "ori": 0, "held": 0, "free": 0}
  for t in range(1, args.steps + 1):
    action = (cmd.joint_pos - offset) / scale
    held_before = (cmd.ref_hand_contact == 1).any(-1)
    length_before = raw.episode_length_buf.clone()
    _, reward, _, _ = env.step(action)
    for n in term_names:
      counts[n] += int(raw.termination_manager.get_term(n).sum())
    hit = raw.termination_manager.get_term("bad_object")
    if hit.any():
      err = raw._hoi_step_terms.values["termination"][hit]
      bad["steps"] += (length_before[hit] + 1).tolist()
      bad["pos"] += int((err[:, 0] > 1).sum())
      bad["ori"] += int((err[:, 1] > 1).sum())
      bad["held"] += int(held_before[hit].sum())
      bad["free"] += int((~held_before[hit]).sum())
    data = raw.sim.wp_data
    nefc = _wp_to_torch(data.nefc)
    peak["nefc"] = max(peak["nefc"], int(nefc.max()))
    nacon = int(_wp_to_torch(data.nacon)[0])
    peak["ncon"] = max(peak["ncon"], nacon // args.num_envs + 1)
    if t % args.every == 0:
      hoi = getattr(raw, "_hoi_step_terms", None)
      terms = {} if hoi is None else {k: round(float(v.float().mean()), 3) for k, v in hoi.values.items() if v.dim() == 1}
      print(
        f"step {t:4d} reward {float(reward.mean()):+.4f} finite={bool(torch.isfinite(reward).all())} "
        f"obj_err {float(cmd.metrics['object_pos_error'].mean()):.3f}m "
        f"nefc max {int(nefc.max())} mean {float(nefc.float().mean()):.0f} contacts/world~{nacon / args.num_envs:.1f}",
        flush=True,
      )
      print(f"   hoi terms {terms}", flush=True)
      print(f"   terminations so far {counts}", flush=True)
  print(f"peak nefc {peak['nefc']}  peak contacts/world (avg over worlds) {peak['ncon']}")
  steps = torch.tensor(bad["steps"], dtype=torch.float)
  if len(steps):
    q = torch.quantile(steps, torch.tensor([0.1, 0.5, 0.9])).tolist()
    print(f"bad_object: {len(steps)} hits, episode step at hit p10/p50/p90 = {q}, "
          f"pos>thr {bad['pos']}, ori>thr {bad['ori']}, reference held {bad['held']}, free {bad['free']}")


if __name__ == "__main__":
  main()
