"""Per-clip data quality of a packed HOI library, measured on the simulator geometry.

For each clip: how deep the reference sinks the robot's collision geometry into
the object, how far the object sinks into the floor, whether the object floats
with no limb near it, and whether the robot or object jumps. Writes
``<library>/hoi/qa.json`` and prints a summary per object type.

  uv run python tools/hoi/qa_hoi_dataset.py <library dir> [<library dir> ...]

Runs on CPU: it compiles the HOI scene for the collision geometry but builds no
simulator.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics

from supermimic.hoi.command import reference_penetration
from supermimic.hoi.env_cfg import configure_hoi_env, hoi_env_cfg
from supermimic.hoi.geometry import object_clearance
from supermimic.hoi.library import HoiLibrary

DT = 0.02
LIMITS = {
  "penetration_max_m": 0.08,  # deepest robot geometry inside the object
  "penetration_p90_m": 0.04,
  "floor_sink_m": 0.03,  # object below the floor
  "floating_frac": 0.05,  # airborne, both hands known apart, no foot touching
  "object_speed_mps": 6.0,
  "root_speed_mps": 5.0,
}
AIRBORNE_M = 0.03


def load(root: str):
  """Reference motion, HOI tables and per-frame reference penetration on CPU."""
  cfg = hoi_env_cfg()
  configure_hoi_env(cfg, root)
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  ix = scene["robot"]._compute_indexing(model, "cpu")
  fk = ForwardKinematics(model, ix.body_ids, ix.joint_q_adr, torch.tensor([0]))
  motion = MotionLoader(root, fk, "cpu")
  lib = HoiLibrary(root, "cpu", cfg.commands["motion"].bin_seconds, DT)
  pen = reference_penetration(
    model, ix.body_ids.tolist(), ix.joint_q_adr.tolist(), ix.geom_ids.tolist(), motion, lib, "cpu"
  )
  return motion, lib, pen, cfg.commands["motion"].start_penetration


def clip_metrics(motion, lib, pen, start_penetration: float, chunk: int = 65536) -> list[dict]:
  total = motion.time_step_total
  clearance = torch.empty(total)
  for lo in range(0, total, chunk):
    f = torch.arange(lo, min(lo + chunk, total))
    clearance[f] = object_clearance(lib, lib.frame_variant[f], lib.object_pos_w[f], lib.object_quat_w[f])
  hands_apart = (lib.hand_contact == 0).all(-1)
  no_foot = (lib.foot_contact != 1).all(-1)
  floating = (clearance > AIRBORNE_M) & hands_apart & no_foot
  obj_speed = lib.object_lin_vel_w.norm(dim=-1)
  root_speed = motion.root_lin_vel_w.norm(dim=-1)
  ends = motion.motion_ends.tolist()
  rows = []
  for c, (s, e) in enumerate(zip([0, *ends[:-1]], ends, strict=True)):
    p = pen[s:e]
    rows.append({
      "name": motion.motion_names[c],
      "variant": lib.variants[int(lib.frame_variant[s])].name,
      "seconds": round((e - s) * DT, 2),
      "penetration_p50_m": round(float(p.median()), 4),
      "penetration_p90_m": round(float(torch.quantile(p, 0.9)), 4),
      "penetration_max_m": round(float(p.max()), 4),
      "start_ok_frac": round(float((p <= start_penetration).float().mean()), 3),
      "floor_sink_m": round(float((-clearance[s:e]).clamp(min=0).max()), 4),
      "floating_frac": round(float(floating[s:e].float().mean()), 3),
      "object_speed_mps": round(float(obj_speed[s:e].max()), 2),
      "root_speed_mps": round(float(root_speed[s:e].max()), 2),
      "hand_unknown_frac": round(float((lib.hand_contact[s:e] < 0).any(-1).float().mean()), 3),
    })
  for r in rows:
    r["flags"] = [k for k, lim in LIMITS.items() if r[k] > lim]
  return rows


def main() -> None:
  for root in sys.argv[1:]:
    rows = clip_metrics(*load(root))
    (Path(root) / "hoi" / "qa.json").write_text(json.dumps({"limits": LIMITS, "clips": rows}, indent=1))
    print(f"=== {root}: {len(rows)} clips")
    by_variant = collections.defaultdict(list)
    for r in rows:
      by_variant[r["variant"]].append(r)
    for v, rs in by_variant.items():
      flagged = [r for r in rs if r["flags"]]
      counts = collections.Counter(f for r in rs for f in r["flags"])
      pen = np.array([r["penetration_p90_m"] for r in rs])
      print(f"{v:26s} clips {len(rs):4d} flagged {len(flagged):4d} {dict(counts)} "
            f"pen_p90 median {np.median(pen):.3f} m; floating clips {sum(r['floating_frac'] > 0 for r in rs)}")


if __name__ == "__main__":
  main()
