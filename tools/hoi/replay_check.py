"""Re-simulate saved clips of ``sim_rollouts.py`` in CPU MuJoCo, one control step at a time.

For every step k of a clip: load the clip's scene, set ``qpos[k]``, ``qvel[k]`` and
``ctrl[k]``, run ``decimation`` physics steps (``physics.json``) and compare with
``qpos[k + 1]``. Reports joint, root and object position differences per object
variant, split by whether the robot touches the object in that step.

  python replay_check.py DIR [--per-variant 3] [--ls-iterations N] [--out DIR/scenes/replay_check.json]

Needs numpy and mujoco (the version in physics.json reads the scenes).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def stats(x: np.ndarray) -> dict:
  if not len(x):
    return {"steps": 0}
  return {
    "steps": len(x),
    "p50": round(float(np.median(x)), 4),
    "p90": round(float(np.percentile(x, 90)), 3),
    "p99": round(float(np.percentile(x, 99)), 2),
    "max": round(float(x.max()), 1),
  }


def check_clip(model: mujoco.MjModel, z, decimation: int) -> dict[str, np.ndarray]:
  d = mujoco.MjData(model)
  out = {"joint_mrad": [], "root_mm": [], "object_mm": [], "touch": []}
  for k in range(len(z["ctrl"])):
    mujoco.mj_resetData(model, d)
    d.qpos[:], d.qvel[:], d.ctrl[:] = z["qpos"][k], z["qvel"][k], z["ctrl"][k]
    for _ in range(decimation):
      mujoco.mj_step(model, d)
    q, nxt = d.qpos, z["qpos"][k + 1]
    out["joint_mrad"].append(1e3 * np.abs(q[7:36] - nxt[7:36]).max())
    out["root_mm"].append(1e3 * np.linalg.norm(q[0:3] - nxt[0:3]))
    out["object_mm"].append(1e3 * np.linalg.norm(q[36:39] - nxt[36:39]))
    out["touch"].append(z["contact_count"][k, -1] > 0)
  return {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument(
    "dir", help="sim_rollouts.py output (clips.csv, clips/, scenes/, physics.json)"
  )
  ap.add_argument(
    "--per-variant", type=int, default=3, help="clips per object variant (0: all)"
  )
  ap.add_argument(
    "--ls-iterations", type=int, help="override the scenes' line-search iterations"
  )
  ap.add_argument("--out", help="write the summary here (JSON)")
  args = ap.parse_args()
  report = check_dir(Path(args.dir), args.per_variant, args.ls_iterations)
  if args.out:
    Path(args.out).write_text(json.dumps(report, indent=1))


def check_dir(
  root: Path, per_variant: int = 3, ls_iterations: int | None = None
) -> dict:
  decimation = int(json.loads((root / "physics.json").read_text())["decimation"])
  rows = list(csv.DictReader(open(root / "clips.csv", encoding="utf-8")))
  report = {
    "decimation": decimation,
    "ls_iterations_override": ls_iterations,
    "variants": {},
  }
  for variant in dict.fromkeys(r["variant"] for r in rows):
    model = mujoco.MjModel.from_xml_path(str(root / "scenes" / variant / "scene.xml"))
    if ls_iterations:
      model.opt.ls_iterations = ls_iterations
    mine = [r for r in rows if r["variant"] == variant]
    if per_variant:
      mine = [
        mine[i]
        for i in np.linspace(0, len(mine) - 1, min(per_variant, len(mine)))
        .round()
        .astype(int)
      ]
    parts = [check_clip(model, np.load(root / r["file"]), decimation) for r in mine]
    s = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    touch, joint = s["touch"], s["joint_mrad"]
    report["variants"][variant] = {
      "clips": [r["clip"] for r in mine],
      "ls_iterations": int(model.opt.ls_iterations),
      "joint_max_err_mrad": stats(joint),
      "joint_max_err_mrad_robot_touches_object": stats(joint[touch]),
      "joint_max_err_mrad_no_robot_object_contact": stats(joint[~touch]),
      "root_pos_err_mm": stats(s["root_mm"]),
      "object_pos_err_mm": stats(s["object_mm"]),
      "steps_within_1_mrad": round(float((joint < 1.0).mean()), 3),
      "steps_over_10_mrad": round(float((joint > 10.0).mean()), 4),
    }
    v = report["variants"][variant]
    print(
      f"{variant:24s} {len(joint):5d} steps, joints p50 {np.median(joint):.3f} p90 {np.percentile(joint, 90):.2f} "
      f"p99 {np.percentile(joint, 99):.1f} max {joint.max():.0f} mrad, within 1 mrad {v['steps_within_1_mrad']:.0%}, "
      f"over 10 mrad {v['steps_over_10_mrad']:.1%}; object p50 {np.median(s['object_mm']):.2f} "
      f"p99 {np.percentile(s['object_mm'], 99):.1f} max {s['object_mm'].max():.1f} mm",
      flush=True,
    )
  return report


if __name__ == "__main__":
  main()
