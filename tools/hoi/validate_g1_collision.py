"""Check cooked G1 HOI collision hulls in compiled MuJoCo models (offline tool).

1. coverage: mount the visual mesh and the hulls on one body the way mjlab's
   g1.xml mounts the visual mesh, compile, and check that every visual vertex
   lies inside the hulls in the world frame.
2. press: a box on a slide joint is pushed into the static left foot with a
   constant force, the way a table leg loads the foot. Penetration is how far
   the box's leading face ends up inside the visual foot surface in front of it
   (mj_ray on the visual mesh only). Scenarios: a 3 cm and a 1.6 cm leg pressed
   down on the top of the foot, and a 2 cm block pushed into its side. Compares
   the old 5-hull recipe with the new hulls under several contact settings and
   records the box's jitter once settled.

Merges the cook manifests into manifest.json and writes validation.json.

  python tools/hoi/validate_g1_collision.py --assets DIR --mesh-dir DIR \
      [--old-foot-hulls left-hulls.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

TOL_M = 5e-4
FOOT_Z = 0.1
DT = 0.005
SETTLE_STEPS = 400
CONTACTS = {
  "mujoco_default": {"option": "", "geom": ""},
  "solref_0.01_elliptic": {
    "option": 'cone="elliptic" impratio="10"',
    "geom": 'solref="0.01 1"',
  },
  "solref_0.01_solimp_0.99_elliptic": {
    "option": 'cone="elliptic" impratio="10"',
    "geom": 'solref="0.01 1" solimp="0.99 0.99 0.01"',
  },
}


@dataclass(frozen=True)
class Press:
  axis: int  # 1 = push along -y (side), 2 = push along -z (top)
  half: tuple[float, float, float]
  force: float
  positions: tuple[tuple[float, float], ...]  # the two coordinates off the axis


SCENARIOS = {
  "top_3cm_leg_20N": Press(
    2,
    (0.015, 0.015, 0.05),
    20.0,
    tuple((x, y) for x in (-0.04, -0.01, 0.02, 0.05, 0.08, 0.11) for y in (-0.015, 0, 0.015)),
  ),
  "top_1.6cm_leg_20N": Press(
    2,
    (0.008, 0.008, 0.05),
    20.0,
    tuple(
      (x, y)
      for x in (-0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13)
      for y in (-0.025, -0.01, 0.0, 0.01, 0.025)
    ),
  ),
  "side_2cm_block_30N": Press(
    1,
    (0.01, 0.01, 0.01),
    30.0,
    tuple((x, z) for x in (-0.05, -0.02, 0.01, 0.04, 0.07, 0.1, 0.13) for z in (-0.025, -0.012, 0.0, 0.012)),
  ),
}


def merge_manifests(assets: Path) -> dict:
  merged = None
  for path in sorted(assets.glob("manifest_*.json")):
    data = json.loads(path.read_text())
    if merged is None:
      merged = data
    else:
      merged["parts"].update(data["parts"])
  if merged is None:
    # Already merged, e.g. the copy checked into the repo.
    return json.loads((assets / "manifest.json").read_text())
  (assets / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n")
  return merged


def hull_assets(files: list[Path]) -> tuple[str, list[str]]:
  names = [f"hull_{i:02d}" for i in range(len(files))]
  xml = "\n".join(f'<mesh name="{n}" file="{f}"/>' for n, f in zip(names, files, strict=True))
  return xml, names


def world_vertices(model, data, geom_id: int) -> np.ndarray:
  mesh = model.geom_dataid[geom_id]
  adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
  verts = model.mesh_vert[adr : adr + num]
  return data.geom_xpos[geom_id] + verts @ data.geom_xmat[geom_id].reshape(3, 3).T


def coverage(assets: Path, mesh_dir: Path, entry: dict) -> dict:
  pos = " ".join(f"{v:.9g}" for v in entry["geom_pos"])
  mesh_xml, names = hull_assets([assets / h["file"] for h in entry["hulls"]])
  geoms = "\n".join(
    f'<geom name="{n}" type="mesh" mesh="{n}" pos="{pos}" group="3"/>' for n in names
  )
  xml = f"""
<mujoco>
  <compiler inertiafromgeom="false"/>
  <asset>
    <mesh name="visual" file="{mesh_dir / entry["source_mesh"]}"/>
    {mesh_xml}
  </asset>
  <worldbody>
    <body name="{entry["link"]}">
      <geom name="visual" type="mesh" mesh="visual" pos="{pos}" contype="0"
            conaffinity="0" group="2"/>
      {geoms}
    </body>
  </worldbody>
</mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  vis = world_vertices(model, data, model.geom("visual").id)
  violation = np.full(len(vis), np.inf)
  for n in names:
    eq = ConvexHull(world_vertices(model, data, model.geom(n).id)).equations
    violation = np.minimum(violation, (vis @ eq[:, :3].T + eq[:, 3]).max(1))
  outside = violation > TOL_M
  return {
    "visual_vertices": int(len(vis)),
    "hull_geoms": len(names),
    "uncovered_vertices": int(outside.sum()),
    "max_plane_violation_mm": round(1e3 * float(max(violation.max(), 0.0)), 3),
    "passed": bool(not outside.any()),
  }


def press_model(foot_stl: Path, hull_files: list[Path], contact: dict, press: Press, a, b):
  mesh_xml, names = hull_assets(hull_files)
  hull_geoms = "\n".join(
    f'<geom name="{n}" type="mesh" mesh="{n}" group="3" condim="3" {contact["geom"]}/>'
    for n in names
  )
  hx, hy, hz = press.half
  if press.axis == 2:
    pos, axis, travel = f"{a} {b} {FOOT_Z + 0.05 + hz}", "0 0 1", 0.2
  else:
    pos, axis, travel = f"{a} {0.06 + hy} {FOOT_Z + b}", "0 1 0", 0.2
  xml = f"""
<mujoco>
  <compiler inertiafromgeom="false"/>
  <option timestep="{DT}" gravity="0 0 0" integrator="implicitfast" iterations="10"
          ls_iterations="20" {contact["option"]}/>
  <asset>
    <mesh name="foot_visual" file="{foot_stl}"/>
    {mesh_xml}
  </asset>
  <worldbody>
    <body name="foot" pos="0 0 {FOOT_Z}">
      <geom name="foot_visual" type="mesh" mesh="foot_visual" contype="0"
            conaffinity="0" group="2"/>
      {hull_geoms}
    </body>
    <body name="pusher" pos="{pos}">
      <joint name="slide" type="slide" axis="{axis}" range="-{travel} 0" damping="20"/>
      <inertial pos="0 0 0" mass="2" diaginertia="0.002 0.002 0.002"/>
      <geom name="pusher" type="box" size="{hx} {hy} {hz}" condim="3" {contact["geom"]}/>
    </body>
  </worldbody>
</mujoco>"""
  return mujoco.MjModel.from_xml_string(xml)


def surface_ahead(model, data, press: Press, center: np.ndarray) -> np.ndarray:
  """Coordinate (along the push axis) of the visual surface in front of the box."""
  group = np.array([0, 0, 1, 0, 0, 0], np.uint8)
  geom_id = np.zeros(1, np.int32)
  direction = np.zeros(3)
  direction[press.axis] = -1.0
  others = [i for i in range(3) if i != press.axis]
  hits = []
  for du in np.linspace(-0.95, 0.95, 5) * press.half[others[0]]:
    for dv in np.linspace(-0.95, 0.95, 5) * press.half[others[1]]:
      pnt = center.copy()
      pnt[others[0]] += du
      pnt[others[1]] += dv
      pnt[press.axis] = 1.0
      dist = mujoco.mj_ray(model, data, pnt, direction, group, 1, -1, geom_id)
      if dist >= 0:
        hits.append(1.0 - dist)
  return np.asarray(hits)


def press_test(foot_stl: Path, hull_files: list[Path], contact: dict, press: Press) -> dict:
  rows = []
  for a, b in press.positions:
    model = press_model(foot_stl, hull_files, contact, press, a, b)
    data = mujoco.MjData(model)
    body = model.body("pusher").id
    faces, dims = [], set()
    for step in range(SETTLE_STEPS):
      data.qfrc_applied[0] = -press.force
      mujoco.mj_step(model, data)
      if step >= SETTLE_STEPS - 100:
        faces.append(data.xpos[body][press.axis] - press.half[press.axis])
        dims.update(int(c.dim) for c in data.contact[: data.ncon])
    hits = surface_ahead(model, data, press, data.xpos[body].copy())
    if len(hits) == 0:
      continue  # the box misses the foot
    rows.append(
      {
        "pos": [a, b],
        "penetration_mm": round(1e3 * (float(hits.max()) - float(np.mean(faces))), 2),
        "jitter_std_mm": round(1e3 * float(np.std(faces)), 3),
        "contact_dims": sorted(dims),
      }
    )
  pen = np.array([r["penetration_mm"] for r in rows])
  return {
    "presses": len(rows),
    "max_penetration_mm": round(float(pen.max()), 2),
    "frac_gt_2mm": round(float((pen > 2).mean()), 3),
    "frac_gt_5mm": round(float((pen > 5).mean()), 3),
    "max_jitter_std_mm": max(r["jitter_std_mm"] for r in rows),
    "contact_dims": sorted({d for r in rows for d in r["contact_dims"]}),
    "per_press": rows,
  }


def old_hull_files(old_json: Path, out: Path) -> list[Path]:
  import trimesh

  files = []
  for i, piece in enumerate(json.loads(old_json.read_text())["pieces"]):
    hull = trimesh.convex.convex_hull(np.asarray(piece["vertices"], float))
    lines = [f"v {a:.9g} {b:.9g} {c:.9g}" for a, b, c in hull.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in hull.faces]
    path = out / f"old_left_foot_hull_{i:02d}.obj"
    path.write_text("\n".join(lines) + "\n")
    files.append(path)
  return files


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--assets", type=Path, required=True)
  parser.add_argument("--mesh-dir", type=Path, required=True)
  parser.add_argument("--old-foot-hulls", type=Path)
  parser.add_argument("--scratch", type=Path, default=Path("validate_scratch"))
  args = parser.parse_args()
  manifest = merge_manifests(args.assets)
  report = {"coverage": {}, "press": {}}
  for key, entry in manifest["parts"].items():
    report["coverage"][key] = coverage(args.assets, args.mesh_dir, entry)
    print(json.dumps({f"coverage/{key}": report["coverage"][key]}), flush=True)

  foot = manifest["parts"]["left_foot"]
  foot_stl = args.mesh_dir / foot["source_mesh"]
  new_files = [args.assets / h["file"] for h in foot["hulls"]]
  variants = {f"new_hulls/{name}": (new_files, c) for name, c in CONTACTS.items()}
  if args.old_foot_hulls is not None:
    args.scratch.mkdir(parents=True, exist_ok=True)
    old = old_hull_files(args.old_foot_hulls, args.scratch)
    variants = {"old_5x16_hulls/mujoco_default": (old, CONTACTS["mujoco_default"])} | variants
  for scenario, press in SCENARIOS.items():
    for name, (files, contact) in variants.items():
      result = press_test(foot_stl, files, contact, press)
      report["press"][f"{scenario}/{name}"] = result
      summary = {k: v for k, v in result.items() if k != "per_press"}
      print(json.dumps({f"{scenario}/{name}": summary}), flush=True)

  (args.assets / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
  return 0 if all(c["passed"] for c in report["coverage"].values()) else 1


if __name__ == "__main__":
  sys.exit(main())
