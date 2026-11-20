"""Cook HOI collision hulls for the G1 rubber hands and feet (offline tool).

Recipe, per part:
  1. CoACD with decimation off. Every hull is the convex hull of one piece of the
     mesh, so the hulls together still contain the whole mesh.
  2. An outer k-DOP per hull: keep only K support planes. Dropping planes can only
     grow a convex hull, so step 2 caps the vertex count without uncovering any of
     the original surface.

The previous recipe (5 hulls, <=16 vertices, decimate on) shrank the hulls inside
the visual surface by up to ~1.6 cm, which is how table legs sank into the feet.

Metrics, measured on the exterior surface of the source mesh:
  gap   = how far the source surface lies outside every hull (must be 0)
  bulge = how far the hull surface lies outside the source mesh

Needs numpy, scipy, trimesh and coacd; none of them is a supermimic dependency.

  python tools/hoi/cook_g1_collision.py --mesh-dir DIR --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import coacd
import numpy as np
import trimesh
from scipy.spatial import HalfspaceIntersection, cKDTree

SCHEMA = "supermimic_hoi_g1_collision_v1"
SIDES = ("left", "right")
KDOP_DIRECTIONS = 64
COMMON_COACD = {"decimate": False, "merge": True, "seed": 0}
PARTS = {
  "hand": {
    "mesh": "{side}_rubber_hand.STL",
    "link": "{side}_wrist_yaw_link",
    # mjlab g1.xml: the rubber-hand visual geom sits at this pos on wrist_yaw_link.
    "geom_pos": {"left": [0.0415, 0.003, 0.0], "right": [0.0415, -0.003, 0.0]},
    "coacd": {
      "threshold": 0.06,
      "max_convex_hull": 20,
      "preprocess_mode": "auto",
      "preprocess_resolution": 50,
    },
    "max_bulge_p95_mm": 3.0,
  },
  "foot": {
    "mesh": "{side}_ankle_roll_link.STL",
    "link": "{side}_ankle_roll_link",
    "geom_pos": {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]},
    # The foot STL is five overlapping shells; "on" remeshes them into one manifold.
    "coacd": {
      "threshold": 0.06,
      "max_convex_hull": 12,
      "preprocess_mode": "on",
      "preprocess_resolution": 80,
    },
    "max_bulge_p95_mm": 7.0,
  },
}
SOURCE_SHA256 = {
  "left_rubber_hand.STL": "cff2221a690fa69303f61fce68f2d155c1517b52efb6ca9262dd56e0bc6e70fe",
  "right_rubber_hand.STL": "99533b778bca6246144fa511bb9d4e555e075c641f2a0251e04372869cd99d67",
  "left_ankle_roll_link.STL": "c4092af943141d4d9f74232f3cfa345afc6565f46a077793b8ae0e68b39dc33f",
  "right_ankle_roll_link.STL": "4b66222ea56653e627711b56d0a8949b4920da5df091da0ceb343f54e884e3a5",
}
MAX_GAP_MM = 0.5


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def fibonacci_directions(k: int) -> np.ndarray:
  i = np.arange(k) + 0.5
  phi = np.arccos(1 - 2 * i / k)
  theta = np.pi * (1 + 5**0.5) * i
  return np.stack(
    [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1
  )


def outer_kdop(vertices: np.ndarray, k: int) -> np.ndarray:
  """Vertices of the k-DOP (plus the 6 axis planes) that contains ``vertices``."""
  dirs = np.concatenate([fibonacci_directions(k), np.eye(3), -np.eye(3)])
  offsets = (vertices @ dirs.T).max(0)
  halfspaces = np.concatenate([dirs, -offsets[:, None]], 1)
  corners = HalfspaceIntersection(halfspaces, vertices.mean(0)).intersections
  return np.asarray(trimesh.convex.convex_hull(corners).vertices)


def surface_samples(mesh: trimesh.Trimesh, n: int, seed: int):
  points, faces = trimesh.sample.sample_surface(mesh, n, seed=seed)
  return points, mesh.face_normals[faces]


def exterior_samples(mesh: trimesh.Trimesh, n: int = 600_000, seed: int = 0):
  """Dense samples (+ normals) of the outer surface of the union of the shells."""
  shells = mesh.split(only_watertight=False)
  pts, nrm, label = [], [], []
  for i, shell in enumerate(shells):
    p, q = surface_samples(shell, max(2000, int(n * shell.area / mesh.area)), seed + i)
    pts.append(p)
    nrm.append(q)
    label.append(np.full(len(p), i))
  pts, nrm, label = np.concatenate(pts), np.concatenate(nrm), np.concatenate(label)
  if len(shells) > 1:
    keep = np.ones(len(pts), bool)
    for j, shell in enumerate(shells):
      ref, ref_n = surface_samples(shell, max(20_000, int(n * shell.area / mesh.area)), 99 + j)
      tree = cKDTree(ref)
      others = np.where(label != j)[0]
      dist, idx = tree.query(pts[others])
      signed = np.einsum("ij,ij->i", pts[others] - ref[idx], ref_n[idx])
      keep[others[(signed < -2e-4) & (dist < 0.05)]] = False
    pts, nrm = pts[keep], nrm[keep]
  return pts, nrm


def halfspaces(hull: trimesh.Trimesh):
  normals = hull.face_normals
  return normals, np.einsum("ij,ij->i", normals, hull.triangles[:, 0])


def inside_any(points, spaces, tol, skip=None) -> np.ndarray:
  out = np.zeros(len(points), bool)
  for k, (a, b) in enumerate(spaces):
    if k != skip:
      out |= (points @ a.T - b).max(1) <= tol
  return out


def summarize(x: np.ndarray) -> dict:
  return {
    "max_mm": round(1e3 * float(x.max()), 2),
    "p99_mm": round(1e3 * float(np.percentile(x, 99)), 2),
    "p95_mm": round(1e3 * float(np.percentile(x, 95)), 2),
    "frac_gt_2mm": round(float((x > 0.002).mean()), 4),
  }


def measure(hulls: list[trimesh.Trimesh], ext_pts, ext_nrm, seed: int = 0) -> dict:
  rng = np.random.default_rng(seed)
  spaces = [halfspaces(h) for h in hulls]
  pick = rng.choice(len(ext_pts), size=min(60_000, len(ext_pts)), replace=False)
  probe = ext_pts[pick]
  covered = inside_any(probe, spaces, tol=1e-4)
  gap = np.zeros(len(probe))
  if (~covered).any():
    hull_pts = np.concatenate([surface_samples(h, 40_000, seed + 1)[0] for h in hulls])
    gap[~covered] = cKDTree(hull_pts).query(probe[~covered])[0]
  tree = cKDTree(ext_pts)
  total_area = sum(h.area for h in hulls)
  bulge = []
  for k, hull in enumerate(hulls):
    n = max(2000, int(30_000 * hull.area / total_area))
    pts, _ = surface_samples(hull, n, seed + 2 + k)
    pts = pts[~inside_any(pts, spaces, tol=-1e-4, skip=k)]
    dist, idx = tree.query(pts)
    signed = np.einsum("ij,ij->i", pts - ext_pts[idx], ext_nrm[idx])
    bulge.append(np.where(signed > 0, dist, 0.0))
  return {"gap": summarize(gap), "bulge": summarize(np.concatenate(bulge))}


def write_obj(path: Path, hull: trimesh.Trimesh) -> None:
  lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in hull.vertices]
  lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in hull.faces]
  path.write_text("\n".join(lines) + "\n")


def cook_part(part: str, side: str, mesh_dir: Path, out: Path) -> dict:
  cfg = PARTS[part]
  name = cfg["mesh"].format(side=side)
  src = mesh_dir / name
  digest = sha256(src)
  if digest != SOURCE_SHA256[name]:
    raise ValueError(f"{src}: sha256 {digest} is not the pinned source mesh")
  mesh = trimesh.load_mesh(src, process=True)
  recipe = {**cfg["coacd"], **COMMON_COACD}
  pieces = coacd.run_coacd(
    coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), **recipe
  )
  hulls = [
    trimesh.convex.convex_hull(outer_kdop(np.asarray(v, float), KDOP_DIRECTIONS))
    for v, _ in pieces
  ]
  ext_pts, ext_nrm = exterior_samples(mesh)
  metrics = measure(hulls, ext_pts, ext_nrm)
  files = []
  for i, hull in enumerate(hulls):
    path = out / f"{side}_{part}_hull_{i:02d}.obj"
    write_obj(path, hull)
    files.append({"file": path.name, "vertices": len(hull.vertices), "sha256": sha256(path)})
  passed = (
    metrics["gap"]["max_mm"] <= MAX_GAP_MM
    and metrics["bulge"]["p95_mm"] <= cfg["max_bulge_p95_mm"]
  )
  entry = {
    "source_mesh": name,
    "source_sha256": digest,
    "link": cfg["link"].format(side=side),
    "geom_pos": cfg["geom_pos"][side],
    "geom_quat": [1.0, 0.0, 0.0, 0.0],
    "coacd": recipe,
    "kdop_directions": KDOP_DIRECTIONS,
    "hulls": files,
    "metrics": metrics,
    "limits": {"max_gap_mm": MAX_GAP_MM, "max_bulge_p95_mm": cfg["max_bulge_p95_mm"]},
    "passed": passed,
  }
  print(json.dumps({f"{side}_{part}": {"hulls": len(files), **metrics, "passed": passed}}))
  return entry


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--mesh-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--parts", default="hand,foot")
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  coacd.set_log_level("warn")
  parts = {}
  for part in args.parts.split(","):
    for side in SIDES:
      parts[f"{side}_{part}"] = cook_part(part, side, args.mesh_dir, args.out)

  def version(pkg: str) -> str:
    try:
      return metadata.version(pkg)
    except metadata.PackageNotFoundError:
      return "unknown"

  manifest_path = args.out / f"manifest_{args.parts.replace(',', '_')}.json"
  manifest_path.write_text(
    json.dumps(
      {
        "schema": SCHEMA,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("coacd", "trimesh", "numpy", "scipy")},
        "parts": parts,
      },
      indent=2,
    )
    + "\n"
  )
  return 0 if all(p["passed"] for p in parts.values()) else 1


if __name__ == "__main__":
  sys.exit(main())
