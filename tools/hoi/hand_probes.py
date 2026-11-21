"""Place hand probe points for the HOI hand-object features (offline tool).

For each rubber hand the probes lie on the medial axis of its collision hulls:
at four stations spread along the hand's length, the point of the cross
section farthest from the hand's outer surface. Each probe's radius is that
distance, so at runtime ``sdf(probe) - radius`` is the gap between the hand
surface and the object (a chain of spheres standing in for the hand).

Writes ``probes.json`` next to the hull manifest. Needs numpy and scipy.

  python tools/hoi/hand_probes.py src/supermimic/hoi/assets/g1_collision
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

FRACTIONS = (0.2, 0.4, 0.6, 0.8)
STEP = 0.002


def load_hulls(asset_dir: Path, entry: dict) -> list[tuple[np.ndarray, np.ndarray]]:
  mount = np.asarray(entry["geom_pos"])
  hulls = []
  for h in entry["hulls"]:
    v, f = [], []
    for line in (asset_dir / h["file"]).read_text().splitlines():
      if line.startswith("v "):
        v.append([float(x) for x in line.split()[1:]])
      elif line.startswith("f "):
        f.append([int(x) - 1 for x in line.split()[1:]])
    hulls.append((np.asarray(v) + mount, np.asarray(f)))
  return hulls


def planes(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
  n = np.cross(b - a, c - a)
  keep = np.linalg.norm(n, axis=1) > 1e-12
  n, a = n[keep] / np.linalg.norm(n[keep], axis=1, keepdims=True), a[keep]
  n[np.einsum("ij,ij->i", n, a - v.mean(0)) < 0] *= -1
  return n, np.einsum("ij,ij->i", n, a)


def exterior_samples(hulls, per_m2: float = 4e6, seed: int = 0) -> np.ndarray:
  """Points on the union's outer surface (hull faces not buried in another hull)."""
  rng = np.random.default_rng(seed)
  hp = [planes(v, f) for v, f in hulls]
  out = []
  for k, (v, f) in enumerate(hulls):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = max(200, int(area.sum() * per_m2))
    tri = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random((2, n))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[tri] + (s * (1 - r2))[:, None] * b[tri] + (s * r2)[:, None] * c[tri]
    buried = np.zeros(n, bool)
    for j, (nn, off) in enumerate(hp):
      if j != k:
        buried |= (pts @ nn.T - off).max(1) < -1e-4
    out.append(pts[~buried])
  return np.concatenate(out)


def inside(hulls_planes, pts: np.ndarray) -> np.ndarray:
  return np.any([(pts @ n.T - off).max(1) < 0 for n, off in hulls_planes], axis=0)


def place(hulls) -> tuple[np.ndarray, np.ndarray]:
  hp = [planes(v, f) for v, f in hulls]
  tree = cKDTree(exterior_samples(hulls))
  verts = np.concatenate([v for v, _ in hulls])
  lo, hi = verts.min(0), verts.max(0)
  ys = np.arange(lo[1], hi[1], STEP)
  zs = np.arange(lo[2], hi[2], STEP)
  yy, zz = np.meshgrid(ys, zs, indexing="ij")

  def deepest(x: float) -> tuple[np.ndarray, float]:
    grid = np.stack([np.full(yy.size, x), yy.ravel(), zz.ravel()], -1)
    grid = grid[inside(hp, grid)]
    if len(grid) == 0:
      return np.zeros(3), 0.0
    d = tree.query(grid)[0]
    i = int(np.argmax(d))
    return grid[i], float(d[i])

  stations = lo[0] + np.asarray(FRACTIONS) * (hi[0] - lo[0])
  found = [deepest(x) for x in stations]
  return np.stack([p for p, _ in found]), np.asarray([r for _, r in found])


def main() -> int:
  asset_dir = Path(sys.argv[1])
  manifest = json.loads((asset_dir / "manifest.json").read_text())
  result = {"frame": "wrist_yaw_link", "fractions": FRACTIONS}
  for side in ("left", "right"):
    pts, radii = place(load_hulls(asset_dir, manifest["parts"][f"{side}_hand"]))
    result[side] = {"points": np.round(pts, 5).tolist(), "radii": np.round(radii, 5).tolist()}
    print(side, np.round(pts, 4).tolist(), np.round(radii, 4).tolist())
  (asset_dir / "probes.json").write_text(json.dumps(result, indent=2) + "\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
