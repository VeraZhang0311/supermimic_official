"""A tiny packed HOI library on disk: two box variants with analytic SDF grids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from supermimic.hoi import OBJECT_TYPES

BOX_HALF = {"largebox": (0.2, 0.2, 0.17), "smallbox": (0.16, 0.18, 0.1)}
VOXEL = 0.01
CLIP_LENGTHS = (120, 75, 60)
CLIP_VARIANT = (0, 1, 0)
UNSAFE = slice(30, 60)
"""Frames of clip 0 whose reference puts the robot inside the object."""


def box_sdf(points: np.ndarray, half: np.ndarray) -> np.ndarray:
  q = np.abs(points) - half
  outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
  inside = np.minimum(q.max(-1), 0.0)
  return outside + inside


def write_box_variant(root: Path, name: str, half: tuple[float, float, float]) -> dict:
  half = np.asarray(half)
  out = root / "hoi" / "objects" / name
  out.mkdir(parents=True)
  lo = -half - 0.12
  shape = np.ceil((2 * half + 0.24) / VOXEL).astype(int) + 1
  axes = [lo[i] + VOXEL * np.arange(shape[i]) for i in range(3)]
  grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
  phi = box_sdf(grid, half)
  g = np.stack(np.gradient(phi, VOXEL), -1)
  g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
  np.savez(out / "sdf.npz", phi=phi.astype(np.float32), normal=g.astype(np.float16),
           origin=lo.astype(np.float32), voxel=np.float32(VOXEL))
  corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
  np.save(out / "verts.npy", corners.astype(np.float32))
  return {"name": name, "type": name, "type_id": OBJECT_TYPES.index(name), "dataset": "test",
          "geometry_sha256": name, "mass": 1.0, "hulls": [], "ipos": [0, 0, 0],
          "iquat": [1, 0, 0, 0], "inertia": [0.01, 0.01, 0.01], "scale": 1.0}


def make_library(root: Path) -> Path:
  root.mkdir(parents=True, exist_ok=True)
  variants = [write_box_variant(root, n, h) for n, h in BOX_HALF.items()]
  total = sum(CLIP_LENGTHS)
  for v, var in enumerate(variants):
    var["clips"] = sum(1 for c in CLIP_VARIANT if c == v)
    var["frames"] = sum(n for n, c in zip(CLIP_LENGTHS, CLIP_VARIANT, strict=True) if c == v)
  hoi = root / "hoi"
  (hoi / "variants.json").write_text(json.dumps({"object_types": list(OBJECT_TYPES), "variants": variants}))
  rng = np.random.default_rng(0)
  quat = rng.normal(size=(total, 4))
  quat /= np.linalg.norm(quat, axis=1, keepdims=True)
  np.save(hoi / "object_pos_w.npy", rng.normal(size=(total, 3)).astype(np.float32))
  np.save(hoi / "object_quat_w.npy", quat.astype(np.float32))
  np.save(hoi / "object_lin_vel_w.npy", rng.normal(size=(total, 3)).astype(np.float32))
  np.save(hoi / "object_ang_vel_w.npy", rng.normal(size=(total, 3)).astype(np.float32))
  np.save(hoi / "hand_contact.npy", rng.integers(-1, 2, size=(total, 2)).astype(np.int8))
  np.save(hoi / "clip_variant.npy", np.asarray(CLIP_VARIANT))
  np.save(root / "motion_ends.npy", np.cumsum(CLIP_LENGTHS))
  return root
