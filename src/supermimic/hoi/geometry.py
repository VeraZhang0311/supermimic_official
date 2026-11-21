"""Differentiable hand-object geometry for reward, termination and WM features.

Everything here is a pure torch function of world-frame poses, so the simulator
and the world model's imagination evaluate exactly the same quantities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from supermimic.hoi.library import HoiLibrary

ASSET_DIR = Path(__file__).parent / "assets" / "g1_collision"

# Each hand is stood in for by a chain of spheres on the medial axis of its
# collision hulls (``tools/hoi/hand_probes.py`` -> ``probes.json``), in its
# wrist_yaw_link frame. ``sdf(probe) - radius`` is the hand-surface-to-object gap.
_PROBES = json.loads((ASSET_DIR / "probes.json").read_text())
NUM_HANDS = 2
NUM_PROBES = len(_PROBES["left"]["points"])
HAND_FEATURE_DIM = NUM_HANDS * NUM_PROBES * 4  # (gap, normal) per probe


def hand_probes(device: str) -> tuple[torch.Tensor, torch.Tensor]:
  """Probe points ``(2, P, 3)`` in wrist_yaw_link frames and their radii ``(2, P)``."""
  points = np.stack([_PROBES[s]["points"] for s in ("left", "right")])
  radii = np.stack([_PROBES[s]["radii"] for s in ("left", "right")])
  return (
    torch.tensor(points, dtype=torch.float32, device=device),
    torch.tensor(radii, dtype=torch.float32, device=device),
  )


def sdf_query(
  lib: HoiLibrary, variant: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Signed distance and outward normal of ``points`` (object frame), trilinear.

  ``variant``: ``(N,)``; ``points``: ``(N, K, 3)``. Points beyond the grid add
  their distance to the grid box. Differentiable in ``points``.
  """
  n, k, _ = points.shape
  # A diverged simulation state must not turn into an out-of-range gather.
  points = torch.nan_to_num(points, nan=0.0, posinf=1e3, neginf=-1e3)
  voxel = lib.sdf_voxel
  origin = lib.sdf_origin[variant][:, None, :]
  dims = lib.sdf_dims[variant][:, None, :].to(points.dtype)
  g = (points - origin) / voxel
  gc = torch.minimum(torch.clamp(g, min=0.0), dims - 1.001)
  outside = torch.linalg.vector_norm((g - gc) * voxel, dim=-1)
  i0 = torch.floor(gc).long()
  f = gc - i0.to(points.dtype)
  d0, d1, d2 = lib.sdf_dmax
  base = variant[:, None] * (d0 * d1 * d2)
  phi = torch.zeros(n, k, device=points.device, dtype=points.dtype)
  normal = torch.zeros(n, k, 3, device=points.device, dtype=points.dtype)
  for dx in (0, 1):
    wx = f[..., 0] if dx else 1 - f[..., 0]
    for dy in (0, 1):
      wy = f[..., 1] if dy else 1 - f[..., 1]
      for dz in (0, 1):
        wz = f[..., 2] if dz else 1 - f[..., 2]
        idx = base + ((i0[..., 0] + dx) * d1 + (i0[..., 1] + dy)) * d2 + (i0[..., 2] + dz)
        w = wx * wy * wz
        phi = phi + w * lib.sdf_phi[idx]
        normal = normal + w[..., None] * lib.sdf_normal[idx]
  normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp(min=1e-6)
  return phi + outside, normal


def hand_object_features(
  lib: HoiLibrary,
  probes: torch.Tensor,
  radii: torch.Tensor,
  variant: torch.Tensor,
  wrist_pos_w: torch.Tensor,
  wrist_quat_w: torch.Tensor,
  obj_pos_w: torch.Tensor,
  obj_quat_w: torch.Tensor,
  root_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Hand-surface gaps ``(N, 2, P)`` and the feature vector ``(N, 24)``.

  ``wrist_*``: ``(N, 2, 3|4)`` world poses of the two wrist_yaw_links. Features
  per probe: the gap and the object surface normal in the robot root frame.
  """
  n = wrist_pos_w.shape[0]
  q = wrist_quat_w[:, :, None, :].expand(n, NUM_HANDS, NUM_PROBES, 4)
  p_w = wrist_pos_w[:, :, None, :] + quat_apply(q, probes.expand(n, -1, -1, -1))
  p_w = p_w.reshape(n, NUM_HANDS * NUM_PROBES, 3)
  oq = obj_quat_w[:, None, :].expand(n, NUM_HANDS * NUM_PROBES, 4)
  local = quat_apply_inverse(oq, p_w - obj_pos_w[:, None, :])
  phi, normal_local = sdf_query(lib, variant, local)
  gap = (phi.reshape(n, NUM_HANDS, NUM_PROBES) - radii).reshape(n, -1)
  rq = root_quat_w[:, None, :].expand(n, NUM_HANDS * NUM_PROBES, 4)
  normal_b = quat_apply_inverse(rq, quat_apply(oq, normal_local))
  features = torch.cat([gap[..., None], normal_b], dim=-1).reshape(n, HAND_FEATURE_DIM)
  return gap.reshape(n, NUM_HANDS, NUM_PROBES), features


def object_clearance(
  lib: HoiLibrary, variant: torch.Tensor, obj_pos_w: torch.Tensor, obj_quat_w: torch.Tensor
) -> torch.Tensor:
  """Height of the object's lowest hull vertex above the ground, ``(N,)``."""
  verts = lib.hull_verts[variant]
  q = obj_quat_w[:, None, :].expand(-1, verts.shape[1], 4)
  z = quat_apply(q, verts)[..., 2] + obj_pos_w[:, None, 2]
  return z.min(dim=1).values
