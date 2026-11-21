"""HOI side of a packed motion library (``tools/hoi/build_hoi_dataset.py``).

The robot reference lives in the mjlab packed directory and loads through
``MotionLoader``; this module adds the per-frame object reference, the hand
contact labels, the object variants and their geometry (SDF grids and hull
vertices), all indexed by the same global frame as ``MotionLoader``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.hoi import OBJECT_TYPES


@dataclass(frozen=True)
class ObjectVariant:
  """One object geometry: a type at one size from one data source."""

  name: str
  type: str
  type_id: int
  scale: float
  mass: float
  ipos: tuple[float, float, float]
  iquat: tuple[float, float, float, float]
  inertia: tuple[float, float, float]
  hull_files: tuple[str, ...]
  frames: int


def read_variants(root: str | Path) -> list[ObjectVariant]:
  root = Path(root)
  meta = json.loads((root / "hoi" / "variants.json").read_text())
  if tuple(meta["object_types"]) != OBJECT_TYPES:
    raise ValueError(f"library object types {meta['object_types']} != {OBJECT_TYPES}")
  out = []
  for v in meta["variants"]:
    obj_dir = root / "hoi" / "objects" / v["name"]
    out.append(
      ObjectVariant(
        name=v["name"],
        type=v["type"],
        type_id=int(v["type_id"]),
        scale=float(v["scale"]),
        mass=float(v["mass"]),
        ipos=tuple(v["ipos"]),
        iquat=tuple(v["iquat"]),
        inertia=tuple(v["inertia"]),
        hull_files=tuple(str(obj_dir / f) for f in v["hulls"]),
        frames=int(v["frames"]),
      )
    )
  return out


def assign_world_variants(
  variants: list[ObjectVariant], num_worlds: int, min_worlds: int
) -> np.ndarray:
  """Fixed variant per world, proportional to each variant's data duration.

  Deterministic in its inputs so every caller (startup event, command, sim
  context) derives the same assignment.
  """
  n = len(variants)
  floor = min(min_worlds, num_worlds // n)
  frames = np.asarray([v.frames for v in variants], float)
  share = frames / frames.sum() * (num_worlds - floor * n)
  counts = floor + np.floor(share).astype(int)
  order = np.argsort(-(share - np.floor(share)))
  counts[order[: num_worlds - counts.sum()]] += 1
  return np.repeat(np.arange(n), counts)


class HoiLibrary:
  """Object reference and geometry, on ``device``, indexed by global frame."""

  def __init__(self, root: str | Path, device: str, bin_seconds: float, dt: float):
    root = Path(root)
    hoi = root / "hoi"
    self.root = root
    self.device = device

    def load(name: str, dtype=torch.float32) -> torch.Tensor:
      return torch.from_numpy(np.load(hoi / f"{name}.npy")).to(device=device, dtype=dtype)

    self.object_pos_w = load("object_pos_w")
    self.object_quat_w = load("object_quat_w")
    self.object_lin_vel_w = load("object_lin_vel_w")
    self.object_ang_vel_w = load("object_ang_vel_w")
    self.hand_contact = load("hand_contact", torch.long)  # (T, 2) in {-1, 0, 1}
    motion_ends = np.load(root / "motion_ends.npy")
    clip_variant = np.load(hoi / "clip_variant.npy")
    total = int(motion_ends[-1])
    self.foot_contact = (
      load("foot_contact", torch.long)
      if (hoi / "foot_contact.npy").exists()
      else torch.full((total, 2), -1, dtype=torch.long, device=device)
    )
    self.variants = read_variants(root)
    self.num_variants = len(self.variants)
    starts = np.r_[0, motion_ends[:-1]]
    lengths = motion_ends - starts
    self.frame_variant = torch.as_tensor(
      np.repeat(clip_variant, lengths), dtype=torch.long, device=device
    )
    self.variant_type = torch.tensor(
      [v.type_id for v in self.variants], dtype=torch.long, device=device
    )
    self.variant_scale = torch.tensor(
      [v.scale for v in self.variants], dtype=torch.float32, device=device
    )

    self._clip_starts, self._clip_lengths, self._clip_variant = starts, lengths, clip_variant
    self._per_bin = max(1, int(round(bin_seconds / dt)))
    self.restrict_starts(np.ones(total, bool))
    self._load_geometry(hoi)

  def restrict_starts(self, start_ok: np.ndarray) -> None:
    """Rebuild the reset bins over the frames in ``start_ok``.

    Each run of allowed frames in a clip is cut into bins of ``bin_seconds``. A
    frame outside every bin (disallowed, or the clip's last) maps to the clip's
    nearest earlier bin, or its first, for return bookkeeping. A clip with no
    allowed frame raises.
    """
    device, per_bin = self.device, self._per_bin
    bin_start, bin_end, bin_variant, clip_first_bin = [], [], [], []
    frame_bin = np.full(len(start_ok), -1, np.int64)
    for c, (s, n) in enumerate(zip(self._clip_starts, self._clip_lengths, strict=True)):
      ok = np.asarray(start_ok[s : s + n], bool).copy()
      ok[-1] = False  # the last frame is never a start
      edges = np.diff(np.r_[0, ok.astype(np.int8), 0])
      runs = zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True)
      first = len(bin_start)
      for a, b in runs:
        for lo in range(s + a, s + b, per_bin):
          frame_bin[lo : min(lo + per_bin, s + b)] = len(bin_start)
          bin_start.append(lo)
          bin_end.append(min(lo + per_bin, s + b))
          bin_variant.append(self._clip_variant[c])
      if len(bin_start) == first:
        raise ValueError(f"clip {c} has no allowed start frame")
      clip_first_bin.append(first)
      fb = frame_bin[s : s + n]
      fb[: np.argmax(fb >= 0)] = first
      frame_bin[s : s + n] = np.maximum.accumulate(fb)
    self.bin_count = len(bin_start)
    self.bin_start = torch.tensor(bin_start, dtype=torch.long, device=device)
    self.bin_end = torch.tensor(bin_end, dtype=torch.long, device=device)
    self.bin_variant = torch.tensor(bin_variant, dtype=torch.long, device=device)
    self.frame_bin = torch.from_numpy(frame_bin).to(device)
    self.clip_first_bin = torch.tensor(clip_first_bin, dtype=torch.long, device=device)
    self.variant_bin_mask = (
      self.bin_variant[None, :] == torch.arange(self.num_variants, device=device)[:, None]
    ).float()
    self.variant_start_mask = torch.zeros_like(self.variant_bin_mask)
    self.variant_start_mask[
      torch.as_tensor(self._clip_variant, device=device), self.clip_first_bin
    ] = 1.0

  def _load_geometry(self, hoi: Path) -> None:
    """Stack the variants' SDF grids (padded) and hull vertices for batched queries."""
    grids = [np.load(hoi / "objects" / v.name / "sdf.npz") for v in self.variants]
    dims = np.stack([g["phi"].shape for g in grids])
    dmax = dims.max(0)
    voxel = {float(g["voxel"]) for g in grids}
    if len(voxel) != 1:
      raise ValueError(f"variants use different SDF voxels {voxel}")
    self.sdf_voxel = voxel.pop()
    phi = np.full((self.num_variants, *dmax), 1.0, np.float32)
    normal = np.zeros((self.num_variants, *dmax, 3), np.float32)
    for i, g in enumerate(grids):
      d = g["phi"].shape
      phi[i, : d[0], : d[1], : d[2]] = g["phi"]
      normal[i, : d[0], : d[1], : d[2]] = g["normal"].astype(np.float32)
    self.sdf_phi = torch.from_numpy(phi).to(self.device).reshape(-1)
    self.sdf_normal = torch.from_numpy(normal).to(self.device).reshape(-1, 3)
    self.sdf_dims = torch.as_tensor(dims, dtype=torch.long, device=self.device)
    self.sdf_dmax = tuple(int(x) for x in dmax)
    self.sdf_origin = torch.as_tensor(
      np.stack([g["origin"] for g in grids]), dtype=torch.float32, device=self.device
    )
    verts = [np.load(hoi / "objects" / v.name / "verts.npy") for v in self.variants]
    m = max(len(v) for v in verts)
    padded = np.stack([np.concatenate([v, np.repeat(v[:1], m - len(v), 0)]) for v in verts])
    self.hull_verts = torch.from_numpy(padded).to(self.device, torch.float32)
