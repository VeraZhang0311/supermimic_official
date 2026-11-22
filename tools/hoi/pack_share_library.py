"""Pack the clips of an HOI share package, full length, into one motion library.

The training library trims a few clips (it keeps only their first run free of
single-hand contact); the share package (``tools/hoi/export_clips.py``) has every
clip whole. This packs the package's clips, train and eval split, at full length
from the pipeline's per-clip outputs, in the package's manifest order, with the
object variants (hulls, nominal inertia) of an existing library. Clips that a
``--check-lib`` library also holds must match it exactly over its span.

  python tools/hoi/pack_share_library.py --manifest PKG/manifest.json \
      --variants-lib TRAIN_LIB --check-lib TRAIN_LIB --check-lib EVAL_LIB --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hoi_dataset import G1_JOINT_NAMES, OBJECT_TYPES, load_clip  # noqa: E402

FIELDS = [
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_pos",
  "joint_vel",
]
HOI_FIELDS = ["object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"]
LABELS = ["hand_contact", "foot_contact", "ref_intersection"]


def library_clips(root: Path) -> dict:
  """(dataset, source clip) -> (clips.json entry, first frame, end frame, variant name)."""
  clips = json.loads((root / "hoi" / "clips.json").read_text())
  ends = np.load(root / "motion_ends.npy")
  variant = np.load(root / "hoi" / "clip_variant.npy")
  names = [
    v["name"]
    for v in json.loads((root / "hoi" / "variants.json").read_text())["variants"]
  ]
  out = {}
  for c, lo, hi, v in zip(clips, np.r_[0, ends[:-1]], ends, variant, strict=True):
    key = (c["dataset"], c["source_clip"])
    if key in out:
      raise ValueError(f"{root}: {key} has several pieces")
    out[key] = (c, int(lo), int(hi), names[v])
  return out


def variant_geometry(root: Path) -> list:
  """A library's variants without their clip counts, plus a digest of their files."""
  out = []
  for v in json.loads((root / "hoi" / "variants.json").read_text())["variants"]:
    files = sorted((root / "hoi" / "objects" / v["name"]).iterdir())
    digest = hashlib.sha256(b"".join(f.name.encode() + f.read_bytes() for f in files))
    out.append(
      {k: x for k, x in v.items() if k not in ("clips", "frames")}
      | {"files": digest.hexdigest()}
    )
  return out


def full_clip(entry: dict) -> dict:
  (piece,) = load_clip({**entry, "trim": False})
  return piece


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--manifest", required=True, help="the share package's manifest.json")
  ap.add_argument("--index", help="pipeline included.jsonl (default: the manifest's)")
  ap.add_argument(
    "--variants-lib", required=True, help="library whose object variants are used"
  )
  ap.add_argument(
    "--check-lib",
    action="append",
    default=[],
    help="library to compare clips with; repeatable",
  )
  ap.add_argument(
    "--per-type",
    type=int,
    default=0,
    help="only the first N clips per object type (smoke tests)",
  )
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  out = Path(args.out)
  if out.exists() and any(out.iterdir()):
    raise FileExistsError(f"{out} is not empty")

  manifest = json.loads(Path(args.manifest).read_text())
  rows = manifest["clips"]
  if args.per_type:
    rows = [
      r
      for t in OBJECT_TYPES
      for r in [r for r in rows if r["object_type"] == t][: args.per_type]
    ]
  index_path = Path(args.index or manifest["index"])
  index = {
    (e["dataset"], e["clip_id"]): e
    for e in map(json.loads, index_path.read_text().splitlines())
    if e
  }
  vlib = Path(args.variants_lib)
  vjson = json.loads((vlib / "hoi" / "variants.json").read_text())
  vnames = [v["name"] for v in vjson["variants"]]
  if json.loads((vlib / "joint_names.json").read_text()) != list(G1_JOINT_NAMES):
    raise ValueError("joint order differs from the variants library's")
  checks = {}
  for lib in map(Path, args.check_lib):
    if variant_geometry(lib) != variant_geometry(vlib):
      raise ValueError(f"{lib} has other object variants than {vlib}")
    for key, val in library_clips(lib).items():
      checks.setdefault(key, (lib, *val))

  with ProcessPoolExecutor(args.workers) as pool:
    clips = list(
      pool.map(full_clip, [index[(r["dataset"], r["clip"])] for r in rows], chunksize=8)
    )

  # Same frames as the package, and the same data as the checked libraries over their spans.
  arrays: dict[Path, dict] = {}
  compared = trimmed = 0
  for r, c in zip(rows, clips, strict=True):
    if len(c["joint_pos"]) != r["frames"]:
      raise ValueError(
        f"{r['clip']}: {len(c['joint_pos'])} frames, the package has {r['frames']}"
      )
    if r["object_collision"] not in vnames:
      raise ValueError(f"{r['clip']}: variant {r['object_collision']} is not in {vlib}")
    if (r["dataset"], r["clip"]) not in checks:
      continue
    lib, entry, lo, hi, vname = checks[(r["dataset"], r["clip"])]
    if vname != r["object_collision"]:
      raise ValueError(
        f"{r['clip']}: variant {vname} in {lib}, {r['object_collision']} in the package"
      )
    a, b = entry["span"]
    if b - a != hi - lo:
      raise ValueError(
        f"{r['clip']}: span {entry['span']} vs {hi - lo} frames in {lib}"
      )
    trimmed += (b - a) != len(c["joint_pos"])
    if lib not in arrays:
      arrays[lib] = {f: np.load(lib / f"{f}.npy", mmap_mode="r") for f in FIELDS}
      arrays[lib] |= {
        f: np.load(lib / "hoi" / f"{f}.npy", mmap_mode="r") for f in HOI_FIELDS + LABELS
      }
    for f in FIELDS + HOI_FIELDS + LABELS:
      mine = c[f][a:b].astype(arrays[lib][f].dtype)
      if not np.array_equal(mine, arrays[lib][f][lo:hi]):
        err = np.abs(mine.astype(float) - arrays[lib][f][lo:hi].astype(float)).max()
        raise ValueError(f"{r['clip']}: {f} differs from {lib} by up to {err:g}")
    compared += 1

  (out / "hoi").mkdir(parents=True)
  shutil.copytree(vlib / "hoi" / "objects", out / "hoi" / "objects")
  for f in FIELDS:
    np.save(out / f"{f}.npy", np.concatenate([c[f] for c in clips]).astype(np.float32))
  for f in HOI_FIELDS:
    np.save(
      out / "hoi" / f"{f}.npy", np.concatenate([c[f] for c in clips]).astype(np.float32)
    )
  for f in LABELS:
    np.save(out / "hoi" / f"{f}.npy", np.concatenate([c[f] for c in clips]))
  np.save(out / "motion_ends.npy", np.cumsum([len(c["joint_pos"]) for c in clips]))
  np.save(
    out / "hoi" / "clip_variant.npy",
    np.asarray([vnames.index(r["object_collision"]) for r in rows], np.int64),
  )
  (out / "joint_names.json").write_text(json.dumps(list(G1_JOINT_NAMES)))
  names = [f"{r['dataset']}:{r['clip']}" for r in rows]
  (out / "motion_names.json").write_text(json.dumps(names))
  (out / "hoi" / "clips.json").write_text(
    json.dumps(
      [
        {
          "name": r["clip"],
          "dataset": r["dataset"],
          "source_clip": r["clip"],
          "categories": r["categories"],
          "trimmed": False,
          "span": [0, len(c["joint_pos"])],
          "source_frames": len(c["joint_pos"]),
          "frames": len(c["joint_pos"]),
          "ref_intersection_frac": round(float(c["ref_intersection"].mean()), 3),
          "foot_contact_frames": int((c["foot_contact"] == 1).any(1).sum()),
          "split": r["split"],
          "package_file": r["file"],
        }
        for r, c in zip(rows, clips, strict=True)
      ],
      indent=1,
    )
  )
  variants = [dict(v) for v in vjson["variants"]]
  for var in variants:
    mine = [
      c
      for r, c in zip(rows, clips, strict=True)
      if r["object_collision"] == var["name"]
    ]
    var["clips"], var["frames"] = len(mine), int(sum(len(c["joint_pos"]) for c in mine))
  (out / "hoi" / "variants.json").write_text(
    json.dumps({**vjson, "variants": variants}, indent=2)
  )
  (out / "build_manifest.json").write_text(
    json.dumps(
      {
        "tool": "tools/hoi/pack_share_library.py",
        "package_manifest": str(Path(args.manifest).resolve()),
        "index": str(index_path.resolve()),
        "variants_lib": str(vlib.resolve()),
        "check_libs": args.check_lib,
        "per_type": args.per_type,
        "clips": len(rows),
        "frames": int(sum(len(c["joint_pos"]) for c in clips)),
        "clip_list_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "splits": {
          s: sum(r["split"] == s for r in rows)
          for s in sorted({r["split"] for r in rows})
        },
      },
      indent=2,
    )
  )
  print(
    f"{len(rows)} clips, {sum(len(c['joint_pos']) for c in clips)} frames -> {out}; "
    f"{compared} matched a checked library ({trimmed} of them trimmed there, now full length)"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
