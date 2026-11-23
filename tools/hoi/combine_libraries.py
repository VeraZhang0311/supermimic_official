"""Combine packed HOI motion libraries, or drop clips from one, without re-packing.

Concatenates the clips of every ``--lib`` in order (their object variants must be the
same geometry: identical ``hoi/objects`` files and variant entries apart from the
per-library clip and frame counts), leaves out the clips named in ``--exclude``
files, and writes a library of the same layout (``tools/hoi/build_hoi_dataset.py``).
A ``--split`` per library is recorded in each clip's ``hoi/clips.json`` entry (e.g.
train / eval, for reports on a combined evaluation library).

  # one evaluation library of every training clip (trimmed ones as trained) and the eval clips
  python tools/hoi/combine_libraries.py --lib TRAIN_LIB --split train --lib EVAL_LIB --split eval --out DIR
  # a training library without some clips
  python tools/hoi/combine_libraries.py --lib TRAIN_LIB --exclude drop.txt --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PER_CLIP = {"motion_ends.npy", "hoi/clip_variant.npy"}
VARIANT_COUNTS = ("clips", "frames")


def per_frame_fields(root: Path) -> list[str]:
  """Every per-frame array of a library, as a path relative to its root."""
  files = [p.name for p in root.glob("*.npy")] + [f"hoi/{p.name}" for p in (root / "hoi").glob("*.npy")]
  return sorted(f for f in files if f not in PER_CLIP)


def object_digest(root: Path) -> dict[str, str]:
  objects = root / "hoi" / "objects"
  return {
    str(p.relative_to(objects)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(objects.rglob("*"))
    if p.is_file()
  }


def geometry(variants: dict) -> dict:
  """variants.json without the per-library counts and provenance (``objects_from``; the
  object files themselves are compared by digest)."""
  return {
    "object_types": variants["object_types"],
    "variants": [{k: v for k, v in x.items() if k not in VARIANT_COUNTS} for x in variants["variants"]],
  }


def read_names(path: Path) -> list[str]:
  """One motion name per line; ``#`` starts a comment at a line's start or after a space
  (a trimmed clip's piece is named ``<clip>#0``)."""
  names = []
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.split(" #", 1)[0].strip()
    if line and not line.startswith("#"):
      names.append(line)
  return names


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--lib", action="append", required=True, help="packed library, in output order")
  ap.add_argument("--split", action="append", default=[], help="label per --lib, kept in clips.json")
  ap.add_argument("--exclude", action="append", default=[], help="file of motion names to leave out")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  libs = [Path(p) for p in args.lib]
  if args.split and len(args.split) != len(libs):
    ap.error("give one --split per --lib, or none")
  out = Path(args.out)
  if out.exists():
    ap.error(f"{out} exists")

  first = libs[0]
  fields = per_frame_fields(first)
  variants = json.loads((first / "hoi" / "variants.json").read_text())
  digest = object_digest(first)
  joint_names = (first / "joint_names.json").read_text()
  for lib in libs[1:]:
    if per_frame_fields(lib) != fields:
      sys.exit(f"{lib} has other per-frame arrays than {first}")
    if geometry(json.loads((lib / "hoi" / "variants.json").read_text())) != geometry(variants):
      sys.exit(f"{lib} has other object variants than {first}")
    if object_digest(lib) != digest:
      sys.exit(f"{lib} has other object files than {first}")
    if (lib / "joint_names.json").read_text() != joint_names:
      sys.exit(f"{lib} has other joints than {first}")

  excluded = {name for path in args.exclude for name in read_names(Path(path))}
  names, clips, clip_variant, spans, sources = [], [], [], [], []
  for k, lib in enumerate(libs):
    lib_names = json.loads((lib / "motion_names.json").read_text())
    lib_clips = json.loads((lib / "hoi" / "clips.json").read_text())
    ends = np.load(lib / "motion_ends.npy")
    variant = np.load(lib / "hoi" / "clip_variant.npy")
    assert len(lib_names) == len(lib_clips) == len(ends) == len(variant), lib
    starts = np.r_[0, ends[:-1]]
    kept = 0
    for i, name in enumerate(lib_names):
      if name in excluded:
        continue
      entry = dict(lib_clips[i])
      if args.split:
        entry["split"] = args.split[k]
      names.append(name)
      clips.append(entry)
      clip_variant.append(int(variant[i]))
      spans.append((k, int(starts[i]), int(ends[i])))
      kept += 1
    sources.append({"lib": str(lib), "split": args.split[k] if args.split else None, "clips": len(lib_names), "kept": kept})
  unknown = excluded - {n for lib in libs for n in json.loads((lib / "motion_names.json").read_text())}
  if unknown:
    sys.exit(f"--exclude names no clip of the libraries: {sorted(unknown)[:5]} ({len(unknown)})")
  if len(set(names)) != len(names):
    sys.exit("the libraries share clip names")

  (out / "hoi").mkdir(parents=True)
  frames = [int(np.load(lib / "motion_ends.npy")[-1]) for lib in libs]
  for field in fields:
    arrays = [np.load(lib / field, mmap_mode="r") for lib in libs]
    if [len(a) for a in arrays] != frames:
      sys.exit(f"{field} is not per frame in every library")
    np.save(out / field, np.concatenate([arrays[k][s:e] for k, s, e in spans]))
  lengths = np.array([e - s for _, s, e in spans])
  np.save(out / "motion_ends.npy", np.cumsum(lengths).astype(np.load(first / "motion_ends.npy").dtype))
  np.save(out / "hoi" / "clip_variant.npy", np.array(clip_variant, dtype=np.load(first / "hoi" / "clip_variant.npy").dtype))
  (out / "motion_names.json").write_text(json.dumps(names, indent=1))
  (out / "joint_names.json").write_text(joint_names)
  (out / "hoi" / "clips.json").write_text(json.dumps(clips, indent=1))
  for v, entry in enumerate(variants["variants"]):
    mine = [i for i, cv in enumerate(clip_variant) if cv == v]
    entry["clips"], entry["frames"] = len(mine), int(lengths[mine].sum())
  (out / "hoi" / "variants.json").write_text(json.dumps(variants, indent=1))
  shutil.copytree(first / "hoi" / "objects", out / "hoi" / "objects")
  manifest = {
    "tool": "tools/hoi/combine_libraries.py",
    "sources": sources,
    "excluded": sorted(excluded),
    "clips": len(names),
    "frames": int(lengths.sum()),
    "clip_list_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
  }
  (out / "build_manifest.json").write_text(json.dumps(manifest, indent=1))
  print(f"{out}: {len(names)} clips, {int(lengths.sum())} frames from " + ", ".join(
    f"{s['lib']} ({s['kept']}/{s['clips']})" for s in sources))
  return 0


if __name__ == "__main__":
  sys.exit(main())
