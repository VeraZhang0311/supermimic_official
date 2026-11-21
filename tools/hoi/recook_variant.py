"""Re-cook one object variant of a packed HOI library into a new library.

The new library hard-links every file of the old one except the variant's
object directory (hulls, SDF, verts.npy), ``variants.json`` and
``build_manifest.json``. The variant is cooked again (CoACD + outer polytopes,
SDF, inertia: ``build_hoi_dataset.cook_variant``) with ``--coacd key=value``
overrides of the builder's CoACD settings; its ``variants.json`` entry records
them under ``coacd`` and the manifest lists the re-cook under ``recooked``.

``--objects-from LIB`` takes the variant's cooked files and entry from another
library instead of cooking (CoACD is not deterministic across runs), so a
held-out eval library gets exactly the training library's hulls.

  python tools/hoi/recook_variant.py --library all_03 --out all_04 \
      --variant woodchair_hoi_retarget --coacd max_convex_hull=24 --coacd threshold=0.04
  python tools/hoi/recook_variant.py --library eval_01 --out eval_02 \
      --variant woodchair_hoi_retarget --objects-from all_04
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hoi_dataset as b  # noqa: E402

COOKED = ("hulls", "hull_vertices_max", "extent_m", "ipos", "iquat", "inertia", "sdf_shape", "coacd")


def parse_override(text: str) -> tuple[str, object]:
  key, _, value = text.partition("=")
  if key not in b.COACD:
    raise ValueError(f"unknown CoACD setting {key!r}; known: {sorted(b.COACD)}")
  kind = type(b.COACD[key])
  if kind is bool:
    return key, value.lower() in ("1", "true", "yes")
  return key, kind(value)


def geometry_path(library: Path, variant: int, sha: str) -> str:
  """The variant's pipeline geometry, found through one of its clips."""
  import numpy as np

  manifest = json.loads((library / "build_manifest.json").read_text())
  clips = json.loads((library / "hoi" / "clips.json").read_text())
  clip = clips[int(np.flatnonzero(np.load(library / "hoi" / "clip_variant.npy") == variant)[0])]
  for line in Path(manifest["index"]).read_text().splitlines():
    e = json.loads(line)
    if (e["dataset"], e["clip_id"]) == (clip["dataset"], clip["source_clip"]):
      meta = json.loads((Path(e["output"]) / "metadata.json").read_text())
      if meta["object_geometry_sha256"] != sha:
        raise ValueError(f"{clip['source_clip']}: geometry {meta['object_geometry_sha256']} is not the variant's {sha}")
      return meta["object_geometry_path"]
  raise ValueError(f"{clip['source_clip']} is not in the index {manifest['index']}")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--library", required=True, help="packed HOI library to start from")
  ap.add_argument("--out", required=True, help="new library directory (must not exist)")
  ap.add_argument("--variant", required=True, help="variant name, e.g. woodchair_hoi_retarget")
  ap.add_argument("--coacd", action="append", default=[], help="CoACD override key=value (repeatable)")
  ap.add_argument("--objects-from", help="copy the variant's cooked files from this library instead of cooking")
  args = ap.parse_args()
  src, out = Path(args.library), Path(args.out)
  if out.exists():
    raise FileExistsError(out)
  variants_doc = json.loads((src / "hoi" / "variants.json").read_text())
  names = [v["name"] for v in variants_doc["variants"]]
  v = names.index(args.variant)
  var = variants_doc["variants"][v]
  obj_dir = Path("hoi") / "objects" / args.variant
  skip = {Path("hoi") / "variants.json", Path("build_manifest.json")}

  def ignore(directory: str, entries: list[str]) -> list[str]:
    rel = Path(directory).relative_to(src)
    return [e for e in entries if rel / e in skip or rel / e == obj_dir]

  shutil.copytree(src, out, copy_function=os.link, ignore=ignore)
  if args.objects_from:
    donor = Path(args.objects_from)
    donor_vars = {d["name"]: d for d in json.loads((donor / "hoi" / "variants.json").read_text())["variants"]}
    shutil.copytree(donor / obj_dir, out / obj_dir)
    info = {k: donor_vars[args.variant][k] for k in COOKED if k in donor_vars[args.variant]}
    record = {"variant": args.variant, "objects_from": str(donor.resolve())}
  else:
    cfg = dict(b.COACD)
    cfg.update(parse_override(o) for o in args.coacd)
    path = geometry_path(src, v, var["geometry_sha256"])
    print(f"[cook] {args.variant} with {cfg}", flush=True)
    info = b.cook_variant(args.variant, path, var["mass"], out / obj_dir, coacd_cfg=cfg)
    info["coacd"] = cfg
    record = {"variant": args.variant, "coacd": cfg, "geometry": path}
  var.update(info)
  (out / "hoi" / "variants.json").write_text(json.dumps(variants_doc, indent=2))
  manifest = json.loads((src / "build_manifest.json").read_text())
  manifest["recooked"] = [*manifest.get("recooked", []), {**record, "from_library": str(src.resolve())}]
  (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
  print(f"{args.variant}: {len(info['hulls'])} hulls, max {info['hull_vertices_max']} vertices, extent {info['extent_m']} -> {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
