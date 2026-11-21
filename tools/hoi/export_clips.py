"""Export HOI clips as one self-contained .npz per clip, for sharing outside SuperMimic.

Reads the clips of a packed HOI library (``tools/hoi/build_hoi_dataset.py``) back
from the data pipeline's per-clip outputs, so every clip comes out full length
(trims applied for training are ignored) on the pipeline's 50 Hz grid.

Output directory:
  clips/<dataset>__<clip>.npz   per clip, see README.md written next to it
  objects/<type>.obj            the OMOMO mesh of each object type (unscaled)
  robot/                        G1 29-DoF MJCF + URDF + meshes (if --robot-dir)
  licenses/                     notices of the sources
  README.md, manifest.json

  python tools/hoi/export_clips.py --library LIB --index included.jsonl \
      --objects-dir HOI_RETARGET/assets/objects/objects --robot-dir ULTRA/ultra/data/assets/g1 \
      --license LICENSE-OBJECTS --per-type 3 --out DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hoi_dataset import (  # noqa: E402
  FOOT_CHANNELS,
  G1_JOINT_NAMES,
  HAND_CHANNELS,
  OBJECT_TYPES,
  geometry_mesh,
  limb_contact,
  object_type,
  unit_wxyz,
)

FPS = 50.0
SCALE_CHECK_TOL_M = 0.01
"""How far the scaled exported mesh's bounds may differ from the pipeline geometry. ULTRA's
geometry is its convex collision decomposition, a few mm larger than the mesh; a wrong scale
(0.83 vs 1) or frame shows up as several cm."""


def pick(names: list[str], n: int) -> list[str]:
  """``n`` names spread evenly over the sorted list (all of them if ``n`` is 0)."""
  names = sorted(names)
  if not n or n >= len(names):
    return names
  return [names[i] for i in np.linspace(0, len(names) - 1, n).round().astype(int)]


def object_scale(meta: dict, arr) -> np.ndarray:
  """Scale from the type's OMOMO mesh to this clip's object (per axis)."""
  asset = meta["object_asset"]
  if "source_scale" in asset:  # HOI-Retarget: one factor for mesh and trajectory
    return np.full(3, float(asset["source_scale"]))
  return np.asarray(arr["metadata/object_asset/exact_scale"], float)  # ULTRA


def check_scale(mesh_bounds: np.ndarray, scale: np.ndarray, geometry_path: str, clip: str) -> None:
  """The scaled mesh must match the geometry the pipeline labelled contacts on."""
  ref = geometry_mesh(geometry_path).bounds
  err = np.abs(mesh_bounds * scale - ref).max()
  if err > SCALE_CHECK_TOL_M:
    raise ValueError(f"{clip}: scaled mesh bounds differ from the pipeline geometry by {err:.4f} m")


def export_clip(entry: dict, info: dict, mesh_bounds: dict, out: Path) -> dict:
  src = Path(entry["output"])
  meta = json.loads((src / "metadata.json").read_text())
  arr = np.load(src / "labels.npz")
  t = arr["timestamps_s"]
  if not np.allclose(np.diff(t), 1.0 / FPS, atol=1e-6):
    raise ValueError(f"{entry['clip_id']}: not on a {FPS:g} Hz grid")
  typ = object_type(entry["clip_id"])
  scale = object_scale(meta, arr)
  check_scale(mesh_bounds[typ], scale, meta["object_geometry_path"], entry["clip_id"])
  channels = list(meta["channels"])
  labels = np.asarray(arr["target_labels"], np.int8)
  intersect = np.asarray(arr["target_intersection"], bool)
  prov = meta["provenance"]
  name = f"{entry['dataset']}__{entry['clip_id'].replace('/', '__')}"
  np.savez_compressed(
    out / "clips" / f"{name}.npz",
    fps=np.float32(FPS),
    up_axis="z",
    quat_order="wxyz",
    root_pos=np.asarray(arr["root_pos"], np.float32),
    root_quat=unit_wxyz(np.asarray(arr["root_quat"], float)).astype(np.float32),
    joint_pos=np.asarray(arr["joint_pos"], np.float32),
    joint_names=np.asarray(G1_JOINT_NAMES),
    object_pos=np.asarray(arr["object_pos"], np.float32),
    object_quat=unit_wxyz(np.asarray(arr["object_quat"], float)).astype(np.float32),
    object_type=typ,
    object_mesh=f"objects/{typ}.obj",
    object_scale=scale.astype(np.float32),
    hand_contact=limb_contact(labels, intersect, channels, HAND_CHANNELS),
    foot_contact=limb_contact(labels, intersect, channels, FOOT_CHANNELS),
    contact_channels=np.asarray(channels),
    contact_labels=labels,
    contact_intersection=intersect,
    contact_distance_m=np.asarray(arr["target_distance_m"], np.float32),
    source_dataset=entry["dataset"],
    source_clip=entry["clip_id"],
    source_fps=np.float32(prov["source_fps"]),
    categories=np.asarray(info.get("categories", []), dtype=str),
  )
  return {"file": f"clips/{name}.npz", "dataset": entry["dataset"], "clip": entry["clip_id"],
          "object_type": typ, "object_scale": scale.round(4).tolist(), "frames": len(t),
          "source_fps": prov["source_fps"], "categories": info.get("categories", [])}


README = """# HOI clips: Unitree G1 (29 DoF) with OMOMO objects

One `.npz` per clip in `clips/`, full length. Load with `np.load(path)`; string fields are
0-d or 1-d numpy string arrays.

## Conventions
- Frame rate: **50 Hz** (`fps`). The pipeline resampled the sources onto this grid
  (HOI-Retarget 30 Hz, ULTRA 60 Hz; `source_fps`): linear for positions and joints,
  slerp for quaternions.
- World frame: **z-up**, metres, ground plane at z = 0.
- Quaternions: **wxyz**, unit, sign-continuous over time.
- `root_*` is the floating base (pelvis) pose in the world.
- `joint_pos` (T, 29) in radians, ordered as `joint_names` (Unitree's 29-DoF order).

## Robot
`robot/` holds the G1 29-DoF model from the ULTRA repository (`g1_29dof.xml`, `g1_29dof.urdf`,
meshes; rubber hands). The joint names and order match `joint_names`. ULTRA clips were generated
with this model. HOI-Retarget clips were retargeted to Unitree's G1 29-DoF in the HOI-Retarget
code (github.com/leggedrobotics/hoi-retarget), which has the same joints and order but its own
hand geometry (it annotates palm pads).

## Objects
`objects/<type>.obj` is the OMOMO mesh of each object type, unscaled, as InterMimic re-exported
it (ULTRA and HOI-Retarget ship byte-identical vertices). The clip's object is that mesh scaled
by `object_scale` (per axis), then placed by `object_pos` / `object_quat`:

    world = object_pos + R(object_quat) @ (object_scale * v)

So the pose is the mesh origin (the OMOMO meshes are centred on it), not the centre of mass.
`object_scale` is 0.83 for HOI-Retarget clips (its G1 setting) and 1.0 for ULTRA clips (the
`_100_100_100` suffix). In ULTRA clip names, the three numbers before that suffix are ULTRA's
per-axis trajectory augmentation (`--xyz`), not an object scale.

## Contacts
Geometric labels from distances between robot collision surfaces and the object surface
(on below 2 cm, off above 4 cm): proximity, **not** measured force contacts.
- `hand_contact`, `foot_contact` (T, 2) int8, [left, right]: 1 touching (or penetrating the
  object), 0 separated, -1 unknown.
- `contact_channels` (40,) names the raw channels (`<robot body>__object`, `object__ground`):
  `contact_labels` (T, 40) int8 in the same {-1, 0, 1}, `contact_intersection` (T, 40) bool
  (surfaces penetrate), `contact_distance_m` (T, 40) float32.

The references are kinematic: some frames have the robot partly inside the object
(`contact_intersection`).

## Sources and licences
- Chairs and tables: HOI-Retarget (leggedrobotics/hoi-retarget), CC BY-NC-SA 4.0, derived from
  OMOMO motion capture (MIT) with InterAct contact annotations (CC BY-NC-SA 4.0).
- Boxes and suitcase: ULTRA `OMOMO_retarget_aug_v4` (code Apache-2.0, InterMimic parts MIT).
- Object meshes: OMOMO geometry via InterMimic (MIT), see `licenses/`.
- Robot model: Unitree, BSD-3-Clause.
Non-commercial use; cite OMOMO, InterMimic, HOI-Retarget and ULTRA. `categories` carries our
own manual tags (chairs and tables only).
"""


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--library", required=True, help="packed HOI library (its clips are exported)")
  ap.add_argument("--index", required=True, help="the pipeline's included.jsonl")
  ap.add_argument("--objects-dir", required=True, help="dir with <type>/<type>.obj (OMOMO meshes)")
  ap.add_argument("--robot-dir", help="G1 model dir to copy into robot/")
  ap.add_argument("--license", action="append", default=[], help="file copied into licenses/")
  ap.add_argument("--per-type", type=int, default=0, help="clips per object type (0: all)")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import trimesh

  out = Path(args.out)
  if out.exists() and any(out.iterdir()):
    raise FileExistsError(f"{out} is not empty")
  (out / "clips").mkdir(parents=True)
  (out / "objects").mkdir()

  index = {(e["dataset"], e["clip_id"]): e for e in map(json.loads, Path(args.index).read_text().splitlines()) if e}
  info = {}
  for c in json.loads((Path(args.library) / "hoi" / "clips.json").read_text()):
    info.setdefault((c["dataset"], c["source_clip"]), c)  # trimmed pieces share a source clip
  by_type = defaultdict(list)
  for ds, clip in info:
    by_type[object_type(clip)].append(f"{ds}\t{clip}")
  chosen = [tuple(k.split("\t")) for t in OBJECT_TYPES for k in pick(by_type[t], args.per_type)]

  mesh_bounds = {}
  for t in sorted({object_type(c) for _, c in chosen}):
    src = Path(args.objects_dir) / t / f"{t}.obj"
    shutil.copyfile(src, out / "objects" / f"{t}.obj")
    mesh_bounds[t] = trimesh.load(src, force="mesh", process=False).bounds
  rows = []
  for ds, clip in chosen:
    rows.append(export_clip(index[(ds, clip)], info[(ds, clip)], mesh_bounds, out))
    print(f"{rows[-1]['file']}  frames={rows[-1]['frames']}  scale={rows[-1]['object_scale'][0]}", flush=True)

  if args.robot_dir:
    shutil.copytree(args.robot_dir, out / "robot")
  if args.license:
    (out / "licenses").mkdir()
    for p in args.license:
      shutil.copyfile(p, out / "licenses" / Path(p).name)
  (out / "README.md").write_text(README)
  (out / "manifest.json").write_text(json.dumps({
    "library": str(Path(args.library).resolve()), "index": str(Path(args.index).resolve()),
    "per_type": args.per_type, "clips": rows,
  }, indent=1))
  print(f"{len(rows)} clips -> {out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
