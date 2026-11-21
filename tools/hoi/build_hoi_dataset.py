"""Pack HOI clips from the offline data pipeline into a SuperMimic motion library.

Input: a clip list and the data pipeline's per-clip outputs (``labels.npz`` holds
50 Hz robot root pose, joints and object pose plus 40 geometric contact channels;
``metadata.json`` points at the object-local collision surface).

Output directory:
  mjlab packed motion  root_*_w.npy, joint_pos.npy, joint_vel.npy, motion_ends.npy,
                       joint_names.json, motion_names.json
  hoi/                 object_{pos,quat,lin_vel,ang_vel}_w.npy, hand_contact.npy and
                       foot_contact.npy (T, 2) int8 in {-1 unknown, 0, 1},
                       ref_intersection.npy (T,) (the pipeline found some robot body
                       touching the inside of the object; any depth), clip_variant.npy,
                       variants.json, clips.json
  hoi/objects/<v>/     hull_XX.obj, sdf.npz (phi + normals, 1 cm), verts.npy

An object *variant* is one geometry: an object type at one size from one source.
Every world in training hosts one variant; ``variants.json`` carries its type,
size relative to the type's canonical size, nominal mass and inertia.

Needs numpy, scipy, trimesh, coacd. Runs with the data pipeline's runtime.

  python tools/hoi/build_hoi_dataset.py --index included.jsonl --clips clips.txt --out DIR
  python tools/hoi/build_hoi_dataset.py --index included.jsonl --objects largebox \
      --datasets hoi_retarget --max-per-variant 30 --out DIR
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
from scipy.spatial.transform import Rotation

OBJECT_TYPES = (
  "largebox",
  "smallbox",
  "plasticbox",
  "suitcase",
  "whitechair",
  "woodchair",
  "largetable",
  "smalltable",
)
NOMINAL_MASS_KG = {
  "whitechair": 2.5,
  "woodchair": 3.0,
  "smalltable": 2.5,
  "largetable": 4.0,
  "smallbox": 0.5,
  "largebox": 2.0,
  "plasticbox": 1.0,
  "suitcase": 2.0,
}
# Canonical size per type: the HOI-Retarget geometry (OMOMO objects at 0.83).
CANONICAL_DATASET = "hoi_retarget"
G1_JOINT_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
  "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint",
  "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
  "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)  # fmt: skip
HAND_CHANNELS = {
  "left": ("left_wrist_yaw_link__object", "left_rubber_hand__object"),
  "right": ("right_wrist_yaw_link__object", "right_rubber_hand__object"),
}
FOOT_CHANNELS = {
  "left": ("left_ankle_roll_link__object",),
  "right": ("right_ankle_roll_link__object",),
}
GROUND_CHANNEL = "object__ground"
HOI_RETARGET_PREFIX = "omomo/unitree_g1/"
DT = 0.02
MIN_FRAMES = 25
MIN_SEGMENT = 50
"""Shortest piece kept after trimming (frames)."""
COACD = {
  "threshold": 0.06,
  "max_convex_hull": 12,
  "decimate": False,
  "merge": True,
  "seed": 0,
  "preprocess_mode": "on",
  "preprocess_resolution": 60,
}
HULL_MAX_PLANES = 38
"""Faces per object hull (a polytope with F faces has at most 2F - 4 vertices)."""
HULL_TOL = 0.001
"""How far an object hull may stick out of the CoACD piece it covers (m)."""
SDF_VOXEL = 0.01
SDF_MARGIN = 0.12


# --------------------------------------------------------------------------- clips


def object_type(clip_id: str) -> str | None:
  return next((t for t in OBJECT_TYPES if re.search(rf"(^|[_/]){t}(_|$)", clip_id)), None)


def ultra_scale(clip_id: str) -> tuple[int, int, int] | None:
  m = re.search(r"_(\d{3})_(\d{3})_(\d{3})$", clip_id)
  return tuple(int(x) for x in m.groups()) if m else None


def read_table(path: Path) -> list[dict[str, str]]:
  """Rows of the user's clip table (.xlsx, .tsv or .csv), keyed by header."""
  if path.suffix.lower() == ".xlsx":
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
      for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
        shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    grid = []
    for row in sheet.iter(f"{{{ns['m']}}}row"):
      cells = {}
      for c in row.findall("m:c", ns):
        col = 0
        for ch in re.match(r"[A-Z]+", c.get("r")).group(0):
          col = col * 26 + ord(ch) - 64
        v = c.find("m:v", ns)
        if c.get("t") == "s" and v is not None:
          cells[col - 1] = shared[int(v.text)]
        elif c.get("t") == "inlineStr":
          cells[col - 1] = "".join(x.text or "" for x in c.iter(f"{{{ns['m']}}}t"))
        else:
          cells[col - 1] = v.text if v is not None else ""
      grid.append([cells.get(i, "") for i in range(max(cells, default=-1) + 1)])
  else:
    import csv

    with open(path, newline="", encoding="utf-8-sig") as f:
      grid = list(csv.reader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
  header = [h.strip().lower() for h in grid[0]]
  return [dict(zip(header, [c.strip() for c in r] + [""] * (len(header) - len(r)))) for r in grid[1:] if r and r[0].strip()]


def entries_from_table(table: str, dataset: str, index: list[dict]) -> list[dict]:
  """Index entries for the table rows, tagged with their categories and trim flag.

  The first column names the clip (HOI-Retarget ids may omit their
  ``omomo/unitree_g1/`` prefix); columns other than ``notes`` are category
  flags; a note mentioning "trim" keeps only the clip's first run free of
  single-hand contact (a clip with no such run is dropped).
  """
  rows = read_table(Path(table))
  by_id = {(e["dataset"], e["clip_id"]): e for e in index}
  first = next(iter(rows[0]))
  out: dict[str, dict] = {}
  row_of: dict[str, int] = {}
  for line, r in enumerate(rows, start=2):
    name = r[first]
    clip_id = name if "/" in name or dataset != "hoi_retarget" else HOI_RETARGET_PREFIX + name
    entry = by_id.get((dataset, clip_id))
    if entry is None:
      raise KeyError(f"table clip {name!r} is not in the index as {dataset}:{clip_id}")
    if clip_id in out:
      # Usually a typo for another clip; the table needs fixing, not a guess.
      raise ValueError(f"{Path(table).name}: {name} is on rows {row_of[clip_id]} and {line}")
    categories = [k for k, v in r.items() if k not in (first, "notes") and v]
    out[clip_id] = {**entry, "categories": categories, "trim": "trim" in r.get("notes", "").lower(),
                    "note": r.get("notes", "")}
    row_of[clip_id] = line
  return list(out.values())


def entries_from_clip_list(path: str, index: list[dict]) -> list[dict]:
  """Index entries for a clip list: one ``clip_id`` or ``dataset:clip_id`` per line."""
  wanted = [c.strip() for c in Path(path).read_text().splitlines()]
  wanted = [c for c in wanted if c and not c.startswith("#")]
  by_id = defaultdict(list)
  for e in index:
    by_id[e["clip_id"]].append(e)
  out = []
  for c in wanted:
    dataset, _, clip = c.partition(":") if ":" in c and not c.startswith("omomo/") else ("", "", c)
    hits = [e for e in by_id.get(clip or c, []) if not dataset or e["dataset"] == dataset]
    if len(hits) != 1:
      raise KeyError(f"clip {c!r} matches {len(hits)} index entries")
    out.append(hits[0])
  return out


def select_entries(args) -> list[dict]:
  """Clips from every ``--table`` and ``--clips`` given (combined), else by filter."""
  index = [json.loads(line) for line in Path(args.index).read_text().splitlines() if line]
  if args.table or args.clips:
    out = [e for t in args.table for e in entries_from_table(t, args.table_dataset, index)]
    out += [e for c in args.clips for e in entries_from_clip_list(c, index)]
    seen = collections.Counter((e["dataset"], e["clip_id"]) for e in out)
    dup = [k for k, n in seen.items() if n > 1]
    if dup:
      raise ValueError(f"clips listed more than once: {dup[:5]}")
    return out
  objects = set(args.objects.split(",")) if args.objects else set(OBJECT_TYPES)
  datasets = set(args.datasets.split(","))
  per_variant = defaultdict(int)
  out = []
  for e in index:
    t = object_type(e["clip_id"])
    if t is None or t not in objects or e["dataset"] not in datasets:
      continue
    if e["dataset"] == "ultra" and ultra_scale(e["clip_id"]) != (100, 100, 100):
      continue
    key = (t, e["dataset"])
    if args.max_per_variant and per_variant[key] >= args.max_per_variant:
      continue
    per_variant[key] += 1
    out.append(e)
  return out


def limb_contact(
  labels: np.ndarray, intersect: np.ndarray, channels: list[str], limb: dict[str, tuple[str, ...]]
) -> np.ndarray:
  """Per side: 1 if any of the limb's channels is in proximity (label 1) or
  intersects the object (the pipeline labels intersections unknown), 0 if all
  are known separated, else -1."""
  out = np.full((labels.shape[0], 2), -1, np.int8)
  for k, side in enumerate(("left", "right")):
    cols = [channels.index(c) for c in limb[side]]
    touch = (labels[:, cols] == 1).any(1) | intersect[:, cols].any(1)
    out[:, k] = np.where(touch, 1, np.where((labels[:, cols] == 0).all(1), 0, -1))
  return out


def keep_segments(keep: np.ndarray, min_len: int) -> list[tuple[int, int]]:
  """``[lo, hi)`` runs of True in ``keep`` that are at least ``min_len`` long."""
  edges = np.diff(np.r_[0, keep.astype(np.int8), 0])
  starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
  return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True) if b - a >= min_len]


def angular_velocity(quat_wxyz: np.ndarray) -> np.ndarray:
  rot = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
  n = len(rot)
  w = np.zeros((n, 3))
  if n >= 3:
    w[1:-1] = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * DT)
  step = (rot[1:] * rot[:-1].inv()).as_rotvec() / DT
  w[0], w[-1] = step[0], step[-1]
  return w


def unit_wxyz(q: np.ndarray) -> np.ndarray:
  q = q / np.linalg.norm(q, axis=1, keepdims=True)
  # Continuous sign so finite differences and interpolation stay on one sheet.
  flip = np.cumsum(np.r_[False, (q[1:] * q[:-1]).sum(1) < 0]) % 2 == 1
  q[flip] *= -1
  return q


PER_FRAME = (
  "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel",
  "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w",
  "hand_contact", "foot_contact", "ref_intersection",
)  # fmt: skip


def load_clip(entry: dict) -> list[dict]:
  """The clip as a list of pieces: the whole clip, or for a trimmed clip its
  first run free of single-hand contact (empty if there is none).

  ``ref_intersection`` marks frames where the pipeline found a robot body
  touching the inside of the object (it records no depth; the simulator task
  measures depth itself to choose reset frames).
  """
  out = Path(entry["output"])
  meta = json.loads((out / "metadata.json").read_text())
  arr = np.load(out / "labels.npz")
  t = arr["timestamps_s"]
  if len(t) < MIN_FRAMES or not np.allclose(np.diff(t), DT, atol=1e-6):
    raise ValueError(f"{entry['clip_id']}: needs a regular 50 Hz grid of >= {MIN_FRAMES} frames")
  channels = list(meta["channels"])
  labels = np.asarray(arr["target_labels"])
  intersect = np.asarray(arr["target_intersection"]).astype(bool)
  robot_cols = [i for i, c in enumerate(channels) if c != GROUND_CHANNEL]
  root_quat = unit_wxyz(np.asarray(arr["root_quat"], float))
  obj_quat = unit_wxyz(np.asarray(arr["object_quat"], float))
  root_pos = np.asarray(arr["root_pos"], float)
  obj_pos = np.asarray(arr["object_pos"], float)
  joint_pos = np.asarray(arr["joint_pos"], float)
  full = {
    "root_pos_w": root_pos,
    "root_quat_w": root_quat,
    "root_lin_vel_w": np.gradient(root_pos, DT, axis=0),
    "root_ang_vel_w": angular_velocity(root_quat),
    "joint_pos": joint_pos,
    "joint_vel": np.gradient(joint_pos, DT, axis=0),
    "object_pos_w": obj_pos,
    "object_quat_w": obj_quat,
    "object_lin_vel_w": np.gradient(obj_pos, DT, axis=0),
    "object_ang_vel_w": angular_velocity(obj_quat),
    "hand_contact": limb_contact(labels, intersect, channels, HAND_CHANNELS),
    "foot_contact": limb_contact(labels, intersect, channels, FOOT_CHANNELS),
    "ref_intersection": intersect[:, robot_cols].any(1),
  }
  n = len(t)
  if entry.get("trim"):
    # Keep only the first run of at least MIN_SEGMENT frames without
    # single-hand contact; everything else in the clip is dropped.
    single_hand = (full["hand_contact"] == 1).sum(1) == 1
    pieces = keep_segments(~single_hand, MIN_SEGMENT)[:1]
  else:
    pieces = [(0, n)]
  base = {
    "clip_id": entry["clip_id"],
    "dataset": entry["dataset"],
    "type": object_type(entry["clip_id"]),
    "geometry_sha256": meta["object_geometry_sha256"],
    "geometry_path": meta["object_geometry_path"],
    "categories": entry.get("categories", []),
    "trimmed": bool(entry.get("trim")),
    "source_frames": n,
  }
  out_pieces = []
  for k, (lo, hi) in enumerate(pieces):
    name = entry["clip_id"] if len(pieces) == 1 and not entry.get("trim") else f"{entry['clip_id']}#{k}"
    out_pieces.append({**base, "name": name, "span": [lo, hi], **{f: full[f][lo:hi] for f in PER_FRAME}})
  return out_pieces


# ------------------------------------------------------------------------ geometry


def geometry_mesh(path: str):
  import trimesh

  geo = json.loads(Path(path).read_text())
  parts = []
  for s in geo["shapes"]:
    if s["kind"] == "mesh":
      parts.append(trimesh.Trimesh(np.asarray(s["vertices"], float), np.asarray(s["faces"]), process=True))
    elif s["kind"] == "cube":
      parts.append(trimesh.creation.box(extents=np.asarray(s["size"], float)))
    else:
      raise ValueError(f"unsupported shape kind {s['kind']}")
  return trimesh.util.concatenate(parts)


def outer_polytope(vertices: np.ndarray, max_planes: int, tol: float) -> np.ndarray:
  """Vertices of a convex polytope of at most ``max_planes`` faces that contains
  the convex hull of ``vertices``: start from the box on the points' principal
  axes and keep adding the hull face plane the polytope crosses most, until no
  corner sticks out more than ``tol``. Every plane supports the hull, so the
  polytope is conservative and follows the real faces whatever their direction."""
  hull = ConvexHull(vertices)
  normals, offsets = hull.equations[:, :3], -hull.equations[:, 3]  # n.x <= offset inside
  _, _, axes = np.linalg.svd(vertices - vertices.mean(0), full_matrices=False)
  dirs = [*axes, *-axes]
  centre = vertices.mean(0)
  while True:
    d = np.asarray(dirs)
    hs = np.concatenate([d, -(vertices @ d.T).max(0)[:, None]], 1)
    corners = HalfspaceIntersection(hs, centre).intersections
    excess = corners @ normals.T - offsets
    if excess.max() <= tol or len(dirs) >= max_planes:
      return corners
    dirs.append(normals[np.unravel_index(excess.argmax(), excess.shape)[1]])


def write_obj(path: Path, hull) -> None:
  lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in hull.vertices]
  lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in hull.faces]
  path.write_text("\n".join(lines) + "\n")


def union_sdf(hulls, origin, shape, voxel):
  """Signed distance to the union of convex hulls on a regular grid (+ normals)."""
  import trimesh

  axes = [origin[i] + voxel * np.arange(shape[i]) for i in range(3)]
  grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
  depth = np.full(len(grid), -np.inf)
  for h in hulls:
    n = h.face_normals
    off = np.einsum("ij,ij->i", n, h.triangles[:, 0])
    depth = np.maximum(depth, (off - grid @ n.T).min(1))  # > 0 inside this hull
  inside = depth > 0
  samples = np.concatenate([trimesh.sample.sample_surface(h, 40_000, seed=i)[0] for i, h in enumerate(hulls)])
  phi = np.empty(len(grid))
  phi[inside] = -depth[inside]
  phi[~inside] = cKDTree(samples).query(grid[~inside])[0]
  phi = phi.reshape(shape)
  g = np.stack(np.gradient(phi, voxel), -1)
  g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
  return phi.astype(np.float32), g.astype(np.float16), inside.reshape(shape)


def cook_variant(name: str, geometry_path: str, mass: float, out: Path) -> dict:
  import coacd
  import trimesh

  coacd.set_log_level("warn")
  mesh = geometry_mesh(geometry_path)
  pieces = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), **COACD)
  hulls = [trimesh.convex.convex_hull(outer_polytope(np.asarray(v, float), HULL_MAX_PLANES, HULL_TOL)) for v, _ in pieces]
  out.mkdir(parents=True, exist_ok=True)
  files = []
  for i, h in enumerate(hulls):
    p = out / f"hull_{i:02d}.obj"
    write_obj(p, h)
    files.append(p.name)
  verts = np.concatenate([h.vertices for h in hulls]).astype(np.float32)
  np.save(out / "verts.npy", verts)
  lo = verts.min(0) - SDF_MARGIN
  shape = np.ceil((verts.max(0) + SDF_MARGIN - lo) / SDF_VOXEL).astype(int) + 1
  phi, normal, inside = union_sdf(hulls, lo, shape, SDF_VOXEL)
  np.savez_compressed(out / "sdf.npz", phi=phi, normal=normal, origin=lo.astype(np.float32), voxel=np.float32(SDF_VOXEL))
  # Solid inertia from the occupied voxels, scaled to the nominal mass.
  cells = np.stack(np.nonzero(inside), -1) * SDF_VOXEL + lo
  com = cells.mean(0)
  r = cells - com
  inertia = (mass / len(cells)) * (np.eye(3) * (r**2).sum() - r.T @ r)
  evals, evecs = np.linalg.eigh(inertia)
  if np.linalg.det(evecs) < 0:
    evecs[:, 0] *= -1
  iquat = Rotation.from_matrix(evecs).as_quat()[[3, 0, 1, 2]]
  return {
    "hulls": files,
    "hull_vertices_max": int(max(len(h.vertices) for h in hulls)),
    "extent_m": (verts.max(0) - verts.min(0)).round(4).tolist(),
    "ipos": com.round(6).tolist(),
    "iquat": iquat.round(8).tolist(),
    "inertia": np.maximum(evals, 1e-6).round(8).tolist(),
    "sdf_shape": shape.tolist(),
  }


# ---------------------------------------------------------------------------- main


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--index", required=True, help="curated selection included.jsonl")
  ap.add_argument("--clips", action="append", default=[],
                  help="text file, one clip_id (or dataset:clip_id) per line; repeatable")
  ap.add_argument("--table", action="append", default=[],
                  help="clip table (.xlsx/.tsv/.csv): clip, category flags, notes; repeatable")
  ap.add_argument("--table-dataset", default="hoi_retarget", help="dataset of the table's clips")
  ap.add_argument("--objects", help="comma list of object types (when --clips is not given)")
  ap.add_argument("--datasets", default="hoi_retarget", help="comma list (when --clips is not given)")
  ap.add_argument("--max-per-variant", type=int, default=0)
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  out = Path(args.out)
  if out.exists() and any(out.iterdir()):
    raise FileExistsError(f"{out} is not empty")
  (out / "hoi" / "objects").mkdir(parents=True, exist_ok=True)

  entries = select_entries(args)
  if not entries:
    raise ValueError("no clips selected")
  with ProcessPoolExecutor(args.workers) as pool:
    pieces = [p for clip in pool.map(load_clip, entries, chunksize=8) for p in clip]
  bad = [c["clip_id"] for c in pieces if c["type"] is None]
  if bad:
    raise ValueError(f"clips without a known object type: {bad[:5]}")
  clips = pieces
  untrimmed = [e["clip_id"] for e in entries if e.get("trim")]
  untrimmed = [c for c in untrimmed if not any(p["clip_id"] == c for p in pieces)]

  # Variants: one per (type, dataset, geometry).
  keys = sorted({(c["type"], c["dataset"], c["geometry_sha256"]) for c in clips}, key=lambda k: (OBJECT_TYPES.index(k[0]), k[1], k[2]))
  variants = []
  extents = {}
  for v, (t, ds, sha) in enumerate(keys):
    name = f"{t}_{ds}" if sum(k[0] == t and k[1] == ds for k in keys) == 1 else f"{t}_{ds}_{sha[:8]}"
    path = next(c["geometry_path"] for c in clips if (c["type"], c["dataset"], c["geometry_sha256"]) == (t, ds, sha))
    print(f"[cook] {name}", flush=True)
    info = cook_variant(name, path, NOMINAL_MASS_KG[t], out / "hoi" / "objects" / name)
    extents[v] = np.linalg.norm(info["extent_m"])
    variants.append({"name": name, "type": t, "type_id": OBJECT_TYPES.index(t), "dataset": ds,
                     "geometry_sha256": sha, "mass": NOMINAL_MASS_KG[t], **info})
  for v, var in enumerate(variants):
    canon = [u for u, w in enumerate(variants) if w["type"] == var["type"] and w["dataset"] == CANONICAL_DATASET]
    var["scale"] = round(float(extents[v] / extents[canon[0]]), 4) if canon else 1.0
  variant_of = {k: v for v, k in enumerate(keys)}

  fields = ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel"]
  hoi_fields = ["object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"]
  for f in fields:
    np.save(out / f"{f}.npy", np.concatenate([c[f] for c in clips]).astype(np.float32))
  for f in hoi_fields:
    np.save(out / "hoi" / f"{f}.npy", np.concatenate([c[f] for c in clips]).astype(np.float32))
  for f in ("hand_contact", "foot_contact", "ref_intersection"):
    np.save(out / "hoi" / f"{f}.npy", np.concatenate([c[f] for c in clips]))
  np.save(out / "motion_ends.npy", np.cumsum([len(c["joint_pos"]) for c in clips]))
  np.save(out / "hoi" / "clip_variant.npy",
          np.asarray([variant_of[(c["type"], c["dataset"], c["geometry_sha256"])] for c in clips], np.int64))
  (out / "joint_names.json").write_text(json.dumps(list(G1_JOINT_NAMES)))
  (out / "motion_names.json").write_text(json.dumps([f"{c['dataset']}:{c['name']}" for c in clips]))
  (out / "hoi" / "clips.json").write_text(json.dumps([
    {"name": c["name"], "dataset": c["dataset"], "source_clip": c["clip_id"], "categories": c["categories"],
     "trimmed": c["trimmed"], "span": c["span"], "source_frames": c["source_frames"], "frames": len(c["joint_pos"]),
     "ref_intersection_frac": round(float(c["ref_intersection"].mean()), 3),
     "foot_contact_frames": int((c["foot_contact"] == 1).any(1).sum())}
    for c in clips
  ], indent=1))
  for var in variants:
    ids = [i for i, c in enumerate(clips) if variant_of[(c["type"], c["dataset"], c["geometry_sha256"])] == variants.index(var)]
    var["clips"] = len(ids)
    var["frames"] = int(sum(len(clips[i]["joint_pos"]) for i in ids))
  (out / "hoi" / "variants.json").write_text(json.dumps({"object_types": list(OBJECT_TYPES), "variants": variants}, indent=2))
  digest = hashlib.sha256("\n".join(f"{c['dataset']}:{c['name']}" for c in clips).encode()).hexdigest()
  inter = np.concatenate([c["ref_intersection"] for c in clips])
  (out / "build_manifest.json").write_text(json.dumps({
    "index": str(Path(args.index).resolve()), "clip_lists": args.clips, "tables": args.table,
    "clip_list_sha256": digest, "source_clips": len(entries), "clips": len(clips),
    "frames": int(sum(len(c["joint_pos"]) for c in clips)),
    "ref_intersection_frac": round(float(inter.mean()), 4), "trim_requested_but_nothing_kept": untrimmed,
    "coacd": COACD, "hull_max_planes": HULL_MAX_PLANES, "hull_tol_m": HULL_TOL, "sdf_voxel_m": SDF_VOXEL,
    "min_segment_frames": MIN_SEGMENT,
  }, indent=2))
  for var in variants:
    print(f"{var['name']:28s} clips={var['clips']:4d} frames={var['frames']:6d} hulls={len(var['hulls'])} "
          f"scale={var['scale']} extent={var['extent_m']}")
  trimmed = [c for c in clips if c["trimmed"]]
  print(f"source clips {len(entries)} -> pieces {len(clips)}; trimmed pieces {len(trimmed)} "
        f"({sum(len(c['joint_pos']) for c in trimmed)} of {sum(c['source_frames'] for c in {x['clip_id']: x for x in trimmed}.values())} frames kept)")
  print(f"frames where the pipeline found a robot body touching the inside of the object: {inter.mean():.1%}")
  print(f"trims that kept nothing: {untrimmed}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
