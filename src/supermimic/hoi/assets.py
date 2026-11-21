"""Scene entities for HOI: the G1 with hand/foot hulls and the object slot.

Collision filtering (contype = own category bit, conaffinity = categories it
accepts; a pair collides if either side accepts the other). The terrain plane
keeps mjlab's default contype = conaffinity = 1.

  category        contype  conaffinity       collides with
  ground (plane)      1        1             body, foot capsules
  body capsules,      2    1|2|4|16 = 23     ground, robot, object
   hand hulls
  foot capsules       4      1|2|4 = 7       ground, robot (not objects)
  foot hulls          8       16             object only
  object             16       32             object floor (+ whoever accepts 16)
  object floor       32        0             object only

The object rests on a finite box floor (top at z = 0, like the plane) instead of
the plane: a plane is a half-space and would catch the parked hulls of unused
variants whenever the object turns them below ground. Parked hulls sit
``PARK_DISTANCE`` from their body, beyond the floor in every direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ARTICULATION,
  KNEES_BENT_KEYFRAME,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from supermimic.hoi.library import ObjectVariant

ASSET_DIR = Path(__file__).parent / "assets" / "g1_collision"
GROUND, BODY, FOOT_CAPSULE, FOOT_HULL, OBJECT, OBJECT_FLOOR = 1, 2, 4, 8, 16, 32
OBJECT_SOLREF = (0.01, 1.0)
OBJECT_FRICTION = (1.0, 0.005, 0.0001)
FLOOR_HALF_SIZE = (100.0, 100.0, 0.5)
PARK_DISTANCE = 1000.0
"""Inactive object hulls sit this far (body +z) from their body: off the floor."""
OBJECT_START_POS = (3.0, 0.0, 1.0)

_FOOT_CAPSULE = r"^(left|right)_foot[1-7]_collision$"
_FOOT_HULL = r"^(left|right)_foot_hull_\d+$"
_HAND_HULL = r"^(left|right)_hand_hull_\d+$"

G1_HOI_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision$", _HAND_HULL, _FOOT_HULL),
  contype={_FOOT_CAPSULE: FOOT_CAPSULE, _FOOT_HULL: FOOT_HULL, ".*": BODY},
  conaffinity={
    _FOOT_CAPSULE: GROUND | BODY | FOOT_CAPSULE,
    _FOOT_HULL: OBJECT,
    ".*": GROUND | BODY | FOOT_CAPSULE | OBJECT,
  },
  # Body capsules stay frictionless among themselves (condim 1, as in mjlab);
  # MuJoCo takes the larger condim per pair, so their contacts with objects
  # (condim 3) have friction.
  condim={_FOOT_CAPSULE: 3, _FOOT_HULL: 3, _HAND_HULL: 3, ".*": 1},
  priority={_FOOT_CAPSULE: 1, ".*": 0},
  friction={_FOOT_CAPSULE: (0.6,)},
  solref={_FOOT_HULL: OBJECT_SOLREF, _HAND_HULL: OBJECT_SOLREF},
)


def _read_obj(path: Path) -> tuple[list[float], list[int]]:
  verts, faces = [], []
  for line in path.read_text().splitlines():
    if line.startswith("v "):
      verts.extend(float(x) for x in line.split()[1:4])
    elif line.startswith("f "):
      faces.extend(int(x.split("/")[0]) - 1 for x in line.split()[1:4])
  return verts, faces


def g1_hoi_spec() -> mujoco.MjSpec:
  """mjlab's G1 with the hand capsules replaced by rubber-hand hulls and foot hulls added."""
  spec = get_spec()
  for side in ("left", "right"):
    spec.delete(spec.geom(f"{side}_hand_collision"))
  manifest = json.loads((ASSET_DIR / "manifest.json").read_text())
  for key, entry in manifest["parts"].items():
    side, part = key.split("_")
    body = spec.body(entry["link"])
    for i, hull in enumerate(entry["hulls"]):
      name = f"{side}_{part}_hull_{i:02d}"
      verts, faces = _read_obj(ASSET_DIR / hull["file"])
      spec.add_mesh(name=name, uservert=verts, userface=faces)
      body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        pos=entry["geom_pos"],
        group=3,
        rgba=(0.2, 0.6, 0.2, 0.3),
      )
  return spec


def get_g1_hoi_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(G1_HOI_COLLISION,),
    spec_fn=g1_hoi_spec,
    articulation=G1_ARTICULATION,
  )


def object_geom_name(variant: int, hull: int) -> str:
  return f"obj_v{variant:02d}_h{hull:02d}"


def object_spec(variants: list[ObjectVariant]) -> mujoco.MjSpec:
  """One free body carrying every variant's hulls; variant 0 active, the rest parked.

  The body's inertia is set explicitly (geoms add no mass); a startup event
  rewrites it per world for that world's variant.
  """
  spec = mujoco.MjSpec()
  # qpos0 (what the first forward sees) keeps the object clear of the robot.
  body = spec.worldbody.add_body(name="object", pos=OBJECT_START_POS)
  body.add_freejoint(name="object_joint")
  v0 = variants[0]
  body.explicitinertial = True
  body.mass = v0.mass
  body.ipos = v0.ipos
  body.iquat = v0.iquat
  body.inertia = v0.inertia
  for v, variant in enumerate(variants):
    for i, path in enumerate(variant.hull_files):
      name = object_geom_name(v, i)
      verts, faces = _read_obj(Path(path))
      spec.add_mesh(name=name, uservert=verts, userface=faces)
      body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        pos=(0.0, 0.0, 0.0 if v == 0 else PARK_DISTANCE),
        contype=OBJECT,
        conaffinity=OBJECT_FLOOR,
        condim=3,
        friction=OBJECT_FRICTION,
        solref=OBJECT_SOLREF,
        group=0,
        rgba=(0.85, 0.55, 0.25, 1.0),
      )
  return spec


def object_floor_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object_floor", pos=(0.0, 0.0, -FLOOR_HALF_SIZE[2]))
  body.add_geom(
    name="object_floor",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=FLOOR_HALF_SIZE,
    contype=OBJECT_FLOOR,
    conaffinity=0,
    condim=3,
    friction=OBJECT_FRICTION,
    solref=OBJECT_SOLREF,
    group=3,
    rgba=(0.5, 0.5, 0.5, 0.0),
  )
  return spec


def get_object_floor_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=object_floor_spec)


def get_object_cfg(variants: list[ObjectVariant]) -> EntityCfg:
  # Away from the robot until the first reset puts it at its reference pose.
  return EntityCfg(
    spec_fn=lambda: object_spec(variants),
    init_state=EntityCfg.InitialStateCfg(pos=OBJECT_START_POS),
  )
