# G1 hand and foot collision hulls for HOI

Convex hulls for the G1 rubber hands and feet. They replace the mjlab G1 hand
capsule and add foot-top collision for objects. Hull vertices are in the frame of
the source mesh; mount each hull with the `geom_pos` given in `manifest.json`.

| part | body | geom pos | hulls | max verts / hull |
|---|---|---|---|---|
| left hand | `left_wrist_yaw_link` | `0.0415 0.003 0` | 20 | 136 |
| right hand | `right_wrist_yaw_link` | `0.0415 -0.003 0` | 20 | 136 |
| left foot | `left_ankle_roll_link` | `0 0 0` | 12 | 136 |
| right foot | `right_ankle_roll_link` | `0 0 0` | 12 | 136 |

These positions match how mjlab's `g1.xml` mounts the rubber-hand and
ankle-roll visual meshes. All four bodies have explicit `<inertial>` elements,
so adding hull geoms does not change any link's inertia.

## Intended use

- Hands: all hand contacts (objects, ground and the rest of the robot). Remove
  `left_hand_collision` / `right_hand_collision`.
- Feet: object contacts only. Ground contact stays on mjlab's sole capsules,
  which the foot-friction randomization and `FootFrictionSpec` look up by name.
- Use `condim="3"` and `solref="0.01 1"` for contacts with objects (see below).

## How the hulls were made

`tools/hoi/cook_g1_collision.py`, from the pinned source STLs (sha256 in
`manifest.json`, identical to the ULTRA repo's meshes):

1. CoACD, threshold 0.06, at most 20 hulls per hand and 12 per foot, decimation
   off. Each hull is the convex hull of one piece of the mesh, so together they
   contain the whole mesh.
2. Outer k-DOP per hull with 64 directions plus the 6 axes. Keeping only support
   planes can only grow a hull, so this caps vertices at 136 per hull without
   uncovering any surface.

The old recipe (5 hulls, at most 16 vertices, decimation on) shrank the hulls
inside the visual surface by up to 1.6 cm.

## Checks

From `manifest.json` (cook) and `validation.json`
(`tools/hoi/validate_g1_collision.py`, MuJoCo 3.15):

| part | uncovered surface | hull outside mesh, p95 / max |
|---|---|---|
| left / right hand | 0 mm | 2.75 / 7.4 mm, 2.57 / 11.4 mm |
| left / right foot | 0 mm | 6.99 / 13.1 mm, 6.87 / 12.7 mm |

In a compiled MuJoCo model, every vertex of all four visual meshes lies inside
the hulls (0.5 mm tolerance).

Press tests on the left foot: a box on a slide joint pushed into the static
foot. Penetration is how far its leading face ends up inside the visual surface
(timestep 5 ms).

| scenario | old 5x16 hulls, MuJoCo default | new hulls, MuJoCo default | new hulls, solref 0.01, elliptic |
|---|---|---|---|
| 3 cm leg on top, 20 N, 18 spots | 3.1 mm | none | none |
| 1.6 cm leg on top, 20 N, 50 spots | 31.3 mm | 8.3 mm | none |
| 2 cm block on the side, 30 N, 25 spots | 19.9 mm | none | none |

"None" means the box stopped 0.5–1.1 mm outside the visual surface. Settled
jitter was at most 0.05 mm with solref 0.01. Adding `solimp="0.99 0.99 0.01"`
(as ULTRA's MuJoCo model does) changed nothing measurable here.

## Not checked yet

- That mjlab_research ships the same STLs (compare its sha256 with the manifest).
- Collision cost in mjwarp with 4096 worlds.
- Hand press tests; the hands were checked for coverage only.
