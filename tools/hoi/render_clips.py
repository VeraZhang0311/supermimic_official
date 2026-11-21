"""Render exported HOI clips (``tools/hoi/export_clips.py``) to mp4 with MuJoCo, offscreen.

Kinematic playback: the G1 follows the reference and the object is a mocap body
carrying its mesh at ``object_scale``. A hand's geoms turn red while its
``hand_contact`` label is 1. The camera follows the midpoint of robot and object.

  MUJOCO_GL=egl python tools/hoi/render_clips.py EXPORT_DIR clips/a.npz clips/b.npz --out videos
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

CONTACT_RGBA = (0.9, 0.15, 0.1, 1.0)


def build_model(root: Path, z) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(root / "robot" / "g1_29dof.xml"))
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=(0.82, 0.83, 0.85), rgb2=(0.72, 0.73, 0.76), width=512, height=512)
  spec.add_material(name="grid", textures=["", "grid"], texrepeat=(8, 8))
  for g in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]:
    spec.delete(g)  # the robot file's own floor would z-fight with ours
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05), material="grid")
  light = spec.worldbody.add_light(pos=(0, 0, 4), dir=(0, 0, -1), diffuse=(0.7, 0.7, 0.7))
  if hasattr(mujoco.mjtLightType, "mjLIGHT_DIRECTIONAL"):
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  else:
    light.directional = True
  spec.add_mesh(name="object", file=str((root / str(z["object_mesh"])).resolve()), scale=z["object_scale"].astype(float))
  body = spec.worldbody.add_body(name="object", mocap=True)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="object", contype=0, conaffinity=0,
                rgba=(0.35, 0.55, 0.85, 1.0))
  return spec.compile()


def hand_geoms(model: mujoco.MjModel) -> dict[int, list[int]]:
  """Geoms of each side's hand (rubber hand body, and wrist yaw link if no hand body)."""
  out = {}
  for k, side in enumerate(("left", "right")):
    bodies = {model.body(f"{side}_rubber_hand").id, model.body(f"{side}_wrist_yaw_link").id}
    out[k] = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
  return out


def render(root: Path, clip: Path, out: Path, width: int, height: int) -> None:
  z = np.load(clip)
  model = build_model(root, z)
  data = mujoco.MjData(model)
  if [model.joint(i).name for i in range(1, model.njnt)] != list(z["joint_names"]):
    raise ValueError("robot joint order differs from the clip")
  hands = hand_geoms(model)
  base_rgba = model.geom_rgba.copy()
  mocap = model.body("object").mocapid[0]
  renderer = mujoco.Renderer(model, height, width)
  cam = mujoco.MjvCamera()
  cam.distance, cam.azimuth, cam.elevation = 2.6, 135.0, -18.0
  fps = float(z["fps"])
  ff = subprocess.Popen(
    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
     "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
    stdin=subprocess.PIPE,
  )
  look = None
  for t in range(len(z["root_pos"])):
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = z["root_pos"][t], z["root_quat"][t], z["joint_pos"][t]
    data.mocap_pos[mocap], data.mocap_quat[mocap] = z["object_pos"][t], z["object_quat"][t]
    mujoco.mj_kinematics(model, data)
    model.geom_rgba[:] = base_rgba
    for k, geoms in hands.items():
      if z["hand_contact"][t, k] == 1:
        model.geom_rgba[geoms] = CONTACT_RGBA
    target = 0.5 * (z["root_pos"][t] + z["object_pos"][t])
    target[2] = 0.6
    look = target if look is None else 0.9 * look + 0.1 * target  # smooth follow
    cam.lookat[:] = look
    renderer.update_scene(data, cam)
    ff.stdin.write(renderer.render().tobytes())
  ff.stdin.close()
  if ff.wait():
    raise RuntimeError(f"ffmpeg failed on {clip}")
  renderer.close()


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("root", help="export directory (has robot/, objects/, clips/)")
  ap.add_argument("clips", nargs="+", help="clip .npz paths relative to root")
  ap.add_argument("--out", required=True)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  args = ap.parse_args()
  root, out = Path(args.root), Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  for c in args.clips:
    dst = out / (Path(c).stem + ".mp4")
    render(root, root / c, dst, args.width, args.height)
    print(dst, flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
