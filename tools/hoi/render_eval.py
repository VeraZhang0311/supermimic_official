"""Replay the protected evaluation with a checkpoint and render chosen clips to mp4.

``rollout`` builds the runner on the training library (a small env), loads a run
directory's actor, EMA actor and world model, plays the held-out library exactly
as the in-training evaluation does (``supermimic.hoi.evaluate``), and saves every
world's simulated and reference state, the per-frame errors and the per-clip
verdicts, plus the scene model (``scene.mjb``).

``bundle`` picks ``--per-type`` clips per object type, ``--passes`` of them that
passed body_and_object_pose and the rest failing ones (topped up from the other
group when one runs short), and cuts them into a small portable bundle (poses,
errors, verdicts, object meshes in the object frame).

``render`` replays a bundle's saved poses (no policy, no physics) on any machine
with MuJoCo, Pillow and ffmpeg: three panels under one camera (simulation,
overlay with the reference as a green ghost, reference), the reference's hands
red while it labels contact, the verdict, live errors and error curves with the
eval's thresholds. It checks its own kinematics (the robot built from a G1 URDF)
against the simulator's body positions.

  python tools/hoi/render_eval.py rollout --train-lib LIB --eval-lib EVAL --run-dir logs/... \
      --wandb-run entity/project/run_id --out DIR                       # cluster, GPU
  python tools/hoi/render_eval.py bundle --out DIR --bundle B --objects OBJS   # cluster
  python tools/hoi/render_eval.py render --bundle B --videos V --urdf g1_29dof.urdf --jobs 4
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- rollout


def rollout(args) -> None:
  import mujoco
  import torch
  import wandb
  from mjlab.tasks.registry import load_env_cfg

  import supermimic.tasks  # noqa: F401
  from supermimic.hoi.env_cfg import configure_hoi_env, configure_hoi_physics
  from supermimic.hoi.evaluate import HoiEvaluator
  from supermimic.runners.run_config import load_runner_cfg
  from supermimic.runners.supermimic_runner import SuperMimicRunner
  from supermimic.scripts.train_utils import create_env

  class RecordingEvaluator(HoiEvaluator):
    """HoiEvaluator that also keeps every world's sim and reference state."""

    def _record(self, rec, valid, robot, obj, origins) -> None:
      super()._record(rec, valid, robot, obj, origins)
      raw, motion, lib, t = (
        self.env.unwrapped,
        self.cmd.motion,
        self.cmd.lib,
        self.cmd.time_steps,
      )
      qpos = raw.sim.data.qpos.clone()
      for adr in self.free_adr:
        qpos[:, adr : adr + 3] -= origins
      ref = torch.zeros_like(qpos)
      ref[:, self.robot_adr : self.robot_adr + 3] = motion.root_pos_w[t]
      ref[:, self.robot_adr + 3 : self.robot_adr + 7] = motion.root_quat_w[t]
      ref[:, self.joint_adr] = motion.joint_pos[t]
      ref[:, self.object_adr : self.object_adr + 3] = lib.object_pos_w[t]
      ref[:, self.object_adr + 3 : self.object_adr + 7] = lib.object_quat_w[t]
      self.frames["qpos"].append(qpos)
      self.frames["ref_qpos"].append(ref)
      self.frames["ref_hand_contact"].append(lib.hand_contact[t] == 1)
      self.recorded = rec

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  run_dir = Path(args.run_dir)
  cfg = load_runner_cfg(wandb.Api().run(args.wandb_run).config)
  cfg.eval_motion_file = None  # built below, as the recording subclass
  env_cfg = load_env_cfg("SuperMimic-HOI-G1", play=False)
  configure_hoi_env(env_cfg, args.train_lib)
  # The run's own physics, as train.py builds it: its sim context and world model match.
  configure_hoi_physics(
    env_cfg, cfg.hoi_physics, push=cfg.hoi_push, action_contract=cfg.hoi_action_contract
  )
  env_cfg.commands["motion"].sampling_mode = (
    "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
  )
  env = create_env(env_cfg, args.num_envs, device)
  runner = SuperMimicRunner(
    env=env, cfg=cfg, task_id="SuperMimic-HOI-G1", device=device
  )

  def load(name: str):
    return torch.load(run_dir / name, map_location=device, weights_only=False)

  runner._wm.load_bundle(load("world_model.pt"))
  runner.policy_learner.actor.load_state_dict(load("actor.pt"))
  if runner.policy_learner.ema_actor is not None:
    runner.policy_learner.ema_actor.load_state_dict(load("actor_ema.pt"))

  ev = RecordingEvaluator(runner, args.eval_lib, physics=args.physics)
  raw = ev.env.unwrapped
  mjm = raw.sim.mj_model
  robot, obj = raw.scene["robot"], ev.cmd.object
  free = [j for j in range(mjm.njnt) if mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  ev.free_adr = [int(mjm.jnt_qposadr[j]) for j in free]
  obj_body = int(obj.indexing.body_ids[0])
  ev.object_adr = next(
    int(mjm.jnt_qposadr[j]) for j in free if mjm.jnt_bodyid[j] == obj_body
  )
  ev.robot_adr = next(a for a in ev.free_adr if a != ev.object_adr)
  ev.joint_adr = robot.indexing.joint_q_adr
  ev.frames = {"qpos": [], "ref_qpos": [], "ref_hand_contact": []}
  row = ev.run()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  table = row["Eval/clips_table"]
  clips = [dict(zip(table.columns, r, strict=True)) for r in table.data]
  frames = {k: torch.stack(v).cpu().numpy() for k, v in ev.frames.items()}
  series = {k: torch.stack(v).cpu().numpy() for k, v in ev.recorded.items()}
  geom_pos = raw.sim.model.geom_pos
  geom_pos = (
    geom_pos.cpu().numpy()
    if geom_pos.ndim == 3
    else np.broadcast_to(geom_pos.cpu().numpy(), (len(clips), *geom_pos.shape))
  )
  np.savez_compressed(
    out / "rollout.npz",
    geom_pos=geom_pos,
    dt=ev.dt,
    robot_adr=ev.robot_adr,
    object_adr=ev.object_adr,
    **frames,
    **series,
  )
  mujoco.mj_saveModel(mjm, str(out / "scene.mjb"), None)
  scalars = {k: v for k, v in row.items() if k != "Eval/clips_table"}
  (out / "eval.json").write_text(
    json.dumps(
      {
        "run_dir": str(run_dir),
        "wandb_run": args.wandb_run,
        "eval_lib": args.eval_lib,
        "physics": ev.physics,
        "scalars": scalars,
        "clips": clips,
      },
      indent=1,
    )
  )
  print(
    f"[EVAL] replay: pass rate body {scalars['Eval/all/phc_pass_rate']:.2f}, object "
    f"{scalars['Eval/all/object_pose_pass_rate']:.2f}, both {scalars['Eval/all/body_and_object_pose_pass_rate']:.2f}, "
    f"fell {scalars['Eval/all/fell_rate']:.2f}; saved {out}"
  )


# ---------------------------------------------------------------- render


def choose(clips: list[dict], per_type: int, passes: int, seed: int) -> list[int]:
  """Indices of ``per_type`` clips per type: ``passes`` passing, the rest failing,
  topping up from the other group. Failing picks mix full losses (body failed
  too) with object-only failures when both exist."""
  rng = random.Random(seed)
  picked = []
  for t in sorted({c["type"] for c in clips}, key=[c["type"] for c in clips].index):
    idx = [i for i, c in enumerate(clips) if c["type"] == t]
    ok = [i for i in idx if clips[i]["body_and_object_pose_pass"]]
    bad = [i for i in idx if not clips[i]["body_and_object_pose_pass"]]
    rng.shuffle(ok)
    lost = [i for i in bad if not clips[i]["phc_pass"]]
    obj_only = [i for i in bad if clips[i]["phc_pass"]]
    rng.shuffle(lost)
    rng.shuffle(obj_only)
    bad = (
      [x for pair in zip(lost, obj_only, strict=False) for x in pair]
      + lost[len(obj_only) :]
      + obj_only[len(lost) :]
    )
    n_ok = min(len(ok), passes, per_type)
    n_bad = min(len(bad), per_type - n_ok)
    n_ok = min(len(ok), per_type - n_bad)
    picked += ok[:n_ok] + bad[:n_bad]
  return picked


SERIES = (
  "body_err_m", "left_hand_err_m", "right_hand_err_m", "object_pos_err_m",
  "object_ori_err_deg", "joint_rmse_rad", "ref_time_s",
)  # fmt: skip


def bundle(args) -> None:
  """Cut the chosen clips out of a rollout into a small portable bundle.

  Per clip: root pose and named joint angles of the simulated and reference
  robot, both object poses, the per-frame errors, the reference hand-contact
  labels, and every robot body's position from the simulator's own model (to
  check a renderer's kinematics against). Runs where ``scene.mjb`` loads (the
  MuJoCo that wrote it).
  """
  import shutil

  import mujoco

  out, dst = Path(args.out), Path(args.bundle)
  (dst / "clips").mkdir(parents=True, exist_ok=True)
  ev = json.loads((out / "eval.json").read_text())
  clips = ev["clips"]
  z = dict(np.load(out / "rollout.npz"))
  model = mujoco.MjModel.from_binary_path(str(out / "scene.mjb"))
  data = mujoco.MjData(model)
  ra, oa = int(z["robot_adr"]), int(z["object_adr"])
  root = next(
    int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_qposadr[j] == ra
  )
  bodies = [b for b in range(model.nbody) if model.body_rootid[b] == root]
  hinges = [
    j
    for j in range(model.njnt)
    if model.body_rootid[model.jnt_bodyid[j]] == root
    and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  jadr = [int(model.jnt_qposadr[j]) for j in hinges]
  names = {
    "joint_names": [model.joint(j).name.split("/")[-1] for j in hinges],
    "body_names": [model.body(b).name.split("/")[-1] for b in bodies],
  }
  index = []
  for c in choose(clips, args.per_type, args.passes, args.seed):
    clip = clips[c]
    frames = np.nonzero(z["valid"][:, c])[0]
    rec = {k: z[k][frames, c] for k in SERIES}
    rec["ref_hand_contact"] = z["ref_hand_contact"][frames, c]
    for side, key in (("sim", "qpos"), ("ref", "ref_qpos")):
      q = z[key][frames, c].astype(np.float64)
      rec[f"{side}_root_pos"], rec[f"{side}_root_quat"] = (
        q[:, ra : ra + 3],
        q[:, ra + 3 : ra + 7],
      )
      rec[f"{side}_joint_pos"] = q[:, jadr]
      rec[f"{side}_object_pos"], rec[f"{side}_object_quat"] = (
        q[:, oa : oa + 3],
        q[:, oa + 3 : oa + 7],
      )
      xpos = np.empty((len(q), len(bodies), 3))
      for i, row in enumerate(q):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        xpos[i] = data.xpos[bodies]
      rec[f"{side}_body_pos"] = xpos
    tag = "pass" if clip["body_and_object_pose_pass"] else "fail"
    name = f"{clip['type']}_{tag}_{clip['clip'].split('/')[-1].split(':')[-1]}"
    np.savez_compressed(dst / "clips" / f"{name}.npz", dt=z["dt"], **names, **rec)
    index.append({"name": name, "frames": len(frames), **clip})
  if args.objects:
    shutil.copytree(args.objects, dst / "objects", dirs_exist_ok=True)
  meta = {k: ev[k] for k in ("run_dir", "wandb_run", "eval_lib")}
  meta |= {"label": args.label, "scalars": ev["scalars"], "clips": index}
  (dst / "index.json").write_text(json.dumps(meta, indent=1))
  print(f"bundled {len(index)} clips into {dst}")


# ---------------------------------------------------------------- local render
# Saved-pose replay: no policy, no physics. Runs on any machine with MuJoCo,
# Pillow, NumPy and ffmpeg. MuJoCo opens no files itself (Windows paths with
# non-ASCII characters fail): every asset is handed over as bytes.

PANEL_W, PANEL_H, HEAD, FOOT, CHART_H, FPS_TOL = 640, 480, 100, 46, 114, 1e-6
CANVAS = (3 * PANEL_W, HEAD + PANEL_H + FOOT + CHART_H + 12)
BG, TEXT, DIM, GRID = "#101b2b", "#d5e3f2", "#9eb3c8", "#36485b"
OK_C, BAD_C = "#70ecae", "#ffaaa4"
TITLES = (
  ("SIMULATION  |  policy", "#7ccaff"),
  ("OVERLAY  |  green = reference", "#ffb773"),
  ("REFERENCE", "#63f28a"),
)
OBJECT_RGBA = (0.90, 0.48, 0.12, 1.0)
REF_RGBA = (0.10, 1.0, 0.25, 1.0)
REF_GHOST_ALPHA = 0.32
CONTACT_RGBA = (1.0, 0.22, 0.18)
TYPE_ZH = {
  "largebox": "大箱", "smallbox": "小箱", "plasticbox": "塑料箱", "suitcase": "行李箱",
  "whitechair": "白椅", "woodchair": "木椅", "largetable": "大桌", "smalltable": "小桌",
}  # fmt: skip
HAND_LINKS = (
  "left_wrist_yaw_link",
  "left_rubber_hand",
  "right_wrist_yaw_link",
  "right_rubber_hand",
)
# group 0: simulated robot and object, 2: reference, 3: ground; 5: hidden
GROUPS = {"sim": (0, 3), "overlay": (0, 2, 3), "ref": (2, 3)}


def _robot_spec(urdf: Path):
  """The URDF's visual geoms as a free-floating MjSpec (collision geoms dropped)."""
  import re

  import mujoco

  text = re.sub(
    r"<mujoco>.*?</mujoco>", "", urdf.read_text(encoding="utf-8"), flags=re.S
  )
  text = re.sub(
    r"(<robot[^>]*>)",
    r'\1<mujoco><compiler discardvisual="false" strippath="true"/></mujoco>',
    text,
    count=1,
  )
  assets = {p.name: p.read_bytes() for p in (urdf.parent / "meshes").iterdir()}
  spec = mujoco.MjSpec.from_string(text, assets=assets)
  for g in list(spec.geoms):
    if g.contype or g.conaffinity:
      spec.delete(g)
  spec.worldbody.first_body().add_freejoint(name="root")
  return spec, assets


def build_scene(urdf: Path, object_obj: Path):
  """Ground, lights, and a simulated plus a reference copy of robot and object."""
  import mujoco

  spec = mujoco.MjSpec()
  spec.option.gravity = (0, 0, 0)
  spec.visual.global_.offwidth, spec.visual.global_.offheight = PANEL_W, PANEL_H
  spec.visual.quality.offsamples = 4
  hl = spec.visual.headlight
  hl.ambient, hl.diffuse, hl.specular = (0.5,) * 3, (0.7,) * 3, (0.15,) * 3
  spec.visual.rgba.haze = (0.08, 0.10, 0.14, 1)
  spec.add_texture(
    name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.16, 0.22, 0.31), rgb2=(0.03, 0.05, 0.08), width=512, height=3072,
  )  # fmt: skip
  spec.add_texture(
    name="groundtex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.20, 0.24, 0.29), rgb2=(0.24, 0.28, 0.33), width=512, height=512,
  )  # fmt: skip
  spec.add_material(
    name="ground",
    textures=["", "groundtex"],
    texrepeat=(12, 12),
    texuniform=True,
    reflectance=0.03,
  )
  world = spec.worldbody
  # Headlight only. On Windows (MuJoCo 3.9 offscreen) any scene light that casts
  # no shadow leaves every geom unlit (black), and shadow-casting lights darken
  # the translucent reference ghost.
  world.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(20, 20, 0.1),
    material="ground",
    group=3,
  )
  assets = {"object.obj": object_obj.read_bytes()}
  spec.add_mesh(name="object", file="object.obj")
  for side, group in (("sim", 0), ("ref", 2)):
    robot, robot_assets = _robot_spec(urdf)
    for g in robot.geoms:
      g.group = group
      if side == "ref":
        g.rgba = REF_RGBA
    assets |= robot_assets
    world.add_frame().attach_body(robot.worldbody.first_body(), f"{side}/", "")
    body = world.add_body(name=f"{side}/object")
    body.add_freejoint(name=f"{side}/object_root")
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_MESH, meshname="object", group=group,
      rgba=OBJECT_RGBA if side == "sim" else REF_RGBA,
    )  # fmt: skip
  spec.assets = assets
  return spec.compile()


def _fonts():
  from PIL import ImageFont

  for path in (
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  ):
    try:
      return {n: ImageFont.truetype(path, n) for n in (14, 16, 18, 20, 24, 28)}
    except OSError:
      continue
  return dict.fromkeys((14, 16, 18, 20, 24, 28), ImageFont.load_default())


def _camera_track(z) -> tuple[np.ndarray, np.ndarray]:
  """Shared camera: centre and distance over both robots and objects, smoothed."""
  pts = np.concatenate(
    [
      z["sim_body_pos"],
      z["ref_body_pos"],
      z["sim_object_pos"][:, None],
      z["ref_object_pos"][:, None],
    ],
    axis=1,
  )
  lo, hi = pts.min(1), pts.max(1)
  centre = (lo + hi) / 2
  dist = np.maximum(
    3.1, np.maximum((hi - lo)[:, :2].max(1) * 1.85, (hi - lo)[:, 2] * 1.65)
  )
  for i in range(1, len(centre)):
    centre[i] = centre[i - 1] * 0.85 + centre[i] * 0.15
    dist[i] = max(dist[i], dist[i - 1] * 0.98)
  return centre, dist


def _charts(z, clip: dict, fonts):
  """Error curves under the panels; returns the strip and each chart's box."""
  from PIL import Image, ImageDraw

  n = len(z["body_err_m"])
  first = clip["first_fail_s"]
  charts = (
    ("Body position error (cm)", [z["body_err_m"] * 100], 50.0),
    (
      "Hand position error (cm)  L / R",
      [z["left_hand_err_m"] * 100, z["right_hand_err_m"] * 100],
      None,
    ),
    ("Object position error (cm)", [z["object_pos_err_m"] * 100], 20.0),
    ("Object orientation error (deg)", [z["object_ori_err_deg"]], 45.0),
  )
  colors = ("#7ccaff", "#ffb773")
  w = CANVAS[0] // len(charts)
  strip = Image.new("RGB", (CANVAS[0], CHART_H), BG)
  draw = ImageDraw.Draw(strip)
  boxes = []
  for k, (title, lines, limit) in enumerate(charts):
    x0, x1, y0, y1 = k * w + 20, (k + 1) * w - 20, 24, CHART_H - 10
    top = max(10.0, max(float(v.max()) for v in lines) * 1.05, (limit or 0) * 1.15)
    draw.text((x0, 0), title, font=fonts[16], fill=TEXT)
    draw.text((x1 - 70, 0), f"max {top:.0f}", font=fonts[14], fill=DIM)
    draw.rectangle((x0, y0, x1, y1), outline=GRID)

    def y(v, y0=y0, y1=y1, top=top):
      return y1 - (y1 - y0) * min(float(v), top) / top

    if limit is not None:  # the eval's failure threshold
      for xs in range(x0, x1, 12):
        draw.line((xs, y(limit), min(xs + 6, x1), y(limit)), fill="#a8505a", width=1)
    if first is not None:
      t = z["ref_time_s"]
      xf = x0 + (x1 - x0) * np.searchsorted(t, first) / max(1, n - 1)
      draw.line((xf, y0, xf, y1), fill="#ff6b6b", width=2)
    for line, color in zip(lines, colors, strict=False):
      draw.line(
        [(x0 + (x1 - x0) * i / max(1, n - 1), y(v)) for i, v in enumerate(line)],
        fill=color,
        width=2,
      )
    boxes.append((x0, x1, y0, y1))
  return strip, boxes


def render_clip(
  urdf: Path,
  bundle_dir: Path,
  clip: dict,
  number: int,
  total: int,
  label: str,
  out: Path,
) -> dict:
  import mujoco
  from PIL import Image, ImageDraw

  z = dict(np.load(bundle_dir / "clips" / f"{clip['name']}.npz"))
  model = build_scene(urdf, bundle_dir / "objects" / f"{clip['type']}.obj")
  data = mujoco.MjData(model)
  n, dt = len(z["body_err_m"]), float(z["dt"])

  def jadr(name: str) -> int:
    return int(model.jnt_qposadr[model.joint(name).id])

  joints = {
    side: [jadr(f"{side}/{j}") for j in z["joint_names"]] for side in ("sim", "ref")
  }
  roots = {side: jadr(f"{side}/root") for side in ("sim", "ref")}
  objs = {side: jadr(f"{side}/object_root") for side in ("sim", "ref")}
  check = {
    side: ([model.body(f"{side}/{b}").id for b in z["body_names"]])
    for side in ("sim", "ref")
  }
  ref_geoms = np.nonzero(model.geom_group == 2)[0]
  ref_rgba = model.geom_rgba[ref_geoms].copy()
  hand_geoms = [
    [
      g
      for g in ref_geoms
      if model.body(model.geom_bodyid[g]).name.split("/")[-1]
      in HAND_LINKS[2 * s : 2 * s + 2]
    ]
    for s in (0, 1)
  ]
  centre, dist = _camera_track(z)
  cam = mujoco.MjvCamera()
  cam.type, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 135.0, -19.0
  options = {}
  for key, groups in GROUPS.items():
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[list(groups)] = 1
    options[key] = opt
  renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
  fonts = _fonts()
  strip, boxes = _charts(z, clip, fonts)
  ok = clip["body_and_object_pose_pass"]
  verdicts = (
    "   |   ".join(
      f"{n_} {'PASS' if clip[k] else 'FAIL'}"
      for n_, k in (
        ("Both", "body_and_object_pose_pass"),
        ("Body", "phc_pass"),
        ("Object", "object_pose_pass"),
      )
    )
    + f"   |   Fell {'YES' if clip['fell'] else 'no'}"
  )
  first = clip["first_fail_s"]
  kind = "手接触" if clip["kind"] == "hand" else "脚接触"
  head = f"{number:02d}/{total}  {TYPE_ZH.get(clip['type'], '')} {clip['type']}  |  {kind}  |  {clip['name'].split('_', 2)[-1]}"
  ff = subprocess.Popen(
    ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CANVAS[0]}x{CANVAS[1]}",
     "-r", f"{1 / dt:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
    stdin=subprocess.PIPE,
  )  # fmt: skip
  fk_err = 0.0
  try:
    for i in range(n):
      for side in ("sim", "ref"):
        data.qpos[roots[side] : roots[side] + 3] = z[f"{side}_root_pos"][i]
        data.qpos[roots[side] + 3 : roots[side] + 7] = z[f"{side}_root_quat"][i]
        data.qpos[joints[side]] = z[f"{side}_joint_pos"][i]
        data.qpos[objs[side] : objs[side] + 3] = z[f"{side}_object_pos"][i]
        data.qpos[objs[side] + 3 : objs[side] + 7] = z[f"{side}_object_quat"][i]
      mujoco.mj_kinematics(model, data)
      for side in ("sim", "ref"):
        fk_err = max(
          fk_err, float(np.abs(data.xpos[check[side]] - z[f"{side}_body_pos"][i]).max())
        )
      cam.lookat[:], cam.distance = centre[i], float(dist[i])
      contact = z["ref_hand_contact"][i]
      frame = Image.new("RGB", CANVAS, BG)
      for col, key in enumerate(("sim", "overlay", "ref")):
        rgba = ref_rgba.copy()
        if key == "overlay":
          rgba[:, 3] = REF_GHOST_ALPHA
        model.geom_rgba[ref_geoms] = rgba
        for s in (0, 1):
          if contact[s]:
            model.geom_rgba[hand_geoms[s], :3] = CONTACT_RGBA
        renderer.update_scene(data, camera=cam, scene_option=options[key])
        frame.paste(Image.fromarray(renderer.render()), (col * PANEL_W, HEAD))
      draw = ImageDraw.Draw(frame)
      t = float(z["ref_time_s"][i])
      draw.text((18, 5), head, font=fonts[28], fill="#eff6ff")
      right = f"{label}   |   {t:.2f} / {z['ref_time_s'][-1]:.2f} s   |   {1 / dt:g} fps"
      draw.text((CANVAS[0] - 18 - draw.textlength(right, font=fonts[18]), 12), right, font=fonts[18], fill=TEXT)
      for col, (title, color) in enumerate(TITLES):
        draw.text((col * PANEL_W + 18, 42), title, font=fonts[20], fill=color)
      draw.text((18, 71), verdicts, font=fonts[16], fill=OK_C if ok else BAD_C)
      failed = first is not None and t >= first
      draw.text((PANEL_W + 18, 71), f"first failure at {first:.2f} s" if first is not None else "no failure",
                font=fonts[16], fill=BAD_C if failed else DIM)  # fmt: skip
      draw.text(
        (2 * PANEL_W + 18, 71),
        "hands turn red while the reference labels contact",
        font=fonts[16],
        fill=DIM,
      )
      for x in (PANEL_W, 2 * PANEL_W):
        draw.line((x, 39, x, HEAD + PANEL_H), fill="#516377", width=2)
      y = HEAD + PANEL_H + 10
      draw.text(
        (18, y),
        f"Now: body {z['body_err_m'][i] * 100:.1f} cm   |   hands {z['left_hand_err_m'][i] * 100:.1f} / "
        f"{z['right_hand_err_m'][i] * 100:.1f} cm   |   object {z['object_pos_err_m'][i] * 100:.1f} cm, "
        f"{z['object_ori_err_deg'][i]:.1f} deg   |   joints {np.degrees(z['joint_rmse_rad'][i]):.1f} deg RMS",
        font=fonts[18], fill=TEXT,
      )  # fmt: skip
      frame.paste(strip, (0, HEAD + PANEL_H + FOOT))
      for x0, x1, y0, y1 in boxes:
        x = x0 + (x1 - x0) * i / max(1, n - 1)
        draw.line(
          (x, HEAD + PANEL_H + FOOT + y0, x, HEAD + PANEL_H + FOOT + y1),
          fill="#f4f8ff",
          width=1,
        )
      ff.stdin.write(np.asarray(frame).tobytes())
      if i in (0, n // 2, n - 1):
        frame.save(out.with_name(f"{out.stem}_{i:04d}.jpg"), quality=92)
  finally:
    renderer.close()
    ff.stdin.close()
    if ff.wait():
      raise RuntimeError(f"ffmpeg failed on {out}")
  return {"video": out.name, "frames": n, "fk_max_error_m": fk_err}


def render(args) -> None:
  bundle_dir, out = Path(args.bundle), Path(args.videos)
  out.mkdir(parents=True, exist_ok=True)
  meta = json.loads((bundle_dir / "index.json").read_text(encoding="utf-8"))
  clips = meta["clips"]
  if args.jobs > 1 and args.shard is None:
    procs = [
      subprocess.Popen([sys.executable, __file__, "render", "--bundle", args.bundle, "--videos", args.videos,
                        "--urdf", args.urdf, "--shard", str(k), "--jobs", str(args.jobs)])
      for k in range(args.jobs)
    ]  # fmt: skip
    if any(p.wait() for p in procs):
      raise SystemExit("a render shard failed")
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*.render.json"))]
    _index_html(out, meta, rows)
    return
  for k, clip in enumerate(clips):
    if args.shard is not None and k % args.jobs != args.shard:
      continue
    t0 = time.time()
    done = render_clip(
      Path(args.urdf),
      bundle_dir,
      clip,
      k + 1,
      len(clips),
      meta.get("label", ""),
      out / f"{clip['name']}.mp4",
    )
    row = {**clip, **done, "order": k, "render_s": time.time() - t0}
    if row["fk_max_error_m"] > args.fk_tol:
      raise SystemExit(
        f"{clip['name']}: renderer kinematics off by {row['fk_max_error_m']:.4f} m"
      )
    (out / f"{clip['name']}.render.json").write_text(json.dumps(row, indent=1))
    print(f"{clip['name']}: {row['frames']} frames in {row['render_s']:.0f} s, kinematics within "
          f"{row['fk_max_error_m'] * 1000:.2f} mm", flush=True)  # fmt: skip
  if args.shard is None:
    _index_html(
      out, meta, [json.loads(p.read_text()) for p in sorted(out.glob("*.render.json"))]
    )


def _index_html(out: Path, meta: dict, rows: list[dict]) -> None:
  """A page with every video, grouped by object type, and the run's eval numbers."""
  import html

  rows = sorted(rows, key=lambda r: r["order"])
  s = meta["scalars"]
  parts = [
    "<!doctype html><meta charset='utf-8'><title>HOI eval videos</title>",
    "<style>body{background:#101b2b;color:#d5e3f2;font-family:'Microsoft YaHei',sans-serif;margin:24px}"
    "video{width:100%;max-width:1200px;display:block;margin:6px 0 22px}h2{margin-top:36px}"
    ".ok{color:#70ecae}.bad{color:#ffaaa4}code{color:#9eb3c8}</style>",
    f"<h1>{html.escape(meta.get('label', ''))}</h1>",
    f"<p>eval replay of all {int(s['Eval/all/clips'])} held-out clips: both pass "
    f"{s['Eval/all/body_and_object_pose_pass_rate']:.2f}, body {s['Eval/all/phc_pass_rate']:.2f}, "
    f"object {s['Eval/all/object_pose_pass_rate']:.2f}, fell {s['Eval/all/fell_rate']:.2f}. "
    f"<code>{html.escape(meta['run_dir'])}</code></p>",
  ]
  last = None
  for r in rows:
    if r["type"] != last:
      last = r["type"]
      parts.append(f"<h2>{TYPE_ZH.get(last, '')} {last}</h2>")
    cls = "ok" if r["body_and_object_pose_pass"] else "bad"
    verdict = "PASS" if r["body_and_object_pose_pass"] else "FAIL"
    first = (
      f", first failure {r['first_fail_s']:.2f} s"
      if r["first_fail_s"] is not None
      else ""
    )
    parts.append(
      f"<p class='{cls}'>{verdict} — {html.escape(r['clip'])} ({r['kind']}, {r['seconds']:.1f} s{first})</p>"
    )
    parts.append(
      f"<video controls preload='metadata' src='{html.escape(r['video'])}'></video>"
    )
  (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
  print(f"wrote {out / 'index.html'}")


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  sub = ap.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("rollout")
  r.add_argument("--train-lib", required=True)
  r.add_argument("--eval-lib", required=True)
  r.add_argument(
    "--run-dir",
    required=True,
    help="directory with actor.pt, actor_ema.pt, world_model.pt",
  )
  r.add_argument(
    "--wandb-run", required=True, help="entity/project/run_id, for the runner config"
  )
  r.add_argument("--num-envs", type=int, default=128)
  r.add_argument(
    "--physics",
    choices=("mjlab", "hw3"),
    help="robot physics of the evaluation (default: the run's); the policy keeps its own action scale",
  )
  r.add_argument("--out", required=True)
  b = sub.add_parser("bundle")
  b.add_argument("--out", required=True, help="the rollout's output directory")
  b.add_argument("--bundle", required=True)
  b.add_argument(
    "--objects", help="dir with <type>.obj, each object's mesh in its body frame"
  )
  b.add_argument(
    "--label", default="", help="shown on every video, e.g. the checkpoint"
  )
  b.add_argument("--per-type", type=int, default=4)
  b.add_argument("--passes", type=int, default=2)
  b.add_argument("--seed", type=int, default=42)
  v = sub.add_parser("render")
  v.add_argument("--bundle", required=True)
  v.add_argument("--videos", required=True, help="output directory")
  v.add_argument("--urdf", required=True, help="G1 29-DoF URDF with meshes/ next to it")
  v.add_argument("--jobs", type=int, default=1, help="render processes")
  v.add_argument("--shard", type=int, help=argparse.SUPPRESS)
  v.add_argument(
    "--fk-tol", type=float, default=0.005, help="max body position mismatch (m)"
  )
  args = ap.parse_args()
  {"rollout": rollout, "bundle": bundle, "render": render}[args.cmd](args)


if __name__ == "__main__":
  main()
