"""Replay a checkpoint on every clip of a library and save the simulated trajectories.

Each clip is played once from its first frame with the protected evaluation's
settings and scored with its rules (``supermimic.hoi.evaluate``: EMA actor, nominal
physics and object mass, no domain randomization, no reset noise, no early
termination, seed 42). Writes into ``--out``:

  clips/<dataset>__<clip>.npz   simulated states, velocities, joint position targets,
                                actuator torques, robot-object contact forces and
                                deepest contact distances, the eval's per-frame errors
                                and the clip's verdicts
  clips.csv                     one row per clip: split, object, contact type, verdicts
  scenes/<variant>/scene.xml    standalone MJCF per object variant: robot, that object,
                                floors and the simulator's options (SCENE_LS_ITERATIONS
                                aside); the robot's meshes in scenes/assets/
  scenes/replay_check.json      the scenes compared with the simulator's model, and
                                saved states re-simulated one control step in CPU
                                MuJoCo on them (replay_check.py, copied next to it)
  physics.json                  simulator options, control rate, actuators, objects

  python tools/hoi/sim_rollouts.py --train-lib LIB --lib FULL_LIB --run-dir logs/... \
      --wandb-run entity/project/run_id --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_check import check_dir  # noqa: E402

LIMBS = {
  "left_hand": "left_wrist_yaw_link",
  "right_hand": "right_wrist_yaw_link",
  "left_foot": "left_ankle_roll_link",
  "right_foot": "right_ankle_roll_link",
}
CONTACT_NAMES = [*LIMBS, "robot"]
CHECK_CLIPS_PER_VARIANT = 10
SCENE_LS_ITERATIONS = 50
"""Line-search iterations of the exported scenes (MuJoCo's default). The simulator
runs MuJoCo Warp with 20; CPU MuJoCo with 20 stops some line searches early in
contact-rich steps and then departs from the saved states by up to ~1 rad in one
control step, with 50 it stays within a few mrad (replay_check.py)."""


def contact_sensors():
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  obj = ContactMatch(mode="body", pattern="object", entity="object")
  return (
    ContactSensorCfg(
      name="limb_object_contact",
      primary=ContactMatch(mode="body", pattern=tuple(LIMBS.values()), entity="robot"),
      secondary=obj,
      fields=("found", "force"),
      reduce="netforce",
    ),
    ContactSensorCfg(
      name="robot_object_contact",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=obj,
      fields=("found", "force"),
      reduce="netforce",
    ),
    # The deepest contact of the same pairs (signed distance, negative inside).
    ContactSensorCfg(
      name="limb_object_depth",
      primary=ContactMatch(mode="body", pattern=tuple(LIMBS.values()), entity="robot"),
      secondary=obj,
      fields=("found", "dist"),
      reduce="mindist",
    ),
    ContactSensorCfg(
      name="robot_object_depth",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=obj,
      fields=("found", "dist"),
      reduce="mindist",
    ),
  )


def make_recorder(base):
  import mujoco
  import torch

  class Recorder(base):
    """HoiEvaluator that also keeps every world's state after reset and after each step."""

    def __init__(self, runner, library):
      super().__init__(runner, library)
      raw = self.env.unwrapped
      m = raw.sim.mj_model
      self.robot = raw.scene["robot"]
      free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
      self.free_adr = [int(m.jnt_qposadr[j]) for j in free]
      obj_body = int(self.cmd.object.indexing.body_ids[0])
      self.object_adr = next(
        int(m.jnt_qposadr[j]) for j in free if m.jnt_bodyid[j] == obj_body
      )
      self.robot_adr = next(a for a in self.free_adr if a != self.object_adr)
      self.limb = raw.scene["limb_object_contact"]
      self.total = raw.scene["robot_object_contact"]
      self.limb_depth = raw.scene["limb_object_depth"]
      self.total_depth = raw.scene["robot_object_depth"]
      self.limb_order, self.limb_depth_order = (
        torch.as_tensor(
          [s.primary_names.index(n) for n in self._limb_primary_names(s)],
          device=raw.device,
        )
        for s in (self.limb, self.limb_depth)
      )
      self.frames, self.steps = defaultdict(list), defaultdict(list)
      reset = self.env.reset

      def reset_and_capture(*a, **k):
        out = reset(*a, **k)
        self.frames.clear()
        self.steps.clear()
        self._capture_state()
        return out

      self.env.reset = reset_and_capture

    @staticmethod
    def _limb_primary_names(sensor):
      names = sensor.primary_names
      return [next(n for n in names if n.endswith(link)) for link in LIMBS.values()]

    def _capture_state(self):
      raw = self.env.unwrapped
      d, origins = raw.sim.data, raw.scene.env_origins
      qpos = d.qpos.clone()
      for a in self.free_adr:
        qpos[:, a : a + 3] -= origins
      r, o = self.robot.data, self.cmd.object.data
      f = self.frames
      f["qpos"].append(qpos)
      f["qvel"].append(d.qvel.clone())
      f["joint_pos"].append(r.joint_pos.clone())
      f["joint_vel"].append(r.joint_vel.clone())
      f["root_lin_vel"].append(r.root_link_lin_vel_w.clone())
      f["root_ang_vel"].append(r.root_link_ang_vel_w.clone())
      f["object_lin_vel"].append(o.root_link_lin_vel_w.clone())
      f["object_ang_vel"].append(o.root_link_ang_vel_w.clone())
      f["time_steps"].append(self.cmd.time_steps.clone())

    def _record(self, rec, valid, robot, obj, origins) -> None:
      super()._record(rec, valid, robot, obj, origins)
      self._capture_state()
      s = self.steps
      s["ctrl"].append(self.env.unwrapped.sim.data.ctrl.clone())
      s["actuator_force"].append(robot.data.actuator_force.clone())
      limb, total = self.limb.data, self.total.data
      s["contact_force"].append(
        torch.cat([limb.force[:, self.limb_order], total.force], 1)
      )
      s["contact_count"].append(
        torch.cat([limb.found[:, self.limb_order], total.found], 1)
      )
      limb, total = self.limb_depth.data, self.total_depth.data
      dist = torch.cat([limb.dist[:, self.limb_depth_order], total.dist], 1).reshape(
        len(valid), -1
      )
      found = torch.cat([limb.found[:, self.limb_depth_order], total.found], 1)
      s["contact_dist"].append(torch.where(found > 0, dist, torch.nan))

    def _report(self, rec, diverged, wandb):
      self.series = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
      self.diverged = diverged
      return super()._report(rec, diverged, wandb)

  return Recorder


def build_runner(args, device):
  import torch
  import wandb
  from mjlab.tasks.registry import load_env_cfg

  import supermimic.tasks  # noqa: F401
  from supermimic.hoi.env_cfg import configure_hoi_env
  from supermimic.runners.run_config import load_runner_cfg
  from supermimic.runners.supermimic_runner import SuperMimicRunner
  from supermimic.scripts.train_utils import create_env

  run_dir = Path(args.run_dir)
  cfg = load_runner_cfg(wandb.Api().run(args.wandb_run).config)
  cfg.eval_motion_file = None  # the recording evaluator is built separately
  env_cfg = load_env_cfg("SuperMimic-HOI-G1", play=False)
  configure_hoi_env(env_cfg, args.train_lib)
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
  return runner


def versions() -> dict:
  import mujoco

  out = {"mujoco": mujoco.__version__}
  for pkg in ("mujoco-warp", "warp-lang", "mjlab", "torch", "numpy"):
    try:
      out[pkg] = metadata.version(pkg)
    except metadata.PackageNotFoundError:
      out[pkg] = None
  return out


def opt_dict(opt) -> dict:
  out = {}
  for name in dir(opt):
    if name.startswith("_"):
      continue
    v = getattr(opt, name)
    if callable(v):
      continue
    out[name] = (
      np.asarray(v).tolist()
      if isinstance(v, np.ndarray)
      else (int(v) if hasattr(v, "__index__") else v)
    )
  return out


def write_scenes(raw, variants, out: Path) -> dict:
  """One standalone MJCF per variant, checked field by field against the simulator's model."""
  import mujoco
  from mjlab.utils.spec import export_spec

  m = raw.sim.mj_model
  base = raw.scene.spec.copy()
  for (
    ent
  ) in raw.scene.entities.values():  # as Scene.write: entity meshes compile from disk
    for mesh in ent.spec.meshes:
      path = Path(ent.spec.modelfiledir) / mesh.compiler.meshdir / mesh.file
      if mesh.file and mesh.file not in base.assets and path.is_file():
        base.assets[mesh.file] = path.read_bytes()
  for name, value in opt_dict(m.opt).items():
    try:
      setattr(base.option, name, value)
    except (AttributeError, TypeError):
      pass
  base.option.ls_iterations = SCENE_LS_ITERATIONS
  shared = out / "scenes" / "assets"
  report = {}
  for v, var in enumerate(variants):
    spec = base.copy()
    prefix = f"object/obj_v{v:02d}_h"
    for g in [g for g in spec.geoms if g.name.startswith("object/obj_v")]:
      if g.name.startswith(prefix):
        g.pos = [0.0, 0.0, 0.0]
      else:
        mesh = g.meshname
        spec.delete(g)
        spec.delete(spec.mesh(mesh))
    body = spec.body("object/object")
    body.mass, body.ipos, body.iquat, body.inertia = (
      var.mass,
      var.ipos,
      var.iquat,
      var.inertia,
    )
    dst = out / "scenes" / var.name
    export_spec(spec, dst)
    # One copy of the robot's mesh files for all scenes (the hulls are inline).
    for f in sorted(p for p in (dst / "assets").rglob("*") if p.is_file()):
      to = shared / f.relative_to(dst / "assets")
      if to.exists():
        if to.read_bytes() != f.read_bytes():
          raise ValueError(f"{f} differs from {to}")
      else:
        to.parent.mkdir(parents=True, exist_ok=True)
        f.replace(to)
    shutil.rmtree(dst / "assets")
    xml = (dst / "scene.xml").read_text()
    if xml.count('meshdir="assets"') != 1:
      raise ValueError(f"{dst}/scene.xml: expected one meshdir")
    (dst / "scene.xml").write_text(
      xml.replace('meshdir="assets"', 'meshdir="../assets"')
    )
    xm = mujoco.MjModel.from_xml_path(str(dst / "scene.xml"))
    sim_opt = opt_dict(m.opt)
    diffs = [
      k
      for k, val in opt_dict(xm.opt).items()
      if k != "ls_iterations" and not np.allclose(val, sim_opt[k], rtol=1e-6, atol=1e-9)
    ]
    for field in ("nq", "nv", "nu", "nbody", "njnt"):
      if getattr(xm, field) != getattr(m, field):
        diffs.append(field)
    names = lambda model, kind, n: [mujoco.mj_id2name(model, kind, i) for i in range(n)]  # noqa: E731
    for kind, n, fields in (
      (
        mujoco.mjtObj.mjOBJ_JOINT,
        "njnt",
        ("jnt_type", "jnt_range", "jnt_qposadr", "jnt_stiffness"),
      ),
      (
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "nu",
        (
          "actuator_gainprm",
          "actuator_biasprm",
          "actuator_forcerange",
          "actuator_ctrlrange",
          "actuator_trnid",
          "actuator_dyntype",
        ),
      ),
      (
        mujoco.mjtObj.mjOBJ_BODY,
        "nbody",
        ("body_mass", "body_inertia", "body_ipos", "body_iquat"),
      ),
    ):
      if names(xm, kind, getattr(xm, n)) != names(m, kind, getattr(m, n)):
        diffs.append(f"{kind.name} names")
        continue
      for f in fields:
        a, b = getattr(xm, f), getattr(m, f)
        if f.startswith(
          "body_"
        ):  # the object body differs on purpose (this variant's inertia)
          ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object/object")
          a, b = np.delete(a, ob, 0), np.delete(b, ob, 0)
        if not np.allclose(a, b, rtol=1e-5, atol=1e-6):
          diffs.append(f)
    for f in ("dof_armature", "dof_damping", "dof_frictionloss"):
      if not np.allclose(getattr(xm, f), getattr(m, f), rtol=1e-5, atol=1e-6):
        diffs.append(f)
    robot_geoms = [
      i
      for i in range(m.ngeom)
      if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("robot/")
    ]
    for i in robot_geoms:
      j = mujoco.mj_name2id(
        xm, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
      )
      for f in (
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_condim",
        "geom_contype",
        "geom_conaffinity",
        "geom_priority",
        "geom_margin",
        "geom_gap",
        "geom_size",
        "geom_type",
      ):
        if j < 0 or not np.allclose(
          getattr(xm, f)[j], getattr(m, f)[i], rtol=1e-5, atol=1e-6
        ):
          diffs.append(f"{f}[{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)}]")
    ob = mujoco.mj_name2id(xm, mujoco.mjtObj.mjOBJ_BODY, "object/object")
    obj_geoms = [i for i in range(xm.ngeom) if xm.geom_bodyid[i] == ob]
    report[var.name] = {
      "xml": f"scenes/{var.name}/scene.xml",
      "differences_from_simulator": sorted(set(diffs)),
      "intended_differences": {
        "ls_iterations": {
          "simulator": int(m.opt.ls_iterations),
          "scene": int(xm.opt.ls_iterations),
        }
      },
      "object_geoms": len(obj_geoms),
      "object_mass": float(xm.body_mass[ob]),
    }
    print(
      f"[scene] {var.name}: {len(obj_geoms)} object hulls, mass {xm.body_mass[ob]:.2f} kg, "
      f"differences {sorted(set(diffs)) or 'none'}",
      flush=True,
    )
  return report


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument(
    "--train-lib", required=True, help="library the runner is built on (same variants)"
  )
  ap.add_argument("--lib", required=True, help="library to replay, one world per clip")
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--wandb-run", required=True)
  ap.add_argument(
    "--num-envs", type=int, default=128, help="worlds of the runner's training env"
  )
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import torch

  from supermimic.hoi import evaluate
  from supermimic.hoi.library import read_variants

  out = Path(args.out)
  if (out / "clips").exists():
    raise FileExistsError(f"{out}/clips exists")
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  runner = build_runner(args, device)

  base_cfg = evaluate.hoi_eval_env_cfg

  def eval_cfg_with_sensors(library, *args, **kwargs):
    cfg = base_cfg(library, *args, **kwargs)
    cfg.scene.sensors = (*cfg.scene.sensors, *contact_sensors())
    return cfg

  evaluate.hoi_eval_env_cfg = eval_cfg_with_sensors
  ev = make_recorder(evaluate.HoiEvaluator)(runner, args.lib)
  raw = ev.env.unwrapped
  m = raw.sim.mj_model
  robot = ev.robot
  lib = Path(args.lib)
  if list(robot.joint_names) != json.loads((lib / "joint_names.json").read_text()):
    raise ValueError("robot joint order differs from the library's")
  lib_ref = {
    "root_pos_w": np.load(lib / "root_pos_w.npy"),
    "joint_pos": np.load(lib / "joint_pos.npy"),
    "object_pos_w": np.load(lib / "hoi" / "object_pos_w.npy"),
  }
  soft_limits = robot.data.soft_joint_pos_limits[0].cpu().numpy()
  joint_ids = robot.indexing.joint_ids.cpu().numpy()
  ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()
  if m.nu != len(ctrl_ids) or (m.actuator_dyntype != 0).any():
    raise ValueError(
      "expected only the robot's 29 position actuators, without activation states"
    )
  act_joint = m.actuator_trnid[ctrl_ids, 0]
  perm = np.asarray(
    [list(act_joint).index(j) for j in joint_ids]
  )  # robot actuator order -> joint order
  ctrl_cols = ctrl_ids[perm]
  ja = robot.indexing.joint_q_adr.cpu().numpy()
  if not (
    ev.robot_adr == 0
    and ev.object_adr == 36
    and np.array_equal(ja, np.arange(7, 36))
    and m.nq == 43
  ):
    raise ValueError(
      f"unexpected qpos layout: robot {ev.robot_adr}, joints {ja}, object {ev.object_adr}"
    )

  row = ev.run()
  table = row["Eval/clips_table"]
  verdicts = [dict(zip(table.columns, r, strict=True)) for r in table.data]
  frames = {
    k: torch.stack(v).cpu().numpy() for k, v in ev.frames.items()
  }  # (steps + 1, N, ...)
  steps = {
    k: torch.stack(v).cpu().numpy() for k, v in ev.steps.items()
  }  # (steps, N, ...)
  series, diverged = ev.series, ev.diverged
  valid = series["valid"]
  clips_json = json.loads((lib / "hoi" / "clips.json").read_text())
  variants = read_variants(lib)
  clip_variant = np.load(lib / "hoi" / "clip_variant.npy")
  ends = np.load(lib / "motion_ends.npy")
  starts = np.r_[0, ends[:-1]]
  if not len(verdicts) == len(clips_json) == raw.num_envs:
    raise ValueError("one world per clip expected")

  (out / "clips").mkdir(parents=True)
  rows = []
  dt = ev.dt
  for c, (info, verdict) in enumerate(zip(clips_json, verdicts, strict=True)):
    n = int(valid[:, c].sum())
    if not valid[:n, c].all():
      raise ValueError(f"world {c}: valid frames are not a prefix")
    # The eval starts a clip at its first allowed start frame (the reference does not
    # sink the robot into the object), frame 0 for most clips. mjlab's tracking
    # command advances its frame index once right after a reset has written that
    # frame's state, and once per step: simulated frame 0 is reference frame
    # start_frame, and frame k >= 1 is scored against reference frame start_frame + k + 1.
    index = frames["time_steps"][: n + 1, c] - starts[c]
    start_frame = int(index[0]) - 1
    if start_frame < 0 or not np.array_equal(index, index[0] + np.arange(n + 1)):
      raise ValueError(
        f"world {c}: simulated frames do not follow the clip's frames: {index[:5]}"
      )
    ref_frame = np.r_[start_frame, index[1:]]
    q = frames["qpos"][: n + 1, c]
    # The reset clips joints to the soft limits and lifts the object out of the floor.
    f0 = starts[c] + start_frame
    ref_joints = np.clip(lib_ref["joint_pos"][f0], soft_limits[:, 0], soft_limits[:, 1])
    if (
      np.abs(q[0, 0:3] - lib_ref["root_pos_w"][f0]).max() > 1e-4
      or np.abs(q[0, 7:36] - ref_joints).max() > 1e-4
      or np.abs(q[0, 36:38] - lib_ref["object_pos_w"][f0, :2]).max() > 1e-4
    ):
      raise ValueError(
        f"world {c}: the start state is not reference frame {start_frame}"
      )
    var = variants[clip_variant[c]]
    p = {
      "fps": np.float32(1.0 / dt),
      "up_axis": "z",
      "quat_order": "wxyz",
      "joint_names": np.asarray(robot.joint_names),
      "root_pos": q[:, 0:3],
      "root_quat": q[:, 3:7],
      "joint_pos": q[:, 7:36],
      "object_pos": q[:, 36:39],
      "object_quat": q[:, 39:43],
      "root_lin_vel": frames["root_lin_vel"][: n + 1, c],
      "root_ang_vel": frames["root_ang_vel"][: n + 1, c],
      "joint_vel": frames["joint_vel"][: n + 1, c],
      "object_lin_vel": frames["object_lin_vel"][: n + 1, c],
      "object_ang_vel": frames["object_ang_vel"][: n + 1, c],
      "qpos": q,
      "qvel": frames["qvel"][: n + 1, c],
      "ctrl": steps["ctrl"][:n, c],
      "joint_pos_target": steps["ctrl"][:n, c][:, ctrl_cols],
      "joint_torque": steps["actuator_force"][:n, c][:, perm],
      "contact_names": np.asarray(CONTACT_NAMES),
      "contact_force": steps["contact_force"][:n, c],
      "contact_count": steps["contact_count"][:n, c].astype(np.int16),
      "contact_min_dist": steps["contact_dist"][:n, c],
      "ref_frame": ref_frame.astype(np.int32),
    }
    for k in (
      "body_err_m",
      "left_hand_err_m",
      "right_hand_err_m",
      "object_pos_err_m",
      "object_ori_err_deg",
    ):
      p[f"err_{k}"] = series[k][:n, c]
    if not np.allclose(p["joint_pos"], frames["joint_pos"][: n + 1, c], atol=1e-6):
      raise ValueError(f"world {c}: joint positions in qpos and entity data differ")
    p = {
      k: (
        v.astype(np.float32)
        if isinstance(v, np.ndarray) and v.dtype == np.float64
        else v
      )
      for k, v in p.items()
    }
    file = info["package_file"]
    first_fail = verdict["first_fail_s"]
    meta = {
      "body_and_object_pose_pass": bool(verdict["body_and_object_pose_pass"]),
      "phc_pass": bool(verdict["phc_pass"]),
      "object_pose_pass": bool(verdict["object_pose_pass"]),
      "fell": bool(verdict["fell"]),
      "diverged": bool(diverged[c]),
      "first_fail_s": np.nan if first_fail is None else float(first_fail),
    }
    np.savez_compressed(
      out / file,
      **p,
      **{k: np.asarray(v) for k, v in meta.items()},
      failed=np.asarray(not meta["body_and_object_pose_pass"]),
      object_type=var.type,
      object_collision=var.name,
      object_mass=np.float32(var.mass),
      scene=f"scenes/{var.name}/scene.xml",
      split=info["split"],
      contact_type=verdict["kind"],
      source_dataset=info["dataset"],
      source_clip=info["source_clip"],
      reference_file=file,
    )
    rows.append(
      {
        "file": file,
        "dataset": info["dataset"],
        "clip": info["source_clip"],
        "split": info["split"],
        "object": var.type,
        "variant": var.name,
        "contact": verdict["kind"],
        "ref_frames": int(ends[c] - starts[c]),
        "start_frame": start_frame,
        "sim_frames": n + 1,
        "failed": int(not meta["body_and_object_pose_pass"]),
        "body_and_object_pose_pass": int(meta["body_and_object_pose_pass"]),
        "phc_pass": int(meta["phc_pass"]),
        "object_pose_pass": int(meta["object_pose_pass"]),
        "fell": int(meta["fell"]),
        "diverged": int(meta["diverged"]),
        "first_fail_s": "" if first_fail is None else round(float(first_fail), 2),
        **{
          k.replace("/", "_"): round(float(verdict[k]), 5)
          for k in (
            "body_err_m/mean",
            "object_pos_err_m/mean",
            "object_ori_err_deg/mean",
          )
        },
      }
    )
  with open(out / "clips.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

  scenes = write_scenes(raw, variants, out)
  decimation = int(raw.cfg.decimation)

  kp = m.actuator_gainprm[ctrl_cols, 0]
  kd = -m.actuator_biasprm[ctrl_cols, 2]
  dof = m.jnt_dofadr[joint_ids]
  physics = {
    "physics_timestep_s": float(m.opt.timestep),
    "decimation": decimation,
    "control_dt_s": float(dt),
    "mujoco_option": opt_dict(m.opt),
    "actuators": [
      {
        "joint": jn,
        "kp": float(kp[i]),
        "kd": float(kd[i]),
        "armature": float(m.dof_armature[dof[i]]),
        "damping": float(m.dof_damping[dof[i]]),
        "frictionloss": float(m.dof_frictionloss[dof[i]]),
        "effort_limit": float(np.abs(m.actuator_forcerange[ctrl_cols[i]]).max()),
        "joint_range": m.jnt_range[joint_ids[i]].tolist(),
      }
      for i, jn in enumerate(robot.joint_names)
    ],
    "objects": [
      {
        "variant": v.name,
        "type": v.type,
        "mass_kg": v.mass,
        "ipos": v.ipos,
        "iquat_wxyz": v.iquat,
        "inertia": v.inertia,
      }
      for v in variants
    ],
    "eval": {
      "seed": evaluate.EVAL_SEED,
      "body_fail_m": evaluate.BODY_FAIL_M,
      "object_pos_fail_m": evaluate.OBJECT_POS_FAIL_M,
      "object_ori_fail_deg": evaluate.OBJECT_ORI_FAIL_DEG,
      "object_fail_s": evaluate.OBJECT_FAIL_S,
      "fall_drop_m": evaluate.FALL_DROP_M,
      "fall_s": evaluate.FALL_S,
    },
    "scene_ls_iterations": SCENE_LS_ITERATIONS,
    "versions": versions(),
    "run_dir": args.run_dir,
    "wandb_run": args.wandb_run,
    "library": str(lib.resolve()),
  }
  (out / "physics.json").write_text(json.dumps(physics, indent=1))
  shutil.copyfile(Path(__file__).with_name("replay_check.py"), out / "replay_check.py")
  check = check_dir(out, CHECK_CLIPS_PER_VARIANT)
  (out / "scenes" / "replay_check.json").write_text(
    json.dumps({"scenes": scenes, "one_step_replays": check}, indent=1)
  )
  scal = {k: v for k, v in row.items() if k != "Eval/clips_table"}
  (out / "eval_scalars.json").write_text(json.dumps(scal, indent=1))
  by_split = defaultdict(list)
  for r in rows:
    by_split[r["split"]].append(r["body_and_object_pose_pass"])
  print(
    "[done] "
    + ", ".join(f"{s}: {sum(v)}/{len(v)} pass" for s, v in sorted(by_split.items()))
    + f"; {sum(1 - r['body_and_object_pose_pass'] for r in rows)} failed; saved {out}",
    flush=True,
  )
  assert math.isclose(
    sum(r["body_and_object_pose_pass"] for r in rows) / len(rows),
    scal["Eval/all/body_and_object_pose_pass_rate"],
  )


if __name__ == "__main__":
  main()
