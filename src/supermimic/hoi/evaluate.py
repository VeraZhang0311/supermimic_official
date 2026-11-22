"""Protected evaluation of the HOI teacher, in the training process.

A second, small env plays every held-out clip once from its start (world i plays
clip i) with the EMA actor, nominal physics (no domain randomization, nominal
object mass), no reset noise and no early terminations, seed 42. Each clip gets:

  phc_pass                   the mean distance of the robot's bodies to the
                             reference is <= 0.5 m in every frame
  object_pose_pass           the object is never off by > 0.2 m or > 45 degrees
                             (full rotation angle, a 180-degree flip counts as
                             180) for longer than 0.3 s
  body_and_object_pose_pass  both
  fell                       the pelvis stays > 0.3 m below the reference's for
                             0.3 s

plus mean / p95 / max / final of the body, hand, object, joint and actuator
diagnostics, and the reference time of the first failure. ``run()`` returns
them as ``Eval/<group>/<stat>`` for W&B, groups ``all``, each object type and
``hand`` / ``foot`` clips, plus a per-clip table.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import quat_apply

from supermimic.hoi.env_cfg import configure_hoi_env, configure_hoi_physics, hoi_env_cfg
from supermimic.hoi.geometry import hand_probes
from supermimic.hoi.library import read_variants
from supermimic.hoi.mdp import WRIST_BODIES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

  from supermimic.runners.supermimic_runner import SuperMimicRunner

EVAL_SEED = 42
BODY_FAIL_M = 0.5
OBJECT_POS_FAIL_M = 0.2
OBJECT_ORI_FAIL_DEG = 45.0
OBJECT_FAIL_S = 0.3
FALL_DROP_M = 0.3
FALL_S = 0.3
SATURATION = 0.99
"""An actuator is saturated at this fraction of its effort limit."""
DIAGNOSTICS = (
  "body_err_m", "left_hand_err_m", "right_hand_err_m", "object_pos_err_m",
  "object_ori_err_deg", "joint_rmse_rad", "action_saturation",
)  # fmt: skip
KEEP_TERMINATIONS = ("time_out", "motion_end", "nonfinite_state")


def hoi_eval_env_cfg(
  library: str, physics: str = "mjlab", action_contract: str | None = None
) -> ManagerBasedRlEnvCfg:
  """The HOI env for evaluation: one world per clip of ``library``, nominal ``physics``
  (``configure_hoi_physics``) with the policy's ``action_contract``."""
  cfg = hoi_env_cfg(play=True)
  configure_hoi_env(cfg, library)
  configure_hoi_physics(cfg, physics, nominal=True, action_contract=action_contract)
  cmd = cfg.commands["motion"]
  cmd.eval_clips = True
  cmd.sampling_mode, cmd.start_of_clip = "uniform", False
  cmd.pose_range, cmd.velocity_range, cmd.joint_position_range = {}, {}, (0.0, 0.0)
  for name in ("foot_friction", "base_com", "encoder_bias", "push_robot"):
    cfg.events.pop(name, None)
  event = cfg.events["object_variant"]
  event.params = {**event.params, "mass_scale_range": (1.0, 1.0)}
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in KEEP_TERMINATIONS}
  cfg.seed = EVAL_SEED
  return cfg


def _runs_at_least(flags: np.ndarray, frames: int) -> int | None:
  """Index of the frame at which ``flags`` has been true for ``frames`` frames in a row."""
  run = 0
  for i, f in enumerate(flags):
    run = run + 1 if f else 0
    if run >= frames:
      return i
  return None


def score_clip(series: dict[str, np.ndarray], ref_time_s: np.ndarray, dt: float, diverged: bool) -> dict:
  """Verdicts and diagnostics of one clip from its per-frame series (valid frames only).

  ``series`` holds DIAGNOSTICS plus ``pelvis_drop_m``; ``ref_time_s`` is each
  frame's time in the reference clip.
  """
  body = series["body_err_m"]
  phc_fail = np.nonzero(body > BODY_FAIL_M)[0]
  phc_at = int(phc_fail[0]) if len(phc_fail) else None
  off = (series["object_pos_err_m"] > OBJECT_POS_FAIL_M) | (series["object_ori_err_deg"] > OBJECT_ORI_FAIL_DEG)
  obj_at = _runs_at_least(off, int(math.floor(OBJECT_FAIL_S / dt + 1e-9)) + 1)  # longer than 0.3 s
  fall_at = _runs_at_least(series["pelvis_drop_m"] > FALL_DROP_M, int(math.ceil(FALL_S / dt - 1e-9)))
  out = {
    "phc_pass": phc_at is None and not diverged,
    "object_pose_pass": obj_at is None and not diverged,
    "fell": fall_at is not None or diverged,
    "diverged": diverged,
    "seconds": len(body) * dt,
  }
  out["body_and_object_pose_pass"] = out["phc_pass"] and out["object_pose_pass"]
  fails = [i for i in (phc_at, obj_at) if i is not None]
  out["first_fail_s"] = float(ref_time_s[min(fails)]) if fails else None
  out["fall_s"] = float(ref_time_s[fall_at]) if fall_at is not None else None
  for name in DIAGNOSTICS:
    x = series[name]
    out |= {f"{name}/mean": float(x.mean()), f"{name}/p95": float(np.percentile(x, 95)),
            f"{name}/max": float(x.max()), f"{name}/final": float(x[-1])}
  return out


def summarize(clips: list[dict]) -> dict[str, float]:
  """``<group>/<stat>`` over clips: pass and fall rates, mean first-failure
  time of the clips that failed, and the clip mean of every diagnostic."""
  groups: dict[str, list[dict]] = {"all": clips}
  for c in clips:
    groups.setdefault(c["type"], []).append(c)
    groups.setdefault(c["kind"], []).append(c)
  out: dict[str, float] = {}
  for g, cs in groups.items():
    for k in ("phc_pass", "object_pose_pass", "body_and_object_pose_pass", "fell"):
      out[f"{g}/{k}_rate"] = float(np.mean([c[k] for c in cs]))
    failed = [c["first_fail_s"] for c in cs if c["first_fail_s"] is not None]
    if failed:
      out[f"{g}/first_fail_s"] = float(np.mean(failed))
    out[f"{g}/clips"] = float(len(cs))
    for name in DIAGNOSTICS:
      for s in ("mean", "p95", "max", "final"):
        out[f"{g}/{name}/{s}"] = float(np.mean([c[f"{name}/{s}"] for c in cs]))
  return out


class HoiEvaluator:
  """Plays the held-out library with the runner's EMA actor; see the module doc."""

  def __init__(self, runner: SuperMimicRunner, library: str, physics: str | None = None):
    """``physics``: the robot physics to evaluate on (default: the run's ``hoi_physics``);
    the policy keeps the action contract of the physics it was trained on."""
    from supermimic.representations import create_representation
    from supermimic.scripts.train_utils import create_env
    from supermimic.tasks.task_spec import make_motion_task

    self.runner = runner
    self.library = library
    train_lib = runner.motion_cmd.lib.root
    names = [v.name for v in read_variants(library)]
    if names != [v.name for v in read_variants(train_lib)]:
      raise ValueError(f"eval library {library} has object variants {names}, unlike training's")
    ends = np.load(Path(library) / "motion_ends.npy")
    device = runner.device
    # The eval env seeds the global RNGs; keep the training run's streams as they were.
    py_state, np_state = random.getstate(), np.random.get_state()
    cuda = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    trained = getattr(runner.cfg, "hoi_physics", "mjlab")
    contract = getattr(runner.cfg, "hoi_action_contract", None) or trained
    self.physics = physics or trained
    with torch.random.fork_rng(devices=cuda):
      self.env = create_env(hoi_eval_env_cfg(library, self.physics, contract), len(ends), device)
    random.setstate(py_state)
    np.random.set_state(np_state)
    raw = self.env.unwrapped
    self.cmd = raw.command_manager.get_term("motion")
    obs, _ = self.env.reset()
    rcfg = runner.cfg
    self.rep = create_representation(
      rcfg.representation, obs["full_state"].shape[-1], self.env.num_actions, raw.step_dt,
      raw.cfg.decimation, raw.scene["robot"].data.default_joint_pos[0].to(device), actor_obs_mode=rcfg.actor_obs,
    ).to(device)
    self.rep.bind(raw, runner._sim_context)
    self.task = make_motion_task(self.rep, runner._num_goals)
    robot = raw.scene["robot"]
    ix = robot.indexing
    self.num_bodies = len(robot.body_names)
    self.fk = ForwardKinematics(
      raw.sim.mj_model, ix.body_ids, ix.joint_q_adr, torch.arange(self.num_bodies), device=device
    )
    self.wrists = [robot.body_names.index(b) for b in WRIST_BODIES]
    probes, _ = hand_probes(device)
    self.hand_offset = probes.mean(dim=-2)  # (2, 3): each hand's probe centre, wrist frame
    limits = raw.sim.mj_model.actuator_forcerange[ix.ctrl_ids.cpu().numpy()]
    self.force_limit = torch.as_tensor(np.abs(limits).max(-1), dtype=torch.float32, device=device)
    self.dt = raw.step_dt
    starts = np.r_[0, ends[:-1]]
    self.clip_frame0 = torch.as_tensor(starts, device=device)
    self.max_steps = int((ends - starts).max()) + 2
    self.meta = self._clip_meta(library, len(ends))

  @staticmethod
  def _clip_meta(library: str, n: int) -> list[dict]:
    """Name, object type and hand/foot kind of each clip (kind from its contact labels)."""
    lib = Path(library)
    names = json.loads((lib / "motion_names.json").read_text())
    variants = read_variants(library)
    clip_variant = np.load(lib / "hoi" / "clip_variant.npy")
    ends = np.load(lib / "motion_ends.npy")
    foot = np.load(lib / "hoi" / "foot_contact.npy")
    out = []
    for c in range(n):
      s = 0 if c == 0 else ends[c - 1]
      kind = "foot" if int((foot[s : ends[c]] == 1).any(-1).sum()) >= 25 else "hand"
      out.append({"clip": names[c], "type": variants[clip_variant[c]].type, "kind": kind})
    return out

  def _actor(self):
    learner = self.runner.policy_learner
    ema = getattr(learner, "ema_actor", None)
    return learner.actor if ema is None else ema

  @torch.inference_mode()
  def run(self) -> dict:
    """Play every clip once; ``Eval/...`` scalars plus the per-clip table."""
    import wandb

    from supermimic.tasks.task_spec import build_actor_obs_td

    env, raw, cmd, rep = self.env, self.env.unwrapped, self.cmd, self.rep
    robot, obj = raw.scene["robot"], cmd.object
    actor = self._actor()
    actor.eval()
    ctx_cols = rep.context_columns
    obs, _ = env.reset()
    n = env.num_envs
    active = torch.ones(n, dtype=torch.bool, device=raw.device)
    diverged = torch.zeros_like(active)
    rec = {k: [] for k in (*DIAGNOSTICS, "pelvis_drop_m", "ref_time_s", "valid")}
    origins = raw.scene.env_origins
    for _ in range(self.max_steps):
      if not bool(active.any()):
        break
      full = obs["full_state"]
      rep_state, aux = rep.represent(full), rep.extract_aux_state(full)
      sim_ctx = self.runner._sim_context.read(raw)
      sensor = rep.compute_actor_sensor(rep_state, sim_ctx, aux_state=aux, sim_actor_obs=None, noisy=False)
      goal = self.task.extract_training_goal(cmd.motion, cmd.time_steps, aux)
      td = build_actor_obs_td(
        {"motion": goal}, sensor.unsqueeze(1), context=None if ctx_cols is None else sim_ctx[:, ctx_cols]
      )
      obs, _, dones, _ = env.step(actor(td))
      finite = torch.isfinite(robot.data.root_link_pos_w).all(-1) & torch.isfinite(obj.data.root_link_pos_w).all(-1)
      diverged |= active & ~finite
      valid = active & ~dones.bool() & finite  # a world that just ended was already reset
      self._record(rec, valid, robot, obj, origins)
      active &= valid
    return self._report(rec, diverged.cpu().numpy(), wandb)

  def _record(self, rec, valid, robot, obj, origins) -> None:
    motion, lib, t = self.cmd.motion, self.cmd.lib, self.cmd.time_steps
    zeros3 = torch.zeros_like(motion.root_pos_w[t])
    ref_pos, ref_quat, _, _ = self.fk(
      motion.root_pos_w[t], motion.root_quat_w[t], motion.joint_pos[t], zeros3, zeros3, torch.zeros_like(motion.joint_pos[t])
    )
    pos = robot.data.body_link_pos_w - origins[:, None, :]
    quat = robot.data.body_link_quat_w
    rec["body_err_m"].append((pos - ref_pos).norm(dim=-1).mean(-1))
    for side, w in zip(("left", "right"), self.wrists, strict=True):
      off = self.hand_offset[0 if side == "left" else 1].expand(len(t), 3)
      hand = pos[:, w] + quat_apply(quat[:, w], off)
      ref_hand = ref_pos[:, w] + quat_apply(ref_quat[:, w], off)
      rec[f"{side}_hand_err_m"].append((hand - ref_hand).norm(dim=-1))
    opos = obj.data.root_link_pos_w - origins
    rec["object_pos_err_m"].append((opos - lib.object_pos_w[t]).norm(dim=-1))
    dot = (obj.data.root_link_quat_w * lib.object_quat_w[t]).sum(-1).abs().clamp(max=1.0)
    rec["object_ori_err_deg"].append(torch.rad2deg(2 * torch.acos(dot)))
    rec["joint_rmse_rad"].append((robot.data.joint_pos - motion.joint_pos[t]).pow(2).mean(-1).sqrt())
    rec["action_saturation"].append(
      (robot.data.actuator_force.abs() >= SATURATION * self.force_limit).float().mean(-1)
    )
    rec["pelvis_drop_m"].append(ref_pos[:, 0, 2] - pos[:, 0, 2])
    clip = torch.searchsorted(motion.motion_ends, t, right=True)
    rec["ref_time_s"].append((t - self.clip_frame0[clip]).float() * self.dt)
    rec["valid"].append(valid)

  def _report(self, rec, diverged: np.ndarray, wandb) -> dict:
    arrays = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}  # (T, N)
    valid = arrays.pop("valid")
    clips = []
    for c, meta in enumerate(self.meta):
      m = valid[:, c]
      if not m.any():  # diverged on the first step
        series = {k: np.full(1, np.inf if k != "action_saturation" else 1.0) for k in (*DIAGNOSTICS, "pelvis_drop_m")}
        ref_t = np.zeros(1)
      else:
        series = {k: arrays[k][m, c] for k in (*DIAGNOSTICS, "pelvis_drop_m")}
        ref_t = arrays["ref_time_s"][m, c]
      clips.append({**meta, **score_clip(series, ref_t, self.dt, bool(diverged[c]))})
    out = {f"Eval/{k}": v for k, v in summarize(clips).items()}
    cols = ["clip", "type", "kind", "seconds", "phc_pass", "object_pose_pass", "body_and_object_pose_pass",
            "fell", "first_fail_s", "fall_s", "body_err_m/mean", "object_pos_err_m/mean", "object_ori_err_deg/mean",
            "left_hand_err_m/mean", "right_hand_err_m/mean", "joint_rmse_rad/mean", "action_saturation/mean"]
    out["Eval/clips_table"] = wandb.Table(columns=cols, data=[[c[k] for k in cols] for c in clips])
    return out
