"""Fengqiao's "hw3" G1 physics, ported from his Isaac Lab env to mjlab / MuJoCo Warp.

Source: gear_sonic (his fork of GR00T-WholeBodyControl), commit 4aed2bb, delivered 2026-11-05 with
INTERN_README.md; every policy of that repo trains and evaluates in it. Ported here (his module in brackets):

- Motors [robots/g1_hw3.py, mdp/actuators.py ``G1MotorActuator``, mdp/motor_model.py]: explicit PD at every 5 ms
  physics step, the torque clipped to the joint's effort limit and then to Unitree's torque-speed envelope of its
  motor type: the drive peak when torque and velocity have the same sign, the brake peak against the motion and
  within |v| <= 0.01 rad/s, fading linearly from v_knee to zero at v_max; the envelope times a torque scale
  U(0.9, 1.1) per env and joint, drawn at every reset. Gains, armature and effort limits per motor type are his:
  kp = armature * w^2, kd = 4 * armature * w (w = 2 pi 10 Hz), as mjlab's G1, but the hip pitch is a 7520-22 (he
  measured it on the lab's G1; mjlab's G1 has a 7520-14: kp 99.1 instead of 40.2, armature 0.0251 instead of
  0.0102, effort limit 139 instead of 88 N m).
- Command delay [G1MotorActuatorCfg.min/max_delay, mdp/events.py ``share_actuator_delay``]: 2-4 physics steps
  (10-20 ms), one draw per env at every reset, the same for every joint.
- Joint friction [mdp/friction_hw3.py]: measured per joint, viscous 0. PhysX has a static (slow-speed) and a dynamic
  (fast-speed) level; MuJoCo has one dry-friction constraint per joint (``frictionloss``, holding and sliding), which
  gets the static level: the fast level is at most 0.27 N m lower, inside the randomization. Scaled per env and joint
  at startup by U(0.8, 2.0), the legs by U(0.8, 2.5).
- Action scale [robots/g1.py ``G1_MODEL_12_ACTION_SCALE``]: 0.25 * effort limit / stiffness per joint.
- Robot definition [urdf/g1/main_rubber_hand_meshcol_handcol_softfinger.urdf]: every link's mass, centre of mass and
  inertia and every joint's frame, axis and range equal mjlab's G1 MJCF once his fixed links are merged into their
  parents (pelvis_contour, logo, head, imu, mid360, LL_FOOT, LR_FOOT; total 33.341 kg); mjlab already has his
  foot and IMU frames as sites, and his lidar frame is added (``mid360``).

Not ported: his collision geometry and soft-finger contact material, the torso / wrist mass randomization (not
wanted), PhysX's joint velocity limits (MuJoCo has none; the envelope already takes the torque to zero at v_max) and
the restitution randomization (MuJoCo contacts have no restitution coefficient).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque
from mjlab.entity import EntityArticulationInfoCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz, as his and mjlab's G1 constants
DAMPING_RATIO = 2.0

# Motor type: (armature kg m^2, drive N m, brake N m, v_knee rad/s, v_max rad/s) [robots/g1.py, motor_model.py].
# The ankles and the waist roll / pitch are linkages of two 5020s: twice one 5020's armature and torques.
MOTORS: dict[str, tuple[float, float, float, float, float]] = {
  "7520-22": (0.025101925, 111.0, 131.0, 14.5, 22.7),
  "7520-14": (0.010177520, 71.0, 83.3, 22.63, 35.52),
  "2x5020": (2 * 0.003609725, 49.6, 63.8, 30.86, 40.13),
  "5020": (0.003609725, 24.8, 31.9, 30.86, 40.13),
  "4010": (0.00425, 4.8, 8.6, 15.3, 24.76),
}
# joint regex -> (motor type, effort limit N m) [g1.py G1_CYLINDER_MODEL_12_DEX_CFG, motor_model.G1_JOINT_MOTORS]
G1_JOINTS: dict[str, tuple[str, float]] = {
  ".*_hip_pitch_joint": ("7520-22", 139.0),
  ".*_hip_roll_joint": ("7520-22", 139.0),
  ".*_knee_joint": ("7520-22", 139.0),
  ".*_hip_yaw_joint": ("7520-14", 88.0),
  "waist_yaw_joint": ("7520-14", 88.0),
  ".*_ankle_pitch_joint": ("2x5020", 50.0),
  ".*_ankle_roll_joint": ("2x5020", 50.0),
  "waist_roll_joint": ("2x5020", 50.0),
  "waist_pitch_joint": ("2x5020", 50.0),
  ".*_shoulder_pitch_joint": ("5020", 25.0),
  ".*_shoulder_roll_joint": ("5020", 25.0),
  ".*_shoulder_yaw_joint": ("5020", 25.0),
  ".*_elbow_joint": ("5020", 25.0),
  ".*_wrist_roll_joint": ("5020", 25.0),
  ".*_wrist_pitch_joint": ("4010", 5.0),
  ".*_wrist_yaw_joint": ("4010", 5.0),
}
STALL_BAND = 0.01
"""|v| (rad/s) below which the brake peak applies in both directions."""

# joint: (slow-speed level, fast-speed level) N m [friction_hw3.G1_FRICTION_LEVELS_HW3]
FRICTION_LEVELS: dict[str, tuple[float, float]] = {
  "left_hip_pitch_joint": (0.893, 0.747),
  "left_hip_roll_joint": (0.686, 0.630),
  "left_hip_yaw_joint": (1.079, 0.879),
  "left_knee_joint": (0.978, 0.713),
  "left_ankle_pitch_joint": (0.729, 0.612),
  "left_ankle_roll_joint": (0.468, 0.385),
  "right_hip_pitch_joint": (0.809, 0.769),
  "right_hip_roll_joint": (0.568, 0.577),
  "right_hip_yaw_joint": (0.963, 0.911),
  "right_knee_joint": (0.848, 0.700),
  "right_ankle_pitch_joint": (0.759, 0.623),
  "right_ankle_roll_joint": (0.529, 0.438),
  "waist_yaw_joint": (0.462, 0.443),
  "waist_roll_joint": (0.869, 0.905),
  "waist_pitch_joint": (
    0.869,
    0.905,
  ),  # not identifiable while hanging: waist roll's (same linkage)
  "left_shoulder_pitch_joint": (0.324, 0.277),
  "left_shoulder_roll_joint": (0.453, 0.395),
  "left_shoulder_yaw_joint": (0.296, 0.271),
  "left_elbow_joint": (0.412, 0.348),
  "left_wrist_roll_joint": (0.395, 0.297),
  "left_wrist_pitch_joint": (0.177, 0.170),
  "left_wrist_yaw_joint": (0.163, 0.162),
  "right_shoulder_pitch_joint": (0.368, 0.304),
  "right_shoulder_roll_joint": (0.461, 0.379),
  "right_shoulder_yaw_joint": (0.258, 0.212),
  "right_elbow_joint": (0.444, 0.367),
  "right_wrist_roll_joint": (0.322, 0.247),
  "right_wrist_pitch_joint": (0.204, 0.199),
  "right_wrist_yaw_joint": (0.134, 0.137),
}
DELAY_STEPS = (2, 4)
"""Command delay, physics steps of 5 ms; one draw per env at every reset."""
NOMINAL_DELAY_STEPS = 3
"""The delay of nominal (evaluation) physics: the middle of DELAY_STEPS, 15 ms."""
TORQUE_SCALE = (0.9, 1.1)
FRICTION_SCALE = (0.8, 2.0)
LEG_FRICTION_SCALE = (0.8, 2.5)
LEG_JOINTS = r".*_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))_joint"
OTHER_JOINTS = (
  r"(waist_.*|.*_(shoulder_(pitch|roll|yaw)|elbow|wrist_(roll|pitch|yaw)))_joint"
)
PUSH_INTERVAL_S = (4.0, 6.0)
PUSH_VELOCITY = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                 "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}  # fmt: skip
MID360 = {"pos": (0.0002835, 0.00003, 0.41618), "rpy": (0.0, 3.101, 3.1415)}
"""His lidar frame on torso_link (mid360_joint origin of his URDF)."""


def gains(motor: str) -> tuple[float, float]:
  armature = MOTORS[motor][0]
  return armature * NATURAL_FREQ**2, 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def action_scale() -> dict[str, float]:
  """His action contract: 0.25 * effort limit / stiffness per joint regex."""
  return {
    expr: 0.25 * effort / gains(motor)[0] for expr, (motor, effort) in G1_JOINTS.items()
  }


def joint_motor(joint: str) -> tuple[str, float]:
  hits = [v for expr, v in G1_JOINTS.items() if re.fullmatch(expr, joint)]
  if len(hits) != 1:
    raise ValueError(f"{joint} matches {len(hits)} motor entries")
  return hits[0]


def envelope_clip(effort, vel, drive, brake, v_knee, v_max, torque_scale, stall_band):
  """Clip a torque to the torque-speed envelope [motor_model.envelope_limit / clip_to_envelope]."""
  peak = torch.where(effort * vel > 0.0, drive, brake)
  peak = torch.where(vel.abs() <= stall_band, brake, peak)
  fade = ((v_max - vel.abs()) / (v_max - v_knee).clamp_min(1e-6)).clamp(0.0, 1.0)
  limit = peak * fade * torque_scale
  return torch.maximum(torch.minimum(effort, limit), -limit)


@dataclass(kw_only=True)
class G1MotorActuatorCfg(IdealPdActuatorCfg):
  """Explicit PD with the effort clip, the torque-speed envelope and its per-reset torque scale."""

  drive: float
  """Peak torque with the torque and the velocity in the same direction (N m)."""
  brake: float
  """Peak torque against the motion, and near zero speed (N m)."""
  v_knee: float
  """Speed up to which the peak torque is available (rad/s)."""
  v_max: float
  """No-load speed: the limit fades linearly to zero between v_knee and v_max (rad/s)."""
  stall_band: float = STALL_BAND
  torque_scale_range: tuple[float, float] = (1.0, 1.0)
  """Envelope scale, one U(lo, hi) draw per env and joint at every reset."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> G1MotorActuator:
    return G1MotorActuator(self, entity, target_ids, target_names)


class G1MotorActuator(IdealPdActuator[G1MotorActuatorCfg]):
  """IdealPdActuator with his motor model. It keeps the shared stateless-law compute, so mjlab fuses every group of
  it into one batched control law behind one delay buffer: the delay is shared by all joints."""

  param_names = (
    *IdealPdActuator.param_names,
    "drive",
    "brake",
    "v_knee",
    "v_max",
    "torque_scale",
    "stall_band",
  )

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    torque = pd_torque(params["stiffness"], params["damping"], cmd)
    torque = torch.clamp(torque, -params["force_limit"], params["force_limit"])
    return envelope_clip(torque, cmd.vel, params["drive"], params["brake"], params["v_knee"], params["v_max"],
                         params["torque_scale"], params["stall_band"])  # fmt: skip

  def __init__(self, cfg, entity, target_ids, target_names) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    for name in ("drive", "brake", "v_knee", "v_max", "torque_scale", "stall_band"):
      setattr(self, name, None)

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    super().initialize(mj_model, model, data, device)
    shape = (data.nworld, len(self._target_names))
    for name in ("drive", "brake", "v_knee", "v_max", "stall_band"):
      setattr(
        self,
        name,
        torch.full(
          shape, float(getattr(self.cfg, name)), dtype=torch.float, device=device
        ),
      )
    self.torque_scale = torch.ones(shape, dtype=torch.float, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    buf = self._delay_buffer
    if buf is not None and buf.max_lag > buf.min_lag:
      # Draw the episode's delay now rather than at the next physics step, so that the
      # sim context read right after a reset sees it. The buffer is shared by the fused
      # group, so the last group's draw holds (hold_prob 1 keeps it to the next reset).
      n = buf.batch_size if isinstance(ids, slice) else len(ids)
      lags = torch.randint(buf.min_lag, buf.max_lag + 1, (n,), device=buf.device)
      buf.set_lags(lags, ids)
    lo, hi = self.cfg.torque_scale_range
    if self.torque_scale is None or lo == hi:
      return
    # In place: after fusion the tensor is a view into the fused group's.
    self.torque_scale[ids] = lo + (hi - lo) * torch.rand_like(self.torque_scale[ids])


@dataclass
class Hw3Options:
  """Which parts of hw3 are on (all of them for training; ``nominal()`` for evaluation)."""

  delay_steps: tuple[int, int] = DELAY_STEPS
  torque_scale: tuple[float, float] = TORQUE_SCALE
  friction: bool = True
  """Measured joint friction (False: none, as mjlab's G1)."""

  @staticmethod
  def nominal() -> Hw3Options:
    return Hw3Options(
      delay_steps=(NOMINAL_DELAY_STEPS, NOMINAL_DELAY_STEPS), torque_scale=(1.0, 1.0)
    )


def hw3_articulation(options: Hw3Options | None = None) -> EntityArticulationInfoCfg:
  """One G1MotorActuatorCfg per joint regex (his gains, armature, effort limit and envelope); the delay with a hold
  probability of 1: drawn once after each reset (the reset clears it), then kept."""
  o = options or Hw3Options()
  lo, hi = o.delay_steps
  acts = []
  for expr, (motor, effort) in G1_JOINTS.items():
    armature, drive, brake, v_knee, v_max = MOTORS[motor]
    kp, kd = gains(motor)
    acts.append(
      G1MotorActuatorCfg(
        target_names_expr=(expr,),
        stiffness=kp,
        damping=kd,
        effort_limit=effort,
        armature=armature,
        frictionloss=None,
        viscous_damping=0.0,
        drive=drive,
        brake=brake,
        v_knee=v_knee,
        v_max=v_max,
        torque_scale_range=tuple(o.torque_scale),
        delay_min_lag=int(lo),
        delay_max_lag=int(hi),
        delay_hold_prob=1.0 if hi > 0 else 0.0,
      )  # fmt: skip
    )
  return EntityArticulationInfoCfg(
    actuators=tuple(acts), soft_joint_pos_limit_factor=0.9
  )


def apply_hw3_spec(spec: mujoco.MjSpec, options: Hw3Options | None = None) -> None:
  """His joint friction (static level) on every joint and his lidar frame as a site."""
  o = options or Hw3Options()
  for joint, (static, _fast) in FRICTION_LEVELS.items():
    spec.joint(joint).frictionloss = static if o.friction else 0.0
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(
    quat, np.asarray(MID360["rpy"]), "XYZ"
  )  # URDF rpy: fixed-axis X, then Y, then Z
  spec.body("torso_link").add_site(
    name="mid360", pos=MID360["pos"], quat=quat.tolist(), size=[0.01, 0, 0]
  )


def friction_scale_events(asset: str = "robot") -> dict:
  """Startup events: each env's joint frictionloss times one draw per joint, legs U(0.8, 2.5), the rest U(0.8, 2.0)."""
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  return {
    f"hw3_joint_friction_{name}": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg(asset, joint_names=(regex,)),
        "operation": "scale",
        "ranges": rng,
      },
    )
    for name, regex, rng in (
      ("legs", LEG_JOINTS, LEG_FRICTION_SCALE),
      ("other", OTHER_JOINTS, FRICTION_SCALE),
    )
  }


def push_event():
  """His optional push: base velocity set within PUSH_VELOCITY every 4-6 s (off in his example)."""
  from mjlab.envs import mdp
  from mjlab.managers.event_manager import EventTermCfg

  return EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=PUSH_INTERVAL_S,
                      params={"velocity_range": dict(PUSH_VELOCITY)})  # fmt: skip
