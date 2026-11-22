"""Fengqiao's hw3 G1 physics (supermimic.hoi.physics_hw3): constants, motor law, robot spec, env switch."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1

from supermimic.hoi import physics_hw3 as hw3
from tests.hoi_fixtures import make_library

JOINTS = list(hw3.FRICTION_LEVELS)


def _mjlab_actuator(joint: str):
  import re

  hits = [
    a
    for a in g1.G1_ARTICULATION.actuators
    if any(re.fullmatch(e, joint) for e in a.target_names_expr)
  ]
  assert len(hits) == 1
  return hits[0]


class TestConstants:
  def test_every_g1_joint_has_one_motor(self):
    assert len(JOINTS) == 29
    for j in JOINTS:
      hw3.joint_motor(j)

  def test_gains_match_mjlab_except_the_hip_pitch(self):
    """Same motor tables and gain formula as mjlab's G1; only the hip pitch motor differs (7520-22 measured)."""
    for j in JOINTS:
      motor, effort = hw3.joint_motor(j)
      kp, kd = hw3.gains(motor)
      a = _mjlab_actuator(j)
      if j.endswith("hip_pitch_joint"):
        assert (motor, effort) == ("7520-22", 139.0)
        assert (
          a.stiffness == pytest.approx(g1.STIFFNESS_7520_14) and a.effort_limit == 88.0
        )
        assert kp == pytest.approx(g1.STIFFNESS_7520_22) and hw3.MOTORS[motor][
          0
        ] == pytest.approx(g1.ARMATURE_7520_22)
        continue
      assert kp == pytest.approx(a.stiffness, rel=1e-6), j
      assert kd == pytest.approx(a.damping, rel=1e-6), j
      assert hw3.MOTORS[motor][0] == pytest.approx(a.armature, rel=1e-6), j
      assert effort == pytest.approx(a.effort_limit), j

  def test_action_scale_is_his_contract(self):
    scale = hw3.action_scale()
    assert scale[".*_hip_pitch_joint"] == pytest.approx(
      0.25 * 139.0 / g1.STIFFNESS_7520_22
    )
    for expr, value in scale.items():
      if expr != ".*_hip_pitch_joint":
        assert value == pytest.approx(g1.G1_ACTION_SCALE[expr], rel=1e-6), expr

  def test_friction_static_is_the_slow_level_and_ranges(self):
    import re

    assert all(slow > 0 and fast > 0 for slow, fast in hw3.FRICTION_LEVELS.values())
    legs = [j for j in JOINTS if re.fullmatch(hw3.LEG_JOINTS, j)]
    other = [j for j in JOINTS if re.fullmatch(hw3.OTHER_JOINTS, j)]
    assert len(legs) == 12 and len(other) == 17 and not set(legs) & set(other)


def _cmd(target, pos, vel):
  def t(x):
    return torch.tensor([x], dtype=torch.float32)

  z = torch.zeros(1, len(target))
  return ActuatorCmd(
    position_target=t(target),
    velocity_target=z,
    effort_target=z,
    pos=t(pos),
    vel=t(vel),
  )


class TestMotorLaw:
  @staticmethod
  def params(n, motor="7520-22", effort=139.0, scale=1.0):
    armature, drive, brake, v_knee, v_max = hw3.MOTORS[motor]
    kp, kd = hw3.gains(motor)
    vals = {"stiffness": kp, "damping": kd, "force_limit": effort, "drive": drive, "brake": brake,
            "v_knee": v_knee, "v_max": v_max, "torque_scale": scale, "stall_band": hw3.STALL_BAND}  # fmt: skip
    return {k: torch.full((1, n), float(v)) for k, v in vals.items()}

  def test_small_error_is_plain_pd(self):
    p = self.params(1)
    tau = hw3.G1MotorActuator.control_law(p, _cmd([0.01], [0.0], [0.2]))
    kp, kd = hw3.gains("7520-22")
    assert tau.item() == pytest.approx(kp * 0.01 - kd * 0.2, rel=1e-5)

  def test_drive_brake_and_stall(self):
    """Torque along the motion is capped at the drive peak, against it at the brake peak, and near zero speed
    at the brake peak in both directions [motor_model.envelope_limit]."""
    p = self.params(3)
    big = 10.0  # rad of error: far beyond every cap
    tau = hw3.G1MotorActuator.control_law(
      p, _cmd([big, -big, big], [0, 0, 0], [1.0, 1.0, 0.005])
    )
    _, drive, brake, v_knee, _ = hw3.MOTORS["7520-22"]
    assert tau[0, 0].item() == pytest.approx(
      drive
    )  # torque + and velocity +: drive (111)
    assert tau[0, 1].item() == pytest.approx(
      -brake
    )  # torque - against velocity +: brake (131)
    assert tau[0, 2].item() == pytest.approx(brake)  # |v| in the stall band: brake

  def test_effort_limit_then_envelope_then_scale(self):
    # 2x5020: effort limit 50 < brake 63.8, so near zero speed the effort limit binds
    p = self.params(1, "2x5020", 50.0)
    assert hw3.G1MotorActuator.control_law(
      p, _cmd([10.0], [0.0], [0.0])
    ).item() == pytest.approx(50.0)
    p = self.params(1, "2x5020", 50.0, scale=0.7)  # 0.7 * 63.8 = 44.7 < 50
    assert hw3.G1MotorActuator.control_law(
      p, _cmd([10.0], [0.0], [0.0])
    ).item() == pytest.approx(0.7 * 63.8)

  def test_fade_to_zero_at_no_load_speed(self):
    _, drive, _, v_knee, v_max = hw3.MOTORS["7520-22"]
    p = self.params(3)
    vel = [v_knee, (v_knee + v_max) / 2, v_max + 1]
    tau = hw3.G1MotorActuator.control_law(p, _cmd([50.0] * 3, [0, 0, 0], vel))
    assert tau[0, 0].item() == pytest.approx(drive)
    assert tau[0, 1].item() == pytest.approx(drive / 2, rel=1e-4)
    assert tau[0, 2].item() == pytest.approx(0.0)


class TestRobot:
  @pytest.fixture(scope="class")
  def model(self):
    from supermimic.hoi.assets import g1_hoi_spec

    return g1_hoi_spec(hw3.Hw3Options()).compile()

  def test_joint_friction_is_the_static_level(self, model):
    for j, (slow, _fast) in hw3.FRICTION_LEVELS.items():
      dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
      assert model.dof_frictionloss[dof] == pytest.approx(slow), j

  def test_mid360_frame_is_his(self, model):
    """URDF rpy (0, 3.101, 3.1415) about fixed X, Y, Z: R = Rz(3.1415) Ry(3.101); his lidar looks down-ish."""
    s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mid360")
    q = model.site_quat[s]
    ry = np.array(
      [[np.cos(3.101), 0, np.sin(3.101)], [0, 1, 0], [-np.sin(3.101), 0, np.cos(3.101)]]
    )
    rz = np.array(
      [
        [np.cos(3.1415), -np.sin(3.1415), 0],
        [np.sin(3.1415), np.cos(3.1415), 0],
        [0, 0, 1],
      ]
    )
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    np.testing.assert_allclose(mat.reshape(3, 3), rz @ ry, atol=1e-6)
    np.testing.assert_allclose(model.site_pos[s], hw3.MID360["pos"])

  def test_no_friction_option(self):
    from supermimic.hoi.assets import g1_hoi_spec

    m = g1_hoi_spec(hw3.Hw3Options(friction=False)).compile()
    assert np.abs(m.dof_frictionloss).max() == 0.0

  def test_articulation_delay_and_scale(self):
    art = hw3.hw3_articulation()
    assert len(art.actuators) == len(hw3.G1_JOINTS)
    for a in art.actuators:
      assert (a.delay_min_lag, a.delay_max_lag, a.delay_hold_prob) == (2, 4, 1.0)
      assert (
        a.torque_scale_range == (0.9, 1.1)
        and a.viscous_damping == 0.0
        and a.frictionloss is None
      )
    nominal = hw3.hw3_articulation(hw3.Hw3Options.nominal())
    for a in nominal.actuators:
      assert (a.delay_min_lag, a.delay_max_lag) == (3, 3) and a.torque_scale_range == (
        1.0,
        1.0,
      )


class TestEnvSwitch:
  def test_hw3_training_env(self, tmp_path):
    from supermimic.hoi.env_cfg import (
      configure_hoi_env,
      configure_hoi_physics,
      hoi_env_cfg,
    )

    cfg = hoi_env_cfg()
    configure_hoi_env(cfg, str(make_library(tmp_path)))
    configure_hoi_physics(cfg, "hw3", push=True)
    acts = cfg.scene.entities["robot"].articulation.actuators
    assert all(isinstance(a, hw3.G1MotorActuatorCfg) for a in acts)
    assert {"hw3_joint_friction_legs", "hw3_joint_friction_other", "push_robot"} <= set(
      cfg.events
    )
    assert (
      cfg.events["hw3_joint_friction_legs"].params["ranges"] == hw3.LEG_FRICTION_SCALE
    )
    assert cfg.actions["joint_pos"].scale == hw3.action_scale()
    # the recipe's other terms stay the env's own
    assert {"foot_friction", "base_com", "encoder_bias"} <= set(cfg.events)

  def test_zero_shot_eval_keeps_the_policy_contract(self, tmp_path):
    from supermimic.hoi.evaluate import hoi_eval_env_cfg

    cfg = hoi_eval_env_cfg(str(make_library(tmp_path)), "hw3", "mjlab")
    acts = cfg.scene.entities["robot"].articulation.actuators
    assert all(
      isinstance(a, hw3.G1MotorActuatorCfg) and a.torque_scale_range == (1.0, 1.0)
      for a in acts
    )
    assert all((a.delay_min_lag, a.delay_max_lag) == (3, 3) for a in acts)
    assert not any(k.startswith("hw3_") or k == "push_robot" for k in cfg.events)
    assert cfg.actions["joint_pos"].scale == g1.G1_ACTION_SCALE

  def test_mjlab_default_is_unchanged(self, tmp_path):
    from supermimic.hoi.evaluate import hoi_eval_env_cfg

    cfg = hoi_eval_env_cfg(str(make_library(tmp_path)))
    assert cfg.scene.entities["robot"].articulation is g1.G1_ARTICULATION
    assert cfg.actions["joint_pos"].scale == g1.G1_ACTION_SCALE

  def test_bad_choices(self, tmp_path):
    from supermimic.hoi.env_cfg import configure_hoi_physics, hoi_env_cfg

    with pytest.raises(ValueError):
      configure_hoi_physics(hoi_env_cfg(), "isaac")
    with pytest.raises(ValueError):
      configure_hoi_physics(hoi_env_cfg(), "mjlab", push=True)
