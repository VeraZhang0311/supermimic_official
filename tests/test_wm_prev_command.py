"""The world model's in-flight-command input, widened weight loading and the hw3 sim-context terms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from supermimic.hoi import physics_hw3 as hw3
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics
from supermimic.representations import create_representation
from supermimic.runners.supermimic_runner import load_widened
from supermimic.sim_context import (
  JointFrictionSpec,
  MotorDelaySpec,
  MotorTorqueScaleSpec,
  SimContext,
)

NUM_JOINTS, NUM_ACTIONS = 20, 4


def _rep():
  return create_representation(
    "gencoord", 13 + 2 * NUM_JOINTS, NUM_ACTIONS, dt=0.02, decimation=4,
    default_joint_pos=torch.zeros(NUM_JOINTS), actor_obs_mode="rep_state",
  )  # fmt: skip


def _dyn(rep, ctx: int, prev: bool, seed: int = 0) -> Dynamics:
  torch.manual_seed(seed)
  return Dynamics(rep, NUM_ACTIONS, ctx, torch.full((NUM_ACTIONS,), 0.5), torch.ones(NUM_ACTIONS),
                  hidden_dims=(16, 16), prev_command=prev)  # fmt: skip


def _inputs(rep, batch=3, seed=1):
  g = torch.Generator().manual_seed(seed)
  rep_state = torch.randn(batch, rep.rep_state_dim, generator=g)
  aux = torch.cat(
    [
      torch.randn(batch, 3, generator=g),
      torch.tensor([[1.0, 0, 0, 0]]).repeat(batch, 1),
    ],
    -1,
  )
  return rep_state, torch.randn(batch, NUM_ACTIONS, generator=g), aux


class TestDynamics:
  def test_without_prev_command_the_layout_is_unchanged(self):
    rep = _rep()
    d = _dyn(rep, 2, prev=False)
    assert d.mlp[0].in_features == rep.rep_state_dim + NUM_ACTIONS + 2
    s, a, aux = _inputs(rep)
    ctx = torch.zeros(3, 2)
    out = d(s, a, aux, ctx)
    out2 = d(s, a, aux, ctx, prev_cmd=torch.randn(3, NUM_ACTIONS))  # ignored
    assert all(torch.equal(x, y) for x, y in zip(out, out2, strict=True))

  def test_prev_command_is_an_input(self):
    rep = _rep()
    d = _dyn(rep, 2, prev=True)
    assert d.mlp[0].in_features == rep.rep_state_dim + 2 * NUM_ACTIONS + 2
    s, a, aux = _inputs(rep)
    ctx = torch.zeros(3, 2)
    base = d(s, a, aux, ctx, prev_cmd=d.proc(a))[1]
    other = d(s, a, aux, ctx, prev_cmd=d.proc(a) + 1.0)[1]
    assert not torch.allclose(base, other)
    assert torch.equal(base, d(s, a, aux, ctx)[1])  # None = no command change


class TestWidenedLoad:
  def test_new_columns_start_at_zero_weight(self):
    """A model trained without the hw3 inputs predicts the same after widening,
    whatever the new inputs are."""
    rep = _rep()
    old = _dyn(rep, 2, prev=False, seed=0)
    new = _dyn(rep, 5, prev=True, seed=1)
    widened = load_widened(new, old.state_dict())
    assert widened == ["mlp.0.weight"]
    s, a, aux = _inputs(rep)
    ctx_old = torch.randn(3, 2)
    ctx_new = torch.cat([ctx_old, torch.randn(3, 3)], -1)
    ref = old(s, a, aux, ctx_old)[1]
    got = new(s, a, aux, ctx_new, prev_cmd=torch.randn(3, NUM_ACTIONS))[1]
    torch.testing.assert_close(got, ref)

  def test_refuses_other_shapes(self):
    rep = _rep()
    with pytest.raises(ValueError):
      load_widened(
        _dyn(rep, 2, prev=False), _dyn(rep, 5, prev=False).state_dict()
      )  # narrower
    bad = _dyn(rep, 2, prev=False).state_dict()
    bad.pop("action_scale")
    with pytest.raises(ValueError):
      load_widened(_dyn(rep, 2, prev=False), bad)


class _Spy:
  """Stands in for Dynamics in a rollout: records each step's prev_cmd."""

  prev_command = True

  def __init__(self, rep_dim):
    self.calls, self.rep_dim = [], rep_dim

  def proc(self, a):
    return 10.0 * a

  def __call__(self, rep_state, action, aux, sim_ctx, prev_cmd=None):
    self.calls.append(prev_cmd)
    return aux, rep_state


def test_rollout_chains_the_in_flight_command():
  rep = _rep()
  learner = WorldModelLearner(
    _dyn(rep, 1, prev=True), rep, WorldModelLearnerCfg(horizon=4), device="cpu"
  )
  spy = _Spy(rep.rep_state_dim)
  learner.dynamics = spy
  s, _, aux = _inputs(rep, batch=2)
  actions = torch.randn(2, 4, NUM_ACTIONS)
  first = torch.randn(2, NUM_ACTIONS)
  learner._rollout(s, actions, aux, torch.zeros(2, 1), first)
  assert len(spy.calls) == 3
  assert torch.equal(spy.calls[0], first)
  for k in (1, 2):
    assert torch.equal(spy.calls[k], 10.0 * actions[:, k - 1])


def test_world_model_env_tracks_the_in_flight_command():
  from supermimic.envs.world_model_env import WorldModelEnv
  from tests.conftest import (
    make_mock_buffer,
    make_mock_command_env,
    make_mock_dynamics,
    make_mock_wm_representation,
  )
  from tests.test_world_model_env import _specs

  dyn = make_mock_dynamics()
  dyn.prev_command = True
  dyn.proc = lambda a: a * 2.0 + 1.0
  env = WorldModelEnv(dynamics=dyn, representation=make_mock_wm_representation(8), env=make_mock_command_env(),
                      batch_size=8, replay_buffer=make_mock_buffer(), task_specs=_specs(), history_length=1,
                      termination_softness=0.02, device="cpu")  # fmt: skip
  batch = make_mock_buffer().sample(1, 8).squeeze(1)
  batch["prev_cmd"] = torch.randn(8, 29)
  env.reset(batch)
  a1, a2 = torch.randn(8, 29), torch.randn(8, 29)
  env.step(a1)
  assert torch.equal(dyn.call_args.args[4], batch["prev_cmd"])
  env.step(a2)
  assert torch.equal(dyn.call_args.args[4], a1 * 2.0 + 1.0)


def _fake_env(num_envs=3):
  """An env with the hw3 motor group, friction events and model fields the specs read."""
  joints = list(hw3.FRICTION_LEVELS)
  acts = [
    SimpleNamespace(cfg=SimpleNamespace(torque_scale_range=(0.9, 1.1)))
    for _ in range(2)
  ]
  group = SimpleNamespace(
    actuator_type=hw3.G1MotorActuator, min_lag=2, max_lag=4, absorbed_actuators=acts,
    target_ids=torch.arange(len(joints)),
    delay_buffer=SimpleNamespace(current_lags=torch.tensor([2, 3, 4])),
    params={"torque_scale": torch.full((num_envs, len(joints)), 1.05)},
  )  # fmt: skip
  nominal = torch.tensor([hw3.FRICTION_LEVELS[j][0] for j in joints])
  dof_adr = torch.arange(6, 6 + len(joints))
  fl = torch.zeros(num_envs, 6 + len(joints))
  fl[:, dof_adr] = nominal * 1.5
  robot = SimpleNamespace(joint_names=joints, indexing=SimpleNamespace(joint_v_adr=dof_adr),
                          _fused_actuator_group=SimpleNamespace(_groups=[group]))  # fmt: skip
  return SimpleNamespace(
    scene={"robot": robot}, cfg=SimpleNamespace(events=hw3.friction_scale_events()), device="cpu",
    num_envs=num_envs, sim=SimpleNamespace(model=SimpleNamespace(dof_frictionloss=fl)),
  )  # fmt: skip


class TestHw3SimContext:
  def test_specs_bind_and_read(self):
    env = _fake_env()
    delay, scale, friction = (
      MotorDelaySpec(),
      MotorTorqueScaleSpec(),
      JointFrictionSpec(),
    )
    assert delay.bind(env) == [(2.0, 4.0)]
    torch.testing.assert_close(delay.read(env), torch.tensor([[2.0], [3.0], [4.0]]))
    assert scale.bind(env) == [(0.9, 1.1)] * 29 and scale.read(env).shape == (3, 29)
    ranges = friction.bind(env)
    assert ranges[0] == hw3.LEG_FRICTION_SCALE and ranges[-1] == hw3.FRICTION_SCALE
    torch.testing.assert_close(friction.read(env), torch.full((3, 29), 1.5))

  def test_unrandomized_friction_field(self):
    env = _fake_env()
    spec = JointFrictionSpec()
    spec.bind(env)
    env.sim.model.dof_frictionloss = env.sim.model.dof_frictionloss[
      0
    ]  # a model field with no world axis
    assert spec.read(env).shape == (3, 29)

  def test_from_env_appends_hw3_terms_last(self):
    ctx = SimContext.from_env(_fake_env())
    assert [s.name for s in ctx.specs] == [
      "motor_delay",
      "motor_torque_scale",
      "joint_friction",
    ]
    assert ctx.output_dim == 1 + 29 + 29
    out = ctx.read(_fake_env())
    assert out.shape == (3, 59) and torch.isfinite(out).all()


def test_replay_window_keeps_prev_cmd():
  from supermimic.replay_buffer import FieldSpec, ReplayBuffer

  buf = ReplayBuffer({"terminated": FieldSpec(dtype=torch.bool), "truncated": FieldSpec(dtype=torch.bool),
                      "prev_cmd": FieldSpec(NUM_ACTIONS)}, buffer_size=8, num_envs=2)  # fmt: skip
  for t in range(5):
    buf.insert(TensorDict({"terminated": torch.zeros(2, dtype=torch.bool), "truncated": torch.zeros(2, dtype=torch.bool),
                           "prev_cmd": torch.full((2, NUM_ACTIONS), float(t))}, batch_size=[2]))  # fmt: skip
  window = buf.sample(3, 4)
  assert window["prev_cmd"].shape == (4, 3, NUM_ACTIONS)
