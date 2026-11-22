"""Trainer state (SuperMimicRunner): a restored run continues exactly where the
saved one stood."""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from supermimic.learners.learner_utils import build_optimizers
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.runners.supermimic_runner import TRAINER_STATE, SuperMimicRunner


class _WorldModel:
  """Stands in for WorldModel: one network behind bundle / load_bundle."""

  def __init__(self, net: nn.Module) -> None:
    self.net = net

  def bundle(self) -> dict:
    return {"dynamics": self.net.state_dict()}

  def load_bundle(self, bundle: dict) -> None:
    self.net.load_state_dict(bundle["dynamics"])


def _mlp() -> nn.Module:
  # Three Linears: Muon takes the middle one, Adam the rest.
  return nn.Sequential(
    nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2)
  )


def _runner(seed: int) -> SimpleNamespace:
  torch.manual_seed(seed)
  wm, actor, critic = _mlp(), _mlp(), _mlp()
  runner = SimpleNamespace(
    _wm=_WorldModel(wm),
    world_model_learner=SimpleNamespace(optimizers=build_optimizers([wm], 1e-3)),
    policy_learner=SimpleNamespace(
      actor=actor,
      actor_optimizers=build_optimizers([actor], 1e-3),
      ema_actor=_mlp(),
      critic=critic,
      critic_optimizers=build_optimizers([critic], 1e-3),
    ),
    _sampler=ReturnSampler(ReturnSamplerCfg(horizon=1), 4, 40, 2, max_step_reward=1.0),
    logger=SimpleNamespace(tot_timesteps=0, tot_time=0.0),
  )
  runner._optimizers = types.MethodType(SuperMimicRunner._optimizers, runner)
  return runner


def _networks(runner: SimpleNamespace) -> dict[str, nn.Module]:
  learner = runner.policy_learner
  return {
    "world_model": runner._wm.net,
    "actor": learner.actor,
    "critic": learner.critic,
  }


def _train(runner: SimpleNamespace, steps: int, seed: int) -> None:
  """Optimizer steps on every network, filling the Adam and Muon state."""
  torch.manual_seed(seed)
  nets = _networks(runner)
  for _ in range(steps):
    for name, opts in runner._optimizers().items():
      loss = nets[name](torch.randn(8, 4)).square().mean()
      for opt in opts:
        opt.zero_grad()
      loss.backward()
      for opt in opts:
        opt.step()


def _record(runner: SimpleNamespace, frame: int, reward: float) -> None:
  n = 2
  runner._sampler.record(
    torch.full((n,), frame),
    torch.full((n,), reward),
    torch.zeros(n, dtype=torch.bool),
    torch.zeros(n, dtype=torch.bool),
    torch.zeros(n, dtype=torch.bool),
  )


def test_restore_continues_exactly(tmp_path) -> None:
  saved = _runner(seed=0)
  _train(saved, steps=3, seed=10)
  _record(saved, 25, 0.3)
  _record(saved, 5, 0.8)
  saved.logger.tot_timesteps, saved.logger.tot_time = 123, 4.5
  state = SuperMimicRunner._trainer_state(saved, step=7, log_count=1, wm_updates=5)
  torch.save(state, tmp_path / TRAINER_STATE)

  resumed = _runner(seed=1)  # other weights, no optimizer state yet
  first = SuperMimicRunner._restore(resumed, torch.load(tmp_path / TRAINER_STATE))
  assert first == (8, 1, 5)
  assert (resumed.logger.tot_timesteps, resumed.logger.tot_time) == (123, 4.5)
  assert torch.equal(resumed._sampler.value, saved._sampler.value)
  for a, b in zip(
    saved.policy_learner.ema_actor.parameters(),
    resumed.policy_learner.ema_actor.parameters(),
    strict=True,
  ):
    assert torch.equal(a, b)

  # Same data from here on: equal weights only if the optimizer states came back.
  _train(saved, steps=2, seed=20)
  _train(resumed, steps=2, seed=20)
  for name, net in _networks(saved).items():
    other = _networks(resumed)[name]
    for a, b in zip(net.parameters(), other.parameters(), strict=True):
      assert torch.equal(a, b), name


def test_restore_without_optimizer_state_starts_them_fresh(tmp_path) -> None:
  saved = _runner(seed=0)
  _train(saved, steps=1, seed=10)
  state = SuperMimicRunner._trainer_state(saved, step=0, log_count=1, wm_updates=1)
  del state["optimizers"], state["sampler"]
  resumed = _runner(seed=1)
  SuperMimicRunner._restore(resumed, state)
  assert all(not opt.state for opts in resumed._optimizers().values() for opt in opts)
  for a, b in zip(
    saved.policy_learner.actor.parameters(),
    resumed.policy_learner.actor.parameters(),
    strict=True,
  ):
    assert torch.equal(a, b)


def test_log_dir_with_a_checkpoint_but_no_trainer_state_is_refused(tmp_path) -> None:
  (tmp_path / "actor.pt").write_bytes(b"")
  runner = SimpleNamespace(cfg=SimpleNamespace(log_dir=str(tmp_path)), device="cpu")
  with pytest.raises(FileExistsError):
    SuperMimicRunner._open_log_dir(runner)
  fresh = tmp_path / "new"
  runner.cfg.log_dir = str(fresh)
  assert SuperMimicRunner._open_log_dir(runner) == (fresh, None)
  assert fresh.is_dir()
