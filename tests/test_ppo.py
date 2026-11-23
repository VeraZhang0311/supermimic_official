"""PPO fine-tuning learner (supermimic.learners.ppo)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.ppo import PpoCfg, PpoLearner, gae, gaussian_kl, gaussian_log_prob
from supermimic.models import _CudaGraphNormalization

OBS, ACT, N = 6, 3, 256


class TinyActor(nn.Module):
  """Deterministic actor over a TensorDict, with a running normalizer like SuperMimic's."""

  def __init__(self):
    super().__init__()
    self.norm = _CudaGraphNormalization(OBS)
    self.net = nn.Sequential(nn.Linear(OBS, 32), nn.Tanh(), nn.Linear(32, ACT))
    self.norm.train()
    self.norm.update(torch.randn(512, OBS) * 3 + 1)

  def forward(self, td: TensorDict) -> torch.Tensor:
    return self.net(self.norm(td["history"][:, -1]))


def rollout(learner: PpoLearner, advantage_fn) -> TensorDict:
  obs = torch.randn(N, 1, OBS)
  td = TensorDict({"history": obs}, batch_size=[N])
  actions, log_probs, means = learner.act(td)
  critic_obs = torch.randn(N, OBS)
  values = learner.value(critic_obs)
  adv = advantage_fn(actions, means)
  return TensorDict(
    {"actor_obs": td, "critic_obs": critic_obs, "actions": actions, "log_probs": log_probs, "means": means,
     "values": values, "advantages": adv, "returns": values + adv},
    batch_size=[N],
  )


def make(**kw) -> PpoLearner:
  torch.manual_seed(0)
  cfg = PpoCfg(**({"critic_warmup": 0, "num_minibatches": 4, "epochs": 2} | kw))
  critic = nn.Sequential(nn.Linear(OBS, 16), nn.Tanh(), nn.Linear(16, 1))
  return PpoLearner(TinyActor(), critic, ACT, cfg, "cpu")


def test_log_prob_and_kl_match_torch():
  mean, log_std = torch.randn(5, ACT), torch.randn(ACT) * 0.3
  actions = mean + torch.randn(5, ACT)
  ref = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1)
  torch.testing.assert_close(gaussian_log_prob(actions, mean, log_std), ref)
  mean2, log_std2 = torch.randn(5, ACT), torch.randn(ACT) * 0.3
  ref_kl = torch.distributions.kl_divergence(
    torch.distributions.Normal(mean, log_std.exp()), torch.distributions.Normal(mean2, log_std2.exp())
  ).sum(-1)
  torch.testing.assert_close(gaussian_kl(mean, log_std, mean2, log_std2), ref_kl)


def test_gae_cuts_at_dones():
  values = torch.tensor([[1.0], [2.0], [3.0]])
  rewards = torch.tensor([[0.5], [1.0], [2.0]])
  dones = torch.tensor([[False], [True], [False]])
  last = torch.tensor([4.0])
  gamma, lam = 0.9, 0.8
  adv, ret = gae(values, last, rewards, dones, gamma, lam)
  d2 = 2.0 + gamma * 4.0 - 3.0
  d1 = 1.0 - 2.0  # episode ends after step 1: no bootstrap from step 2
  d0 = 0.5 + gamma * 2.0 - 1.0
  expected = torch.tensor([[d0 + gamma * lam * d1], [d1], [d2]])
  torch.testing.assert_close(adv, expected)
  torch.testing.assert_close(ret, expected + values)


def test_critic_warmup_freezes_actor_and_std():
  learner = make(critic_warmup=2)
  actor0 = [p.detach().clone() for p in learner.actor.parameters()]
  critic0 = [p.detach().clone() for p in learner.critic.parameters()]
  std0 = learner.log_std.detach().clone()
  for _ in range(2):
    stats = learner.update(rollout(learner, lambda a, m: a[:, 0] - m[:, 0]))
    assert stats["PPO/actor_frozen"] == 1.0 and stats["PPO/actor_updates"] == 0.0
  assert all(torch.equal(a, b) for a, b in zip(actor0, learner.actor.parameters(), strict=True))
  assert torch.equal(std0, learner.log_std)
  assert not all(torch.equal(a, b) for a, b in zip(critic0, learner.critic.parameters(), strict=True))
  stats = learner.update(rollout(learner, lambda a, m: a[:, 0] - m[:, 0]))
  assert stats["PPO/actor_frozen"] == 0.0 and stats["PPO/actor_updates"] > 0


def test_actor_learns_the_advantage_and_keeps_normalizer_and_bounds():
  learner = make(actor_lr=3e-3, max_lr=3e-3, desired_kl=1.0, kl_stop=1e9, epochs=4)
  norm = {k: v.clone() for k, v in learner.actor.norm.state_dict().items()}
  probe = TensorDict({"history": torch.randn(512, 1, OBS)}, batch_size=[512])
  with torch.no_grad():
    before = learner.actor(probe)[:, 0].mean()
    ema_before = learner.ema_actor(probe)[:, 0].mean()
  for _ in range(15):
    learner.update(rollout(learner, lambda a, m: a[:, 0] - m[:, 0]))  # better: larger action 0
  with torch.no_grad():
    after = learner.actor(probe)[:, 0].mean()
    ema_after = learner.ema_actor(probe)[:, 0].mean()
  assert after > before + 0.05
  assert ema_before < ema_after < after  # the EMA follows, lagging
  for k, v in learner.actor.norm.state_dict().items():
    assert torch.equal(v, norm[k]), k
  std = learner.std
  assert (std >= learner.cfg.min_std - 1e-6).all() and (std <= learner.cfg.max_std + 1e-6).all()


def test_kl_stop_and_lr_adaptation():
  learner = make(actor_lr=1e-1, max_lr=1e-1, desired_kl=1e-6, kl_stop=1.0, epochs=3)
  stats = learner.update(rollout(learner, lambda a, m: torch.randn(N)))
  assert stats["PPO/kl_stopped"] == 1.0 and stats["PPO/actor_updates"] == 1.0
  assert learner.lr == 1e-1 / 1.5  # the KL far above target lowered it once
  assert all(g["lr"] == learner.lr for g in learner.actor_optimizer.param_groups)
  learner = make(actor_lr=1e-6, desired_kl=10.0, kl_stop=1e9)
  learner.update(rollout(learner, lambda a, m: torch.randn(N)))
  assert learner.lr == learner.cfg.max_lr or learner.lr > 1e-6  # tiny KL raises it, within bounds
  assert learner.lr <= learner.cfg.max_lr


def test_kl_is_measured_after_the_step():
  """The first minibatch of a rollout already moves the policy: its KL is not ~0."""
  learner = make(actor_lr=1e-2, max_lr=1e-2, desired_kl=1e-9, kl_stop=1e12, epochs=1, num_minibatches=1)
  stats = learner.update(rollout(learner, lambda a, m: a[:, 0] - m[:, 0]))
  assert stats["PPO/actor_updates"] == 1.0 and float(stats["PPO/kl"]) > 1e-6
  assert learner.lr < 1e-2  # lowered, not raised


def test_initial_std_and_state_round_trip():
  learner = make(init_std=0.08)
  torch.testing.assert_close(learner.std, torch.full((ACT,), 0.08))
  learner.update(rollout(learner, lambda a, m: a[:, 0] - m[:, 0]))
  state = learner.state_dict()
  other = make(init_std=0.08)
  other.load_state_dict(state)
  assert torch.equal(other.log_std, learner.log_std) and other.lr == learner.lr and other.iteration == 1
  assert math.isclose(other.actor_optimizer.param_groups[0]["lr"], learner.lr)
