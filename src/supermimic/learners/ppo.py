"""PPO fine-tuning of a SuperMimic actor in the simulator.

The SuperMimic actor is deterministic (its gradient comes through the world model).
PPO makes it the mean of a Gaussian with a state-independent per-joint std in the
policy's raw action units, and guards the pretrained policy while it learns from real
transitions:

- a small initial std (``init_std``), kept within ``[min_std, max_std]``, no entropy bonus;
- ``critic_warmup`` iterations that fit only the critic (the pretrained critic learned
  imagined returns), the actor and std frozen;
- the actor's observation/goal normalizers are never updated (the runner never calls
  ``update_normalization``), so its input mapping stays the pretrained one;
- clipped surrogate, a learning rate adapted to ``desired_kl`` and an early stop of the
  epochs when the KL to the collecting policy passes ``kl_stop`` times it;
- an EMA copy of the actor for evaluation, as in SuperMimic.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict


@dataclass
class PpoCfg:
  num_steps: int = 24
  """Env steps per env and iteration (the rollout length)."""
  epochs: int = 3
  num_minibatches: int = 4
  clip: float = 0.2
  value_clip: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  init_std: float = 0.08
  """Initial action std (raw policy units). 0.08: the pretrained policy's eval96 pass rate
  is unchanged with up to 0.1 of i.i.d. action noise and drops from 0.15."""
  min_std: float = 0.02
  max_std: float = 0.15
  entropy_coef: float = 0.0
  actor_lr: float = 1e-5
  """Initial actor (and std) learning rate; then adapted to ``desired_kl``."""
  min_lr: float = 1e-6
  max_lr: float = 1e-4
  desired_kl: float = 0.01
  kl_stop: float = 2.5
  """Stop an iteration's epochs once a minibatch's KL exceeds this many ``desired_kl``."""
  critic_lr: float = 1e-4
  max_grad_norm: float = 1.0
  critic_warmup: int = 100
  """Iterations that update only the critic."""
  ema_decay: float = 0.999
  """Per actor update; the EMA actor is evaluated and exported."""


def gaussian_log_prob(actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
  """Log density of ``actions`` under N(mean, exp(log_std)^2), summed over action dims."""
  z = (actions - mean) * torch.exp(-log_std)
  return (-0.5 * z.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)


def gaussian_kl(
  mean_old: torch.Tensor, log_std_old: torch.Tensor, mean_new: torch.Tensor, log_std_new: torch.Tensor
) -> torch.Tensor:
  """KL(old || new) per sample, summed over action dims."""
  var_old, var_new = torch.exp(2 * log_std_old), torch.exp(2 * log_std_new)
  kl = log_std_new - log_std_old + (var_old + (mean_old - mean_new).pow(2)) / (2 * var_new) - 0.5
  return kl.sum(-1)


def gae(
  values: torch.Tensor,
  last_values: torch.Tensor,
  rewards: torch.Tensor,
  dones: torch.Tensor,
  gamma: float,
  lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """``(advantages, returns)`` of a time-major rollout ``(T, B)``. ``dones[t]`` ends the
  episode after step ``t`` (the next row is a reset); time-outs are bootstrapped into
  ``rewards`` beforehand."""
  advantages = torch.zeros_like(values)
  running = torch.zeros_like(last_values)
  next_values = last_values
  for t in reversed(range(values.shape[0])):
    not_done = 1.0 - dones[t].float()
    delta = rewards[t] + gamma * not_done * next_values - values[t]
    running = delta + gamma * lam * not_done * running
    advantages[t] = running
    next_values = values[t]
  return advantages, advantages + values


class PpoLearner:
  """PPO on a pretrained deterministic actor (the Gaussian mean) and critic."""

  def __init__(
    self,
    actor: nn.Module,
    critic: nn.Module,
    num_actions: int,
    cfg: PpoCfg,
    device: str,
    ema_actor: nn.Module | None = None,
  ):
    """``ema_actor``: the EMA copy to keep (e.g. the one the evaluator reads); a new
    copy of ``actor`` if None."""
    self.cfg = cfg
    self.actor, self.critic = actor, critic
    self.log_std = nn.Parameter(torch.full((num_actions,), math.log(cfg.init_std), device=device))
    self.lr = cfg.actor_lr
    self.actor_optimizer = torch.optim.Adam([*actor.parameters(), self.log_std], lr=self.lr)
    self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    self.ema_actor = ema_actor if ema_actor is not None else copy.deepcopy(actor).eval().requires_grad_(False)
    self.iteration = 0

  @property
  def std(self) -> torch.Tensor:
    return self.log_std.exp()

  @torch.no_grad()
  def act(self, actor_td: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(actions, log_probs, means)`` sampled from the current policy."""
    self.actor.eval()
    mean = self.actor(actor_td)
    actions = mean + self.std * torch.randn_like(mean)
    return actions, gaussian_log_prob(actions, mean, self.log_std), mean

  @torch.no_grad()
  def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.critic(critic_obs).squeeze(-1)

  def update(self, batch: TensorDict) -> dict[str, torch.Tensor | float]:
    """PPO epochs on one rollout, flattened to ``(N,)``: ``actor_obs``, ``critic_obs``,
    ``actions``, ``log_probs``, ``means``, ``values``, ``advantages``, ``returns``."""
    cfg = self.cfg
    train_actor = self.iteration >= cfg.critic_warmup
    adv = batch["advantages"]
    batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    log_std_old = self.log_std.detach().clone()
    n = batch.shape[0]
    size = n // cfg.num_minibatches
    stats: dict[str, list[torch.Tensor]] = {k: [] for k in ("value_loss", "surrogate", "kl", "clip_fraction")}
    stopped = False
    updates = 0
    # The actor is only ever run in eval mode: no auxiliary (latent BC) loss, and the
    # normalizers are frozen whatever the mode.
    self.actor.eval()
    for _ in range(cfg.epochs):
      perm = torch.randperm(n, device=adv.device)
      for k in range(cfg.num_minibatches):
        mb = batch[perm[k * size : (k + 1) * size]]
        values = self.critic(mb["critic_obs"]).squeeze(-1)
        v_clipped = mb["values"] + (values - mb["values"]).clamp(-cfg.value_clip, cfg.value_clip)
        value_loss = torch.maximum(
          (values - mb["returns"]).pow(2), (v_clipped - mb["returns"]).pow(2)
        ).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()
        stats["value_loss"].append(value_loss.detach())
        if not train_actor:
          continue

        mean = self.actor(mb["actor_obs"])
        log_prob = gaussian_log_prob(mb["actions"], mean, self.log_std)
        ratio = torch.exp(log_prob - mb["log_probs"])
        a = mb["advantages"]
        surrogate = -torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
        entropy = (self.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
        loss = surrogate - cfg.entropy_coef * entropy
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([*self.actor.parameters(), self.log_std], cfg.max_grad_norm)
        self.actor_optimizer.step()
        with torch.no_grad():
          self.log_std.clamp_(math.log(cfg.min_std), math.log(cfg.max_std))
          # The KL after this step, to the collecting policy: what the learning rate and
          # the early stop act on (before the step, a rollout's first minibatch always
          # reads ~0 and would raise the learning rate every iteration).
          kl = gaussian_kl(mb["means"], log_std_old, self.actor(mb["actor_obs"]), self.log_std).mean()
        self._update_ema()
        updates += 1
        stats["surrogate"].append(surrogate.detach())
        stats["kl"].append(kl)
        stats["clip_fraction"].append(((ratio - 1).abs() > cfg.clip).float().mean().detach())
        kl_value = float(kl)
        self._adapt_lr(kl_value)
        if kl_value > cfg.kl_stop * cfg.desired_kl:
          stopped = True
          break
      if stopped:
        break

    with torch.no_grad():
      residual = batch["returns"] - self.critic(batch["critic_obs"]).squeeze(-1)
      explained = 1 - residual.var() / (batch["returns"].var() + 1e-8)
    self.iteration += 1
    out: dict[str, torch.Tensor | float] = {
      f"PPO/{k}": torch.stack(v).mean() for k, v in stats.items() if v
    }
    return out | {
      "PPO/explained_variance": explained,
      "PPO/actor_lr": self.lr,
      "PPO/std_mean": self.std.mean().detach(),
      "PPO/std_min": self.std.min().detach(),
      "PPO/std_max": self.std.max().detach(),
      "PPO/actor_updates": float(updates),
      "PPO/kl_stopped": float(stopped),
      "PPO/actor_frozen": float(not train_actor),
      "PPO/advantage_std": adv.std().detach(),
    }

  def _adapt_lr(self, kl: float) -> None:
    cfg = self.cfg
    if kl > 2.0 * cfg.desired_kl:
      self.lr = max(cfg.min_lr, self.lr / 1.5)
    elif kl < 0.5 * cfg.desired_kl:
      self.lr = min(cfg.max_lr, self.lr * 1.5)
    for group in self.actor_optimizer.param_groups:
      group["lr"] = self.lr

  @torch.no_grad()
  def _update_ema(self) -> None:
    ema = list(self.ema_actor.parameters())
    torch._foreach_lerp_(ema, list(self.actor.parameters()), 1.0 - self.cfg.ema_decay)
    for ema_buf, buf in zip(self.ema_actor.buffers(), self.actor.buffers(), strict=True):
      ema_buf.copy_(buf)

  def state_dict(self) -> dict:
    return {
      "log_std": self.log_std.detach().clone(),
      "lr": self.lr,
      "iteration": self.iteration,
      "actor_optimizer": self.actor_optimizer.state_dict(),
      "critic_optimizer": self.critic_optimizer.state_dict(),
    }

  def load_state_dict(self, state: dict) -> None:
    with torch.no_grad():
      self.log_std.copy_(state["log_std"])
    self.lr, self.iteration = state["lr"], state["iteration"]
    self.actor_optimizer.load_state_dict(state["actor_optimizer"])
    self.critic_optimizer.load_state_dict(state["critic_optimizer"])
    for group in self.actor_optimizer.param_groups:
      group["lr"] = self.lr
