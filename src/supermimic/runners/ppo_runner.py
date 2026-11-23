"""PPO fine-tuning of a trained SuperMimic actor in the simulator.

The world model is fixed (and only saved along with the checkpoints, so every tool that
reads a SuperMimic run directory reads these too); the actor and critic continue from
``init_from`` and learn from real transitions with ``supermimic.learners.ppo``. The env,
the actor's observations, the critic's input (representation state, world-model goal,
sim context), the reset sampler, the protected evaluation and the checkpoints are the
SuperMimic runner's. One logged iteration is one PPO iteration (``ppo.num_steps`` env
steps per env, then the PPO epochs); ``max_iterations``, ``eval_interval`` and
``save_interval`` count them.
"""

from __future__ import annotations

import dataclasses
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import wandb
from tensordict import TensorDict

from supermimic.learners.ppo import PpoCfg, PpoLearner, gae
from supermimic.runners.supermimic_runner import (
  EVAL_CHECKPOINTS,
  TRAINER_STATE,
  SuperMimicRunner,
  SuperMimicRunnerCfg,
  _save_atomic,
)
from supermimic.utils.wandb_artifacts import log_model_artifact

PPO_FILE = "ppo.pt"
"""Checkpoint file with the PPO action std (``log_std``)."""


@dataclass
class PpoRunnerCfg(SuperMimicRunnerCfg):
  ppo: PpoCfg = field(default_factory=PpoCfg)
  init_actor: Literal["actor_ema", "actor"] = "actor_ema"
  """Which actor of ``init_from`` starts PPO (and its EMA): ``actor_ema`` (the evaluated
  policy) or ``actor``."""


class PpoRunner(SuperMimicRunner):
  cfg: PpoRunnerCfg

  def __init__(self, env, cfg: PpoRunnerCfg, task_id: str, registry_name: str | None = None, device: str = "cuda:0"):
    # The replay buffer only feeds world-model training, which PPO does not do.
    super().__init__(env, dataclasses.replace(cfg, buffer_size=2), task_id, registry_name, device)
    learner = self.policy_learner
    if getattr(learner, "critic", None) is None or learner.ema_actor is None:
      raise ValueError("PPO fine-tuning needs the actor-critic policy learner with an EMA actor")
    self.ppo = PpoLearner(
      learner.actor, learner.critic, self._num_actions, self.cfg.ppo, device, ema_actor=learner.ema_actor
    )

  # ------------------------------------------------------------ weights

  def _init_from_weights(self, src: Path) -> None:
    """Actor and EMA actor from ``src``'s ``init_actor``, its critic and world model
    (whose sim context statistics the actor's context and the critic read)."""

    def load(name: str):
      return torch.load(src / name, map_location=self.device, weights_only=False)

    bundle = load("world_model.pt")
    for name in ("action_scale", "action_offset"):
      old, new = bundle["dynamics"][name].to(self.device), getattr(self._dynamics, name)
      if old.shape != new.shape or not torch.allclose(old, new, atol=1e-6):
        raise ValueError(f"init_from: {src} was trained with another {name} than this env's")
    self._wm.load_bundle(bundle)
    actor = load(f"{self.cfg.init_actor}.pt")
    self.policy_learner.actor.load_state_dict(actor)
    self.policy_learner.ema_actor.load_state_dict(actor)
    self.policy_learner.critic.load_state_dict(load("critic.pt"))
    print(f"[INFO] PPO starts from {src}: {self.cfg.init_actor}.pt (actor and EMA), critic, world model")

  def _write_checkpoint(self, directory: Path) -> list[str]:
    files = super()._write_checkpoint(directory)
    path = directory / PPO_FILE
    torch.save({"log_std": self.ppo.log_std.detach().cpu()}, path)
    return [*files, str(path)]

  def _ppo_state(self, iteration: int) -> dict:
    learner = self.policy_learner
    state = {
      "step": iteration,
      "log_count": iteration + 1,
      "wandb_run_id": wandb.run.id if wandb.run is not None else None,
      "logger": {"tot_timesteps": self.logger.tot_timesteps, "tot_time": self.logger.tot_time},
      "world_model": self._wm.bundle(),
      "actor": learner.actor.state_dict(),
      "actor_ema": learner.ema_actor.state_dict(),
      "critic": learner.critic.state_dict(),
      "ppo": self.ppo.state_dict(),
    }
    if self._sampler is not None:
      state["sampler"] = self._sampler.state_dict()
    return state

  def _restore_ppo(self, state: dict) -> int:
    """Load a ``_ppo_state``; returns the next iteration."""
    learner = self.policy_learner
    self._wm.load_bundle(state["world_model"])
    learner.actor.load_state_dict(state["actor"])
    learner.ema_actor.load_state_dict(state["actor_ema"])
    learner.critic.load_state_dict(state["critic"])
    self.ppo.load_state_dict(state["ppo"])
    if self._sampler is not None and "sampler" in state:
      self._sampler.load_state_dict(state["sampler"])
    self.logger.tot_timesteps = state["logger"]["tot_timesteps"]
    self.logger.tot_time = state["logger"]["tot_time"]
    return state["step"] + 1

  # ------------------------------------------------------------ collection

  def _observe(self) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
    """The actor's input, the critic's input and the pre-step clip frames."""
    obs = self.env.get_observations()
    curr_states = obs["full_state"].to(self.device)
    motion_times = self.motion_cmd.time_steps.to(self.device).clone()
    rep_state = self.world_model_learner.represent(curr_states)
    aux_state = self._representation.extract_aux_state(curr_states)
    sim_ctx = self._sim_context.read(self.env.unwrapped)
    actor_td = self._build_sim_obs_td(rep_state, aux_state, obs["actor"].to(self.device), motion_times, sim_ctx)
    wm_goal, _ = self._representation.compute_wm_goal(self.motion, motion_times, aux_state)
    critic_obs = torch.cat([rep_state, wm_goal, sim_ctx], dim=-1)
    return actor_td, critic_obs, motion_times

  def _collect(self) -> TensorDict:
    """One rollout of ``ppo.num_steps`` env steps with the sampling policy, with its
    advantages and returns."""
    cfg = self.cfg.ppo
    steps = []
    with torch.no_grad():
      for _ in range(cfg.num_steps):
        actor_td, critic_obs, motion_times = self._observe()
        actions, log_probs, means = self.ppo.act(actor_td)
        values = self.ppo.value(critic_obs)
        _, rewards, dones, extras = self.env.step(actions)
        rewards = rewards.to(self.device)
        manager = self.env.unwrapped.termination_manager
        time_outs = manager.time_outs.to(self.device, torch.bool)
        terminated = manager.terminated
        self.logger.process_env_step(rewards, terminated, extras)
        if self._sampler is not None:
          clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
          weights = self._sampler.record(
            motion_times, rewards, terminated.to(self.device, torch.bool) & ~time_outs, clip_end, time_outs & ~clip_end
          )
          if weights is not None:
            self.motion_cmd.set_sampling_distribution(weights)
        steps.append(
          TensorDict(
            {
              "actor_obs": actor_td,
              "critic_obs": critic_obs,
              "actions": actions,
              "log_probs": log_probs,
              "means": means,
              "values": values,
              # A time-out is not the end of the task: its next value is bootstrapped.
              "rewards": rewards + cfg.gamma * values * time_outs.float(),
              "dones": dones.to(self.device).bool(),
            },
            batch_size=[self.env.num_envs],
          )
        )
      _, last_critic_obs, _ = self._observe()
      rollout = torch.stack(steps)
      advantages, returns = gae(
        rollout["values"], self.ppo.value(last_critic_obs), rollout["rewards"], rollout["dones"], cfg.gamma, cfg.lam
      )
    rollout["advantages"], rollout["returns"] = advantages, returns
    return rollout.flatten(0, 1)

  # ------------------------------------------------------------ loop

  def run(self) -> None:
    self._stop_requested = False
    previous_handler = signal.signal(signal.SIGTERM, self._request_stop)
    log_dir, state = self._open_log_dir()
    self._init_wandb(log_dir, run_id=state["wandb_run_id"] if state else None)
    self.env.reset()

    start = 0
    if state is not None:
      start = self._restore_ppo(state)
      del state
    elif self.cfg.init_from:
      self._init_from_weights(Path(self.cfg.init_from))
    else:
      raise ValueError("PPO fine-tuning starts from a trained run: give --runner.init-from (or a log_dir to resume)")

    uploader = ThreadPoolExecutor(max_workers=1)
    upload: Future | None = None

    def keep_eval_checkpoint(iteration: int) -> Future:
      eval_dir = log_dir / EVAL_CHECKPOINTS / f"iter_{iteration:06d}"
      files = self._write_checkpoint(eval_dir)
      print(f"[EVAL] iteration {iteration}: checkpoint kept in {eval_dir}")
      return uploader.submit(log_model_artifact, files, aliases=[f"eval-{iteration}"])

    if start == 0 and self.evaluator is not None:
      self._evaluate(0)  # the starting policy, on this run's eval
      if self.cfg.keep_eval_checkpoints:
        upload = keep_eval_checkpoint(0)

    print(f"[INFO] PPO from iteration {start} to {self.cfg.max_iterations}")
    for it in range(start, self.cfg.max_iterations):
      t0 = time.time()
      rollout = self._collect()
      t1 = time.time()
      stats = self.ppo.update(rollout)
      t2 = time.time()
      self.logger.log(
        it=it,
        total_it=self.cfg.max_iterations,
        collect_time=t1 - t0,
        wm_learn_time=0.0,
        policy_learn_time=t2 - t1,
        loss_dict=stats,
        num_collected_steps=self.cfg.ppo.num_steps * self.env.num_envs,
        extra_row=self.motion_cmd.log_stats() if hasattr(self.motion_cmd, "log_stats") else None,
      )
      done_it = it + 1
      if self.evaluator is not None and done_it % self.cfg.eval_interval == 0:
        self._evaluate(done_it)
        if self.cfg.keep_eval_checkpoints:
          if upload is not None:
            upload.result()
          upload = keep_eval_checkpoint(done_it)
      if done_it % self.cfg.save_interval == 0 or done_it == self.cfg.max_iterations:
        if upload is not None:
          upload.result()  # the previous version must be logged before its files are overwritten
        files = self._write_checkpoint(log_dir)
        _save_atomic(self._ppo_state(it), log_dir / TRAINER_STATE)
        upload = uploader.submit(log_model_artifact, files)
      if self._stop_requested:
        _save_atomic(self._ppo_state(it), log_dir / TRAINER_STATE)
        print(f"[INFO] SIGTERM: saved {log_dir / TRAINER_STATE} after iteration {it}")
        wandb.mark_preempting()
        break

    signal.signal(signal.SIGTERM, previous_handler)
    if upload is not None:
      upload.result()
    uploader.shutdown()
    self._cleanup()
