from __future__ import annotations

import dataclasses
import os
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import torch
import tyro
import wandb
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import StateRepresentation, create_representation
from supermimic.runners.exporter import export_actor_as_onnx
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  TasksCfg,
  build_actor_obs_td,
  resolve_task_specs,
)
from supermimic.utils.helpers import (
  extract_action_scale_offset,
  unpack_obs_dims,
)
from supermimic.utils.logger import Logger
from supermimic.utils.wandb_artifacts import download_model_dir, log_model_artifact

if TYPE_CHECKING:
  from mjlab.rl import RslRlVecEnvWrapper

BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]

TRAINER_STATE = "trainer_state.pt"
"""Everything a run needs to continue, written with every checkpoint and on SIGTERM."""


def _save_atomic(obj: object, path: Path) -> None:
  """``torch.save`` that never leaves a partly written ``path`` (a kill mid-write
  keeps the previous file)."""
  tmp = path.with_name(path.name + ".tmp")
  torch.save(obj, tmp)
  os.replace(tmp, path)


@dataclass
class SuperMimicRunnerCfg:
  wandb_project: str = "supermimic"
  run_name: str | None = None
  log_dir: str | None = None
  """Checkpoint directory in place of a new timestamped one under ``logs/<task>``.
  If it holds a ``trainer_state.pt``, training continues from it: the weights, the
  optimizer states, the counters, the reset sampler and the W&B run (the replay
  buffer is refilled first). A requeued Slurm job resumes this way."""
  wandb_run_path: str | None = None
  headless: bool = True
  buffer_size: int = 1000
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  actor_obs: str = "rep_state"  # "rep_state" or "proprio"
  log_interval: int = 10
  save_interval: int = 1000
  history_length: int = 1
  termination_softness: float = 0.05
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
  max_iterations: int = 1000000
  replay_ratio: int = 16
  latent_dim: int = 0
  """Goal latent width. 0 = auto: 64 for HOI, 32 otherwise."""
  wm_warmup_updates: int = -1
  """Update rounds that train only the world model before the policy starts.
  -1 = auto: 500 for HOI, 0 otherwise."""
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_bc_weight: float = 1e-1
  """Weight on the latent BC loss pulling non-anchor encoders toward the
  anchor encoder's latent on the same state. 0 disables."""
  tasks: TasksCfg = field(default_factory=TasksCfg)
  """Per-task hyperparameters."""
  slot_mix: dict[str, float] = field(
    default_factory=lambda: {"motion": 1.0, "velocity": 0.0}
  )
  """Fractional slot allocation per task. Must sum to 1.0."""
  policy: BpttCfg | ActorCriticCfg = field(default_factory=ActorCriticCfg)

  sampling_mode: Literal["uniform", "adaptive", "start", "return"] = "return"
  """RSI reset distribution: 'return' (concentrate resets on the bins where the
  policy earns low return; see ReturnSampler) or one of mjlab's native modes.
  train.py resolves this into the motion command's cfg before the env is built
  ('return' drives mjlab's 'external' mode)."""
  return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)
  eval_motion_file: str | None = None
  """Held-out HOI library played in-process by ``supermimic.hoi.evaluate`` every
  ``eval_interval`` logged iterations (results under ``Eval/`` on W&B)."""
  eval_interval: int = 2000
  hoi_physics: Literal["mjlab", "hw3"] = "mjlab"
  """HOI robot physics (``supermimic.hoi.env_cfg.configure_hoi_physics``): "mjlab" (mjlab's
  G1 actuators) or "hw3" (Fengqiao's lab G1: motor envelope, delay, joint friction, his
  action scale). train.py applies it to the env; the protected evaluation runs it nominal."""
  hoi_push: bool = False
  """With hw3: his periodic base-velocity pushes during training."""
  hoi_action_contract: Literal["mjlab", "hw3"] | None = None
  """Whose action scale the policy speaks (None: ``hoi_physics``'). A policy trained
  on mjlab physics keeps "mjlab" when it continues on hw3: his scale differs on the
  hip pitch (0.351 vs 0.548)."""
  wm_prev_command: bool | None = None
  """The dynamics also sees the command in flight (``Dynamics.prev_command``).
  None = auto: on with hw3 physics, whose command delay applies it in the first
  2-4 physics steps of each step."""
  init_from: str | None = None
  """Run directory (actor.pt, actor_ema.pt, critic.pt, world_model.pt) whose weights
  start this new run; the inputs this run adds (sim context columns, the in-flight
  command) start at zero weight. Ignored when ``log_dir`` resumes a run."""


@dataclass
class WorldModel:
  """The world model plus the env dimensions that sized it."""

  representation: StateRepresentation
  dynamics: Dynamics
  sim_context: SimContext
  num_goals: int
  num_actions: int

  def bundle(self) -> dict:
    """The ``world_model.pt`` contents."""
    return {
      "dynamics": self.dynamics.state_dict(),
      "sim_context": self.sim_context.state_dict(),
    }

  def load_bundle(self, bundle: dict) -> None:
    """Load a ``world_model.pt`` bundle."""
    self.dynamics.load_state_dict(bundle["dynamics"])
    self.sim_context.load_state_dict(bundle["sim_context"])


def build_world_model(
  cfg: SuperMimicRunnerCfg, env: RslRlVecEnvWrapper, device: str
) -> WorldModel:
  """Construct the representation and dynamics for ``env`` sized by ``cfg``.

  Resets ``env`` to read its observation dimensions.
  """
  if cfg.history_length > 1 and cfg.actor_obs != "proprio":
    raise RuntimeError(
      "history_length > 1 requires actor_obs='proprio' (--runner.actor-obs proprio)"
    )

  obs, _ = env.reset()
  num_full_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

  sim_context = SimContext.from_env(env.unwrapped)
  sim_ctx_dim = sim_context.output_dim
  print(
    f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})"
  )

  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  representation = create_representation(
    cfg.representation,
    num_full_state,
    num_actions,
    env.unwrapped.step_dt,
    env.unwrapped.cfg.decimation,
    default_joint_pos,
    actor_obs_mode=cfg.actor_obs,
  ).to(device)

  representation.bind(env.unwrapped, sim_context)

  print(
    f"[INFO] Representation: {cfg.representation} "
    f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
    f"pred={representation.prediction_dim})"
  )

  action_scale, action_offset = extract_action_scale_offset(env)
  dynamics = Dynamics(
    representation,
    num_actions,
    sim_ctx_dim,
    action_scale,
    action_offset,
    hidden_dims=cfg.dynamics_hidden_dims,
    prev_command=bool(cfg.wm_prev_command),
  ).to(device)

  return WorldModel(
    representation=representation,
    dynamics=dynamics,
    sim_context=sim_context,
    num_goals=num_goals,
    num_actions=num_actions,
  )


def resolve_auto_cfg(cfg: SuperMimicRunnerCfg, hoi: bool) -> SuperMimicRunnerCfg:
  """Fill the ``auto`` fields (0 / -1) with the HOI or the motion-tracking default."""
  cfg = dataclasses.replace(cfg)
  if cfg.latent_dim == 0:
    cfg.latent_dim = 64 if hoi else 32
  if cfg.wm_warmup_updates < 0:
    cfg.wm_warmup_updates = 500 if hoi else 0
  if cfg.wm_prev_command is None:
    cfg.wm_prev_command = hoi and cfg.hoi_physics == "hw3"
  policy = dataclasses.replace(cfg.policy)
  if policy.grad_accum_chunks == 0:
    policy.grad_accum_chunks = 2 if hoi else 1  # HOI at 4096 envs: 39.5 GB on an L40; 1 chunk OOMs
  if policy.ema_decay < 0:
    policy.ema_decay = 0.999 if hoi else 0.0
  cfg.policy = policy
  return cfg


def context_dim(representation: StateRepresentation) -> int:
  cols = representation.context_columns
  return 0 if cols is None else cols.stop - cols.start


def build_actor(
  cfg: SuperMimicRunnerCfg, wm: WorldModel, device: str
) -> MultiEncoderAE:
  """Actor sized by ``cfg``; the resolved task specs live on ``actor.task_specs``."""
  representation = wm.representation
  obs_dim = (
    representation.rep_state_dim
    if cfg.actor_obs == "rep_state"
    else representation.proprio_obs_dim(wm.num_actions)
  )
  return MultiEncoderAE(
    context_dim=context_dim(representation),
    obs_dim=obs_dim,
    action_dim=wm.num_actions,
    task_specs=resolve_task_specs(cfg.tasks, representation, wm.num_goals, device),
    slot_mix=cfg.slot_mix,
    latent_dim=cfg.latent_dim,
    hidden_dims=cfg.actor_hidden_dims,
    latent_bc_weight=cfg.latent_bc_weight,
    history_length=cfg.history_length,
  ).to(device)


def load_widened(module: torch.nn.Module, state: dict) -> list[str]:
  """Load ``state`` into ``module`` where a 2-D weight of ``module`` may have more
  input columns than in ``state``: the old ones are copied, the new last ones are
  zero. Every other tensor must match in shape. Returns the widened keys."""
  own = module.state_dict()
  if set(own) != set(state):
    raise ValueError(f"state dict keys differ: {sorted(set(own) ^ set(state))}")
  out, widened = {}, []
  for key, new in own.items():
    old = state[key].to(new.device, new.dtype)
    if old.shape == new.shape:
      out[key] = old
    elif old.ndim == 2 and old.shape[0] == new.shape[0] and old.shape[1] < new.shape[1]:
      grown = torch.zeros_like(new)
      grown[:, : old.shape[1]] = old
      out[key] = grown
      widened.append(key)
    else:
      raise ValueError(f"{key}: {tuple(old.shape)} cannot be widened to {tuple(new.shape)}")
  module.load_state_dict(out)
  return widened


def load_wandb_checkpoint(wandb_run_path: str, wm: WorldModel, device: str) -> None:
  """Warm-start ``wm`` from another W&B run's world-model bundle."""
  print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
  run = wandb.Api().run(wandb_run_path)
  wm.load_bundle(
    torch.load(download_model_dir(run) / "world_model.pt", map_location=device)
  )


class Ratio:
  def __init__(self, ratio: float) -> None:
    self._ratio = ratio
    self._prev: int | None = None

  def __call__(self, step: int) -> int:
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class SuperMimicRunner:
  """Trains the world model and the multi-encoder actor.

  Each task owns a contiguous slot of the env batch and its own goal encoder.
  All tasks share the motion-tracking reward.
  """

  cfg: SuperMimicRunnerCfg

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    cfg: SuperMimicRunnerCfg,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    from supermimic.hoi.env_cfg import is_hoi_env

    self.env = env
    self.is_hoi = is_hoi_env(env.unwrapped.cfg)
    self.cfg = cfg = resolve_auto_cfg(cfg, self.is_hoi)
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device

    self._init_common()
    self._init_policy()

    self.logger = Logger(
      run_name=cfg.run_name,
      num_envs=env.num_envs,
      max_episode_length=env.max_episode_length,
      device=device,
    )
    self.evaluator = None
    if cfg.eval_motion_file:
      if not self.is_hoi:
        raise ValueError("eval_motion_file is the HOI protected evaluation; this task is not HOI")
      from supermimic.hoi.evaluate import HoiEvaluator

      self.evaluator = HoiEvaluator(self, cfg.eval_motion_file)

  def _init_common(self) -> None:
    wm = build_world_model(self.cfg, self.env, self.device)
    self._wm = wm
    self._representation = wm.representation
    self._dynamics = wm.dynamics
    self._sim_context = wm.sim_context
    self._num_goals = wm.num_goals
    self._num_actions = wm.num_actions

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)

    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion

    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
    )
    self.world_model_learner = WorldModelLearner(
      self._dynamics, self._representation, self.cfg.wm, device=self.device
    )

    self._pending_norm_updates: list[tuple] = []
    self._sim_history: torch.Tensor | None = None

  def _seed_sim_history(self) -> None:
    """Seed the real-sim obs-history ring with the noised current sensor."""
    if self.cfg.history_length == 1:
      return
    seed = self._representation.add_proprio_noise(
      self.env.get_observations()["actor"].to(self.device)
    )
    self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

  def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window for a real-sim actor step.
    history_length > 1 needs ``_seed_sim_history()`` first."""
    if self.cfg.history_length == 1:
      return sensor.unsqueeze(1)
    assert self._sim_history is not None
    return self._sim_history

  def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
    """Append the noised next sensor; rows whose episode just reset repeat that
    single realization, matching the warm-start padding."""
    if self.cfg.history_length == 1:
      return
    assert self._sim_history is not None
    next_sensor = self._representation.add_proprio_noise(next_actor)
    shifted = torch.cat([self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1)
    reset = dones.bool().view(-1, 1, 1)
    self._sim_history = torch.where(reset, next_sensor.unsqueeze(1), shifted)

  @property
  def _stores_actor_obs(self) -> bool:
    """The buffer carries the real sensor only for its sole reader, the
    history warm-start (``WorldModelEnv.sample_initial_states``)."""
    return (
      self._representation.actor_obs_mode == "proprio" and self.cfg.history_length > 1
    )

  def _buffer_schema(self) -> dict[str, FieldSpec]:
    """Replay buffer fields."""
    schema = {
      "state": FieldSpec(self._representation.full_state_dim),
      "action": FieldSpec(self._num_actions),
      "prev_action": FieldSpec(self._num_actions),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }
    if self._stores_actor_obs:
      schema["actor_obs"] = FieldSpec(
        self._representation.proprio_obs_dim(self._num_actions)
      )
    schema["sim_ctx"] = FieldSpec(self._sim_context.output_dim)
    if self._dynamics.prev_command:
      schema["prev_cmd"] = FieldSpec(self._num_actions)
    return schema

  def _in_flight_command(self, prev_action: torch.Tensor) -> torch.Tensor:
    """The command still in flight when this step's action is issued, in the
    dynamics' proc space: the previous action's, or right after a reset the held
    pose (mjlab backfills a reset world's actuator targets, and so its command
    delay, with the reset joint positions; proc space carries the encoder bias)."""
    term = self.action_manager.get_term(self.action_manager.active_terms[0])
    held = term._entity.data.joint_pos_biased[:, term._target_ids]
    just_reset = (self.env.unwrapped.episode_length_buf == 0).unsqueeze(-1)
    return torch.where(just_reset, held, self._dynamics.proc(prev_action))

  def _step_and_store(
    self,
    actions: torch.Tensor,
    curr_states: torch.Tensor,
    actor_td: TensorDict,
    extra_fields: dict[str, torch.Tensor],
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Step the env, write the transition to the replay buffer and queue
    ``actor_td`` for the actor-normalizer update."""
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    motion_times = motion_cmd.time_steps.clone()

    raw_actions = actions.detach().clone()
    prev_action = self.action_manager.action.clone()
    if self._dynamics.prev_command:
      extra_fields = {**extra_fields, "prev_cmd": self._in_flight_command(prev_action)}

    next_obs, rewards, dones, extras = self.env.step(actions)

    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    terminated = dones.bool() & ~truncated

    fields: dict[str, torch.Tensor] = {
      "state": curr_states,
      "action": raw_actions,
      "prev_action": prev_action,
      "motion_time": motion_times,
      "terminated": terminated,
      "truncated": truncated,
      **extra_fields,
    }
    self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))

    self._pending_norm_updates.append(
      (curr_states.clone(), raw_actions.clone(), actor_td.detach().clone())
    )
    return next_obs, rewards, dones, extras

  def _flush_normalization(self) -> None:
    """Apply queued normalizer updates. Call outside ``inference_mode()``."""
    if not self._pending_norm_updates:
      return
    states, actions, actor_inputs = zip(*self._pending_norm_updates, strict=True)
    self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
    self._update_actor_normalization(actor_inputs)
    self._pending_norm_updates.clear()

  def _open_log_dir(self) -> tuple[Path, dict | None]:
    """The log directory (``cfg.log_dir``, or a new timestamped one) and the
    trainer state it holds, if any."""
    if self.cfg.log_dir is None:
      log_dir = Path("logs", self.task_id, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
      log_dir = Path(self.cfg.log_dir)
      state_path = log_dir / TRAINER_STATE
      if state_path.exists():
        state = torch.load(state_path, map_location=self.device)
        print(
          f"[INFO] Resuming {log_dir} after step {state['step']} "
          f"(iteration {state['log_count'] - 1})"
        )
        return log_dir, state
      if (log_dir / "actor.pt").exists():
        raise FileExistsError(
          f"{log_dir} holds a checkpoint but no {TRAINER_STATE}; a new run there "
          "would overwrite it"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    return log_dir, None

  def _init_wandb(self, log_dir: Path, run_id: str | None = None) -> None:
    """Initialize the W&B run (or continue run ``run_id``) and link the registry
    artifacts."""
    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    default_name = f"{self.task_id}-{log_dir.name}"
    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or default_name,
      config=wandb_config,
      id=run_id,
      resume="allow" if run_id else None,
    )

    if self.registry_name:
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        wandb.run.use_artifact(artifact)

  def _cleanup(self) -> None:
    """Close environment and finish W&B run."""
    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()
    wandb.finish()

  def _optimizers(self) -> dict[str, list[torch.optim.Optimizer]]:
    learner = self.policy_learner
    groups = {
      "world_model": self.world_model_learner.optimizers,
      "actor": learner.actor_optimizers,
    }
    if getattr(learner, "critic_optimizers", None) is not None:
      groups["critic"] = learner.critic_optimizers
    return groups

  def _trainer_state(self, step: int, log_count: int, wm_updates: int) -> dict:
    """What ``_restore`` continues from after loop step ``step``: the weights, the
    optimizer states, the counters, the reset sampler and the W&B run id. The
    replay buffer is left out (``_refill_replay_buffer``)."""
    learner = self.policy_learner
    state = {
      "step": step,
      "log_count": log_count,
      "wm_updates": wm_updates,
      "wandb_run_id": wandb.run.id if wandb.run is not None else None,
      "logger": {
        "tot_timesteps": self.logger.tot_timesteps,
        "tot_time": self.logger.tot_time,
      },
      "world_model": self._wm.bundle(),
      "actor": learner.actor.state_dict(),
      "optimizers": {
        name: [o.state_dict() for o in opts] for name, opts in self._optimizers().items()
      },
    }
    for name, module in (
      ("actor_ema", getattr(learner, "ema_actor", None)),
      ("critic", getattr(learner, "critic", None)),
    ):
      if module is not None:
        state[name] = module.state_dict()
    if self._sampler is not None:
      state["sampler"] = self._sampler.state_dict()
    return state

  def _init_from_weights(self, src: Path) -> None:
    """Start a new run from the weights of run directory ``src`` (``init_from``):
    actor, EMA actor, critic and world model; optimizers, counters, sampler and
    replay buffer start fresh, and the world-model warm-up runs as for any new run.
    Inputs this run adds are new last columns of the first layers (critic: sim
    context; dynamics: sim context, in-flight command) and start at zero weight."""

    def load(name: str):
      return torch.load(src / name, map_location=self.device, weights_only=False)

    learner = self.policy_learner
    learner.actor.load_state_dict(load("actor.pt"))
    ema = getattr(learner, "ema_actor", None)
    if ema is not None:
      ema.load_state_dict(load("actor_ema.pt") if (src / "actor_ema.pt").exists() else learner.actor.state_dict())
    critic = getattr(learner, "critic", None)
    widened = []
    if critic is not None and (src / "critic.pt").exists():
      widened += load_widened(critic, load("critic.pt"))
    bundle = load("world_model.pt")
    for name in ("action_scale", "action_offset"):
      old, new = bundle["dynamics"][name].to(self.device), getattr(self._dynamics, name)
      if old.shape != new.shape or not torch.allclose(old, new, atol=1e-6):
        raise ValueError(
          f"init_from: {src} was trained with another {name} than this env's; pick the same "
          "action contract (--runner.hoi-action-contract)"
        )
    widened += [f"dynamics.{k}" for k in load_widened(self._dynamics, bundle["dynamics"])]
    old_ctx, ctx = bundle["sim_context"], self._sim_context
    n = old_ctx["mean"].numel()
    if not (
      torch.allclose(old_ctx["mean"].to(self.device), ctx.mean[:n])
      and torch.allclose(old_ctx["scale"].to(self.device), ctx.scale[:n])
    ):
      raise ValueError(f"init_from: {src}'s sim context columns are not this run's first {n}")
    print(
      f"[INFO] Initialized from {src}: actor, EMA actor, critic, world model; widened "
      f"{widened or 'nothing'} (sim context {n} -> {ctx.output_dim} columns)"
    )

  def _restore(self, state: dict) -> tuple[int, int, int]:
    """Load a ``_trainer_state``; returns (first loop step, log_count, wm_updates).

    Runs before any update: the updates are captured as CUDA graphs over the
    optimizer state they first see. Optimizer or sampler state missing from
    ``state`` (one written by hand for a run checkpointed before trainer_state.pt
    existed) starts fresh."""
    learner = self.policy_learner
    self._wm.load_bundle(state["world_model"])
    learner.actor.load_state_dict(state["actor"])
    for name, module in (
      ("actor_ema", getattr(learner, "ema_actor", None)),
      ("critic", getattr(learner, "critic", None)),
    ):
      if module is not None:
        module.load_state_dict(state[name])
    saved = state.get("optimizers", {})
    for name, opts in self._optimizers().items():
      if name not in saved:
        print(f"[WARN] Resume: no {name} optimizer state; it starts fresh")
        continue
      for opt, opt_state in zip(opts, saved[name], strict=True):
        opt.load_state_dict(opt_state)
    if self._sampler is not None:
      if "sampler" in state:
        self._sampler.load_state_dict(state["sampler"])
      else:
        print("[WARN] Resume: no reset-sampler state; every bin starts unmeasured")
    self.logger.tot_timesteps = state["logger"]["tot_timesteps"]
    self.logger.tot_time = state["logger"]["tot_time"]
    return state["step"] + 1, state["log_count"], state["wm_updates"]

  def _refill_replay_buffer(self) -> None:
    """Collect with the restored policy, without updates, until the replay buffer
    is full again, as it was when the run stopped."""
    t0 = time.time()
    while (
      self.replay_buffer.num_transitions < self.cfg.buffer_size
      and not self._stop_requested
    ):
      self.step()
    print(
      f"[INFO] Refilled the replay buffer: {self.cfg.buffer_size} steps in "
      f"{time.time() - t0:.0f} s"
    )

  def _request_stop(self, signum: int, frame: object) -> None:
    """SIGTERM (Slurm preemption, scancel, time limit): stop after this step and
    write the trainer state. Printing here could re-enter a print in progress."""
    del signum, frame
    self._stop_requested = True

  def _init_policy(self) -> None:
    actor = build_actor(self.cfg, self._wm, self.device)
    self._active_task_specs = [
      ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name, 0.0) > 0.0
    ]

    print("[INFO] Sim slot ranges:")
    for ts, (lo, hi) in zip(
      actor.task_specs, actor.slot_ranges(self.env.num_envs), strict=True
    ):
      print(f"  {ts.name}: [{lo}, {hi}) — goal_dim={ts.goal_dim}")
    critic_input_dim = (
      self._representation.rep_state_dim
      + self._num_goals
      + self._sim_context.output_dim
    )
    critic = build_mlp(
      critic_input_dim,
      self.cfg.critic_hidden_dims,
      output_dim=1,
    ).to(self.device)

    policy_cfg = self.cfg.policy

    self.wm_env = WorldModelEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=policy_cfg.batch_size,
      replay_buffer=self.replay_buffer,
      task_specs=self._active_task_specs,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )
    if torch.cuda.is_available():
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)

    self._configure_reset_sampling()

    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor, critic, self.wm_env, policy_cfg, device=self.device
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor, self.wm_env, policy_cfg, device=self.device
      )

  def _build_sim_obs_td(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_actor_obs: torch.Tensor,
    motion_times: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> TensorDict:
    proprio = self.cfg.actor_obs == "proprio"
    sensor = self._representation.compute_actor_sensor(
      rep_state,
      sim_ctx,
      aux_state=aux_state,
      sim_actor_obs=sim_actor_obs if proprio else None,
      noisy=proprio,
    )
    goals = {
      ts.name: ts.extract_training_goal(self.motion, motion_times, aux_state)
      for ts in self._active_task_specs
    }
    cols = self._representation.context_columns
    context = None if cols is None else sim_ctx[:, cols]
    return build_actor_obs_td(goals, self._collection_history(sensor), context=context)

  def _configure_reset_sampling(self) -> None:
    self._sampler: ReturnSampler | None = None
    env_mode = self.motion_cmd.cfg.sampling_mode
    expected = (
      "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
    )
    if env_mode != expected:
      raise RuntimeError(
        f"runner sampling_mode='{self.cfg.sampling_mode}' needs the motion command "
        f"in '{expected}' mode but the env was built with '{env_mode}'; resolve it "
        "into the env cfg before env creation ('external' also needs an mjlab with "
        "MotionCommand external-mode support)"
      )
    if self.cfg.sampling_mode != "return":
      return
    self._sampler = ReturnSampler(
      cfg=self.cfg.return_sampler,
      bin_count=self.motion_cmd.bin_count,
      frame_count=self.motion.time_step_total,
      num_envs=self.env.num_envs,
      max_step_reward=self.wm_env.max_step_reward,
      device=self.device,
      frame_bins=getattr(getattr(self.motion_cmd, "lib", None), "frame_bin", None),
    )

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)

      # Pre-step frames; env.step() advances time_steps in place.
      motion_times = self.motion_cmd.time_steps.to(self.device).clone()

      rep_state = self.world_model_learner.represent(curr_states)
      aux_state = self._representation.extract_aux_state(curr_states)
      sim_ctx = self._sim_context.read(self.env.unwrapped)
      actor_td = self._build_sim_obs_td(
        rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx
      )

      actions = self.policy_learner.actor(actor_td)

      extra_fields = {"sim_ctx": sim_ctx}
      if self._stores_actor_obs:
        extra_fields["actor_obs"] = sim_actor_obs

      next_obs, rewards, dones, extras = self._step_and_store(
        actions,
        curr_states,
        actor_td=actor_td,
        extra_fields=extra_fields,
      )
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

      if self._sampler is not None:
        # A failure on a step that also times out counts as the time-out.
        manager = self.env.unwrapped.termination_manager
        truncated = manager.time_outs.to(self.device, torch.bool)
        clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
        weights = self._sampler.record(
          motion_times,
          rewards,
          terminated.to(self.device, torch.bool) & ~truncated,
          clip_end,
          truncated & ~clip_end,
        )
        if weights is not None:
          self.motion_cmd.set_sampling_distribution(weights)

    self._flush_normalization()
    return self.env.num_envs

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    self.policy_learner.actor.train()
    for actor_td in actor_inputs:
      self.policy_learner.actor.update_normalization(actor_td)

  def _evaluate(self, log_count: int) -> None:
    """Protected evaluation, logged at this iteration's W&B step."""
    t0 = time.time()
    row = self.evaluator.run()
    wandb.log(row, step=log_count)
    print(
      f"[EVAL] iteration {log_count}: pass rate body {row['Eval/all/phc_pass_rate']:.2f}, "
      f"object {row['Eval/all/object_pose_pass_rate']:.2f}, both {row['Eval/all/body_and_object_pose_pass_rate']:.2f}, "
      f"fell {row['Eval/all/fell_rate']:.2f} ({time.time() - t0:.0f} s)"
    )

  def run(self) -> None:
    self._stop_requested = False
    previous_handler = signal.signal(signal.SIGTERM, self._request_stop)
    log_dir, state = self._open_log_dir()
    self._init_wandb(log_dir, run_id=state["wandb_run_id"] if state else None)

    self.env.reset()
    self._seed_sim_history()

    start, log_count, wm_updates = 0, 0, 0
    if state is not None:
      start, log_count, wm_updates = self._restore(state)
      del state
      self._refill_replay_buffer()
    elif self.cfg.init_from:
      self._init_from_weights(Path(self.cfg.init_from))

    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")

    should_train = Ratio(
      self.env.num_envs
      / self.cfg.wm.num_batches
      / self.cfg.wm.batch_size
      / self.cfg.wm.horizon
      * self.cfg.replay_ratio
    )

    num_collected_steps = 0
    start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict: dict[str, torch.Tensor] = {}
    uploader = ThreadPoolExecutor(max_workers=1)
    upload: Future | None = None

    for step in range(start, self.cfg.max_iterations):
      num_collected_steps += self.step()

      for _ in range(should_train(step)):
        if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
          continue

        data = (
          self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
          for _ in range(self.cfg.wm.num_batches)
        )

        start_wm_time = time.time()
        wm_logs = self.world_model_learner.update(data)
        wm_learn_time += time.time() - start_wm_time
        wm_updates += 1
        loss_dict.update(wm_logs)
        if wm_updates <= self.cfg.wm_warmup_updates:
          continue  # world-model warm-up: no policy updates yet

        start_policy_time = time.time()
        policy_logs = self.policy_learner.update()
        policy_learn_time += time.time() - start_policy_time
        loss_dict.update(policy_logs)

      if step % self.cfg.log_interval == 0:
        elapsed_time = time.time() - start_time
        total_train_time = wm_learn_time + policy_learn_time
        collect_time = elapsed_time - total_train_time

        self.logger.log(
          it=log_count,
          total_it=self.cfg.max_iterations // self.cfg.log_interval,
          collect_time=collect_time,
          wm_learn_time=wm_learn_time,
          policy_learn_time=policy_learn_time,
          loss_dict=loss_dict,
          num_collected_steps=num_collected_steps,
          extra_row=self.motion_cmd.log_stats() if hasattr(self.motion_cmd, "log_stats") else None,
        )
        if self.evaluator is not None and log_count > 0 and log_count % self.cfg.eval_interval == 0:
          self._evaluate(log_count)

        log_count += 1
        start_time = time.time()
        wm_learn_time = 0
        policy_learn_time = 0
        num_collected_steps = 0

      if step % self.cfg.save_interval == 0:
        assert wandb.run is not None  # _init_wandb ran before the loop
        # The previous version must be logged before its files are overwritten.
        if upload is not None:
          upload.result()
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(self._wm.bundle(), world_model_path)
        torch.save(self.policy_learner.actor.state_dict(), actor_path)
        checkpoint_files = [world_model_path, actor_path]
        ema = getattr(self.policy_learner, "ema_actor", None)
        if ema is not None:
          ema_path = str(log_dir / "actor_ema.pt")
          torch.save(ema.state_dict(), ema_path)
          checkpoint_files.append(ema_path)
        critic = getattr(self.policy_learner, "critic", None)
        if critic is not None:
          critic_path = str(log_dir / "critic.pt")
          torch.save(critic.state_dict(), critic_path)
          checkpoint_files.append(critic_path)
        # Skip ONNX export on huge datasets.
        if (
          self._representation.actor_obs_mode == "proprio"
          and self.motion.time_step_total <= 15_000
        ):
          checkpoint_files += export_actor_as_onnx(
            actor=self.policy_learner.actor,
            env=self.env.unwrapped,
            path=str(log_dir),
            run_name=wandb.run.name,
            term_widths=self._representation.proprio_term_widths(self._num_actions),
          )
        _save_atomic(
          self._trainer_state(step, log_count, wm_updates), log_dir / TRAINER_STATE
        )
        upload = uploader.submit(log_model_artifact, checkpoint_files)

      if self._stop_requested:
        _save_atomic(
          self._trainer_state(step, log_count, wm_updates), log_dir / TRAINER_STATE
        )
        print(f"[INFO] SIGTERM: saved {log_dir / TRAINER_STATE} after step {step}")
        wandb.mark_preempting()
        break

    signal.signal(signal.SIGTERM, previous_handler)
    uploader.shutdown()
    self._cleanup()
