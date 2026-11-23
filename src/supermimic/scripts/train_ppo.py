"""PPO fine-tuning of a trained SuperMimic HOI run in the simulator (``PpoRunner``).

  python src/supermimic/scripts/train_ppo.py SuperMimic-HOI-G1 --motion_file LIB \
      --runner.init_from RUN_DIR --runner.hoi_physics hw3 --runner.hoi_action_contract mjlab \
      --runner.eval_motion_file EVAL_LIB --runner.max_iterations 3000 ...
"""

from __future__ import annotations

from dataclasses import dataclass

import mjlab
import mjlab.tasks  # noqa: F401
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.hoi.env_cfg import configure_hoi_env, configure_hoi_physics, is_hoi_env
from supermimic.runners.ppo_runner import PpoRunner, PpoRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device


@dataclass
class PpoTrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: PpoRunnerCfg
  num_envs: int = 4096
  device: str | None = None
  motion_file: str | None = None

  @staticmethod
  def from_task(task_id: str) -> PpoTrainConfig:
    return PpoTrainConfig(env=load_env_cfg(task_id, play=False), runner=PpoRunnerCfg())


def main() -> None:
  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(PpoTrainConfig, args=remaining, default=PpoTrainConfig.from_task(task), config=mjlab.TYRO_FLAGS)
  if not is_hoi_env(args.env):
    raise ValueError("PPO fine-tuning is set up for the HOI tasks")
  if not args.motion_file:
    raise ValueError("give --motion_file <packed HOI library directory>")
  device = setup_device(args.device)
  configure_hoi_env(args.env, args.motion_file)
  configure_hoi_physics(
    args.env, args.runner.hoi_physics, push=args.runner.hoi_push, action_contract=args.runner.hoi_action_contract
  )
  if args.runner.representation == "gencoord":
    args.runner.representation = "hoi_gencoord"
  args.env.commands["motion"].sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )
  print(f"[INFO] PPO on {args.motion_file}, robot physics {args.runner.hoi_physics}, init {args.runner.init_from}")
  env = create_env(args.env, args.num_envs, device)
  PpoRunner(env=env, cfg=args.runner, task_id=task, device=device).run()


if __name__ == "__main__":
  main()
