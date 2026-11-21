#!/bin/bash
# Short HOI training run that exercises the whole loop on a small library:
# env build, physics, rewards, WM warm-up, then WM + policy updates.
#
#   sbatch tools/hoi/smoke_train.sbatch <packed HOI library dir>
#SBATCH --job-name=hoi-smoke
#SBATCH --account=liu-account
#SBATCH --partition=liu-compute
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.log

set -euo pipefail
LIB=${1:?usage: sbatch smoke_train.sbatch <library dir>}
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/projects/ll/tianzizh/.uv_cache}
export WANDB_MODE=offline
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
nvidia-smi -L
uv run --no-sync python src/supermimic/scripts/train.py SuperMimic-HOI-G1 \
  --motion_file "$LIB" \
  --num_envs 512 \
  --runner.run_name hoi-smoke \
  --runner.max_iterations 400 \
  --runner.log_interval 20 \
  --runner.save_interval 1000000 \
  --runner.wm_warmup_updates 50 \
  --runner.wm.num_batches 2 \
  --runner.wm.batch_size 512 \
  --runner.policy.batch_size 512
