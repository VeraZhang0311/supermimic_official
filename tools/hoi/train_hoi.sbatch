#!/bin/bash
# Full HOI teacher training on one L40, logged online to W&B.
#
#   sbatch tools/hoi/train_hoi.sbatch <packed HOI library dir> [extra train.py args...]
#
# W&B project: $WANDB_PROJECT (default supermimic_hoi). Compiled kernels
# are cached under $TORCHINDUCTOR_CACHE_DIR so relaunches skip most of the
# first-update compile.
#SBATCH --job-name=hoi-train
#SBATCH --account=liu-account
#SBATCH --partition=liu-compute
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
LIB=${1:?usage: sbatch train_hoi.sbatch <library dir> [train.py args...]}
shift
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/projects/ll/tianzizh/.uv_cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/projects/ll/tianzizh/.inductor_cache}
export WANDB_PROJECT=${WANDB_PROJECT:-supermimic_hoi}
export PYTHONUNBUFFERED=1  # progress lines reach the job log as they are printed
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
nvidia-smi -L
# Some nodes carry a driver too old for this torch build; train.py would then
# fall back to CPU silently. Stop instead.
uv run --no-sync python -c "import socket, sys, torch; sys.exit(0 if torch.cuda.is_available() else f'no usable CUDA on {socket.gethostname()} (driver too old?); resubmit with --exclude')"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 \
  > "${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu.csv" &
if [ "${PRECHECK:-0}" = 1 ]; then
  # Same GPU, before the run: per-part collection timing, then 300 offline
  # iterations (policy updates start after 20) for the iteration time. Any
  # failure stops the job here (set -e).
  WANDB_MODE=offline uv run --no-sync python tools/hoi/profile_runner.py "$LIB"
  WANDB_MODE=offline uv run --no-sync python src/supermimic/scripts/train.py SuperMimic-HOI-G1 \
    --motion_file "$LIB" --num_envs 4096 "$@" \
    --runner.run_name precheck --runner.max_iterations 300 --runner.log_interval 10 \
    --runner.save_interval 1000000 --runner.wm_warmup_updates 20 --runner.eval_interval 10
fi
uv run --no-sync python src/supermimic/scripts/train.py SuperMimic-HOI-G1 \
  --motion_file "$LIB" \
  --num_envs 4096 \
  --runner.wandb_project "$WANDB_PROJECT" \
  --runner.save_interval 2500 \
  "$@"
