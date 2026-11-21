#!/bin/bash
# Run one command from the repository root on an L40, e.g.
#   sbatch tools/hoi/gpu_job.sbatch python tools/hoi/replay_reference.py <lib>
#SBATCH --job-name=hoi-gpu
#SBATCH --account=liu-account
#SBATCH --partition=liu-compute
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log

set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/projects/ll/tianzizh/.uv_cache}
export WANDB_MODE=offline
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
nvidia-smi -L
uv run --no-sync "$@"
