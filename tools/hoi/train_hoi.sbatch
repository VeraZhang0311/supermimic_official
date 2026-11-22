#!/bin/bash
# Full HOI teacher training on one L40, logged online to W&B.
#
#   sbatch tools/hoi/train_hoi.sbatch <packed HOI library dir> [extra train.py args...]
#
# Checkpoints go to $RUN_DIR (default logs/SuperMimic-HOI-G1/slurm-<job id>).
# A preempted job is requeued by Slurm and continues from $RUN_DIR/trainer_state.pt,
# written every checkpoint and on the SIGTERM before the kill; to continue a run in
# a new job, submit with --export=ALL,RUN_DIR=<its log dir>.
#
# Python environment: $VENV (default .venv). .venv's torch is built for CUDA 13.2
# and needs NVIDIA driver 580 or newer (ll-l40-2, al-l40s-0). VENV=.venv-cu12
# (tools/hoi/build_venv_cu12.sh: torch 2.14.1+cu126, warp +cu12) runs on driver
# 535 (ll-l40-0, ll-l40-1, checked on an A40) and newer, but not on 525.
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
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail
# Slurm signals every process of the job, then kills them 30 s (KillWait) later.
# train.py saves its trainer state on SIGTERM; this script must not exit first.
trap '' TERM
LIB=${1:?usage: sbatch train_hoi.sbatch <library dir> [train.py args...]}
shift
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/projects/ll/tianzizh/.uv_cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/projects/ll/tianzizh/.inductor_cache}
export WANDB_PROJECT=${WANDB_PROJECT:-supermimic_hoi}
export PYTHONUNBUFFERED=1  # progress lines reach the job log as they are printed
export UV_PROJECT_ENVIRONMENT=${VENV:-.venv}  # what `uv run` runs in
if [ "$UV_PROJECT_ENVIRONMENT" != .venv ]; then
  # Kernels compiled by another build of the same warp version must not be reused.
  export WARP_CACHE_PATH=${WARP_CACHE_PATH:-/mnt/projects/ll/tianzizh/.warp_cache${UV_PROJECT_ENVIRONMENT#.venv}}
fi
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
RUN_DIR=${RUN_DIR:-logs/SuperMimic-HOI-G1/slurm-$SLURM_JOB_ID}
echo "[$(date)] start $((${SLURM_RESTART_COUNT:-0} + 1)) on $(hostname), run dir $RUN_DIR, env $UV_PROJECT_ENVIRONMENT"
nvidia-smi -L
# A driver too old for this torch build makes train.py fall back to CPU silently
# (CUDA 13 builds), or fail at the first allocation while torch.cuda.is_available()
# still says yes (.venv-cu12 on driver 525). Stop here instead.
uv run --no-sync python -c "import socket, sys, torch; torch.cuda.is_available() or sys.exit(f'no usable CUDA on {socket.gethostname()} (driver too old?); resubmit with VENV=.venv-cu12 or --exclude'); torch.zeros(1, device='cuda')"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 \
  >> "${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu.csv" &
if [ "${PRECHECK:-0}" = 1 ] && [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then
  # Same GPU, before the run: per-part collection timing, then 300 offline
  # iterations (policy updates start after 20) for the iteration time, then a
  # resume from their step-200 trainer state that must save again on SIGTERM.
  # Any failure stops the job here (set -e).
  WANDB_MODE=offline uv run --no-sync python tools/hoi/profile_runner.py "$LIB"
  PRE_DIR=logs/SuperMimic-HOI-G1/precheck-$SLURM_JOB_ID
  PRE=(src/supermimic/scripts/train.py SuperMimic-HOI-G1 --motion_file "$LIB" --num_envs 4096 "$@"
    --runner.run_name precheck --runner.log_interval 10 --runner.save_interval 100
    --runner.wm_warmup_updates 20 --runner.eval_interval 10 --runner.log_dir "$PRE_DIR")
  WANDB_MODE=offline uv run --no-sync python "${PRE[@]}" --runner.max_iterations 300
  WANDB_MODE=offline "$UV_PROJECT_ENVIRONMENT/bin/python" "${PRE[@]}" --runner.max_iterations 100000 \
    > "$PRE_DIR/resume.log" 2>&1 &
  pid=$!
  until grep -q 'Learning iteration 40/' "$PRE_DIR/resume.log" || ! kill -0 $pid 2>/dev/null; do
    sleep 5
  done
  kill -TERM $pid 2>/dev/null || true
  status=0
  wait $pid || status=$?
  cat "$PRE_DIR/resume.log"
  if [ $status != 0 ] || ! grep -q 'Resuming' "$PRE_DIR/resume.log" \
    || ! grep -q 'SIGTERM: saved' "$PRE_DIR/resume.log"; then
    echo "precheck: the resume check failed (exit $status)"
    exit 1
  fi
fi
uv run --no-sync python src/supermimic/scripts/train.py SuperMimic-HOI-G1 \
  --motion_file "$LIB" \
  --num_envs 4096 \
  --runner.wandb_project "$WANDB_PROJECT" \
  --runner.save_interval 2500 \
  --runner.log_dir "$RUN_DIR" \
  "$@"
