#!/bin/bash
# Build .venv-cu12 beside .venv: the same packages, but torch 2.14.1+cu126 and
# warp 1.19.0+cu12 instead of the CUDA 13 builds. torch 2.15.0 is published for
# CUDA 13.2 only, which needs NVIDIA driver 580 or newer. This env runs on 535
# (checked on an A40 with 535.216.01, the ll-l40-0/1 driver) and 550; on 525.60
# torch sees the GPU but its first allocation fails. Use it with
#
#   sbatch --export=ALL,VENV=.venv-cu12,... tools/hoi/train_hoi.sbatch ...
#
# Run from the repository root, with .venv already synced. Every package other
# than torch, its CUDA libraries, triton and warp is pinned to its .venv version.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/projects/ll/tianzizh/.uv_cache}
V=.venv-cu12
TORCH=${TORCH:-2.14.1}
WARP_WHEEL=https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl
if [ -e $V ]; then echo "$V exists; remove it first"; exit 1; fi
uv venv $V --python "$(readlink -f .venv/bin/python)"
uv pip freeze --python .venv/bin/python > $V/freeze-cu13.txt
grep -vE '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z-]+|warp-lang|mjlab|supermimic)(==| @)' $V/freeze-cu13.txt \
  | grep -v '^-e ' > $V/rest.txt
uv pip install --python $V/bin/python "torch==$TORCH" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python $V/bin/python --no-deps "warp-lang @ $WARP_WHEEL"
uv pip install --python $V/bin/python --no-deps -r $V/rest.txt
uv pip install --python $V/bin/python --no-deps -e ../mjlab_research -e .
uv pip check --python $V/bin/python
$V/bin/python -c "import torch, warp, mujoco_warp, mjlab, supermimic; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'warp', warp.__version__)"
