#!/bin/bash
# ============================================================================
# QKV-Steer: interactive pipeline
# ============================================================================
# Run this file command-by-command in an interactive SLURM terminal. It is not
# meant to be executed as a whole -- copy-paste one section at a time and look
# at the output before moving on.
# ============================================================================


# ── STEP 0: ALLOCATE ───────────────────────────────────────────────────────
# Run ONE of these.

salloc --gpus-per-node=l40s:1 --cpus-per-task=24 --mem=24G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:1 --cpus-per-task=8 --mem=16G --time=1:00:00 --account=aip-lsigal


# Longer / bigger:
# salloc --gpus-per-node=h100:1 --cpus-per-task=24 --mem=60G --time=1:00:00 --account=aip-lsigal


# ── STEP 1: ENVIRONMENT ────────────────────────────────────────────────────
# There is no requirements.txt -- scripts/install.sh is the single source of
# truth for dependencies, and every SLURM script calls it too.

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# HF_TOKEN must already be set in your environment -- never hardcode a
# token here.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache

# Core deps only -- enough for the ACT-ViT comparison (exact_match labels):
bash scripts/install.sh

# ...OR with BLEURT, required for the HalluShift comparison. Adds TensorFlow-CPU
# and downloads the ~1.5GB BLEURT-20-D12 checkpoint into models/ (cached, so
# later allocations reuse it). It self-tests and fails loudly if broken.
# bash scripts/install.sh --bleurt

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name())"


bash all-datasets_extract.sh
bash all=datasets_run.sh