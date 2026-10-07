#!/bin/bash
# Batch job for the detector stage (extract + train + test).
#SBATCH --job-name=mains
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=12:00:00

set -uo pipefail

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# HF_TOKEN must be set by the caller's environment (e.g. exported in
# ~/.bashrc or passed via `sbatch --export`) -- never hardcode a token here.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache

bash scripts/install.sh --bleurt --baselines || exit 1

# bash all-datasets_extract.sh
# bash all-datasets_run.sh

bash scripts/experiments/all_experiments.sh