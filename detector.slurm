#!/bin/bash
# Batch job for the DETECTOR stage (extract + train + test). This fits f_theta,
# which the steering stage then consumes; it does not steer anything itself.
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
#SBATCH --time=1-00:00:00

set -uo pipefail

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache
export HF_TOKEN=REDACTED_HF_TOKEN

bash scripts/install.sh --bleurt --baselines || exit 1

# bash all-datasets_extract.sh
bash all-datasets_run.sh

# DATASETS=coqa MODELS=llama2_7b BATCH=2 bash scripts/experiments/run_all_baselines.sh 
# RERUN=1 DATASETS=coqa MODELS=opt_6.7b BATCH=2 bash scripts/experiments/run_all_baselines.sh


python detector.py --config configs/coqa/opt_6.7b.yaml train --run-name opt_6.7b_coqa
python detector.py --config configs/coqa/opt_6.7b.yaml test --checkpoint runs/opt_6.7b_coqa/best.pt --dataset coqa
DATASETS=coqa MODELS=opt_6.7b BATCH=2 bash scripts/experiments/run_all_baselines.sh
