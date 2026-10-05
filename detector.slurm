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

# HF_TOKEN must be set by the caller's environment (e.g. exported in
# ~/.bashrc or passed via `sbatch --export`) -- never hardcode a token here.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache

bash scripts/install.sh --bleurt --baselines || exit 1

# bash all-datasets_extract.sh
# bash all-datasets_run.sh





# LLaMA-3.1-8B
python detector.py --config configs/triviaqa/llama3.1_8b.yaml train --run-name flatmlp_llama3.1_8b_triviaqa --set model.backbone=flat_mlp
python detector.py --config configs/triviaqa/llama3.1_8b.yaml test --checkpoint runs/flatmlp_llama3.1_8b_triviaqa/best.pt --dataset triviaqa --set model.backbone=flat_mlp

# OPT-6.7B
python detector.py --config configs/triviaqa/opt_6.7b.yaml train --run-name flatmlp_opt_6.7b_triviaqa --set model.backbone=flat_mlp
python detector.py --config configs/triviaqa/opt_6.7b.yaml test --checkpoint runs/flatmlp_opt_6.7b_triviaqa/best.pt --dataset triviaqa --set model.backbone=flat_mlp

# Qwen2.5-7B
python detector.py --config configs/triviaqa/qwen2.5_7b.yaml train --run-name flatmlp_qwen2.5_7b_triviaqa --set model.backbone=flat_mlp
python detector.py --config configs/triviaqa/qwen2.5_7b.yaml test --checkpoint runs/flatmlp_qwen2.5_7b_triviaqa/best.pt --dataset triviaqa --set model.backbone=flat_mlp



# ------------------------------------------------------------------------------------------

# LLaMA-3.1-8B
python detector.py --config configs/triviaqa/llama3.1_8b.yaml train --run-name permL42_llama3.1_8b_triviaqa --set model.layer_permute_seed=42
python detector.py --config configs/triviaqa/llama3.1_8b.yaml test --checkpoint runs/permL42_llama3.1_8b_triviaqa/best.pt --dataset triviaqa --set model.layer_permute_seed=42

# OPT-6.7B
python detector.py --config configs/triviaqa/opt_6.7b.yaml train --run-name permL42_opt_6.7b_triviaqa --set model.layer_permute_seed=42
python detector.py --config configs/triviaqa/opt_6.7b.yaml test --checkpoint runs/permL42_opt_6.7b_triviaqa/best.pt --dataset triviaqa --set model.layer_permute_seed=42

# Qwen2.5-7B
python detector.py --config configs/triviaqa/qwen2.5_7b.yaml train --run-name permL42_qwen2.5_7b_triviaqa --set model.layer_permute_seed=42
python detector.py --config configs/triviaqa/qwen2.5_7b.yaml test --checkpoint runs/permL42_qwen2.5_7b_triviaqa/best.pt --dataset triviaqa --set model.layer_permute_seed=42