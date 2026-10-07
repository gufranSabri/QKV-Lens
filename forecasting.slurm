#!/bin/bash
# Prefix-forecasting sweep over every trained detector, then the paper
# figure. Needs runs/<llm>_<dataset>/best.pt for every cell -- never trains.
# Cost scales with --limit x median response length; --limit 0 (full
# held-out split) is ~5x the 500-example default and needs --time raised.
# Raw trajectories cache under docs/forecasting_cache/, so redrawing figures
# afterwards (`run_all.py --summary-only`) needs no GPU.
#
#   sbatch forecasting.slurm
#SBATCH --job-name=forecast
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -uo pipefail

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache

bash scripts/install.sh || exit 1

python scripts/analysis/run_all.py --limit 500

echo "figures  -> docs/figures/forecasting/"
echo "tables   -> docs/tables/forecasting/"
echo "reports  -> docs/reports/forecasting/"
