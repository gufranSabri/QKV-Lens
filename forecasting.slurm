#!/bin/bash
# Prefix-forecasting sweep over every trained detector, then the paper figure.
#
# Needs `runs/<llm>_<dataset>/best.pt` for every cell -- this job only runs the
# detector over prefixes, it never trains. Run detector.slurm first.
#
#   sbatch forecasting.slurm
#
# The sweep is T detector passes per example, so cost scales with --limit x the
# median response length. 500 examples/cell across 12 cells is comfortable in
# the walltime below; --limit 0 (the whole held-out split, ~2.5k/cell) is
# roughly 5x that and needs --time raised to match.
#
# Every cell's raw trajectories are cached under docs/forecasting_cache/, so
# re-drawing the figures afterwards needs no GPU at all:
#
#   python scripts/analysis/run_all.py --summary-only
#
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

# Sweeps every run under runs/, writes each cell's figure + report, then pools
# them into docs/figures/forecasting/forecasting_summary.png and
# docs/reports/forecasting/report.md.
python scripts/analysis/run_all.py --limit 500

echo "figures  -> docs/figures/forecasting/"
echo "tables   -> docs/tables/forecasting/"
echo "reports  -> docs/reports/forecasting/"
