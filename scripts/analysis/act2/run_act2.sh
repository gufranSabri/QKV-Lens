#!/bin/bash
# Act 2, control 4 -- PCA-matched capacity: does QKV's edge over H survive
# once both are truncated to the same number of usable dimensions?
# docs/notes/narrative.md

cd "$(dirname "$0")/../../.." || exit 1

CONFIGS=(
    "llama3.1_8b=configs/triviaqa/llama3.1_8b.yaml"
    "opt_6.7b=configs/triviaqa/opt_6.7b.yaml"
    "qwen2.5_7b=configs/triviaqa/qwen2.5_7b.yaml"
)

python scripts/analysis/act2/pca_capacity_control.py --configs "${CONFIGS[@]}" \
    --dataset triviaqa --n-examples 600
