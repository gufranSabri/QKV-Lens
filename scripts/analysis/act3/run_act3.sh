#!/bin/bash
# Act 3 -- explain the gap mechanistically. docs/notes/narrative.md

cd "$(dirname "$0")/../../.." || exit 1

CONFIGS=(
    "llama3.1_8b=configs/triviaqa/llama3.1_8b.yaml"
    "opt_6.7b=configs/triviaqa/opt_6.7b.yaml"
    "qwen2.5_7b=configs/triviaqa/qwen2.5_7b.yaml"
)
MODELS=(llama3.1_8b opt_6.7b qwen2.5_7b)

# Read-only over already-trained representation_ablation.sh runs -- no GPU, instant.
python scripts/analysis/act3/head_alignment.py --models "${MODELS[@]}"

# Reads already-pooled fields off disk (default/hs variants); no GPU, no new extraction.
python scripts/analysis/act3/cell_signal_maps.py --configs "${CONFIGS[@]}" \
    --dataset triviaqa --n-examples 600

python scripts/analysis/act3/augmentation_hypothesis.py --configs "${CONFIGS[@]}" \
    --dataset triviaqa --n-examples 600
