#!/bin/bash
# Act 4 -- unique information across Q, K, V. docs/notes/narrative.md

cd "$(dirname "$0")/../../.." || exit 1

CONFIGS=(
    "llama3.1_8b=configs/triviaqa/llama3.1_8b.yaml"
    "opt_6.7b=configs/triviaqa/opt_6.7b.yaml"
    "qwen2.5_7b=configs/triviaqa/qwen2.5_7b.yaml"
)

# Reads already-pooled fields off disk (default/hs variants); no GPU, no new extraction.
python scripts/analysis/act4/view_disagreement.py --configs "${CONFIGS[@]}" \
    --dataset triviaqa --n-examples 600
