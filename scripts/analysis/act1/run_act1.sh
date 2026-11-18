#!/bin/bash
# Act 1 -- the projection itself adds nothing; the gap enters at pooling.
# docs/notes/narrative.md

cd "$(dirname "$0")/../../.." || exit 1

CONFIGS=(
    "llama3.1_8b=configs/triviaqa/llama3.1_8b.yaml"
    "opt_6.7b=configs/triviaqa/opt_6.7b.yaml"
    "qwen2.5_7b=configs/triviaqa/qwen2.5_7b.yaml"
)

python scripts/analysis/act1/spectral_rank.py --configs "${CONFIGS[@]}"
python scripts/analysis/act1/probe_parity.py --configs "${CONFIGS[@]}" --n-examples 300
