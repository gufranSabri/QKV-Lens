#!/bin/bash
# Extract the QKV feature field for every (dataset x LLM).
# One call per pair writes {data_root}/{dataset}/{llm_alias}/.

DATASETS=(
    triviaqa
    truthfulqa
    coqa
)

MODELS=(
    qwen2.5_7b
    # llama2_7b
    # llama3.1_8b
    # opt_6.7b
)

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        echo "========================================"
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        echo "========================================"

        python detector.py --config configs/$DATASET/$MODEL.yaml extract \
            --set extract.batch_size=8
    done
done
