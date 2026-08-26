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

# truthfulqa/triviaqa/coqa all mirror HalluShift's single-split protocol
# (src/extract/datasets.py SPLIT_SOURCES): each loads ONE upstream split and
# gets a stratified held-out slice carved out of it at train time, so there is
# no separately extracted `<name>_test` corpus to pull here.

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
