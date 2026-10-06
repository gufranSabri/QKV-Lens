#!/bin/bash
# Train + test the detector on every (dataset x LLM) with default settings.
# Run all-datasets_extract.sh first so the feature fields are on disk.

# RERUN=0 (default): skip a train/test call if its output already exists --
# train is skipped when runs/$RUN_NAME/results.json exists (only written after
# a run finishes, so a crashed/partial run is correctly NOT treated as done);
# test is skipped when its checkpoint's test_$DATASET.json already exists.
# RERUN=1: always run everything, overwriting prior results.
RERUN=${RERUN:-0}

DATASETS=(
    triviaqa
    truthfulqa
    coqa
)

MODELS=(
    llama2_7b
    llama3.1_8b
    qwen2.5_7b
    opt_6.7b
)

# run_train <run_name> [--set key=val ...]
run_train() {
    local run_name="$1"; shift
    if [[ "$RERUN" != "1" && -f "runs/${run_name}/results.json" ]]; then
        echo "  [skip] train runs/${run_name} (results.json already exists; RERUN=1 to force)"
        return
    fi
    python detector.py --config configs/$DATASET/$MODEL.yaml train --run-name "${run_name}" "$@"
}

# run_test <run_name> [--set key=val ...]
run_test() {
    local run_name="$1"; shift
    local dest="runs/${run_name}/test_${DATASET}.json"
    if [[ "$RERUN" != "1" && -f "$dest" ]]; then
        echo "  [skip] test runs/${run_name} ($dest already exists; RERUN=1 to force)"
        return
    fi
    python detector.py --config configs/$DATASET/$MODEL.yaml test --checkpoint "runs/${run_name}/best.pt" --dataset $DATASET "$@"
}

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        echo "========================================"
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        echo "========================================"

        RUN_NAME=${MODEL}_${DATASET}

        run_train "$RUN_NAME"
        run_test "$RUN_NAME"

    done
done

python detector.py --config configs/triviaqa/llama3.1_8b.yaml train --run-name "test" --set model.backbone=conv_segment
