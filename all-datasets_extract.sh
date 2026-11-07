#!/bin/bash
# Phase 1 (extract, GPU) for every (dataset x LLM), then phase 2 (label, judge API, no GPU); each skips finished work.
#
#   bash all-datasets_extract.sh            # both phases
#   bash all-datasets_extract.sh extract    # phase 1 only
#   bash all-datasets_extract.sh label      # phase 2 only
#   LABEL_MODE=export bash all-datasets_extract.sh label   # export / import CSV sheets instead of the API
#   VARIANTS=all bash all-datasets_extract.sh extract

PHASE=${1:-all}
case "$PHASE" in
    extract|label|all) ;;
    *) echo "usage: $0 [extract|label|all]"; exit 1 ;;
esac

DATASETS=(
    triviaqa
    truthfulqa
    coqa
)

MODELS=(
    qwen2.5_7b
    llama2_7b
    llama3.1_8b
    opt_6.7b
)

if [[ "$PHASE" != "label" ]]; then
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo "========================================"
            echo "Extract | Dataset: $DATASET | Model: $MODEL"
            echo "========================================"

            python detector.py --config "configs/$DATASET/$MODEL.yaml" extract \
                --set extract.batch_size=8 ${VARIANTS:+--variants "$VARIANTS"}
        done
    done
fi

if [[ "$PHASE" != "extract" ]]; then
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo "========================================"
            echo "Label | Dataset: $DATASET | Model: $MODEL"
            echo "========================================"

            python detector.py --config "configs/$DATASET/$MODEL.yaml" label --mode "${LABEL_MODE:-api}"
        done
    done
fi
