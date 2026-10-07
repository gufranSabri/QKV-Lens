#!/bin/bash
# QKV-Steer dependency install -- the single source of truth, there is no
# requirements.txt. Every SLURM script and troubleshooting.sh sources this.
# Assumes the venv is already created and activated.
#
#   bash scripts/install.sh              # core deps only (exact_match labeling)
#   bash scripts/install.sh --bleurt     # + BLEURT (needed for HalluShift work)
#   bash scripts/install.sh --baselines  # + the training-free baselines' deps

set -euo pipefail

WITH_BLEURT=0
WITH_BASELINES=0
for arg in "$@"; do
  [ "$arg" = "--bleurt" ] && WITH_BLEURT=1
  [ "$arg" = "--baselines" ] && WITH_BASELINES=1
done

echo "[install] core dependencies"

PIP_FLAGS=""
if [ -n "${CC_CLUSTER:-}${SLURM_JOB_ID:-}" ] && pip install --no-index --dry-run pip >/dev/null 2>&1; then
  PIP_FLAGS="--no-index"   # Compute Canada: install from the local wheelhouse
fi

pip install $PIP_FLAGS --upgrade pip

pip install $PIP_FLAGS \
  torch \
  torchvision \
  transformers \
  datasets \
  numpy \
  scikit-learn \
  pyyaml \
  tqdm \
  matplotlib \
  accelerate \
  PyWavelets \
  pytest

echo "[install] core dependencies OK"

# Only needed by scripts/experiments/run_all_baselines.sh. Two baselines run
# the original authors' code out of reproducing_baselines/, so this installs
# what those modules import at load time, not just what scoring needs.
if [ "$WITH_BASELINES" -eq 1 ]; then
  echo "[install] training-free baseline dependencies"
  pip install $PIP_FLAGS rouge_score sentencepiece protobuf
  # spacy resolves from the cluster wheelhouse; from PyPI it tries to build
  # pydantic-core from source (needs Rust) and fails on this cluster.
  pip install $PIP_FLAGS spacy
  # bert_score is not in the wheelhouse; --no-deps avoids a conflicting torch/transformers pin
  pip install --no-deps bert_score
  pip install $PIP_FLAGS pandarallel ipdb evaluate
  # persist_to_disk is the only one absent from the wheelhouse; --no-deps is
  # essential here -- without it pip backtracks huggingface-hub below the
  # version transformers 5.x requires, breaking every `from transformers import`.
  pip install --no-deps persist_to_disk

  python -m spacy download en_core_web_sm 2>/dev/null \
    || echo "[install] en_core_web_sm unavailable; regex sentence split will be used"
  echo "[install] baseline dependencies OK"
fi

# Only needed for labeling.scheme=bleurt (the HalluShift comparison).
# Git install, so it can NOT come from the offline wheelhouse -- these lines
# deliberately drop $PIP_FLAGS and hit the network.
if [ "$WITH_BLEURT" -eq 1 ]; then
  echo "[install] BLEURT (TensorFlow-CPU)"

  # pip install tensorflow-cpu

  if [ ! -d bleurt ]; then
    git clone --depth 1 https://github.com/google-research/bleurt.git
  fi
  pip install ./bleurt

  mkdir -p models
  if [ ! -d models/BLEURT-20-D12 ]; then
    echo "[install] downloading BLEURT-20-D12 checkpoint"
    wget -q https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
    unzip -q BLEURT-20-D12.zip -d models/
    rm -f BLEURT-20-D12.zip
  fi

  python - <<'PY'
from bleurt import score
s = score.BleurtScorer("models/BLEURT-20-D12")
hi, lo = s.score(references=["Paris", "Paris"], candidates=["Paris", "a fish"])
assert hi > lo, "BLEURT is loaded but scoring nonsense"
print(f"[install] BLEURT OK  (match={hi:.3f}  mismatch={lo:.3f})")
PY
fi

echo "[install] done"
