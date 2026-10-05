#!/bin/bash
# ============================================================================
# QKV-Steer: dependency install
# ============================================================================
# The single source of truth for dependencies -- there is no requirements.txt.
# Every SLURM script and the interactive troubleshooting flow sources this.
#
#   bash scripts/install.sh              # core deps only (exact_match labeling)
#   bash scripts/install.sh --bleurt     # + BLEURT (needed for HalluShift work)
#   bash scripts/install.sh --baselines  # + the training-free baselines' deps
#
# Assumes the venv is already created and ACTIVATED by the caller.
# ============================================================================
set -euo pipefail

WITH_BLEURT=0
WITH_BASELINES=0
for arg in "$@"; do
  [ "$arg" = "--bleurt" ] && WITH_BLEURT=1
  [ "$arg" = "--baselines" ] && WITH_BASELINES=1
done

echo "[install] core dependencies"

# On Compute Canada, --no-index installs from the local wheelhouse and is far
# faster. Fall back to PyPI anywhere else.
PIP_FLAGS=""
if [ -n "${CC_CLUSTER:-}${SLURM_JOB_ID:-}" ] && pip install --no-index --dry-run pip >/dev/null 2>&1; then
  PIP_FLAGS="--no-index"
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

# -- training-free baselines --------------------------------------------------
# Only needed by scripts/experiments/run_all_baselines.sh. Kept behind a flag so an
# extraction/training allocation does not pay for them.
# Two of these baselines run the ORIGINAL authors' code straight out of
# scripts/reproducing_baselines/ (see scripts/baselines/upstream.py), so this installs
# what THEIR modules import at load time, not just what the scoring needs:
#   spacy         selfcheckgpt/modeling_selfcheck.py imports it at module
#                 level; it is also the sentence splitter their README uses.
#   bert_score    same file, for the SelfCheckBERTScore variant we never call.
#                 Not in the Compute Canada wheelhouse -> PyPI, --no-deps.
#   evaluate      lexical_similarity/dataeval/load_worker.py does
#                 evaluate.load('rouge') at import; that IS their Rouge-L.
#   persist_to_disk, pandarallel, ipdb
#                 also imported at module level by that same file.
#   sentencepiece + protobuf   DeBERTa-v3's tokenizer, for SelfCheckGPT-NLI
#   rouge_score   not used upstream; kept for local sanity checks
# NOT installed: openai. lexical_similarity wants the pre-1.0 SDK for a GPT-3.5
# path we never call; upstream.py stubs that import instead of pinning a 2023
# SDK into the environment.
if [ "$WITH_BASELINES" -eq 1 ]; then
  echo "[install] training-free baseline dependencies"
  pip install $PIP_FLAGS rouge_score sentencepiece protobuf
  # spacy resolves from the cluster wheelhouse; from PyPI it tries to build
  # pydantic-core from source (needs Rust) and fails on this cluster.
  pip install $PIP_FLAGS spacy
  # bert_score is not in the wheelhouse. --no-deps because its pins would drag
  # in a conflicting torch/transformers; nothing we call touches them.
  pip install --no-deps bert_score
  # pandarallel, ipdb and evaluate ARE in the wheelhouse -- keep $PIP_FLAGS so
  # they resolve there and cannot perturb the core install.
  pip install $PIP_FLAGS pandarallel ipdb evaluate
  # persist_to_disk is the ONLY one genuinely absent from the wheelhouse, so it
  # is the only line that may reach PyPI. --no-deps is essential: without it
  # pip mixes PyPI metadata with the wheelhouse and backtracks huggingface-hub
  # from 1.30 down to 1.4.1, below the >=1.5.0 that transformers 5.x requires,
  # which breaks every `from transformers import ...` in the run that follows.
  # persist_to_disk's own deps (pandas, numpy, pyyaml) are already installed.
  pip install --no-deps persist_to_disk

  # spaCy's small English model -- SelfCheckGPT's README splits sentences with
  # it. upstream.selfcheck_sentences falls back to a regex if it is missing,
  # so a failed download is a warning, not an error.
  python -m spacy download en_core_web_sm 2>/dev/null \
    || echo "[install] en_core_web_sm unavailable; regex sentence split will be used"
  echo "[install] baseline dependencies OK"
fi

# ── BLEURT ─────────────────────────────────────────────────────────────────
# Only needed for labeling.scheme=bleurt, i.e. the HalluShift comparison.
# The ACT-ViT comparison uses exact_match and needs none of this.
#
# BLEURT is a git install, so it can NOT come from the offline wheelhouse --
# these two lines deliberately drop $PIP_FLAGS and hit the network.
#
# tensorflow-CPU is preferred: the GPU build reserves VRAM at import and
# collides with the torch CUDA context during extraction. NOTE: the
# `pip install tensorflow-cpu` pre-install that would guarantee this is
# currently commented out below, so `pip install ./bleurt` is free to pull
# in plain (GPU) tensorflow via its own setup.py deps -- known gap, not
# fixed here.
if [ "$WITH_BLEURT" -eq 1 ]; then
  echo "[install] BLEURT (TensorFlow-CPU)"

  # pip install tensorflow-cpu

  if [ ! -d bleurt ]; then
    git clone --depth 1 https://github.com/google-research/bleurt.git
  fi
  pip install ./bleurt

  # Checkpoint (~1.5GB). Cached in the repo, so later allocations reuse it.
  mkdir -p models
  if [ ! -d models/BLEURT-20-D12 ]; then
    echo "[install] downloading BLEURT-20-D12 checkpoint"
    wget -q https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
    unzip -q BLEURT-20-D12.zip -d models/
    rm -f BLEURT-20-D12.zip
  fi

  # Fail here, loudly, rather than six hours into a generation run.
  python - <<'PY'
from bleurt import score
s = score.BleurtScorer("models/BLEURT-20-D12")
hi, lo = s.score(references=["Paris", "Paris"], candidates=["Paris", "a fish"])
assert hi > lo, "BLEURT is loaded but scoring nonsense"
print(f"[install] BLEURT OK  (match={hi:.3f}  mismatch={lo:.3f})")
PY
fi

echo "[install] done"
