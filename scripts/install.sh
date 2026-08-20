#!/bin/bash
# ============================================================================
# QKV-Steer: dependency install
# ============================================================================
# The single source of truth for dependencies -- there is no requirements.txt.
# Every SLURM script and the interactive troubleshooting flow sources this.
#
#   bash scripts/install.sh              # core deps only (exact_match labeling)
#   bash scripts/install.sh --bleurt     # + BLEURT (needed for HalluShift work)
#
# Assumes the venv is already created and ACTIVATED by the caller.
# ============================================================================
set -euo pipefail

WITH_BLEURT=0
for arg in "$@"; do
  [ "$arg" = "--bleurt" ] && WITH_BLEURT=1
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

# ── BLEURT ─────────────────────────────────────────────────────────────────
# Only needed for labeling.scheme=bleurt, i.e. the HalluShift comparison.
# The ACT-ViT comparison uses exact_match and needs none of this.
#
# BLEURT is a git install, so it can NOT come from the offline wheelhouse --
# these two lines deliberately drop $PIP_FLAGS and hit the network.
#
# tensorflow-CPU on purpose: the GPU build reserves VRAM at import and collides
# with the torch CUDA context during extraction.
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
