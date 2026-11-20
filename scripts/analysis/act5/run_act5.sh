#!/bin/bash
# Act 5 -- tie the linear-probe diagnostics (Acts 1-4) back to the real
# detector. docs/notes/narrative.md

cd "$(dirname "$0")/../../.." || exit 1

# Reads already-trained runs/repr_* results and Act 2's PCA-capacity control
# CSV off disk; no GPU, no new extraction or training.
python scripts/analysis/act5/probe_vs_detector.py
