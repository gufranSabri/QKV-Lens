# Everything around the pipeline that is not the pipeline itself
# (detector.py at the repo root runs extract/train/test).
#
#   install.sh / troubleshooting.sh   dependency install / annotated pipeline
#   experiments/  sweep drivers (.sh) that call detector.py and populate runs/
#   baselines/    training-free baselines, run on our prompts and splits
#   tables/       scan runs/ and emit the paper's tables into docs/tables/
#   figures/      figure generators and the shared matplotlib style
#   analysis/     statistical analyses and data exports
