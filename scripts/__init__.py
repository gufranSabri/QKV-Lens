"""Everything around the pipeline that is not the pipeline itself.

`detector.py` at the repo root runs extract/train/test. This package holds the
machinery that sets that up, sweeps it, and turns `runs/` into the paper's
artifacts. Only setup lives at this level; everything else is grouped by what
it produces:

    install.sh          dependency install -- the single source of truth,
                        sourced by every SLURM script (there is no
                        requirements.txt)
    troubleshooting.sh  the same pipeline annotated step-by-step, to be run
                        command-by-command in an interactive session

    experiments/  sweep drivers (.sh) that call detector.py and populate runs/
    baselines/    the training-free baselines, run on our prompts and splits
    tables/       scan runs/ and emit the paper's tables into docs/tables/
    figures/      figure generators and the shared matplotlib style
    analysis/     statistical analyses and data exports that are not a
                  headline table or figure

Every module here is importable as `scripts.<package>.<module>` and runnable
as `python scripts/<package>/<module>.py` from the repo root; the scripts that
need the repo on `sys.path` put it there themselves.
"""
