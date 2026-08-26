#!/usr/bin/env python3
"""CLI: how early in a response can the detector call a hallucination?

    python analysis/run_forecasting.py \\
        --config configs/triviaqa/llama2_7b.yaml \\
        --checkpoint runs/llama2_7b_triviaqa/best.pt

Writes `<run_dir>/forecasting/` next to the checkpoint: report.md, five figures,
and trajectories.npz. See analysis/forecasting.py for the method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python analysis/run_forecasting.py` from the repo root without an
# editable install -- the analysis package sits beside src/, not inside it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecasting import compute_trajectories          # noqa: E402
from analysis.forecasting_report import build_all              # noqa: E402
from src.config import load_config                             # noqa: E402
from src.utils.logger import get_logger, setup_logging         # noqa: E402
from src.utils.seed import seed_everything                     # noqa: E402

logger = get_logger(__name__)

#: Default cap on evaluated examples. The sweep is T detector passes per
#: example (~64 here), so the full 2490-row held-out split is ~159k passes.
#: 500 gives a trustworthy picture in a fraction of the time; --limit 0 runs all.
DEFAULT_LIMIT = 500


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="run_forecasting.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--checkpoint", required=True, help="a trained best.pt")
    p.add_argument("--dataset", default=None, help="defaults to the config's dataset")
    p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"evaluate at most this many held-out examples "
             f"(default {DEFAULT_LIMIT}; 0 = the whole held-out split)",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="examples per detector call at a fixed prefix length (default 64)",
    )
    p.add_argument(
        "--out", default=None,
        help="output directory (default: <checkpoint dir>/forecasting)",
    )
    p.add_argument(
        "--set", action="append", metavar="KEY=VAL", default=[],
        help="override a config key, e.g. --set data_root=/path/to/data",
    )
    args = p.parse_args(argv)

    setup_logging()

    overrides: dict = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        import yaml

        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)

    cfg = load_config(args.config, overrides=overrides)
    seed_everything(cfg.train.seed)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    dataset_name = args.dataset or cfg.dataset.name
    out_dir = Path(args.out) if args.out else checkpoint.parent / "forecasting"

    trajectories = compute_trajectories(
        cfg,
        checkpoint=checkpoint,
        dataset_name=dataset_name,
        limit=None if args.limit == 0 else args.limit,
        batch_size=args.batch_size,
    )
    if not trajectories:
        raise SystemExit("no examples evaluated; nothing to report")

    report = build_all(
        trajectories,
        out_dir,
        provenance={
            "checkpoint": str(checkpoint),
            "llm": cfg.llm.alias,
            "dataset": dataset_name,
        },
    )
    logger.info("done -> %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
