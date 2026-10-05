#!/usr/bin/env python3
"""CLI: how early in a response can the detector call a hallucination? (one cell)

    python scripts/analysis/run_forecasting.py \\
        --config configs/triviaqa/llama2_7b.yaml \\
        --checkpoint runs/llama2_7b_triviaqa/best.pt

Runs the prefix sweep for ONE (LLM, dataset) cell and writes, under `docs/`:

    figures/forecasting/cells/<llm>_<dataset>.png (+ .pdf)
    reports/forecasting/cells/<llm>_<dataset>.md
    forecasting_cache/<llm>_<dataset>.npz          (raw trajectories)

The cache is what makes the figures cheap to iterate on: the sweep needs a GPU
and the extracted field, redrawing does not. Pass `--from-cache` to rebuild the
figure and report from a previous sweep without touching either.

For the whole 4x3 grid and the pooled summary figure, use `run_all.py` — that is
the entry point that produces the paper figure. See `forecasting.py` for the
method and `forecasting_report.py` for what is drawn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/analysis/run_forecasting.py` from the repo root
# without an editable install -- scripts/analysis sits beside src/, not
# inside it, so the repo root needs to be on sys.path for both.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.forecasting import (                    # noqa: E402
    compute_trajectories, load_trajectories, save_trajectories,
)
from scripts.analysis.forecasting_report import (             # noqa: E402
    Cell, ForecastPaths, build_cell,
)
from src.config import load_config                             # noqa: E402
from src.utils.logger import get_logger, setup_logging         # noqa: E402
from src.utils.seed import seed_everything                     # noqa: E402

logger = get_logger(__name__)

#: Default cap on evaluated examples. The sweep is T detector passes per
#: example (~64 here), so the full 2490-row held-out split is ~159k passes.
#: 500 gives a trustworthy picture in a fraction of the time; --limit 0 runs all.
DEFAULT_LIMIT = 500


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared with run_all.py, so the two cannot drift apart."""
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
        "--docs-root", default=str(REPO_ROOT / "docs"),
        help="where figures/, tables/ and reports/ are written (default: docs/)",
    )
    p.add_argument(
        "--from-cache", action="store_true",
        help="rebuild figures/reports from a previous sweep's cached "
             "trajectories instead of re-running the detector (no GPU needed)",
    )


def run_cell(
    *,
    config_path: str,
    checkpoint: Path,
    dataset_name: str | None,
    limit: int,
    batch_size: int,
    docs: ForecastPaths,
    from_cache: bool,
    overrides: dict | None = None,
) -> Cell:
    """Sweep (or load) one cell and write its figure and report.

    Returns the `Cell` so `run_all.py` can pool it into the summary without
    recomputing anything.
    """
    cfg = load_config(config_path, overrides=overrides or {})
    seed_everything(cfg.train.seed)

    dataset_name = dataset_name or cfg.dataset.name
    llm_alias = cfg.llm.alias
    key = f"{llm_alias}_{dataset_name}"
    cache = docs.cache_file(key)

    provenance = {
        "checkpoint": str(checkpoint),
        "llm": llm_alias,
        "dataset": dataset_name,
        "limit": limit,
    }

    if from_cache:
        if not cache.exists():
            raise SystemExit(
                f"--from-cache but no cached sweep at {cache}; "
                f"run without --from-cache first"
            )
        trajectories, provenance = load_trajectories(cache)
        logger.info("loaded %d cached trajectories from %s",
                    len(trajectories), cache)
    else:
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        trajectories = compute_trajectories(
            cfg,
            checkpoint=checkpoint,
            dataset_name=dataset_name,
            limit=None if limit == 0 else limit,
            batch_size=batch_size,
        )
        if not trajectories:
            raise SystemExit("no examples evaluated; nothing to report")
        save_trajectories(trajectories, cache, provenance)
        logger.info("cached %d trajectories -> %s", len(trajectories), cache)

    cell = Cell.from_trajectories(trajectories, llm=llm_alias, dataset=dataset_name)
    report = build_cell(cell, docs, provenance)
    logger.info("%s: AUROC %.4f, usable at %.0f%% of the response -> %s",
                key, cell.final_auroc, 100 * cell.earliness, report)
    return cell


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="run_forecasting.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--checkpoint", required=True, help="a trained best.pt")
    p.add_argument("--dataset", default=None, help="defaults to the config's dataset")
    add_common_args(p)
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

    docs = ForecastPaths(Path(args.docs_root))
    docs.mkdirs()

    run_cell(
        config_path=args.config,
        checkpoint=Path(args.checkpoint),
        dataset_name=args.dataset,
        limit=args.limit,
        batch_size=args.batch_size,
        docs=docs,
        from_cache=args.from_cache,
        overrides=overrides,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
