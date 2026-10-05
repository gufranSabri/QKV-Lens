#!/usr/bin/env python3
"""Prefix-forecasting over the whole run grid, plus the pooled summary figure.

    python scripts/analysis/run_all.py                  # every run under runs/
    python scripts/analysis/run_all.py llama2_7b_coqa   # just this one
    python scripts/analysis/run_all.py --limit 0        # the full held-out split
    python scripts/analysis/run_all.py --from-cache     # redraw, no GPU needed

This is the entry point that produces the paper figure. It sweeps every
(LLM, dataset) cell, writes each cell's own figure and report, and then pools
them into:

    docs/figures/forecasting/forecasting_summary.png (+ .pdf)
    docs/tables/forecasting/forecasting_summary.csv
    docs/tables/forecasting/forecasting_curves.csv
    docs/reports/forecasting/report.md

For each run directory the (dataset, LLM) config is reconstructed from that
run's own saved `config.json` -- not by parsing the run-name string -- so it is
exact even for a run whose name doesn't match `configs/{dataset}/{llm}.yaml`
verbatim (e.g. a `-old` suffixed rerun).

A run with no `best.pt` is skipped (training never finished). A cell that fails
is skipped with a warning and the summary is still built from the rest, so one
bad checkpoint does not cost the whole grid -- the report states how many cells
it covers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.forecasting_report import (            # noqa: E402
    ForecastPaths, Grid, build_summary,
)
from scripts.analysis.run_forecasting import (               # noqa: E402
    add_common_args, run_cell,
)
from src.utils.logger import get_logger, setup_logging       # noqa: E402

logger = get_logger(__name__)


def config_path_for(run_dir: Path) -> tuple[Path, str]:
    """The configs/{dataset}/{llm_alias}.yaml that produced this run, and its dataset.

    Read from the run's own config.json rather than the run directory's
    name, so it is correct even when the name doesn't follow the
    `{llm_alias}_{dataset}` convention.
    """
    config_json = run_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"no config.json in {run_dir}")
    saved = json.loads(config_json.read_text())
    dataset = saved["dataset"]["name"]
    llm_alias = saved["llm"]["alias"]
    path = REPO_ROOT / "configs" / dataset / f"{llm_alias}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{run_dir}: config.json points to dataset={dataset!r} "
            f"llm_alias={llm_alias!r}, but {path} does not exist"
        )
    return path, dataset


def discover_runs(runs_root: Path) -> list[Path]:
    return sorted(p for p in runs_root.iterdir() if p.is_dir())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="run_all.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "run_name", nargs="?", default=None,
        help="only analyze runs/<run_name> (default: every run under --runs-root)",
    )
    p.add_argument("--runs-root", default="runs", help="default: runs")
    p.add_argument(
        "--summary-only", action="store_true",
        help="skip the sweep entirely and rebuild only the pooled summary from "
             "whatever cells are already cached",
    )
    add_common_args(p)
    args = p.parse_args(argv)

    setup_logging()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = REPO_ROOT / runs_root

    docs = ForecastPaths(Path(args.docs_root))
    docs.mkdirs()

    if args.run_name:
        run_dirs = [runs_root / args.run_name]
        if not run_dirs[0].is_dir():
            raise SystemExit(f"no such run: {run_dirs[0]}")
    else:
        run_dirs = discover_runs(runs_root)

    cells, n_skip, n_fail = [], 0, 0
    for run_dir in run_dirs:
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            logger.info("[skip] %s: no best.pt (training incomplete)", run_dir.name)
            n_skip += 1
            continue

        try:
            config_path, dataset = config_path_for(run_dir)
        except (FileNotFoundError, KeyError) as e:
            logger.warning("[fail] %s: %s", run_dir.name, e)
            n_fail += 1
            continue

        # --summary-only means "use what is cached"; a cell with no cache is
        # simply absent from the summary rather than an error.
        if args.summary_only:
            continue

        logger.info("[run] %s (config=%s)", run_dir.name, config_path)
        try:
            cells.append(run_cell(
                config_path=str(config_path),
                checkpoint=checkpoint,
                dataset_name=dataset,
                limit=args.limit,
                batch_size=args.batch_size,
                docs=docs,
                from_cache=args.from_cache,
            ))
        except (SystemExit, RuntimeError, ValueError, OSError) as e:
            logger.warning("[fail] %s: %s", run_dir.name, e)
            n_fail += 1

    if args.summary_only:
        cells = _cells_from_cache(docs)

    if not cells:
        logger.error("no cells succeeded; nothing to summarise")
        return 1

    build_summary(Grid(cells), docs)
    logger.info("done: %d cells, %d skipped, %d failed", len(cells), n_skip, n_fail)
    return 1 if n_fail else 0


def _cells_from_cache(docs: ForecastPaths) -> list:
    """Every cached sweep, rebuilt into Cells without touching a checkpoint."""
    from scripts.analysis.forecasting import load_trajectories
    from scripts.analysis.forecasting_report import Cell

    cells = []
    for npz in sorted(docs.cache.glob("*.npz")):
        trajs, prov = load_trajectories(npz)
        cells.append(Cell.from_trajectories(
            trajs, llm=prov["llm"], dataset=prov["dataset"]
        ))
        logger.info("[cache] %s (%d examples)", npz.stem, len(trajs))
    return cells


if __name__ == "__main__":
    sys.exit(main())
