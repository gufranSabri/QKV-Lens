#!/usr/bin/env python3
"""Train + evaluate one or more baselines on already-extracted features.

    python run_training.py --config configs/coqa/llama2_7b.yaml
    python run_training.py --config configs/coqa/llama2_7b.yaml --methods qkv-steer,hallushift
    python run_training.py --config configs/coqa/llama2_7b.yaml --methods hallushift

Each selected method trains on its OWN already-extracted features, using its
OWN training code (this script only dispatches, it does not reimplement any
method):

    qkv-steer   src/train.train -- the (T,L,M,3) field, CNN detector.
    hallushift  reproducing_baselines/hallushift/classifier.train_combined_model
                -- rows written by `detector.py extract --methods ...,hallushift`
                (see src/extract/run_extraction.py), reusing QKV-Steer's own
                train/test split (see build_hallushift_dataframe below) so
                both methods are compared on the identical partition.
    haloscope   not yet implemented (its extraction doesn't exist yet either
                -- see src/extract/run_extraction.py's module docstring).

QKV-Steer is trained FIRST whenever both it and hallushift are requested
together (regardless of --methods order): hallushift's split reuse needs
QKV-Steer's split.json, which src/train.train only writes as a side effect of
training QKV-Steer itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "reproducing_baselines" / "hallushift"))

from src.config import load_config  # noqa: E402
from src.extract.run_extraction import hallushift_dir  # noqa: E402
from src.utils.logger import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

KNOWN_METHODS = ("qkv-steer", "hallushift", "haloscope")


def qkv_steer_run_dir(cfg, dataset_name: str) -> Path:
    """Where QKV-Steer's own training writes split.json/best.pt/etc, matching
    all-datasets_run.sh's naming (`{llm_alias}_{dataset}`) -- NOT
    src/train.default_run_name's `{llm_alias}_{dataset}_{backbone}`, which
    all-datasets_run.sh always overrides via --run-name. Reading the wrong
    one here would silently look for a split.json that a real training run
    never actually wrote.
    """
    return Path(cfg.runs_root) / f"{cfg.llm.alias}_{dataset_name}"


def train_qkv_steer(cfg, dataset_name: str) -> dict:
    from src.train import train

    logger.info("=== training qkv-steer: %s / %s ===", cfg.llm.alias, dataset_name)
    # train()'s run_dir is Path(cfg.runs_root) / run_name -- passing this
    # fixed `{alias}_{dataset}` name is what makes qkv_steer_run_dir's own
    # Path(cfg.runs_root) / f"{alias}_{dataset}" the SAME directory train()
    # just wrote split.json/best.pt into, matching all-datasets_run.sh.
    run_name = f"{cfg.llm.alias}_{dataset_name}"
    return train(cfg, dataset_name, run_name=run_name)


def load_manifest_labels(manifest_path: Path) -> dict[int, int]:
    """{idx: label} from QKV-Steer's manifest.jsonl -- the single source of
    labels for BOTH methods (same response, same BLEURT score, same
    threshold; see src/extract/run_extraction.py's module docstring).

    Excludes n_tokens == 0 examples (response truncated to nothing --
    truncate_runon's "marker appears essentially immediately" case): their
    hallushift row has empty max/min-probability lists (zero decode steps
    captured), and functions.data_preparation's
    `max(a - b for a, b in zip(...))` raises ValueError on an empty zip.
    QKV-Steer's own QKVFieldDataset drops these same examples for the same
    reason (see src/data/dataset.py) -- dropping them here keeps both
    methods training on the identical example set.
    """
    labels: dict[int, int] = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = str(rec.get("label", "")).strip()
            if label in ("0", "1") and rec.get("n_tokens", 0) > 0:
                labels[rec["idx"]] = int(label)
    return labels


def load_qkv_steer_split(split_json_path: Path, manifest_path: Path) -> tuple[list[int], list[int]]:
    """Map QKV-Steer's split.json positions to hallushift's dataframe row
    positions.

    split.json's train/test lists are POSITIONS into QKV-Steer's FILTERED
    example list -- QKVFieldDataset drops any example whose response
    truncated to zero tokens (n_tokens == 0), then make_split() operates on
    that filtered, renumbered list (see src/data/dataset.py,
    src/utils/splits.py). hallushift's dataframe here is built directly from
    manifest.jsonl in idx order WITHOUT dropping empty examples (row i ==
    manifest idx i, see build_hallushift_dataframe), so reusing split.json
    requires:

        split.json position -> QKV-Steer manifest.jsonl idx (drop n_tokens==0,
                                keep order) -> hallushift dataframe row (identity)

    Raises ValueError if the manifest's post-drop example count doesn't
    match split.json's `n` -- that mismatch means the two are for different
    extractions and applying the split would silently select the wrong rows.
    """
    split = json.loads(split_json_path.read_text())

    manifest_records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_records.append(json.loads(line))
    manifest_records.sort(key=lambda r: r["idx"])

    filtered_idxs = [r["idx"] for r in manifest_records if r.get("n_tokens", 0) > 0]

    if len(filtered_idxs) != split["n"]:
        raise ValueError(
            f"split.json says n={split['n']} but {manifest_path} has "
            f"{len(filtered_idxs)} examples with n_tokens > 0. The split's "
            "index positions would not refer to the examples you think they "
            "do -- refusing to guess. Re-check that split_json_path and "
            "manifest_path are for the same (dataset, model) extraction."
        )

    train_rows = [filtered_idxs[p] for p in split["train"]]
    test_rows = [filtered_idxs[p] for p in split["test"]]
    return train_rows, test_rows


def build_hallushift_dataframe(rows_path: Path, manifest_path: Path):
    """hallushift's `data` dataframe (functions.data_preparation's output),
    built from rows.jsonl + QKV-Steer's manifest -- no re-scoring, no
    re-generation.

    Row i is manifest idx i (rows.jsonl is written keyed by idx during
    extraction; sorting by idx here makes row order match manifest order
    exactly, which load_qkv_steer_split's position mapping assumes).
    """
    import pandas as pd

    sys.path.insert(0, str(REPO_ROOT / "reproducing_baselines" / "hallushift"))
    import functions  # reproducing_baselines/hallushift/functions.py

    rows_by_idx: dict[int, list] = {}
    with open(rows_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows_by_idx[rec["idx"]] = rec["row"]

    labels = load_manifest_labels(manifest_path)

    common_idxs = sorted(set(rows_by_idx) & set(labels))
    missing_row = set(labels) - set(rows_by_idx)
    missing_label = set(rows_by_idx) - set(labels)
    if missing_row:
        logger.warning(
            "%d manifest example(s) have no hallushift row (extracted before "
            "hallushift was requested?) -- excluded: %s",
            len(missing_row), sorted(missing_row)[:10],
        )
    if missing_label:
        logger.warning(
            "%d hallushift row(s) have no manifest label -- excluded: %s",
            len(missing_label), sorted(missing_label)[:10],
        )

    df_1 = pd.DataFrame([rows_by_idx[i] for i in common_idxs])
    # Drop the response-text column (the row's last element, build_hallushift_row's
    # `+ [response]`) BEFORE data_preparation -- hal_detection.py's own main()
    # does exactly this (`answers = df.iloc[:, -1]; df = df.iloc[:, :-1]`)
    # right after generation, so data_preparation never sees it there either.
    # Without this, the response string ends up inside train_combined_model's
    # feature matrix X and torch.tensor(X, dtype=torch.float32) raises
    # "could not convert string to float" (confirmed).
    df_1 = df_1.iloc[:, :-1]
    df_2 = pd.DataFrame({"hallucination": [labels[i] for i in common_idxs]})

    with open(rows_path.parent / "geometry.json") as f:
        num_layers = json.load(f)["num_layers"]

    data = functions.data_preparation(df_1, df_2, num_layers)
    return data, num_layers, common_idxs


def train_hallushift(cfg, dataset_name: str, qkv_run_dir: Path) -> dict:
    import torch

    sys.path.insert(0, str(REPO_ROOT / "reproducing_baselines" / "hallushift"))
    import classifier

    hs_dir = hallushift_dir(cfg)
    rows_path = hs_dir / "rows.jsonl"
    manifest_path = cfg.example_dir() / "manifest.jsonl"

    if not rows_path.exists():
        raise FileNotFoundError(
            f"no hallushift rows at {rows_path} -- run "
            f"`detector.py extract --methods qkv-steer,hallushift` first"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"no QKV-Steer manifest at {manifest_path}")

    logger.info("=== training hallushift: %s / %s ===", cfg.llm.alias, dataset_name)

    data, num_layers, common_idxs = build_hallushift_dataframe(rows_path, manifest_path)
    logger.info("hallushift dataframe: %d examples, num_layers=%d", len(data), num_layers)

    split_json_path = qkv_run_dir / "split.json"
    train_idx = test_idx = None
    if split_json_path.exists():
        # common_idxs may be a STRICT SUBSET of manifest idx (rows missing
        # for some examples) -- split.json's positions are into the full
        # filtered manifest, so map to a position within `data` by looking
        # up each mapped idx's position in common_idxs, not assuming they
        # coincide.
        full_train_rows, full_test_rows = load_qkv_steer_split(split_json_path, manifest_path)
        idx_to_row = {idx: row for row, idx in enumerate(common_idxs)}
        train_idx = [idx_to_row[i] for i in full_train_rows if i in idx_to_row]
        test_idx = [idx_to_row[i] for i in full_test_rows if i in idx_to_row]
        logger.info(
            "reusing QKV-Steer split from %s: %d train / %d test rows",
            split_json_path, len(train_idx), len(test_idx),
        )
    else:
        logger.warning(
            "no split.json at %s -- hallushift will use its OWN fresh "
            "train_test_split, NOT the same partition QKV-Steer uses. Train "
            "qkv-steer first (or include it in --methods) for a fair "
            "comparison.",
            split_json_path,
        )

    model, metrics = classifier.train_combined_model(
        data, num_layers, test_size=cfg.train.test_fraction,
        train_idx=train_idx, test_idx=test_idx,
    )

    hs_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), hs_dir / "best.pt")
    metrics_out = {"llm_alias": cfg.llm.alias, "dataset": dataset_name, **metrics}
    (hs_dir / "results.json").write_text(json.dumps(metrics_out, indent=2))
    logger.info("hallushift results saved to %s", hs_dir / "results.json")
    return metrics_out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_training.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--dataset", default=None, help="defaults to the config's dataset")
    parser.add_argument(
        "--methods", default="qkv-steer",
        help="comma-separated: which method(s) to train+eval (default: qkv-steer only)",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VAL",
        help="override a config key, e.g. --set train.epochs=30",
    )
    args = parser.parse_args(argv)

    setup_logging()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = set(methods) - set(KNOWN_METHODS)
    if unknown:
        raise SystemExit(f"unknown method(s) {sorted(unknown)}; known: {KNOWN_METHODS}")
    if "haloscope" in methods:
        raise SystemExit(
            "haloscope training is not implemented yet -- its extraction "
            "doesn't exist either (see src/extract/run_extraction.py)."
        )

    overrides: dict = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        import yaml

        node = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)

    cfg = load_config(args.config, overrides=overrides)
    dataset_name = args.dataset or cfg.dataset.name
    qkv_run_dir = qkv_steer_run_dir(cfg, dataset_name)

    results: dict[str, dict] = {}

    # qkv-steer always runs FIRST when both are requested: hallushift's split
    # reuse needs split.json, which only src/train.train's own run produces.
    if "qkv-steer" in methods:
        results["qkv-steer"] = train_qkv_steer(cfg, dataset_name)

    if "hallushift" in methods:
        results["hallushift"] = train_hallushift(cfg, dataset_name, qkv_run_dir)

    logger.info("done: trained %s", sorted(results))
    for method, metrics in results.items():
        if "auc_roc" in metrics:
            logger.info("  %-12s AUC-ROC=%.4f", method, metrics["auc_roc"])
        elif "val_auroc" in metrics:
            logger.info("  %-12s AUC-ROC=%.4f", method, metrics["val_auroc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
