from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines.common import METHODS as BASELINE_METHODS  # noqa: E402
from src.extract.run_extraction import parse_meta  # noqa: E402

# Top-level entries under data_root that are method outputs, not (dataset, llm_alias) trees.
EXCLUDED_TOP_LEVEL = {"hallushift", "baselines", *BASELINE_METHODS}


def load_data_root(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    return Path(cfg["data_root"])


def discover_pairs(root: Path) -> list[tuple[str, str]]:
    pairs = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in EXCLUDED_TOP_LEVEL:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if model_dir.is_dir() and (model_dir / "manifest.jsonl").exists():
                pairs.append((dataset_dir.name, model_dir.name))
    return pairs


def read_manifest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["idx"])
    return records


def export_pair(dataset: str, llm_alias: str, root: Path, out_dir: Path) -> int:
    pair_dir = root / dataset / llm_alias
    records = read_manifest(pair_dir / "manifest.jsonl")

    out_path = out_dir / f"{dataset}_{llm_alias}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "prompt", "gold", "generated_response", "bleurt_score", "hallucination_label"])
        for rec in records:
            meta = parse_meta(pair_dir / rec["dir"] / "meta.txt")
            writer.writerow([
                rec["idx"],
                meta.get("prompt", ""),
                meta.get("gold", ""),
                meta.get("response", ""),
                rec.get("score", ""),
                rec.get("label", ""),
            ])
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "exports"))
    args = parser.parse_args()

    root = load_data_root(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, llm_alias in discover_pairs(root):
        n = export_pair(dataset, llm_alias, root, out_dir)
        print(f"{dataset}/{llm_alias}: wrote {n} rows -> {out_dir / f'{dataset}_{llm_alias}.csv'}")
