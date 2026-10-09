from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines.common import METHODS as BASELINE_METHODS  # noqa: E402


def load_data_root(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    return Path(cfg["data_root"])


def read_manifest(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stats_for_pair(dataset: str, llm_alias: str, root: Path) -> dict:
    pair_dir = root / dataset / llm_alias
    manifest_path = pair_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {
            "dataset": dataset, "llm": llm_alias,
            "total": 0, "labeled": 0, "unlabeled": 0,
            "hallucinated": 0, "hallucination_rate": None,
            "mean_score": None, "mean_n_tokens": None,
            "missing_manifest": True,
        }

    records = read_manifest(manifest_path)
    total = len(records)
    labeled = [r for r in records if str(r.get("label", "-1")) in ("0", "1")]
    unlabeled = total - len(labeled)

    n_hall = sum(1 for r in labeled if int(r["label"]) == 1)
    hall_rate = n_hall / len(labeled) if labeled else None
    mean_score = (
        sum(r["score"] for r in labeled) / len(labeled) if labeled else None
    )
    mean_n_tokens = (
        sum(r["n_tokens"] for r in records) / total if total else None
    )

    return {
        "dataset": dataset, "llm": llm_alias,
        "total": total, "labeled": len(labeled), "unlabeled": unlabeled,
        "hallucinated": n_hall, "hallucination_rate": hall_rate,
        "mean_score": mean_score, "mean_n_tokens": mean_n_tokens,
        "missing_manifest": False,
    }


# Top-level entries under data_root that are method outputs, not (dataset, llm_alias) trees.
EXCLUDED_TOP_LEVEL = {"hallushift", "baselines", *BASELINE_METHODS}


def discover_pairs(root: Path) -> list[tuple[str, str]]:
    pairs = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in EXCLUDED_TOP_LEVEL:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if model_dir.is_dir():
                pairs.append((dataset_dir.name, model_dir.name))
    return pairs


def fmt_pct(x: float | None) -> str:
    return f"{100 * x:.1f}%" if x is not None else "—"


def fmt_num(x: float | None, digits=1) -> str:
    return f"{x:.{digits}f}" if x is not None else "—"


def build_report(rows: list[dict]) -> str:
    lines = []
    lines.append("# QKV-Lens migrated corpus stats\n")
    lines.append(
        "Computed from `manifest.jsonl` under the migrated "
        "`{data_root}/{dataset}/{llm_alias}/` tree. Pairs whose BLEURT "
        "relabeling hasn't finished yet show `unlabeled` > 0 and an empty "
        "hallucination rate for the missing rows.\n"
    )

    lines.append("## Per (dataset, model)\n")
    lines.append(
        "| Dataset | Model | Total | Labeled | Unlabeled | Hallucinated | "
        "Hallucination rate | Mean BLEURT | Mean response tokens |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["missing_manifest"]:
            lines.append(
                f"| {r['dataset']} | {r['llm']} | — | — | — | — | "
                f"— | — | — (no manifest.jsonl found) |"
            )
            continue
        lines.append(
            f"| {r['dataset']} | {r['llm']} | {r['total']} | {r['labeled']} | "
            f"{r['unlabeled']} | {r['hallucinated']} | "
            f"{fmt_pct(r['hallucination_rate'])} | "
            f"{fmt_num(r['mean_score'], 3)} | {fmt_num(r['mean_n_tokens'], 1)} |"
        )

    lines.append("\n## Per dataset (across all models)\n")
    lines.append("| Dataset | Total | Labeled | Hallucinated | Hallucination rate |")
    lines.append("|---|---:|---:|---:|---:|")
    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)
    for dataset, group in by_dataset.items():
        total = sum(r["total"] for r in group)
        labeled = sum(r["labeled"] for r in group)
        hall = sum(r["hallucinated"] for r in group)
        rate = hall / labeled if labeled else None
        lines.append(
            f"| {dataset} | {total} | {labeled} | {hall} | {fmt_pct(rate)} |"
        )

    lines.append("\n## Per model (across all datasets)\n")
    lines.append("| Model | Total | Labeled | Hallucinated | Hallucination rate |")
    lines.append("|---|---:|---:|---:|---:|")
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["llm"], []).append(r)
    for llm, group in by_model.items():
        total = sum(r["total"] for r in group)
        labeled = sum(r["labeled"] for r in group)
        hall = sum(r["hallucinated"] for r in group)
        rate = hall / labeled if labeled else None
        lines.append(
            f"| {llm} | {total} | {labeled} | {hall} | {fmt_pct(rate)} |"
        )

    total = sum(r["total"] for r in rows)
    labeled = sum(r["labeled"] for r in rows)
    unlabeled = sum(r["unlabeled"] for r in rows)
    hall = sum(r["hallucinated"] for r in rows)
    rate = hall / labeled if labeled else None
    lines.append("\n## Overall\n")
    lines.append(f"- Total examples: **{total}**")
    lines.append(f"- Labeled: **{labeled}** (unlabeled: {unlabeled})")
    lines.append(f"- Hallucinated: **{hall}**")
    lines.append(f"- Hallucination rate (over labeled examples): **{fmt_pct(rate)}**")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "docs" / "tables" / "corpus_stats.md"),
    )
    args = parser.parse_args()

    root = load_data_root(args.config)
    pairs = discover_pairs(root)
    rows = [stats_for_pair(dataset, llm, root) for dataset, llm in pairs]

    report = build_report(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")
