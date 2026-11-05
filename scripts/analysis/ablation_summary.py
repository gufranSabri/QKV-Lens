#!/usr/bin/env python3
# Reads CSVs/reports the sweeps already wrote (all_experiments.sh); never recomputes.

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

TABLES = REPO_ROOT / "docs" / "tables"
REPORTS = REPO_ROOT / "docs" / "reports"
OUT = REPORTS / "ablation_summary.md"

PRETTY_LLM = {
    "llama2_7b": "LLaMA-2-7B", "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}


def _read_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fnum(x, default="-"):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return default


def _pct(x, default="-"):
    try:
        return f"{100 * float(x):.2f}"
    except (TypeError, ValueError):
        return default


def section_hidden_vs_qkv(lines: list[str]) -> None:
    lines.append("## 1. Hidden states vs Q/K/V: linear probes on token-mean features (TriviaQA)")
    rows = _read_csv(TABLES / "hidden_vs_qkv" / "probes.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `python scripts/analysis/hidden_vs_qkv.py`.\n")
        return

    by = {(r["model"], r["rep"]): r["auroc"] for r in rows}
    models = [m for m in PRETTY_LLM if any(k[0] == m for k in by)]
    reps = ("H", "Q", "K", "V", "QKV", "H+QKV")
    lines.append("")
    lines.append("| Model | " + " | ".join(reps) + " |")
    lines.append("|---|" + "---|" * len(reps))
    for m in models:
        lines.append(f"| {PRETTY_LLM[m]} | " + " | ".join(_pct(by.get((m, r))) for r in reps) + " |")
    lines.append("")
    lines.append("Full report: `docs/reports/hidden_vs_qkv.md`.")
    lines.append("")


def section_component(lines: list[str]) -> None:
    lines.append("## 2. Component ablation (llama3.1_8b, TriviaQA)")
    rows = _read_csv(TABLES / "component_ablation.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/component_ablation.sh`.\n")
        return

    labels = {
        "full": "Full detector", "no_proj": "- flat_mlp projection", "no_conv": "- Conv1d",
        "no_lstm": "- BiLSTM", "no_attn_pool": "- attention pool (masked mean)",
        "probe": "Probe (flatten -> masked mean -> head)",
    }
    lines.append("")
    lines.append("| Detector | AUROC | $\\Delta$ vs full | Params |")
    lines.append("|---|---|---|---|")
    for r in rows:
        n = int(r["n_params"]) if r.get("n_params") else None
        params = "-" if n is None else f"{n / 1e6:.1f} M" if n >= 1e5 else f"{n / 1e3:.1f} k"
        lines.append(
            f"| {labels.get(r['arm'], r['arm'])} | {_pct(r['auroc'])} "
            f"| {r['delta_vs_full'] or '--'} | {params} |"
        )
    lines.append("")
    lines.append(
        "A large drop when a component is removed means the detector relies on "
        "it. `no_proj` has more parameters than full and `no_lstm` loses all "
        "long-range context, so read those with their param counts."
    )
    lines.append("")
    lines.append("Full table: `docs/tables/component_ablation.md`.")
    lines.append("")


def section_representation(lines: list[str]) -> None:
    lines.append("## 3a. Representation ablation (Q/K/V + hidden states)")
    rows = _read_csv(TABLES / "representation_ablation.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/representation_ablation.sh`.\n")
        return

    best_per_model: dict[str, tuple[str, float]] = {}
    for r in rows:
        if not r["auroc"]:
            continue
        auroc = float(r["auroc"])
        cur = best_per_model.get(r["model"])
        if cur is None or auroc > cur[1]:
            best_per_model[r["model"]] = (r["representation"], auroc)

    lines.append("")
    lines.append("Best representation per model (by held-out AUROC):")
    lines.append("")
    lines.append("| Model | Best representation | AUROC | Full QKV | Hidden states |")
    lines.append("|---|---|---|---|---|")
    by_model_combo = {(r["model"], r["representation"]): r["auroc"] for r in rows}
    for model in PRETTY_LLM:
        best = best_per_model.get(model)
        if best is None:
            continue
        qkv = by_model_combo.get((model, "QKV"))
        hs = by_model_combo.get((model, "HS"))
        lines.append(
            f"| {PRETTY_LLM[model]} | {best[0]} | {_pct(best[1])} "
            f"| {_pct(qkv)} | {_pct(hs)} |"
        )
    lines.append("")
    lines.append(
        "This result feeds section 4 below: a majority vote over each model's "
        "own best row here picks the representation used for the main "
        "results (ties/no-majority default to QKV)."
    )
    lines.append("")
    lines.append("Full table: `docs/tables/representation_ablation.md`.")
    lines.append("")


def section_pooling(lines: list[str]) -> None:
    lines.append("## 3b. Pooling ablation: segments M and pooling mode (llama3.1_8b, TriviaQA)")
    rows = _read_csv(TABLES / "pooling_ablation.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/pooling_ablation.sh`.\n")
        return

    valid = [r for r in rows if r["auroc"]]
    if not valid:
        lines.append("\n_Run but no cell produced a result yet._\n")
        return
    field = {"qkv": "QKV", "hidden-states": "Hidden states"}
    best = max(valid, key=lambda r: float(r["auroc"]))
    lines.append("")
    lines.append(
        f"Best cell: **{field.get(best['source'], best['source'])}, {best['pool']} pooling, "
        f"M={best['n_segments']}** -- AUROC {_pct(best['auroc'])}."
    )
    lines.append("")
    lines.append("| Field | Pool | M | AUROC |")
    lines.append("|---|---|---|---|")
    for r in sorted(valid, key=lambda r: -float(r["auroc"]))[:5]:
        lines.append(
            f"| {field.get(r['source'], r['source'])} | {r['pool']} | {r['n_segments']} "
            f"| {_pct(r['auroc'])} |"
        )
    lines.append("")
    lines.append("Full table: `docs/tables/pooling_ablation.md`.")
    lines.append("")


def section_collapse(lines: list[str]) -> None:
    lines.append("## 3c. Feature-pooling (collapse-axis) ablation (llama3.1_8b, TriviaQA)")
    rows = _read_csv(TABLES / "collapse_axis_ablation.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/collapse_axis_ablation.sh`.\n")
        return

    by_field = {r["field"]: r for r in rows}
    full = by_field.get("full")
    lines.append("")
    lines.append("| Field | AUROC | $\\Delta$ vs full |")
    lines.append("|---|---|---|")
    if full is not None:
        lines.append(f"| **Full (no collapse)** | **{_pct(full['auroc'])}** | -- |")
    for axis, label in (("collapse_M", "Collapse M (segments)"),
                        ("collapse_L", "Collapse L (layers)"),
                        ("collapse_channels", "Collapse channels (Q/K/V)")):
        r = by_field.get(axis)
        if r is None:
            lines.append(f"| {label} | - | - |")
        else:
            lines.append(f"| {label} | {_pct(r['auroc'])} | {r['delta_vs_full'] or '-'} |")
    lines.append("")
    lines.append(
        "A large drop for any row is evidence that axis carries information "
        "the full field uses and a collapsed/aggregated vector loses."
    )
    lines.append("")
    lines.append("Full table: `docs/tables/collapse_axis_ablation.md`.")
    lines.append("")


def section_permutation(lines: list[str]) -> None:
    lines.append("## 3d. Permutation ablation: does true L / M order matter? (llama3.1_8b, TriviaQA)")
    rows = _read_csv(TABLES / "permutation_ablation.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/permutation_ablation.sh`.\n")
        return

    baseline = next((r["auroc"] for r in rows if r["axis"] == "baseline"), None)
    lines.append("")
    lines.append("| Permuted axis | Mean AUROC over seeds | $\\Delta$ vs true order |")
    lines.append("|---|---|---|")
    lines.append(f"| None (true order) | {_pct(baseline)} | -- |")
    for axis, label in (("L", "Layers (L)"), ("M", "Segments (M)"), ("LM", "Both")):
        vals = [float(r["auroc"]) for r in rows if r["axis"] == axis and r["auroc"]]
        mean = sum(vals) / len(vals) if vals else None
        delta = (f"{100 * (mean - float(baseline)):+.2f}"
                 if mean is not None and baseline else "-")
        lines.append(f"| {label} | {_pct(mean)} | {delta} |")
    lines.append("")
    lines.append(
        "A drop under permutation means the true axis order carries information "
        "beyond the raw values."
    )
    lines.append("")
    lines.append("Full table: `docs/tables/permutation_ablation.md`.")
    lines.append("")


def section_main_results(lines: list[str]) -> None:
    lines.append("## 4. Winning representation + main results (4 models x 3 datasets)")
    rows = _read_csv(TABLES / "representation_ablation.csv")
    if rows is None:
        lines.append(
            "\n_Not yet run._ `all_experiments.sh` reads "
            "`docs/tables/representation_ablation.csv`, picks the majority-"
            "vote winning representation, and trains+tests all 4 models x "
            "3 datasets against it (`runs/{model}_{dataset}/`).\n"
        )
        return

    from collections import Counter

    best_per_model: dict[str, tuple[str, float]] = {}
    for r in rows:
        if not r["auroc"]:
            continue
        auroc = float(r["auroc"])
        cur = best_per_model.get(r["model"])
        if cur is None or auroc > cur[1]:
            best_per_model[r["model"]] = (r["representation"], auroc)

    if not best_per_model:
        lines.append("\n_Representation ablation ran but produced no result yet._\n")
        return

    votes = Counter(repr_ for repr_, _ in best_per_model.values())
    top, top_n = votes.most_common(1)[0]
    n_models = len(best_per_model)
    winner = top if top_n > n_models / 2 else "QKV"

    lines.append("")
    lines.append(
        f"Majority vote over {n_models} model(s)' best representation: "
        f"{dict(votes)} -> **{winner}**"
        + ("" if winner == top else " (no majority, defaulted to QKV)") + "."
    )
    lines.append("")

    datasets = ["triviaqa", "truthfulqa", "coqa"]
    dataset_pretty = {"triviaqa": "TriviaQA", "truthfulqa": "TruthfulQA", "coqa": "CoQA"}
    lines.append(f"Main results (representation = **{winner}**):")
    lines.append("")
    lines.append("| Model | " + " | ".join(dataset_pretty[d] for d in datasets) + " |")
    lines.append("|---|" + "---|" * len(datasets))
    from scripts.tables.collect_runs import load_runs
    runs = load_runs(REPO_ROOT / "runs")
    by_model_ds = {(r["llm"], r["dataset"]): r["auroc"] for r in runs
                   if r["run"] == f"{r['llm']}_{r['dataset']}"}
    for model in PRETTY_LLM:
        cells = [_pct(by_model_ds.get((model, d))) for d in datasets]
        lines.append(f"| {PRETTY_LLM[model]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Checkpoints: `runs/{model}_{dataset}/best.pt` -- same layout as "
        "`all-datasets_run.sh`'s own main-results runs, so section 5's "
        "significance/prototype/IG/forecasting figures below read these "
        "directly."
    )
    lines.append("")


def section_significance(lines: list[str]) -> None:
    lines.append("## 5a. Significance / diffuseness analysis (all 4 models)")
    rows = _read_csv(TABLES / "significance_stats.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `python scripts/analysis/significance.py`.\n")
        return

    n_sig = sum(1 for r in rows if r.get("p") and float(r["p"]) < 0.05)
    lines.append("")
    lines.append(
        f"{n_sig} of {len(rows)} (dataset, model) settings show a statistically "
        f"significant global class difference (p < 0.05)."
    )
    lines.append("")
    lines.append(r"| Dataset | Model | z | p | mean \|d\| Q | mean \|d\| K | mean \|d\| V |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        sig = "**" if r.get("p") and float(r["p"]) < 0.05 else ""
        lines.append(
            f"| {r['dataset']} | {PRETTY_LLM.get(r['model'], r['model'])} "
            f"| {sig}{_fnum(r.get('z'))}{sig} | {_fnum(r.get('p'))} "
            f"| {_fnum(r.get('mean_abs_d_Q'))} | {_fnum(r.get('mean_abs_d_K'))} "
            f"| {_fnum(r.get('mean_abs_d_V'))} |"
        )
    lines.append("")
    lines.append(
        "Small mean |d| alongside significant z means the signal is "
        "**distributed**, not concentrated in a few locations -- see "
        "`docs/figures/significance.png`."
    )
    lines.append("")


def section_qualitative(lines: list[str]) -> None:
    lines.append("## 5b. Qualitative maps: confusion examples + class-average prototypes (TriviaQA)")
    stats_path = REPO_ROOT / "docs" / "figures" / "prototype_stats.json"
    if not stats_path.exists():
        lines.append(
            "\n_Not yet run._ See `python scripts/figures/qualitative_maps.py "
            "--model <model>` (run per model by `all_experiments.sh`).\n"
        )
        return

    import json
    stats = json.loads(stats_path.read_text())
    lines.append("")
    lines.append(
        "Per-layer mean |difference| between hallucinated and non-"
        "hallucinated class-average feature maps (most recently run model):"
    )
    lines.append("")
    lines.append("| Projection | Peak layer | Peak |diff| | Mean |diff| |")
    lines.append("|---|---|---|---|")
    for proj, s in stats.items():
        if s.get("zeroed"):
            lines.append(f"| {proj} | zeroed (not in keep_channels) | - | - |")
            continue
        lines.append(
            f"| {proj} | {s['peak_layer']} | {s['peak_value']:.5f} | {s['mean_abs_diff']:.5f} |"
        )
    lines.append("")
    lines.append(
        "Figures: `docs/figures/confusion_examples.png` (one real example per "
        "confusion-matrix cell) and `docs/figures/prototype_maps.png` (what a "
        "hallucination looks like on average)."
    )
    lines.append("")


def section_attribution(lines: list[str]) -> None:
    lines.append("## 5c. Integrated Gradients attribution heatmap (TriviaQA, 4 models)")
    fig_path = REPO_ROOT / "docs" / "figures" / "f1a_cam_response_heatmap.png"
    if not fig_path.exists():
        lines.append(
            "\n_Not yet run._ See `python scripts/figures/attribution_heatmap.py`.\n"
        )
        return
    lines.append("")
    lines.append(
        "Where the detector looks across a generated response, per model "
        "(rows) and relative position in the response (columns): "
        "`docs/figures/f1a_cam_response_heatmap.png`. See `src/cam.py`'s "
        "docstring for why Integrated Gradients needs no model-specific hooking."
    )
    lines.append("")


def section_forecasting(lines: list[str]) -> None:
    lines.append("## 5d. Forecasting: how early is the verdict usable? (TriviaQA, 4 models)")
    rows = _read_csv(TABLES / "forecasting" / "forecasting_summary.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `python scripts/analysis/run_all.py`.\n")
        return

    lines.append("")
    lines.append("| Model | Final AUROC | Earliness (95% of final) | AUROC @ 25% seen |")
    lines.append("|---|---|---|---|")
    for r in rows:
        earliness = r.get("earliness_95")
        earliness_str = f"{_pct(earliness)}%" if earliness else "-"
        lines.append(
            f"| {PRETTY_LLM.get(r['llm'], r['llm'])} | {_pct(r.get('final_auroc'))} "
            f"| {earliness_str} | {_pct(r.get('auroc_at_25'))} |"
        )
    lines.append("")
    lines.append(
        "Figure (one line per model): `docs/figures/forecasting/forecasting_by_model.png`. "
        "Full report: `docs/reports/forecasting/report.md`."
    )
    lines.append("")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ablation & analysis summary",
        "",
        "Headline numbers from every ablation/analysis sweep "
        "(`scripts/experiments/all_experiments.sh`), in one place. Each "
        "section links to its own full table/figure for the complete "
        "picture. Section 4 picks the winning representation and runs the actual main "
        "results, so by the time this report is read, "
        "`runs/{model}_{dataset}/` already reflects everything above it.",
        "",
    ]

    for fn in (
        section_hidden_vs_qkv,
        section_component,
        section_representation,
        section_pooling,
        section_collapse,
        section_permutation,
        section_main_results,
        section_significance,
        section_qualitative,
        section_attribution,
        section_forecasting,
    ):
        fn(lines)

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
