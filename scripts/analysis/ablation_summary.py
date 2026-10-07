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


def section_layer_grid_grid(lines: list[str]) -> None:
    lines.append("## 1. LayerGrid grid search: KERNEL_SIZE / HIDDEN (llama3.1_8b, TriviaQA)")
    rows = _read_csv(TABLES / "layer_grid_grid.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/layer_grid_grid.sh`.\n")
        return

    valid = [r for r in rows if r["auroc"]]
    if not valid:
        lines.append("\n_Run but no cell produced a result yet._\n")
        return
    best = max(valid, key=lambda r: float(r["auroc"]))
    lines.append("")
    lines.append(
        f"Best cell: **KERNEL_SIZE={best['kernel_size']}, "
        f"HIDDEN={best['hidden']}** -- AUROC {_pct(best['auroc'])}. "
        "`layer_grid.py` is left set to this combo by the sweep script, so "
        "every `model.backbone=layer_grid` run below (this repo's MAIN "
        "backbone -- representation/pooling/collapse ablations, structure "
        "analysis, main results) already trains against it."
    )
    lines.append("")
    lines.append("| KERNEL_SIZE | HIDDEN | AUROC |")
    lines.append("|---|---|---|")
    for r in sorted(valid, key=lambda r: -float(r["auroc"])):
        lines.append(f"| {r['kernel_size']} | {r['hidden']} | {_pct(r['auroc'])} |")
    lines.append("")
    lines.append("Full table: `docs/tables/layer_grid_grid.md`.")
    lines.append("")


def section_representation(lines: list[str]) -> None:
    lines.append("## 2a. Representation ablation (Q/K/V + hidden states)")
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
        "This result feeds step 5 below: a majority vote over each model's "
        "own best row here picks the representation used for the main "
        "results (ties/no-majority default to QKV)."
    )
    lines.append("")
    lines.append("Full table: `docs/tables/representation_ablation.md`.")
    lines.append("")


def section_pooling(lines: list[str]) -> None:
    lines.append("## 2b. n_segments / pooling ablation (llama3.1_8b, TriviaQA)")
    rows = _read_csv(TABLES / "pooling_ablation.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/pooling_ablation.sh`.\n")
        return

    valid = [r for r in rows if r["auroc"]]
    if not valid:
        lines.append("\n_Run but no cell produced a result yet._\n")
        return
    best = max(valid, key=lambda r: float(r["auroc"]))
    lines.append("")
    lines.append(
        f"Best cell: **{best['pool']} pooling, M={best['n_segments']}** "
        f"-- AUROC {_pct(best['auroc'])}."
    )
    lines.append("")
    lines.append("| Pool | M | AUROC |")
    lines.append("|---|---|---|")
    for r in sorted(valid, key=lambda r: -float(r["auroc"]))[:5]:
        lines.append(f"| {r['pool']} | {r['n_segments']} | {_pct(r['auroc'])} |")
    lines.append("")
    lines.append("Full table: `docs/tables/pooling_ablation.md`.")
    lines.append("")


def section_collapse(lines: list[str]) -> None:
    lines.append("## 2c. Feature-pooling (collapse-axis) ablation (llama3.1_8b, TriviaQA)")
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


def section_structure(lines: list[str]) -> None:
    lines.append("## 3. Structure-preservation analysis (llama3.1_8b, TriviaQA)")
    rows = _read_csv(TABLES / "structure_analysis.csv")
    if rows is None:
        lines.append("\n_Not yet run._ See `scripts/experiments/structure_analysis.sh`.\n")
        return

    by_arm = {r["arm"]: r for r in rows}
    lines.append("")
    lines.append("| Arm | AUROC | $\\Delta$ vs flat_mlp |")
    lines.append("|---|---|---|")
    for arm, label in (("flat", "flat_mlp (no spatial structure)"),
                       ("gridbb", "layer_grid (MAIN backbone, gated local conv over L)"),
                       ("permL42", "flat_mlp + layer permutation")):
        r = by_arm.get(arm)
        if r is None:
            lines.append(f"| {label} | - | - |")
        else:
            lines.append(f"| {label} | {_pct(r['auroc'])} | {r['delta_vs_flat'] or '--'} |")
    lines.append("")

    flat_r = by_arm.get("flat")
    main_r = by_arm.get("gridbb")
    perm_r = by_arm.get("permL42")
    if flat_r and flat_r["auroc"] and main_r and main_r["auroc"]:
        flat_auroc, main_auroc = float(flat_r["auroc"]), float(main_r["auroc"])
        verdict = "layer_grid beats flat_mlp" if main_auroc > flat_auroc else "flat_mlp holds or beats layer_grid"
        lines.append(
            f"- **Spatial structure (gated):** {verdict} "
            f"(layer_grid {_pct(main_auroc)} vs flat_mlp {_pct(flat_auroc)})."
        )
    if flat_r and flat_r["auroc"] and perm_r and perm_r["auroc"]:
        flat_auroc, perm_auroc = float(flat_r["auroc"]), float(perm_r["auroc"])
        verdict = ("layer order matters (permuted is worse)" if perm_auroc < flat_auroc
                   else "layer order doesn't clearly matter (permuted is not worse)")
        lines.append(
            f"- **Layer order:** {verdict} "
            f"(permuted {_pct(perm_auroc)} vs true order {_pct(flat_auroc)})."
        )
    lines.append("")
    lines.append("Full table: `docs/tables/structure_analysis.md`.")
    lines.append("")


def section_main_results(lines: list[str]) -> None:
    lines.append("## 5. Winning representation + main results (4 models x 3 datasets)")
    rows = _read_csv(TABLES / "representation_ablation.csv")
    if rows is None:
        lines.append(
            "\n_Not yet run._ `all_experiments.sh` step 5 reads "
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
        "`all-datasets_run.sh`'s own main-results runs, so step 6's "
        "significance/prototype/IG/forecasting figures below read these "
        "directly."
    )
    lines.append("")


def section_significance(lines: list[str]) -> None:
    lines.append("## 6a. Significance / diffuseness analysis (all 4 models)")
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
    lines.append("## 6b. Qualitative maps: confusion examples + class-average prototypes (TriviaQA)")
    stats_path = REPO_ROOT / "docs" / "figures" / "prototype_stats.json"
    if not stats_path.exists():
        lines.append(
            "\n_Not yet run._ See `python scripts/figures/qualitative_maps.py "
            "--model <model>` (run per model by `all_experiments.sh` step 6).\n"
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
    lines.append("## 6c. Integrated Gradients attribution heatmap (TriviaQA, 4 models)")
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
        "`docs/figures/f1a_cam_response_heatmap.png`. Works for any backbone "
        "(layer_grid, flat_mlp) -- see `src/cam.py`'s docstring for why "
        "Integrated Gradients needs no backbone-specific hooking."
    )
    lines.append("")


def section_forecasting(lines: list[str]) -> None:
    lines.append("## 6d. Forecasting: how early is the verdict usable? (TriviaQA, 4 models)")
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
        "picture. `layer_grid` is this repo's MAIN backbone (see step 1); "
        "step 5 picks the winning representation and runs the actual main "
        "results, so by the time this report is read, "
        "`runs/{model}_{dataset}/` already reflects everything above it.",
        "",
    ]

    for fn in (
        section_layer_grid_grid,
        section_representation,
        section_pooling,
        section_collapse,
        section_structure,
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
