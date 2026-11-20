from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt

from scripts.figures import style as fm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS_ROOT = REPO_ROOT / "runs"
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act3"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act3"
M_GRID = (8, 16, 32, 64, 128)
MODEL_PRETTY = {
    "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}
HEAD_DIM = {
    "llama3.1_8b": 128, "opt_6.7b": 128, "qwen2.5_7b": 128,
}
HIDDEN_SIZE = {
    "llama3.1_8b": 4096, "opt_6.7b": 4096, "qwen2.5_7b": 3584,
}


def run_name(model_alias: str, M: int) -> str:
    tag = "repr_hs" if M == 32 else f"repr_hs_M{M}"
    return f"{tag}_{model_alias}_triviaqa"


def read_auroc(model_alias: str, M: int) -> float | None:
    # These are already-trained full-detector runs from
    # scripts/experiments/representation_ablation.sh step 2 -- no training here.
    path = RUNS_ROOT / run_name(model_alias, M) / "test_triviaqa.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["metrics"]["auroc"]


def analyse(model_alias: str) -> dict:
    hidden = HIDDEN_SIZE[model_alias]
    head_dim = HEAD_DIM[model_alias]
    per_m = {}
    for M in M_GRID:
        auroc = read_auroc(model_alias, M)
        if auroc is None:
            continue
        seg_size = hidden / M
        per_m[M] = {"auroc": auroc, "seg_size": seg_size, "head_aligned": seg_size == head_dim}
    return {"model": model_alias, "per_m": per_m, "head_dim": head_dim, "hidden": hidden}


def write_table(results: list[dict]) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "head_alignment.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "M", "segment_size", "head_dim", "head_aligned", "auroc"])
        for res in results:
            for M, row in res["per_m"].items():
                w.writerow([res["model"], M, row["seg_size"], res["head_dim"],
                            row["head_aligned"], f"{row['auroc']:.5f}"])
    print("wrote", path)


def fig_head_alignment(results: list[dict]) -> None:
    fm.apply()
    tags = "abc"
    fig, axes = plt.subplots(1, len(results),
                              figsize=(fm.PANEL_W * len(results), fm.PANEL_H), squeeze=False)
    for i, (ax, res) in enumerate(zip(axes[0], results)):
        ms = sorted(res["per_m"])
        aurocs = [100 * res["per_m"][m]["auroc"] for m in ms]
        colours = [fm.AQUA if res["per_m"][m]["head_aligned"] else fm.GRAY for m in ms]
        ax.scatter(ms, aurocs, c=colours, s=70, zorder=3, edgecolor="white", linewidth=0.8)
        ax.plot(ms, aurocs, color=fm.GRAY, lw=1.2, zorder=1, path_effects=fm.HALO)
        ax.set_xscale("log")
        ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in ms])
        ax.set_xlabel("Segments (M)")
        if i == 0:
            ax.set_ylabel("AUROC")
        ax.scatter([], [], c=fm.AQUA, s=70, edgecolor="white", linewidth=0.8, label="Head-Aligned")
        ax.scatter([], [], c=fm.GRAY, s=70, edgecolor="white", linewidth=0.8, label="Not Aligned")
        fm.legend(ax)
        fm.panel_tag(ax, f"({tags[i]}) {MODEL_PRETTY.get(res['model'], res['model'])}")
    paths = fm.save(fig, OUT_FIG, "head_alignment")

    def best_m(res):
        m = max(res["per_m"], key=lambda m: res["per_m"][m]["auroc"])
        return m, 100 * res["per_m"][m]["auroc"], res["per_m"][m]["head_aligned"]

    fm.write_how_to_read(
        OUT_FIG, "head_alignment",
        how_to_read=(
            "Each panel is one LLM; x-axis is the number of pooling segments "
            "M used for the hidden-states (H) field (log scale), y-axis is "
            "the trained conv+BiLSTM+attention detector's test AUROC at that "
            "M (already-trained runs from "
            "scripts/experiments/representation_ablation.sh, not retrained "
            "here). A point is coloured aqua when that M's segment size "
            "(hidden_size / M) exactly equals the model's head_dim -- i.e. "
            "pooling happens to align with attention-head boundaries -- and "
            "gray otherwise. H has no head structure of its own; this checks "
            "whether pooling it in head-sized chunks (the only thing Q/K/V "
            "get for free) helps even when imposed on H artificially."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY.get(res['model'], res['model'])}: best M={best_m(res)[0]} "
                f"(AUROC {best_m(res)[1]:.1f}, head-aligned={best_m(res)[2]})"
                for res in results
            )
        ),
    )
    print("wrote", paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    results = [analyse(m) for m in args.models]
    results = [r for r in results if r["per_m"]]
    missing = set(args.models) - {r["model"] for r in results}
    if missing:
        print(f"no runs found for: {sorted(missing)} -- skipping", flush=True)
    if not results:
        raise SystemExit("no runs found for any requested model")

    write_table(results)
    fig_head_alignment(results)


if __name__ == "__main__":
    main()
