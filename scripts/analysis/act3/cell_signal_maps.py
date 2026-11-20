from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from scripts.figures import style as fm
from scripts.analysis.act2.pca_capacity_control import balanced_indices, load_source_for

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act3"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act3"
SEED = 0
MODEL_PRETTY = {
    "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}
VIEWS = ("Q", "K", "V", "H")


def per_token_mean_fields(source, idx: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    # {view: (N, L, M)}, token-meaned per example (no PCA flatten here --
    # cell maps need the (layer, segment) grid intact). images: (T, C, L, M).
    is_hidden = source.is_hidden_states
    per_view = {v: [] for v in (("H",) if is_hidden else ("Q", "K", "V"))}
    y = []
    for i in idx:
        images, label, _origin = source[i]           # (T, C, L, M)
        mean = images.mean(dim=0).numpy()             # (C, L, M)
        for c, v in enumerate(per_view):
            per_view[v].append(mean[c])
        y.append(label)
    return {v: np.stack(rows) for v, rows in per_view.items()}, np.array(y, dtype=np.int64)


def cohens_d(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # X: (N, L, M); returns (L, M). Pooled-SD Cohen's d, hallucinated minus clean.
    pos, neg = X[y == 1], X[y == 0]
    sd = np.sqrt((pos.var(0, ddof=1) + neg.var(0, ddof=1)) / 2)
    return (pos.mean(0) - neg.mean(0)) / np.maximum(sd, 1e-12)


def cell_auroc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Rank-based AUROC per cell, no fold held out -- descriptive map, not a
    # probe evaluation (that's Act 1/2's job). Returns (L, M) in [0.5, 1].
    flat = X.reshape(len(X), -1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    ranks = rankdata(flat, axis=0)
    auc = (ranks[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc = np.maximum(auc, 1 - auc)
    return auc.reshape(X.shape[1:])


def analyse(model_alias: str, cfg_path: str, dataset: str, n_examples: int) -> dict:
    qkv_source = load_source_for(cfg_path, "default", dataset)
    hs_source = load_source_for(cfg_path, "hs", dataset)
    idx = balanced_indices(qkv_source.labels, n_examples)
    assert [hs_source.labels[i] for i in idx] == [qkv_source.labels[i] for i in idx], (
        f"{model_alias}: default/hs label mismatch at shared idx -- corpora out of sync"
    )

    qkv_fields, y = per_token_mean_fields(qkv_source, idx)
    hs_fields, y_h = per_token_mean_fields(hs_source, idx)
    assert np.array_equal(y, y_h)
    fields = {**qkv_fields, **hs_fields}

    d_maps, auc_maps = {}, {}
    for v in VIEWS:
        d_maps[v] = cohens_d(fields[v], y)
        auc_maps[v] = cell_auroc(fields[v], y)
        print(f"  {v}: max|d|={np.abs(d_maps[v]).max():.3f} mean|d|={np.abs(d_maps[v]).mean():.3f} "
              f"best-cell AUROC={auc_maps[v].max():.3f} cells>=0.65={(auc_maps[v] >= 0.65).sum()}", flush=True)

    return {"model": model_alias, "n": len(y), "d_maps": d_maps, "auc_maps": auc_maps}


def write_table(results: list[dict]) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "cell_signal_maps.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "view", "max_abs_d", "mean_abs_d", "best_cell_auroc",
                    "n_cells_auroc_ge_0.60", "n_cells_auroc_ge_0.65", "n_cells_auroc_ge_0.70"])
        for res in results:
            for v in VIEWS:
                d, auc = res["d_maps"][v], res["auc_maps"][v]
                w.writerow([res["model"], v, f"{np.abs(d).max():.5f}", f"{np.abs(d).mean():.5f}",
                            f"{auc.max():.5f}", int((auc >= 0.60).sum()),
                            int((auc >= 0.65).sum()), int((auc >= 0.70).sum())])
    print("wrote", path)


def fig_cell_maps(results: list[dict]) -> None:
    fm.apply()
    fig, axes = plt.subplots(len(results), len(VIEWS),
                              figsize=(fm.PANEL_W * len(VIEWS) * 0.75, fm.PANEL_H * len(results)),
                              squeeze=False)
    vmax = max(np.abs(res["d_maps"][v]).max() for res in results for v in VIEWS)
    tags = "abcdefghijklmnop"
    tag_i = 0
    im = None
    for r, (row, res) in enumerate(zip(axes, results)):
        for ax, v in zip(row, VIEWS):
            im = ax.imshow(res["d_maps"][v], cmap=fm.DIVERGING, vmin=-vmax, vmax=vmax,
                           aspect="auto", origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            fm.panel_tag(ax, f"({tags[tag_i]})")
            tag_i += 1
        row[0].set_ylabel(f"{MODEL_PRETTY.get(res['model'], res['model'])}\nLayer")
    for ax, v in zip(axes[0], VIEWS):
        ax.text(0.5, 1.18, v, transform=ax.transAxes, ha="center", va="bottom",
                fontweight="bold", fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("Segment")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Cohen's d")
    paths = fm.save(fig, OUT_FIG, "cell_signal_maps")

    fm.write_how_to_read(
        OUT_FIG, "cell_signal_maps",
        how_to_read=(
            "Each tile is one (model, view) pair: x-axis is pooling segment, "
            "y-axis is transformer layer (row 1 at bottom). Colour is Cohen's "
            "d between hallucinated and clean examples at that cell "
            "(diverging, symmetric scale shared across every tile so "
            "magnitudes are comparable). Computed on the token-mean, "
            "already-pooled field (no held-out split -- descriptive, not a "
            "probe evaluation)."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY.get(res['model'], res['model'])}: " + ", ".join(
                    f"{v} max|d|={np.abs(res['d_maps'][v]).max():.2f}" for v in VIEWS
                )
                for res in results
            )
        ),
    )
    print("wrote", paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                     help="pairs of alias=path, e.g. llama3.1_8b=configs/triviaqa/llama3.1_8b.yaml")
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n-examples", type=int, default=600)
    args = ap.parse_args()

    results = []
    for pair in args.configs:
        alias, path = pair.split("=", 1)
        print(f"=== {alias} ===", flush=True)
        results.append(analyse(alias, path, args.dataset, args.n_examples))

    write_table(results)
    fig_cell_maps(results)


if __name__ == "__main__":
    main()
