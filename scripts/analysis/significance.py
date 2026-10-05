"""Class-separability statistics for QKV feature fields, across every model x
dataset.

Tests the claim behind the "What does hallucination look like?" question: the
hallucinated and non-hallucinated class-average feature fields differ
*systematically* (a permutation test rejects label exchangeability) even
though the per-location effect sizes are small (Cohen's d), i.e. the signal is
weak locally but distributed across the field.

For each (dataset, LLM) pair this computes, on the (token x layer) field
mean-pooled over the segment axis M:

  * a permutation test on the mean absolute class difference, giving a z-score
    against the label-shuffled null;
  * per-cell Cohen's d, summarised as mean |d| and the fraction of cells
    exceeding |d| > 0.5.

Reads through `QKVFieldDataset` (src/data/dataset.py) rather than loading
tokens.npy directly, so both native (T, L, M, 3) corpora and QKV-Lens-era
legacy corpora (src/data/legacy.py) are handled identically.

All responses are used. Responses have variable length, so each (token, layer)
cell is averaged over only the responses that have that token (padding is
masked, never averaged), and only token positions with >= MIN_PER_CLASS
responses in both classes are analysed. The permutation null shuffles labels
across all responses, so class imbalance is built into the null.

Outputs `docs/tables/significance_stats.csv` and the figure
`docs/figures/significance.png`.

Usage:
    python scripts/analysis/significance.py [--n-perm 500]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.extract.tensor_ops import PROJECTIONS
from src.train import load_source

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_FIG = REPO_ROOT / "docs" / "figures"
OUT_TAB = REPO_ROOT / "docs" / "tables"

DATASETS = ["coqa", "triviaqa", "truthfulqa"]
MODELS = ["llama2_7b", "llama3.1_8b", "opt_6.7b", "qwen2.5_7b"]
PRETTY = {
    "llama2_7b": "Llama-2-7B", "llama3.1_8b": "Llama-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}


def load_field_source(ds: str, model: str):
    """QKVFieldDataset for (ds, model), via the normal Config path -- no
    training, no normalisation (stats=None, so _finish only permutes channels
    into conv position). `source.records[i]` gives label/n_tokens straight
    from the manifest, with no tensor.npy read."""
    cfg = load_config(str(REPO_ROOT / "configs" / ds / f"{model}.yaml"))
    return load_source(cfg, ds, cfg.llm.alias)


T_MAX = 64          # tokens kept per response (matches extraction cap)
MIN_PER_CLASS = 30  # a token position is analysed only if BOTH classes have this many responses


def collect(ds: str, model: str):
    """Every response of (ds, model), as a zero-padded field plus a validity mask.

    Responses are short and vary in length (median 2-5 tokens), so cropping to a
    common window discards most of the corpus. Instead nothing is dropped:
    X is (N, 3, T_MAX, L) with a (N, T_MAX) mask marking real tokens, and
    statistics average each (token, layer) cell over only the responses that
    actually have that token. Padded slots are never averaged in. Only token
    positions where BOTH classes have >= MIN_PER_CLASS responses are analysed.
    """
    source = load_field_source(ds, model)   # stats=None -> raw field
    n = len(source.records)
    labels = np.array([int(r["label"]) for r in source.records], dtype=bool)
    X = M = None
    for i in range(n):
        images, _label, _origin = source[i]              # (T, 3, L, M)
        field = images.mean(dim=-1).numpy()[:T_MAX]      # (T, 3, L)
        if X is None:
            X = np.zeros((n, 3, T_MAX, field.shape[-1]), np.float32)
            M = np.zeros((n, T_MAX), bool)
        t = field.shape[0]
        X[i, :, :t] = field.transpose(1, 0, 2)
        M[i, :t] = True
    cnt1, cnt0 = M[labels].sum(0), M[~labels].sum(0)
    t_use = int(np.sum((cnt1 >= MIN_PER_CLASS) & (cnt0 >= MIN_PER_CLASS)))
    if t_use == 0:
        print(f"  [skip] {ds}/{model}: no token position with >= {MIN_PER_CLASS} "
              f"responses in both classes ({labels.sum()} hallucinated / {(~labels).sum()} clean)")
        return None
    print(f"  {ds}/{model}: {labels.sum()} hallucinated / {(~labels).sum()} clean, "
          f"{t_use} token position(s) analysed")
    return X[:, :, :t_use], M[:, :t_use], labels, t_use


def stats_for(X: np.ndarray, M: np.ndarray, labels: np.ndarray, n_perm: int, rng) -> dict:
    """Permutation z-score plus Cohen's d summaries over masked cells.

    X (N, 3, T, L) zero-padded, M (N, T) validity, labels (N,) bool.
    Label permutation keeps group sizes, so class imbalance is part of the null.
    """
    N, _, T, L = X.shape
    Xf = X.reshape(N, -1)
    Xsq = Xf ** 2
    Mf = np.broadcast_to(M[:, None, :, None], X.shape).reshape(N, -1).astype(np.float32)
    tot_x, tot_sq, tot_m = Xf.sum(0), Xsq.sum(0), Mf.sum(0)

    def group_stats(w):
        sx, ssq, c = w @ Xf, w @ Xsq, w @ Mf
        return sx, ssq, c

    def mean_abs_diff(w):
        sx1, _, c1 = group_stats(w)
        sx0, c0 = tot_x - sx1, tot_m - (w @ Mf)
        return np.abs(sx1 / c1 - sx0 / c0).mean()

    w_obs = labels.astype(np.float32)
    obs = mean_abs_diff(w_obs)
    n1 = int(labels.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        w = np.zeros(N, np.float32)
        w[rng.permutation(N)[:n1]] = 1.0
        null[i] = mean_abs_diff(w)

    z = (obs - null.mean()) / (null.std() + 1e-12)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)

    sx1, ssq1, c1 = group_stats(w_obs)
    sx0, ssq0, c0 = tot_x - sx1, tot_sq - ssq1, tot_m - c1
    m1, m0 = sx1 / c1, sx0 / c0
    v1 = (ssq1 - c1 * m1 ** 2) / (c1 - 1)
    v0 = (ssq0 - c0 * m0 ** 2) / (c0 - 1)
    d = ((m1 - m0) / (np.sqrt((v1 + v0) / 2) + 1e-12)).reshape(3, T, L)
    d_per_proj = {}
    for vi, vn in enumerate(PROJECTIONS):
        ad = np.abs(d[vi])
        d_per_proj[vn] = {"mean_abs_d": float(ad.mean()), "max_abs_d": float(ad.max()),
                          "frac_gt_05": float((ad > 0.5).mean())}

    return {"obs": float(obs), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "z": float(z), "p": float(p),
            "d": d_per_proj}


def build_figure(rows: list[dict]) -> None:
    """Three panels, each readable without the surrounding prose."""
    from scripts.figures import style_modern as fm

    x = np.arange(len(rows))
    ds_colour = {"coqa": "#4C72B0", "triviaqa": "#DD8452", "truthfulqa": "#55A868"}
    ds_pretty = {"coqa": "CoQA", "triviaqa": "TriviaQA", "truthfulqa": "TruthfulQA"}
    colours = [ds_colour[r["dataset"]] for r in rows]
    labels = [r["model_pretty"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.patch.set_facecolor("white")

    zs = [r["z"] for r in rows]
    ratio = [r["obs"] / r["null_mean"] for r in rows]
    band = float(np.mean([3 * r["null_sd"] / r["null_mean"] for r in rows]))

    # ── panel 1: is the difference real? ────────────────────────────────
    ax = axes[0]
    ax.bar(x, zs, color=colours, edgecolor="black", linewidth=0.4)
    ax.axhline(1.96, ls="--", lw=1.2, color="crimson", zorder=4)
    ax.text(len(rows) - 0.45, max(zs) * 0.6, "dashed: $p=0.05$ ($z=1.96$)",
            fontsize=7.5, color="crimson", ha="right", va="bottom",
            fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.set_ylim(-0.5, max(zs) * 1.15)
    for xi, z in zip(x, zs):
        ax.text(xi, max(z, 0) + 0.3, f"{z + 0.0:.1f}" if abs(z) >= 0.1 else "0.0", ha="center",
                va="bottom", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("permutation $z$-score", fontsize=9)
    ax.set_title("Permutation $z$-score of the class difference",
                 fontsize=10, fontweight="bold")

    # ── panel 2: how large is it at any one location? ───────────────────
    ax = axes[1]
    w = 0.26
    proj_colour = {"Q": "#4C72B0", "K": "#DD8452", "V": "#55A868"}
    for i, vn in enumerate(PROJECTIONS):
        ax.bar(x + (i - 1) * w, [r["mean_abs_d"][vn] for r in rows], width=w,
               label=f"{vn} projection", color=proj_colour[vn],
               edgecolor="black", linewidth=0.3)
    ax.axhline(0.5, ls="--", lw=1.2, color="crimson", zorder=4)
    ax.axhline(0.2, ls=":", lw=1.2, color="dimgrey", zorder=4)
    ax.text(len(rows) - 0.45, 0.515, "medium effect  $|d|=0.5$", fontsize=7.5,
            color="crimson", ha="right", va="bottom", fontweight="bold")
    ax.text(-0.35, 0.215, "small effect  $|d|=0.2$", fontsize=7.5,
            color="dimgrey", ha="left", va="bottom", fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("mean $|$Cohen's $d|$ per cell", fontsize=9)
    ax.set_title("Mean per-cell effect size, by projection",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.0))

    # ── panel 3: how far above chance, on a common scale? ───────────────
    ax = axes[2]
    ax.bar(x, ratio, color=colours, edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="black", lw=1.2, zorder=4)
    ax.fill_between([-0.6, len(rows) - 0.4], 1 - band, 1 + band,
                    color="crimson", alpha=0.25, zorder=3)
    ax.text(len(rows) - 0.45, max(ratio) * 1.14,
            "shuffled-label null $\\pm3$ SD  (chance $=1.0$)",
            fontsize=7.5, color="crimson", ha="right", va="bottom",
            fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    for xi, rt in zip(x, ratio):
        ax.text(xi, rt + 0.05, f"{rt:.1f}x", ha="center", va="bottom", fontsize=6.5)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, max(ratio) * 1.3)
    ax.set_ylabel("observed / null class difference", fontsize=9)
    ax.set_title("Class difference normalised by its own null",
                 fontsize=10, fontweight="bold")

    for ax, letter in zip(axes, "abc"):
        ax.text(-0.135, 1.06, f"({letter})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="left")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=38, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", lw=0.4, alpha=0.25)

        start = 0
        for i in range(len(rows) + 1):
            if i == len(rows) or rows[i]["dataset"] != rows[start]["dataset"]:
                ds = rows[start]["dataset"]
                ax.plot([start - 0.4, i - 0.6], [-0.30, -0.30],
                        transform=ax.get_xaxis_transform(), clip_on=False,
                        color=ds_colour[ds], lw=3.0, solid_capstyle="butt")
                ax.text((start + i - 1) / 2, -0.345, ds_pretty[ds],
                        transform=ax.get_xaxis_transform(), clip_on=False,
                        ha="center", va="top", fontsize=8.5,
                        fontweight="bold", color=ds_colour[ds])
                start = i

    fig.tight_layout()
    fig.savefig(OUT_FIG / "significance.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_FIG / "significance.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT_FIG / "significance.png")


def rows_from_csv() -> list[dict]:
    rows = []
    with open(OUT_TAB / "significance_stats.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "dataset": r["dataset"], "model": r["model"],
                "model_pretty": PRETTY[r["model"]], "z": float(r["z"]),
                "obs": float(r["observed_mean_abs_diff"]),
                "null_mean": float(r["null_mean"]), "null_sd": float(r["null_sd"]),
                "mean_abs_d": {v: float(r[f"mean_abs_d_{v}"]) for v in PROJECTIONS},
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from the existing CSV (no recomputation)")
    args = ap.parse_args()

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        build_figure(rows_from_csv())
        return
    rng = np.random.default_rng(0)

    rows = []
    for ds in DATASETS:
        for model in MODELS:
            got = collect(ds, model)
            if got is None:
                continue
            X, Mk, lab, t_use = got
            st = stats_for(X, Mk, lab, args.n_perm, rng)
            rows.append({
                "dataset": ds, "model": model, "model_pretty": PRETTY[model],
                "n_hallu": int(lab.sum()), "n_clean": int((~lab).sum()), "t_use": t_use,
                "z": st["z"], "p": st["p"], "obs": st["obs"],
                "null_mean": st["null_mean"], "null_sd": st["null_sd"],
                "mean_abs_d": {v: st["d"][v]["mean_abs_d"] for v in PROJECTIONS},
                "max_abs_d": {v: st["d"][v]["max_abs_d"] for v in PROJECTIONS},
                "frac_gt_05": {v: st["d"][v]["frac_gt_05"] for v in PROJECTIONS},
            })
            print(f"    z={st['z']:.1f} p={st['p']:.4f} "
                  f"mean|d| Q={st['d']['Q']['mean_abs_d']:.3f} "
                  f"K={st['d']['K']['mean_abs_d']:.3f} "
                  f"V={st['d']['V']['mean_abs_d']:.3f}")

    if not rows:
        print("no (dataset, model) pair had enough balanced data -- nothing to write")
        return

    with open(OUT_TAB / "significance_stats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "n_hallucinated", "n_clean", "tokens_used", "z", "p",
                    "observed_mean_abs_diff", "null_mean", "null_sd",
                    *[f"mean_abs_d_{v}" for v in PROJECTIONS],
                    *[f"max_abs_d_{v}" for v in PROJECTIONS],
                    *[f"frac_absd_gt_0.5_{v}" for v in PROJECTIONS]])
        for r in rows:
            w.writerow([r["dataset"], r["model"], r["n_hallu"], r["n_clean"], r["t_use"],
                        f"{r['z']:.2f}", f"{r['p']:.5f}",
                        f"{r['obs']:.6f}", f"{r['null_mean']:.6f}",
                        f"{r['null_sd']:.6f}",
                        *[f"{r['mean_abs_d'][v]:.4f}" for v in PROJECTIONS],
                        *[f"{r['max_abs_d'][v]:.4f}" for v in PROJECTIONS],
                        *[f"{r['frac_gt_05'][v]:.4f}" for v in PROJECTIONS]])
    print("wrote", OUT_TAB / "significance_stats.csv")

    build_figure(rows)


if __name__ == "__main__":
    main()
