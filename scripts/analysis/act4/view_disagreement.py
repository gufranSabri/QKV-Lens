from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.figures import style as fm
from scripts.analysis.act2.pca_capacity_control import balanced_indices, load_source_for
from scripts.analysis.act3.cell_signal_maps import per_token_mean_fields

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act4"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act4"
SEED = 0
N_FOLDS = 5
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
VIEWS = ("Q", "K", "V", "H", "QKV")
MODEL_PRETTY = {
    "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}

# Neutral white-to-ink ramp for the McNemar significance matrix -- no hue to
# get wrong, significance reads purely as darkness, standard for a journal
# significance matrix.
MCNEMAR_CMAP = LinearSegmentedColormap.from_list(
    "mcnemar", ["#ffffff", "#d4d4d2", "#a8a7a3", "#7a7975", fm.INK],
)


def _mix(hex_colour: str, towards: str = "#ffffff", frac: float = 0.35) -> tuple:
    a = np.array(plt.matplotlib.colors.to_rgb(hex_colour))
    b = np.array(plt.matplotlib.colors.to_rgb(towards))
    return tuple(a * (1 - frac) + b * frac)


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def _lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000, solver="liblinear")


def cv_oof_proba(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    # Out-of-fold predicted P(label=1), one value per example -- needed for
    # McNemar and "only this view correct", which score per-example
    # correctness rather than fold-mean AUROC. C is picked on an inner
    # holdout of the training fold, same as every other act's probe.
    proba = np.empty(len(y), dtype=np.float64)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        tr_in, tr_val, y_in, y_val = train_test_split(
            Xtr, y[tr], test_size=0.2, stratify=y[tr], random_state=SEED)
        best = max(C_GRID, key=lambda C: roc_auc_score(
            y_val, _lr(C).fit(tr_in, y_in).decision_function(tr_val)))
        clf = _lr(best).fit(Xtr, y[tr])
        proba[te] = clf.predict_proba(Xte)[:, 1]
    return proba


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    # n10: A right, B wrong. n01: A wrong, B right. Exact binomial test on the
    # discordant pairs against p=0.5 -- avoids a statsmodels dependency and is
    # the right-sized test at n~600 (chi-square with continuity correction is
    # only needed once n10+n01 is large).
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n_disc = n10 + n01
    p = 1.0 if n_disc == 0 else binomtest(min(n10, n01), n_disc, 0.5).pvalue
    return {"n10": n10, "n01": n01, "n_discordant": n_disc, "p": p}


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

    single = {v: flatten(qkv_fields[v]) for v in "QKV"}
    single["H"] = flatten(hs_fields["H"])
    single["QKV"] = np.concatenate([single[v] for v in "QKV"], axis=1)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    proba = {v: cv_oof_proba(single[v], y, folds) for v in VIEWS}
    correct = {v: (proba[v] >= 0.5).astype(int) == y for v in VIEWS}

    mcnemar = {}
    for a, b in combinations(VIEWS, 2):
        mcnemar[(a, b)] = mcnemar_exact(correct[a], correct[b])
        print(f"  McNemar {a} vs {b}: n10={mcnemar[(a, b)]['n10']} n01={mcnemar[(a, b)]['n01']} "
              f"p={mcnemar[(a, b)]['p']:.4f}", flush=True)

    right_all = np.stack([correct[v] for v in "QKVH"], axis=1)  # (N, 4)
    n_right = right_all.sum(axis=1)
    only_one = {v: float(np.mean((n_right == 1) & right_all[:, i])) for i, v in enumerate("QKVH")}
    only_one_right = {
        "all_right": float(np.mean(n_right == 4)),
        "all_wrong": float(np.mean(n_right == 0)),
        "exactly_one_Q": only_one["Q"], "exactly_one_K": only_one["K"],
        "exactly_one_V": only_one["V"], "exactly_one_H": only_one["H"],
    }
    print(f"  only-one-right: {only_one_right}", flush=True)

    return {"model": model_alias, "n": len(y), "mcnemar": mcnemar, "only_one_right": only_one_right}


def write_tables(results: list[dict]) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    path = OUT_TAB / "mcnemar.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "view_a", "view_b", "n10", "n01", "n_discordant", "p"])
        for res in results:
            for (a, b), t in res["mcnemar"].items():
                w.writerow([res["model"], res["n"], a, b, t["n10"], t["n01"], t["n_discordant"], f"{t['p']:.6f}"])
    print("wrote", path)

    path = OUT_TAB / "only_one_right.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        cats = ["all_right", "all_wrong", "exactly_one_Q", "exactly_one_K", "exactly_one_V", "exactly_one_H"]
        w.writerow(["model", "n", "category", "pct"])
        for res in results:
            for cat in cats:
                w.writerow([res["model"], res["n"], cat, f"{100*res['only_one_right'][cat]:.3f}"])
    print("wrote", path)


def load_tables() -> list[dict]:
    # Reconstructs analyse()'s shape from the two CSVs so --stage plot can
    # redraw without rerunning the CV sweep.
    by_model: dict[str, dict] = {}

    with open(OUT_TAB / "mcnemar.csv") as fh:
        for row in csv.DictReader(fh):
            m = row["model"]
            entry = by_model.setdefault(m, {"model": m, "n": int(row["n"]), "mcnemar": {}})
            entry["mcnemar"][(row["view_a"], row["view_b"])] = {
                "n10": int(row["n10"]), "n01": int(row["n01"]),
                "n_discordant": int(row["n_discordant"]), "p": float(row["p"]),
            }

    with open(OUT_TAB / "only_one_right.csv") as fh:
        for row in csv.DictReader(fh):
            entry = by_model[row["model"]]
            entry.setdefault("only_one_right", {})[row["category"]] = float(row["pct"]) / 100.0

    return list(by_model.values())


def _mcnemar_matrix(results: list[dict]) -> tuple[list, float]:
    # Full symmetric matrix (both triangles filled), diagonal left NaN/blank.
    n_views = len(VIEWS)
    mats = []
    for res in results:
        mat = np.full((n_views, n_views), np.nan)
        for (a, b), t in res["mcnemar"].items():
            i, j = VIEWS.index(a), VIEWS.index(b)
            neglog10p = -np.log10(max(t["p"], 1e-300))
            mat[i, j] = neglog10p
            mat[j, i] = neglog10p
        mats.append(mat)
    vmax = max(np.nanmax(m) for m in mats)
    return mats, vmax


def _sig_pairs(res):
    return sorted(
        ((a, b, t["p"]) for (a, b), t in res["mcnemar"].items() if t["p"] < 0.05),
        key=lambda x: x[2],
    )


def _mcnemar_how_to_read(results: list[dict], extra: str) -> str:
    return (
        "Each panel is one LLM; both axes list the 5 views (Q, K, V, H, "
        "QKV). Each cell is an exact McNemar test (binomial test on "
        "discordant pairs, n=600 examples) comparing whether two views' "
        "linear probes get the SAME examples right -- not whether their "
        "mean AUROC differs; the matrix is symmetric by construction "
        "(McNemar(A,B) == McNemar(B,A)) and the diagonal is blank. " + extra
    )


def _mcnemar_interpretation(results: list[dict]) -> str:
    return "; ".join(
        f"{MODEL_PRETTY.get(res['model'], res['model'])}: "
        + (", ".join(f"{a}-{b} p={p:.4f}" for a, b, p in _sig_pairs(res)[:3])
           if _sig_pairs(res) else "no pair significant at p<0.05")
        for res in results
    )


def fig_mcnemar(results: list[dict]) -> None:
    # Full matrix, tight panel spacing, deeper non-washed-out colour ramp,
    # subtle cell borders instead of thick white gridlines.
    fm.apply()
    n_views = len(VIEWS)
    fig, axes = plt.subplots(1, len(results),
                              figsize=(fm.PANEL_H * len(results) + 0.3, fm.PANEL_H),
                              squeeze=False, gridspec_kw={"wspace": 0.35})
    tags = "abc"
    mats, vmax = _mcnemar_matrix(results)

    im = None
    for i, (ax, res, mat) in enumerate(zip(axes[0], results, mats)):
        im = ax.imshow(mat, cmap=MCNEMAR_CMAP, vmin=0, vmax=vmax)
        for k in range(n_views + 1):
            ax.axhline(k - 0.5, color="white", linewidth=0.8, zorder=3)
            ax.axvline(k - 0.5, color="white", linewidth=0.8, zorder=3)
        ax.add_patch(Rectangle((-0.5, -0.5), n_views, n_views, fill=False,
                                edgecolor=fm.INK, linewidth=0.8, zorder=4))

        for (a, b), t in res["mcnemar"].items():
            if t["p"] < 0.05:
                pi, pj = VIEWS.index(a), VIEWS.index(b)
                mark_colour = "white" if mat[pi, pj] > 0.55 * vmax else fm.INK
                ax.plot(pj, pi, marker="*", markersize=5, color=mark_colour,
                        markeredgecolor="none", zorder=5)
                ax.plot(pi, pj, marker="*", markersize=5, color=mark_colour,
                        markeredgecolor="none", zorder=5)

        ax.set_xticks(range(n_views))
        ax.set_yticks(range(n_views))
        ax.set_xticklabels(VIEWS)
        ax.set_yticklabels(VIEWS)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fm.panel_tag(ax, f"({tags[i]}) {MODEL_PRETTY.get(res['model'], res['model'])}")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="-Log10(P)",
                       pad=0.02, aspect=22)
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor(fm.INK)
    paths = fm.save(fig, OUT_FIG, "mcnemar")

    fm.write_how_to_read(
        OUT_FIG, "mcnemar",
        how_to_read=_mcnemar_how_to_read(
            results,
            "Color is -log10(p) (larger = more significant: a p-value of "
            "0.0001 maps to 4, a p-value of 0.5 maps to ~0.3), from an exact "
            "McNemar test -- a binomial test on the pair's discordant "
            "examples (cases where the two views disagree) checking whether "
            "one view rescues more of the other's misses than chance would "
            "predict. A star marks pairs significant at p<0.05, the usual "
            "threshold for treating the disagreement as real rather than "
            "noise."
        ),
        interpretation=_mcnemar_interpretation(results),
    )
    print("wrote", paths[0])


def fig_only_one_right(results: list[dict]) -> None:
    fm.apply()
    fig, ax = plt.subplots(figsize=(fm.PANEL_W * 1.7, fm.PANEL_H * 1.15))

    cats = ["all_right", "exactly_one_Q", "exactly_one_K", "exactly_one_V", "exactly_one_H", "all_wrong"]
    cat_labels = ["All Correct", "Only Q Correct", "Only K Correct",
                  "Only V Correct", "Only H Correct", "All Wrong"]
    # Softened per-model hues (blended 35% toward white) -- same identity as
    # every other figure's model colouring, calmer saturation for a dense
    # grouped-bar layout.
    model_colour = {r["model"]: _mix(c, "#ffffff", 0.35)
                     for r, c in zip(results, fm.PALETTE)}

    n_models = len(results)
    bar_h = 0.8 / n_models
    ys = np.arange(len(cats))
    xmax = 0.0
    for m_i, res in enumerate(results):
        vals = [100 * res["only_one_right"][c] for c in cats]
        xmax = max(xmax, max(vals))
        offset = (m_i - (n_models - 1) / 2) * bar_h
        bars = ax.barh(ys + offset, vals, height=bar_h * 0.88,
                        color=model_colour[res["model"]], edgecolor=fm.INK, linewidth=0.6,
                        label=MODEL_PRETTY.get(res["model"], res["model"]), zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + xmax * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}", va="center", ha="left", fontsize=6.5, color=fm.INK, zorder=3)

    # Divider + gap between the 3 structural groups (all-correct / single-view
    # breakdown / all-wrong) so the grouping logic is visible, not just implied
    # by the labels.
    for boundary in (0.5, 4.5):
        ax.axhline(boundary, color=fm.INK, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)

    ax.set_yticks(list(ys))
    ax.set_yticklabels(cat_labels)
    ax.set_ylim(len(cats) - 0.4, -0.6)
    ax.set_xlim(0, xmax * 1.16)
    ax.set_xlabel("% Of Examples")
    ax.xaxis.grid(True, color=fm.GRAY, linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    fm.legend(ax, loc="lower right")
    paths = fm.save(fig, OUT_FIG, "only_one_right")

    fm.write_how_to_read(
        OUT_FIG, "only_one_right",
        how_to_read=(
            "Horizontal grouped bar chart: one row per outcome category, one "
            "bar colour per LLM, value labels at each bar's tip. For each of "
            "the 600 examples, Q/K/V/H probes are each thresholded at P=0.5 "
            "(out-of-fold prediction) into correct/wrong, then examples are "
            "bucketed by how many of the 4 views agree with the true label: "
            "'All Correct' / 'All Wrong' (all 4 views agree, correctly or "
            "not) and 'Only Q/K/V/H Correct' (exactly 1 of 4 correct, split "
            "by which; dashed lines separate these three structural groups). "
            "The x-axis is the % of all 600 examples falling in that bucket "
            "for that model."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY.get(res['model'], res['model'])}: "
                f"{100*res['only_one_right']['all_right']:.1f}% all-correct, "
                f"{100*sum(res['only_one_right'][f'exactly_one_{v}'] for v in 'QKVH'):.1f}% "
                f"exactly-one-correct (any view)"
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
    ap.add_argument("--stage", choices=("run", "plot"), default="run",
                     help="run: full CV sweep + tables + figures. "
                          "plot: figures only, reads the existing act4 CSVs.")
    args = ap.parse_args()

    pairs = [p.split("=", 1) for p in args.configs]
    wanted = [alias for alias, _ in pairs]

    if args.stage == "plot":
        tables = {OUT_TAB / f"{n}.csv" for n in ("mcnemar", "only_one_right")}
        missing = [p for p in tables if not p.exists()]
        if missing:
            raise SystemExit(f"missing table(s) {missing} -- run --stage run first")
        results = [r for r in load_tables() if r["model"] in wanted]
        missing_models = set(wanted) - {r["model"] for r in results}
        if missing_models:
            raise SystemExit(f"act4 tables have no rows for: {sorted(missing_models)}")
        fig_mcnemar(results)
        fig_only_one_right(results)
        return

    results = []
    for alias, path in pairs:
        print(f"=== {alias} ===", flush=True)
        results.append(analyse(alias, path, args.dataset, args.n_examples))

    write_tables(results)
    fig_mcnemar(results)
    fig_only_one_right(results)


if __name__ == "__main__":
    main()
