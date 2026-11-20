from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.figures import style as fm
from src.config import load_config
from src.data.dataset import _set_stats, compute_stats
from src.train import load_source

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act2"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act2"
N_FOLDS = 5
SEED = 0
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
K_GRID = (1, 2, 5, 10, 20, 50, 100, 200)
VIEW_COLOUR = {"QKV": fm.BLUE, "H": fm.GRAY}
MODEL_PRETTY = {
    "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}


def balanced_indices(labels: list[int], n: int, seed: int = SEED) -> list[int]:
    # n//2 per class, shuffled -- same rationale as Act 1's
    # load_labeled_examples: an unbalanced natural ratio would let one
    # class's features dominate the PCA fit and the StandardScaler.
    labels = np.asarray(labels)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    per_class = n // 2
    if min(len(pos), len(neg)) < per_class:
        raise ValueError(
            f"need {per_class} examples per class but have {len(pos)} positive, "
            f"{len(neg)} negative"
        )
    chosen = np.concatenate([
        rng.choice(pos, per_class, replace=False),
        rng.choice(neg, per_class, replace=False),
    ])
    return sorted(chosen.tolist())


def load_source_for(cfg_path: str, variant: str, dataset: str):
    cfg = load_config(cfg_path)
    cfg = cfg.for_variant(variant) if variant != "default" else cfg
    # The configs' own model.keep_channels defaults to [Q, K] (the detector's
    # representation-ablation pick). This control needs FULL QKV on the qkv
    # corpus, and QKVFieldDataset rejects any non-null keep_channels on a
    # single-channel hidden-states corpus -- so null it out for both.
    cfg.model.keep_channels = None
    return load_source(cfg, dataset, cfg.llm.alias)


def flat_fields(source, idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    # One flat (token- and layer/segment-mean) vector per example, matching
    # Act 1's token-mean convention; channel axis flattened in so QKV lands
    # at 3x H's raw width (L x M x 3 vs L x M x 1) for the SAME model/field.
    stats = compute_stats(source, idx, max_examples=len(idx))
    _set_stats(source, stats)

    X, y = [], []
    for i in idx:
        images, label, _origin = source[i]           # (T, C, L, M)
        X.append(images.mean(dim=0).reshape(-1).numpy())  # token-mean, then flatten C/L/M
        y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)


def _lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000, solver="liblinear")


def cv_auroc_at_k(X: np.ndarray, y: np.ndarray, k: int, folds) -> np.ndarray:
    # Matched-capacity control: PCA fit on the training fold only, truncated
    # to the SAME k components for QKV and H, so neither view gets more
    # usable dimensions than the other going into the probe.
    aucs = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        k_eff = min(k, len(tr) - 1, Xtr.shape[1])
        pca = PCA(n_components=k_eff, random_state=SEED).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

        tr_in, tr_val, y_in, y_val = train_test_split(
            Ztr, y[tr], test_size=0.2, stratify=y[tr], random_state=SEED)
        best = max(C_GRID, key=lambda C: roc_auc_score(
            y_val, _lr(C).fit(tr_in, y_in).decision_function(tr_val)))
        s = _lr(best).fit(Ztr, y[tr]).decision_function(Zte)
        aucs.append(roc_auc_score(y[te], s))
    return np.array(aucs)


def analyse(model_alias: str, cfg_path: str, dataset: str, n_examples: int) -> dict:
    qkv_source = load_source_for(cfg_path, "default", dataset)
    hs_source = load_source_for(cfg_path, "hs", dataset)
    # Both variants come from the same generation (src/extract/run_extraction.py
    # writes every requested variant from one decode), so their labels at a
    # given idx must agree -- pick ONE balanced idx set and use it for both,
    # so the two views are compared on the exact same examples.
    idx = balanced_indices(qkv_source.labels, n_examples)
    assert [hs_source.labels[i] for i in idx] == [qkv_source.labels[i] for i in idx], (
        f"{model_alias}: default/hs label mismatch at shared idx -- corpora out of sync"
    )

    Xq, y = flat_fields(qkv_source, idx)
    Xh, y_h = flat_fields(hs_source, idx)
    assert np.array_equal(y, y_h)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    per_k = {}
    for k in K_GRID:
        q_auc = cv_auroc_at_k(Xq, y, k, folds)
        h_auc = cv_auroc_at_k(Xh, y, k, folds)
        per_k[k] = {"QKV": q_auc, "H": h_auc}
        print(f"  k={k:4d}  QKV {q_auc.mean():.3f}  H {h_auc.mean():.3f}  "
              f"Δ {100*(q_auc.mean()-h_auc.mean()):+.2f}", flush=True)

    return {"model": model_alias, "n": len(y), "d_qkv": Xq.shape[1], "d_h": Xh.shape[1], "per_k": per_k}


def write_table(results: list[dict]) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "pca_capacity_control.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "d_qkv", "d_h", "k", "view", "auroc_mean", "auroc_sd"])
        for res in results:
            for k, row in res["per_k"].items():
                for v in ("QKV", "H"):
                    w.writerow([res["model"], res["n"], res["d_qkv"], res["d_h"], k, v,
                                f"{row[v].mean():.5f}", f"{row[v].std():.5f}"])
    print("wrote", path)


def load_table(path: Path) -> list[dict]:
    # {model -> {k -> {view -> mean_auroc}}}, read straight from the CSV so
    # the plot can be redrawn without rerunning the PCA/nested-CV sweep.
    by_model: dict[str, dict[int, dict[str, float]]] = {}
    dims: dict[str, tuple[int, int]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            m, k, v = row["model"], int(row["k"]), row["view"]
            by_model.setdefault(m, {}).setdefault(k, {})[v] = float(row["auroc_mean"])
            dims[m] = (int(row["d_qkv"]), int(row["d_h"]))
    results = []
    for model, per_k in by_model.items():
        d_qkv, d_h = dims[model]
        results.append({"model": model, "d_qkv": d_qkv, "d_h": d_h, "per_k": per_k})
    return results


def fig_capacity(results: list[dict]) -> None:
    # AUROC is reported on a 0-100 scale throughout this figure, not 0-1.
    def _mean(v):
        return 100.0 * (float(v.mean()) if hasattr(v, "mean") else float(v))

    fm.apply()
    tags = "abc"
    fig, axes = plt.subplots(1, len(results),
                              figsize=(fm.PANEL_W * len(results), fm.PANEL_H), squeeze=False)
    lo = min(_mean(res["per_k"][k][v]) for res in results for k in res["per_k"] for v in ("QKV", "H"))
    hi = max(_mean(res["per_k"][k][v]) for res in results for k in res["per_k"] for v in ("QKV", "H"))
    pad = 0.03 * (hi - lo)
    for i, (ax, res) in enumerate(zip(axes[0], results)):
        ks = sorted(res["per_k"])
        ys = range(len(ks))
        h_vals = [_mean(res["per_k"][k]["H"]) for k in ks]
        q_vals = [_mean(res["per_k"][k]["QKV"]) for k in ks]
        for y, h, q in zip(ys, h_vals, q_vals):
            ax.plot([h, q], [y, y], color=fm.GRAY, lw=1.2, zorder=1,
                    path_effects=fm.HALO, solid_capstyle="round")
        ax.scatter(h_vals, ys, color=VIEW_COLOUR["H"], s=36, zorder=2,
                   edgecolor="white", linewidth=0.8, label="H")
        ax.scatter(q_vals, ys, color=VIEW_COLOUR["QKV"], s=36, zorder=2,
                   edgecolor="white", linewidth=0.8, label="QKV")
        ax.set_yticks(list(ys))
        ax.set_yticklabels([str(k) for k in ks])
        ax.set_ylim(-0.6, len(ks) - 0.4)
        ax.invert_yaxis()
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel("AUROC")
        if i == 0:
            ax.set_ylabel("PCA Components")
        fm.legend(ax)
        fm.panel_tag(ax, f"({tags[i]}) {MODEL_PRETTY.get(res['model'], res['model'])}")
    paths = fm.save(fig, OUT_FIG, "pca_capacity_control")

    def best_k(res, v):
        k = max(res["per_k"], key=lambda k: _mean(res["per_k"][k][v]))
        return k, _mean(res["per_k"][k][v])

    def delta_at(res, k):
        return _mean(res["per_k"][k]["QKV"]) - _mean(res["per_k"][k]["H"])

    fm.write_how_to_read(
        OUT_FIG, "pca_capacity_control",
        how_to_read=(
            "Each panel is one LLM; each row is one PCA component count k "
            "the probe was restricted to, fit on the training fold only and "
            "truncated to the SAME k for both views -- so the raw "
            "dimensionality gap (QKV's pooled field is 3x H's, one channel "
            "vs three) is irrelevant here by construction: both views get "
            "exactly k usable components. The dumbbell at each row connects "
            "H's mean 5-fold CV AUROC (gray dot) to QKV's (blue dot) on the "
            "pooled field (token-mean, flattened layer x segment x channel); "
            "the line's length is the gap at that k. If QKV's advantage over "
            "H were just more usable capacity, equalising k should have "
            "closed every gap to near zero."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY.get(res['model'], res['model'])}: Δ(QKV-H) "
                f"{delta_at(res, min(res['per_k'])):+.1f} pts at k={min(res['per_k'])}, "
                f"{delta_at(res, max(res['per_k'])):+.1f} pts at k={max(res['per_k'])}"
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
    ap.add_argument("--n-examples", type=int, default=600,
                     help="balanced (n//2 per class) examples; this control reads already-pooled "
                          "fields off disk, so it can afford more than Act 1's 300")
    ap.add_argument("--stage", choices=("run", "plot"), default="run",
                     help="run: full PCA/nested-CV sweep + table + figure. "
                          "plot: figure only, reads the existing pca_capacity_control.csv.")
    args = ap.parse_args()

    pairs = [p.split("=", 1) for p in args.configs]

    if args.stage == "plot":
        table_path = OUT_TAB / "pca_capacity_control.csv"
        if not table_path.exists():
            raise SystemExit(f"{table_path} missing -- run --stage run first")
        wanted = [alias for alias, _ in pairs]
        results = [r for r in load_table(table_path) if r["model"] in wanted]
        missing = set(wanted) - {r["model"] for r in results}
        if missing:
            raise SystemExit(f"{table_path} has no rows for: {sorted(missing)}")
        fig_capacity(results)
        return

    results = []
    for alias, path in pairs:
        print(f"=== {alias} ===", flush=True)
        results.append(analyse(alias, path, args.dataset, args.n_examples))

    write_table(results)
    fig_capacity(results)


if __name__ == "__main__":
    main()
