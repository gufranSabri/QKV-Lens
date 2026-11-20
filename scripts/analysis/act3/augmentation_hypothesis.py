from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.figures import style as fm
from scripts.analysis.act2.pca_capacity_control import balanced_indices, load_source_for
from scripts.analysis.act3.cell_signal_maps import per_token_mean_fields

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act3"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act3"
SEED = 0
N_FOLDS = 5
N_NULL = 20  # random 3-way splits of H sampled for the null distribution
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
MODEL_PRETTY = {
    "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def cka(A: np.ndarray, B: np.ndarray) -> float:
    # Linear CKA between two (N, D) views of the SAME N examples -- the
    # disagreement/diversity metric for prediction (a): lower CKA means the
    # two views carry more different structure.
    a = A - A.mean(0)
    b = B - B.mean(0)
    cross = np.linalg.norm(a.T @ b) ** 2
    denom = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return float(cross / denom) if denom > 0 else 0.0


def random_3way_split(D: int, rng: np.random.Generator) -> list[np.ndarray]:
    # Partitions D dims into 3 groups as evenly as possible, matching Q/K/V's
    # own channel count -- a fair null for "3 different views of the same field."
    perm = rng.permutation(D)
    return np.array_split(perm, 3)


def _lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000, solver="liblinear")


def cv_auroc(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    aucs = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        tr_in, tr_val, y_in, y_val = train_test_split(
            Xtr, y[tr], test_size=0.2, stratify=y[tr], random_state=SEED)
        best = max(C_GRID, key=lambda C: roc_auc_score(
            y_val, _lr(C).fit(tr_in, y_in).decision_function(tr_val)))
        s = _lr(best).fit(Xtr, y[tr]).decision_function(Xte)
        aucs.append(roc_auc_score(y[te], s))
    return np.array(aucs)


def prediction_a(views: dict[str, np.ndarray], h_flat: np.ndarray, rng: np.random.Generator) -> dict:
    # Real: pairwise CKA among Q, K, V. Null: pairwise CKA among 3 random
    # splits of H, repeated N_NULL times for a distribution to compare against.
    real_cka = [cka(views[a], views[b]) for i, a in enumerate("QKV") for b in "QKV"[i + 1:]]

    null_cka = []
    for _ in range(N_NULL):
        groups = random_3way_split(h_flat.shape[1], rng)
        parts = [h_flat[:, g] for g in groups]
        null_cka += [cka(parts[i], parts[j]) for i in range(3) for j in range(i + 1, 3)]

    return {
        "real_mean_cka": float(np.mean(real_cka)), "real_cka": real_cka,
        "null_mean_cka": float(np.mean(null_cka)), "null_std_cka": float(np.std(null_cka)),
    }


def prediction_b(views: dict[str, np.ndarray], h_flat: np.ndarray, y: np.ndarray,
                  folds, rng: np.random.Generator) -> dict:
    # Real: gain from concatenating Q+K+V over the best single one of them.
    single_auc = {v: cv_auroc(views[v], y, folds) for v in "QKV"}
    concat_auc = cv_auroc(np.concatenate([views[v] for v in "QKV"], axis=1), y, folds)
    best_single = max(single_auc, key=lambda v: single_auc[v].mean())
    real_gain = float(concat_auc.mean() - single_auc[best_single].mean())

    # Null: same computation on N_NULL random 3-way splits of H.
    null_gains = []
    for _ in range(N_NULL):
        groups = random_3way_split(h_flat.shape[1], rng)
        parts = [h_flat[:, g] for g in groups]
        part_auc = [cv_auroc(p, y, folds) for p in parts]
        concat = cv_auroc(np.concatenate(parts, axis=1), y, folds)
        null_gains.append(float(concat.mean() - max(a.mean() for a in part_auc)))

    return {
        "real_gain": real_gain, "best_single_view": best_single,
        "null_mean_gain": float(np.mean(null_gains)), "null_std_gain": float(np.std(null_gains)),
        "null_gains": null_gains,
    }


def analyse(model_alias: str, cfg_path: str, dataset: str, n_examples: int, seed: int) -> dict:
    qkv_source = load_source_for(cfg_path, "default", dataset)
    hs_source = load_source_for(cfg_path, "hs", dataset)
    idx = balanced_indices(qkv_source.labels, n_examples)
    assert [hs_source.labels[i] for i in idx] == [qkv_source.labels[i] for i in idx], (
        f"{model_alias}: default/hs label mismatch at shared idx -- corpora out of sync"
    )

    qkv_fields, y = per_token_mean_fields(qkv_source, idx)
    hs_fields, y_h = per_token_mean_fields(hs_source, idx)
    assert np.array_equal(y, y_h)

    views = {v: flatten(qkv_fields[v]) for v in "QKV"}
    h_flat = flatten(hs_fields["H"])

    rng = np.random.default_rng(seed)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    a = prediction_a(views, h_flat, rng)
    b = prediction_b(views, h_flat, y, folds, rng)
    print(f"  (a) disagreement: real mean CKA={a['real_mean_cka']:.3f}, "
          f"null mean CKA={a['null_mean_cka']:.3f} ± {a['null_std_cka']:.3f} "
          f"(lower CKA = more diverse)", flush=True)
    print(f"  (b) concat gain: real +{100*b['real_gain']:.2f} pts (over best single {b['best_single_view']}), "
          f"null +{100*b['null_mean_gain']:.2f} ± {100*b['null_std_gain']:.2f} pts", flush=True)

    return {"model": model_alias, "n": len(y), "a": a, "b": b}


def write_table(results: list[dict]) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "augmentation_hypothesis.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "metric", "real", "null_mean", "null_std"])
        for res in results:
            w.writerow([res["model"], res["n"], "pairwise_cka",
                        f"{res['a']['real_mean_cka']:.5f}", f"{res['a']['null_mean_cka']:.5f}",
                        f"{res['a']['null_std_cka']:.5f}"])
            w.writerow([res["model"], res["n"], "concat_gain",
                        f"{res['b']['real_gain']:.5f}", f"{res['b']['null_mean_gain']:.5f}",
                        f"{res['b']['null_std_gain']:.5f}"])
    print("wrote", path)


def load_table(path: Path) -> list[dict]:
    # Reconstructs analyse()'s {model: {a: {...}, b: {...}}} shape from the
    # CSV so the figure can be redrawn without rerunning the nested-CV sweep.
    by_model: dict[str, dict] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            entry = by_model.setdefault(row["model"], {"model": row["model"], "n": int(row["n"])})
            if row["metric"] == "pairwise_cka":
                entry["a"] = {"real_mean_cka": float(row["real"]), "null_mean_cka": float(row["null_mean"]),
                              "null_std_cka": float(row["null_std"])}
            else:
                entry["b"] = {"real_gain": float(row["real"]), "null_mean_gain": float(row["null_mean"]),
                              "null_std_gain": float(row["null_std"])}
    return list(by_model.values())


def _z(real: float, null_mean: float, null_std: float) -> float:
    return (real - null_mean) / max(null_std, 1e-12)


def fig_augmentation(results: list[dict]) -> None:
    fm.apply()
    fig, ax = plt.subplots(figsize=(fm.PANEL_W * 1.6, fm.PANEL_H))

    models = [MODEL_PRETTY.get(r["model"], r["model"]) for r in results]
    # Prediction (a) wants LOWER CKA than the null (more diverse), so its
    # z-score is negated: for both marks, positive = supports the hypothesis.
    z_diversity = [-_z(r["a"]["real_mean_cka"], r["a"]["null_mean_cka"], r["a"]["null_std_cka"])
                   for r in results]
    z_gain = [_z(r["b"]["real_gain"], r["b"]["null_mean_gain"], r["b"]["null_std_gain"])
              for r in results]

    n = len(results)
    ys = np.arange(n)
    for y, zd, zg in zip(ys, z_diversity, z_gain):
        ax.plot([zd, zg], [y, y], color=fm.GRAY, lw=1.2, zorder=1,
                path_effects=fm.HALO, solid_capstyle="round")
    ax.scatter(z_diversity, ys, color=fm.BLUE, s=70, zorder=2,
               edgecolor="white", linewidth=0.8, label="Diversity (Prediction a)")
    ax.scatter(z_gain, ys, color=fm.VIOLET, s=70, zorder=2,
               edgecolor="white", linewidth=0.8, label="Concat. Gain (Prediction b)")
    ax.axvline(0, color=fm.INK, lw=0.8, zorder=0)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(models)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Z-Score vs. Null")
    fm.legend(ax)
    paths = fm.save(fig, OUT_FIG, "augmentation_hypothesis")
    fm.write_how_to_read(
        OUT_FIG, "augmentation_hypothesis",
        how_to_read=(
            "One row per model, testing the projection-as-augmentation "
            "hypothesis against a null of 3 random splits of H's own dims "
            "(resampled 20 times). Each point is a z-score: how many null "
            "standard deviations the real Q/K/V value sits from the null "
            "mean, oriented so positive = supports the hypothesis. 'Diversity' "
            "(blue) is -1x the z-score of pairwise linear CKA among Q, K, V "
            "(the hypothesis predicts LOWER CKA, i.e. more mutual difference, "
            "than 3 arbitrary views of the same H would have). 'Concatenation "
            "Gain' (violet) is the z-score of the AUROC gain from "
            "concatenating Q+K+V over the best single one of them, against "
            "the same gain computed for 3 random H splits. The connecting "
            "line is only a visual pairing of the two predictions per model, "
            "not a trend. A point at 0 means the real value is "
            "indistinguishable from chance; |z|>~2 is a meaningful departure "
            "from the null."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY.get(r['model'], r['model'])}: diversity z="
                f"{-_z(r['a']['real_mean_cka'], r['a']['null_mean_cka'], r['a']['null_std_cka']):+.2f}, "
                f"concat-gain z={_z(r['b']['real_gain'], r['b']['null_mean_gain'], r['b']['null_std_gain']):+.2f}"
                for r in results
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
                     help="run: full analysis + table + figure. "
                          "plot: figure only, reads the existing augmentation_hypothesis.csv.")
    args = ap.parse_args()

    pairs = [p.split("=", 1) for p in args.configs]

    if args.stage == "plot":
        table_path = OUT_TAB / "augmentation_hypothesis.csv"
        if not table_path.exists():
            raise SystemExit(f"{table_path} missing -- run --stage run first")
        wanted = [alias for alias, _ in pairs]
        results = [r for r in load_table(table_path) if r["model"] in wanted]
        missing = set(wanted) - {r["model"] for r in results}
        if missing:
            raise SystemExit(f"{table_path} has no rows for: {sorted(missing)}")
        fig_augmentation(results)
        return

    results = []
    for alias, path in pairs:
        print(f"=== {alias} ===", flush=True)
        results.append(analyse(alias, path, args.dataset, args.n_examples, SEED))

    write_table(results)
    fig_augmentation(results)


if __name__ == "__main__":
    main()
