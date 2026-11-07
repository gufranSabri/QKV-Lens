from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_FIG = REPO_ROOT / "docs" / "figures" / "hidden_vs_qkv"
OUT_TAB = REPO_ROOT / "docs" / "tables" / "hidden_vs_qkv"
OUT_REPORT = REPO_ROOT / "docs" / "reports" / "hidden_vs_qkv.md"
OUT_CACHE = REPO_ROOT / "docs" / "hidden_vs_qkv_cache"

DATASET = "triviaqa"
MODELS = ["llama3.1_8b", "opt_6.7b", "qwen2.5_7b"]
PRETTY = {"llama3.1_8b": "Llama-3.1-8B", "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B"}
VIEWS = ("H", "Q", "K", "V")
PAIRS = tuple(f"{a}+{b}" for i, a in enumerate(VIEWS) for b in VIEWS[i + 1:])
REPS = (*VIEWS, *PAIRS, "QKV", "H+QKV")
BAR_REPS = ("H", "Q", "K", "V", "QKV", "H+QKV")
PCA_REPS = ("H", "Q", "K", "V", "QKV")
COLOURS = {"H": "#7F7F7F", "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
           "QKV": "#C44E52", "H+QKV": "#937860"}
C_GRID = (1e-4, 1e-3, 1e-2)
PCA_KS = (1, 2, 5, 10, 20, 50, 100)
TOPK_CELLS = (1, 4, 16, 64)
CELL_THRESHOLDS = (0.6, 0.65, 0.7)
N_FOLDS = 5
N_BOOT = 1000
SEED = 0


def _manifest(root: Path) -> dict[int, dict]:
    with open(root / "manifest.jsonl") as fh:
        return {r["idx"]: r for r in map(json.loads, filter(str.strip, fh))}


def _response(meta: Path) -> str:
    return meta.read_text().split("response:", 1)[1].split("\ngold:", 1)[0]


def load_aligned(model: str, cfg, refresh: bool) -> dict:
    cache = OUT_CACHE / f"{model}_{DATASET}.npz"
    qkv_root = cfg.dataset_dir_for(DATASET, model)
    hs_root = cfg.for_variant("hs").dataset_dir_for(DATASET, model)
    newest = max((r / "manifest.jsonl").stat().st_mtime for r in (qkv_root, hs_root))
    if cache.exists() and not refresh and cache.stat().st_mtime > newest:
        return dict(np.load(cache))
    qm, hm = _manifest(qkv_root), _manifest(hs_root)
    common = sorted(set(qm) & set(hm))
    nonempty = [i for i in common if qm[i]["n_tokens"] > 0 and hm[i]["n_tokens"] > 0]
    same_shape = [i for i in nonempty
                  if qm[i]["n_tokens"] == hm[i]["n_tokens"] and qm[i]["label"] == hm[i]["label"]]

    def same_response(i):
        return (_response(qkv_root / qm[i]["dir"] / "meta.txt")
                == _response(hs_root / hm[i]["dir"] / "meta.txt"))

    first = [int(bool(json.loads((r / "geometry.json").read_text()).get("prompt_row"))) for r in (qkv_root, hs_root)]

    def token_mean(i):
        q = np.load(qkv_root / qm[i]["dir"] / "tokens.npy")[first[0]:][: cfg.extract.max_tokens]
        h = np.load(hs_root / hm[i]["dir"] / "tokens.npy")[first[1]:][: cfg.extract.max_tokens, ..., 0]
        return q.astype(np.float32).mean(0), h.astype(np.float32).mean(0)

    with ThreadPoolExecutor(16) as ex:
        keep = [i for i, ok in zip(same_shape, ex.map(same_response, same_shape)) if ok]
        fields = list(ex.map(token_mean, keep))

    data = {
        "qkv": np.stack([f[0] for f in fields]),
        "h": np.stack([f[1] for f in fields]),
        "y": np.array([qm[i]["label"] for i in keep], dtype=np.int64),
        "idx": np.array(keep),
        "n_total": np.array(len(common)),
        "n_empty": np.array(len(common) - len(nonempty)),
        "n_divergent": np.array(len(nonempty) - len(keep)),
    }
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **data)
    os.replace(tmp, cache)
    return data


def segment_widths(cfg, model: str) -> dict:
    g = json.loads((cfg.dataset_dir_for(DATASET, model) / "geometry.json").read_text())
    geo, m = g["geometry"], g["n_segments"]
    kv = geo["n_kv_heads"] * geo["head_dim"] / m
    return {"head_dim": geo["head_dim"], "H": geo["hidden_size"] / m,
            "Q": geo["n_heads"] * geo["head_dim"] / m, "K": kv, "V": kv}


def build_rep(fields: dict, name: str) -> np.ndarray:
    n = len(fields["H"])
    return np.concatenate(
        [fields[p].reshape(n, -1) for p in name.replace("QKV", "Q+K+V").split("+")], 1)


def _lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000)


def _auroc_cols(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    n1 = y.sum()
    n0 = len(y) - n1
    return (rankdata(S, axis=0)[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def cv_probe(X: np.ndarray, y: np.ndarray, folds) -> tuple[np.ndarray, np.ndarray, list]:
    oof, aucs, cs = np.zeros(len(y)), [], []
    for k, (tr, te) in enumerate(folds):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        a, b, ya, yb = train_test_split(Xtr, y[tr], test_size=0.2, stratify=y[tr], random_state=k)
        best = max(C_GRID, key=lambda C: roc_auc_score(yb, _lr(C).fit(a, ya).decision_function(b)))
        oof[te] = _lr(best).fit(Xtr, y[tr]).decision_function(Xte)
        aucs.append(roc_auc_score(y[te], oof[te]))
        cs.append(best)
    return np.array(aucs), oof, cs


def topk_probe(X: np.ndarray, y: np.ndarray, folds) -> dict[int, float]:
    out = {k: [] for k in TOPK_CELLS}
    for tr, te in folds:
        order = np.argsort(np.abs(_auroc_cols(y[tr], X[tr]) - 0.5))[::-1]
        for k in TOPK_CELLS:
            cols = order[:k]
            sc = StandardScaler().fit(X[tr][:, cols])
            s = _lr(1.0).fit(sc.transform(X[tr][:, cols]), y[tr]).decision_function(sc.transform(X[te][:, cols]))
            out[k].append(roc_auc_score(y[te], s))
    return {k: float(np.mean(v)) for k, v in out.items()}


def univariate(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, neg = X[y == 1], X[y == 0]
    sd = np.sqrt((pos.var(0, ddof=1) + neg.var(0, ddof=1)) / 2)
    d = (pos.mean(0) - neg.mean(0)) / np.maximum(sd, 1e-12)
    auc = _auroc_cols(y, X)
    return d, np.maximum(auc, 1 - auc)


def spectrum(X: np.ndarray) -> dict:
    Xc = (X - X.mean(0)).astype(np.float64)
    raw_var = Xc.var(0)
    raw_top = np.linalg.eigvalsh(Xc.T @ Xc / (len(X) - 1))[-1]
    Z = Xc / np.maximum(Xc.std(0), 1e-12)
    lam = np.clip(np.linalg.eigvalsh(Z.T @ Z / (len(X) - 1))[::-1], 0, None)
    cum = np.cumsum(lam / lam.sum())
    top = max(1, len(raw_var) // 100)
    return {"cum": cum, "pcs_90": int(np.searchsorted(cum, 0.9) + 1),
            "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "raw_pc1_share": float(raw_top / raw_var.sum()),
            "top1pct_var_share": float(np.sort(raw_var)[::-1][:top].sum() / raw_var.sum())}


def pca_probe(X: np.ndarray, y: np.ndarray, folds) -> dict[int, float]:
    out = {k: [] for k in PCA_KS}
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        p = PCA(n_components=max(PCA_KS), svd_solver="randomized", random_state=SEED)
        Ztr = p.fit_transform(sc.transform(X[tr]))
        Zte = p.transform(sc.transform(X[te]))
        for k in PCA_KS:
            s = _lr(1.0).fit(Ztr[:, :k], y[tr]).decision_function(Zte[:, :k])
            out[k].append(roc_auc_score(y[te], s))
    return {k: float(np.mean(v)) for k, v in out.items()}


def pca_2d(X: np.ndarray, y: np.ndarray) -> dict:
    Z = StandardScaler().fit_transform(X)
    p = PCA(n_components=2, random_state=SEED).fit(Z)
    pc = p.transform(Z)
    auc = _auroc_cols(y, pc)
    return {"pc": pc, "evr": p.explained_variance_ratio_, "auc": np.maximum(auc, 1 - auc)}


def layer_probe(field: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    res = np.zeros(field.shape[1])
    for l in range(field.shape[1]):
        X = field[:, l, :]
        aucs = []
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            s = _lr(1.0).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
            aucs.append(roc_auc_score(y[te], s))
        res[l] = np.mean(aucs)
    return res


def prior_matched(score: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    # Each view flags the same fraction as positive, so error counts are comparable.
    pred = np.zeros(len(y), dtype=bool)
    for tr, te in folds:
        pred[te] = score[te] >= np.quantile(score[te], 1 - y[tr].mean())
    return pred


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if n01 + n10 == 0:
        return 1.0
    return float(chi2.sf((abs(n01 - n10) - 1) ** 2 / (n01 + n10), 1))


def _zscore(A: np.ndarray) -> np.ndarray:
    flat = A.reshape(len(A), -1).astype(np.float64)
    return (flat - flat.mean(0)) / np.maximum(flat.std(0), 1e-12)


def cka_layers(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    _, L, M = A.shape
    a, b = _zscore(A), _zscore(B)
    cross = ((a.T @ b).reshape(L, M, L, M) ** 2).sum((1, 3))
    ar = np.arange(L)
    na = np.sqrt(((a.T @ a).reshape(L, M, L, M)[ar, :, ar, :] ** 2).sum((1, 2)))
    nb = np.sqrt(((b.T @ b).reshape(L, M, L, M)[ar, :, ar, :] ** 2).sum((1, 2)))
    return cross / (na[:, None] * nb[None, :])


def cka_full(A: np.ndarray, B: np.ndarray) -> float:
    a, b = _zscore(A), _zscore(B)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def _fmt_delta(t: tuple) -> str:
    return f"{100 * t[0]:+.2f} [{100 * t[1]:+.2f}, {100 * t[2]:+.2f}]"


def analyse(model: str, refresh: bool, limit: int | None, rng) -> dict:
    t0 = time.time()
    cfg = load_config(str(REPO_ROOT / "configs" / DATASET / f"{model}.yaml"))
    d = load_aligned(model, cfg, refresh)
    if limit:
        d = {k: (v[:limit] if v.ndim else v) for k, v in d.items()}
    y = d["y"]
    n = len(y)
    fields = {"H": d["h"], **{c: d["qkv"][..., k] for k, c in enumerate("QKV")}}
    L, M = fields["H"].shape[1:]
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n), y))
    print(f"\n=== {PRETTY[model]} / {DATASET}: {n} aligned examples (of {int(d['n_total'])}; "
          f"{int(d['n_empty'])} empty, {int(d['n_divergent'])} divergent between the two "
          f"generations), hallucination rate {y.mean():.3f}, field {L}x{M}", flush=True)

    probe, oof = {}, {}
    for name in REPS:
        aucs, oof[name], cs = cv_probe(build_rep(fields, name), y, folds)
        probe[name] = {"auroc": float(aucs.mean()), "sd": float(aucs.std()),
                       "C": max(set(cs), key=cs.count),
                       "n_features": L * M * len(name.replace("QKV", "Q+K+V").split("+"))}
        print(f"  probe {name:6s} AUROC {aucs.mean():.4f} ± {aucs.std():.4f}", flush=True)

    col = {r: i for i, r in enumerate(REPS)}
    S = np.stack([oof[r] for r in REPS], 1)
    full = _auroc_cols(y, S)
    boot = np.empty((N_BOOT, len(REPS)))
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        boot[b] = _auroc_cols(y[s], S[s])

    def delta(a: str, b: str) -> tuple[float, float, float]:
        lo, hi = np.percentile(boot[:, col[a]] - boot[:, col[b]], [2.5, 97.5])
        return float(full[col[a]] - full[col[b]]), float(lo), float(hi)

    def pair(a: str, b: str) -> str:
        return "+".join(sorted((a, b), key=VIEWS.index))

    vs_h = {r: delta(r, "H") for r in REPS if r != "H"}
    unique = {(a, b): delta(pair(a, b), a) for a in VIEWS for b in VIEWS if a != b}
    h_adds = delta("H+QKV", "QKV")
    print("  Δ vs H [95% CI]: " + ", ".join(f"{r} {_fmt_delta(vs_h[r])}" for r in ("Q", "K", "V", "QKV")))
    print("  gain from adding a view (base -> base+added):")
    for a in VIEWS:
        print("    " + a + ": " + ", ".join(f"+{b} {_fmt_delta(unique[(a, b)])}" for b in VIEWS if b != a))
    print(f"  H on top of QKV: {_fmt_delta(h_adds)}", flush=True)

    clarity = {}
    for v in VIEWS:
        X = fields[v].reshape(n, -1)
        dv, cell = univariate(X, y)
        clarity[v] = {"d": dv.reshape(L, M), "cell_auc": cell, "topk": topk_probe(X, y, folds),
                      "mean_abs_d": float(np.abs(dv).mean()), "max_abs_d": float(np.abs(dv).max())}
        print(f"  clarity {v}: best cell AUROC {cell.max():.3f} | "
              + " | ".join(f"cells>={t} {int((cell >= t).sum())}" for t in CELL_THRESHOLDS)
              + " | top-k cell probe " + " ".join(f"k={k}:{a:.3f}" for k, a in clarity[v]["topk"].items()),
              flush=True)

    layers = {v: layer_probe(fields[v], y, folds) for v in VIEWS}
    won = np.stack([layers[v] for v in VIEWS]).argmax(0)
    layers_won = {v: int((won == i).sum()) for i, v in enumerate(VIEWS)}
    print("  per-layer probe, layers won: " + ", ".join(f"{v} {layers_won[v]}" for v in VIEWS)
          + " | best layer: " + ", ".join(
              f"{v} {layers[v].max():.3f}@L{int(layers[v].argmax()) + 1}" for v in VIEWS), flush=True)

    correct = {v: prior_matched(oof[v], y, folds) == y for v in VIEWS}
    right_wrong = np.array([[np.mean(correct[a] & ~correct[b]) for b in VIEWS] for a in VIEWS])
    mcn = np.array([[mcnemar_p(correct[a], correct[b]) for b in VIEWS] for a in VIEWS])
    only_right = {v: float(np.mean(correct[v] & ~np.any([correct[u] for u in VIEWS if u != v], 0)))
                  for v in VIEWS}
    rho = spearmanr(np.stack([oof[v] for v in VIEWS], 1)).statistic
    print("  only this view correct: " + ", ".join(f"{v} {100 * only_right[v]:.2f}%" for v in VIEWS)
          + " | Spearman(H, Q/K/V scores): " + ", ".join(f"{v} {rho[0, i]:.3f}" for i, v in enumerate(VIEWS) if v != "H"),
          flush=True)

    pca = {}
    for r in PCA_REPS:
        X = build_rep(fields, r)
        pca[r] = {**spectrum(X), "probe": pca_probe(X, y, folds)}
        print(f"  PCA {r:4s} PCs@90% {pca[r]['pcs_90']:4d} | PR {pca[r]['participation_ratio']:6.1f} | "
              f"raw PC1 {pca[r]['raw_pc1_share']:.3f} | top-1% cells {pca[r]['top1pct_var_share']:.3f} | "
              f"probe on top-k PCs " + " ".join(f"{k}:{a:.3f}" for k, a in pca[r]["probe"].items()), flush=True)
    scatter = {r: pca_2d(build_rep(fields, r), y) for r in ("H", "QKV")}

    cka = {c: cka_layers(fields["H"], fields[c]) for c in "QKV"}
    # Q/K/V at layer l are computed from H at layer l-1, hence diag offset 1.
    cka_summary = {c: (float(np.diag(cka[c]).mean()), float(np.diag(cka[c], k=1).mean()),
                       cka_full(fields["H"], fields[c])) for c in "QKV"}
    print("  CKA(H, ·) same-layer / input-layer / full-field: " + ", ".join(
        f"{c} {s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f}" for c, s in cka_summary.items()))
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    return {"model": model, "n": n, "n_total": int(d["n_total"]), "n_empty": int(d["n_empty"]),
            "n_divergent": int(d["n_divergent"]), "rate": float(y.mean()), "L": L, "M": M, "y": y,
            "widths": segment_widths(cfg, model), "probe": probe, "vs_h": vs_h, "unique": unique,
            "h_adds": h_adds, "clarity": clarity, "layers": layers, "layers_won": layers_won,
            "right_wrong": right_wrong, "mcnemar": mcn, "only_right": only_right, "rho": rho,
            "pca": pca, "scatter": scatter, "cka": cka, "cka_summary": cka_summary}


def _save(fig, name: str) -> None:
    fig.savefig(OUT_FIG / f"{name}.png", dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT_FIG / f"{name}.png")


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)


def _row(results: list[dict], h: float = 3.8):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), h), squeeze=False)
    return fig, axes[0]


def fig_probe(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(BAR_REPS)
    lo = min(res["probe"][r]["auroc"] for res in results for r in BAR_REPS)
    for i, rep in enumerate(BAR_REPS):
        x = np.arange(len(results)) + (i - (len(BAR_REPS) - 1) / 2) * w
        ax.bar(x, [res["probe"][rep]["auroc"] for res in results], w,
               yerr=[res["probe"][rep]["sd"] for res in results],
               color=COLOURS[rep], edgecolor="black", linewidth=0.3, capsize=2, label=rep)
    ax.set_xticks(np.arange(len(results)))
    ax.set_xticklabels([PRETTY[r["model"]] for r in results])
    ax.set_ylim(lo - 0.03, None)
    ax.set_ylabel("linear-probe AUROC (5-fold CV)")
    ax.set_title("Linear probe on pooled, token-averaged fields", fontweight="bold")
    ax.legend(ncol=len(BAR_REPS), fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _style(ax)
    _save(fig, "probe_auroc")


def fig_clarity(results: list[dict]) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(5 * len(results), 7.5), squeeze=False)
    for j, res in enumerate(results):
        ax = axes[0, j]
        for v in VIEWS:
            cell = np.sort(res["clarity"][v]["cell_auc"])[::-1]
            ax.plot(np.arange(1, len(cell) + 1), cell, color=COLOURS[v], lw=1.8, label=v)
        ax.set_xscale("log")
        ax.set_xlabel("cell rank")
        ax.set_title(PRETTY[res["model"]], fontweight="bold")
        _style(ax)
        ax = axes[1, j]
        for v in VIEWS:
            ax.plot(TOPK_CELLS, [res["clarity"][v]["topk"][k] for k in TOPK_CELLS], "o-", ms=3,
                    color=COLOURS[v], lw=1.8, label=v)
        ax.set_xscale("log")
        ax.set_xlabel("# most discriminative cells used (picked on train fold)")
        _style(ax)
    axes[0, 0].set_ylabel("single-cell AUROC (sorted)")
    axes[1, 0].set_ylabel("probe AUROC on top-k cells")
    axes[0, 0].legend(frameon=False)
    _save(fig, "clarity")


def _matrix(ax, mat: np.ndarray, text: list[list[str]], cmap: str, vmin: float, vmax: float, title: str):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(len(VIEWS)):
        for j in range(len(VIEWS)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(VIEWS)))
    ax.set_yticks(range(len(VIEWS)))
    ax.set_xticklabels(VIEWS)
    ax.set_yticklabels(VIEWS)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def fig_unique(results: list[dict]) -> None:
    fig, axes = _row(results, 4.2)
    vmax = max(abs(100 * t[0]) for res in results for t in res["unique"].values())
    for ax, res in zip(axes, results):
        mat = np.full((4, 4), np.nan)
        text = [[""] * 4 for _ in VIEWS]
        for (a, b), t in res["unique"].items():
            i, j = VIEWS.index(a), VIEWS.index(b)
            mat[i, j] = 100 * t[0]
            text[i][j] = f"{100 * t[0]:+.2f}" + ("*" if t[1] > 0 or t[2] < 0 else "")
        im = _matrix(ax, mat, text, "RdBu_r", -vmax, vmax, PRETTY[res["model"]])
        ax.set_xlabel("added view")
        ax.set_ylabel("base view")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85, label="Δ AUROC ×100  (* = 95% CI excludes 0)")
    _save(fig, "unique_info")


def fig_disagreement(results: list[dict]) -> None:
    fig, axes = _row(results, 4.2)
    vmax = max(100 * res["right_wrong"].max() for res in results)
    for ax, res in zip(axes, results):
        mat = 100 * res["right_wrong"]
        np.fill_diagonal(mat, np.nan)
        text = [["" if i == j else f"{mat[i, j]:.1f}" for j in range(4)] for i in range(4)]
        im = _matrix(ax, mat, text, "Purples", 0, vmax, PRETTY[res["model"]])
        ax.set_xlabel("view B (wrong)")
        ax.set_ylabel("view A (right)")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85, label="% A right, B wrong")
    _save(fig, "disagreement")


def fig_layers(results: list[dict]) -> None:
    fig, axes = _row(results)
    for ax, res in zip(axes, results):
        for v in VIEWS:
            ax.plot(np.arange(1, res["L"] + 1), res["layers"][v], color=COLOURS[v], lw=1.8, label=v)
        ax.set_title(PRETTY[res["model"]], fontweight="bold")
        ax.set_xlabel("layer")
        _style(ax)
    axes[0].set_ylabel("per-layer probe AUROC")
    axes[0].legend(frameon=False)
    _save(fig, "layer_probe")


def fig_dmaps(results: list[dict]) -> None:
    fig, axes = plt.subplots(len(results), 4, figsize=(14, 3.4 * len(results)), squeeze=False)
    for row, res in zip(axes, results):
        vmax = max(res["clarity"][v]["max_abs_d"] for v in VIEWS)
        for ax, v in zip(row, VIEWS):
            im = ax.imshow(res["clarity"][v]["d"], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto", origin="lower")
            ax.set_title(f"{PRETTY[res['model']]} — {v}", fontsize=10)
            ax.set_xlabel("segment (M)")
            ax.set_ylabel("layer (L)")
        fig.colorbar(im, ax=row.tolist(), shrink=0.85, label="Cohen's d (halluc. − clean)")
    _save(fig, "cohens_d_maps")


def fig_pca(results: list[dict]) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(5 * len(results), 7.5), squeeze=False)
    for j, res in enumerate(results):
        ax = axes[0, j]
        for r in PCA_REPS:
            cum = res["pca"][r]["cum"][:300]
            ax.plot(np.arange(1, len(cum) + 1), cum, color=COLOURS[r], lw=1.8, label=r)
        ax.axhline(0.9, ls="--", lw=0.8, color="black")
        ax.set_xscale("log")
        ax.set_xlabel("# principal components")
        ax.set_title(PRETTY[res["model"]], fontweight="bold")
        _style(ax)
        ax = axes[1, j]
        for r in PCA_REPS:
            ax.plot(PCA_KS, [res["pca"][r]["probe"][k] for k in PCA_KS], "o-", ms=3,
                    color=COLOURS[r], lw=1.8, label=r)
        ax.set_xscale("log")
        ax.set_xlabel("# principal components used by the probe")
        _style(ax)
    axes[0, 0].set_ylabel("cumulative explained variance\n(standardised features)")
    axes[1, 0].set_ylabel("probe AUROC on top-k PCs")
    axes[0, 0].legend(frameon=False)
    _save(fig, "pca_spectrum")


def fig_pca_scatter(results: list[dict]) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(10, 4.2 * len(results)), squeeze=False)
    rng = np.random.default_rng(SEED)
    for row, res in zip(axes, results):
        y = res["y"]
        sub = rng.choice(len(y), size=min(4000, len(y)), replace=False)
        for ax, rep in zip(row, ("H", "QKV")):
            sc = res["scatter"][rep]
            pc = sc["pc"][sub]
            for lab, colour, name in ((0, "#4C72B0", "clean"), (1, "#C44E52", "hallucinated")):
                m = y[sub] == lab
                ax.scatter(pc[m, 0], pc[m, 1], s=3, alpha=0.35, color=colour, label=name, rasterized=True)
            ax.set_title(f"{PRETTY[res['model']]} — {rep}\n"
                         f"PC1 {100 * sc['evr'][0]:.1f}% var (AUROC {sc['auc'][0]:.3f}), "
                         f"PC2 {100 * sc['evr'][1]:.1f}% var (AUROC {sc['auc'][1]:.3f})", fontsize=9)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            _style(ax)
    axes[0, 0].legend(frameon=False, markerscale=4)
    _save(fig, "pca_scatter")


def fig_cka(results: list[dict]) -> None:
    fig, axes = plt.subplots(len(results), 3, figsize=(12, 3.8 * len(results)), squeeze=False)
    for row, res in zip(axes, results):
        for ax, c in zip(row, "QKV"):
            im = ax.imshow(res["cka"][c], cmap="viridis", vmin=0, vmax=1, origin="lower")
            ax.set_title(f"{PRETTY[res['model']]} — CKA(H, {c})", fontsize=10)
            ax.set_xlabel(f"{c} layer")
            ax.set_ylabel("H layer (block output)")
        fig.colorbar(im, ax=row.tolist(), shrink=0.85, label="linear CKA")
    _save(fig, "cka")


def _write_csv(name: str, header: list[str], rows: list[list]) -> None:
    with open(OUT_TAB / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{c:.5f}" if isinstance(c, float) else c for c in r])


def write_tables(results: list[dict]) -> None:
    _write_csv("probes.csv",
               ["model", "rep", "n_features", "auroc", "auroc_sd", "C", "delta_vs_h", "delta_lo", "delta_hi"],
               [[res["model"], r, p["n_features"], p["auroc"], p["sd"], p["C"],
                 *(res["vs_h"].get(r, (0.0, 0.0, 0.0)))]
                for res in results for r, p in res["probe"].items()])
    _write_csv("unique_info.csv", ["model", "base", "added", "delta", "delta_lo", "delta_hi"],
               [[res["model"], a, b, *t] for res in results for (a, b), t in res["unique"].items()])
    _write_csv("clarity.csv",
               ["model", "view", "best_cell_auroc", *[f"n_cells_auroc_ge_{t}" for t in CELL_THRESHOLDS],
                "mean_abs_d", "max_abs_d", *[f"top{k}_cells_auroc" for k in TOPK_CELLS],
                "layers_won", "only_this_view_correct"],
               [[res["model"], v, float(c["cell_auc"].max()),
                 *[int((c["cell_auc"] >= t).sum()) for t in CELL_THRESHOLDS],
                 c["mean_abs_d"], c["max_abs_d"], *[c["topk"][k] for k in TOPK_CELLS],
                 res["layers_won"][v], res["only_right"][v]]
                for res in results for v, c in res["clarity"].items()])
    _write_csv("disagreement.csv",
               ["model", "view_a", "view_b", "a_right_b_wrong", "b_right_a_wrong", "mcnemar_p", "spearman"],
               [[res["model"], a, b, float(res["right_wrong"][i, j]), float(res["right_wrong"][j, i]),
                 float(res["mcnemar"][i, j]), float(res["rho"][i, j])]
                for res in results for i, a in enumerate(VIEWS) for j, b in enumerate(VIEWS) if i < j])
    _write_csv("layer_probe.csv", ["model", "view", "layer", "auroc"],
               [[res["model"], v, l + 1, float(a)] for res in results for v in VIEWS
                for l, a in enumerate(res["layers"][v])])
    _write_csv("pca.csv",
               ["model", "rep", "pcs_90", "participation_ratio", "raw_pc1_share", "top1pct_var_share",
                *[f"pca{k}_auroc" for k in PCA_KS]],
               [[res["model"], r, p["pcs_90"], p["participation_ratio"], p["raw_pc1_share"],
                 p["top1pct_var_share"], *[p["probe"][k] for k in PCA_KS]]
                for res in results for r, p in res["pca"].items()])
    _write_csv("cka_summary.csv", ["model", "channel", "cka_same_layer", "cka_input_layer", "cka_full_field"],
               [[res["model"], c, *s] for res in results for c, s in res["cka_summary"].items()])
    print("wrote", OUT_TAB)


def _md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _sig(t: tuple) -> bool:
    return t[1] > 0 or t[2] < 0


def write_report(results: list[dict]) -> None:
    pct = lambda v: f"{100 * v:.2f}"
    first = lambda res, i: PRETTY[res["model"]] if i == 0 else ""
    L = ["# Hidden states vs Q/K/V after D→M pooling (TriviaQA)", "",
         "**Claim under test:** Q, K and V are three linear views of the hidden state, and after D→M "
         "pooling some discriminative information is clearer in some views than in others (and than in H).", "",
         "Q/K/V at layer l are linear maps of the normalised H at layer l−1, so before pooling they cannot "
         "carry more label information than H (data-processing inequality). Pooling D→M is a fixed, lossy "
         "compression, and what it keeps depends on the basis it is applied in. The tests below therefore "
         "measure how much label signal survives pooling in each view, how few features are needed to read "
         "it out, and whether different views keep different parts of it.", "",
         "Setup: both fields are mean-pooled over D into M segments per layer, then averaged over generated "
         "tokens. Only examples whose two generations gave the same response and label are kept. Probes: "
         f"standardised features, logistic regression, {N_FOLDS}-fold stratified CV, C picked from "
         f"{list(C_GRID)} on an inner 20% holdout. Δ CIs: {N_BOOT}-sample paired bootstrap on out-of-fold "
         "scores. Cell = one (layer, segment) feature.", ""]

    L += [_md_table(["Model", "Aligned N", "Dropped (empty / divergent)", "Halluc. rate", "L×M",
                     "Segment width H / Q / K / V (dims)", "head_dim"],
                    [[PRETTY[r["model"]], r["n"], f"{r['n_empty']} / {r['n_divergent']}", f"{r['rate']:.3f}",
                      f"{r['L']}×{r['M']}", " / ".join(f"{r['widths'][c]:g}" for c in VIEWS),
                      r["widths"]["head_dim"]] for r in results]), ""]

    L += ["## Summary", ""]
    for res in results:
        c = res["clarity"]
        qkv_pairs = [(a, b) for a in "QKV" for b in "QKV" if a != b]
        n_sig = sum(_sig(res["unique"][p]) and res["unique"][p][0] > 0 for p in qkv_pairs)
        L += [f"**{PRETTY[res['model']]}**", "",
              "- Projected view vs H (Δ AUROC): " + ", ".join(f"{v} {_fmt_delta(res['vs_h'][v])}" for v in "QKV")
              + f"; QKV {_fmt_delta(res['vs_h']['QKV'])}.",
              "- Best single cell AUROC: " + ", ".join(f"{v} {c[v]['cell_auc'].max():.3f}" for v in VIEWS) + ".",
              f"- Cells with AUROC ≥ {CELL_THRESHOLDS[1]}: "
              + ", ".join(f"{v} {int((c[v]['cell_auc'] >= CELL_THRESHOLDS[1]).sum())}" for v in VIEWS) + ".",
              f"- Probe on the {TOPK_CELLS[1]} best cells: "
              + ", ".join(f"{v} {pct(c[v]['topk'][TOPK_CELLS[1]])}" for v in VIEWS) + ".",
              f"- Q/K/V add to each other: {n_sig}/6 ordered pairs give a significant gain; "
              f"H on top of QKV {_fmt_delta(res['h_adds'])}.",
              "- Layers where each view is the best probe: "
              + ", ".join(f"{v} {res['layers_won'][v]}" for v in VIEWS) + f" (of {res['L']}).",
              "- Examples only this view classifies correctly: "
              + ", ".join(f"{v} {100 * res['only_right'][v]:.2f}%" for v in VIEWS) + ".", ""]

    L += ["## 1. Does a projected view keep more label signal than H?", "",
          _md_table(["Model", "Rep", "#feat", "AUROC (CV)", "Δ vs H [95% CI]"],
                    [[first(res, i), r, p["n_features"], f"{pct(p['auroc'])} ± {pct(p['sd'])}",
                      "—" if r == "H" else _fmt_delta(res["vs_h"][r])]
                     for res in results for i, (r, p) in enumerate(res["probe"].items())]), "",
          "![probe](../figures/hidden_vs_qkv/probe_auroc.png)", ""]

    L += ["## 2. Is the signal clearer, i.e. readable from fewer features?", "",
          "Single-cell AUROC is the AUROC of one (layer, segment) feature on its own. The top-k probe "
          "picks the k most discriminative cells on the training fold only.", "",
          _md_table(["Model", "View", "best cell AUROC",
                     *[f"#cells ≥ {t}" for t in CELL_THRESHOLDS], "mean |d|", "max |d|",
                     *[f"top-{k} probe" for k in TOPK_CELLS]],
                    [[first(res, i), v, f"{c['cell_auc'].max():.3f}",
                      *[int((c["cell_auc"] >= t).sum()) for t in CELL_THRESHOLDS],
                      f"{c['mean_abs_d']:.3f}", f"{c['max_abs_d']:.3f}", *[pct(c["topk"][k]) for k in TOPK_CELLS]]
                     for res in results for i, (v, c) in enumerate(res["clarity"].items())]), "",
          "![clarity](../figures/hidden_vs_qkv/clarity.png)", "",
          "![d maps](../figures/hidden_vs_qkv/cohens_d_maps.png)", ""]

    L += ["## 3. Do different views keep different information?", "",
          "Gain from adding a view: Δ AUROC of the probe on base+added over the probe on base alone. A "
          "significant gain means the added view holds label signal that the base view's pooled features do "
          "not expose linearly.", "",
          _md_table(["Model", "Base", *[f"+{b}" for b in VIEWS]],
                    [[first(res, i), a, *["—" if a == b else _fmt_delta(res["unique"][(a, b)]) + ("*" if _sig(res["unique"][(a, b)]) else "")
                                          for b in VIEWS]]
                     for res in results for i, a in enumerate(VIEWS)]), "",
          "![unique](../figures/hidden_vs_qkv/unique_info.png)", "",
          "Per-example disagreement: each view's out-of-fold probe flags the same fraction of examples as "
          "hallucinated (the training prior), then we count examples one view gets right and the other gets "
          "wrong.", "",
          _md_table(["Model", "A", "B", "A right, B wrong", "B right, A wrong", "McNemar p", "Spearman(scores)"],
                    [[PRETTY[res["model"]] if (i, j) == (0, 1) else "", a, b,
                      f"{100 * res['right_wrong'][i, j]:.2f}%", f"{100 * res['right_wrong'][j, i]:.2f}%",
                      f"{res['mcnemar'][i, j]:.2g}", f"{res['rho'][i, j]:.3f}"]
                     for res in results for i, a in enumerate(VIEWS) for j, b in enumerate(VIEWS) if i < j]), "",
          "![disagreement](../figures/hidden_vs_qkv/disagreement.png)", "",
          "![layers](../figures/hidden_vs_qkv/layer_probe.png)", ""]

    L += ["## 4. PCA", "",
          "`PCs@90%` and participation ratio are on standardised features (higher = variance spread over "
          "more directions). `raw PC1` and `top-1% cells` are on raw features: how much raw variance one "
          "direction / the top 1% of cells hold. The probe columns use only the top-k PCs, fitted without "
          "labels inside each training fold.", "",
          _md_table(["Model", "Rep", "PCs@90%", "Participation ratio", "raw PC1", "top-1% cells",
                     *[f"k={k}" for k in PCA_KS]],
                    [[first(res, i), r, p["pcs_90"], f"{p['participation_ratio']:.1f}",
                      f"{p['raw_pc1_share']:.3f}", f"{p['top1pct_var_share']:.3f}",
                      *[pct(p["probe"][k]) for k in PCA_KS]]
                     for res in results for i, (r, p) in enumerate(res["pca"].items())]), "",
          "![pca spectrum](../figures/hidden_vs_qkv/pca_spectrum.png)", "",
          "![pca scatter](../figures/hidden_vs_qkv/pca_scatter.png)", ""]

    L += ["## 5. How similar are the pooled views to H? (linear CKA)", "",
          "`same layer` = mean CKA(H_l, X_l); `input layer` = mean CKA(H_{l−1}, X_l), the H that feeds the "
          "projection; `full field` = CKA over all layers at once. Cells are standardised first.", "",
          _md_table(["Model", "Channel", "same layer", "input layer", "full field"],
                    [[first(res, i), c, *[f"{x:.3f}" for x in s]]
                     for res in results for i, (c, s) in enumerate(res["cka_summary"].items())]), "",
          "![cka](../figures/hidden_vs_qkv/cka.png)", "",
          "Generated by `scripts/analysis/hidden_vs_qkv.py`."]
    OUT_REPORT.write_text("\n".join(L) + "\n")
    print("wrote", OUT_REPORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--refresh", action="store_true", help="rebuild the token-mean feature cache")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N examples (quick check)")
    args = ap.parse_args()

    for d in (OUT_FIG, OUT_TAB, OUT_REPORT.parent):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = [analyse(m, args.refresh, args.limit, rng) for m in args.models]

    write_tables(results)
    fig_probe(results)
    fig_clarity(results)
    fig_unique(results)
    fig_disagreement(results)
    fig_layers(results)
    fig_dmaps(results)
    fig_pca(results)
    fig_pca_scatter(results)
    fig_cka(results)
    write_report(results)


if __name__ == "__main__":
    main()
