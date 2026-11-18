from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.extract.qkv_hooks import capture_all, left_pad_batch, read_geometry
from src.extract.run_extraction import load_llm, parse_meta

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act1"
OUT_CACHE = REPO_ROOT / "docs" / "act1_cache"
N_FOLDS = 5
N_BOOT = 1000
SEED = 0
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)


def load_labeled_examples(cfg, n: int) -> list[dict]:
    # Reads prompt/gold/label straight off the already-extracted, already-
    # labeled `default` corpus -- no dataset re-download, no judge call, and
    # independent of any trained detector under runs/.
    root = cfg.example_dir()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} missing -- run `detector.py extract` + `label` first")

    with open(manifest_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if r["label"] in (0, 1)]
    records.sort(key=lambda r: r["idx"])
    records = records[:n]

    out = []
    for r in records:
        meta = parse_meta(root / r["dir"] / "meta.txt")
        out.append({"idx": r["idx"], "prompt": meta["prompt"], "label": r["label"]})
    return out


@torch.no_grad()
def collect_unpooled(model, tokenizer, examples: list[dict], max_new_tokens: int) -> dict:
    device = next(model.parameters()).device
    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    per_view = {"Q": [], "K": [], "V": [], "HN": []}
    labels = []
    for ex in examples:
        ids = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].to(device)
        input_ids, attention_mask = left_pad_batch([ids[0]], pad_id, device)
        (activations, gen_ids), = capture_all(
            model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=stop_ids,
            attention_mask=attention_mask, capture_normed_hidden_states=True,
        )
        if gen_ids.numel() == 0:
            continue
        for v in per_view:
            per_view[v].append(activations[v].mean(0))  # (L, D) token-mean, matches existing convention
        labels.append(ex["label"])

    return {v: torch.stack(rows).numpy() for v, rows in per_view.items()} | {
        "y": np.array(labels, dtype=np.int64)
    }


def _lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000)


def cv_auroc(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    aucs = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        best = max(C_GRID, key=lambda C: roc_auc_score(
            y[tr], _lr(C).fit(Xtr, y[tr]).decision_function(Xtr)))
        s = _lr(best).fit(Xtr, y[tr]).decision_function(Xte)
        aucs.append(roc_auc_score(y[te], s))
    return np.array(aucs)


def bootstrap_delta(a_auroc: np.ndarray, b_auroc: np.ndarray, rng) -> tuple[float, float, float]:
    # Paired bootstrap over the fold-level AUROCs (cheap stand-in for a full
    # per-example bootstrap; folds are already the unit of variance here).
    diffs = a_auroc - b_auroc
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def analyse(model_alias: str, cfg_path: str, n_examples: int, max_new_tokens: int, refresh: bool, rng) -> dict:
    cfg = load_config(cfg_path)
    cache = OUT_CACHE / f"{model_alias}.npz"
    if cache.exists() and not refresh:
        data = dict(np.load(cache))
    else:
        examples = load_labeled_examples(cfg, n_examples)
        model, tokenizer = load_llm(cfg)
        data = collect_unpooled(model, tokenizer, examples, max_new_tokens)
        del model
        torch.cuda.empty_cache()
        OUT_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(fh, **data)
        tmp.rename(cache)

    y = data["y"]
    L = data["HN"].shape[1]
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    per_layer = {}
    for l in range(L):
        row = {}
        for v in ("Q", "K", "V", "HN"):
            row[v] = cv_auroc(data[v][:, l, :], y, folds)
        per_layer[l] = row
        delta_q = bootstrap_delta(row["Q"], row["HN"], rng)
        print(f"  layer {l:2d}  HN {row['HN'].mean():.3f}  Q {row['Q'].mean():.3f}  "
              f"K {row['K'].mean():.3f}  V {row['V'].mean():.3f}  "
              f"ΔQ-HN {100*delta_q[0]:+.2f} [{100*delta_q[1]:+.2f},{100*delta_q[2]:+.2f}]", flush=True)

    return {"model": model_alias, "n": len(y), "L": L, "per_layer": per_layer}


def write_table(results: list[dict], rng) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "probe_parity.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "layer", "view", "auroc_mean", "auroc_sd",
                    "delta_vs_HN", "delta_lo", "delta_hi"])
        for res in results:
            for l, row in res["per_layer"].items():
                for v in ("Q", "K", "V", "HN"):
                    delta = (0.0, 0.0, 0.0) if v == "HN" else bootstrap_delta(row[v], row["HN"], rng)
                    w.writerow([res["model"], res["n"], l, v,
                                f"{row[v].mean():.5f}", f"{row[v].std():.5f}",
                                f"{delta[0]:.5f}", f"{delta[1]:.5f}", f"{delta[2]:.5f}"])
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                     help="pairs of alias=path, e.g. llama3.1_8b=configs/triviaqa/llama3.1_8b.yaml")
    ap.add_argument("--n-examples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    results = []
    for pair in args.configs:
        alias, path = pair.split("=", 1)
        print(f"=== {alias} ===", flush=True)
        results.append(analyse(alias, path, args.n_examples, args.max_new_tokens, args.refresh, rng))

    write_table(results, rng)


if __name__ == "__main__":
    main()
