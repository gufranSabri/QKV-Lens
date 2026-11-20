from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.figures import style as fm
from scripts.analysis.act2.pca_capacity_control import balanced_indices
from src.config import load_config
from src.extract.qkv_hooks import capture_all, left_pad_batch
from src.extract.run_extraction import load_llm, parse_meta
from src.utils.progress import progress

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act1"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act1"
OUT_CACHE = REPO_ROOT / "docs" / "act1_cache"
N_FOLDS = 5
N_BOOT = 1000
SEED = 0
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
CAPTURED_VIEWS = ("Q", "K", "V", "H")   # stored per example
PROBE_VIEWS = ("Q", "K", "V", "QKV", "H")   # QKV is Q+K+V concatenated at probe time
PLOT_VIEWS = ("Q", "K", "V", "H")   # QKV kept in the CSV but left off the figure
VIEW_COLOUR = {"Q": fm.BLUE, "K": fm.ORANGE, "V": fm.AQUA,
               "QKV": fm.VIOLET, "H": fm.GRAY}
MODEL_PRETTY = {
    "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}


def load_labeled_examples(cfg, n: int, seed: int = SEED) -> list[dict]:
    # Reads prompt/gold/label straight off the already-extracted, already-
    # labeled `default` corpus -- no dataset re-download, no judge call, and
    # independent of any trained detector under runs/. Selection goes through
    # the SAME balanced_indices() as Acts 2-4 (manifest order, n//2 per
    # class), so every act draws from the same example set at the same seed
    # instead of relying on two independently-written selectors agreeing.
    root = cfg.example_dir()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} missing -- run `detector.py extract` + `label` first")

    with open(manifest_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    labels = [int(r["label"]) for r in records]
    idx = balanced_indices(labels, n, seed=seed)

    out = []
    for i in idx:
        r = records[i]
        meta = parse_meta(root / r["dir"] / "meta.txt")
        out.append({"idx": r["idx"], "prompt": meta["prompt"], "label": r["label"]})
    return out


def _part_path(model_alias: str) -> Path:
    return OUT_CACHE / f"{model_alias}.partial.jsonl"


def _index_path(model_alias: str) -> Path:
    # Tiny sidecar of just the completed idx values. .partial.jsonl carries
    # full per-layer float arrays for 5 views and gets into the GB range for
    # only a few hundred examples -- re-parsing all of that just to answer
    # "which idx are already done" made every resumed run grind for minutes
    # before doing any real work. This file is the cheap thing to read instead.
    return OUT_CACHE / f"{model_alias}.partial.idx"


def _cache_path(model_alias: str) -> Path:
    return OUT_CACHE / f"{model_alias}.npz"


def _done_indices(model_alias: str) -> set[int]:
    index_path = _index_path(model_alias)
    if index_path.exists():
        with open(index_path) as f:
            return {int(line) for line in f if line.strip()}

    # No sidecar yet (first run after this change, on an existing .partial.jsonl):
    # rebuild it once from the big file, then every later call is cheap again.
    part_path = _part_path(model_alias)
    if not part_path.exists():
        return set()
    done: set[int] = set()
    with open(part_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["idx"])
            except json.JSONDecodeError:
                continue
    with open(index_path, "w") as f:
        f.writelines(f"{i}\n" for i in sorted(done))
    return done


@torch.no_grad()
def capture_batch(model, tokenizer, batch: list[dict], max_new_tokens: int) -> list[dict | None]:
    device = next(model.parameters()).device
    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    prompt_ids = [tokenizer(ex["prompt"], return_tensors="pt")["input_ids"][0].to(device) for ex in batch]
    input_ids, attention_mask = left_pad_batch(prompt_ids, pad_id, device)
    batch_out = capture_all(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=stop_ids,
        attention_mask=attention_mask, capture_hidden_states=True,
    )

    out = []
    for ex, (activations, gen_ids) in zip(batch, batch_out):
        if gen_ids.numel() == 0:
            out.append(None)
            continue
        out.append({
            "idx": ex["idx"], "label": ex["label"],
            **{v: activations[v].mean(0).tolist() for v in CAPTURED_VIEWS},  # (L, D) token-mean
        })
    return out


def run_capture(model_alias: str, cfg_path: str, n_examples: int, max_new_tokens: int,
                 batch_size: int, refresh: bool) -> bool:
    # Returns whether anything was actually captured this call.
    # Checkpoints one JSON line per example to a .partial.jsonl as soon as its
    # batch finishes, so a crash/preemption loses at most one in-flight batch,
    # never the whole run. Re-running the same command resumes from there.
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    part_path = _part_path(model_alias)
    index_path = _index_path(model_alias)
    if refresh:
        for p in (part_path, index_path):
            if p.exists():
                p.unlink()

    cfg = load_config(cfg_path)
    examples = load_labeled_examples(cfg, n_examples)
    done = _done_indices(model_alias)
    todo = [ex for ex in examples if ex["idx"] not in done]
    if not todo:
        print(f"{model_alias}: all {len(examples)} examples already captured", flush=True)
        return False

    print(f"{model_alias}: {len(todo)}/{len(examples)} example(s) left to capture", flush=True)
    model, tokenizer = load_llm(cfg)
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    try:
        with open(part_path, "a") as fh, open(index_path, "a") as idx_fh:
            for batch in progress(batches, desc=f"act1 capture {model_alias}", ncols=100):
                for row in capture_batch(model, tokenizer, batch, max_new_tokens):
                    if row is not None:
                        fh.write(json.dumps(row) + "\n")
                        idx_fh.write(f"{row['idx']}\n")
                fh.flush()
                idx_fh.flush()
    finally:
        del model
        torch.cuda.empty_cache()
    return True


def finalize_cache(model_alias: str, force: bool = False) -> dict | None:
    # Folds the resumable .partial.jsonl into the one .npz the probe/plot
    # stage reads, so that stage never needs the GPU or the partial file.
    # Parsing .partial.jsonl means re-reading every captured example's full
    # per-layer float arrays, which gets into the GB range fast -- skip that
    # entirely when the cache is already at least as new as the partial file.
    part_path = _part_path(model_alias)
    if not part_path.exists():
        raise FileNotFoundError(f"{part_path} missing -- run with --capture first")
    cache = _cache_path(model_alias)
    if not force and cache.exists() and cache.stat().st_mtime >= part_path.stat().st_mtime:
        print(f"{cache} already up to date with {part_path} -- skipping", flush=True)
        return None
    rows_by_idx: dict[int, dict] = {}
    with open(part_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from a hard kill mid-write
            rows_by_idx[row["idx"]] = row  # last write for a given idx wins
    rows = sorted(rows_by_idx.values(), key=lambda r: r["idx"])

    data = {v: np.array([r[v] for r in rows], dtype=np.float32) for v in CAPTURED_VIEWS}
    data["y"] = np.array([r["label"] for r in rows], dtype=np.int64)

    tmp = cache.with_name(cache.name + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, **data)
    tmp.rename(cache)
    print(f"wrote {cache} ({len(rows)} examples)")
    return data


def _lr(C: float) -> LogisticRegression:
    # liblinear (coordinate descent) converges far faster than lbfgs here:
    # N=600 << D=4096 for Q/H, a severely underdetermined L2 fit.
    return LogisticRegression(C=C, max_iter=3000, solver="liblinear")


def cv_auroc(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    # C is picked on an INNER holdout of the training fold, never on the
    # training fold it was fit on and never on the test fold -- scoring C by
    # training-set AUROC would favour less regularisation (and so a better
    # training-set fit) as D grows, which would quietly favour QKV's larger
    # concatenated dimensionality over Q/K/V/H with no real signal behind it.
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


def bootstrap_delta(a_auroc: np.ndarray, b_auroc: np.ndarray, rng) -> tuple[float, float, float]:
    # Paired bootstrap over the fold-level AUROCs (cheap stand-in for a full
    # per-example bootstrap; folds are already the unit of variance here).
    diffs = a_auroc - b_auroc
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def view_features(data: dict, view: str, l: int) -> np.ndarray:
    if view == "QKV":
        return np.concatenate([data[v][:, l, :] for v in "QKV"], axis=1)
    return data[view][:, l, :]


def probe(model_alias: str, rng) -> dict:
    # Reads ONLY the finalized .npz cache -- no model load, no GPU. Safe to
    # re-run as often as needed while tuning the probe or the plots.
    data = dict(np.load(_cache_path(model_alias)))
    y = data["y"]
    L = data["H"].shape[1]
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    per_layer = {}
    for l in range(L):
        row = {v: cv_auroc(view_features(data, v, l), y, folds) for v in PROBE_VIEWS}
        per_layer[l] = row
        delta_q = bootstrap_delta(row["Q"], row["H"], rng)
        print(f"  layer {l:2d}  " + "  ".join(f"{v} {row[v].mean():.3f}" for v in PROBE_VIEWS)
              + f"  ΔQ-H {100*delta_q[0]:+.2f} [{100*delta_q[1]:+.2f},{100*delta_q[2]:+.2f}]", flush=True)

    return {"model": model_alias, "n": len(y), "L": L, "per_layer": per_layer}


def write_table(results: list[dict], rng) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "probe_parity.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "layer", "view", "auroc_mean", "auroc_sd",
                    "delta_vs_H", "delta_lo", "delta_hi"])
        for res in results:
            for l, row in res["per_layer"].items():
                for v in PROBE_VIEWS:
                    delta = (0.0, 0.0, 0.0) if v == "H" else bootstrap_delta(row[v], row["H"], rng)
                    w.writerow([res["model"], res["n"], l, v,
                                f"{row[v].mean():.5f}", f"{row[v].std():.5f}",
                                f"{delta[0]:.5f}", f"{delta[1]:.5f}", f"{delta[2]:.5f}"])
    print("wrote", path)


def _tick_step(n_layers: int, step: int = 5) -> list[int]:
    return list(range(step, n_layers + 1, step))


def load_table(path: Path) -> list[dict]:
    # {model -> {layer -> {view -> mean_auroc}}}, read straight from the CSV
    # so the plot can be redrawn without rerunning capture/probe.
    by_model: dict[str, dict[int, dict[str, float]]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            m, l, v = row["model"], int(row["layer"]), row["view"]
            by_model.setdefault(m, {}).setdefault(l, {})[v] = float(row["auroc_mean"])
    results = []
    for model, per_layer in by_model.items():
        results.append({"model": model, "L": max(per_layer) + 1, "per_layer": per_layer})
    return results


def fig_layer_auroc(results: list[dict]) -> None:
    # Accepts either probe()'s {layer: {view: np.ndarray-of-fold-aucs}} or
    # load_table()'s {layer: {view: float}} -- both expose .mean()-able or
    # plain-float entries, so _mean() below normalises the two. AUROC is
    # reported on a 0-100 scale throughout this figure, not 0-1.
    def _mean(v):
        return 100.0 * (float(v.mean()) if hasattr(v, "mean") else float(v))

    fm.apply()
    tags = "abc"
    fig, axes = plt.subplots(1, len(results),
                              figsize=(fm.PANEL_W * len(results), fm.PANEL_H), squeeze=False)
    for i, (ax, res) in enumerate(zip(axes[0], results)):
        layers = sorted(res["per_layer"])
        all_means = []
        for v in PLOT_VIEWS:
            means = [_mean(res["per_layer"][l][v]) for l in layers]
            all_means += means
            fm.halo_line(ax, [l + 1 for l in layers], means, VIEW_COLOUR[v], label=v)
        fm.layer_axis(ax, res["L"])
        ax.set_xticks(_tick_step(res["L"]))
        ax.set_ylim(top=max(all_means) + 10)
        if i == 0:
            ax.set_ylabel("AUROC")
        fm.legend(ax)
        fm.panel_tag(ax, f"({tags[i]}) {MODEL_PRETTY.get(res['model'], res['model'])}")
    paths = fm.save(fig, OUT_FIG, "probe_parity_by_layer")

    def best_layer(res, v):
        l = max(res["per_layer"], key=lambda l: _mean(res["per_layer"][l][v]))
        return l + 1, _mean(res["per_layer"][l][v])

    fm.write_how_to_read(
        OUT_FIG, "probe_parity_by_layer",
        how_to_read=(
            "Each panel is one LLM; x-axis is transformer layer (1-indexed, "
            "ticked every 5). Each line is a linear probe's mean 5-fold CV "
            "AUROC at that layer, trained on that layer's raw, unpooled "
            "per-token features: Q, K, V (the pre-attention projections) and "
            "H (the raw residual stream). QKV is computed and kept in "
            "probe_parity.csv but left off this figure to reduce clutter."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY.get(res['model'], res['model'])}: peak Q "
                f"{best_layer(res, 'Q')[1]:.1f} at layer {best_layer(res, 'Q')[0]}, "
                f"peak H {best_layer(res, 'H')[1]:.1f} at layer {best_layer(res, 'H')[0]}"
                for res in results
            )
        ),
    )
    print("wrote", paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                     help="pairs of alias=path, e.g. llama3.1_8b=configs/triviaqa/llama3.1_8b.yaml")
    ap.add_argument("--n-examples", type=int, default=600)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stage", choices=("capture", "probe", "plot", "all"), default="all",
                     help="capture: GPU generation, resumable. probe: CV + tables, cache-only. "
                          "plot: figure only, reads the existing probe_parity.csv, no cache/GPU needed.")
    ap.add_argument("--refresh", action="store_true", help="discard the partial capture and start over")
    args = ap.parse_args()

    pairs = [p.split("=", 1) for p in args.configs]

    if args.stage in ("capture", "all"):
        for alias, path in pairs:
            print(f"=== capture {alias} ===", flush=True)
            captured = run_capture(alias, path, args.n_examples, args.max_new_tokens,
                                    args.batch_size, args.refresh)
            finalize_cache(alias, force=captured or args.refresh)

    if args.stage in ("probe", "all"):
        table_path = OUT_TAB / "probe_parity.csv"
        wanted = [alias for alias, _ in pairs]
        table_fresh = (
            not args.refresh
            and table_path.exists()
            and all(_cache_path(a).exists() for a in wanted)
            and table_path.stat().st_mtime >= max(_cache_path(a).stat().st_mtime for a in wanted)
        )
        if table_fresh:
            existing = {r["model"] for r in load_table(table_path)}
            table_fresh = set(wanted) <= existing

        if table_fresh:
            print(f"{table_path} is already up to date for {wanted} -- skipping retrain "
                  f"(pass --refresh, or delete the cache/table, to force it)", flush=True)
            fig_layer_auroc([r for r in load_table(table_path) if r["model"] in wanted])
        else:
            rng = np.random.default_rng(SEED)
            results = []
            for alias, _ in pairs:
                print(f"=== probe {alias} ===", flush=True)
                results.append(probe(alias, rng))
            write_table(results, rng)
            fig_layer_auroc(results)

    if args.stage == "plot":
        table_path = OUT_TAB / "probe_parity.csv"
        if not table_path.exists():
            raise SystemExit(f"{table_path} missing -- run --stage probe (or all) first")
        wanted = [alias for alias, _ in pairs]
        results = [r for r in load_table(table_path) if r["model"] in wanted]
        missing = set(wanted) - {r["model"] for r in results}
        if missing:
            raise SystemExit(f"{table_path} has no rows for: {sorted(missing)}")
        fig_layer_auroc(results)


if __name__ == "__main__":
    main()
