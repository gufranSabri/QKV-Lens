from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.figures import style as fm
from src.config import load_config
from src.extract.qkv_hooks import get_decoder_layers, read_geometry

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act1"
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act1"
VIEW_COLOUR = {"Q": fm.BLUE, "K": fm.ORANGE, "V": fm.AQUA}

PROJ_ATTR = {"Q": "q_proj", "K": "k_proj", "V": "v_proj"}
NORM_ATTR_CANDIDATES = ("input_layernorm", "self_attn_layer_norm")


def norm_gain(layer) -> torch.Tensor | None:
    for attr in NORM_ATTR_CANDIDATES:
        norm = getattr(layer, attr, None)
        if norm is not None and hasattr(norm, "weight"):
            return norm.weight.detach()
    return None


def layer_spectrum(layer, view: str) -> dict:
    proj = getattr(layer.self_attn, PROJ_ATTR[view])
    W = proj.weight.detach().to(torch.float32)  # (D_out, D_in)
    g = norm_gain(layer)
    # Effective map the detector's row actually reads: W @ diag(g), since the
    # norm's learned per-channel gain is applied before the projection.
    W_eff = W * g.unsqueeze(0) if g is not None else W
    s = torch.linalg.svdvals(W_eff).cpu()
    rank = int((s > s[0] * 1e-6).sum())
    return {
        "d_in": W.shape[1], "d_out": W.shape[0], "rank": rank,
        "s_max": float(s[0]), "s_min": float(s[-1]),
        "cond": float(s[0] / s[-1].clamp_min(1e-12)),
        "s": s,
    }


def analyse(model_alias: str, cfg_path: str) -> dict:
    cfg = load_config(cfg_path)
    from src.extract.run_extraction import load_llm

    model, _ = load_llm(cfg)
    layers = get_decoder_layers(model)
    geom = read_geometry(model)

    per_view = {v: [layer_spectrum(layers[l], v) for l in range(geom.n_layers)] for v in "QKV"}
    del model
    torch.cuda.empty_cache()
    return {"model": model_alias, "L": geom.n_layers, "per_view": per_view}


def write_table(results: list[dict]) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "spectral_rank.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "view", "layer", "d_in", "d_out", "rank", "full_rank",
                    "cond_number", "s_max", "s_min"])
        for res in results:
            for v in "QKV":
                for l, sp in enumerate(res["per_view"][v]):
                    w.writerow([res["model"], v, l, sp["d_in"], sp["d_out"], sp["rank"],
                                sp["rank"] == min(sp["d_in"], sp["d_out"]),
                                f"{sp['cond']:.3f}", f"{sp['s_max']:.5f}", f"{sp['s_min']:.5f}"])
    print("wrote", path)


def write_summary_table(results: list[dict]) -> None:
    # One row per (model, view): the paper-ready version of spectral_rank.csv,
    # collapsed across layers.
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "spectral_rank_summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "view", "d_in", "pct_rank_retained_min",
                    "pct_rank_retained_median", "cond_number_median"])
        for res in results:
            for v in "QKV":
                layers = res["per_view"][v]
                pct = [100 * sp["rank"] / min(sp["d_in"], sp["d_out"]) for sp in layers]
                cond = [sp["cond"] for sp in layers]
                w.writerow([res["model"], v, layers[0]["d_in"],
                            f"{min(pct):.2f}", f"{float(np.median(pct)):.2f}",
                            f"{float(np.median(cond)):.3e}"])
    print("wrote", path)


def fig_spectrum(results: list[dict]) -> None:
    fm.apply()
    fig, axes = plt.subplots(1, len(results),
                              figsize=(fm.PANEL_W * len(results), fm.PANEL_H), squeeze=False)
    tags = "abc"
    for i, (ax, res) in enumerate(zip(axes[0], results)):
        for v in "QKV":
            mid = res["per_view"][v][res["L"] // 2]
            s = mid["s"].numpy()
            fm.halo_line(ax, range(1, len(s) + 1), s / s[0], VIEW_COLOUR[v], label=v)
        ax.set_yscale("log")
        ax.set_xlabel(f"Singular Value Rank ({res['model']})")
        if i == 0:
            ax.set_ylabel("Singular Value / Max")
        fm.legend(ax)
        fm.panel_tag(ax, f"({tags[i]})")
    paths = fm.save(fig, OUT_FIG, "spectrum")
    fm.write_how_to_read(
        OUT_FIG, "spectrum",
        how_to_read=(
            "Each panel is one LLM. Each line is one projection (Q, K, V) at "
            "that model's middle transformer layer: its singular values, "
            "sorted largest to smallest, normalised by the largest one, on a "
            "log y-axis. A line that stays near 1.0 across most of the "
            "x-axis and only drops at the very end means the projection "
            "preserves nearly all input directions; an early, steep drop "
            "would mean most directions are discarded."
        ),
        interpretation=(
            "; ".join(
                f"{res['model']}: " + ", ".join(
                    f"{v} min singular value {res['per_view'][v][res['L']//2]['s_min']:.2e}"
                    f" of max {res['per_view'][v][res['L']//2]['s_max']:.2f}"
                    for v in "QKV"
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
    args = ap.parse_args()

    results = []
    for pair in args.configs:
        alias, path = pair.split("=", 1)
        print(f"=== {alias} ===", flush=True)
        results.append(analyse(alias, path))

    write_table(results)
    write_summary_table(results)
    fig_spectrum(results)


if __name__ == "__main__":
    main()
