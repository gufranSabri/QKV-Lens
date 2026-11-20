from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures import style as fm
from scripts.tables.representation_ablation import best_combo, load_cells, pick

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_TAB = REPO_ROOT / "docs" / "tables" / "act5"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "act5"
PCA_CSV = REPO_ROOT / "docs" / "tables" / "act2" / "pca_capacity_control.csv"
PROBE_K = 200  # near-max PCA capacity in Act 2's sweep -- closest to an
               # uncapped, full-capacity linear-probe comparison.
MODELS = ("llama3.1_8b", "opt_6.7b", "qwen2.5_7b")
MODEL_PRETTY = {
    "llama3.1_8b": "LLaMA-3.1-8B", "opt_6.7b": "OPT-6.7B", "qwen2.5_7b": "Qwen2.5-7B",
}


def load_probe_auroc() -> dict[str, dict[str, float]]:
    # {model: {"QKV": auroc, "H": auroc}} at k=200 from Act 2's PCA-capacity
    # control -- the closest existing linear-probe number to an uncapped
    # full-capacity comparison (narrative.md's Act 2 section).
    if not PCA_CSV.exists():
        raise FileNotFoundError(f"{PCA_CSV} missing -- run scripts/analysis/act2/run_act2.sh first")
    out: dict[str, dict[str, float]] = {}
    with open(PCA_CSV) as fh:
        for row in csv.DictReader(fh):
            if int(row["k"]) != PROBE_K:
                continue
            out.setdefault(row["model"], {})[row["view"]] = float(row["auroc_mean"])
    return out


def load_detector_auroc() -> dict[str, dict[str, float]]:
    # {model: {"QKV": test_auroc, "hs": test_auroc, "hsn": test_auroc}} --
    # the real conv+BiLSTM+attention detector's picked-on-val, reported-on-
    # test cells from scripts/experiments/representation_ablation.sh's runs.
    # "hs" (raw hidden states) is the field matched to the probe side below,
    # since Act 2's PCA-capacity control only has raw H, not HN, at k=200.
    cells = load_cells(REPO_ROOT / "runs")
    best, _, _ = best_combo(cells)
    out: dict[str, dict[str, float]] = {}
    for m in MODELS:
        out[m] = {}
        for field in ("QKV", "hs", "hsn"):
            got = pick(cells, m, field, best)
            if got is None:
                raise SystemExit(f"no real-detector cell for {m}/{field} -- is runs/ populated?")
            _, (_, test_auroc) = got
            out[m][field] = test_auroc
    return out


def assemble(probe: dict, detector: dict) -> list[dict]:
    results = []
    for m in MODELS:
        if m not in probe:
            raise SystemExit(f"no Act 2 PCA-control probe cell for {m} at k={PROBE_K}")
        p_qkv, p_h = 100 * probe[m]["QKV"], 100 * probe[m]["H"]
        d_qkv, d_h = 100 * detector[m]["QKV"], 100 * detector[m]["hs"]
        results.append({
            "model": m,
            "probe_qkv": p_qkv, "probe_h": p_h, "probe_gap": p_qkv - p_h,
            "detector_qkv": d_qkv, "detector_h": d_h, "detector_gap": d_qkv - d_h,
        })
        print(f"  {m}: probe QKV {p_qkv:.2f} H {p_h:.2f} gap {p_qkv - p_h:+.2f}  |  "
              f"detector QKV {d_qkv:.2f} hs {d_h:.2f} gap {d_qkv - d_h:+.2f}", flush=True)
    return results


def write_table(results: list[dict]) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    path = OUT_TAB / "probe_vs_detector.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "setting", "qkv_auroc", "h_auroc", "gap"])
        for res in results:
            w.writerow([res["model"], "probe", f"{res['probe_qkv']:.3f}",
                        f"{res['probe_h']:.3f}", f"{res['probe_gap']:.3f}"])
            w.writerow([res["model"], "detector", f"{res['detector_qkv']:.3f}",
                        f"{res['detector_h']:.3f}", f"{res['detector_gap']:.3f}"])
    print("wrote", path)


def load_table() -> list[dict]:
    by_model: dict[str, dict] = {}
    with open(OUT_TAB / "probe_vs_detector.csv") as fh:
        for row in csv.DictReader(fh):
            m = row["model"]
            entry = by_model.setdefault(m, {"model": m})
            prefix = "probe" if row["setting"] == "probe" else "detector"
            entry[f"{prefix}_qkv"] = float(row["qkv_auroc"])
            entry[f"{prefix}_h"] = float(row["h_auroc"])
            entry[f"{prefix}_gap"] = float(row["gap"])
    return [by_model[m] for m in MODELS if m in by_model]


def fig_dumbbell(results: list[dict]) -> None:
    # One row per (model, setting): H-to-QKV dumbbell, Probe and Detector
    # rows interleaved per model so the shrinkage reads directly top-to-bottom.
    fm.apply()
    fig, ax = plt.subplots(figsize=(fm.PANEL_W * 1.5, fm.PANEL_H * 1.3))

    rows = []
    for res in results:
        rows.append((f"{MODEL_PRETTY[res['model']]} — Detector", res["detector_h"], res["detector_qkv"]))
        rows.append((f"{MODEL_PRETTY[res['model']]} — Probe", res["probe_h"], res["probe_qkv"]))

    ys = np.arange(len(rows))
    for y, (_, h, qkv) in zip(ys, rows):
        ax.plot([h, qkv], [y, y], color=fm.GRAY, lw=1.2, zorder=1,
                path_effects=fm.HALO, solid_capstyle="round")
    ax.scatter([r[1] for r in rows], ys, color=fm.GRAY, s=50, zorder=2,
               edgecolor="white", linewidth=0.8, label="H")
    ax.scatter([r[2] for r in rows], ys, color=fm.VIOLET, s=50, zorder=2,
               edgecolor="white", linewidth=0.8, label="QKV")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    for boundary in np.arange(1.5, len(rows) - 1, 2):
        ax.axhline(boundary, color=fm.INK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlabel("AUROC")
    fm.legend(ax, loc="lower right")
    paths = fm.save(fig, OUT_FIG, "probe_vs_detector_dumbbell")

    fm.write_how_to_read(
        OUT_FIG, "probe_vs_detector_dumbbell",
        how_to_read=(
            "One row per model x setting, dashed lines separating models. "
            "'Detector' rows are the real conv+BiLSTM+attention detector's "
            "best picked-on-val cell (test AUROC) for QKV and for raw hidden "
            "states (hs), from scripts/experiments/representation_ablation.sh's "
            "TriviaQA runs. "
            "'Probe' rows are a linear probe on the same pooled fields at "
            "matched near-max PCA capacity (k=200, Act 2's capacity control), "
            "gray H to violet QKV. Comparing each model's two rows shows "
            "whether the real detector preserves the probe-diagnosed QKV-over-H "
            "gap, shrinks it, or erases it."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY[res['model']]}: probe gap {res['probe_gap']:+.2f}, "
                f"detector gap {res['detector_gap']:+.2f}"
                for res in results
            )
        ),
    )
    print("wrote", paths[0])


def fig_gap_bars(results: list[dict]) -> None:
    # Grouped horizontal bars of the gap itself (QKV - H), Probe vs Detector,
    # one pair per model -- directly answers "does the gap shrink."
    fm.apply()
    fig, ax = plt.subplots(figsize=(fm.PANEL_W * 1.3, fm.PANEL_H))

    ys = np.arange(len(results))
    bar_h = 0.32
    probe_vals = [res["probe_gap"] for res in results]
    det_vals = [res["detector_gap"] for res in results]

    bars_p = ax.barh(ys + bar_h / 2 + 0.02, probe_vals, height=bar_h,
                      color=fm.VIOLET, edgecolor=fm.INK, linewidth=0.6,
                      label="Linear Probe", zorder=2)
    bars_d = ax.barh(ys - bar_h / 2 - 0.02, det_vals, height=bar_h,
                      color=fm.BLUE, edgecolor=fm.INK, linewidth=0.6,
                      label="Real Detector", zorder=2)
    for bars in (bars_p, bars_d):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.15, bar.get_y() + bar.get_height() / 2, f"{w:.1f}",
                    va="center", ha="left", fontsize=6.5, color=fm.INK, zorder=3)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([MODEL_PRETTY[res["model"]] for res in results])
    ax.set_ylim(len(results) - 0.5, -0.5)
    ax.set_xlim(0, max(probe_vals + det_vals) * 1.22)  # headroom for value labels + legend
    ax.axvline(0, color=fm.INK, linewidth=0.8, zorder=1)
    ax.set_xlabel("AUROC Gap (QKV − H)")
    ax.xaxis.grid(True, color=fm.GRAY, linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    fm.legend(ax, loc="upper right")
    paths = fm.save(fig, OUT_FIG, "probe_vs_detector_gap")

    fm.write_how_to_read(
        OUT_FIG, "probe_vs_detector_gap",
        how_to_read=(
            "One row per model, two bars: the QKV-minus-H AUROC gap measured "
            "by a linear probe (violet, Act 2's PCA-capacity control at "
            "k=200) vs. the same gap measured by the real conv+BiLSTM+"
            "attention detector (blue, best picked-on-val cell, test AUROC). "
            "Both bars positive means QKV beats H in both settings; a "
            "shorter blue bar than violet means the real detector narrows "
            "the gap the linear probe diagnoses."
        ),
        interpretation=(
            "; ".join(
                f"{MODEL_PRETTY[res['model']]}: probe {res['probe_gap']:+.2f} "
                f"-> detector {res['detector_gap']:+.2f} "
                f"({'shrinks' if abs(res['detector_gap']) < abs(res['probe_gap']) else 'grows'})"
                for res in results
            )
        ),
    )
    print("wrote", paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("run", "plot"), default="run",
                     help="run: read runs/ + Act 2's CSV, write table + figures. "
                          "plot: figures only, reads the existing act5 CSV.")
    args = ap.parse_args()

    if args.stage == "plot":
        table_path = OUT_TAB / "probe_vs_detector.csv"
        if not table_path.exists():
            raise SystemExit(f"{table_path} missing -- run --stage run first")
        results = load_table()
        fig_dumbbell(results)
        fig_gap_bars(results)
        return

    probe = load_probe_auroc()
    detector = load_detector_auroc()
    results = assemble(probe, detector)
    write_table(results)
    fig_dumbbell(results)
    fig_gap_bars(results)


if __name__ == "__main__":
    main()
