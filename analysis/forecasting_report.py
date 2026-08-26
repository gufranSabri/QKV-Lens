"""Figures and report.md for the prefix-forecasting analysis.

Five figures, all pooled over every evaluated example rather than split by
response length. Length-splitting was the original plan, but on this corpus
max_new_tokens=64 truncates almost everything: 2481 of 2490 held-out TriviaQA
responses are exactly 64 tokens, and the remaining 7 length buckets hold 1-2
examples each. One figure per length would therefore have produced 8 files, 7 of
them statistically empty. Pooling on PREFIX FRACTION keeps every example in
every panel, so each point is backed by the full n.

  1_detection_counts.png   the signed-count bars, per class (the original idea)
  2_earliest_detection.png when detection first becomes correct-and-stable
  3_auroc_curve.png        AUROC / accuracy as a function of prefix length
  4_prob_trajectory.png    mean p(hallucinated) per class, with spread
  5_raster.png             per-example correctness, one row per response
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics

from analysis.forecasting import FRACTIONS, THRESHOLD, Trajectory

logger = get_logger(__name__)

#: Colour-blind-safe pair used consistently for the two directions in every
#: figure: green = the detector was right, orange = it was wrong.
RIGHT = "#2A9D5C"
WRONG = "#D97A28"
#: Per-class accents for the trajectory plot.
HALLU = "#B5416B"
CLEAN = "#3A6EA5"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _pct_labels() -> list[str]:
    return [f"{int(round(f * 100))}%" for f in FRACTIONS]


def signed_counts(trajs: list[Trajectory], label: int) -> tuple[np.ndarray, np.ndarray]:
    """(right, wrong) counts at each prefix fraction, for one true class.

    This is the original brief's panel, generalised from "one figure per token
    count" to a fraction axis. For hallucinated examples (label=1) `right` means
    the detector said hallucination; for clean ones it means it said clean. The
    caller plots `right` up and `wrong` down.
    """
    sel = [t for t in trajs if t.label == label]
    right = np.zeros(len(FRACTIONS), dtype=int)
    wrong = np.zeros(len(FRACTIONS), dtype=int)
    for t in sel:
        hit = t.fraction_of(t.correct)
        right += hit.astype(int)
        wrong += (~hit).astype(int)
    return right, wrong


def fig_detection_counts(trajs: list[Trajectory], dest: Path) -> None:
    """The signed-count bars: correct above the axis, incorrect below."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(FRACTIONS))

    for ax, label, title in (
        (axes[0], 1, "Hallucinated responses"),
        (axes[1], 0, "Clean responses"),
    ):
        right, wrong = signed_counts(trajs, label)
        n = right[0] + wrong[0]
        ax.bar(x, right, color=RIGHT, label="detected correctly")
        ax.bar(x, -wrong, color=WRONG, label="missed")
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(_pct_labels(), rotation=45, fontsize=8)
        ax.set_xlabel("fraction of response seen")
        ax.set_title(f"{title}  (n={n})", fontsize=11)
        ax.grid(axis="y", alpha=0.25, ls=":")
        # Annotate the crossover: the first fraction where right > wrong.
        crossed = np.argmax(right > wrong) if (right > wrong).any() else None
        if crossed is not None and (right > wrong).any():
            ax.axvline(crossed, color="black", ls="--", lw=0.8, alpha=0.6)
            ax.text(
                crossed, ax.get_ylim()[1] * 0.92,
                f" majority correct\n from {_pct_labels()[crossed]}",
                fontsize=7.5, va="top",
            )

    axes[0].set_ylabel("examples  (correct up / wrong down)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Detector correctness vs. how much of the response it has seen",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_earliest_detection(trajs: list[Trajectory], dest: Path) -> None:
    """Distribution of the earliest prefix that is correct AND stays correct."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

    for ax, label, title, colour in (
        (axes[0], 1, "Hallucinated", HALLU),
        (axes[1], 0, "Clean", CLEAN),
    ):
        sel = [t for t in trajs if t.label == label]
        fracs, never = [], 0
        for t in sel:
            first = t.first_stable_correct()
            if first is None:
                never += 1
            else:
                fracs.append(first / t.n_tokens)

        if fracs:
            ax.hist(fracs, bins=20, range=(0, 1), color=colour, alpha=0.85)
            med = float(np.median(fracs))
            ax.axvline(med, color="black", ls="--", lw=1.2)
            ax.text(
                med, ax.get_ylim()[1] * 0.95, f" median {med:.0%}",
                fontsize=9, va="top",
            )
        ax.set_xlabel("fraction of response before the call locks in")
        ax.set_title(
            f"{title}  (n={len(sel)}, never correct at the end: {never})",
            fontsize=11,
        )
        ax.grid(axis="y", alpha=0.25, ls=":")

    axes[0].set_ylabel("examples")
    fig.suptitle(
        "Earliest prefix at which the verdict becomes correct and stays correct",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def curve_metrics(trajs: list[Trajectory]) -> dict[str, np.ndarray]:
    """AUROC / accuracy at each prefix fraction, over all examples."""
    y = np.array([t.label for t in trajs])
    aurocs, accs = [], []
    for k in range(len(FRACTIONS)):
        p = np.array([t.fraction_of(t.probs)[k] for t in trajs])
        m = compute_metrics(y, p, threshold=THRESHOLD)
        aurocs.append(m["auroc"])
        accs.append(m["accuracy"])
    return {"auroc": np.array(aurocs), "accuracy": np.array(accs)}


def fig_auroc_curve(curves: dict[str, np.ndarray], dest: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(FRACTIONS))

    ax.plot(x, curves["auroc"], "o-", color=HALLU, lw=2, label="AUROC")
    ax.plot(x, curves["accuracy"], "s--", color=CLEAN, lw=1.6, label="accuracy")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")

    final = curves["auroc"][-1]
    # Where does the prefix reach 95% / 99% of the full-response AUROC?
    for frac, style in ((0.95, "--"), (0.99, ":")):
        target = 0.5 + frac * (final - 0.5)
        hit = np.argmax(curves["auroc"] >= target) if (curves["auroc"] >= target).any() else None
        if hit is not None and (curves["auroc"] >= target).any():
            ax.axvline(hit, color="black", ls=style, lw=0.9, alpha=0.7)
            ax.text(
                hit, 0.52, f" {frac:.0%} of final\n at {_pct_labels()[hit]}",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(_pct_labels(), rotation=45, fontsize=8)
    ax.set_xlabel("fraction of response seen")
    ax.set_ylabel("score")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Detector performance vs. prefix length", fontsize=12)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_prob_trajectory(trajs: list[Trajectory], dest: Path) -> None:
    """Mean p(hallucinated) per true class, with an interquartile band."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(FRACTIONS))

    for label, name, colour in ((1, "true hallucinated", HALLU), (0, "true clean", CLEAN)):
        sel = [t for t in trajs if t.label == label]
        if not sel:
            continue
        mat = np.stack([t.fraction_of(t.probs) for t in sel])   # (n, F)
        mean = mat.mean(axis=0)
        lo, hi = np.percentile(mat, [25, 75], axis=0)
        ax.plot(x, mean, "o-", color=colour, lw=2, label=f"{name} (n={len(sel)})")
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18)

    ax.axhline(THRESHOLD, color="black", ls="--", lw=1, label=f"threshold {THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(_pct_labels(), rotation=45, fontsize=8)
    ax.set_xlabel("fraction of response seen")
    ax.set_ylabel("p(hallucinated)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9)
    ax.set_title(
        "Predicted probability as the response unfolds (band = IQR)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_raster(trajs: list[Trajectory], dest: Path, max_rows: int = 400) -> None:
    """One row per example, one column per prefix fraction: right vs wrong.

    Sorted by how early the verdict locks in, so the shape of the boundary is
    the finding: a clean diagonal means detection time varies smoothly, a block
    means most examples resolve at the same point.
    """
    plt = _plt()
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, label, title in (
        (axes[0], 1, "Hallucinated"),
        (axes[1], 0, "Clean"),
    ):
        sel = [t for t in trajs if t.label == label]
        if not sel:
            ax.set_visible(False)
            continue

        def sort_key(t: Trajectory):
            first = t.first_stable_correct()
            # Never-correct rows sort last.
            return (1e9 if first is None else first / t.n_tokens)

        sel = sorted(sel, key=sort_key)
        if len(sel) > max_rows:
            # Even stride, so the picture stays representative rather than
            # showing only the earliest-detected examples.
            sel = [sel[i] for i in np.linspace(0, len(sel) - 1, max_rows).astype(int)]

        mat = np.stack([t.fraction_of(t.correct).astype(float) for t in sel])
        ax.imshow(
            mat, aspect="auto", interpolation="nearest",
            cmap=ListedColormap([WRONG, RIGHT]), vmin=0, vmax=1,
        )
        ax.set_xticks(np.arange(len(FRACTIONS)))
        ax.set_xticklabels(_pct_labels(), rotation=45, fontsize=7.5)
        ax.set_xlabel("fraction of response seen")
        ax.set_title(f"{title}  ({len(sel)} of {len([t for t in trajs if t.label==label])} shown)",
                     fontsize=11)
        ax.set_ylabel("examples, sorted by lock-in point")

    fig.suptitle(
        "Per-example correctness (green = correct, orange = wrong)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def write_report(
    trajs: list[Trajectory],
    curves: dict[str, np.ndarray],
    out_dir: Path,
    provenance: dict,
) -> Path:
    """Assemble report.md around the figures, with the numbers stated inline."""
    n = len(trajs)
    n_hallu = sum(1 for t in trajs if t.label == 1)
    lengths = sorted({t.n_tokens for t in trajs})

    def _median_lockin(label: int) -> tuple[str, int, int]:
        sel = [t for t in trajs if t.label == label]
        vals = [
            t.first_stable_correct() / t.n_tokens
            for t in sel
            if t.first_stable_correct() is not None
        ]
        never = len(sel) - len(vals)
        med = f"{np.median(vals):.0%}" if vals else "n/a"
        return med, never, len(sel)

    hall_med, hall_never, hall_n = _median_lockin(1)
    clean_med, clean_never, clean_n = _median_lockin(0)

    final_auroc = curves["auroc"][-1]
    target = 0.5 + 0.95 * (final_auroc - 0.5)
    reached = np.where(curves["auroc"] >= target)[0]
    early_at = _pct_labels()[int(reached[0])] if len(reached) else "never"

    rows = "\n".join(
        f"| {lab} | {curves['auroc'][k]:.4f} | {curves['accuracy'][k]:.4f} |"
        for k, lab in enumerate(_pct_labels())
    )

    text = f"""# Prefix forecasting — how early can the detector tell?

The detector was re-run on every prefix of every held-out response: at step `t`
it sees tokens `1..t` and nothing after. This measures whether a hallucination
verdict is available *during* generation, which is what the closed-loop steering
mode in [docs/plan.md](../../docs/plan.md) §1.5(c) depends on.

## Setup

| | |
|---|---|
| checkpoint | `{provenance['checkpoint']}` |
| LLM / dataset | {provenance['llm']} / {provenance['dataset']} |
| examples | {n} held-out ({n_hallu} hallucinated, {n - n_hallu} clean) |
| response lengths | {lengths[0]}–{lengths[-1]} tokens ({len(lengths)} distinct) |
| detector passes | {sum(t.n_tokens for t in trajs):,} |
| threshold | {THRESHOLD} on p(hallucinated) |

Prefixes are **sliced, not masked** — see `analysis/forecasting.py` for why the
two are not equivalent through the temporal Conv1d.

## Headline

- **Full-response AUROC: {final_auroc:.4f}.**
- **95% of that AUROC is already reached at {early_at} of the response.**
- Hallucinated responses lock in a correct verdict at a median of
  **{hall_med}** of the way through ({hall_never}/{hall_n} are still wrong at the end).
- Clean responses lock in at a median of **{clean_med}**
  ({clean_never}/{clean_n} still wrong at the end).

## Figures

### 1. Detection counts — `1_detection_counts.png`
The signed-count view: at each prefix fraction, correct calls are drawn above
the axis and incorrect ones below, separately for truly-hallucinated and truly-
clean responses. The dashed line marks where correct calls first outnumber
wrong ones.

### 2. Earliest stable detection — `2_earliest_detection.png`
Per example, the earliest prefix after which the verdict is correct *and never
flips again*. Transient early hits do not count — they would overstate how early
detection really happens. Examples whose final verdict is wrong are excluded and
reported separately.

### 3. Performance vs prefix — `3_auroc_curve.png`
AUROC and accuracy over the whole evaluated set at each prefix fraction. The
reference lines mark where the prefix reaches 95% and 99% of the full-response
AUROC, measured above chance.

### 4. Probability trajectory — `4_prob_trajectory.png`
Mean `p(hallucinated)` for each true class with an interquartile band. If the
two classes separate early, an early intervention has a signal to act on.

### 5. Per-example raster — `5_raster.png`
One row per response, sorted by lock-in point. Shows whether errors are spread
across many examples or concentrated in a stubborn few.

## Metric table

| prefix | AUROC | accuracy |
|---|---|---|
{rows}

## What this means for steering

The closed-loop drift mode refreshes the attribution map every `k` tokens from
the prefix so far. Its value depends entirely on the curve in figure 3: the
earlier AUROC saturates, the earlier a *trustworthy* map exists, and the more of
the response an intervention can still affect. If AUROC only approaches its
final value near 100%, closed-loop steering degenerates to a late, one-shot
intervention and the response-level aggregate map (§1.5d) is the cheaper
equivalent.

Note this measures the detector's own confidence trajectory, not the quality of
its Grad-CAM attribution at that prefix — a correct verdict does not guarantee a
correctly *localised* one. That is a separate check.
"""
    dest = out_dir / "report.md"
    dest.write_text(text, encoding="utf-8")
    return dest


def build_all(trajs: list[Trajectory], out_dir: Path, provenance: dict) -> Path:
    """Write every figure plus report.md into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = curve_metrics(trajs)

    fig_detection_counts(trajs, out_dir / "1_detection_counts.png")
    fig_earliest_detection(trajs, out_dir / "2_earliest_detection.png")
    fig_auroc_curve(curves, out_dir / "3_auroc_curve.png")
    fig_prob_trajectory(trajs, out_dir / "4_prob_trajectory.png")
    fig_raster(trajs, out_dir / "5_raster.png")

    # Raw trajectories, so a figure can be redrawn without re-running the sweep.
    np.savez_compressed(
        out_dir / "trajectories.npz",
        idx=np.array([t.idx for t in trajs]),
        label=np.array([t.label for t in trajs]),
        n_tokens=np.array([t.n_tokens for t in trajs]),
        probs_at_fraction=np.stack([t.fraction_of(t.probs) for t in trajs]),
        fractions=FRACTIONS,
        auroc=curves["auroc"],
        accuracy=curves["accuracy"],
    )

    report = write_report(trajs, curves, out_dir, provenance)
    logger.info("wrote %s and 5 figures to %s", report.name, out_dir)
    return report
