"""Figures, tables and reports for the prefix-forecasting analysis.

Two questions drive every artifact here, and nothing else is drawn:

  1. HOW EARLY IS THE VERDICT USABLE?  AUROC as a function of how much of the
     response the detector has seen, and the prefix at which that curve reaches
     95% of its own full-response value.
  2. WHEN DOES THE VERDICT STOP CHANGING?  Per example, the earliest prefix
     after which the call is correct and never flips again
     (`Trajectory.first_stable_correct`) -- the point from which the detector is
     right from there on out.

Everything is produced at two scales: one compact panel per (LLM, dataset)
cell, and one dense summary across the whole grid. The summary is the figure
meant for the paper; the per-cell panels are the supplement that backs it.

WHY ONE SUMMARY FIGURE AND NOT A WALL OF THEM
---------------------------------------------
The grid is 4 LLMs x 3 datasets. Drawn as separate per-cell figures that is 12
pages that no reader will cross-reference. The summary is instead two pooled
curves, the first carrying the across-cell spread as quantile bands, so the
variation is visible without a panel per cell. The per-cell NUMBERS are not in
the figure at all -- they are in the report's markdown tables and in
forecasting_summary.csv, which is where a reader checks a specific cell anyway.

The summary figure carries no title and no in-plot annotations: it is meant to
be dropped into a paper, where the caption is LaTeX's job.

COLOUR
------
Palettes come from `scripts/figures/style.py`, which documents why only TWO
categorical hues are CVD-safe in this band. So hue encodes the one contrast
that matters -- hallucinated vs clean -- and never the dataset or the LLM.
The per-cell spread is carried by quantile bands in a single hue instead, and
the per-cell numbers by the report's tables.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.analysis.forecasting import FRACTIONS, THRESHOLD, Trajectory
from scripts.figures import style as st
from scripts.figures import style_modern as fm
from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics

logger = get_logger(__name__)

#: Hue is reserved for the one contrast that matters. `style.py` explains why a
#: third categorical hue is not available CVD-safely in this lightness band.
HALLU = st.SECOND      # truly hallucinated
CLEAN = st.PRIMARY     # truly clean

#: Fraction of the full-response AUROC (measured above chance, so the bar is not
#: trivially met by a detector that is barely better than a coin) that counts as
#: "the verdict is now usable". Reported as `earliness`.
USABLE = 0.95

#: Headline prefix for the "a quarter of the way in" statistic in the report.
HEADLINE_FRACTION = 0.25

PRETTY_LLM = {
    "llama2_7b": "LLaMA-2-7B",
    "llama3.1_8b": "LLaMA-3.1-8B",
    "opt_6.7b": "OPT-6.7B",
    "qwen2.5_7b": "Qwen2.5-7B",
}
PRETTY_DATASET = {
    "triviaqa": "TriviaQA",
    "truthfulqa": "TruthfulQA",
    "coqa": "CoQA",
}
#: Display order when the cells are laid out as a grid. Cells outside these
#: lists still render -- they are appended in sorted order (see `_axis_order`).
LLM_ORDER = ("llama2_7b", "llama3.1_8b", "opt_6.7b", "qwen2.5_7b")
DATASET_ORDER = ("truthfulqa", "triviaqa", "coqa")

#: Grid the lock-in CDF is evaluated on. Finer than FRACTIONS because lock-in is
#: a per-token statistic and its CDF is a step function -- a coarse grid would
#: hide the steps that carry the shape.
CDF_GRID = np.linspace(0.0, 1.0, 201)


def pct(x: float, digits: int = 0) -> str:
    """Percent, or an em dash for an undefined value."""
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}%}"


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """Everything the figures need about one (LLM, dataset) pair.

    Built once from the trajectories and then treated as read-only, so the
    summary figure and the per-cell figure cannot disagree about a number.
    """

    llm: str
    dataset: str
    labels: np.ndarray       # (N,) 1 = hallucinated
    lockin: np.ndarray       # (N,) lock-in as a fraction of the response, NaN = never
    probs_at: np.ndarray     # (N, len(FRACTIONS)) p(hallucinated) at each prefix
    n_tokens: np.ndarray     # (N,)
    auroc: np.ndarray        # (len(FRACTIONS),)
    accuracy: np.ndarray     # (len(FRACTIONS),)

    @classmethod
    def from_trajectories(cls, trajs: list[Trajectory], llm: str, dataset: str) -> "Cell":
        labels = np.array([t.label for t in trajs])
        probs_at = np.stack([t.fraction_of(t.probs) for t in trajs])
        aurocs, accs = [], []
        for k in range(len(FRACTIONS)):
            m = compute_metrics(labels, probs_at[:, k], threshold=THRESHOLD)
            aurocs.append(m["auroc"])
            accs.append(m["accuracy"])
        return cls(
            llm=llm,
            dataset=dataset,
            labels=labels,
            lockin=np.array([t.lockin_fraction() for t in trajs]),
            probs_at=probs_at,
            n_tokens=np.array([t.n_tokens for t in trajs]),
            auroc=np.array(aurocs),
            accuracy=np.array(accs),
        )

    # -- identity ----------------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.llm}_{self.dataset}"

    @property
    def pretty(self) -> str:
        return (f"{PRETTY_LLM.get(self.llm, self.llm)} / "
                f"{PRETTY_DATASET.get(self.dataset, self.dataset)}")

    @property
    def n(self) -> int:
        return len(self.labels)

    # -- question 1: how early is the verdict usable? ----------------------
    @property
    def final_auroc(self) -> float:
        return float(self.auroc[-1])

    @property
    def earliness(self) -> float:
        """Smallest prefix fraction whose AUROC reaches `USABLE` of the final.

        Measured above chance: the target is 0.5 + USABLE * (final - 0.5), so a
        detector that finishes at 0.55 does not clear the bar just by starting
        near 0.5. Returns a value ON the FRACTIONS grid rather than an
        interpolated one -- each grid point is a real measurement, and
        interpolating would report a prefix the detector was never run at
        (the same reason `Trajectory.fraction_of` resamples by nearest token).
        NaN if the curve never gets there.
        """
        target = 0.5 + USABLE * (self.final_auroc - 0.5)
        hit = np.where(self.auroc >= target)[0]
        return float(FRACTIONS[hit[0]]) if len(hit) else float("nan")

    def auroc_at(self, fraction: float) -> float:
        """AUROC at the FRACTIONS grid point nearest `fraction`."""
        return float(self.auroc[int(np.argmin(np.abs(FRACTIONS - fraction)))])

    @property
    def retained_at_headline(self) -> float:
        """Share of the full-response AUROC lift already present at 25%."""
        lift = self.final_auroc - 0.5
        if lift <= 0:
            return float("nan")
        return (self.auroc_at(HEADLINE_FRACTION) - 0.5) / lift

    # -- question 2: when does the verdict stop changing? ------------------
    def lockin_of(self, label: int | None = None) -> np.ndarray:
        """Lock-in fractions for one class (or all), NaNs included."""
        return self.lockin if label is None else self.lockin[self.labels == label]

    def median_lockin(self, label: int | None = None) -> float:
        """Median lock-in among examples that DO lock in. NaN if none do."""
        vals = self.lockin_of(label)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if len(vals) else float("nan")

    def never_rate(self, label: int | None = None) -> float:
        """Share of examples whose final verdict is wrong, so they never lock in."""
        vals = self.lockin_of(label)
        return float(np.mean(~np.isfinite(vals))) if len(vals) else float("nan")

    def lockin_cdf(self, label: int | None = None) -> np.ndarray:
        """P(locked in by prefix f) on CDF_GRID.

        The denominator is EVERY example of the class, including those that
        never lock in, so the curve plateaus at 1 - never_rate rather than being
        renormalised to 1. A curve forced to 1 would show a detector that is
        wrong on 20% of examples as though it eventually got them all.
        """
        vals = self.lockin_of(label)
        if not len(vals):
            return np.full_like(CDF_GRID, np.nan)
        # NaN <= f is False, so non-locking examples correctly never count.
        return np.array([np.mean(vals <= f) for f in CDF_GRID])

    def mean_prob_band(self, label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Median and IQR of p(hallucinated) at each prefix, for one class."""
        sel = self.probs_at[self.labels == label]
        if not len(sel):
            nan = np.full(len(FRACTIONS), np.nan)
            return nan, nan, nan
        return (np.median(sel, axis=0),
                np.percentile(sel, 25, axis=0),
                np.percentile(sel, 75, axis=0))


# ---------------------------------------------------------------------------
# Pooled views over the grid
# ---------------------------------------------------------------------------


def _axis_order(values: set[str], preferred: tuple[str, ...]) -> list[str]:
    """`preferred` first (those that are present), then any extras, sorted."""
    known = [v for v in preferred if v in values]
    return known + sorted(values - set(known))


class Grid:
    """The set of cells, with the pooled statistics the summary figure draws."""

    def __init__(self, cells: list[Cell]):
        if not cells:
            raise ValueError("no cells to summarise")
        self.cells = cells
        self.llms = _axis_order({c.llm for c in cells}, LLM_ORDER)
        self.datasets = _axis_order({c.dataset for c in cells}, DATASET_ORDER)
        self._by_key = {(c.llm, c.dataset): c for c in cells}

    def get(self, llm: str, dataset: str) -> Cell | None:
        return self._by_key.get((llm, dataset))

    @property
    def mean_auroc(self) -> np.ndarray:
        """AUROC at each prefix, averaged over CELLS (not over examples).

        Cell-weighted, so a dataset with more held-out rows does not dominate
        the headline curve -- the claim being made is about the method across
        settings, not about one corpus.
        """
        return np.nanmean(np.stack([c.auroc for c in self.cells]), axis=0)

    @property
    def mean_final_auroc(self) -> float:
        return float(np.nanmean([c.final_auroc for c in self.cells]))

    @property
    def pooled_earliness(self) -> float:
        """Where the CELL-MEAN AUROC curve reaches USABLE of its final value."""
        final = float(self.mean_auroc[-1])
        target = 0.5 + USABLE * (final - 0.5)
        hit = np.where(self.mean_auroc >= target)[0]
        return float(FRACTIONS[hit[0]]) if len(hit) else float("nan")

    def pooled_lockin_cdf(self, label: int | None = None) -> np.ndarray:
        """Cell-mean of the per-cell lock-in CDFs, for the same reason as above."""
        return np.nanmean(
            np.stack([c.lockin_cdf(label) for c in self.cells]), axis=0
        )

    def pooled_median_lockin(self, label: int | None = None) -> float:
        return float(np.nanmedian([c.median_lockin(label) for c in self.cells]))



# ---------------------------------------------------------------------------
# Figure: the summary (this is the one for the paper)
# ---------------------------------------------------------------------------


def _frac_axis(ax, label: str = "% of the response seen") -> None:
    """Shared x-axis treatment: prefix fraction, ticked as bare percentages.

    The "%" lives in the axis label rather than on every tick -- four ticks each
    carrying their own "%" is what pushes this axis into overlapping at
    half-panel width.
    """
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel(label)


def _pct_axis(ax, label: str) -> None:
    """Shared y-axis treatment for a 0-1 share."""
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel(label)


def _panel_head(ax, letter: str, ylabel: str, pad: int = 26) -> None:
    """Panel letter above a horizontal y-axis label.

    The y-label is set flat at the top-left rather than rotated up the spine:
    rotated labels force the reader to tilt their head for the one string that
    says what the axis IS, and they eat left margin that the plot could use.
    Both sit inside the space the title pad reserves, which is what keeps
    constrained_layout aware of them (free-floating text is not).
    """
    ax.set_ylabel("")
    ax.set_title(letter, loc="left", pad=pad, fontsize=11.5,
                 fontweight="bold", color=st.INK)
    ax.text(0.0, 1.015, ylabel, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=fm.INK_SOFT)


def _legend(ax, handles=None, loc="lower right"):
    """Frameless-looking legend that still masks what runs underneath it.

    `frameon=False` let the plateau rules and band edges cross the label text.
    A surface-coloured patch with no edge reads as frameless but occludes.
    """
    kw = dict(loc=loc, handlelength=1.5, borderpad=0.45, labelspacing=0.45,
              frameon=True, framealpha=1.0, facecolor=fm.SURFACE,
              edgecolor="none")
    return ax.legend(handles=handles, **kw) if handles else ax.legend(**kw)


def _polish(ax) -> None:
    """Editorial axis treatment: hairline horizontal rules, no box."""
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=fm.HAIRLINE, lw=0.8)
    ax.grid(False, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(fm.HAIRLINE)
    ax.tick_params(colors=fm.INK_FAINT, labelsize=9)


def fig_summary(grid: Grid, out_dir: Path, name: str = "forecasting_summary"):
    """The paper figure: the two pooled curves, and nothing else.

    Deliberately bare -- no suptitle, no in-plot callouts. Every number that
    used to be annotated onto the marks (the crossing point, the medians) is in
    the report and the CSVs, and in a paper the explanation belongs in the
    LaTeX caption, not burned into the image.

    The per-cell breakdown that used to sit here as two heatmaps now lives only
    in the report's markdown tables and `forecasting_summary.csv` -- the same
    numbers, without spending half the figure on them.

    SPREAD IS DRAWN AS BANDS, NOT AS 12 LINES
    -----------------------------------------
    The earlier draft drew every cell as its own faint line. Twelve overlapping
    polylines read as noise, and the eye cannot recover a distribution from
    them. Nested quantile bands (10-90 and 25-75 across cells) carry the same
    spread as a shape, and let the mean stay the only line in the panel.
    """
    fm.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.patheffects import withStroke

    # A white casing under each line lifts it off the band it sits in without
    # a heavier stroke, which would swamp a 9-inch-wide figure.
    casing = [withStroke(linewidth=4.5, foreground=fm.SURFACE)]

    fig, (ax_auroc, ax_cdf) = plt.subplots(
        1, 2, figsize=(9.6, 3.9), gridspec_kw={"wspace": 0.05},
    )

    # -- (a) AUROC vs prefix ------------------------------------------------
    stack = np.stack([c.auroc for c in grid.cells])
    p10, p25, p75, p90 = np.nanpercentile(stack, [10, 25, 75, 90], axis=0)
    mean = grid.mean_auroc

    ax_auroc.fill_between(FRACTIONS, p10, p90, color=CLEAN, alpha=0.11, lw=0,
                          zorder=1)
    ax_auroc.fill_between(FRACTIONS, p25, p75, color=CLEAN, alpha=0.20, lw=0,
                          zorder=2)
    ax_auroc.plot(FRACTIONS, mean, color=CLEAN, lw=2.9, zorder=4,
                  path_effects=casing)
    ax_auroc.plot(FRACTIONS, mean, "o", color=CLEAN, ms=4.2, zorder=5,
                  mec=fm.SURFACE, mew=1.1)
    ax_auroc.axhline(0.5, color=fm.HAIRLINE, ls=(0, (2, 3)), lw=1.1, zorder=0)

    _frac_axis(ax_auroc)
    ax_auroc.set_ylim(0.45, min(1.0, max(0.75, float(np.nanmax(p90)) + 0.03)))
    _polish(ax_auroc)
    _panel_head(ax_auroc, "(a)", "AUROC")
    _legend(ax_auroc, handles=[
        Line2D([], [], color=CLEAN, lw=2.9,
               label=f"mean of {len(grid.cells)} cells"),
        Patch(facecolor=CLEAN, alpha=0.20, lw=0, label="middle 50% of cells"),
        Patch(facecolor=CLEAN, alpha=0.11, lw=0, label="10th–90th percentile"),
    ])

    # -- (b) lock-in CDF ----------------------------------------------------
    # NOT filled under the curves. The two classes overlap almost everywhere,
    # so two tinted areas stack into the grey-mauve that style.py warns about
    # ("two mid-tones average to mud") and the hues stop being readable. The
    # plateau each curve tops out at -- the share that never locks in -- is
    # carried by a hairline rule in the curve's own colour instead.
    for label, colour, lab in ((1, HALLU, "hallucinated"), (0, CLEAN, "clean")):
        cdf = grid.pooled_lockin_cdf(label)
        ax_cdf.axhline(cdf[-1], color=colour, ls=(0, (2, 3)), lw=1.1,
                       alpha=0.45, zorder=1)
        ax_cdf.plot(CDF_GRID, cdf, color=colour, lw=2.7, label=lab, zorder=4,
                    path_effects=casing)
        med = grid.pooled_median_lockin(label)
        if np.isfinite(med):
            ax_cdf.plot([med], [float(np.interp(med, CDF_GRID, cdf))], "o",
                        color=colour, ms=7.5, zorder=5,
                        mec=fm.SURFACE, mew=1.4)

    _frac_axis(ax_cdf)
    _pct_axis(ax_cdf, "")
    _polish(ax_cdf)
    _panel_head(ax_cdf, "(b)", "% of responses locked in")
    # Upper left is empty by construction in a CDF panel.
    _legend(ax_cdf, loc="upper left")

    return fm.save(fig, out_dir, name)


def fig_summary_by_model(grid: Grid, out_dir: Path, name: str = "forecasting_by_model"):
    """Per-MODEL AUROC-vs-prefix lines, for a grid that is ONE dataset x
    several models (e.g. the TriviaQA-only structure-analysis forecasting
    sweep) -- fig_summary's pooled-band design is for a many-cell grid where
    individual lines are noise; here there are only as many cells as models
    (3-4), and which MODEL a line belongs to is exactly the comparison this
    figure exists to make, so pooling it away would erase the point.

    COLOUR: style.py validates only TWO CVD-safe categorical hues (see its
    docstring), not enough for 4 models. Lines instead use ONE hue's
    light-to-dark lightness ramp (style.SEQUENTIAL, sampled at 4 fixed
    points) -- lightness differences stay perceptible under CVD where a 3rd+
    arbitrary hue would not (style.py's own rejected-3rd-hue finding). Every
    line is ALSO labelled directly at its right end, so identifying a line
    never depends on colour discrimination alone.
    """
    fm.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    casing = [withStroke(linewidth=4.5, foreground=fm.SURFACE)]

    # Fixed lightness steps (not len(cells)-dependent), so a given model's
    # line is the same shade across different figures/subsets -- picked
    # light-to-dark in LLM_ORDER so earlier-released/smaller models read
    # lighter. Sampled from style.SEQUENTIAL, the one validated single-hue
    # ramp (see style.py's docstring: "Light->dark keeps magnitude readable
    # in greyscale and under CVD").
    shades = [st.SEQUENTIAL(x) for x in (0.35, 0.58, 0.78, 0.97)]

    cells = sorted(grid.cells, key=lambda c: grid.llms.index(c.llm) if c.llm in grid.llms else 99)
    datasets = {c.dataset for c in cells}
    dataset_label = (PRETTY_DATASET.get(next(iter(datasets)), next(iter(datasets)))
                      if len(datasets) == 1 else "mixed datasets")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # Colour-matched legend handles, used INSTEAD OF end-of-line annotate()
    # text: annotate(..., annotation_clip=False) draws outside the axes,
    # which constrained_layout (set globally by fm.apply()) does not reserve
    # room for, so the longest label silently clips at the figure edge
    # (verified: fig.subplots_adjust is a no-op under constrained_layout and
    # only raises a warning, it does not fix the clipping). A legend IS
    # something constrained_layout accounts for automatically.
    from matplotlib.lines import Line2D
    handles = []
    for i, cell in enumerate(cells):
        colour = shades[i % len(shades)]
        label = PRETTY_LLM.get(cell.llm, cell.llm)
        ax.plot(FRACTIONS, cell.auroc, color=colour, lw=2.6, zorder=4,
                path_effects=casing)
        ax.plot(FRACTIONS, cell.auroc, "o", color=colour, ms=3.8, zorder=5,
                mec=fm.SURFACE, mew=1.0)
        handles.append(Line2D([], [], color=colour, lw=2.6, label=label))

    ax.axhline(0.5, color=fm.HAIRLINE, ls=(0, (2, 3)), lw=1.1, zorder=0)
    _frac_axis(ax)
    all_auroc = np.concatenate([c.auroc for c in cells])
    ax.set_ylim(0.45, min(1.0, max(0.75, float(np.nanmax(all_auroc)) + 0.03)))
    _polish(ax)
    _panel_head(ax, "", f"AUROC — {dataset_label}")
    _legend(ax, handles=handles, loc="lower right")

    return fm.save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# Figure: one compact row per cell
# ---------------------------------------------------------------------------


def fig_cell(cell: Cell, out_dir: Path):
    """Three panels on one row: the curve, the lock-in CDF, the separation."""
    fm.apply()
    import matplotlib.pyplot as plt

    from matplotlib.patheffects import withStroke
    casing = [withStroke(linewidth=4.0, foreground=fm.SURFACE)]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(11.8, 3.7), gridspec_kw={"wspace": 0.06},
    )

    # -- AUROC / accuracy ---------------------------------------------------
    ax_a.plot(FRACTIONS, cell.auroc, "-", color=CLEAN, lw=2.5,
              label="AUROC", path_effects=casing, zorder=4)
    ax_a.plot(FRACTIONS, cell.auroc, "o", color=CLEAN, ms=4.2, zorder=5,
              mec=fm.SURFACE, mew=1.0)
    ax_a.plot(FRACTIONS, cell.accuracy, "s--", color=st.NEUTRAL, lw=1.6, ms=3.8,
              label="accuracy", zorder=3)
    ax_a.axhline(0.5, color=st.GRID, ls=":", lw=1.2)
    e = cell.earliness
    lo, hi = 0.45, min(1.0, max(0.75, float(np.nanmax(cell.auroc)) + 0.06))
    if np.isfinite(e):
        ax_a.axvline(e, color=HALLU, ls="--", lw=1.4)
        # Ride the top of the vline, flipping to its left half once the
        # crossing is far enough right that the label would leave the axes.
        right = e > 0.55
        ax_a.text(e + (-0.02 if right else 0.02), hi - 0.005,
                  f"{USABLE:.0%} of final\nby {pct(e)}",
                  fontsize=9, color=HALLU, va="top", linespacing=1.3,
                  ha="right" if right else "left")
    _frac_axis(ax_a)
    ax_a.set_ylim(lo, hi)
    _polish(ax_a)
    _panel_head(ax_a, "(a)  Performance vs prefix", "score")
    _legend(ax_a)

    # -- lock-in CDF --------------------------------------------------------
    # Upper left is empty by construction in a CDF panel, which is where the
    # one colour key for the whole figure goes -- HALLU/CLEAN mean the same
    # thing in (b) and (c), so (c) does not repeat it.
    for label, colour, lab in ((1, HALLU, "hallucinated"), (0, CLEAN, "clean")):
        cdf = cell.lockin_cdf(label)
        ax_b.axhline(cdf[-1], color=colour, ls=(0, (2, 3)), lw=1.1,
                     alpha=0.45, zorder=1)
        ax_b.plot(CDF_GRID, cdf, color=colour, lw=2.5, label=lab, zorder=4,
                  path_effects=casing)
        med = cell.median_lockin(label)
        if np.isfinite(med):
            ax_b.plot([med], [float(np.interp(med, CDF_GRID, cdf))], "o",
                      color=colour, ms=7, zorder=5, mec=fm.SURFACE, mew=1.3)
    _frac_axis(ax_b)
    _pct_axis(ax_b, "")
    _polish(ax_b)
    _panel_head(ax_b, "(b)  Lock-in point", "% locked in")
    _legend(ax_b, loc="upper left")

    # -- class separation ---------------------------------------------------
    for label, colour in ((1, HALLU), (0, CLEAN)):
        med, q1, q3 = cell.mean_prob_band(label)
        # Two overlapping washes average to mud (see style.py). A thin edge in
        # the band's own hue keeps each one's extent readable through the
        # overlap, so the alpha can stay low enough not to muddy at all.
        ax_c.fill_between(FRACTIONS, q1, q3, color=colour, alpha=0.12, lw=0,
                          zorder=2)
        for edge in (q1, q3):
            ax_c.plot(FRACTIONS, edge, color=colour, lw=0.9, alpha=0.55,
                      zorder=3)
        ax_c.plot(FRACTIONS, med, color=colour, lw=2.5, zorder=4,
                  path_effects=casing)
    ax_c.axhline(THRESHOLD, color=fm.HAIRLINE, ls=(0, (2, 3)), lw=1.1, zorder=1)
    ax_c.text(0.995, THRESHOLD + 0.012, "threshold", ha="right", va="bottom",
              fontsize=8.5, color=fm.INK_FAINT)
    _frac_axis(ax_c)
    ax_c.set_ylim(0, 1)
    _polish(ax_c)
    _panel_head(ax_c, "(c)  Class separation", "p(hallucinated), median + IQR")

    fm.title(fig, cell.pretty,
             f"{cell.n:,} held-out responses · full-response AUROC "
             f"{cell.final_auroc:.3f} · usable by {pct(cell.earliness)}")
    return fm.save(fig, out_dir, cell.key)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


SUMMARY_COLUMNS = [
    "llm", "dataset", "n", "n_hallucinated", "final_auroc",
    f"earliness_{int(USABLE * 100)}", "auroc_at_25", "retained_at_25",
    "median_lockin_hallucinated", "median_lockin_clean",
    "never_lockin_rate", "median_response_tokens",
]


def _summary_row(c: Cell) -> dict:
    return {
        "llm": c.llm,
        "dataset": c.dataset,
        "n": c.n,
        "n_hallucinated": int(c.labels.sum()),
        "final_auroc": round(c.final_auroc, 4),
        f"earliness_{int(USABLE * 100)}": round(c.earliness, 4),
        "auroc_at_25": round(c.auroc_at(HEADLINE_FRACTION), 4),
        "retained_at_25": round(c.retained_at_headline, 4),
        "median_lockin_hallucinated": round(c.median_lockin(1), 4),
        "median_lockin_clean": round(c.median_lockin(0), 4),
        "never_lockin_rate": round(c.never_rate(), 4),
        "median_response_tokens": int(np.median(c.n_tokens)),
    }


def write_tables(grid: Grid, out_dir: Path) -> list[Path]:
    """Per-cell summary and the full AUROC/accuracy curves, as CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    summary = out_dir / "forecasting_summary.csv"
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for c in grid.cells:
            w.writerow(_summary_row(c))
    written.append(summary)

    curves = out_dir / "forecasting_curves.csv"
    with open(curves, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["llm", "dataset", "prefix_fraction", "auroc", "accuracy"])
        for c in grid.cells:
            for k, f in enumerate(FRACTIONS):
                w.writerow([c.llm, c.dataset, f"{f:.2f}",
                            f"{c.auroc[k]:.4f}", f"{c.accuracy[k]:.4f}"])
    written.append(curves)
    return written


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _grid_table(grid: Grid, fn, fmt=lambda v: pct(v)) -> str:
    """Markdown 4x3 table of `fn(cell)`, datasets down, LLMs across."""
    head = "| | " + " | ".join(PRETTY_LLM.get(m, m) for m in grid.llms) + " |"
    rule = "|---" * (len(grid.llms) + 1) + "|"
    rows = []
    for ds in grid.datasets:
        cells = []
        for llm in grid.llms:
            c = grid.get(llm, ds)
            cells.append("—" if c is None else fmt(fn(c)))
        rows.append(f"| **{PRETTY_DATASET.get(ds, ds)}** | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def write_summary_report(grid: Grid, out_dir: Path, figure_rel: str) -> Path:
    """The pooled report: the headline numbers, the grid, and what they mean."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = sum(c.n for c in grid.cells)
    mean_final = grid.mean_final_auroc
    pooled_early = grid.pooled_earliness
    retained = np.array([c.retained_at_headline for c in grid.cells])
    med_h = grid.pooled_median_lockin(1)
    med_c = grid.pooled_median_lockin(0)

    # Best/worst must range over cells that actually HAVE the statistic. A cell
    # whose AUROC is undefined (a single-class slice, say) is unmeasurable, not
    # the worst performer, and naming it as the worst misreports the grid.
    measured = [c for c in grid.cells if np.isfinite(c.earliness)]
    early_vals = np.array([c.earliness for c in measured])
    locked = [c for c in grid.cells if np.isfinite(c.median_lockin(1))]
    n_unmeasured = len(grid.cells) - len(measured)

    earliest = min(measured, key=lambda c: c.earliness) if measured else None
    latest = max(measured, key=lambda c: c.earliness) if measured else None
    best_lock = min(locked, key=lambda c: c.median_lockin(1)) if locked else None

    if measured:
        spread = (
            f"Best: **{earliest.pretty}** at {pct(earliest.earliness)}. Worst: "
            f"**{latest.pretty}** at {pct(latest.earliness)}. Across the "
            f"measured cells the spread is {pct(early_vals.min())}–"
            f"{pct(early_vals.max())}, median {pct(np.median(early_vals))}."
        )
    else:
        spread = "No cell reached the threshold, so there is no best or worst."
    if n_unmeasured:
        spread += (f" {n_unmeasured} of {len(grid.cells)} cells are excluded "
                   f"here — their AUROC is undefined (single-class slice).")

    lock_line = (
        f"Earliest: **{best_lock.pretty}** at {pct(best_lock.median_lockin(1))}."
        if best_lock else
        "No cell has a defined median lock-in for hallucinated responses."
    )

    # Claims about SHAPE have to be read off the data, not asserted: which
    # class locks in first, and where the curve stops climbing, both flip
    # depending on the corpus. An earlier draft hard-coded both and was wrong
    # the moment the numbers moved.
    mean_curve = grid.mean_auroc
    # The band the figure draws, quoted at the headline prefix so the prose and
    # the panel cannot describe different things.
    k_head = int(np.argmin(np.abs(FRACTIONS - HEADLINE_FRACTION)))
    auroc_stack = np.stack([c.auroc for c in grid.cells])
    band_lo, band_hi = np.nanpercentile(auroc_stack[:, k_head], [10, 90])
    lift = mean_curve[-1] - 0.5
    knee_idx = (np.where(mean_curve >= 0.5 + 0.90 * lift)[0] if lift > 0
                else np.array([], dtype=int))
    knee = float(FRACTIONS[knee_idx[0]]) if len(knee_idx) else float("nan")

    if np.isfinite(med_h) and np.isfinite(med_c) and abs(med_h - med_c) > 0.02:
        first, second = (("hallucinated", "clean") if med_h < med_c
                         else ("clean", "hallucinated"))
        asymmetry = (
            f"The {first} curve rises to the left of the {second} one: a "
            f"verdict on a {first} response settles earlier "
            f"({pct(min(med_h, med_c))} vs {pct(max(med_h, med_c))} median). "
        )
        if first == "hallucinated":
            asymmetry += ("That is the useful direction — the class worth "
                          "acting on early is the one that is decided first.")
        else:
            asymmetry += ("That is the awkward direction — the class worth "
                          "acting on early is the one that settles last, so "
                          "an early positive call carries more risk of "
                          "flipping than an early negative one.")
    else:
        asymmetry = (
            f"The two curves sit on top of each other ({pct(med_h)} vs "
            f"{pct(med_c)} median), so lead time is the same whichever way "
            "the verdict goes."
        )

    text = f"""# How early can the detector call a hallucination?

![forecasting summary]({figure_rel})

The detector was re-run on every prefix of every held-out response: at step `t`
it sees tokens `1..t` and nothing after, so each point is a verdict the detector
could genuinely have produced mid-generation. Prefixes are **sliced, not
masked** — see `scripts/analysis/forecasting.py` for why the two are not
equivalent through the temporal `Conv1d`.

Two questions, one per half of the figure: **how early does the verdict become
usable**, and **when does it stop changing**.

## Headline

- Across all {len(grid.cells)} LLM × dataset cells ({n_total:,} held-out responses),
  mean full-response **AUROC is {mean_final:.3f}**.
- The cell-mean AUROC curve reaches **{USABLE:.0%} of that value by
  {pct(pooled_early)} of the response** — the detector does not need to see the
  answer finish.
- At just **{HEADLINE_FRACTION:.0%}** of the response, the detector already
  retains a mean **{pct(np.nanmean(retained))}** of its full-response
  discriminative lift (range {pct(np.nanmin(retained))}–{pct(np.nanmax(retained))}).
- A hallucinated response becomes correctly flagged, and stays flagged, at a
  median of **{pct(med_h)}** of the way through; a clean one at **{pct(med_c)}**.

## Per-cell: prefix needed to reach {USABLE:.0%} of full-response AUROC

{_grid_table(grid, lambda c: c.earliness)}

{spread}

## Per-cell: median lock-in point (hallucinated responses)

{_grid_table(grid, lambda c: c.median_lockin(1))}

This is the earliest prefix after which the call is correct *and never flips
again*. Transient early hits deliberately do not count — they would overstate
how early a usable decision exists. {lock_line}

## Per-cell: full-response AUROC (for reference)

{_grid_table(grid, lambda c: c.final_auroc, fmt=lambda v: f"{v:.3f}")}

## Reading the figure

**Panel (a)** is the forecasting claim: the cell-mean AUROC reaches 90% of its
full-response lift by **{pct(knee)}** of the response and is close to flat
after that. The bands are the spread across cells — the middle 50%, and the
10th–90th percentile — not a confidence interval on the mean. At
{HEADLINE_FRACTION:.0%} of the response that band spans AUROC
{band_lo:.3f}–{band_hi:.3f}, so the shape is a property of the method rather
than an average over settings that disagree.

**Panel (b)** answers the stricter question. The curves are cumulative: the
share of responses whose verdict is already correct and never changes after
that point. They plateau **below 100%**, and the gap is exactly the share the
detector gets wrong at the end — those examples never lock in, and inflating
the curve to 1 would hide them. {asymmetry}

Neither panel breaks the grid out per cell — the tables above do that, so a
claim about the mean can be checked against the worst case rather than taken
on trust.

## What this means

The practical consequence is that a verdict is available while the response is
still being written. A detector that only worked on the finished response could
support flagging or re-generation after the fact; one that is {pct(np.nanmean(retained))}
of the way to its final accuracy at {HEADLINE_FRACTION:.0%} of the tokens can
support intervention *during* decoding, when most of the response is still
unwritten and can still be changed.

The lock-in statistic is the stricter bar, and the honest caveat: a median lock-in
of {pct(med_h)} on hallucinated responses means half of them are still capable
of flipping past that point. Early intervention on the basis of a single prefix
verdict trades some precision for lead time. The two panels together say what
that trade costs at any chosen cut-off.

One limit worth stating: this measures the detector's confidence trajectory, not
the quality of its attribution map at that prefix. A correct verdict does not
guarantee a correctly *localised* one — that is a separate check.

## Method

| | |
|---|---|
| examples | {n_total:,} held-out, across {len(grid.cells)} cells |
| prefix grid | {len(FRACTIONS)} fractions, {FRACTIONS[0]:.0%}–{FRACTIONS[-1]:.0%} of each response |
| threshold | {THRESHOLD} on p(hallucinated) |
| lock-in | earliest `t` after which the call is correct through the end |
| "usable" | AUROC ≥ {USABLE:.0%} of the full-response value, measured above chance |

Per-cell figures and reports are in `cells/`. Raw numbers are in
`docs/tables/forecasting/`.
"""
    dest = out_dir / "report.md"
    dest.write_text(text, encoding="utf-8")
    return dest


def write_cell_report(cell: Cell, out_dir: Path, figure_rel: str,
                      provenance: dict) -> Path:
    """One page per cell, with the same statistics as the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {f:.0%} | {cell.auroc[k]:.4f} | {cell.accuracy[k]:.4f} |"
        for k, f in enumerate(FRACTIONS)
    )
    text = f"""# Prefix forecasting — {cell.pretty}

![{cell.key}]({figure_rel})

| | |
|---|---|
| checkpoint | `{provenance.get('checkpoint', '?')}` |
| LLM / dataset | {cell.llm} / {cell.dataset} |
| examples | {cell.n:,} held-out ({int(cell.labels.sum())} hallucinated, {int((1 - cell.labels).sum())} clean) |
| response length | median {int(np.median(cell.n_tokens))} tokens (max {int(cell.n_tokens.max())}) |
| detector passes | {int(cell.n_tokens.sum()):,} |
| threshold | {THRESHOLD} on p(hallucinated) |

## Headline

- Full-response AUROC **{cell.final_auroc:.4f}**.
- Reaches {USABLE:.0%} of that at **{pct(cell.earliness)}** of the response.
- At {HEADLINE_FRACTION:.0%} of the response: AUROC {cell.auroc_at(HEADLINE_FRACTION):.4f}
  (**{pct(cell.retained_at_headline)}** of the full-response lift).
- Median lock-in: **{pct(cell.median_lockin(1))}** hallucinated,
  **{pct(cell.median_lockin(0))}** clean.
- Never locks in (final call wrong): **{pct(cell.never_rate(), 1)}** overall —
  {pct(cell.never_rate(1), 1)} of hallucinated, {pct(cell.never_rate(0), 1)} of clean.

## AUROC and accuracy by prefix

| prefix | AUROC | accuracy |
|---|---|---|
{rows}

See [../report.md](../report.md) for the pooled view across every cell.
"""
    dest = out_dir / f"{cell.key}.md"
    dest.write_text(text, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def build_cell(cell: Cell, docs: "ForecastPaths", provenance: dict) -> Path:
    """Per-cell figure + report."""
    fig_cell(cell, docs.cell_figures)
    return write_cell_report(
        cell, docs.cell_reports,
        figure_rel=docs.rel_cell_figure(cell.key),
        provenance=provenance,
    )


def build_summary(grid: Grid, docs: "ForecastPaths") -> Path:
    """Pooled figure, tables, and the summary report.

    When the grid spans exactly one dataset (e.g. a TriviaQA-only sweep
    across models, as opposed to the full LLM x dataset grid), ALSO draws
    fig_summary_by_model -- fig_summary's pooled bands make sense across many
    cells, but with only as many cells as models, the per-model identity IS
    the comparison, so it gets its own line, not an average. A multi-dataset
    grid skips this (it would conflate "which model" with "which dataset" in
    one set of lines, which fig_summary_by_model was not designed to show).
    """
    fig_summary(grid, docs.figures)
    if len({c.dataset for c in grid.cells}) == 1:
        fig_summary_by_model(grid, docs.figures)
    write_tables(grid, docs.tables)
    report = write_summary_report(grid, docs.reports,
                                  figure_rel=docs.rel_summary_figure())
    logger.info(
        "wrote summary over %d cells -> %s", len(grid.cells), report
    )
    return report


@dataclass
class ForecastPaths:
    """Where every forecasting artifact goes under `docs/`.

    One object so the figure, the report and the markdown link that points from
    one to the other cannot drift apart.
    """

    root: Path

    @property
    def figures(self) -> Path:
        return self.root / "figures" / "forecasting"

    @property
    def cell_figures(self) -> Path:
        return self.figures / "cells"

    @property
    def tables(self) -> Path:
        return self.root / "tables" / "forecasting"

    @property
    def reports(self) -> Path:
        return self.root / "reports" / "forecasting"

    @property
    def cell_reports(self) -> Path:
        return self.reports / "cells"

    @property
    def cache(self) -> Path:
        return self.root / "forecasting_cache"

    def cache_file(self, key: str) -> Path:
        return self.cache / f"{key}.npz"

    # -- links, written from the report's own directory --------------------
    def rel_summary_figure(self) -> str:
        return "../../figures/forecasting/forecasting_summary.png"

    def rel_cell_figure(self, key: str) -> str:
        return f"../../../figures/forecasting/cells/{key}.png"

    def mkdirs(self) -> None:
        for p in (self.figures, self.cell_figures, self.tables,
                  self.reports, self.cell_reports, self.cache):
            p.mkdir(parents=True, exist_ok=True)
