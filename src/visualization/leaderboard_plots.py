"""
Pillar 5, Group A -- leaderboard and ranking figures (Figs 1-5).

Fig 1  Leaderboard bar chart
Fig 2  Multi-metric grouped bars
Fig 3  Per-series error distribution boxplots
Fig 4  Critical Difference diagram (Demsar 2006)
Fig 5  Multiple Comparisons with the Best (Koning et al. 2005)

Figs 4 and 5 are also emitted side by side by `plot_significance_pair`. That
pairing is deliberate and is the most important figure this project produces:
the pooled and rank-based tests disagree about which model leads, and the
disagreement is a finding about M5's structure rather than something to
reconcile away. Presenting them apart would let a reader take either as
"the" answer.

References
----------
Demsar (2006), "Statistical comparisons of classifiers over multiple data
    sets", JMLR 7:1-30.  [CD diagram]
Koning, Franses, Hibon & Stekler (2005), "The M3 competition: Statistical
    tests of the results", IJF 21(3):397-409.  [MCB]
Makridakis, Spiliotis & Assimakopoulos (2022), IJF 38(4):1346-1364 -- reports
    the same WRMSSE/MCB rank divergence within the M5 competition itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.logging import get_logger
from src.visualization.style import (
    TIER_COLORS,
    apply_style,
    color_for,
    save_figure,
    tier_legend_handles,
    tier_of,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fig 1 -- leaderboard
# ---------------------------------------------------------------------------


def plot_leaderboard(
    scores: pd.DataFrame,
    out_dir: Path | str,
    metric: str = "wape",
    model_col: str = "model",
    reference_model: str = "seasonal_naive",
    name: str = "fig01_leaderboard",
) -> Path:
    """
    Horizontal bars, one per model, coloured by tier and sorted best-first.

    A vertical reference line marks the seasonal naive baseline, because RMSSE
    is *defined* relative to a naive forecast: a model to the right of that
    line has not earned its complexity. Without the line a reader has no way to
    judge whether the whole field is good or the whole field is poor.
    """
    apply_style()
    df = scores[[model_col, metric]].dropna().sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * len(df))))
    colors = [color_for(m) for m in df[model_col]]
    bars = ax.barh(df[model_col].astype(str), df[metric], color=colors, height=0.72)
    ax.invert_yaxis()  # best model at the top

    span = float(df[metric].max() - df[metric].min()) or 1.0
    for bar, value in zip(bars, df[metric]):
        ax.text(
            bar.get_width() + span * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8.5,
        )

    ref = df[df[model_col].astype(str) == reference_model]
    if len(ref):
        ax.axvline(
            float(ref[metric].iloc[0]),
            color="#c1444a",
            ls="--",
            lw=1.4,
            label=f"{reference_model} baseline",
            zorder=0,
        )

    present = [t for t in TIER_COLORS if any(tier_of(m) == t for m in df[model_col])]
    handles = tier_legend_handles(present)
    if len(ref):
        handles.append(ax.get_lines()[-1])
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    ax.set_xlabel(f"{metric.upper()}  (lower is better)")
    ax.set_title(f"Fig 1 — Model leaderboard by {metric.upper()}")
    ax.margins(x=0.12)

    return save_figure(
        fig,
        out_dir,
        name,
        caption=(
            f"Models ranked by {metric.upper()}, coloured by tier. The dashed line "
            f"marks {reference_model}: RMSSE is defined relative to a naive "
            f"forecast, so any model at or beyond that line has not justified its "
            f"complexity. Bar ORDER is not itself a finding -- see Figs 4 and 5 for "
            f"which gaps are statistically separable."
        ),
    )


# ---------------------------------------------------------------------------
# Fig 2 -- multi-metric comparison
# ---------------------------------------------------------------------------


def plot_multi_metric(
    scores: pd.DataFrame,
    out_dir: Path | str,
    metrics: Sequence[str] = ("wape", "rmsse", "mase"),
    model_col: str = "model",
    name: str = "fig02_multi_metric",
) -> Path:
    """
    Grouped bars across several metrics, each normalised to its own best model.

    Normalisation is what makes the comparison legible: WAPE is a percentage in
    the 70-100 range while RMSSE sits near 1, so raw bars would put them on
    incomparable scales. Plotting each as a ratio to the best model on that
    metric puts every panel on "1.0 = best" and makes DISAGREEMENT between
    metrics the visible feature -- which is the point of the figure.
    """
    apply_style()
    available = [m for m in metrics if m in scores.columns]
    if not available:
        raise ValueError(f"none of {list(metrics)} present in scores")

    df = scores[[model_col, *available]].dropna(subset=available, how="all").copy()
    df = df.sort_values(available[0])

    for m in available:
        best = df[m].min()
        df[f"{m}_rel"] = df[m] / best if best and np.isfinite(best) else np.nan

    x = np.arange(len(df))
    width = 0.8 / len(available)

    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(df)), 5))
    for i, m in enumerate(available):
        ax.bar(
            x + i * width - 0.4 + width / 2, df[f"{m}_rel"], width=width, label=m.upper(), alpha=0.9
        )

    ax.axhline(1.0, color="#c1444a", ls="--", lw=1.2, label="best on metric")
    ax.set_xticks(x)
    ax.set_xticklabels(df[model_col].astype(str), rotation=42, ha="right")
    ax.set_ylabel("Score relative to best (1.0 = best)")
    ax.set_title("Fig 2 — Metric agreement and disagreement")
    ax.legend(fontsize=8, ncol=len(available) + 1)

    return save_figure(
        fig,
        out_dir,
        name,
        caption=(
            "Each metric normalised to its own best model, so 1.0 is the best "
            "score on that metric. Where a model's bars differ in height across "
            "metrics, the metrics disagree about it -- MASE is less sensitive to "
            "a few large misses than RMSSE, so a model better on MASE and worse "
            "on RMSSE is winning on typical days and losing on spikes."
        ),
    )


# ---------------------------------------------------------------------------
# Fig 3 -- per-series error distribution
# ---------------------------------------------------------------------------


def plot_error_distribution(
    per_series: pd.DataFrame,
    out_dir: Path | str,
    metric_label: str = "WAPE",
    max_value: float | None = 300.0,
    name: str = "fig03_error_distribution",
) -> Path:
    """
    Boxplots of the PER-SERIES metric, one box per model.

    A leaderboard reports a single pooled number per model. Two models can
    share it while one is consistently mediocre and the other is excellent on
    most series and catastrophic on a few. In a replenishment setting those are
    very different risks, and only the distribution shows it.

    Args:
        per_series: (series x model) matrix, as produced by
            `evaluation.significance.per_series_scores`.
        max_value: y-axis clip. Per-series WAPE is unbounded above -- a series
            selling 1 unit in the window and forecast 4 scores 300% -- so a few
            such series would otherwise compress every box into a line. The clip
            is stated in the caption rather than applied silently.
    """
    apply_style()
    order = per_series.median().sort_values().index.tolist()
    data = [per_series[m].dropna().to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(order)), 5.5))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, model in zip(bp["boxes"], order):
        patch.set_facecolor(color_for(model))
        patch.set_alpha(0.78)

    # Mean marker: with a right-skewed distribution the mean sits well above
    # the median, and the gap is exactly the tail risk being described.
    means = [np.nanmean(d) for d in data]
    ax.scatter(
        range(1, len(order) + 1),
        means,
        marker="D",
        s=26,
        color="white",
        edgecolor="black",
        zorder=5,
        label="mean",
    )

    ax.set_xticklabels(order, rotation=42, ha="right")
    ax.set_ylabel(f"Per-series {metric_label}")
    if max_value is not None:
        ax.set_ylim(0, max_value)
    ax.set_title(f"Fig 3 — Per-series {metric_label} distribution")
    ax.legend(fontsize=8)

    clip_note = (
        f" Y-axis clipped at {max_value:.0f}; per-series {metric_label} is "
        f"unbounded above for near-zero series."
        if max_value
        else ""
    )
    return save_figure(
        fig,
        out_dir,
        name,
        caption=(
            f"Distribution of {metric_label} across individual series, not the "
            f"pooled score. Boxes show quartiles, diamonds the mean. A large "
            f"mean-median gap indicates right skew: the model is fine on most "
            f"series and badly wrong on a few.{clip_note} Outliers hidden for "
            f"legibility."
        ),
    )


# ---------------------------------------------------------------------------
# Fig 4 -- Critical Difference diagram
# ---------------------------------------------------------------------------


def plot_critical_difference(
    average_ranks: pd.Series,
    critical_difference: float,
    out_dir: Path | str,
    cliques: Sequence[Sequence[str]] | None = None,
    n_series: int | None = None,
    ax: plt.Axes | None = None,
    name: str = "fig04_critical_difference",
) -> Path | plt.Axes:
    """
    Demsar critical-difference diagram.

    Models are placed on a rank axis (best on the left). Any set whose ranks
    all fall within CD of each other is joined by a horizontal bar, meaning
    they are statistically indistinguishable. This is the most rigorous
    multi-model comparison figure available and very few portfolio projects
    include one.

    Read the CD value as well as the bars: CD shrinks as sqrt(1/N), so at large
    N almost any rank gap clears it. At N = 2,950 here, CD = 0.249 -- a
    caveat that belongs beside the figure, not buried in a methods section.

    Pass `ax` to draw into an existing axes (used by `plot_significance_pair`).
    """
    apply_style()
    ranks = average_ranks.sort_values()
    names = list(ranks.index)
    values = ranks.to_numpy(dtype=float)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, max(3.2, 0.42 * len(names) + 1.8)))
    else:
        fig = ax.figure

    # Rank axis spans the data; the plotted x-range is wider so the model
    # labels sit INSIDE the axes. Drawing them outside (at the data limits)
    # makes them collide with a neighbouring subplot in the paired figure.
    rank_lo, rank_hi = float(values.min()), float(values.max())
    rank_span = (rank_hi - rank_lo) or critical_difference or 1.0
    gutter = rank_span * 0.85  # room for the longest label
    lo, hi = rank_lo - gutter, rank_hi + gutter

    n_left = int(np.ceil(len(names) / 2))  # left half gets the better ranks
    rows = max(n_left, len(names) - n_left)

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, rows + 2.6)

    # A CD diagram has exactly ONE axis: the rank scale drawn manually below.
    # Matplotlib's own axes would show the padded range including the label
    # gutters, which is a wider span than the ranks actually cover and reads as
    # a contradictory second scale.
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    axis_y = rows + 1.4
    ax.hlines(axis_y, rank_lo, rank_hi, color="black", lw=1.2)
    for tick in np.linspace(rank_lo, rank_hi, 5):
        ax.vlines(tick, axis_y, axis_y + 0.14, color="black", lw=1.0)
        ax.text(tick, axis_y + 0.26, f"{tick:.2f}", ha="center", fontsize=8)

    # Model stems. Better-ranked half exits left, worse half exits right, each
    # stepping DOWN independently so no two labels share a row.
    for i, (nm, val) in enumerate(zip(names, values)):
        left = i < n_left
        depth = i if left else i - n_left
        y = axis_y - 0.85 - depth * 0.92
        edge = rank_lo - gutter * 0.06 if left else rank_hi + gutter * 0.06

        ax.vlines(val, y, axis_y, color=color_for(nm), lw=1.5)
        ax.hlines(y, val, edge, color=color_for(nm), lw=1.5)
        ax.text(
            edge + (-gutter * 0.04 if left else gutter * 0.04),
            y,
            f"{nm} ({val:.2f})",
            ha="right" if left else "left",
            va="center",
            fontsize=8.5,
        )

    # CD ruler, above the rank axis.
    ax.hlines(axis_y + 0.72, rank_lo, rank_lo + critical_difference, color="black", lw=2.2)
    ax.vlines(
        [rank_lo, rank_lo + critical_difference],
        axis_y + 0.63,
        axis_y + 0.81,
        color="black",
        lw=2.2,
    )
    ax.text(
        rank_lo + critical_difference / 2,
        axis_y + 0.92,
        f"CD = {critical_difference:.3f}",
        ha="left",
        fontsize=9,
        weight="bold",
    )

    # Clique bars, stacked just below the rank axis so they read as one group.
    for depth, clique in enumerate(cliques or []):
        members = [m for m in clique if m in ranks.index]
        if len(members) < 2:
            continue
        vals = [float(ranks[m]) for m in members]
        y = axis_y - 0.24 - depth * 0.17
        ax.hlines(y, min(vals), max(vals), color="black", lw=3.6, alpha=0.8)

    ax.text(
        (rank_lo + rank_hi) / 2,
        axis_y + 0.52,
        "Average rank  (lower is better)",
        ha="center",
        fontsize=9,
        color="#444",
    )
    ax.set_title("Fig 4 — Critical Difference diagram (Nemenyi)", pad=18)

    if not standalone:
        return ax

    n_note = f" N = {n_series:,} series." if n_series else ""
    return save_figure(
        fig,
        out_dir,
        name,
        caption=(
            f"Average rank per model; models joined by a heavy horizontal bar are "
            f"statistically indistinguishable at alpha=0.05 (Nemenyi post-hoc, "
            f"CD = {critical_difference:.3f}).{n_note} CD shrinks as sqrt(1/N), so "
            f"at this sample size nearly any rank gap clears it -- statistical "
            f"separability here is not the same as a difference worth acting on. "
            f"The test also assumes the compared forecasts are independent, which "
            f"is violated for series nested in a hierarchy."
        ),
    )


# ---------------------------------------------------------------------------
# Fig 5 -- Multiple Comparisons with the Best
# ---------------------------------------------------------------------------


def plot_mcb(
    mcb: pd.DataFrame,
    out_dir: Path | str,
    ax: plt.Axes | None = None,
    name: str = "fig05_mcb",
) -> Path | plt.Axes:
    """
    MCB intervals: every model against the best, the question a reader asks.

    Each model gets its average rank and a CD/2 interval. A model whose
    interval does not reach the best model's upper bound is significantly
    worse. This is the comparison plot used in the official M5 paper.
    """
    apply_style()
    df = mcb.sort_values("average_rank").reset_index(drop=True)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8.5, max(4, 0.40 * len(df) + 1)))
    else:
        fig = ax.figure

    y = np.arange(len(df))
    best_upper = float(df["ci_upper"].iloc[0])

    for i, row in df.iterrows():
        indistinguishable = bool(row["indistinguishable_from_best"])
        color = color_for(str(row["model"]))
        ax.hlines(
            i,
            row["ci_lower"],
            row["ci_upper"],
            color=color,
            lw=2.6,
            alpha=1.0 if indistinguishable else 0.45,
        )
        ax.plot(
            row["average_rank"],
            i,
            "o",
            color=color,
            markersize=7,
            markeredgecolor="black" if row["is_best"] else "none",
            markeredgewidth=1.2,
            zorder=5,
        )

    ax.axvline(best_upper, color="#c1444a", ls="--", lw=1.3, label="upper bound of best model")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['model']}{'  ★' if r['is_best'] else ''}" for _, r in df.iterrows()],
        fontsize=8.5,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Average rank  (lower is better)")
    ax.set_title("Fig 5 — Multiple Comparisons with the Best")
    ax.legend(fontsize=8, loc="lower right")

    if not standalone:
        return ax

    n_tied = int(df["indistinguishable_from_best"].sum()) - 1
    return save_figure(
        fig,
        out_dir,
        name,
        caption=(
            f"Average rank with a CD/2 interval per model. Models whose interval "
            f"crosses the dashed line are not significantly worse than the best "
            f"(★); {n_tied} model(s) qualify here. Faded intervals are "
            f"significantly worse. Compare against Fig 4: MCB ranks series "
            f"equally, while a pooled loss metric weights by volume, and on M5 "
            f"the two disagree about the leader."
        ),
    )


# ---------------------------------------------------------------------------
# Figs 4 + 5 side by side -- the headline figure
# ---------------------------------------------------------------------------


def plot_significance_pair(
    average_ranks: pd.Series,
    critical_difference: float,
    mcb: pd.DataFrame,
    out_dir: Path | str,
    pooled_leader: str | None = None,
    rank_leader: str | None = None,
    cliques: Sequence[Sequence[str]] | None = None,
    n_series: int | None = None,
    name: str = "fig04_05_significance_pair",
) -> Path:
    """
    Figs 4 and 5 in one frame, captioned on their disagreement.

    The spec requires these side by side, and the reason is substantive rather
    than cosmetic. A pooled per-observation loss (Diebold-Mariano) is
    volume-weighted and is dominated by the ~5% of M5 series carrying most
    units; per-series ranks (Nemenyi, MCB) weight every series equally and are
    dominated by the 95% that are sparse. They can therefore name different
    winners while both being correct.

    Makridakis et al. (2022) observed exactly this inside the competition: the
    M5 winner ranked 1st on WRMSSE but 13th on MCB, and another entry ranked
    8th on WRMSSE and 1st on MCB. Their reading was that winning teams
    optimised for expensive fast-moving products at the expense of the rest.
    """
    apply_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(19, max(5.5, 0.46 * len(average_ranks) + 2.4)),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    plot_critical_difference(
        average_ranks, critical_difference, out_dir, cliques=cliques, ax=axes[0]
    )
    plot_mcb(mcb, out_dir, ax=axes[1])

    disagreement = ""
    if pooled_leader and rank_leader and pooled_leader != rank_leader:
        disagreement = (
            f" On this data they disagree: the volume-weighted pooled test names "
            f"{pooled_leader}, while equal-weighted per-series ranks name "
            f"{rank_leader}."
        )
        fig.suptitle(
            f"Rank-based significance — pooled leader: {pooled_leader}   |   "
            f"rank leader: {rank_leader}",
            fontsize=12.5,
            y=1.005,
        )

    n_note = f" N = {n_series:,} series." if n_series else ""
    return save_figure(
        fig,
        out_dir,
        name,
        caption=(
            f"Left: Nemenyi critical-difference diagram. Right: Multiple "
            f"Comparisons with the Best. Both are rank-based and equal-weight "
            f"every series, unlike a pooled loss metric which weights by "
            f"volume.{disagreement} This is a property of M5's structure, not a "
            f"defect: Makridakis et al. (2022) found the same divergence inside "
            f"the competition, where the winner ranked 1st on WRMSSE but 13th on "
            f"MCB.{n_note} Two caveats apply to both panels: CD = "
            f"{critical_difference:.3f} is small enough at this N that nearly any "
            f"rank gap clears it, and the independence assumption is violated for "
            f"hierarchically nested series."
        ),
    )
