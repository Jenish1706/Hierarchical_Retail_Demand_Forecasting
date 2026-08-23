"""
Pillar 5, Group B/C -- segment and temporal figures.

Fig 6   Model x demand-class heatmap
Fig 7   Model x hierarchy-level heatmap
Fig 8   Model x category grouped bars
Fig 11  Horizon degradation curves

Fig 6 is the one that decides a design question rather than merely describing
results. The router sends INTERMITTENT series to Croston and LUMPY series to
ADIDA on the strength of a literature rule; this heatmap is the direct test of
whether that rule holds on M5. If a global learner wins inside the sparse
quadrants, the routing strategy is wrong and the figure says so.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.logging import get_logger
from src.config.settings import M5
from src.visualization.style import apply_style, color_for, save_figure

log = get_logger(__name__)

# Ordered sparsest-last so the heatmap reads left-to-right as increasing
# intermittency, matching how the SBT quadrants are usually discussed.
DEMAND_CLASS_ORDER = ("SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY")


def _wape(g: pd.DataFrame) -> float:
    denom = np.abs(g["y_true"]).sum()
    if denom <= 0:
        return np.nan
    return float(100.0 * np.abs(g["y_true"] - g["y_pred"]).sum() / denom)


def _annotated_heatmap(
    matrix: pd.DataFrame,
    ax: plt.Axes,
    cbar_label: str,
    highlight_best_per_column: bool = True,
) -> None:
    """
    Heatmap with per-cell values, normalised WITHIN each column.

    Column-wise normalisation is the important detail. Absolute WAPE differs by
    an order of magnitude between segments -- sparse series score 90%+ while
    dense ones score 30% -- so a globally-normalised heatmap would just
    re-display "which segment is hard" in every row. Normalising per column
    makes the colour answer the question actually being asked: within THIS
    segment, which model is best?
    """
    values = matrix.to_numpy(dtype=float)

    # Column-wise min-max to [0, 1]; a column with no spread renders neutral.
    col_min = np.nanmin(values, axis=0)
    col_max = np.nanmax(values, axis=0)
    span = np.where((col_max - col_min) > 0, col_max - col_min, np.nan)
    normed = (values - col_min) / span

    im = ax.imshow(normed, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.grid(False)

    best_row = np.nanargmin(values, axis=0) if highlight_best_per_column else None

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999")
                continue
            is_best = best_row is not None and best_row[j] == i
            ax.text(
                j, i, f"{v:.1f}",
                ha="center", va="center", fontsize=8.5,
                weight="bold" if is_best else "normal",
                color="black",
            )
            if is_best:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor="black", lw=2.0)
                )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Fig 6 -- model x demand class
# ---------------------------------------------------------------------------

def plot_model_by_demand_class(
    predictions: pd.DataFrame,
    classification: pd.DataFrame,
    out_dir: Path | str,
    name: str = "fig06_model_by_demand_class",
) -> Path:
    """
    WAPE for every model within each SBT demand class.

    Args:
        predictions: long frame with model, series_id, y_true, y_pred.
        classification: series_id -> pattern, from `reports/eda/`.
    """
    apply_style()

    df = predictions.merge(
        classification[["series_id", "pattern"]], on="series_id", how="inner"
    )
    if df.empty:
        raise ValueError("no overlap between predictions and the SBT classification")

    rows = [
        {"model": str(model), "pattern": str(pattern), "wape": _wape(g)}
        for (model, pattern), g in df.groupby(["model", "pattern"], observed=True)
    ]
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="model", columns="pattern", values="wape")

    present = [c for c in DEMAND_CLASS_ORDER if c in matrix.columns]
    matrix = matrix[present]
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]

    counts = (
        classification[classification["series_id"].isin(df["series_id"].unique())]
        .groupby("pattern", observed=True).size()
    )

    fig, ax = plt.subplots(figsize=(1.9 * len(present) + 5.5, 0.42 * len(matrix) + 3))
    _annotated_heatmap(matrix, ax, "Relative WAPE within class\n(green = best)")

    ax.set_xticklabels(
        [f"{c}\n(n={counts.get(c, 0):,})" for c in present], rotation=0, fontsize=9
    )
    ax.set_title("Fig 6 — Model accuracy by demand class (WAPE)")
    ax.set_xlabel("")

    # Which model actually wins each sparse quadrant -- the routing test.
    winners = {c: str(matrix[c].idxmin()) for c in present if matrix[c].notna().any()}
    sparse = [c for c in ("INTERMITTENT", "LUMPY") if c in winners]
    routing_note = ""
    if sparse:
        routing_note = " Winners in the sparse quadrants: " + ", ".join(
            f"{c} -> {winners[c]}" for c in sparse
        ) + "."

    return save_figure(
        fig, out_dir, name,
        caption=(
            f"WAPE per model within each Syntetos-Boylan-Teunter demand class. "
            f"Colour is normalised WITHIN each column, because absolute WAPE "
            f"differs by roughly an order of magnitude between dense and sparse "
            f"segments; a globally-scaled heatmap would only restate which "
            f"segment is hard. Boxed cells are the best model in that class."
            f"{routing_note} This figure is the direct test of the routing rule: "
            f"the router sends INTERMITTENT to Croston and LUMPY to ADIDA on the "
            f"strength of a literature convention, and if a global learner wins "
            f"those columns instead, the rule does not hold on M5."
        ),
    )


# ---------------------------------------------------------------------------
# Fig 7 -- model x hierarchy level
# ---------------------------------------------------------------------------

def plot_model_by_level(
    by_level: pd.DataFrame,
    out_dir: Path | str,
    metric: str = "wape",
    name: str = "fig07_model_by_level",
) -> Path:
    """
    Model accuracy across aggregation levels.

    Accuracy should improve markedly with aggregation, because summing series
    cancels idiosyncratic error. The interesting question is which models
    benefit MOST -- a model that is mediocre at item level but excellent at
    store level is exactly what a replenishment planner buying at store level
    should use.
    """
    apply_style()
    matrix = by_level.pivot_table(index="model", columns="level", values=metric,
                                  observed=True)

    order = [lvl for lvl in M5.HIERARCHY_LEVELS if lvl in matrix.columns]
    matrix = matrix[order]
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(1.55 * len(order) + 5.5, 0.42 * len(matrix) + 3))
    _annotated_heatmap(matrix, ax, f"Relative {metric.upper()} within level")
    ax.set_title(f"Fig 7 — Model accuracy by hierarchy level ({metric.upper()})")
    ax.set_xlabel("Aggregation level (bottom-up)")

    improvement = ""
    if {"item_store", "total"} <= set(matrix.columns):
        lo = float(matrix["item_store"].min())
        hi = float(matrix["total"].min())
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
            improvement = (
                f" Best {metric.upper()} improves from {lo:.1f} at item-store to "
                f"{hi:.1f} at total, a {100 * (lo - hi) / lo:.0f}% reduction from "
                f"aggregation alone."
            )

    return save_figure(
        fig, out_dir, name,
        caption=(
            f"{metric.upper()} per model at each of the six aggregation levels, "
            f"forecasts summed bottom-up. Colour normalised within each column so "
            f"the comparison is between models at a level, not between levels."
            f"{improvement} Error cancels as series are aggregated, so a model's "
            f"rank can change with the level a decision is actually taken at."
        ),
    )


# ---------------------------------------------------------------------------
# Fig 8 -- model x category
# ---------------------------------------------------------------------------

def plot_model_by_category(
    predictions: pd.DataFrame,
    out_dir: Path | str,
    name: str = "fig08_model_by_category",
) -> Path:
    """Grouped bars: model accuracy across FOODS, HOBBIES, HOUSEHOLD."""
    apply_style()
    if "cat_id" not in predictions.columns:
        raise KeyError("predictions must carry cat_id")

    rows = [
        {"model": str(model), "cat_id": str(cat), "wape": _wape(g)}
        for (model, cat), g in predictions.groupby(["model", "cat_id"], observed=True)
    ]
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="model", columns="cat_id", values="wape")
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]

    cats = list(matrix.columns)
    x = np.arange(len(matrix))
    width = 0.8 / len(cats)

    fig, ax = plt.subplots(figsize=(max(9, 0.72 * len(matrix)), 5))
    for i, cat in enumerate(cats):
        ax.bar(x + i * width - 0.4 + width / 2, matrix[cat], width=width, label=cat)

    ax.set_xticks(x)
    ax.set_xticklabels(matrix.index, rotation=40, ha="right")
    ax.set_ylabel("WAPE")
    ax.set_title("Fig 8 — Model accuracy by product category")
    ax.legend(fontsize=8, title="Category")

    return save_figure(
        fig, out_dir, name,
        caption=(
            "WAPE per model within each product category. HOBBIES is the sparsest "
            "category and the hardest for every model; FOODS is the densest and "
            "the easiest. A model whose relative standing changes between "
            "categories is responding to sparsity rather than to category as such."
        ),
    )


# ---------------------------------------------------------------------------
# Fig 11 -- horizon degradation
# ---------------------------------------------------------------------------

def plot_horizon_degradation(
    predictions: pd.DataFrame,
    out_dir: Path | str,
    top_n: int | None = 8,
    name: str = "fig11_horizon_degradation",
) -> Path:
    """
    WAPE by forecast day, 1 through h.

    Determines which model suits which planning horizon. A flat curve forecasts
    day 28 about as well as day 1; a steeply rising one is only usable for
    short-term replenishment. A single pooled number over the whole window
    hides that distinction entirely.

    Per-series methods that emit a constant rate (Croston, ADIDA, SES) should be
    flat by construction -- their forecast does not vary with h. Curves that
    rise are the ones carrying genuine short-horizon information, and the gap
    between the two families is the figure's real content.
    """
    apply_style()
    df = predictions.copy()

    if "horizon_day" not in df.columns:
        # Derive day-within-window from the date rank inside each (model,
        # origin, series) block.
        group_cols = [c for c in ("model", "origin", "series_id") if c in df.columns]
        df["horizon_day"] = df.groupby(group_cols, observed=True)["date"].rank(
            method="dense"
        ).astype(int)

    rows = [
        {"model": str(model), "horizon_day": int(day), "wape": _wape(g)}
        for (model, day), g in df.groupby(["model", "horizon_day"], observed=True)
    ]
    curves = pd.DataFrame(rows)

    keep = (
        curves.groupby("model")["wape"].mean().sort_values().head(top_n).index
        if top_n else curves["model"].unique()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in keep:
        sub = curves[curves["model"] == model].sort_values("horizon_day")
        ax.plot(sub["horizon_day"], sub["wape"], marker="o", markersize=3.2,
                lw=1.7, label=str(model), color=color_for(str(model)))

    ax.set_xlabel("Forecast day ahead")
    ax.set_ylabel("WAPE")
    ax.set_title("Fig 11 — Accuracy degradation across the forecast horizon")
    ax.legend(fontsize=8, ncol=2)

    # Quantify the slope so the caption states a number, not an impression.
    slopes = {}
    for model in keep:
        sub = curves[curves["model"] == model].sort_values("horizon_day")
        if len(sub) > 2:
            slopes[str(model)] = float(
                np.polyfit(sub["horizon_day"], sub["wape"], 1)[0]
            )
    flattest = min(slopes, key=slopes.get) if slopes else None
    steepest = max(slopes, key=slopes.get) if slopes else None

    note = ""
    if flattest and steepest and flattest != steepest:
        note = (
            f" Flattest: {flattest} ({slopes[flattest]:+.3f} WAPE/day). "
            f"Steepest: {steepest} ({slopes[steepest]:+.3f} WAPE/day)."
        )

    return save_figure(
        fig, out_dir, name,
        caption=(
            f"WAPE at each day of the forecast horizon, top {len(keep)} models by "
            f"mean accuracy.{note} Methods that emit a constant demand rate "
            f"(Croston, ADIDA, SES) are flat by construction, since their forecast "
            f"does not depend on how far ahead it is made. A rising curve indicates "
            f"a model exploiting short-horizon information that decays -- useful "
            f"for next-week replenishment, less so at four weeks."
        ),
    )
