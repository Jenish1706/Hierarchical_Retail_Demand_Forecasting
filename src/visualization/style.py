"""
Shared plotting style for Pillar 5.

One module owns colour, typography and figure conventions so twenty figures
read as one document rather than twenty. Tier colouring in particular is
load-bearing: the central question of this study is which model FAMILY wins,
so family must be visible at a glance in every figure that lists models.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; no display required

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------------------
# Tier palette
#
# Chosen to stay distinguishable in greyscale print and for the most common
# colour-vision deficiencies: the five hues differ in lightness as well as hue,
# so a red/green confusion does not collapse two tiers into one.
# ---------------------------------------------------------------------------
TIER_COLORS: dict[str, str] = {
    "naive": "#9aa0a6",  # grey -- the reference floor
    "statistical": "#2a7fb8",  # blue
    "intermittent": "#7a5ea8",  # purple
    "classical_ml": "#e07b39",  # orange
    "probabilistic": "#3f9e8c",  # teal
    "deep_learning": "#c1444a",  # red
    "ensemble": "#4c9a52",  # green
    "other": "#666666",
}

CATEGORY_COLORS: dict[str, str] = {
    "FOODS": "#2a7fb8",
    "HOBBIES": "#e07b39",
    "HOUSEHOLD": "#4c9a52",
}

DEMAND_CLASS_COLORS: dict[str, str] = {
    "SMOOTH": "#2a7fb8",
    "ERRATIC": "#e07b39",
    "INTERMITTENT": "#7a5ea8",
    "LUMPY": "#c1444a",
    "NO_DEMAND": "#9aa0a6",
}

# Model name -> tier. Prefix matching, longest first, so `lightgbm_tweedie`
# resolves before `lightgbm` and `sf_seasonal_naive` before `sf_`.
_TIER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("routed_ensemble", "ensemble"),
    ("ensemble_", "ensemble"),
    ("croston", "intermittent"),
    ("adida", "intermittent"),
    ("imapa", "intermittent"),
    ("tsb", "intermittent"),
    ("lightgbm", "classical_ml"),
    ("xgboost", "classical_ml"),
    ("catboost", "classical_ml"),
    ("random_forest", "classical_ml"),
    ("extra_trees", "classical_ml"),
    ("ridge", "classical_ml"),
    ("naive", "naive"),
    ("sf_naive", "naive"),
    ("seasonal_naive", "naive"),
    ("sf_seasonal", "naive"),
    ("moving_average", "naive"),
    ("sf_window", "naive"),
    ("autoets", "statistical"),
    ("ets_", "statistical"),
    ("ses", "statistical"),
    ("holt", "statistical"),
    ("theta", "statistical"),
    ("autoces", "statistical"),
    ("arima", "statistical"),
    ("mstl", "statistical"),
    ("quantile", "probabilistic"),
    ("conformal", "probabilistic"),
    ("ngboost", "probabilistic"),
    ("deepar", "deep_learning"),
    ("nbeats", "deep_learning"),
    ("nhits", "deep_learning"),
    ("dlinear", "deep_learning"),
    ("patchtst", "deep_learning"),
    ("tide", "deep_learning"),
    ("lstm", "deep_learning"),
    ("gru", "deep_learning"),
    ("tcn", "deep_learning"),
    ("mlp", "deep_learning"),
)


def tier_of(model: str) -> str:
    """Map a model name to its tier for colouring and grouping."""
    name = str(model).lower()
    # Longest pattern first so specific names beat generic prefixes.
    for pattern, tier in sorted(_TIER_PATTERNS, key=lambda kv: -len(kv[0])):
        if pattern in name:
            return tier
    return "other"


def color_for(model: str) -> str:
    return TIER_COLORS[tier_of(model)]


def apply_style() -> None:
    """Set the global matplotlib/seaborn defaults for every Pillar 5 figure."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.alpha": 0.3,
            "font.family": "DejaVu Sans",
        }
    )


def tier_legend_handles(tiers: Iterable[str] | None = None) -> list[Any]:
    """Proxy artists for a tier legend, for figures that colour bars by tier."""
    from matplotlib.patches import Patch

    keys = list(tiers) if tiers is not None else list(TIER_COLORS)
    return [
        Patch(facecolor=TIER_COLORS[t], label=t.replace("_", " ").title())
        for t in keys
        if t in TIER_COLORS
    ]


def save_figure(
    fig: plt.Figure, out_dir: Path | str, name: str, caption: str | None = None
) -> Path:
    """
    Write a figure to `out_dir/name.png`, plus its caption as a sidecar `.md`.

    Captions live beside the image rather than only in the report so a figure
    is never separated from the claim it supports -- spec constraint 9 requires
    every claim to have a figure, and the inverse discipline keeps a figure
    from floating free of its interpretation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)

    if caption:
        (out_dir / f"{name}.md").write_text(f"# {name}\n\n{caption.strip()}\n", encoding="utf-8")
    return path
