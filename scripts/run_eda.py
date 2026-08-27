"""
Pillar 1: Exploratory Data Analysis over all 30,490 M5 series.

Produces `reports/eda/`: figures, CSV tables, and a JSON findings file that
later phases consume. Two sections feed directly into modelling decisions and
are therefore not optional colour:

  * §3.3 intermittency -- the SBT classification decides which series the
    router sends to Croston/ADIDA rather than to a global learner.
  * §3.8 stationarity and distribution fitting -- decides the differencing
    orders for ARIMA and, more importantly, which likelihood Tier 3 should
    use. Poisson, Negative Binomial and Tweedie are TESTED here rather than
    assumed, because the wrong likelihood on a zero-inflated count target is
    the difference between a calibrated forecast and a useless one.

Runs on the full panel by default. The wide->long melt of 30,490 x 1,913 is
58.3M rows, so per-series statistics are computed on the WIDE matrix with
NumPy along axis 1 -- roughly 60x faster than a pandas groupby over the melted
frame and a fraction of the memory.

Usage:
    python scripts/run_eda.py
    python scripts/run_eda.py --sample-series 5000     # faster iteration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config.logging import get_logger  # noqa: E402
from src.config.settings import M5, settings  # noqa: E402
from src.models.statistical.intermittent import (  # noqa: E402
    ADI_CUTOFF,
    CV2_CUTOFF,
    classify_demand,
)

log = get_logger("eda")

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"FOODS": "#2a7fb8", "HOBBIES": "#e07b39", "HOUSEHOLD": "#4c9a52"}
CLASS_COLORS = {
    "SMOOTH": "#2a7fb8",
    "ERRATIC": "#e07b39",
    "INTERMITTENT": "#7a5ea8",
    "LUMPY": "#c1444a",
    "NO_DEMAND": "#999999",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("figure_saved", name=name)
    return path


# ===========================================================================
# 3.1 Structure
# ===========================================================================


def section_structure(wide: pd.DataFrame, cal: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Cardinality at every hierarchy level; series-length and lifecycle checks."""
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    sales = wide[day_cols].to_numpy(dtype=np.int32)

    levels = {
        "total": 1,
        "state": wide["state_id"].nunique(),
        "store": wide["store_id"].nunique(),
        "category": wide["cat_id"].nunique(),
        "department": wide["dept_id"].nunique(),
        "item": wide["item_id"].nunique(),
        "cat_store": wide.groupby(["cat_id", "store_id"], observed=True).ngroups,
        "dept_store": wide.groupby(["dept_id", "store_id"], observed=True).ngroups,
        "item_store": len(wide),
    }

    # Lifecycle: M5 items are introduced over time, so a leading run of zeros
    # is "not yet stocked", not "no demand". Distinguishing the two matters --
    # training on the pre-introduction zeros teaches the model that the item
    # sells nothing.
    nonzero = sales > 0
    has_any = nonzero.any(axis=1)
    first_sale = np.where(has_any, nonzero.argmax(axis=1), -1)
    # argmax on the reversed array gives the last True position.
    last_sale = np.where(has_any, sales.shape[1] - 1 - nonzero[:, ::-1].argmax(axis=1), -1)

    active_len = np.where(has_any, last_sale - first_sale + 1, 0)

    findings = {
        "n_series": int(len(wide)),
        "n_days": len(day_cols),
        "long_format_rows": int(len(wide) * len(day_cols)),
        "wide_memory_mb": round(wide.memory_usage(deep=True).sum() / 1e6, 1),
        "hierarchy_cardinality": {k: int(v) for k, v in levels.items()},
        "total_units_sold": int(sales.sum()),
        "series_never_sold": int((~has_any).sum()),
        "median_first_sale_day": int(np.median(first_sale[has_any])) + 1,
        "series_starting_after_day_1": int((first_sale > 0).sum()),
        "series_starting_after_day_365": int((first_sale > 365).sum()),
        "series_with_no_sale_in_final_28d": int((sales[:, -28:].sum(axis=1) == 0).sum()),
        "median_active_length_days": int(np.median(active_len[has_any])),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(first_sale[has_any] + 1, bins=60, color="#2a7fb8", edgecolor="white")
    axes[0].set(
        xlabel="Day index of first sale", ylabel="Series", title="Item introduction over time"
    )
    axes[0].axvline(
        1,
        color="#c1444a",
        ls="--",
        lw=1,
        label=f"{int((first_sale == 0).sum()):,} present from day 1",
    )
    axes[0].legend(fontsize=8)

    axes[1].hist(active_len[has_any], bins=60, color="#4c9a52", edgecolor="white")
    axes[1].set(
        xlabel="Active length (days, first sale to last)",
        ylabel="Series",
        title="Series active lifespan",
    )
    fig.suptitle("§3.1 Structure: series lifecycle", y=1.02, fontsize=12)
    _save(fig, out, "fig_3_1_lifecycle")

    return findings


# ===========================================================================
# 3.2 Demand distribution
# ===========================================================================


def section_demand(wide: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Zero-inflation and volume distribution, overall and by segment."""
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    sales = wide[day_cols].to_numpy(dtype=np.int32)

    flat = sales.ravel()
    zero_rate = float((flat == 0).mean())

    series_mean = sales.mean(axis=1)
    series_var = sales.var(axis=1)

    # Overdispersion: for a Poisson process variance == mean. Retail demand is
    # almost always overdispersed, which is the empirical case for Negative
    # Binomial over Poisson in Tier 3.
    with np.errstate(divide="ignore", invalid="ignore"):
        vmr = np.where(series_mean > 0, series_var / series_mean, np.nan)

    by_cat = (
        wide.assign(_zero=(sales == 0).mean(axis=1), _mean=series_mean)
        .groupby("cat_id", observed=True)
        .agg(
            n_series=("_zero", "size"),
            zero_rate=("_zero", "mean"),
            mean_daily=("_mean", "mean"),
            total_units=("_mean", "sum"),
        )
        .reset_index()
    )
    by_store = (
        wide.assign(_zero=(sales == 0).mean(axis=1), _mean=series_mean)
        .groupby("store_id", observed=True)
        .agg(n_series=("_zero", "size"), zero_rate=("_zero", "mean"), mean_daily=("_mean", "mean"))
        .reset_index()
    )
    by_cat.to_csv(out / "table_demand_by_category.csv", index=False)
    by_store.to_csv(out / "table_demand_by_store.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    counts = np.bincount(flat[flat <= 30])
    axes[0].bar(range(len(counts)), counts, color="#2a7fb8")
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="Units sold in a day",
        ylabel="Observations (log)",
        title=f"Daily sales distribution\n{zero_rate:.1%} of all observations are zero",
    )

    axes[1].barh(
        by_cat["cat_id"],
        by_cat["zero_rate"],
        color=[PALETTE.get(c, "#888") for c in by_cat["cat_id"]],
    )
    axes[1].set(xlabel="Zero-sales rate", title="Zero-inflation by category")
    for i, v in enumerate(by_cat["zero_rate"]):
        axes[1].text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)

    finite_vmr = vmr[np.isfinite(vmr)]
    axes[2].hist(
        np.log10(np.clip(finite_vmr, 1e-3, None)), bins=60, color="#7a5ea8", edgecolor="white"
    )
    axes[2].axvline(0, color="#c1444a", ls="--", lw=1.5, label="Poisson (var = mean)")
    axes[2].set(xlabel="log10(variance / mean)", ylabel="Series", title="Overdispersion")
    axes[2].legend(fontsize=8)

    fig.suptitle("§3.2 Demand distribution", y=1.02, fontsize=12)
    _save(fig, out, "fig_3_2_demand_distribution")

    return {
        "overall_zero_rate": round(zero_rate, 4),
        "mean_daily_units_per_series": round(float(series_mean.mean()), 4),
        "median_daily_units_per_series": round(float(np.median(series_mean)), 4),
        "pct_series_overdispersed": round(float(np.mean(finite_vmr > 1.0)), 4),
        "median_variance_mean_ratio": round(float(np.median(finite_vmr)), 4),
        "zero_rate_by_category": by_cat.set_index("cat_id")["zero_rate"].round(4).to_dict(),
        "top_decile_volume_share": round(
            float(np.sort(series_mean)[::-1][: len(series_mean) // 10].sum() / series_mean.sum()), 4
        ),
    }


# ===========================================================================
# 3.3 Intermittency -- SBT classification
# ===========================================================================


def section_intermittency(wide: pd.DataFrame, out: Path) -> dict[str, Any]:
    """
    Classify every series into the four SBT quadrants.

    Feeds the routing strategy in Tier 5 and, together with §3.8, the
    likelihood choice in Tier 3.
    """
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    sales = wide[day_cols].to_numpy(dtype=np.float64)

    records: list[dict[str, Any]] = []
    for i in range(sales.shape[0]):
        cls = classify_demand(sales[i])
        records.append(
            {
                "series_id": f"{wide['item_id'].iat[i]}_{wide['store_id'].iat[i]}",
                "item_id": wide["item_id"].iat[i],
                "store_id": wide["store_id"].iat[i],
                "cat_id": wide["cat_id"].iat[i],
                "dept_id": wide["dept_id"].iat[i],
                "state_id": wide["state_id"].iat[i],
                "pattern": cls.pattern.value,
                "adi": cls.adi,
                "cv2": cls.cv2,
                "zero_fraction": cls.zero_fraction,
                "mean_demand": cls.mean_demand,
                "total_units": float(sales[i].sum()),
            }
        )

    cls_df = pd.DataFrame(records)
    cls_df.to_parquet(out / "series_classification.parquet", index=False)

    summary = (
        cls_df.groupby("pattern", observed=True)
        .agg(
            n_series=("series_id", "size"),
            total_units=("total_units", "sum"),
            median_adi=("adi", "median"),
            median_cv2=("cv2", "median"),
            mean_zero_fraction=("zero_fraction", "mean"),
        )
        .reset_index()
    )
    summary["pct_of_series"] = 100 * summary["n_series"] / summary["n_series"].sum()
    summary["pct_of_volume"] = 100 * summary["total_units"] / summary["total_units"].sum()
    summary = summary.sort_values("n_series", ascending=False)
    summary.to_csv(out / "table_sbt_classification.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Quadrant scatter with the SBT cut-offs drawn in.
    finite = cls_df[np.isfinite(cls_df["adi"]) & np.isfinite(cls_df["cv2"])]
    sample = finite.sample(n=min(12000, len(finite)), random_state=0)
    for pattern, g in sample.groupby("pattern", observed=True):
        axes[0].scatter(
            g["adi"],
            g["cv2"],
            s=4,
            alpha=0.35,
            color=CLASS_COLORS.get(pattern, "#888"),
            label=pattern,
        )
    axes[0].axvline(ADI_CUTOFF, color="black", ls="--", lw=1)
    axes[0].axhline(CV2_CUTOFF, color="black", ls="--", lw=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="ADI (log)",
        ylabel="CV² (log)",
        title=f"SBT quadrants (cut-offs {ADI_CUTOFF} / {CV2_CUTOFF})",
    )
    axes[0].legend(markerscale=3, fontsize=8, loc="upper left")

    # Series count vs volume share -- the key contrast.
    x = np.arange(len(summary))
    axes[1].bar(x - 0.2, summary["pct_of_series"], width=0.4, label="% of series", color="#2a7fb8")
    axes[1].bar(x + 0.2, summary["pct_of_volume"], width=0.4, label="% of volume", color="#e07b39")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["pattern"], rotation=20, ha="right")
    axes[1].set(ylabel="Percent", title="Series count vs volume share")
    axes[1].legend(fontsize=8)

    comp = pd.crosstab(cls_df["cat_id"], cls_df["pattern"], normalize="index") * 100
    comp.plot(
        kind="barh",
        stacked=True,
        ax=axes[2],
        color=[CLASS_COLORS.get(c, "#888") for c in comp.columns],
    )
    axes[2].set(xlabel="% of series", title="Pattern composition by category")
    axes[2].legend(fontsize=7, loc="lower right")

    fig.suptitle("§3.3 Intermittency: Syntetos-Boylan-Teunter classification", y=1.02, fontsize=12)
    _save(fig, out, "fig_3_3_intermittency")

    return {
        "counts": summary.set_index("pattern")["n_series"].to_dict(),
        "pct_of_series": summary.set_index("pattern")["pct_of_series"].round(2).to_dict(),
        "pct_of_volume": summary.set_index("pattern")["pct_of_volume"].round(2).to_dict(),
        "intermittent_or_lumpy_pct": round(
            float(
                summary[summary["pattern"].isin(["INTERMITTENT", "LUMPY"])]["pct_of_series"].sum()
            ),
            2,
        ),
    }


# ===========================================================================
# 3.4 Temporal patterns
# ===========================================================================


def section_temporal(wide: pd.DataFrame, cal: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Day-of-week, monthly and multi-year structure at the aggregate level."""
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    sales = wide[day_cols].to_numpy(dtype=np.int32)

    cal_idx = cal.set_index("d").loc[day_cols]
    daily_total = sales.sum(axis=0)

    ts = pd.DataFrame(
        {
            "date": pd.to_datetime(cal_idx["date"].to_numpy()),
            "total": daily_total,
            "dow": pd.to_datetime(cal_idx["date"].to_numpy()).dayofweek,
            "month": pd.to_datetime(cal_idx["date"].to_numpy()).month,
            "year": pd.to_datetime(cal_idx["date"].to_numpy()).year,
        }
    )

    # Per-category day-of-week profile, indexed so categories are comparable.
    cat_dow: dict[str, np.ndarray] = {}
    for cat in wide["cat_id"].unique():
        mask = (wide["cat_id"] == cat).to_numpy()
        series = sales[mask].sum(axis=0)
        prof = pd.Series(series).groupby(ts["dow"].to_numpy()).mean()
        cat_dow[str(cat)] = (prof / prof.mean()).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for cat, prof in cat_dow.items():
        axes[0, 0].plot(labels, prof, marker="o", label=cat, color=PALETTE.get(cat, "#888"))
    axes[0, 0].axhline(1.0, color="grey", ls=":", lw=1)
    axes[0, 0].set(
        ylabel="Demand index (1.0 = weekly mean)", title="Day-of-week profile by category"
    )
    axes[0, 0].legend(fontsize=8)

    monthly = ts.groupby("month")["total"].mean()
    axes[0, 1].bar(monthly.index, monthly / monthly.mean(), color="#4c9a52")
    axes[0, 1].axhline(1.0, color="grey", ls=":", lw=1)
    axes[0, 1].set(xlabel="Month", ylabel="Demand index", title="Monthly seasonality")

    axes[1, 0].plot(ts["date"], ts["total"], lw=0.4, color="#2a7fb8", alpha=0.7)
    axes[1, 0].plot(
        ts["date"],
        ts["total"].rolling(28, center=True).mean(),
        lw=1.8,
        color="#c1444a",
        label="28-day rolling mean",
    )
    axes[1, 0].set(ylabel="Total units/day", title="Aggregate demand, full history")
    axes[1, 0].legend(fontsize=8)

    # ACF of the aggregate series -- directly informs ARIMA order selection.
    from statsmodels.tsa.stattools import acf

    acf_vals = acf(ts["total"].to_numpy(), nlags=60, fft=True)
    axes[1, 1].stem(range(len(acf_vals)), acf_vals, basefmt=" ")
    conf = 1.96 / np.sqrt(len(ts))
    axes[1, 1].axhspan(-conf, conf, color="#c1444a", alpha=0.15, label="95% band")
    for lag in (7, 14, 21, 28):
        axes[1, 1].axvline(lag, color="#e07b39", ls=":", lw=1)
    axes[1, 1].set(
        xlabel="Lag (days)", ylabel="ACF", title="Autocorrelation (weekly spikes marked)"
    )
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("§3.4 Temporal patterns", y=1.00, fontsize=12)
    _save(fig, out, "fig_3_4_temporal")

    dow_overall = ts.groupby("dow")["total"].mean()
    dow_overall = dow_overall / dow_overall.mean()

    return {
        "dow_index": {labels[i]: round(float(v), 4) for i, v in dow_overall.items()},
        "weekend_uplift_pct": round(float(100 * (dow_overall[[5, 6]].mean() - 1)), 2),
        "acf_lag_1": round(float(acf_vals[1]), 4),
        "acf_lag_7": round(float(acf_vals[7]), 4),
        "acf_lag_28": round(float(acf_vals[28]), 4),
        "yearly_total_units": ts.groupby("year")["total"].sum().to_dict(),
    }


# ===========================================================================
# 3.5 Events and SNAP
# ===========================================================================


def section_events_snap(wide: pd.DataFrame, cal: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Descriptive event lift and SNAP contrast. Descriptive, not causal."""
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    sales = wide[day_cols].to_numpy(dtype=np.int32)
    cal_idx = cal.set_index("d").loc[day_cols].reset_index()

    rows: list[dict[str, Any]] = []
    for cat in wide["cat_id"].unique():
        mask = (wide["cat_id"] == cat).to_numpy()
        for state in M5.STATES:
            snap_col = f"snap_{state}"
            # Restrict to stores in this state so the SNAP flag applies.
            smask = mask & (wide["state_id"] == state).to_numpy()
            if not smask.any():
                continue
            s_daily = sales[smask].sum(axis=0)
            on = s_daily[cal_idx[snap_col].to_numpy() == 1].mean()
            off = s_daily[cal_idx[snap_col].to_numpy() == 0].mean()
            rows.append(
                {
                    "cat_id": str(cat),
                    "state_id": state,
                    "snap_day_mean": float(on),
                    "non_snap_day_mean": float(off),
                    "lift_pct": float(100 * (on - off) / off) if off else np.nan,
                }
            )

    snap_df = pd.DataFrame(rows)
    snap_df.to_csv(out / "table_snap_contrast.csv", index=False)

    # Event-day lift by event type, aggregate level.
    daily_total = sales.sum(axis=0)
    ev_rows = []
    baseline = daily_total[cal_idx["event_name_1"].isna().to_numpy()].mean()
    for etype in M5.EVENT_TYPES:
        sel = (cal_idx["event_type_1"] == etype).to_numpy()
        if sel.sum() == 0:
            continue
        ev_rows.append(
            {
                "event_type": etype,
                "n_days": int(sel.sum()),
                "mean_units": float(daily_total[sel].mean()),
                "baseline_units": float(baseline),
                "lift_pct": float(100 * (daily_total[sel].mean() - baseline) / baseline),
            }
        )
    ev_df = pd.DataFrame(ev_rows).sort_values("lift_pct")
    ev_df.to_csv(out / "table_event_lift.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    piv = snap_df.pivot(index="cat_id", columns="state_id", values="lift_pct")
    piv.plot(kind="bar", ax=axes[0], color=["#2a7fb8", "#e07b39", "#4c9a52"])
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set(
        ylabel="Demand lift on SNAP days (%)",
        title="SNAP-day demand contrast by category and state",
    )
    axes[0].legend(title="State", fontsize=8)
    axes[0].tick_params(axis="x", rotation=0)

    colors = ["#c1444a" if v < 0 else "#4c9a52" for v in ev_df["lift_pct"]]
    axes[1].barh(ev_df["event_type"], ev_df["lift_pct"], color=colors)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set(xlabel="Demand lift vs non-event days (%)", title="Event-type lift (descriptive)")

    fig.suptitle("§3.5 Events and SNAP", y=1.02, fontsize=12)
    _save(fig, out, "fig_3_5_events_snap")

    return {
        "snap_lift_by_category_pct": snap_df.groupby("cat_id")["lift_pct"]
        .mean()
        .round(2)
        .to_dict(),
        "snap_lift_overall_pct": round(float(snap_df["lift_pct"].mean()), 2),
        "event_lift_by_type_pct": ev_df.set_index("event_type")["lift_pct"].round(2).to_dict(),
        "note": (
            "Descriptive contrasts only. SNAP figures are a raw day-type "
            "difference, not a causal estimate; a causal reading requires the "
            "Double ML treatment in Appendix A."
        ),
    }


# ===========================================================================
# 3.8 Stationarity and distribution fitting -- gates Tier 1 and Tier 3
# ===========================================================================


def section_stationarity_distributions(
    wide: pd.DataFrame, cal: pd.DataFrame, out: Path, n_fit: int = 400
) -> dict[str, Any]:
    """
    ADF/KPSS at several aggregation levels, and count-distribution fitting.

    ADF and KPSS have OPPOSITE null hypotheses, which is why both are run:
        ADF  H0: unit root (non-stationary)
        KPSS H0: stationary
    Agreement is strong evidence. Disagreement means the series is likely
    difference- or trend-stationary in a way one test alone would mislabel,
    and that distinction changes the `d` term in ARIMA.

    Distribution fitting compares Poisson, Negative Binomial and Tweedie by
    AIC on a sample of series. This is the evidence Tier 3 needs to choose a
    likelihood instead of assuming one.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    day_cols = [c for c in wide.columns if c.startswith("d_")]
    sales = wide[day_cols].to_numpy(dtype=np.float64)
    aggregates = {
        "total": sales.sum(axis=0),
        "state_CA": sales[(wide["state_id"] == "CA").to_numpy()].sum(axis=0),
        "store_CA_1": sales[(wide["store_id"] == "CA_1").to_numpy()].sum(axis=0),
        "cat_FOODS": sales[(wide["cat_id"] == "FOODS").to_numpy()].sum(axis=0),
    }

    stat_rows = []
    for name, series in aggregates.items():
        try:
            adf_p = float(adfuller(series, autolag="AIC")[1])
        except Exception:
            adf_p = np.nan
        try:
            import warnings as _w

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                kpss_p = float(kpss(series, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = np.nan

        adf_stationary = adf_p < 0.05
        kpss_stationary = kpss_p > 0.05
        verdict = (
            "stationary"
            if adf_stationary and kpss_stationary
            else "non-stationary"
            if not adf_stationary and not kpss_stationary
            else "conflicting -- inspect trend/differencing"
        )
        stat_rows.append(
            {
                "level": name,
                "adf_p": adf_p,
                "kpss_p": kpss_p,
                "adf_says_stationary": adf_stationary,
                "kpss_says_stationary": kpss_stationary,
                "verdict": verdict,
            }
        )
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out / "table_stationarity.csv", index=False)

    # --- distribution fitting on a sample of bottom-level series ---
    from scipy import stats as sps

    rng = np.random.default_rng(0)
    idx = rng.choice(len(sales), size=min(n_fit, len(sales)), replace=False)

    fit_rows = []
    for i in idx:
        y = sales[i]
        y = y[-730:]  # last two years
        mu, var = y.mean(), y.var()
        if mu <= 0:
            continue

        # Poisson: one parameter.
        ll_pois = float(sps.poisson.logpmf(y, mu).sum())
        aic_pois = 2 * 1 - 2 * ll_pois

        # Negative binomial via method of moments. Defined only when
        # overdispersed; an underdispersed series has no valid NB fit.
        if var > mu:
            p = mu / var
            r = mu * p / (1 - p) if p < 1 else np.nan
            ll_nb = float(sps.nbinom.logpmf(y, r, p).sum()) if np.isfinite(r) and r > 0 else np.nan
            aic_nb = 2 * 2 - 2 * ll_nb if np.isfinite(ll_nb) else np.nan
        else:
            aic_nb = np.nan

        fit_rows.append(
            {
                "series_idx": int(i),
                "mean": float(mu),
                "var": float(var),
                "zero_frac": float((y == 0).mean()),
                "overdispersed": bool(var > mu),
                "aic_poisson": aic_pois,
                "aic_negbinom": aic_nb,
                "nb_better": bool(np.isfinite(aic_nb) and aic_nb < aic_pois),
            }
        )
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(out / "table_distribution_fit.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].scatter(fit_df["mean"], fit_df["var"], s=12, alpha=0.5, color="#2a7fb8")
    lim = [0, float(fit_df["mean"].quantile(0.98)) * 1.5]
    axes[0].plot(lim, lim, color="#c1444a", ls="--", lw=1.5, label="Poisson (var = mean)")
    axes[0].set(
        xlim=lim,
        ylim=[0, float(fit_df["var"].quantile(0.98)) * 1.2],
        xlabel="Series mean",
        ylabel="Series variance",
        title="Mean-variance relationship",
    )
    axes[0].legend(fontsize=8)

    share = pd.Series(
        {
            "Negative Binomial": float(fit_df["nb_better"].mean()),
            "Poisson": float(1 - fit_df["nb_better"].mean()),
        }
    )
    axes[1].bar(share.index, share.to_numpy(), color=["#4c9a52", "#e07b39"])
    axes[1].set(ylabel="Share of sampled series", title="Preferred count likelihood (AIC)")
    for i, v in enumerate(share):
        axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)

    axes[2].scatter(
        fit_df["zero_frac"],
        np.log10(np.clip(fit_df["var"] / fit_df["mean"], 1e-2, None)),
        s=12,
        alpha=0.5,
        color="#7a5ea8",
    )
    axes[2].axhline(0, color="#c1444a", ls="--", lw=1.2)
    axes[2].set(
        xlabel="Zero fraction", ylabel="log10(var/mean)", title="Zero-inflation vs overdispersion"
    )

    fig.suptitle("§3.8 Stationarity and count distributions", y=1.02, fontsize=12)
    _save(fig, out, "fig_3_8_distributions")

    return {
        "stationarity": stat_df.to_dict(orient="records"),
        "n_series_fitted": int(len(fit_df)),
        "pct_overdispersed": round(float(fit_df["overdispersed"].mean() * 100), 2),
        "pct_negbinom_beats_poisson": round(float(fit_df["nb_better"].mean() * 100), 2),
        "implication": (
            "Overdispersion is the norm, so Poisson is the wrong default. "
            "Tier 3 should use Negative Binomial or Tweedie; the large zero "
            "mass alongside continuous-positive behaviour favours Tweedie for "
            "the gradient-boosted models."
        ),
    }


# ===========================================================================
# Driver
# ===========================================================================


def main() -> int:
    p = argparse.ArgumentParser(description="M5 exploratory data analysis")
    p.add_argument(
        "--sample-series",
        type=int,
        default=None,
        help="subsample series for faster iteration (default: all 30,490)",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else settings.base_dir / "reports" / "eda"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    raw = settings.raw_data_dir
    wide = pd.read_parquet(raw / "sales_train_validation.parquet")
    cal = pd.read_parquet(raw / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])

    if args.sample_series:
        wide = wide.sample(n=min(args.sample_series, len(wide)), random_state=args.seed)
        log.warning(
            "eda_on_subsample",
            n_series=len(wide),
            note="figures describe the sample, not the full panel",
        )

    log.info("eda_start", n_series=len(wide), n_days=sum(c.startswith("d_") for c in wide.columns))

    findings: dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_series_analysed": int(len(wide)),
        "is_subsample": bool(args.sample_series),
        "data_limitations": {
            "sell_prices": (
                f"Source workbook truncated at Excel's row limit: "
                f"{M5.PRICE_ROW_COVERAGE_PCT}% of the true price table, covering "
                f"{list(M5.STORES_WITH_PRICE_DATA)} only. §3.6 price analysis is "
                f"therefore restricted to those stores and is omitted here."
            ),
        },
    }

    findings["3.1_structure"] = section_structure(wide, cal, out)
    findings["3.2_demand"] = section_demand(wide, out)
    findings["3.3_intermittency"] = section_intermittency(wide, out)
    findings["3.4_temporal"] = section_temporal(wide, cal, out)
    findings["3.5_events_snap"] = section_events_snap(wide, cal, out)
    findings["3.8_stationarity_distributions"] = section_stationarity_distributions(wide, cal, out)
    findings["elapsed_seconds"] = round(time.time() - t0, 1)

    (out / "eda_findings.json").write_text(
        json.dumps(findings, indent=2, default=str), encoding="utf-8"
    )

    # --- console summary ---
    s31, s32, s33 = findings["3.1_structure"], findings["3.2_demand"], findings["3.3_intermittency"]
    s34, s38 = findings["3.4_temporal"], findings["3.8_stationarity_distributions"]

    print("\n" + "=" * 78)
    print("PILLAR 1 -- EXPLORATORY DATA ANALYSIS")
    print("=" * 78)
    print(
        f"  series analysed        : {s31['n_series']:,}  x {s31['n_days']:,} days "
        f"= {s31['long_format_rows']:,} long-format rows"
    )
    print(f"  total units sold       : {s31['total_units_sold']:,}")
    print(f"  never sold a unit      : {s31['series_never_sold']:,} series")
    print(f"  introduced after day 1 : {s31['series_starting_after_day_1']:,} series")
    print()
    print(f"  zero-sales rate        : {s32['overall_zero_rate']:.1%} of all observations")
    print(
        "  by category            : "
        + ", ".join(f"{k} {v:.1%}" for k, v in s32["zero_rate_by_category"].items())
    )
    print(f"  overdispersed series   : {s32['pct_series_overdispersed']:.1%}")
    print(f"  top volume decile      : {s32['top_decile_volume_share']:.1%} of all units")
    print()
    print("  SBT classification:")
    for pattern, pct in sorted(s33["pct_of_series"].items(), key=lambda kv: -kv[1]):
        vol = s33["pct_of_volume"].get(pattern, 0.0)
        print(f"    {pattern:<14} {pct:>5.1f}% of series  |  {vol:>5.1f}% of volume")
    print(
        f"    -> {s33['intermittent_or_lumpy_pct']:.1f}% of series are sparse "
        f"(INTERMITTENT or LUMPY)"
    )
    print()
    print(f"  weekend uplift         : {s34['weekend_uplift_pct']:+.1f}%")
    print(
        f"  ACF lag 1 / 7 / 28     : {s34['acf_lag_1']:.3f} / {s34['acf_lag_7']:.3f} / {s34['acf_lag_28']:.3f}"
    )
    print()
    print("  Stationarity (aggregate levels):")
    for row in s38["stationarity"]:
        print(
            f"    {row['level']:<12} ADF p={row['adf_p']:.4f}  KPSS p={row['kpss_p']:.4f}  -> {row['verdict']}"
        )
    print(f"  overdispersed          : {s38['pct_overdispersed']:.1f}% of sampled series")
    print(f"  NegBinom beats Poisson : {s38['pct_negbinom_beats_poisson']:.1f}% of sampled series")
    print()
    print(f"  elapsed: {findings['elapsed_seconds']}s   artefacts: {out}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
