"""
Pillar 5: generate figures from whatever results currently exist.

Built now, against Tier 1 + 2 results, deliberately. Every later tier then adds
ROWS to figures that already work, instead of requiring a big-bang
visualisation build at the end when time is shortest.

Figures produced at this stage:
    Fig 1      leaderboard bar chart
    Fig 2      multi-metric agreement
    Fig 3      per-series error distribution
    Fig 4 + 5  critical difference and MCB, SIDE BY SIDE

Usage:
    python scripts/pipeline/run_figures.py
    python scripts/pipeline/run_figures.py --results-dir reports/tier1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.logging import get_logger
from src.config.settings import settings
from src.evaluation.significance import (
    friedman_test,
    mcb_test,
    nemenyi_test,
    per_series_scores,
)
from src.visualization.leaderboard_plots import (
    plot_error_distribution,
    plot_leaderboard,
    plot_multi_metric,
    plot_significance_pair,
)
from src.visualization.segment_plots import (
    plot_horizon_degradation,
    plot_model_by_category,
    plot_model_by_demand_class,
    plot_model_by_level,
)

log = get_logger("figures")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Pillar 5 figures")
    p.add_argument(
        "--results-dir",
        default="reports/backtest_main",
        help="directory containing predictions.parquet",
    )
    p.add_argument("--out-dir", default="reports/results")
    p.add_argument("--metric", default="wape")
    p.add_argument("--horizon", type=int, default=28)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = settings.base_dir / results_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = settings.base_dir / out_dir

    preds_path = results_dir / "predictions.parquet"
    if not preds_path.exists():
        log.error("predictions_missing", path=str(preds_path))
        print(f"\nNo predictions at {preds_path}. Run a backtest or tier first.\n")
        return 1

    preds = pd.read_parquet(preds_path).dropna(subset=["y_true", "y_pred"])
    log.info(
        "loaded_predictions",
        rows=len(preds),
        models=preds["model"].nunique(),
        series=preds["series_id"].nunique(),
    )

    # --- pooled leaderboard ---
    rows = []
    for model, g in preds.groupby("model", observed=True):
        yt = g["y_true"].to_numpy(float)
        yp = np.clip(g["y_pred"].to_numpy(float), 0.0, None)
        denom = np.abs(yt).sum()
        row = {
            "model": str(model),
            "wape": 100.0 * np.abs(yt - yp).sum() / denom if denom > 0 else np.nan,
            "mae": float(np.abs(yt - yp).mean()),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "bias": float(np.mean(yp - yt)),
        }
        if "scale" in g.columns:
            scales = g["scale"].to_numpy(float)
            ok = np.isfinite(scales) & (scales > 0)
            if ok.any():
                row["rmsse"] = float(np.sqrt(np.mean((yt[ok] - yp[ok]) ** 2 / scales[ok])))
                row["mase"] = float(np.mean(np.abs(yt[ok] - yp[ok]) / np.sqrt(scales[ok])))
        rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)
    # RMSSE here must be the canonical per-(series, origin) mean from accuracy_overall.csv. The pooled
    # sqrt(mean(e^2 / scale)) computed above is a different quantity (it ranks LightGBM first and
    # contradicts the README), so it is replaced whenever the canonical table is present.
    canonical_csv = results_dir / "accuracy_overall.csv"
    if canonical_csv.exists():
        canonical = pd.read_csv(canonical_csv).set_index("model")["rmsse"]
        leaderboard["rmsse"] = leaderboard["model"].map(canonical)
    leaderboard.to_csv(
        out_dir.parent / "results" / "leaderboard.csv", index=False
    ) if out_dir.name == "results" else None

    written: list[Path] = []
    written.append(plot_leaderboard(leaderboard, out_dir, metric=args.metric))
    written.append(plot_multi_metric(leaderboard, out_dir))

    # --- per-series matrix drives Figs 3, 4, 5 ---
    matrix = per_series_scores(preds, metric=args.metric)
    complete = matrix.dropna(axis=0, how="any")
    log.info("per_series_matrix", series=len(complete), models=complete.shape[1])

    written.append(plot_error_distribution(complete, out_dir, metric_label=args.metric.upper()))

    friedman = friedman_test(complete, alpha=args.alpha)
    if not friedman.significant:
        print("\nFriedman did not reject -- Figs 4/5 skipped, as post-hoc tests")
        print("are not licensed. The models are statistically indistinguishable.\n")
    else:
        nem = nemenyi_test(complete, alpha=args.alpha)
        mcb = mcb_test(complete, alpha=args.alpha)

        # Pooled (volume-weighted) leader for the disagreement caption.
        pooled_leader = str(leaderboard.iloc[0]["model"])
        rank_leader = str(nem["average_ranks"].index[0])

        written.append(
            plot_significance_pair(
                average_ranks=nem["average_ranks"],
                critical_difference=nem["critical_difference"],
                mcb=mcb,
                out_dir=out_dir,
                pooled_leader=pooled_leader,
                rank_leader=rank_leader,
                cliques=nem["cliques"],
                n_series=len(complete),
            )
        )

    # --- Group B/C: segment and temporal ---
    # Fig 6 needs the SBC classification the EDA produced. Missing inputs skip
    # with a logged reason rather than raising, so a fresh clone can generate
    # the figures it has data for instead of failing on the first gap.
    cls_path = settings.base_dir / "reports" / "eda" / "series_classification.parquet"
    if cls_path.exists():
        try:
            written.append(plot_model_by_demand_class(preds, pd.read_parquet(cls_path), out_dir))
        except (ValueError, KeyError) as exc:
            log.warning("fig06_skipped", reason=str(exc))
    else:
        log.warning("fig06_skipped", reason="run scripts/pipeline/run_eda.py first")

    if "cat_id" in preds.columns:
        written.append(plot_model_by_category(preds, out_dir))

    level_path = results_dir / "accuracy_by_hierarchy_level.csv"
    if level_path.exists():
        by_level = pd.read_csv(level_path)
        if {"model", "level", "wape"} <= set(by_level.columns):
            written.append(plot_model_by_level(by_level, out_dir))

    written.append(plot_horizon_degradation(preds, out_dir))

    print("\n" + "=" * 76)
    print("PILLAR 5 -- FIGURES GENERATED")
    print("=" * 76)
    print(f"  source : {preds_path.relative_to(settings.base_dir)}")
    print(f"  models : {preds['model'].nunique()}   series: {len(complete):,}")
    for path in written:
        print(f"    {path.relative_to(settings.base_dir)}")
    print("\n  Each figure has a .md sidecar carrying its caption.")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
