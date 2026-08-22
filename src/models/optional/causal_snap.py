"""
Causal Machine Learning Module.

Moves beyond correlation to estimate true causal effects of:
  - Promotions on demand (promotion lift)
  - Holidays on demand
  - Price changes on demand (causal elasticity)

Methods:
  - Double Machine Learning (DML) — Robinson (1988) / Chernozhukov et al. (2018)
  - Difference-in-Differences (DiD) — natural experiments
  - Causal graph via DoWhy (when available)

Why causal vs correlational?
  A naive regression of sales ~ promotion is biased because:
  1. Promotions are not random (confounders: seasonality, inventory levels)
  2. Stores select into promotions based on expected demand

DML solves this with the "partialling out" trick:
  Step 1: Regress T (treatment) on X (controls) → get residuals Ẽ
  Step 2: Regress Y (outcome) on X (controls) → get residuals Ỹ
  Step 3: Regress Ỹ on Ẽ → estimate is causal under ML regularity conditions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold

from src.config.logging import get_logger

log = get_logger(__name__)


@dataclass
class CausalEffect:
    """Estimated causal effect with uncertainty."""
    treatment: str
    outcome: str
    effect: float              # Average treatment effect (ATE)
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_obs: int
    method: str
    is_significant: bool

    def __str__(self) -> str:
        direction = "increases" if self.effect > 0 else "decreases"
        return (
            f"{self.treatment} causally {direction} {self.outcome} "
            f"by {abs(self.effect):.3f} units "
            f"(95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
            f"p={self.p_value:.4f})"
        )


class DoubleMLEstimator:
    """
    Double Machine Learning (DML) for causal effect estimation.

    Suitable for:
      - Continuous treatments: price → demand
      - Binary treatments: promotion → demand, holiday → demand

    Cross-fitting procedure (K=5 folds):
      For each fold k:
        1. Train nuisance models on -k folds
        2. Get out-of-fold residuals on fold k
      Final estimate: OLS of outcome_residuals ~ treatment_residuals

    Reference: Chernozhukov, V. et al. (2018) "Double/Debiased Machine Learning"
    """

    def __init__(
        self,
        n_folds: int = 5,
        ml_model: Any = None,
    ):
        self.n_folds = n_folds
        self.ml_model = ml_model  # If None, uses LightGBM

    def estimate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        controls: list[str],
        treatment_type: str = "continuous",  # "continuous" or "binary"
    ) -> CausalEffect:
        """
        Estimate causal effect of treatment on outcome controlling for covariates.

        Args:
            df: DataFrame with all columns
            treatment: Column name of treatment variable (e.g., "is_on_sale")
            outcome: Column name of outcome variable (e.g., "sales")
            controls: List of control/confounder column names
            treatment_type: "continuous" or "binary"

        Returns:
            CausalEffect with ATE estimate and uncertainty bounds
        """
        df_clean = df[controls + [treatment, outcome]].dropna()
        if len(df_clean) < 100:
            log.warning("insufficient_data_for_dml", n=len(df_clean))
            return self._null_effect(treatment, outcome, len(df_clean))

        X = df_clean[controls].fillna(0).values
        T = df_clean[treatment].values.astype(float)
        Y = df_clean[outcome].values.astype(float)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        T_residuals = np.zeros_like(T)
        Y_residuals = np.zeros_like(Y)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # Step 1: Partial out treatment
            if treatment_type == "binary":
                m_model = LGBMClassifier(
                    n_estimators=200, random_state=42, verbose=-1
                )
                m_model.fit(X_train, T_train.astype(int))
                T_hat = m_model.predict_proba(X_val)[:, 1]
            else:
                m_model = self._get_ml_model()
                m_model.fit(X_train, T_train)
                T_hat = m_model.predict(X_val)

            T_residuals[val_idx] = T_val - T_hat

            # Step 2: Partial out outcome
            g_model = self._get_ml_model()
            g_model.fit(X_train, Y_train)
            Y_hat = g_model.predict(X_val)
            Y_residuals[val_idx] = Y_val - Y_hat

        # Step 3: Final OLS on residuals — this is the causal estimate
        import statsmodels.api as sm
        T_res_const = sm.add_constant(T_residuals)
        ols_result = sm.OLS(Y_residuals, T_res_const).fit(
            cov_type="HC3"  # Robust standard errors
        )

        effect = float(ols_result.params[1])
        se = float(ols_result.bse[1])
        ci = ols_result.conf_int()[1]
        p_value = float(ols_result.pvalues[1])

        result = CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=effect,
            std_error=se,
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=p_value,
            n_obs=len(df_clean),
            method="double_ml",
            is_significant=(p_value < 0.05),
        )
        log.info("dml_estimate_complete", result=str(result))
        return result

    def _get_ml_model(self) -> Any:
        if self.ml_model is not None:
            return type(self.ml_model)(**self.ml_model.get_params())
        return LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )

    def _null_effect(self, treatment: str, outcome: str, n: int) -> CausalEffect:
        return CausalEffect(
            treatment=treatment, outcome=outcome,
            effect=0.0, std_error=0.0, ci_lower=0.0, ci_upper=0.0,
            p_value=1.0, n_obs=n, method="double_ml_failed",
            is_significant=False,
        )


class DiDEstimator:
    """
    Difference-in-Differences estimator for event impact.

    Useful for: "What was the lift from a promotion that ran in specific stores?"
    Assumes parallel trends between treated and control stores pre-event.

    DiD = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
    """

    def estimate(
        self,
        df: pd.DataFrame,
        treatment_col: str,      # 1 = treated store/product, 0 = control
        post_col: str,           # 1 = post-event period, 0 = pre-event
        outcome_col: str,
        entity_col: str = "store_id",
        time_col: str = "date",
    ) -> CausalEffect:
        """
        Estimate DiD effect with two-way fixed effects regression.

        Regression: Y_{it} = α_i + γ_t + β * (Treated_i × Post_t) + ε_{it}
        β is the DiD estimator.
        """
        df_clean = df[[entity_col, time_col, treatment_col, post_col, outcome_col]].dropna()

        if len(df_clean) < 50:
            return self._null_effect(treatment_col, outcome_col, len(df_clean))

        # Create interaction term
        df_clean = df_clean.copy()
        df_clean["treated_post"] = df_clean[treatment_col] * df_clean[post_col]

        try:
            import statsmodels.formula.api as smf

            # Two-way FE (entity + time dummies absorbed by within transformation)
            model = smf.ols(
                f"{outcome_col} ~ treated_post + {treatment_col} + {post_col} + "
                f"C({entity_col}) + C({time_col})",
                data=df_clean,
            ).fit(cov_type="cluster", cov_kwds={"groups": df_clean[entity_col]})

            effect = float(model.params.get("treated_post", 0))
            se = float(model.bse.get("treated_post", 0))
            ci = model.conf_int().loc["treated_post"] if "treated_post" in model.conf_int().index else [0, 0]
            p_value = float(model.pvalues.get("treated_post", 1.0))

        except Exception as e:
            log.warning("did_estimation_failed", error=str(e))
            return self._null_effect(treatment_col, outcome_col, len(df_clean))

        return CausalEffect(
            treatment=f"{treatment_col} × {post_col}",
            outcome=outcome_col,
            effect=effect,
            std_error=se,
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=p_value,
            n_obs=len(df_clean),
            method="diff_in_diff_twfe",
            is_significant=(p_value < 0.05),
        )

    def _null_effect(self, treatment: str, outcome: str, n: int) -> CausalEffect:
        return CausalEffect(
            treatment=treatment, outcome=outcome,
            effect=0.0, std_error=0.0, ci_lower=0.0, ci_upper=0.0,
            p_value=1.0, n_obs=n, method="did_failed", is_significant=False,
        )


class CausalAnalysisPipeline:
    """
    Runs all causal analyses for a supply chain dataset.
    Answers key business questions with causal estimates.
    """

    def __init__(self):
        self.dml = DoubleMLEstimator(n_folds=5)
        self.did = DiDEstimator()
        self._log = get_logger("CausalAnalysisPipeline")

    def run_all(self, df: pd.DataFrame) -> dict[str, CausalEffect]:
        """
        Run all causal analyses. Returns dict of {question: CausalEffect}.
        """
        control_cols = [
            c for c in [
                "lag_7", "rolling_mean_28", "day_of_week", "month",
                "is_holiday", "snap_flag", "store_id_enc", "dept_id_enc",
            ] if c in df.columns
        ]

        results = {}

        # Q1: What is the causal effect of being on sale on demand?
        if "is_on_sale" in df.columns and "sales" in df.columns:
            self._log.info("estimating_promotion_lift")
            results["promotion_lift"] = self.dml.estimate(
                df=df,
                treatment="is_on_sale",
                outcome="sales",
                controls=control_cols,
                treatment_type="binary",
            )

        # Q2: What is the causal price elasticity?
        if "sell_price" in df.columns and "sales" in df.columns:
            self._log.info("estimating_causal_price_elasticity")
            df_log = df.copy()
            df_log["log_sales"] = np.log1p(df_log["sales"])
            df_log["log_price"] = np.log(df_log["sell_price"].clip(lower=0.01))
            results["price_elasticity_causal"] = self.dml.estimate(
                df=df_log,
                treatment="log_price",
                outcome="log_sales",
                controls=control_cols,
                treatment_type="continuous",
            )

        # Q3: Causal effect of holidays on demand
        if "is_holiday" in df.columns and "sales" in df.columns:
            self._log.info("estimating_holiday_effect")
            results["holiday_effect"] = self.dml.estimate(
                df=df,
                treatment="is_holiday",
                outcome="sales",
                controls=[c for c in control_cols if c != "is_holiday"],
                treatment_type="binary",
            )

        # Q4: SNAP benefit day effect
        if "snap_flag" in df.columns and "sales" in df.columns:
            self._log.info("estimating_snap_effect")
            results["snap_effect"] = self.dml.estimate(
                df=df,
                treatment="snap_flag",
                outcome="sales",
                controls=[c for c in control_cols if c != "snap_flag"],
                treatment_type="binary",
            )

        # Summary
        for name, effect in results.items():
            self._log.info(
                "causal_result",
                analysis=name,
                effect=round(effect.effect, 4),
                significant=effect.is_significant,
                p_value=round(effect.p_value, 4),
            )

        return results

    def generate_report(self, results: dict[str, CausalEffect]) -> str:
        """Generate a plain-English causal analysis report."""
        lines = ["# Causal Analysis Report\n"]
        lines.append("## Key Findings\n")

        labels = {
            "promotion_lift": "Promotion Effect on Daily Sales",
            "price_elasticity_causal": "Causal Price Elasticity (log-log)",
            "holiday_effect": "Holiday Effect on Daily Sales",
            "snap_effect": "SNAP Benefit Day Effect on Daily Sales",
        }

        for key, effect in results.items():
            label = labels.get(key, key)
            sig = "✓ Significant" if effect.is_significant else "✗ Not significant"
            lines.append(f"### {label}")
            lines.append(f"- {str(effect)}")
            lines.append(f"- Statistical significance: {sig} (p={effect.p_value:.4f})")
            lines.append(f"- Based on {effect.n_obs:,} observations\n")

        return "\n".join(lines)
