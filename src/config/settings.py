"""
Core configuration for the M5 Forecasting Model Laboratory.

This module has two distinct halves, and the separation is deliberate:

  1. `M5` -- IMMUTABLE FACTS about the M5 competition dataset. These are not
     configuration. They are properties of the data itself. Code and tests
     assert against them so a silently-corrupted ingest fails loudly instead
     of training on garbage.

     Several of these were VERIFIED against the supplied workbook and found to
     contradict the project brief. Where they differ, the data won and the
     discrepancy is documented inline: the sales window ends 2016-04-24 (not
     2016-06-19), and the calendar carries 30 event names over 162 event days
     (not "161 named events").

  2. The Pydantic settings classes -- things that genuinely vary by
     environment: paths, split windows, tuning budgets, tracking URIs.

Every number that is not in the dataset must be traceable to a named source
and must appear in the run manifest that accompanies each result, rather than
hiding in code.

Reference: Makridakis, Spiliotis & Assimakopoulos (2022), "M5 accuracy
competition: Results, findings and conclusions", International Journal of
Forecasting 38(4):1346-1364.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Mapping

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ===========================================================================
# SECTION 1 -- M5 DATASET FACTS (immutable; do not make these configurable)
# ===========================================================================


@dataclass(frozen=True)
class M5Facts:
    """
    Ground-truth cardinalities and ranges of the M5 dataset.

    Frozen dataclass rather than Pydantic settings on purpose: these must NOT
    be overridable by an environment variable. If an ingest produces 3,048
    items, that is a data bug to be raised -- not a config value to relax.
    """

    # --- Series cardinality ------------------------------------------------
    N_ITEMS: Final[int] = 3_049          # unique item_id values
    N_STORES: Final[int] = 10            # CA_1..CA_4, TX_1..TX_3, WI_1..WI_3
    N_SERIES: Final[int] = 30_490        # == N_ITEMS * N_STORES (bottom level)
    N_STATES: Final[int] = 3
    N_CATEGORIES: Final[int] = 3
    N_DEPARTMENTS: Final[int] = 7

    # --- Temporal extent ---------------------------------------------------
    # VERIFIED against the supplied workbook. The project brief stated the
    # 1,913 sales days run to 2016-06-19; they do not. 2011-01-29 + 1912 days
    # is 2016-04-24. 2016-06-19 is the last CALENDAR row (1,969), which extends
    # 56 days past the sales data precisely so the 28-day forecast window has
    # its events and SNAP flags available in advance.
    N_DAYS: Final[int] = 1_913           # d_1 .. d_1913 in sales_train_validation
    START_DATE: Final[date] = date(2011, 1, 29)
    END_DATE: Final[date] = date(2016, 4, 24)      # last day WITH sales actuals
    N_CALENDAR_ROWS: Final[int] = 1_969
    CALENDAR_END_DATE: Final[date] = date(2016, 6, 19)   # last calendar row

    # --- Forecasting task --------------------------------------------------
    HORIZON_DAYS: Final[int] = 28        # F1..F28 in the submission template

    # --- Event structure ---------------------------------------------------
    # Verified against the supplied workbook, because the project brief had
    # these conflated. The calendar carries 30 distinct names in event_name_1
    # (plus 4 more that only ever appear in event_name_2, all of which also
    # occur in column 1), spread over 162 dated event days. "161 events" was a
    # count of event DAYS, not names, and off by one.
    N_EVENT_NAMES: Final[int] = 30       # distinct event_name_1 values
    N_EVENT_DAYS: Final[int] = 162       # calendar rows with a non-null event
    EVENT_TYPES: Final[tuple[str, ...]] = (
        "Sporting", "Cultural", "National", "Religious",
    )

    # --- SNAP ---------------------------------------------------------------
    # Each state issues benefits on 650 of the 1,969 calendar days (~33%), but
    # on DIFFERENT days. That cross-state divergence on a shared date is what
    # identifies the SNAP effect: it supplies within-day, between-state
    # variation that is not confounded with national demand seasonality.
    SNAP_DAYS_PER_STATE: Final[int] = 650
    SNAP_COLUMNS: Final[tuple[str, ...]] = ("snap_CA", "snap_TX", "snap_WI")

    # --- Categorical domains -----------------------------------------------
    STATES: Final[tuple[str, ...]] = ("CA", "TX", "WI")
    STORES: Final[tuple[str, ...]] = (
        "CA_1", "CA_2", "CA_3", "CA_4",
        "TX_1", "TX_2", "TX_3",
        "WI_1", "WI_2", "WI_3",
    )
    CATEGORIES: Final[tuple[str, ...]] = ("HOBBIES", "HOUSEHOLD", "FOODS")
    DEPARTMENTS: Final[tuple[str, ...]] = (
        "HOBBIES_1", "HOBBIES_2",
        "HOUSEHOLD_1", "HOUSEHOLD_2",
        "FOODS_1", "FOODS_2", "FOODS_3",
    )

    # --- Hierarchy ---------------------------------------------------------
    # Six aggregation levels used by the MinT reconciler. Ordered bottom-up so
    # index 0 is always the disaggregate level the models are fitted at.
    HIERARCHY_LEVELS: Final[tuple[str, ...]] = (
        "item_store",   # 30,490 series -- the level models train at
        "dept_store",   #     70 series (7 depts x 10 stores)
        "cat_store",    #     30 series (3 cats  x 10 stores)
        "store",        #     10 series
        "state",        #      3 series
        "total",        #      1 series
    )

    # --- Units -------------------------------------------------------------
    # Sales are INTEGER UNIT COUNTS. Not revenue, not dollars. Any code that
    # formats a forecast as currency is a bug.
    TARGET_UNIT: Final[str] = "units"

    # --- Price coverage in THIS copy of the data ----------------------------
    # The dataset was supplied as a single Excel workbook, and the sell_prices
    # sheet hit Excel's 1,048,576-row ceiling: 1,048,575 of the true 6,841,121
    # rows survived (15.3%). The cut is positional, not a sample -- the CSV was
    # pasted in store-by-store and stopped partway through CA_2 -- so price
    # data exists for exactly two stores and is entirely absent for the other
    # eight.
    #
    # This is a property of the supplied file, not of M5 itself, which is why
    # it lives beside the facts rather than among them. Sales, calendar, SNAP
    # and event data are all COMPLETE, so forecasting, hierarchical
    # reconciliation, SNAP causal estimation and holiday lift are unaffected.
    # Anything price-derived must be restricted to the covered stores.
    STORES_WITH_PRICE_DATA: Final[tuple[str, ...]] = ("CA_1", "CA_2")
    PRICE_ROW_COVERAGE_PCT: Final[float] = 15.33

    def has_price_data(self, store_id: str) -> bool:
        """True if this store has sell_price rows in the supplied workbook."""
        return store_id in self.STORES_WITH_PRICE_DATA

    # --- Canonical evaluation splits ---------------------------------------
    # The spec proposed test = 2016-04-25..2016-05-22. Those dates are
    # d_1914..d_1941, and `Sales_train_validation` stops at d_1913 -- that is
    # the M5 competition's held-back validation period, for which this release
    # contains NO actuals. Scoring against it is impossible, so the windows
    # below shift back to the last 56 days that DO have ground truth, keeping
    # the intended 28/28 chronological structure.
    #
    #   train      d_1    .. d_1857   2011-01-29 .. 2016-02-28
    #   validation d_1858 .. d_1885   2016-02-29 .. 2016-03-27
    #   test       d_1886 .. d_1913   2016-03-28 .. 2016-04-24   (held out)
    TRAIN_END: Final[date] = date(2016, 2, 28)
    VAL_START: Final[date] = date(2016, 2, 29)
    VAL_END: Final[date] = date(2016, 3, 27)
    TEST_START: Final[date] = date(2016, 3, 28)
    TEST_END: Final[date] = date(2016, 4, 24)

    def store_to_state(self, store_id: str) -> str:
        """`CA_3` -> `CA`. Store IDs encode their state as the prefix."""
        return store_id.split("_")[0]

    def d_to_date(self, d: str | int) -> date:
        """`d_1913` (or 1913) -> the calendar date it refers to."""
        n = int(str(d).replace("d_", ""))
        if not 1 <= n <= self.N_CALENDAR_ROWS:
            raise ValueError(f"d index {n} outside calendar range 1..{self.N_CALENDAR_ROWS}")
        return self.START_DATE + timedelta(days=n - 1)

    def date_to_d(self, when: date) -> str:
        """Inverse of `d_to_date`."""
        offset = (when - self.START_DATE).days
        if offset < 0 or offset >= self.N_CALENDAR_ROWS:
            raise ValueError(f"{when} outside the M5 calendar")
        return f"d_{offset + 1}"

    def has_actuals(self, when: date) -> bool:
        """True if `when` falls inside the window that has sales ground truth."""
        return self.START_DATE <= when <= self.END_DATE

    def dept_to_category(self, dept_id: str) -> str:
        """`FOODS_3` -> `FOODS`. Department IDs encode their category."""
        return dept_id.rsplit("_", 1)[0]

    def validate_cardinalities(self, observed: Mapping[str, int]) -> list[str]:
        """
        Compare observed counts from an ingest against the known truth.

        Returns a list of human-readable violations (empty == clean). Used by
        the validation layer so a partial or duplicated load cannot silently
        proceed to training.
        """
        expected = {
            "n_items": self.N_ITEMS,
            "n_stores": self.N_STORES,
            "n_series": self.N_SERIES,
            "n_states": self.N_STATES,
            "n_categories": self.N_CATEGORIES,
            "n_departments": self.N_DEPARTMENTS,
            "n_days": self.N_DAYS,
        }
        violations: list[str] = []
        for key, want in expected.items():
            if key in observed and observed[key] != want:
                violations.append(
                    f"{key}: expected {want:,} (M5 fact), observed {observed[key]:,}"
                )
        return violations


M5: Final[M5Facts] = M5Facts()


# ===========================================================================
# SECTION 2 -- DOCUMENTED ASSUMPTIONS (data M5 does not contain)
# ===========================================================================


@dataclass(frozen=True)
class Assumption:
    """
    A single parameter that is NOT in the M5 dataset.

    Every field is mandatory. An assumption without a stated source is
    indistinguishable from a fabricated number, which is the exact failure
    mode this platform was rebuilt to eliminate.
    """

    key: str
    value: Any
    unit: str
    source: str          # where the number came from -- never just "assumed"
    used_by: str         # which module depends on it
    rationale: str       # why this value and not another


class AssumptionRegistry:
    """
    Single source of truth for every non-M5 parameter.

    API routers and the dashboard read from this registry so assumptions
    travel with the results they influence. If you add a parameter the
    dataset does not contain, register it here -- do not inline it.
    """

    _REGISTRY: Final[tuple[Assumption, ...]] = (
        Assumption(
            key="store_locations",
            value={
                # M5 gives state only. Coordinates are inferred from the
                # largest metros in each state so a weather join is possible
                # at all. Store-level precision is NOT claimed -- this is a
                # state-level proxy and is labelled as such downstream.
                "CA_1": (34.0522, -118.2437),   # Los Angeles
                "CA_2": (34.0522, -118.2437),   # Los Angeles
                "CA_3": (37.7749, -122.4194),   # San Francisco
                "CA_4": (37.7749, -122.4194),   # San Francisco
                "TX_1": (32.7767, -96.7970),    # Dallas
                "TX_2": (29.7604, -95.3698),    # Houston
                "TX_3": (32.7767, -96.7970),    # Dallas
                "WI_1": (43.0389, -87.9065),    # Milwaukee
                "WI_2": (43.0731, -89.4012),    # Madison
                "WI_3": (44.5133, -88.0133),    # Green Bay
            },
            unit="(latitude, longitude)",
            source="Geographic inference from state_id; not in M5",
            used_by="ingestion.WeatherIngester",
            rationale=(
                "M5 provides state_id but no store address. Weather is a "
                "regional signal, so metro-level coordinates capture most of "
                "the usable variation. Treated as a state proxy, not a "
                "store location."
            ),
        ),
        Assumption(
            key="unit_cost_ratio",
            value=0.60,
            unit="fraction of sell_price",
            source="APICS/ASCM grocery gross-margin benchmark (~40% margin)",
            used_by="analytics.holiday_and_inventory",
            rationale=(
                "M5 contains sell_price but no COGS. A 60% cost ratio is the "
                "midpoint of published grocery benchmarks. Consequence: this "
                "platform reports REVENUE, never profit. Any profit figure "
                "derived from this ratio would be an artefact of the "
                "assumption rather than a finding about the retailer."
            ),
        ),
        Assumption(
            key="ordering_cost",
            value=50.0,
            unit="USD per purchase order",
            source="APICS/ASCM benchmark for retail replenishment",
            used_by="analytics.holiday_and_inventory (EOQ)",
            rationale=(
                "EOQ requires a fixed per-order cost. M5 has no procurement "
                "data. Sensitivity to this value is reported alongside every "
                "EOQ result rather than assumed away."
            ),
        ),
        Assumption(
            key="holding_cost_rate",
            value=0.25,
            unit="fraction of unit value per year",
            source="APICS/ASCM benchmark (capital + storage + obsolescence)",
            used_by="analytics.holiday_and_inventory (EOQ, safety stock)",
            rationale=(
                "Standard retail carrying-cost rate. Combined with the unit "
                "cost assumption, so EOQ inherits both uncertainties."
            ),
        ),
        Assumption(
            key="lead_time_mean_days",
            value=7.0,
            unit="days",
            source="Typical DC-to-store replenishment cycle",
            used_by="analytics.holiday_and_inventory (safety stock, ROP)",
            rationale=(
                "M5 has no supplier or lead-time data. Seven days matches a "
                "weekly replenishment cadence, consistent with the weekly "
                "granularity of the sell_prices table."
            ),
        ),
        Assumption(
            key="lead_time_std_days",
            value=1.5,
            unit="days",
            source="Typical DC-to-store variability",
            used_by="analytics.holiday_and_inventory (safety stock)",
            rationale=(
                "Safety stock under stochastic lead time needs a variance "
                "term. 1.5 days is ~21% CV, a common planning default."
            ),
        ),
    )

    @classmethod
    def all(cls) -> tuple[Assumption, ...]:
        return cls._REGISTRY

    @classmethod
    def get(cls, key: str) -> Assumption:
        for a in cls._REGISTRY:
            if a.key == key:
                return a
        raise KeyError(
            f"Unknown assumption {key!r}. Register it in AssumptionRegistry "
            f"rather than inlining an undocumented constant."
        )

    @classmethod
    def value(cls, key: str) -> Any:
        return cls.get(key).value

    @classmethod
    def as_disclosure(cls, keys: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Serialisable form for attaching to API responses and dashboards.

        Callers pass the subset of keys their result actually depends on, so
        the disclosure is specific rather than a blanket footer users learn
        to ignore.
        """
        items = cls._REGISTRY if keys is None else [cls.get(k) for k in keys]
        return [
            {
                "assumption": a.key,
                "value": a.value,
                "unit": a.unit,
                "source": a.source,
                "rationale": a.rationale,
            }
            for a in items
        ]


# Warnings that must accompany specific classes of result. Defined once here
# so the exact wording cannot drift between API, dashboard and docs.
ENDOGENEITY_WARNING: Final[str] = (
    "ENDOGENEITY_WARNING: This elasticity is CORRELATIONAL, not causal. "
    "Prices in M5 are set by the retailer and may respond to anticipated "
    "demand (e.g. markdowns ahead of expected softness), biasing OLS "
    "estimates toward zero. No instrument or randomised experiment is "
    "available in this dataset. Do not use this coefficient to set prices."
)

INVENTORY_ASSUMPTION_WARNING: Final[str] = (
    "ASSUMPTION_WARNING: M5 contains no inventory levels, lead times, or unit "
    "costs. These figures are a policy SIMULATION driven by documented "
    "benchmark assumptions, not a measurement of this retailer's operations. "
    "See the `assumptions` field for every value the result depends on."
)

SNAP_CAUSAL_NOTE: Final[str] = (
    "SNAP disbursement dates are fixed by state government and are plausibly "
    "exogenous to item-level demand, which supports a causal reading. "
    "Estimated via Double ML with cross-fitting (Chernozhukov et al., 2018)."
)


# ===========================================================================
# SECTION 3 -- ENVIRONMENT CONFIGURATION (genuinely variable)
# ===========================================================================


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="m5_platform")
    user: str = Field(default="m5")
    password: str = Field(default="changeme")
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=10)
    pool_timeout: int = Field(default=30)
    echo: bool = Field(default=False)

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def async_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class MLflowSettings(BaseSettings):
    """MLflow tracking + model registry configuration."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_prefix: str = Field(default="m5")
    artifact_root: str = Field(default="./mlruns")
    registry_uri: str = Field(default="")

    # Name under which the champion is registered. The promotion gate compares
    # a new challenger against whatever currently holds the Production stage.
    registered_model_name: str = Field(default="m5_demand_forecaster")


class ModelSettings(BaseSettings):
    """Training and evaluation configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    # Horizon defaults to the M5 fact but is overridable for ad-hoc backtests.
    forecast_horizon: int = Field(default=M5.HORIZON_DAYS)

    # Nested CV: outer folds report unbiased performance, inner folds select
    # hyperparameters. See Cawley & Talbot (2010), JMLR 11:2079-2107.
    outer_cv_splits: int = Field(default=3)
    inner_cv_splits: int = Field(default=3)
    tune_trials: int = Field(default=30)

    min_train_days: int = Field(default=365)
    feature_version: int = Field(default=2)

    quantiles: list[float] = Field(default=[0.10, 0.25, 0.50, 0.75, 0.90])

    # Promotion gate: a challenger must beat the incumbent by at least this
    # relative margin on WAPE before promotion. Guards against churning the
    # Production model on fold-to-fold noise.
    promotion_min_improvement_pct: float = Field(default=1.0)

    # Retraining triggers.
    wape_threshold: float = Field(default=15.0)   # percent
    rmsse_threshold: float = Field(default=1.0)   # >1 means worse than naive
    bias_threshold: float = Field(default=0.10)


class MonitoringSettings(BaseSettings):
    """Drift detection and alerting configuration."""

    model_config = SettingsConfigDict(env_prefix="MONITOR_", extra="ignore")

    # PSI convention (Siddiqi, 2006): <0.10 stable, 0.10-0.25 moderate shift,
    # >0.25 significant shift warranting investigation.
    psi_warn_threshold: float = Field(default=0.10)
    psi_alert_threshold: float = Field(default=0.25)
    ks_significance_level: float = Field(default=0.05)

    drift_feature_pct_trigger: float = Field(default=0.30)
    alert_email: str = Field(default="")
    slack_webhook: str = Field(default="")
    prometheus_port: int = Field(default=8001)


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="M5 Retail Demand Intelligence Platform")
    app_version: str = Field(default="2.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    base_dir: Path = Field(default=Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default=Path("data"))
    model_dir: Path = Field(default=Path("models"))
    reports_dir: Path = Field(default=Path("reports"))
    logs_dir: Path = Field(default=Path("logs"))

    api_secret_key: str = Field(default="dev-secret-key-change-in-production")
    api_key_header: str = Field(default="X-API-Key")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {sorted(allowed)}")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def raw_data_dir(self) -> Path:
        """Where the four M5 source files are expected to live."""
        return self.base_dir / self.data_dir / "raw"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cached settings instance -- import this, do not construct AppSettings."""
    return AppSettings()


settings = get_settings()
