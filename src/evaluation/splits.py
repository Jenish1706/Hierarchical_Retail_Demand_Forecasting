"""
Chronological splitting and walk-forward origins.

Standing constraint 7: chronological splits only, never random.

Why this is not negotiable
--------------------------
A random split of a time series lets the model train on 2016-04-20 to predict
2016-04-15. Every metric improves, and every improvement is fictional. The
model has seen the future of the very series it is scoring, including the
demand spike it is supposed to anticipate. On M5 this inflates accuracy by
enough to reverse a leaderboard.

Two things this module guarantees:

  1. Every split boundary is a DATE, applied to all 30,490 series at once.
     Splitting per-series at different dates would let series A's future leak
     into series B's past through any cross-series feature or global model.

  2. Test windows never overlap training windows, and rolling origins are
     non-overlapping with each other, so averaging across origins is closer to
     averaging independent draws than to double-counting the same days.

References
----------
Tashman (2000), "Out-of-sample tests of forecasting accuracy: an analysis and
    review", IJF 16(4):437-450.  [rolling-origin evaluation]
Bergmeir & Benitez (2012), "On the use of cross-validation for time series
    predictor evaluation", Information Sciences 191:192-213.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.config.logging import get_logger
from src.config.settings import M5

log = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    """One train/test boundary, expressed in dates."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def horizon(self) -> int:
        return int((self.test_end - self.test_start).days) + 1

    @property
    def n_train_days(self) -> int:
        return int((self.train_end - self.train_start).days) + 1

    def describe(self) -> str:
        return (
            f"{self.name}: train {self.train_start.date()}..{self.train_end.date()} "
            f"({self.n_train_days}d) -> test {self.test_start.date()}..{self.test_end.date()} "
            f"({self.horizon}d)"
        )

    def split_frame(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply this split to a long panel, returning (train, test)."""
        dates = pd.to_datetime(df[date_col])
        train = df[(dates >= self.train_start) & (dates <= self.train_end)]
        test = df[(dates >= self.test_start) & (dates <= self.test_end)]
        return train, test

    def to_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "horizon": self.horizon,
            "n_train_days": self.n_train_days,
        }


def canonical_splits() -> dict[str, Split]:
    """
    The project's fixed train / validation / test windows.

    Sourced from `M5Facts`, which shifted them back from the values in the
    project brief: the brief's test window (2016-04-25..2016-05-22) is
    d_1914..d_1941, and this release of the data stops at d_1913, so it
    contains no actuals to score against. See the note in settings.py.

    The test split is HELD OUT until the final report. Model selection uses
    validation and the walk-forward origins only.
    """
    val = Split(
        name="validation",
        train_start=pd.Timestamp(M5.START_DATE),
        train_end=pd.Timestamp(M5.TRAIN_END),
        test_start=pd.Timestamp(M5.VAL_START),
        test_end=pd.Timestamp(M5.VAL_END),
    )
    test = Split(
        name="test",
        train_start=pd.Timestamp(M5.START_DATE),
        # The test model may train on everything up to the test window, which
        # includes the validation period. That is correct: at deployment time
        # those actuals exist. Only the TEST window itself is withheld.
        train_end=pd.Timestamp(M5.VAL_END),
        test_start=pd.Timestamp(M5.TEST_START),
        test_end=pd.Timestamp(M5.TEST_END),
    )
    return {"validation": val, "test": test}


def rolling_origins(
    n_origins: int = 5,
    horizon: int = M5.HORIZON_DAYS,
    last_date: date | pd.Timestamp | None = None,
    first_date: date | pd.Timestamp | None = None,
    step: int | None = None,
    expanding: bool = True,
) -> list[Split]:
    """
    Non-overlapping rolling origins ending at `last_date`.

        |------------ train ------------|== test ==|          origin 4 (latest)
        |---------- train ----------|== test ==|               origin 3
        |-------- train --------|== test ==|                   origin 2
        ...

    Args:
        n_origins: how many origins. The spec requires >= 5 for a stability
            estimate; fewer than that and an origin-to-origin ranking flip is
            indistinguishable from noise.
        horizon: test-window length in days.
        last_date: latest date with actuals. Defaults to M5's END_DATE, i.e.
            2016-04-24 -- NOT the calendar end, which has no sales.
        step: days between consecutive origins. Defaults to `horizon`, which
            makes the test windows exactly non-overlapping.
        expanding: True gives an expanding training window (all history up to
            each origin), which is the realistic production setting. False
            gives a fixed-width rolling window, useful for testing whether old
            data still helps.

    IMPORTANT: the default `last_date` deliberately excludes the held-out test
    window is NOT applied here -- rolling origins are allowed to cover it only
    when explicitly requested, so pass `last_date=M5.VAL_END` while selecting
    models to keep the test period untouched.
    """
    if n_origins < 1:
        raise ValueError(f"n_origins must be >= 1, got {n_origins}")

    last = pd.Timestamp(last_date or M5.END_DATE)
    first = pd.Timestamp(first_date or M5.START_DATE)
    step = step or horizon

    splits: list[Split] = []
    for i in range(n_origins):
        test_end = last - pd.Timedelta(days=i * step)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)

        if expanding:
            train_start = first
        else:
            # Fixed window matched to the longest training span available at
            # the earliest origin, so every origin trains on the same amount.
            span = (last - pd.Timedelta(days=(n_origins - 1) * step + horizon)) - first
            train_start = train_end - span

        if train_end <= train_start:
            raise ValueError(
                f"origin {i} leaves no training data: need more history before {train_start.date()}"
            )

        splits.append(
            Split(
                name=f"origin_{n_origins - 1 - i}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    splits = list(reversed(splits))  # chronological order
    for s in splits:
        log.info("rolling_origin", detail=s.describe())
    return splits


def assert_no_temporal_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "date",
) -> None:
    """
    Fail loudly if any training row is dated at or after the first test row.

    Cheap to run and worth running on every split. The failure mode it catches
    -- an off-by-one that includes the first test day in training -- is
    invisible in aggregate metrics but shifts them consistently.
    """
    if train.empty or test.empty:
        raise ValueError(f"empty split: train={len(train)} rows, test={len(test)} rows")

    train_max = pd.to_datetime(train[date_col]).max()
    test_min = pd.to_datetime(test[date_col]).min()

    if train_max >= test_min:
        raise AssertionError(
            f"TEMPORAL LEAKAGE: training data extends to {train_max.date()} but the "
            f"test window opens {test_min.date()}. Training must end strictly before "
            f"the test window."
        )


def summarise_splits(splits: Sequence[Split]) -> pd.DataFrame:
    """Tabulate splits for the run manifest and the report."""
    return pd.DataFrame([s.to_dict() for s in splits])
