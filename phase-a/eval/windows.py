"""Rolling evaluation windows built from the Parquet corpus.

Splits partition forecast *targets*, not input windows. A window's 30-step target lies
inside the split; its 400-bar lookback is drawn from whatever precedes it, crossing the
split boundary backwards. That is not leakage -- at serving time the model always
conditions on the most recent bars regardless of period. What must not leak is *training*
on target-period data.

Windows are keyed by forecast start date, which is also the block identifier for the
bootstrap: every ticker forecasting from the same date shares that date's market
conditions, so they succeed or fail together and must be resampled together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from common.preprocess import CANONICAL_COLUMNS
from common.splits import SPLITS

LOOKBACK = 400
HORIZON = 30
STRIDE = 30


@dataclass
class EvalGrid:
    """One evaluation grid: aligned windows plus everything needed to score them."""

    tickers: list[str]
    start_dates: list[pd.Timestamp]
    x_frames: list[pd.DataFrame]
    x_timestamps: list[pd.Series]
    y_timestamps: list[pd.Series]
    y_close: np.ndarray  # (n_windows, horizon) realized closes
    history_close: np.ndarray  # (n_windows, lookback) for the baselines
    atr_pct: np.ndarray  # (n_windows,) volatility of the lookback, for regime slices
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tickers)

    @property
    def block_ids(self) -> np.ndarray:
        """Per-observation block labels, shaped like ``y_close``.

        The forecast start date is the block: all tickers and all horizon steps sharing
        a date move together.
        """
        codes = pd.factorize(pd.to_datetime(self.start_dates))[0]
        return np.repeat(codes[:, None], self.y_close.shape[1], axis=1)

    def atr_tercile(self) -> np.ndarray:
        """0 = calm, 1 = mid, 2 = volatile, by lookback ATR across the whole grid."""
        edges = np.quantile(self.atr_pct, [1 / 3, 2 / 3])
        return np.digitize(self.atr_pct, edges)

    def subsample(self, n: int, seed: int = 0) -> EvalGrid:
        """A smaller grid, stratified by volatility regime and spread over time.

        Used for iteration and for the sampling-policy sweep, where the question is a
        relative comparison rather than a headline number.
        """
        if n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        terciles = self.atr_tercile()
        picks: list[int] = []
        for t in (0, 1, 2):
            idx = np.flatnonzero(terciles == t)
            take = min(len(idx), n // 3 + (1 if t < n % 3 else 0))
            picks.extend(rng.choice(idx, size=take, replace=False).tolist())
        picks = sorted(picks)
        return EvalGrid(
            tickers=[self.tickers[i] for i in picks],
            start_dates=[self.start_dates[i] for i in picks],
            x_frames=[self.x_frames[i] for i in picks],
            x_timestamps=[self.x_timestamps[i] for i in picks],
            y_timestamps=[self.y_timestamps[i] for i in picks],
            y_close=self.y_close[picks],
            history_close=self.history_close[picks],
            atr_pct=self.atr_pct[picks],
            meta={
                **self.meta,
                "subsampled_from": len(self),
                "subsample_seed": seed,
                # Recount rather than inheriting: the parent's totals describe a grid
                # this one is no longer.
                "n_windows": len(picks),
                "n_tickers": len({self.tickers[i] for i in picks}),
                "n_distinct_start_dates": len({self.start_dates[i] for i in picks}),
            },
        )


def average_true_range_pct(df: pd.DataFrame) -> float:
    """ATR over the frame as a fraction of mean close, so it compares across tickers."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.mean() / df["close"].mean())


def load_corpus(parquet_root) -> pd.DataFrame:
    glob = (parquet_root / "*" / "*.parquet").as_posix()
    cols = ", ".join(CANONICAL_COLUMNS)
    return duckdb.sql(
        f"SELECT ticker, {cols} FROM read_parquet('{glob}', hive_partitioning=true) "
        "ORDER BY ticker, timestamps"
    ).to_df()


def build_grid(
    corpus: pd.DataFrame,
    split: str,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    stride: int = STRIDE,
    tickers: list[str] | None = None,
) -> EvalGrid:
    """Enumerate every admissible rolling window whose target lies inside ``split``."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}")
    start_bound, end_bound = (pd.Timestamp(d) for d in SPLITS[split])

    out = EvalGrid([], [], [], [], [], np.empty((0, horizon)), np.empty((0, lookback)), np.empty(0))
    y_close, hist_close, atrs = [], [], []

    for ticker, g in corpus.groupby("ticker", sort=True):
        if tickers is not None and ticker not in tickers:
            continue
        g = g.reset_index(drop=True)
        in_split = np.flatnonzero((g["timestamps"] >= start_bound) & (g["timestamps"] <= end_bound))
        if len(in_split) == 0:
            continue

        for target_start in range(in_split[0], in_split[-1] + 1, stride):
            # The whole target must lie inside the split and inside the data.
            target_end = target_start + horizon
            if target_end > len(g) or target_start < lookback:
                continue
            if g.loc[target_end - 1, "timestamps"] > end_bound:
                continue

            x = g.iloc[target_start - lookback : target_start]
            y = g.iloc[target_start:target_end]

            out.tickers.append(ticker)
            out.start_dates.append(y["timestamps"].iloc[0])
            out.x_frames.append(x[CANONICAL_COLUMNS[1:]].reset_index(drop=True))
            out.x_timestamps.append(x["timestamps"].reset_index(drop=True))
            out.y_timestamps.append(y["timestamps"].reset_index(drop=True))
            y_close.append(y["close"].to_numpy(dtype=float))
            hist_close.append(x["close"].to_numpy(dtype=float))
            atrs.append(average_true_range_pct(x))

    out.y_close = np.array(y_close) if y_close else np.empty((0, horizon))
    out.history_close = np.array(hist_close) if hist_close else np.empty((0, lookback))
    out.atr_pct = np.array(atrs)
    out.meta = {
        "split": split,
        "split_range": SPLITS[split],
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "n_windows": len(out.tickers),
        "n_tickers": len(set(out.tickers)),
        "n_distinct_start_dates": len(set(out.start_dates)),
    }
    return out
