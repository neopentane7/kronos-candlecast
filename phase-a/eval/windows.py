"""Rolling evaluation windows built from the Parquet corpus.

Splits partition forecast *targets*, not input windows. A window's 30-step target lies
inside the split; its 400-bar lookback is drawn from whatever precedes it, crossing the
split boundary backwards. That is not leakage -- at serving time the model always
conditions on the most recent bars regardless of period. What must not leak is *training*
on target-period data.

Windows are keyed by forecast start date, which is also the block identifier for the
bootstrap: every ticker forecasting from the same date shares that date's market
conditions, so they succeed or fail together and must be resampled together.

Memory
------
Windows are stored as **integer offsets into a compact per-ticker array**, and the pandas
objects the sampler needs are built one batch at a time. An earlier version materialised
every window eagerly -- 708 DataFrames plus 1,416 Series held for the run's lifetime --
which was fine at 45 windows and drove the machine into paging at 708: 16 GB resident,
1.9 GB free, the GPU idle at 0% while one core thrashed. The underlying numbers are only
about 3 MB; the rest was pandas per-object overhead.
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

FEATURE_COLUMNS = CANONICAL_COLUMNS[1:]  # open, high, low, close, volume, amount


@dataclass
class EvalGrid:
    """Windows as offsets into per-ticker arrays, plus everything needed to score them."""

    features: dict[str, np.ndarray]  # ticker -> (n_rows, 6) float64
    timestamps: dict[str, pd.DatetimeIndex]  # ticker -> (n_rows,)
    tickers: list[str]  # per window
    offsets: list[int]  # per window: row index of the first target bar
    start_dates: list[pd.Timestamp]
    y_close: np.ndarray  # (n_windows, horizon)
    history_close: np.ndarray  # (n_windows, lookback)
    atr_pct: np.ndarray  # (n_windows,)
    lookback: int = LOOKBACK
    horizon: int = HORIZON
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tickers)

    # -- lazy window construction --------------------------------------------------

    def batch(self, start: int, stop: int) -> tuple[list, list, list]:
        """Build the pandas objects for windows ``[start, stop)`` on demand.

        Returns ``(df_list, x_timestamp_list, y_timestamp_list)`` in the shape the sampler
        expects. Nothing is cached: the caller holds one batch at a time.
        """
        dfs, x_ts, y_ts = [], [], []
        for i in range(start, min(stop, len(self))):
            t, off = self.tickers[i], self.offsets[i]
            feats = self.features[t]
            stamps = self.timestamps[t]
            lo = off - self.lookback
            dfs.append(pd.DataFrame(feats[lo:off], columns=FEATURE_COLUMNS))
            x_ts.append(pd.Series(stamps[lo:off], name="timestamps").reset_index(drop=True))
            y_ts.append(
                pd.Series(stamps[off : off + self.horizon], name="timestamps").reset_index(
                    drop=True
                )
            )
        return dfs, x_ts, y_ts

    # -- alignment -----------------------------------------------------------------

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

    def subsample_by_block(self, per_block: int, seed: int = 0) -> EvalGrid:
        """A balanced panel: the same ``per_block`` tickers at every forecast date.

        ``subsample`` stratifies by volatility and lets date coverage fall where it may.
        A sampling-policy sweep needs the opposite guarantee -- every arm scored on
        *identical* windows, with all date blocks present so the horizon curve is not
        dominated by one period.

        Because the arms are compared against each other rather than against the full
        grid, the panel does not have to be representative; it has to be identical across
        arms and spread over time. That is also why a sweep must never be compared to
        full-grid aggregates: different windows, different level.
        """
        names = sorted({t for t in self.tickers})
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(names, size=min(per_block, len(names)), replace=False).tolist())
        picks = sorted(i for i, t in enumerate(self.tickers) if t in chosen)
        if not picks:
            raise ValueError("block-balanced subsample selected no windows")

        blocks = {self.start_dates[i] for i in picks}
        sub = EvalGrid(
            features={k: v for k, v in self.features.items() if k in chosen},
            timestamps={k: v for k, v in self.timestamps.items() if k in chosen},
            tickers=[self.tickers[i] for i in picks],
            offsets=[self.offsets[i] for i in picks],
            start_dates=[self.start_dates[i] for i in picks],
            y_close=self.y_close[picks],
            history_close=self.history_close[picks],
            atr_pct=self.atr_pct[picks],
            lookback=self.lookback,
            horizon=self.horizon,
        )
        sub.meta = {
            **self.meta,
            "subsampled_from": len(self),
            "subsample_seed": seed,
            "subsample_mode": "block_balanced",
            "panel_tickers": sorted(chosen),
            "n_windows": len(picks),
            "n_tickers": len(chosen),
            "n_blocks": len(blocks),
        }
        return sub

    def subsample(self, n: int, seed: int = 0) -> EvalGrid:
        """A smaller grid, stratified by volatility regime and spread over time."""
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
        kept = {self.tickers[i] for i in picks}
        return EvalGrid(
            # Share the backing arrays rather than copying; they are read-only here.
            features={k: v for k, v in self.features.items() if k in kept},
            timestamps={k: v for k, v in self.timestamps.items() if k in kept},
            tickers=[self.tickers[i] for i in picks],
            offsets=[self.offsets[i] for i in picks],
            start_dates=[self.start_dates[i] for i in picks],
            y_close=self.y_close[picks],
            history_close=self.history_close[picks],
            atr_pct=self.atr_pct[picks],
            lookback=self.lookback,
            horizon=self.horizon,
            meta={
                **self.meta,
                "subsampled_from": len(self),
                "subsample_seed": seed,
                "n_windows": len(picks),
                "n_tickers": len(kept),
                "n_distinct_start_dates": len({self.start_dates[i] for i in picks}),
            },
        )


def average_true_range_pct(high, low, close) -> float:
    """ATR over the window as a fraction of mean close, so it compares across tickers."""
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(
        np.stack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0
    )
    return float(np.nanmean(tr) / close.mean())


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

    features: dict[str, np.ndarray] = {}
    stamps: dict[str, pd.DatetimeIndex] = {}
    win_tickers: list[str] = []
    offsets: list[int] = []
    start_dates: list[pd.Timestamp] = []
    y_close, hist_close, atrs = [], [], []

    close_i = FEATURE_COLUMNS.index("close")
    high_i = FEATURE_COLUMNS.index("high")
    low_i = FEATURE_COLUMNS.index("low")

    for ticker, g in corpus.groupby("ticker", sort=True):
        if tickers is not None and ticker not in tickers:
            continue
        ts = pd.DatetimeIndex(g["timestamps"])
        feats = g[FEATURE_COLUMNS].to_numpy(dtype=float)

        in_split = np.flatnonzero((ts >= start_bound) & (ts <= end_bound))
        if len(in_split) == 0:
            continue

        used = False
        for target_start in range(in_split[0], in_split[-1] + 1, stride):
            target_end = target_start + horizon
            if target_end > len(feats) or target_start < lookback:
                continue
            if ts[target_end - 1] > end_bound:
                continue

            win_tickers.append(ticker)
            offsets.append(int(target_start))
            start_dates.append(ts[target_start])
            y_close.append(feats[target_start:target_end, close_i])
            hist_close.append(feats[target_start - lookback : target_start, close_i])
            atrs.append(
                average_true_range_pct(
                    feats[target_start - lookback : target_start, high_i],
                    feats[target_start - lookback : target_start, low_i],
                    feats[target_start - lookback : target_start, close_i],
                )
            )
            used = True

        if used:  # only retain arrays a window actually references
            features[ticker] = feats
            stamps[ticker] = ts

    grid = EvalGrid(
        features=features,
        timestamps=stamps,
        tickers=win_tickers,
        offsets=offsets,
        start_dates=start_dates,
        y_close=np.array(y_close) if y_close else np.empty((0, horizon)),
        history_close=np.array(hist_close) if hist_close else np.empty((0, lookback)),
        atr_pct=np.array(atrs),
        lookback=lookback,
        horizon=horizon,
    )
    grid.meta = {
        "split": split,
        "split_range": SPLITS[split],
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "n_windows": len(win_tickers),
        "n_tickers": len(features),
        "n_distinct_start_dates": len(set(start_dates)),
    }
    return grid
