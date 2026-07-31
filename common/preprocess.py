"""Canonical schema + validation. The one place market data is shaped (constraint 6).

Train time and serve time both call into here, so a source difference cannot silently
become a feature difference. Every source adapter returns the same canonical frame and
every canonical frame is validated by the same ``pandera`` schema.

Canonical column order (hard constraint 4)::

    timestamps, open, high, low, close, volume, amount     with amount = close * volume

``amount`` is deliberately the ``close * volume`` proxy rather than real exchange
turnover, in both phases. Measured against NSE's own ``VALUE`` field the proxy has a
median ratio of 1.0009, so the parity gain costs ~0.1% in fidelity.

Price adjustment policy
-----------------------
Canonical prices are **split/bonus back-adjusted**, as yfinance returns them.

This resolves a conflict in the project's data-source strategy. jugaad-data returns raw
as-traded NSE prices; yfinance back-adjusts. For RELIANCE's 1:1 bonus (ex-date
2024-10-28) the two differ by exactly 2.0x on price and 0.5x on volume, and the raw
series shows a -49.76% single-day "crash" where the adjusted series shows +0.49%. Both
series pass any schema you can write -- they simply describe different worlds -- so the
shared schema alone does NOT buy train/serve parity, as the resolved-conflict note
assumed. Choosing the adjusted series everywhere buys it by construction. On days with
no corporate action the two sources agree on returns to 5 decimal places, so nothing is
given up. jugaad-data remains useful as an independent cross-check and as the source of
real turnover for the ablation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

from common.calendar import nse_calendar

CANONICAL_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Upstream finetune_csv expects close BEFORE high (see its README sample table and
# finetune_csv/data/HK_ali_09988_kline_5min_all.csv). Ours is the OHLC order above, so
# build_dataset.py must reorder on export -- that is what this constant is for.
UPSTREAM_CSV_COLUMNS = ["timestamps", "open", "close", "high", "low", "volume", "amount"]

# Corporate-action detection (hard constraint 6), calibrated on the corpus rather than
# guessed. Measured over 108,880 bars from 52 NIFTY-100 names, 2018-01 to 2026-06:
#
#   |overnight gap| = |open_t / close_{t-1} - 1|   p99 3.6%   p99.99 10.0%   max 16.2%
#   |close-to-close|                               p99.99 20.1%              max 28.2%
#
# An unadjusted corporate action is an *overnight* discontinuity: a 1:1 bonus gaps -50%,
# and VEDL's 2026-04-30 demerger gapped -62.6%. The largest genuine gap in the corpus is
# ONGC's -16.2% on 2020-03-23, so a 25% gap bound has ~34 points of clear air on both
# sides.
#
# A close-to-close bound cannot do this job: INDUSINDBK genuinely rallied +44.7% intraday
# on 2020-03-26 (gapping only +5.0% first), which is a *larger* move than the -50% a
# bonus produces. Any threshold tight enough to catch the bonus rejects the real rally.
# So the gap is the corporate-action test and close-to-close is only a loose backstop
# against grossly corrupt data.
MAX_OVERNIGHT_GAP = 0.25
MAX_DAILY_MOVE = 0.60

# jugaad-data returns tz-naive timestamps at 18:30:00, i.e. midnight IST expressed in
# UTC. Normalizing without this shift moves every bar to the PREVIOUS calendar day.
JUGAAD_TZ_OFFSET = pd.Timedelta(hours=5, minutes=30)

_TOL = 1e-6


def _ohlc_consistent(df: pd.DataFrame) -> pd.Series:
    """low <= min(open, close) <= max(open, close) <= high, within float tolerance."""
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    scale = df["close"].abs() * _TOL
    return (
        (df["low"] <= body_low + scale)
        & (body_high <= df["high"] + scale)
        & (df["low"] <= df["high"] + scale)
    )


def _daily_move_within_bounds(df: pd.DataFrame, max_move: float = MAX_DAILY_MOVE) -> pd.Series:
    # fill_method=None: a NaN close must stay NaN here and be caught by the nullable
    # check, not be forward-filled into a spurious 0% move.
    returns = df["close"].pct_change(fill_method=None)
    return returns.isna() | (returns.abs() <= max_move)


def _overnight_gap_within_bounds(df: pd.DataFrame, max_gap: float = MAX_OVERNIGHT_GAP) -> pd.Series:
    """The corporate-action test: open_t against close_{t-1}."""
    gap = df["open"] / df["close"].shift(1) - 1.0
    return gap.isna() | (gap.abs() <= max_gap)


def _amount_is_close_times_volume(df: pd.DataFrame) -> pd.Series:
    """Mechanically enforce hard constraint 4, so real turnover cannot leak in."""
    return pd.Series(
        np.isclose(df["amount"], df["close"] * df["volume"], rtol=1e-6, atol=1.0),
        index=df.index,
    )


def canonical_schema(
    max_daily_move: float = MAX_DAILY_MOVE,
    max_overnight_gap: float = MAX_OVERNIGHT_GAP,
) -> pa.DataFrameSchema:
    """The schema that validates market data at train time and at serve time alike."""
    positive_price = pa.Column(
        float, checks=pa.Check.gt(0), nullable=False, coerce=True, required=True
    )
    return pa.DataFrameSchema(
        columns={
            "timestamps": pa.Column("datetime64[ns]", nullable=False, coerce=True),
            "open": positive_price,
            "high": positive_price,
            "low": positive_price,
            "close": positive_price,
            # volume > 0 per constraint 6: a zero-volume session means a halt or a
            # data gap, never a tradeable bar.
            "volume": pa.Column(float, checks=pa.Check.gt(0), nullable=False, coerce=True),
            "amount": pa.Column(float, checks=pa.Check.gt(0), nullable=False, coerce=True),
        },
        checks=[
            pa.Check(
                lambda df: df["timestamps"].is_monotonic_increasing,
                name="timestamps_monotonic_increasing",
                error="timestamps must be sorted ascending",
            ),
            pa.Check(
                lambda df: not df["timestamps"].duplicated().any(),
                name="timestamps_unique",
                error="duplicate timestamps",
            ),
            pa.Check(_ohlc_consistent, name="ohlc_consistent", error="OHLC bounds violated"),
            pa.Check(
                lambda df: _overnight_gap_within_bounds(df, max_overnight_gap),
                name="overnight_gap_within_bounds",
                error=(
                    f"overnight gap exceeds {max_overnight_gap:.0%} (unadjusted corporate action?)"
                ),
            ),
            pa.Check(
                lambda df: _daily_move_within_bounds(df, max_daily_move),
                name="daily_move_within_bounds",
                error=f"close-to-close move exceeds {max_daily_move:.0%}",
            ),
            pa.Check(
                _amount_is_close_times_volume,
                name="amount_is_close_times_volume",
                error="amount must equal close * volume (hard constraint 4)",
            ),
        ],
        strict=True,
        ordered=True,
        name="nse_daily_canonical",
    )


def validate(
    df: pd.DataFrame,
    max_daily_move: float = MAX_DAILY_MOVE,
    max_overnight_gap: float = MAX_OVERNIGHT_GAP,
) -> pd.DataFrame:
    """Validate a canonical frame, collecting every failure rather than the first."""
    return canonical_schema(max_daily_move, max_overnight_gap).validate(df, lazy=True)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop duplicate sessions, and derive ``amount``. Shared by all adapters."""
    df = df.dropna(subset=CANONICAL_COLUMNS[:-1])
    df = df.sort_values("timestamps")
    df = df.drop_duplicates(subset="timestamps", keep="last").reset_index(drop=True)
    df["amount"] = df["close"] * df["volume"]
    return df[CANONICAL_COLUMNS]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows that are not tradeable bars, returning the frame and what was removed.

    **Zero volume is the only drop rule**, and it is enough. yfinance fabricates
    placeholder bars on NSE market holidays (2026-01-15, 2026-05-01, 2026-05-28,
    2026-06-26 were all observed market-wide), but every one of them is flat --
    ``open == high == low == close`` -- with ``volume == 0``, because no trading
    happened. The same rule also catches defective bars on genuine sessions, e.g.
    2025-03-18 was flat and zero-volume for RELIANCE/TCS/HDFCBANK but not ITC.
    Constraint 6 requires ``volume > 0`` regardless.

    Rows the exchange calendar does not recognise are **reported but kept**, because
    dropping them would discard real data: ``XBOM`` omits the annual Muhurat (Diwali)
    session, and those bars are genuine -- 2024-11-01 carries 2.1M shares of RELIANCE
    volume and NSE's own feed reports the identical close of 1338.65. Consumers that
    need calendar-aligned timestamps (the eval harness, the nightly job) must therefore
    tolerate a session the calendar cannot generate.

    Dropping rather than forward-filling is deliberate: an invented bar would become a
    training sample. A gap is visible in the manifest; a fabricated bar is not.
    """
    if df.empty:
        return df, {
            "rows_in": 0,
            "rows_out": 0,
            "dropped_zero_volume": 0,
            "kept_off_calendar": 0,
        }

    rows_in = len(df)

    zero_vol = df["volume"] <= 0
    zero_vol_dates = [str(t.date()) for t in df.loc[zero_vol, "timestamps"]]
    df = df.loc[~zero_vol].reset_index(drop=True)

    # Informational only: sessions the packaged calendar has no record of.
    cal = nse_calendar()
    in_range = df["timestamps"] <= cal.last_session
    off_calendar = [
        str(ts.date())
        for ts, ok in zip(df["timestamps"], in_range, strict=True)
        if ok and not cal.is_session(ts)
    ]

    report = {
        "rows_in": rows_in,
        "rows_out": len(df),
        "dropped_zero_volume": len(zero_vol_dates),
        "dropped_zero_volume_dates": zero_vol_dates[:20],
        "kept_off_calendar": len(off_calendar),
        "kept_off_calendar_dates": off_calendar[:20],
    }
    return df, report


def canonicalize_yfinance(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Adapter + clean in one call. Train time and serve time both use this.

    Bundling the two steps is the mechanical part of constraint 6: there is no way to
    shape yfinance data without also applying the cleaning rules.
    """
    return clean(from_yfinance(raw))


def canonicalize_jugaad(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Adapter + clean for the cross-check source. Prices remain unadjusted."""
    return clean(from_jugaad(raw))


def from_yfinance(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize ``yf.Ticker(...).history()`` output. This is the canonical source.

    Expects ``auto_adjust=False``: yfinance still back-adjusts prices for splits and
    bonuses in ``Close`` (that is not what ``auto_adjust`` controls), which is exactly
    the adjustment policy documented above. ``Adj Close`` additionally folds in
    dividends and is deliberately ignored -- dividend adjustment would break the
    OHLC-consistency invariant, since only the close would be restated.
    """
    if raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    missing = {"Open", "High", "Low", "Close", "Volume"} - set(raw.columns)
    if missing:
        raise ValueError(f"yfinance frame missing columns: {sorted(missing)}")

    index = pd.DatetimeIndex(raw.index)
    if index.tz is not None:
        index = index.tz_localize(None)

    df = pd.DataFrame(
        {
            "timestamps": index.normalize(),
            "open": raw["Open"].to_numpy(dtype=float),
            "high": raw["High"].to_numpy(dtype=float),
            "low": raw["Low"].to_numpy(dtype=float),
            "close": raw["Close"].to_numpy(dtype=float),
            "volume": raw["Volume"].to_numpy(dtype=float),
            "amount": np.nan,
        }
    )
    return _finalize(df)


def from_jugaad(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize ``jugaad_data.nse.stock_df`` output.

    Cross-check / ablation source only -- prices here are raw as-traded, so a corpus
    built from this path is NOT interchangeable with the yfinance one across corporate
    actions. Applies the +5:30 fix for the off-by-one-day ``DATE`` column.
    """
    if raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    missing = {"DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"} - set(raw.columns)
    if missing:
        raise ValueError(f"jugaad frame missing columns: {sorted(missing)}")

    df = pd.DataFrame(
        {
            "timestamps": (pd.to_datetime(raw["DATE"]) + JUGAAD_TZ_OFFSET).dt.normalize(),
            "open": raw["OPEN"].astype(float).to_numpy(),
            "high": raw["HIGH"].astype(float).to_numpy(),
            "low": raw["LOW"].astype(float).to_numpy(),
            "close": raw["CLOSE"].astype(float).to_numpy(),
            "volume": raw["VOLUME"].astype(float).to_numpy(),
            "amount": np.nan,
        }
    )
    return _finalize(df)
