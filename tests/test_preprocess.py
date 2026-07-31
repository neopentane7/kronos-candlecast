"""Tests for the canonical schema (hard constraints 4 and 6).

The corruption tests are the A2 acceptance requirement: a deliberately broken row must
fail validation rather than reach training.
"""

import numpy as np
import pandas as pd
import pandera.errors as pae
import pytest

from common.preprocess import (
    CANONICAL_COLUMNS,
    JUGAAD_TZ_OFFSET,
    MAX_OVERNIGHT_GAP,
    UPSTREAM_CSV_COLUMNS,
    canonicalize_yfinance,
    clean,
    from_jugaad,
    from_yfinance,
    validate,
)


def failed_checks(err: pae.SchemaErrors) -> str:
    """The error strings pandera collected. Its `check` column holds `error=`, not `name=`."""
    return " | ".join(str(c) for c in err.failure_cases["check"].unique())


def make_canonical(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A small, well-formed canonical frame: gentle random walk, valid OHLC."""
    rng = np.random.default_rng(seed)
    close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    spread = close * rng.uniform(0.003, 0.015, n)
    open_ = close + rng.normal(0, 0.3, n) * spread
    df = pd.DataFrame(
        {
            "timestamps": pd.bdate_range("2022-01-03", periods=n),
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
        }
    )
    df["amount"] = df["close"] * df["volume"]
    return df[CANONICAL_COLUMNS]


def test_wellformed_frame_validates():
    df = make_canonical()
    out = validate(df)
    assert len(out) == len(df)
    assert list(out.columns) == CANONICAL_COLUMNS


# --- A2 acceptance: deliberately corrupted rows must fail ------------------------


def test_nan_close_fails():
    df = make_canonical()
    df.loc[10, "close"] = np.nan
    with pytest.raises(pae.SchemaErrors):
        validate(df)


def test_zero_volume_fails():
    df = make_canonical()
    df.loc[10, "volume"] = 0.0
    df.loc[10, "amount"] = 0.0
    with pytest.raises(pae.SchemaErrors):
        validate(df)


def test_negative_price_fails():
    df = make_canonical()
    df.loc[10, "low"] = -1.0
    with pytest.raises(pae.SchemaErrors):
        validate(df)


def test_high_below_close_fails():
    """low <= min(open, close) <= max(open, close) <= high."""
    df = make_canonical()
    df.loc[10, "high"] = df.loc[10, "close"] * 0.5
    with pytest.raises(pae.SchemaErrors) as exc:
        validate(df)
    assert "OHLC bounds violated" in failed_checks(exc.value)


def test_unsorted_timestamps_fail():
    df = make_canonical()
    df.loc[[5, 6], "timestamps"] = df.loc[[6, 5], "timestamps"].to_numpy()
    with pytest.raises(pae.SchemaErrors):
        validate(df)


def test_duplicate_timestamps_fail():
    df = make_canonical()
    df.loc[6, "timestamps"] = df.loc[5, "timestamps"]
    with pytest.raises(pae.SchemaErrors):
        validate(df)


def test_extra_column_fails():
    df = make_canonical()
    df["turnover_real"] = 1.0
    with pytest.raises(pae.SchemaErrors):
        validate(df)


def test_amount_must_be_the_close_times_volume_proxy():
    """Hard constraint 4: real turnover must not be able to sneak in as `amount`."""
    df = make_canonical()
    df.loc[10, "amount"] = df.loc[10, "amount"] * 1.5  # e.g. VWAP-based turnover
    with pytest.raises(pae.SchemaErrors) as exc:
        validate(df)
    assert "amount must equal close * volume" in failed_checks(exc.value)


# --- corporate-action detection -------------------------------------------------


def test_unadjusted_bonus_gap_is_rejected():
    """A 1:1 bonus halves the price overnight; that is what the gap check is for."""
    df = make_canonical()
    for col in ["open", "high", "low", "close"]:
        df.loc[30:, col] = df.loc[30:, col] / 2
    df["amount"] = df["close"] * df["volume"]
    with pytest.raises(pae.SchemaErrors) as exc:
        validate(df)
    assert "overnight gap exceeds" in failed_checks(exc.value)


def test_large_intraday_move_is_accepted():
    """INDUSINDBK rallied +44.7% intraday on 2020-03-26 after gapping only +5%.

    A real move opens near the previous close, so the gap check must let it through.
    """
    df = make_canonical()
    i = 30
    prev_close = df.loc[i - 1, "close"]
    df.loc[i, "open"] = prev_close * 1.05
    df.loc[i, "close"] = prev_close * 1.447
    df.loc[i, "high"] = prev_close * 1.50
    df.loc[i, "low"] = prev_close * 1.00
    df.loc[i:, "amount"] = df.loc[i:, "close"] * df.loc[i:, "volume"]
    # Keep the following bar's gap small so only the intraday move is under test.
    df.loc[i + 1, "open"] = df.loc[i, "close"] * 1.01
    df.loc[i + 1, "high"] = max(df.loc[i + 1, "high"], df.loc[i + 1, "open"])

    gap = df.loc[i, "open"] / prev_close - 1
    assert abs(gap) < MAX_OVERNIGHT_GAP
    validate(df)  # must not raise


# --- source adapters ------------------------------------------------------------


def test_from_yfinance_derives_amount_and_drops_tz():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=5, tz="Asia/Kolkata"))
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11, 12, 13, 14],
            "High": [11.0, 12, 13, 14, 15],
            "Low": [9.0, 10, 11, 12, 13],
            "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Volume": [100.0, 200, 300, 400, 500],
        },
        index=idx,
    )
    out = from_yfinance(raw)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["timestamps"].dt.tz is None
    assert np.allclose(out["amount"], out["close"] * out["volume"])


def test_from_jugaad_corrects_the_off_by_one_day_timestamp():
    """jugaad returns 18:30:00 (midnight IST in UTC); normalizing loses a day."""
    raw = pd.DataFrame(
        {
            "DATE": [pd.Timestamp("2024-01-30 18:30:00")],
            "OPEN": [100.0],
            "HIGH": [105.0],
            "LOW": [99.0],
            "CLOSE": [104.0],
            "VOLUME": [1000.0],
        }
    )
    out = from_jugaad(raw)
    assert out["timestamps"].iloc[0] == pd.Timestamp("2024-01-31")
    assert pd.Timedelta(hours=5, minutes=30) == JUGAAD_TZ_OFFSET


# --- cleaning -------------------------------------------------------------------


def test_clean_drops_fabricated_zero_volume_bars():
    """yfinance emits flat zero-volume bars on NSE holidays; they must not survive."""
    df = make_canonical()
    df.loc[20, ["open", "high", "low", "close"]] = df.loc[20, "close"]
    df.loc[20, ["volume", "amount"]] = 0.0
    dropped_date = df.loc[20, "timestamps"]

    out, report = clean(df)
    assert report["dropped_zero_volume"] == 1
    assert str(dropped_date.date()) in report["dropped_zero_volume_dates"]
    assert dropped_date not in set(out["timestamps"])
    assert len(out) == len(df) - 1


def test_clean_keeps_muhurat_sessions_the_calendar_does_not_know():
    """XBOM omits the Diwali Muhurat session, but those bars are real (constraint 6)."""
    muhurat = pd.Timestamp("2024-11-01")  # a real NSE session, absent from XBOM
    df = make_canonical(n=5)
    df["timestamps"] = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-10-30"),
            pd.Timestamp("2024-10-31"),
            muhurat,
            pd.Timestamp("2024-11-04"),
            pd.Timestamp("2024-11-05"),
        ]
    )
    out, report = clean(df)
    assert muhurat in set(out["timestamps"]), "a real Muhurat session was dropped"
    assert report["kept_off_calendar"] >= 1
    assert str(muhurat.date()) in report["kept_off_calendar_dates"]


def test_canonicalize_yfinance_bundles_adapter_and_clean():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=4))
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11, 12, 13],
            "High": [11.0, 12, 13, 14],
            "Low": [9.0, 10, 11, 12],
            "Close": [10.5, 11.5, 12.5, 13.5],
            "Volume": [100.0, 0.0, 300, 400],  # row 1 is a fabricated bar
        },
        index=idx,
    )
    out, report = canonicalize_yfinance(raw)
    assert len(out) == 3
    assert report["dropped_zero_volume"] == 1


def test_upstream_csv_order_puts_close_before_high():
    """Upstream finetune_csv's column order differs from ours; build_dataset reorders."""
    assert UPSTREAM_CSV_COLUMNS == [
        "timestamps",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
    ]
    assert UPSTREAM_CSV_COLUMNS.index("close") < UPSTREAM_CSV_COLUMNS.index("high")
    assert set(UPSTREAM_CSV_COLUMNS) == set(CANONICAL_COLUMNS)
