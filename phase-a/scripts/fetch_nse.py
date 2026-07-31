"""Fetch the NSE daily corpus into partitioned Parquet, with a manifest.

yfinance is the canonical source for prices (see the adjustment-policy note in
``common/preprocess.py``). jugaad-data is available as an independent cross-check via
``--cross-check``: it reports where NSE's own as-traded prices disagree with yfinance's
adjusted series beyond what corporate actions explain, and it is the source of real
turnover for the constraint-4 ablation.

Every ticker is canonicalized and validated before it is written. A ticker that fails
validation is skipped and recorded in the manifest with its reason -- never silently
substituted or patched.

Usage (PowerShell):
    uv run python phase-a/scripts/fetch_nse.py
    uv run python phase-a/scripts/fetch_nse.py --tickers RELIANCE TCS --cross-check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandera.errors as pae  # noqa: E402

from common.calendar import sessions_in_range  # noqa: E402
from common.preprocess import canonicalize_jugaad, canonicalize_yfinance, validate  # noqa: E402
from common.results import DISCLAIMER  # noqa: E402
from common.splits import CORPUS_START, SPLITS, slice_split  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "parquet"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"

# Survivors: liquid names that are NIFTY-100 constituents today. Selecting only these
# and backfilling to 2018 is survivorship bias, which is why SURVIVORSHIP_ADDITIONS
# below exists -- see the note there before editing either list.
# NSE symbols; yfinance needs the `.NS` suffix.
# fmt: off
CURRENT_CONSTITUENTS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO", "ONGC",
    # TATAMOTORS is absent: Yahoo 404s the symbol after the 2025 demerger, and the
    # successor TMPV.NS has only ~1 month of history. TATAPOWER takes its place.
    "NTPC", "POWERGRID", "TATAPOWER", "TATASTEEL", "JSWSTEEL",
    "HINDALCO", "COALINDIA", "NESTLEIND", "TECHM", "HCLTECH",
    "ADANIENT", "ADANIPORTS", "GRASIM", "CIPLA", "DRREDDY",
    "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "BAJAJFINSV",
    "INDUSINDBK", "TATACONSUM", "APOLLOHOSP", "PIDILITIND", "DABUR",
    "GODREJCP", "HAVELLS", "SIEMENS", "DMART", "VEDL",
    "GAIL", "IOC", "BPCL", "AMBUJACEM", "MARICO",
]

# Names that WERE NIFTY-100 constituents during 2018-2021 and subsequently collapsed,
# were de-rated, or left the index. Without them the panel silently excludes exactly the
# regimes a calibration study most needs to be honest about.
#
# Measured, on the same 2018-2026 window:
#
#   YESBANK     12 bars moving >20%, max drawdown -97.3%
#   IDEA        20 bars moving >20%, max drawdown -95.8%
#   ZEEL         5 bars moving >20%, max drawdown -88.9%
#   BANDHANBNK   2 bars moving >20%, max drawdown -82.1%
#   RBLBANK      2 bars moving >20%, max drawdown -89.0%
#
# For comparison the 54 survivors contribute 17 such bars in total across 113,068 rows.
# These five roughly triple the tail population for under 10% more data, and all pass
# the canonical schema unmodified -- nothing technical excluded them, only the choice
# of universe.
#
# This is NOT a true point-in-time reconstruction, which would need historical index
# membership tables that are not freely available. It is a documented approximation and
# must be described as such wherever coverage numbers are published.
SURVIVORSHIP_ADDITIONS = [
    "YESBANK", "IDEA", "ZEEL", "BANDHANBNK", "RBLBANK",
]

UNIVERSE = CURRENT_CONSTITUENTS + SURVIVORSHIP_ADDITIONS
# fmt: on

MIN_ROWS = 1400  # acceptance bar per ticker


def fetch_yfinance(
    symbol: str, start: str, end: str, retries: int = 3
) -> tuple[pd.DataFrame, dict]:
    """Canonical price history for ``symbol``, with retries on transient failures."""
    import yfinance as yf

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(f"{symbol}.NS").history(
                start=start, end=end, auto_adjust=False, actions=True
            )
            if raw.empty:
                raise ValueError("yfinance returned no rows")
            return canonicalize_yfinance(raw)
        except Exception as exc:  # noqa: BLE001 - retried, then reported in the manifest
            last_exc = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"yfinance failed after {retries} attempts: {last_exc}")


def fetch_jugaad(symbol: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    """As-traded NSE history for cross-checking. Unadjusted -- not the canonical path."""
    from jugaad_data.nse import stock_df

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = stock_df(
                symbol=symbol,
                from_date=date.fromisoformat(start),
                to_date=date.fromisoformat(end),
                series="EQ",
            )
            if raw is None or raw.empty:
                raise ValueError("jugaad-data returned no rows")
            return canonicalize_jugaad(raw)[0]
        except Exception as exc:  # noqa: BLE001 - cross-check is best-effort
            last_exc = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"jugaad-data failed after {retries} attempts: {last_exc}")


def cross_check(canonical: pd.DataFrame, as_traded: pd.DataFrame) -> dict:
    """Compare adjusted vs as-traded returns on shared sessions.

    Corporate actions make the price *levels* differ by design, so we compare daily
    returns instead: those agree to ~1e-5 except on an action's ex-date.
    """
    merged = canonical.merge(as_traded, on="timestamps", suffixes=("_adj", "_raw"))
    if len(merged) < 2:
        return {"shared_sessions": len(merged), "status": "insufficient_overlap"}

    r_adj = merged["close_adj"].pct_change()
    r_raw = merged["close_raw"].pct_change()
    diff = (r_adj - r_raw).abs().dropna()
    outliers = diff[diff > 1e-3]
    return {
        "shared_sessions": int(len(merged)),
        "max_return_diff": float(diff.max()),
        "median_return_diff": float(diff.median()),
        "action_dates": [
            str(merged.loc[i, "timestamps"].date()) for i in outliers.index if i in merged.index
        ],
        "status": "ok",
    }


def session_gaps(df: pd.DataFrame, start: str, end: str) -> list[str]:
    """Trading sessions the exchange calendar has but this ticker does not."""
    expected = sessions_in_range(max(start, df["timestamps"].min().strftime("%Y-%m-%d")), end)
    have = pd.DatetimeIndex(df["timestamps"])
    return [str(d.date()) for d in expected.difference(have)]


def write_partition(df: pd.DataFrame, symbol: str) -> Path:
    """Hive-partitioned Parquet: ``data/parquet/ticker=<SYMBOL>/data.parquet``."""
    out_dir = DATA_ROOT / f"ticker={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None, help="defaults to the full universe")
    ap.add_argument("--start", default=CORPUS_START)
    ap.add_argument("--end", default=SPLITS["test"][1])
    ap.add_argument("--cross-check", action="store_true", help="also fetch jugaad-data")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    args = ap.parse_args()

    tickers = args.tickers if args.tickers else UNIVERSE
    # yfinance treats `end` as exclusive.
    fetch_end = (pd.Timestamp(args.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    entries, skipped = [], []

    for i, symbol in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {symbol} ... ", end="", flush=True)
        try:
            df, clean_report = fetch_yfinance(symbol, args.start, fetch_end)
            df = validate(df)
        except pae.SchemaErrors as exc:
            # Summarize which checks failed and how often, not pandera's full dump.
            fc = exc.failure_cases
            counts = fc.groupby(["column", "check"], dropna=False).size()
            reason = "; ".join(f"{col}/{chk}: {n} rows" for (col, chk), n in counts.items())
            print(f"SKIP (schema: {reason[:100]})")
            skipped.append(
                {
                    "ticker": symbol,
                    "stage": "validate",
                    "reason": reason[:600],
                    "example_failures": fc.head(5).to_dict("records"),
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001 - reported, never substituted
            reason = str(exc).splitlines()[0][:400]
            print(f"SKIP ({type(exc).__name__}: {reason[:80]})")
            skipped.append({"ticker": symbol, "stage": "fetch", "reason": reason})
            continue

        if len(df) < args.min_rows:
            print(f"SKIP (only {len(df)} rows, need {args.min_rows})")
            skipped.append({"ticker": symbol, "stage": "min_rows", "reason": f"{len(df)} rows"})
            continue

        gaps = session_gaps(df, args.start, args.end)
        entry = {
            "ticker": symbol,
            "source": "yfinance",
            "adjustment": "split_bonus_adjusted",
            "rows": int(len(df)),
            "range": [str(df["timestamps"].iloc[0].date()), str(df["timestamps"].iloc[-1].date())],
            "rows_per_split": {name: int(len(slice_split(df, name))) for name in SPLITS},
            "cleaning": clean_report,
            "missing_sessions": len(gaps),
            "missing_session_dates": gaps[:20],
        }

        if args.cross_check:
            try:
                entry["cross_check"] = cross_check(df, fetch_jugaad(symbol, args.start, args.end))
            except Exception as exc:  # noqa: BLE001 - best-effort
                entry["cross_check"] = {"status": "failed", "reason": str(exc)[:200]}

        write_partition(df, symbol)
        entries.append(entry)
        print(f"OK {len(df)} rows, {len(gaps)} missing sessions")

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "disclaimer": DISCLAIMER,
        "requested_range": [args.start, args.end],
        "canonical_source": "yfinance",
        "adjustment_policy": "split_bonus_adjusted (see common/preprocess.py)",
        # Back-adjusted history is not stable over time: a future split retroactively
        # rewrites every earlier bar. Two corpora fetched on different dates are
        # therefore NOT interchangeable, so the fetch date is part of the corpus
        # identity and any result must cite it alongside the commit sha.
        "downloaded_at": datetime.now().astimezone().isoformat(),
        "adjustment_stability_note": (
            "Prices are back-adjusted as of downloaded_at. A corporate action after "
            "that date will change the adjusted history for affected tickers; re-fetch "
            "and re-run rather than mixing corpora across fetch dates."
        ),
        "amount_definition": "close * volume (hard constraint 4)",
        "splits": SPLITS,
        "universe": {
            "current_constituents": len(CURRENT_CONSTITUENTS),
            "survivorship_additions": SURVIVORSHIP_ADDITIONS,
            "point_in_time": False,
            "note": (
                "Approximate, not point-in-time. Current index constituents plus named "
                "2018-2021 members that later collapsed or left the index. True "
                "point-in-time reconstruction needs historical membership tables that "
                "are not freely available. Residual survivorship bias remains and must "
                "be disclosed wherever coverage numbers are published."
            ),
        },
        "tickers_requested": len(tickers),
        "tickers_written": len(entries),
        "tickers_skipped": len(skipped),
        "skipped": skipped,
        "tickers": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # default=str: pandera failure cases carry Timestamps, which json cannot encode.
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n--- fetch summary ---")
    print(f"written  : {len(entries)}")
    print(f"skipped  : {len(skipped)}")
    for s in skipped:
        print(f"    {s['ticker']}: {s['reason'][:120]}")
    if entries:
        rows = [e["rows"] for e in entries]
        print(f"rows     : min={min(rows)} median={int(pd.Series(rows).median())} max={max(rows)}")
    print(f"manifest : {MANIFEST_PATH}")
    print(f"parquet  : {DATA_ROOT}")
    return 0 if len(entries) >= 45 else 1


if __name__ == "__main__":
    raise SystemExit(main())
