"""Rebuild the live scoring ledger from the committed archive.

**Why this exists.** The nightly workflow's commit step named state files individually and
did not include `track_record_live.json`, so for seven nights the ledger was written on the
runner and thrown away. `append_live_day` starts from whatever is on disk, so each run began
from nothing and the published payload's `live` array was replaced with a single day.

**Why the loss is recoverable here, when the module docstring says it is not.** That warning
is about the *corpus*, which stops months before serving began. But the bands themselves are
committed in `pipeline/archive_data`, and closes for dates now in the past are ordinary
history the vendor still serves. So the scoring can be redone exactly: same bands, same
outcomes, same arithmetic as `score_and_update`.

This is a one-off repair, not part of the nightly path. Run it once, commit the ledger, and
the fixed workflow keeps it from happening again.

    uv run python pipeline/backfill_live_ledger.py            # show what it would write
    uv run python pipeline/backfill_live_ledger.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline import archive  # noqa: E402
from pipeline.run_nightly import ARCHIVE_ROOT, LIVE_LEDGER, fetch_live_bars  # noqa: E402


def live_dates(df: pd.DataFrame) -> list[str]:
    """Forecast dates made live, oldest first. Replayed dates are not live days."""
    live = df[~df["backfilled"]]
    return sorted(live["forecast_date"].unique().tolist())


def closes_from_vendor(tickers: list[str]) -> dict[str, dict[str, float]]:
    """`{ticker: {date: close}}` for the serving universe, over recent history."""
    out: dict[str, dict[str, float]] = {}
    for i, full in enumerate(tickers, 1):
        base = full[:-3] if full.endswith(".NS") else full
        try:
            bars = fetch_live_bars(base)
        except Exception as exc:  # noqa: BLE001 -- a missing ticker is skipped, never faked
            print(f"  [{i:>2}/{len(tickers)}] {full}: {type(exc).__name__}: {exc}")
            continue
        stamps = pd.to_datetime(bars["timestamps"]).dt.strftime("%Y-%m-%d")
        out[full] = dict(zip(stamps, bars["close"], strict=True))
        if i % 15 == 0 or i == len(tickers):
            print(f"  [{i:>2}/{len(tickers)}] fetched", flush=True)
    return out


def score_day(df: pd.DataFrame, target_date: str, closes: dict) -> dict | None:
    """Reproduce `score_and_update`'s per-level block for one scoring date.

    Same selection (rows whose horizon lands on `target_date`) and same hit test, so a
    backfilled entry is indistinguishable from one written on the night.
    """
    due = df[df["target_date"] == target_date]
    if due.empty:
        return None

    per_level = {}
    for lvl, grp in due.groupby("level"):
        actual = grp["ticker"].map(lambda t: closes.get(t, {}).get(target_date))
        ok = actual.notna()
        if not ok.any():
            continue
        g, a = grp[ok], actual[ok]
        hits = (g["lo"] <= a) & (a <= g["hi"])
        per_level[f"{float(lvl):.2f}"] = {
            "n": int(len(g)),
            "hits": int(hits.sum()),
            "tickers": int(g["ticker"].nunique()),
            "empirical": round(float(hits.mean()), 4),
            "nominal": float(lvl),
        }
    return per_level or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the ledger; otherwise dry run")
    args = ap.parse_args(argv)

    df = archive.read_all(ARCHIVE_ROOT)
    if df.empty:
        raise SystemExit("empty archive")

    dates = live_dates(df)
    if not dates:
        raise SystemExit("no live forecast dates in the archive")
    print(f"live forecast dates in the archive: {len(dates)}  ({dates[0]} -> {dates[-1]})")

    existing = (
        {d["date"] for d in json.loads(LIVE_LEDGER.read_text(encoding="utf-8")).get("days", [])}
        if LIVE_LEDGER.exists()
        else set()
    )
    print(f"already in the ledger: {len(existing)}")

    tickers = sorted(df[~df["backfilled"]]["ticker"].unique().tolist())
    print(f"\nfetching closes for {len(tickers)} tickers ...")
    closes = closes_from_vendor(tickers)
    if not closes:
        raise SystemExit("no closes fetched; refusing to write an empty ledger")

    # Accumulated in memory and written once. append_live_day() reads the file on every
    # call, so folding through it here would make a dry run silently produce a one-day
    # ledger while reporting nine -- the script's own version of the bug it repairs.
    days = {}
    if LIVE_LEDGER.exists():
        for d in json.loads(LIVE_LEDGER.read_text(encoding="utf-8")).get("days", []):
            days[d["date"]] = d

    recovered = []
    for date in dates:
        per_level = score_day(df, date, closes)
        if per_level is None:
            print(f"  {date}: nothing matured yet, skipping")
            continue
        days[date] = {"date": date, "by_level": per_level}
        recovered.append((date, per_level))

    ledger = {"days": [days[k] for k in sorted(days)], "rebuilt_from": "pipeline/archive_data"}

    print(f"\n{'date':<12}{'cov50':>8}{'cov80':>8}{'cov90':>8}{'n@80':>7}")
    for date, pl in recovered:
        row = f"{date:<12}"
        for lvl in ("0.50", "0.80", "0.90"):
            row += f"{pl[lvl]['empirical']:>8.4f}" if lvl in pl else f"{'-':>8}"
        row += f"{pl.get('0.80', {}).get('n', 0):>7}"
        print(row)

    if args.write:
        LIVE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LIVE_LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten: {LIVE_LEDGER}  ({len(ledger['days'])} days)")
    else:
        print(f"\ndry run -- {len(recovered)} day(s) would be written. Re-run with --write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
