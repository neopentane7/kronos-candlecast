"""Audit every ticker's session index against the universe and the exchange calendar.

The ITC defect (a bar on 2025-03-18 that no other ticker carried) was found only because
it split the evaluation grid's blocks and corrupted a statistic that was being read at the
time. A gap that never lands on a window boundary would not have announced itself. This
scans the whole corpus once, so a corpus correction is a single event rather than a series
of them, each churning the golden numbers.

Three kinds of disagreement, which mean different things:

* **orphan** -- the ticker has a bar on a date the universe does not trade. Grid semantics
  are the intersection of dates: a window cannot be forecast against a universe with no
  data there, so these rows are dropped.
* **hole** -- the ticker lacks a bar on a date the universe does trade, inside its own
  listing range. Not fixable without refetching, which back-adjustment forbids
  (see SETUP.md), so these are reported and carried as a known limitation.
* **off-calendar** -- the whole universe trades on a date the calendar package does not
  list. These are real NSE special sessions (Diwali Muhurat) and are kept.

"Universe session" means a date carried by at least half the tickers listed on it, so a
single ticker can never define the calendar for the rest.

Usage (PowerShell):
    uv run python phase-a/scripts/audit_calendar.py
    uv run python phase-a/scripts/audit_calendar.py --json audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.calendar import NSE_CALENDAR_CODE  # noqa: E402

PARQUET_GLOB = "data/parquet/*/*.parquet"


def load_dates(parquet_glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    return con.sql(f"""
        SELECT ticker, CAST(timestamps AS DATE) AS d
        FROM read_parquet('{parquet_glob}', hive_partitioning=1)
        GROUP BY ticker, d ORDER BY ticker, d
    """).df()


def audit(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    by_ticker = {t: set(g["d"]) for t, g in df.groupby("ticker")}
    tickers = sorted(by_ticker)

    span = {t: (min(ds), max(ds)) for t, ds in by_ticker.items()}
    # A date is a universe session if at least half the tickers *listed on it* carry it.
    counts = df.groupby("d")["ticker"].nunique()
    listed = pd.Series(
        {d: sum(1 for t in tickers if span[t][0] <= d <= span[t][1]) for d in counts.index}
    )
    universe = {d for d in counts.index if counts[d] >= max(2, listed[d] / 2)}

    import exchange_calendars as xc

    cal = xc.get_calendar(NSE_CALENDAR_CODE)
    lo, hi = min(universe), max(universe)
    sessions = {pd.Timestamp(s).normalize() for s in cal.sessions_in_range(lo, hi)}

    per_ticker = {}
    for t in tickers:
        first, last = span[t]
        expected = {d for d in universe if first <= d <= last}
        orphans = sorted(by_ticker[t] - universe)
        holes = sorted(expected - by_ticker[t])
        if orphans or holes:
            per_ticker[t] = {
                "n_sessions": len(by_ticker[t]),
                "first": str(first.date()),
                "last": str(last.date()),
                "orphan_dates": [str(d.date()) for d in orphans],
                "hole_dates": [str(d.date()) for d in holes],
            }

    return {
        "calendar": NSE_CALENDAR_CODE,
        "n_tickers": len(tickers),
        "range": [str(lo.date()), str(hi.date())],
        "n_universe_sessions": len(universe),
        "n_calendar_sessions": len(sessions),
        "universe_off_calendar": [str(d.date()) for d in sorted(universe - sessions)],
        "calendar_absent_from_universe": [str(d.date()) for d in sorted(sessions - universe)],
        "tickers_with_disagreements": per_ticker,
        "n_orphan_rows": sum(len(v["orphan_dates"]) for v in per_ticker.values()),
        "n_hole_rows": sum(len(v["hole_dates"]) for v in per_ticker.values()),
        "clean": not per_ticker,
    }


def drop_orphans(rep: dict, root: Path) -> list[str]:
    """Remove orphan rows from the partitions that carry them.

    Only orphans. Holes cannot be filled without refetching, and refetching re-runs
    back-adjustment against a later corporate-action history, which would silently
    rewrite prices across the whole series rather than patch one date.
    """
    changed = []
    for ticker, info in sorted(rep["tickers_with_disagreements"].items()):
        if not info["orphan_dates"]:
            continue
        path = root / f"ticker={ticker}" / "data.parquet"
        df = pd.read_parquet(path)
        drop = pd.to_datetime(pd.Series(info["orphan_dates"])).dt.normalize()
        keep = ~pd.to_datetime(df["timestamps"]).dt.normalize().isin(set(drop))
        if int((~keep).sum()) == 0:
            continue
        df[keep].reset_index(drop=True).to_parquet(path, index=False)
        changed.append(f"{ticker}: dropped {int((~keep).sum())} row(s) {info['orphan_dates']}")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=PARQUET_GLOB)
    ap.add_argument("--json", type=Path, default=None, help="also write the report here")
    ap.add_argument(
        "--fix",
        action="store_true",
        help="drop orphan rows in place; re-runs the audit afterwards to confirm",
    )
    args = ap.parse_args()

    rep = audit(load_dates(args.parquet))

    lo, hi = rep["range"]
    print(f"calendar {rep['calendar']}  |  {rep['n_tickers']} tickers  |  {lo} .. {hi}")
    print(
        f"universe sessions: {rep['n_universe_sessions']}   "
        f"calendar sessions: {rep['n_calendar_sessions']}"
    )

    print(f"\nuniverse trades, calendar does not list it ({len(rep['universe_off_calendar'])}):")
    print("  " + (", ".join(rep["universe_off_calendar"]) or "none"))
    print("  -> NSE special sessions (Muhurat). Kept: they are real trading days.")

    print(
        f"\ncalendar lists it, universe has no data ({len(rep['calendar_absent_from_universe'])}):"
    )
    print("  " + (", ".join(rep["calendar_absent_from_universe"]) or "none"))

    print(f"\ntickers disagreeing with the universe: {len(rep['tickers_with_disagreements'])}")
    for t, v in sorted(rep["tickers_with_disagreements"].items()):
        bits = []
        if v["orphan_dates"]:
            bits.append(f"orphan {v['orphan_dates']}")
        if v["hole_dates"]:
            h = v["hole_dates"]
            bits.append(f"holes({len(h)}) {h if len(h) <= 6 else h[:6] + ['...']}")
        print(f"  {t:<12} {v['n_sessions']:>5} sessions  " + "  ".join(bits))

    print(
        f"\norphan rows to drop: {rep['n_orphan_rows']}   holes (not fixable): {rep['n_hole_rows']}"
    )
    print("CLEAN" if rep["clean"] else "corpus needs a correction pass")

    if args.fix and rep["n_orphan_rows"]:
        root = Path(args.parquet).parent.parent
        print(f"\n--fix: dropping {rep['n_orphan_rows']} orphan row(s) under {root}")
        for line in drop_orphans(rep, root):
            print("  " + line)
        rep = audit(load_dates(args.parquet))
        print(f"re-audit: orphan rows now {rep['n_orphan_rows']}, holes {rep['n_hole_rows']}")
        if rep["n_orphan_rows"]:
            print("orphans survived the fix")
            return 1

    if args.json:
        args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
