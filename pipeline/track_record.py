"""Realized coverage over time — the track record, on the axis that can show convergence.

Two things this module refuses to do, because both flatter the result and neither is
visible in the number it produces.

**It never pools across dates.** Sixty tickers priced on one date share a market and thirty
horizon steps share a path, so one session is one observation, not eighteen hundred.
Everything here reduces to a per-date hit rate first and then counts *days*. Report section
17b argues the same denominator for the research grid; a serving surface that quoted the
row count would be making exactly the mistake the corpus audit caught (R6).

**It never reports the cumulative mean.** ACI adapts online, so early forecasts ran on a
cold state and later ones on a warm one. On this project's own archive the cumulative
figures read 0.4554 / 0.7492 / 0.8584 against nominal 0.50 / 0.80 / 0.90, while the last
twenty comparable dates read 0.5071 / 0.8086 / 0.9078. The running average is dragging a
cold-start deficit the engine has already corrected across every future reading of it, and
understates the served bands by five points at 80%.

**And it never averages dates that have matured to different depths.** This one was found
the hard way. The newest dates in the archive have only their first step or two resolved,
short horizons are easier to cover than long ones, and averaging a one-step date against a
thirty-step date reads as improvement that has not happened. Before the `complete` filter,
this module reported the trailing window at 0.8458 against a true 0.8086 -- a manufactured
+4.6pp overshoot, produced by exactly the unequal-units pooling described above.

The unit of the time axis is the **forecast date**, not the date the outcome arrived. A
band produced while ACI was cold is a cold band no matter when it matures, so grouping by
forecast date is what makes the convergence question answerable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LEVELS = (0.50, 0.80, 0.90)

# A forecast date is comparable to another only once the same steps have matured on
# both. The newest dates in an archive have only their first step or two resolved, and
# averaging those against fully-resolved dates is the R6 mistake in another costume:
# unequal units pooled as though they were equal. Dates short of the full horizon are
# carried in the series -- the shape is still informative -- but never averaged.
HORIZON = 30

# Twenty sessions is about a trading month: long enough that a single volatile day does not
# swing the line, short enough that a plateau lasting weeks cannot hide inside it.
WINDOW = 20


def realized_closes(corpus: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """`{ticker.NS: {date: close}}` — the outcomes a forecast is scored against."""
    out: dict[str, dict[str, float]] = {}
    for ticker, bars in corpus.items():
        stamps = pd.to_datetime(bars["timestamps"]).dt.strftime("%Y-%m-%d")
        out[f"{ticker}.NS"] = dict(zip(stamps, bars["close"], strict=True))
    return out


def score_rows(df: pd.DataFrame, closes: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Attach the outcome and the hit indicator; drop forecasts that have not matured."""
    if df.empty:
        return df.assign(actual=[], hit=[])
    actual = [
        closes.get(t, {}).get(d) for t, d in zip(df["ticker"], df["target_date"], strict=True)
    ]
    scored = df.assign(actual=actual).dropna(subset=["actual"])
    if scored.empty:
        return scored.assign(hit=[])
    inside = (scored["lo"] <= scored["actual"]) & (scored["actual"] <= scored["hi"])
    return scored.assign(hit=inside)


def per_date_series(scored: pd.DataFrame) -> dict[str, list[dict]]:
    """One entry per forecast date per level: the hit rate on that date, and its size.

    `days` is the count a reader should be shown. `rows` is carried only so a reader can
    see how much larger it is, never as a sample size.
    """
    out: dict[str, list[dict]] = {}
    if scored.empty:
        return {f"{lvl:.2f}": [] for lvl in LEVELS}

    for lvl in LEVELS:
        grp = scored[scored["level"] == lvl]
        entries: list[dict] = []
        for date, day in grp.groupby("forecast_date", sort=True):
            entries.append(
                {
                    "date": str(date),
                    "coverage": round(float(day["hit"].mean()), 4),
                    "rows": int(len(day)),
                    "tickers": int(day["ticker"].nunique()),
                    "steps": int(day["step"].max()),
                    "complete": bool(day["step"].max() >= HORIZON),
                    # A date is live only if nothing on it was a replayed forecast.
                    "live": bool(not day["backfilled"].any()),
                }
            )
        out[f"{lvl:.2f}"] = entries
    return out


def trailing(entries: list[dict], window: int = WINDOW) -> dict | None:
    """Mean coverage over the last `window` *fully matured* forecast dates.

    Incomplete dates are excluded, not down-weighted. A date with one step resolved is not
    a small sample of the same quantity -- it is a different quantity, because short
    horizons are easier to cover than long ones. Including them tilts the recent end of the
    series upward and would read as improvement.

    Returns None rather than a number when nothing qualifies: a panel with no data must say
    so, not render a zero.
    """
    ready = [e for e in entries if e.get("complete", True)]
    if not ready:
        return None
    tail = ready[-window:]
    return {
        "coverage": round(sum(e["coverage"] for e in tail) / len(tail), 4),
        "days": len(tail),
        "window": window,
        "excluded_incomplete": len(entries) - len(ready),
        "from": tail[0]["date"],
        "to": tail[-1]["date"],
    }


def build(archive_df: pd.DataFrame, corpus: dict[str, pd.DataFrame]) -> dict:
    """The full payload the site publishes."""
    scored = score_rows(archive_df, realized_closes(corpus))
    series = per_date_series(scored)

    levels: dict[str, dict] = {}
    for key, entries in series.items():
        live = [e for e in entries if e["live"]]
        levels[key] = {
            "nominal": float(key),
            "series": entries,
            "trailing": trailing(entries),
            "days": len(entries),
            "live_days": len(live),
            # Kept, labelled, and deliberately not the headline: it is the number a reader
            # would compute themselves and be misled by, so the panel shows it losing.
            "cumulative": (
                round(sum(e["coverage"] for e in ready) / len(ready), 4)
                if (ready := [e for e in entries if e["complete"]])
                else None
            ),
        }

    any_entries = next((v["series"] for v in levels.values() if v["series"]), [])
    return {
        "window": WINDOW,
        "unit": "forecast date",
        "days": len(any_entries),
        "live_days": sum(1 for e in any_entries if e["live"]),
        "first": any_entries[0]["date"] if any_entries else None,
        "last": any_entries[-1]["date"] if any_entries else None,
        "levels": levels,
        "note": (
            "One forecast date is one observation. Coverage is the mean over dates, and the "
            "headline figure is a trailing window: ACI adapts online, so a cumulative mean "
            "blends the cold-start transient into the steady state and can read near-nominal "
            "by cancellation."
        ),
    }


def append_live_day(ledger_path: Path, forecast_date: str, per_level: dict) -> dict:
    """Append one live scoring day to the durable ledger, replacing any existing entry.

    Live outcomes cannot be recomputed offline: the committed corpus stops months before
    the serving path starts, so a live day that is not written down when it happens is lost.
    Re-running the same date overwrites rather than duplicates, because the nightly job is
    idempotent and a rerun must not count twice.
    """
    ledger = {"days": []}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    days = [d for d in ledger.get("days", []) if d["date"] != forecast_date]
    days.append({"date": forecast_date, "by_level": per_level})
    ledger["days"] = sorted(days, key=lambda d: d["date"])
    return ledger
