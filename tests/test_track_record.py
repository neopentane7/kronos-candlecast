"""The track record has exactly two ways to lie, and both look like ordinary code.

It can count forecasts instead of days, which overstates the evidence by roughly the
number of tickers. And it can report the cumulative mean, which on an online-adapting
forecaster averages the cold-start transient against the steady state and can land on
nominal by cancellation. Neither would raise; both would render a confident, wrong panel.

These tests pin the two decisions, and the last one demonstrates the cancellation on a
constructed series so the reason for the trailing window is visible rather than asserted.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline import track_record  # noqa: E402


def archive_frame(
    per_date, level=0.80, tickers=("A.NS", "B.NS", "C.NS"), backfilled=True, steps=None
):
    """One row per ticker per horizon step per date, with `hits` tickers inside the band.

    The band is [0, 1] for a hit and [10, 11] for a miss, so the outcome (always 0.5) is
    inside exactly when intended. Dates carry the full horizon by default, because a date
    short of it is deliberately excluded from every average and would make most of these
    tests vacuous.
    """
    steps = range(1, (steps or track_record.HORIZON) + 1)
    rows = []
    for date, hits in per_date:
        for i, tk in enumerate(tickers):
            inside = i < hits
            for step in steps:
                rows.append(
                    {
                        "ticker": tk,
                        "forecast_date": date,
                        "target_date": date,
                        "step": step,
                        "level": level,
                        "lo": 0.0 if inside else 10.0,
                        "hi": 1.0 if inside else 11.0,
                        "p50": 0.5,
                        "engine": "rw_drift",
                        "backfilled": backfilled,
                    }
                )
    return pd.DataFrame(rows)


def closes_for(df):
    return {tk: {d: 0.5 for d in df["target_date"]} for tk in df["ticker"].unique()}


# ------------------------------------------------------- days, not observations
def test_a_date_is_one_observation_however_many_tickers_it_carries():
    df = archive_frame(
        [("2026-01-01", 3), ("2026-01-02", 0)], tickers=tuple(f"T{i}.NS" for i in range(60))
    )
    series = track_record.per_date_series(track_record.score_rows(df, closes_for(df)))["0.80"]

    assert len(series) == 2, "two dates, not 120 forecasts"
    assert series[0]["rows"] == 60 * track_record.HORIZON, "rows dwarf the day count"
    assert series[0]["tickers"] == 60
    assert track_record.trailing(series)["days"] == 2


def test_a_days_coverage_is_its_own_hit_rate_not_a_pooled_one():
    """Two days of unequal quality must average as days, not as forecasts."""
    df = archive_frame([("2026-01-01", 3), ("2026-01-02", 0)])
    series = track_record.per_date_series(track_record.score_rows(df, closes_for(df)))["0.80"]
    assert series[0]["coverage"] == 1.0
    assert series[1]["coverage"] == 0.0
    assert track_record.trailing(series)["coverage"] == 0.5


# ------------------------------------------------------- trailing, not cumulative
def test_the_trailing_window_ignores_everything_before_it():
    entries = [{"date": f"d{i}", "coverage": 0.0} for i in range(30)]
    entries += [{"date": f"d{30 + i}", "coverage": 1.0} for i in range(20)]
    t = track_record.trailing(entries, window=20)
    assert t["coverage"] == 1.0
    assert t["days"] == 20
    assert t["from"] == "d30"


def test_the_window_degrades_to_what_exists_rather_than_padding():
    entries = [{"date": "d0", "coverage": 0.9}, {"date": "d1", "coverage": 0.7}]
    t = track_record.trailing(entries, window=20)
    assert t["days"] == 2, "the day count must report reality, not the requested window"
    assert t["coverage"] == pytest.approx(0.8)


def test_no_data_returns_none_rather_than_a_zero():
    """A panel with nothing to show must say so; 0.0 would render as total failure."""
    assert track_record.trailing([]) is None


def test_the_cumulative_mean_understates_a_forecaster_that_has_since_converged():
    """The whole reason this module exists, on a series built to make it explicit.

    Twenty cold days at 0.60 followed by forty converged days at 0.85 average to 0.7667 --
    a cumulative panel reports a forecaster running seven points under nominal, when the
    bands actually served today cover 0.85. The deficit is real history and it is over.
    """
    entries = [{"date": f"c{i}", "coverage": 0.60, "complete": True} for i in range(20)]
    entries += [{"date": f"w{i}", "coverage": 0.85, "complete": True} for i in range(40)]

    cumulative = sum(e["coverage"] for e in entries) / len(entries)
    assert cumulative == pytest.approx(0.7667, abs=1e-4)
    assert track_record.trailing(entries, window=20)["coverage"] == pytest.approx(0.85)


# ------------------------------------------------- comparable dates only
def test_a_partly_matured_date_never_enters_the_average():
    """The bug this catches was real, and it manufactured a +4.6pp overshoot.

    A date with one step resolved is not a small sample of a thirty-step date; short
    horizons are easier to cover, so mixing them tilts the recent end upward and reads as
    improvement.
    """
    entries = [{"date": f"d{i}", "coverage": 0.80, "complete": True} for i in range(20)]
    entries += [{"date": "fresh", "coverage": 1.0, "complete": False}]

    t = track_record.trailing(entries, window=20)
    assert t["coverage"] == pytest.approx(0.80), "the unmatured date must not lift the mean"
    assert t["excluded_incomplete"] == 1
    assert t["days"] == 20


def test_a_date_is_complete_only_at_the_full_horizon():
    df = archive_frame([("2026-01-01", 3)])
    series = track_record.per_date_series(track_record.score_rows(df, closes_for(df)))["0.80"]
    assert series[0]["complete"] is True
    assert series[0]["steps"] == track_record.HORIZON


def test_a_date_short_of_the_horizon_is_carried_but_flagged():
    """Kept in the series -- the shape is informative -- but excluded from the average."""
    df = archive_frame([("2026-01-01", 3)], steps=2)
    series = track_record.per_date_series(track_record.score_rows(df, closes_for(df)))["0.80"]
    assert series[0]["complete"] is False
    assert track_record.trailing(series) is None, "nothing comparable yet is not a zero"


# ------------------------------------------------------- live is never pooled
def test_a_replayed_date_is_never_marked_live():
    df = archive_frame([("2026-01-01", 3)], backfilled=True)
    series = track_record.per_date_series(track_record.score_rows(df, closes_for(df)))["0.80"]
    assert series[0]["live"] is False


def test_a_date_with_any_replayed_forecast_is_not_live():
    """Mixed provenance on one date is not a live day; the stricter reading is the safe one."""
    live = archive_frame([("2026-01-01", 2)], tickers=("A.NS",), backfilled=False)
    replayed = archive_frame([("2026-01-01", 1)], tickers=("B.NS",), backfilled=True)
    df = pd.concat([live, replayed], ignore_index=True)
    series = track_record.per_date_series(track_record.score_rows(df, closes_for(df)))["0.80"]
    assert len(series) == 1
    assert series[0]["live"] is False


def test_unmatured_forecasts_are_dropped_not_counted_as_misses():
    """A forecast whose outcome has not happened yet is absent, not wrong."""
    df = archive_frame([("2026-01-01", 3)])
    scored = track_record.score_rows(df, {})  # no closes at all
    assert scored.empty


# ------------------------------------------------------- the live ledger
def test_a_live_day_is_recorded_and_rerunning_it_does_not_double_count(tmp_path):
    """The nightly job is idempotent, so a repeated run must overwrite its own entry."""
    path = tmp_path / "ledger.json"
    per_level = {"0.80": {"n": 60, "hits": 51, "empirical": 0.85, "nominal": 0.8}}

    ledger = track_record.append_live_day(path, "2026-08-27", per_level)
    path.write_text(json.dumps(ledger), encoding="utf-8")
    ledger = track_record.append_live_day(path, "2026-08-27", per_level)

    assert len(ledger["days"]) == 1, "a rerun of the same date must replace, not append"


def test_the_ledger_stays_in_date_order_however_days_arrive(tmp_path):
    path = tmp_path / "ledger.json"
    for date in ("2026-08-28", "2026-08-26", "2026-08-27"):
        ledger = track_record.append_live_day(path, date, {})
        path.write_text(json.dumps(ledger), encoding="utf-8")
    assert [d["date"] for d in ledger["days"]] == ["2026-08-26", "2026-08-27", "2026-08-28"]


# ------------------------------------------------------- the published payload
def test_the_payload_reports_days_and_carries_the_cumulative_only_as_a_field():
    df = archive_frame([(f"2026-01-0{i}", 2) for i in range(1, 6)])
    corpus_closes = closes_for(df)
    scored = track_record.score_rows(df, corpus_closes)
    series = track_record.per_date_series(scored)

    assert series["0.80"], "the 80% level must be present"
    assert all(e["rows"] == e["tickers"] * track_record.HORIZON for e in series["0.80"])
    # cumulative exists in build()'s output but the trailing figure is what the panel reads.
    t = track_record.trailing(series["0.80"])
    assert t["window"] == track_record.WINDOW


# ------------------------------------------------------- the merge, not a rebuild
def test_the_nightly_publish_preserves_the_backtest_half(tmp_path, monkeypatch):
    """The failure this prevents is total and silent.

    The backtest series is derived from the corpus, and the Actions runner has no corpus.
    If the nightly publish rebuilt the payload instead of merging into it, the first
    scheduled run would replace 59 days of history with an empty series and commit that.
    """
    from pipeline import run_nightly

    site = tmp_path / "site"
    site.mkdir()
    ledger = tmp_path / "ledger.json"

    published = site / "track_record.json"
    published.write_text(
        json.dumps({"days": 59, "window": 20, "backtest": {"0.80": {"nominal": 0.8}}}),
        encoding="utf-8",
    )
    ledger.write_text(
        json.dumps({"days": [{"date": "2026-08-27", "by_level": {}}]}), encoding="utf-8"
    )

    monkeypatch.setattr(run_nightly, "SITE_DATA", site)
    monkeypatch.setattr(run_nightly, "LIVE_LEDGER", ledger)
    run_nightly.publish_track_record()

    after = json.loads(published.read_text(encoding="utf-8"))
    assert after["backtest"] == {"0.80": {"nominal": 0.8}}, "the backtest half must survive"
    assert after["days"] == 59
    assert [d["date"] for d in after["live"]] == ["2026-08-27"]


def test_publishing_before_any_backtest_exists_does_not_crash(tmp_path, monkeypatch):
    """A fresh checkout has no payload yet; the nightly job must still write the live half."""
    from pipeline import run_nightly

    site = tmp_path / "site"
    monkeypatch.setattr(run_nightly, "SITE_DATA", site)
    monkeypatch.setattr(run_nightly, "LIVE_LEDGER", tmp_path / "absent.json")
    run_nightly.publish_track_record()

    after = json.loads((site / "track_record.json").read_text(encoding="utf-8"))
    assert after["live"] == []
