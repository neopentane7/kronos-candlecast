"""The nightly job's guarantees: valid contracts, idempotency, honest flags.

These are the checks that keep the site from quietly serving something wrong. A forecast
that fails validation must never reach the site, a re-run must not double-count the
archive, and seeded history must be distinguishable from a forecast actually made that
morning (resolved conflict #10).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline import archive  # noqa: E402
from pipeline.aci import ACIState  # noqa: E402
from pipeline.contract import CONTRACT_VERSION, DISCLAIMER, Forecast, validate_file  # noqa: E402
from pipeline.run_nightly import forecast_one  # noqa: E402

SITE = REPO / "site" / "data"
ARCHIVE = REPO / "pipeline" / "archive_data"
needs_run = pytest.mark.skipif(
    not (SITE / "index.json").exists(), reason="run pipeline/run_nightly.py first"
)


def bars(n=460, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=n)))
    return pd.DataFrame(
        {
            "timestamps": pd.bdate_range("2024-01-01", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
            "amount": close * 1e6,
        }
    )


# ---------------------------------------------------------------- contract
def test_forecast_one_produces_a_valid_contract():
    fc, rows = forecast_one(
        "TEST", bars(), ACIState(), 1.0, "2026-06-30T18:00:00+05:30", False, 1234
    )
    assert fc.contract_version == CONTRACT_VERSION
    assert fc.engine == "rw_drift"
    assert fc.challenger is None
    assert len(rows) == 30 * 3, "one archive row per horizon step per level"


def test_crossed_quantiles_are_rejected():
    """A crossed band would render the cone inside out; the schema must refuse it."""
    fc, _ = forecast_one("T", bars(), ACIState(), 1.0, "2026-06-30T18:00:00+05:30", False, 1)
    bad = fc.model_dump()
    bad["quantiles"]["p10"][0] = bad["quantiles"]["p90"][0] + 1.0
    with pytest.raises(ValueError, match="quantiles cross"):
        Forecast.model_validate(bad)


def test_the_disclaimer_is_not_editable():
    fc, _ = forecast_one("T", bars(), ACIState(), 1.0, "2026-06-30T18:00:00+05:30", False, 1)
    bad = fc.model_dump()
    bad["disclaimer"] = "Guaranteed returns"
    with pytest.raises(ValueError, match="not editable"):
        Forecast.model_validate(bad)


@pytest.mark.parametrize(
    "field,value,msg",
    [
        ("prob_above_last_close", [2.0] * 30, "must lie in"),
        ("prob_vol_exceeds_recent", 1.5, "must lie in"),
        ("last_close", -1.0, "must be positive"),
        ("timestamps", ["2026-01-01"], "timestamps has"),
    ],
)
def test_out_of_range_fields_are_rejected(field, value, msg):
    fc, _ = forecast_one("T", bars(), ACIState(), 1.0, "2026-06-30T18:00:00+05:30", False, 1)
    bad = fc.model_dump()
    bad[field] = value
    with pytest.raises(ValueError, match=msg):
        Forecast.model_validate(bad)


def test_backfilled_flag_is_carried_through_to_the_archive():
    """Seeded history must never be mistaken for a forecast made that morning."""
    fc, rows = forecast_one("T", bars(), ACIState(), 1.0, "2026-06-30T18:00:00+05:30", True, 1)
    assert fc.backfilled is True
    assert rows["backfilled"].all()


# ---------------------------------------------------------------- archive
def test_rerunning_a_day_replaces_it_and_appends_nothing(tmp_path):
    """Idempotency: a workflow re-run on the same day is routine."""

    def day(value):
        return pd.DataFrame(
            [
                {
                    "ticker": "A.NS",
                    "forecast_date": "2026-06-30",
                    "target_date": "2026-07-01",
                    "step": 1,
                    "level": 0.8,
                    "lo": value,
                    "hi": value + 1,
                    "p50": value + 0.5,
                    "engine": "rw_drift",
                    "backfilled": False,
                }
            ]
        )

    archive.write_day(tmp_path, "2026-06-30", day(10.0))
    archive.write_day(tmp_path, "2026-06-30", day(20.0))
    got = archive.read_all(tmp_path)
    assert len(got) == 1, "a re-run must replace the day, not append to it"
    assert got["lo"].iloc[0] == 20.0
    assert archive.partitions(tmp_path) == ["2026-06-30"]


def test_writing_a_day_never_touches_another_day(tmp_path):
    """Append-only in the sense that matters: history is not rewritten."""
    base = {
        "ticker": "A.NS",
        "target_date": "2026-07-01",
        "step": 1,
        "level": 0.8,
        "lo": 1.0,
        "hi": 2.0,
        "p50": 1.5,
        "engine": "rw_drift",
        "backfilled": False,
    }
    archive.write_day(
        tmp_path, "2026-06-29", pd.DataFrame([{**base, "forecast_date": "2026-06-29"}])
    )
    archive.write_day(
        tmp_path, "2026-06-30", pd.DataFrame([{**base, "forecast_date": "2026-06-30"}])
    )
    archive.write_day(
        tmp_path, "2026-06-30", pd.DataFrame([{**base, "forecast_date": "2026-06-30"}])
    )
    assert archive.partitions(tmp_path) == ["2026-06-29", "2026-06-30"]
    assert len(archive.read_all(tmp_path)) == 2


def test_a_partition_must_be_internally_consistent(tmp_path):
    mixed = pd.DataFrame(
        [
            {
                "ticker": "A",
                "forecast_date": "2026-06-30",
                "target_date": "x",
                "step": 1,
                "level": 0.8,
                "lo": 1,
                "hi": 2,
                "p50": 1.5,
                "engine": "e",
                "backfilled": False,
            },
            {
                "ticker": "A",
                "forecast_date": "2026-06-29",
                "target_date": "x",
                "step": 1,
                "level": 0.8,
                "lo": 1,
                "hi": 2,
                "p50": 1.5,
                "engine": "e",
                "backfilled": False,
            },
        ]
    )
    with pytest.raises(ValueError, match="share its forecast_date"):
        archive.write_day(tmp_path, "2026-06-30", mixed)


# ---------------------------------------------------------------- committed output
@needs_run
def test_every_committed_forecast_validates():
    files = sorted((SITE / "forecasts").glob("*.json"))
    assert files, "no forecasts emitted"
    for f in files:
        fc = validate_file(f)
        assert fc.engine == "rw_drift"
        assert fc.disclaimer == DISCLAIMER
        assert len(fc.timestamps) == fc.horizon


@needs_run
def test_index_and_history_agree_with_the_forecasts():
    index = json.loads((SITE / "index.json").read_text())
    listed = set(index["tickers"])
    on_disk = {p.stem for p in (SITE / "forecasts").glob("*.json")}
    history = {p.stem for p in (SITE / "history").glob("*.json")}
    assert listed == on_disk == history
    assert index["contract_version"] == CONTRACT_VERSION


@needs_run
def test_seeded_records_are_all_flagged_backfilled():
    df = archive.read_all(ARCHIVE)
    assert not df.empty
    assert df["backfilled"].all(), "seeded archive rows must be flagged (conflict #10)"
