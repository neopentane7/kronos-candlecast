"""Tests for hard constraint 5: future timestamps come from the exchange calendar."""

import pandas as pd
import pytest

from common.calendar import NSE_CALENDAR_CODE, future_sessions, nse_calendar, sessions_in_range


def test_calendar_is_indian_equities():
    cal = nse_calendar()
    assert cal.name == NSE_CALENDAR_CODE
    assert str(cal.tz) == "Asia/Kolkata"


def test_known_indian_holidays_are_not_sessions():
    # Republic Day and Gandhi Jayanti are weekday market holidays in India.
    sessions = sessions_in_range("2025-01-01", "2025-12-31")
    assert pd.Timestamp("2025-01-26") not in sessions
    assert pd.Timestamp("2025-10-02") not in sessions
    assert pd.Timestamp("2025-01-27") in sessions  # the Monday after Republic Day


def test_future_sessions_skips_weekends_and_holidays():
    # 2025-01-24 was a Friday; the next 5 sessions must skip the weekend and 26 Jan.
    out = future_sessions(pd.Timestamp("2025-01-24"), 5)

    assert len(out) == 5
    assert out[0] == pd.Timestamp("2025-01-27")
    assert out.tz is None
    assert out.is_monotonic_increasing
    assert not any(ts.weekday() >= 5 for ts in out)
    assert pd.Timestamp("2025-01-26") not in out


def test_future_sessions_anchors_from_a_non_session():
    """A Sunday anchor rolls back to Friday, so the next session is the Monday."""
    from_sunday = future_sessions(pd.Timestamp("2025-01-26"), 3)
    from_friday = future_sessions(pd.Timestamp("2025-01-24"), 3)
    assert list(from_sunday) == list(from_friday)


def test_future_sessions_rejects_bad_count():
    with pytest.raises(ValueError, match="count must be"):
        future_sessions(pd.Timestamp("2025-01-24"), 0)


def test_future_sessions_refuses_to_run_past_published_holidays():
    """Better to fail loudly than to invent sessions on unpublished holidays."""
    last = nse_calendar().last_session
    with pytest.raises(ValueError, match="holiday list"):
        future_sessions(last, 30)
