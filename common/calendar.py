"""The single source of trading sessions for the whole project (hard constraint 5).

Nothing anywhere may use ``pd.bdate_range`` or calendar days to build future
timestamps -- import from here instead.

**Deviation from hard constraint 5, recorded deliberately:** the constraint names the
NSE calendar code ``XNSE``, but ``exchange_calendars`` does not define it. India is
covered only by ``XBOM``/``BSE`` (see ``exchange_calendars.get_calendar_names()``).
NSE and BSE trade on the same SEBI-published holiday list and the same 09:15-15:30 IST
session, so ``XBOM`` is the NSE calendar in all but name. If upstream ever adds a real
``XNSE``, changing ``NSE_CALENDAR_CODE`` below is the entire migration.
"""

from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

NSE_CALENDAR_CODE = "XBOM"


def nse_calendar() -> xcals.ExchangeCalendar:
    """The NSE trading calendar. Cached internally by ``exchange_calendars``."""
    return xcals.get_calendar(NSE_CALENDAR_CODE)


def _as_session(cal: xcals.ExchangeCalendar, ts) -> pd.Timestamp:
    """Snap ``ts`` back to the most recent trading session on or before it."""
    return cal.date_to_session(pd.Timestamp(ts).normalize(), direction="previous")


def future_sessions(last_ts, count: int) -> pd.DatetimeIndex:
    """The next ``count`` trading sessions strictly after ``last_ts``.

    Raises if the request runs past the end of the packaged holiday list rather than
    silently emitting sessions on holidays that have not been published yet.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    cal = nse_calendar()
    anchor = _as_session(cal, last_ts)
    horizon_end = anchor + pd.Timedelta(days=count * 3 + 14)
    if horizon_end > cal.last_session:
        raise ValueError(
            f"{count} sessions after {anchor.date()} runs past the end of the "
            f"{NSE_CALENDAR_CODE} holiday list ({cal.last_session.date()}). "
            "Upgrade exchange_calendars to pick up the next year's holidays."
        )

    sessions = cal.sessions_window(anchor, count + 1)[1:]
    return pd.DatetimeIndex(sessions).tz_localize(None)


def sessions_in_range(start, end) -> pd.DatetimeIndex:
    """All trading sessions in ``[start, end]``, tz-naive."""
    cal = nse_calendar()
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return pd.DatetimeIndex(sessions).tz_localize(None)
