"""Append-only forecast archive, partitioned by the date the forecast was made.

Two requirements pull against each other. The archive must be append-only, because it is
the evidence for a future track-record view and a rewritten past would make that view a
lie. It must also be idempotent, because a workflow re-run on the same day is routine and
must not double-count.

Partitioning by forecast date satisfies both without a dedup pass: a day is one file, a
re-run replaces that day's file, and no other day is ever touched. Appending twice is not
possible because appending is not what the writer does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "ticker",
    "forecast_date",
    "target_date",
    "step",
    "level",
    "lo",
    "hi",
    "p50",
    "engine",
    "backfilled",
]


def _day_path(root: Path, forecast_date: str) -> Path:
    return Path(root) / f"date={forecast_date}" / "forecasts.parquet"


def write_day(root: Path, forecast_date: str, rows: pd.DataFrame) -> Path:
    """Replace this forecast date's partition. Other dates are never read or rewritten."""
    missing = set(COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"archive rows are missing columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("refusing to write an empty archive partition")
    if set(rows["forecast_date"].unique()) != {forecast_date}:
        raise ValueError("every row in a partition must share its forecast_date")

    path = _day_path(root, forecast_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows[COLUMNS].to_parquet(path, index=False)
    return path


def read_all(root: Path) -> pd.DataFrame:
    root = Path(root)
    files = sorted(root.glob("date=*/forecasts.parquet"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def due_for_scoring(root: Path, target_date: str) -> pd.DataFrame:
    """Rows whose forecast horizon lands on ``target_date`` -- i.e. now observable.

    Every past forecast contributes exactly one step on any given session: the h-step
    prediction made h sessions ago. That is what feeds the ACI update.
    """
    df = read_all(root)
    if df.empty:
        return df
    return df[df["target_date"] == target_date].copy()


def partitions(root: Path) -> list[str]:
    found = Path(root).glob("date=*/forecasts.parquet")
    return sorted(p.parent.name.split("=", 1)[1] for p in found)
