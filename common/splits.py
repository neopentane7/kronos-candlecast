"""Time-based splits (hard constraint 7). Never split by ticker.

Every ticker contributes to every split; the boundaries are dates, so no information
from a later period can reach an earlier one. 2024 doubles as the validation year and
the conformal calibration set (A5). The test period stays untouched until final eval.
"""

from __future__ import annotations

import pandas as pd

SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2018-01-01", "2023-12-31"),
    "val": ("2024-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-06-30"),
}

CORPUS_START = SPLITS["train"][0]


def split_of(ts) -> str | None:
    """Which split a timestamp belongs to, or ``None`` if it falls outside all of them."""
    ts = pd.Timestamp(ts).normalize()
    for name, (start, end) in SPLITS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return name
    return None


def slice_split(df: pd.DataFrame, split: str, column: str = "timestamps") -> pd.DataFrame:
    """Rows of ``df`` falling inside ``split``."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    start, end = SPLITS[split]
    ts = pd.to_datetime(df[column])
    mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)
