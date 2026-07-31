"""Tests for hard constraint 7: time splits, never ticker splits, never overlapping."""

import pandas as pd

from common.splits import SPLITS, slice_split, split_of


def test_splits_are_ordered_and_do_not_overlap():
    bounds = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in SPLITS.values()]
    for (_, end), (start, _) in zip(bounds, bounds[1:], strict=False):
        assert end < start, "splits must not overlap"


def test_calibration_year_is_2024():
    """2024 is both the validation split and the conformal calibration set (A5)."""
    assert SPLITS["val"] == ("2024-01-01", "2024-12-31")


def test_split_of_boundaries():
    assert split_of("2018-01-01") == "train"
    assert split_of("2023-12-31") == "train"
    assert split_of("2024-01-01") == "val"
    assert split_of("2024-12-31") == "val"
    assert split_of("2025-01-01") == "test"
    assert split_of("2026-06-30") == "test"
    assert split_of("2017-12-31") is None
    assert split_of("2026-07-01") is None


def test_slice_split_partitions_without_loss_or_overlap():
    df = pd.DataFrame({"timestamps": pd.date_range("2018-01-01", "2026-06-30", freq="D")})
    parts = {name: slice_split(df, name) for name in SPLITS}

    total = sum(len(p) for p in parts.values())
    assert total == len(df), "date splits must cover the corpus range exactly once"

    seen = pd.concat([p["timestamps"] for p in parts.values()])
    assert not seen.duplicated().any()
