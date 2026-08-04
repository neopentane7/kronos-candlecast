"""Tests for evaluation-window construction.

The memory test is the reason this file exists: eagerly materialising every window was
fine at 45 windows and drove a 32 GB machine into paging at 708.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a"))

from eval.windows import (  # noqa: E402
    FEATURE_COLUMNS,
    average_true_range_pct,
    build_grid,
)


def synthetic_corpus(n_tickers=4, n_rows=900, seed=0) -> pd.DataFrame:
    """A corpus spanning the test split, with enough history for a 400-bar lookback."""
    rng = np.random.default_rng(seed)
    frames = []
    dates = pd.bdate_range("2023-06-01", periods=n_rows)
    for t in range(n_tickers):
        close = 100.0 * (t + 1) * np.exp(np.cumsum(rng.normal(0, 0.01, n_rows)))
        spread = close * rng.uniform(0.003, 0.02, n_rows)
        open_ = close + rng.normal(0, 0.3, n_rows) * spread
        vol = rng.uniform(1e5, 5e5, n_rows)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"T{t}",
                    "timestamps": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) + spread,
                    "low": np.minimum(open_, close) - spread,
                    "close": close,
                    "volume": vol,
                    "amount": close * vol,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_grid_enumerates_windows_inside_the_split():
    grid = build_grid(synthetic_corpus(), "test", stride=30)
    assert len(grid) > 0
    assert grid.meta["n_tickers"] == 4
    # Every start date must fall inside the test split.
    for d in grid.start_dates:
        assert pd.Timestamp("2025-01-01") <= d <= pd.Timestamp("2026-06-30")


def test_batch_reconstructs_the_expected_window():
    """The lazy frames must match a direct slice of the corpus."""
    corpus = synthetic_corpus()
    grid = build_grid(corpus, "test", stride=30)
    dfs, x_ts, y_ts = grid.batch(0, 3)

    assert len(dfs) == len(x_ts) == len(y_ts) == 3
    for k in range(3):
        ticker, off = grid.tickers[k], grid.offsets[k]
        g = corpus[corpus["ticker"] == ticker].reset_index(drop=True)

        assert list(dfs[k].columns) == FEATURE_COLUMNS
        assert len(dfs[k]) == grid.lookback
        assert len(y_ts[k]) == grid.horizon

        expected_x = g[FEATURE_COLUMNS].to_numpy(float)[off - grid.lookback : off]
        np.testing.assert_allclose(dfs[k].to_numpy(float), expected_x)
        assert x_ts[k].iloc[-1] == g["timestamps"].iloc[off - 1]
        assert y_ts[k].iloc[0] == g["timestamps"].iloc[off]
        # The scored close must be the same rows the batch will be conditioned against.
        np.testing.assert_allclose(
            grid.y_close[k], g["close"].to_numpy(float)[off : off + grid.horizon]
        )
        np.testing.assert_allclose(
            grid.history_close[k], g["close"].to_numpy(float)[off - grid.lookback : off]
        )


def test_batch_timestamps_support_the_dt_accessor():
    """Kronos calls `.dt` on these, so they must be Series and not Index objects."""
    grid = build_grid(synthetic_corpus(), "test", stride=30)
    _, x_ts, y_ts = grid.batch(0, 1)
    assert x_ts[0].dt.weekday is not None
    assert y_ts[0].dt.month is not None


def test_grid_does_not_retain_a_frame_per_window():
    """Memory guard: backing storage must scale with the corpus, not the window count.

    Eager construction held one DataFrame and two Series per window. This asserts the
    grid keeps one array per *ticker* instead, which is what makes the full 708-window
    run fit in memory.
    """
    grid = build_grid(synthetic_corpus(n_tickers=4), "test", stride=5)
    assert len(grid) > 4 * 10, "expected many windows at stride 5"
    assert len(grid.features) == 4
    assert len(grid.timestamps) == 4

    backing = sum(v.nbytes for v in grid.features.values())
    per_window = backing / len(grid)
    assert per_window < 50_000, f"{per_window:.0f} bytes/window suggests eager storage"


def test_subsample_shares_backing_arrays_and_recounts_meta():
    grid = build_grid(synthetic_corpus(n_tickers=4), "test", stride=5)
    sub = grid.subsample(9, seed=0)

    assert len(sub) == 9
    assert sub.meta["subsampled_from"] == len(grid)
    assert sub.meta["n_windows"] == 9
    assert sub.meta["n_tickers"] == len(sub.features)
    # Backing arrays are shared, not copied.
    for t in sub.features:
        assert sub.features[t] is grid.features[t]


def test_subsample_batches_still_reconstruct_correctly():
    """Offsets must survive subsampling — an index bug here would silently misalign."""
    grid = build_grid(synthetic_corpus(), "test", stride=5)
    sub = grid.subsample(6, seed=1)
    dfs, _, _ = sub.batch(0, len(sub))
    for k in range(len(sub)):
        t, off = sub.tickers[k], sub.offsets[k]
        expected = grid.features[t][off - grid.lookback : off]
        np.testing.assert_allclose(dfs[k].to_numpy(float), expected)


def test_block_ids_group_by_start_date():
    grid = build_grid(synthetic_corpus(), "test", stride=30)
    blocks = grid.block_ids
    assert blocks.shape == grid.y_close.shape
    # Two windows sharing a start date must share a block id.
    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            if grid.start_dates[i] == grid.start_dates[j]:
                assert blocks[i, 0] == blocks[j, 0]
                break


def test_atr_is_scale_free():
    """ATR is normalised by mean close so tickers at different prices are comparable."""
    n = 200
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high, low = close * 1.01, close * 0.99
    a = average_true_range_pct(high, low, close)
    b = average_true_range_pct(high * 50, low * 50, close * 50)
    assert a == pytest.approx(b, rel=1e-9)


def test_unknown_split_is_rejected():
    with pytest.raises(KeyError, match="unknown split"):
        build_grid(synthetic_corpus(), "nonexistent")
