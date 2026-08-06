"""The diagnostics decide what gets pre-registered, so they get the same scrutiny.

Each test below pins a property that, if it silently broke, would produce a plausible
number rather than an error -- which is the only kind of bug that matters here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.diagnose import (  # noqa: E402
    _blocks,
    _chronological_blocks,
    _elasticity,
    _spearman,
    alignment_report,
    blocks_per_stratum,
    split_feasibility,
    spread_vs_input_vol,
)

HORIZON = 30


def make_npz(block_of_window, tercile, tickers, start_dates, n_samples=30, seed=0):
    """A dict shaped like the real ensembles.npz, with only the fields under test."""
    n = len(block_of_window)
    rng = np.random.default_rng(seed)
    hist = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 401)), axis=1))
    y = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, HORIZON)), axis=1))
    return {
        "block_ids": np.repeat(np.asarray(block_of_window)[:, None], HORIZON, axis=1),
        "atr_tercile": np.asarray(tercile),
        "tickers": np.asarray(tickers, dtype="<U10"),
        "start_dates": np.asarray(start_dates, dtype="<U10"),
        "history_close": hist,
        "y_close": y,
        "ens__m": rng.lognormal(0, 0.1, size=(n, HORIZON, n_samples)),
        "files": [],
    }


class Npz(dict):
    """np.load returns an object with .files; a plain dict does not."""

    @property
    def files(self):
        return [k for k in self if k != "files"]


def aligned_grid(n_blocks=6, n_tickers=4):
    b, t, k, d = [], [], [], []
    for i in range(n_blocks):
        for j in range(n_tickers):
            b.append(i)
            t.append(j % 3)
            k.append(f"T{j}")
            d.append(f"2025-{i + 1:02d}-01")
    return Npz(make_npz(b, t, k, d))


def test_blocks_rejects_ids_that_vary_within_a_window():
    """The whole collapse to one label per window rests on this being constant."""
    bad = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="vary within a window"):
        _blocks(bad)


def test_alignment_report_is_clean_on_an_aligned_grid():
    rep = alignment_report(aligned_grid())
    assert rep["aligned"]
    assert rep["n_blocks_orphan"] == 0
    assert rep["effective_blocks_corrected"] == rep["n_blocks_reported"]


def test_alignment_report_finds_the_single_misaligned_ticker():
    """The real defect: one ticker with an extra session invents its own blocks.

    Reported effective sample size then counts one ticker's private dates as if they
    were independent forecast dates shared by the universe.
    """
    g = aligned_grid(n_blocks=4, n_tickers=4)
    b = g["block_ids"][:, 0].copy()
    tick = g["tickers"].copy()
    dates = g["start_dates"].copy()
    # T3 drifts off the shared calendar from its second window onward.
    for new_block, idx in enumerate(np.flatnonzero((tick == "T3") & (b > 0)), start=100):
        b[idx] = new_block
        dates[idx] = f"2026-{new_block - 99:02d}-01"
    g["block_ids"] = np.repeat(b[:, None], HORIZON, axis=1)
    g["start_dates"] = dates

    rep = alignment_report(g)
    assert not rep["aligned"]
    assert rep["n_blocks_orphan"] == 3
    assert rep["tickers_causing_orphans"] == {"T3": 3}
    assert rep["effective_blocks_corrected"] == 4  # the shared dates, not 4 + 3
    assert rep["n_blocks_reported"] == 7


def test_chronological_blocks_orders_by_date_not_by_label():
    """factorize numbers by first appearance, which is not chronological order.

    Sorting labels produced a 'temporal' split that put a 2025 block after a 2026 one.
    """
    blocks = np.array([0, 1, 2])
    dates = np.array(["2026-01-01", "2025-01-01", "2025-06-01"], dtype="<U10")
    assert list(_chronological_blocks(blocks, dates)) == [1, 2, 0]


def test_blocks_per_stratum_counts_blocks_not_windows():
    """Many windows sharing one date are one observation, not many."""
    b = [0] * 30 + [1] * 30
    t = [2] * 60  # all volatile
    g = Npz(make_npz(b, t, ["T0"] * 60, ["2025-01-01"] * 30 + ["2025-02-01"] * 30))
    out = blocks_per_stratum(g)
    vol = out["strata"]["volatile"]
    assert vol["n_windows"] == 60
    assert vol["effective_blocks"] == 2
    assert vol["feasible_by_windows"]["80"] is True  # 60 >= 9
    assert vol["feasible_by_blocks"]["80"] is False  # 2 < 9 -- the honest answer


def test_split_feasibility_requires_both_sides_to_clear_the_floor():
    g = aligned_grid(n_blocks=6, n_tickers=3)
    out = split_feasibility(g, level=0.8)  # floor 9, only 6 blocks exist
    assert out["n_feasible_splits"] == 0
    assert "no temporal split" in out["verdict"]

    wide = aligned_grid(n_blocks=24, n_tickers=3)
    out2 = split_feasibility(wide, level=0.8)
    assert out2["n_feasible_splits"] > 0
    # Every workable split must leave >= 9 blocks on each side in every stratum.
    for k in out2["feasible_cal_block_counts"]:
        assert 9 <= k <= 24 - 9


def test_split_feasibility_is_ordered_in_time():
    """Calibration must precede test, or the quantile is fitted on the future."""
    g = aligned_grid(n_blocks=4, n_tickers=3)
    out = split_feasibility(g, level=0.5)
    assert [r["n_cal_blocks"] for r in out["splits"]] == [1, 2, 3]
    assert [r["n_test_blocks"] for r in out["splits"]] == [3, 2, 1]


def test_elasticity_recovers_a_known_slope():
    """A forecaster whose spread is vol**0.5 must measure beta = 0.5."""
    vol = np.linspace(0.01, 0.05, 200)
    assert _elasticity(vol**0.5, vol) == pytest.approx(0.5, abs=1e-9)
    assert _elasticity(vol, vol) == pytest.approx(1.0, abs=1e-9)
    assert _elasticity(np.full_like(vol, 3.0), vol) == pytest.approx(0.0, abs=1e-9)


def test_elasticity_and_rank_correlation_come_apart():
    """The distinction the diagnosis rests on.

    A forecaster that ranks volatility perfectly but barely responds to it has rank
    correlation 1 and elasticity near 0. Reporting only the correlation would call it
    regime-aware; reporting only the elasticity would call it blind. Both are needed.
    """
    vol = np.linspace(0.01, 0.05, 200)
    compressed = vol**0.05
    assert _spearman(compressed, vol) == pytest.approx(1.0, abs=1e-9)
    assert _elasticity(compressed, vol) == pytest.approx(0.05, abs=1e-9)


def test_spearman_is_rank_based_not_linear():
    x = np.arange(50, dtype=float)
    assert _spearman(np.exp(x / 10), x) == pytest.approx(1.0, abs=1e-9)
    assert _spearman(-x, x) == pytest.approx(-1.0, abs=1e-9)


def test_spread_vs_input_vol_flags_a_degenerate_forecaster():
    """last_value has zero spread; a correlation against it is undefined, not zero."""
    g = aligned_grid(n_blocks=12, n_tickers=4)
    g["ens__flat"] = np.repeat(g["y_close"][:, :, None], 30, axis=2) * 0 + 5.0
    out = spread_vs_input_vol(g, model_keys=["flat"])
    assert out["models"]["flat"]["degenerate"] is True
