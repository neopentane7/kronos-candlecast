"""A4's sweep must compare arms on identical windows and decide by the pinned rule.

The sweep exists to answer one pre-registered question, so the parts that could quietly
make it answer a different one get pinned: the panel must be block-balanced, the arms must
share it, and the reference must be chosen by CRPS rather than by the metric the
intervention is designed to move.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "phase-a"))
sys.path.insert(0, str(REPO))

from eval.analysis import spread_ratio_by_horizon  # noqa: E402


def test_spread_ratio_is_one_for_a_correctly_sized_ensemble():
    """A forecaster whose spread matches the realized spread must measure 1.0."""
    rng = np.random.default_rng(0)
    n, h, m = 400, 6, 200
    last = np.full(n, 100.0)
    history = np.tile(last[:, None], (1, 10))

    # Realized moves and the ensemble are drawn from the same distribution, so the
    # predicted spread should equal the realized spread at every step.
    sigma = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    obs = last[:, None] * (1.0 + rng.normal(0, sigma, size=(n, h)))
    ens = last[:, None, None] * (1.0 + rng.normal(0, sigma[None, :, None], size=(n, h, m)))

    out = spread_ratio_by_horizon(obs, ens, history)
    for v in out["ratio_by_step"]:
        assert v == pytest.approx(1.0, abs=0.08)


def test_spread_ratio_detects_a_narrow_ensemble():
    """Halving the ensemble spread must halve the ratio, at every step."""
    rng = np.random.default_rng(1)
    n, h, m = 400, 4, 200
    last = np.full(n, 50.0)
    history = np.tile(last[:, None], (1, 10))
    sigma = 0.03

    obs = last[:, None] * (1.0 + rng.normal(0, sigma, size=(n, h)))
    ens = last[:, None, None] * (1.0 + rng.normal(0, sigma / 2, size=(n, h, m)))

    out = spread_ratio_by_horizon(obs, ens, history)
    assert out["ratio_h1"] == pytest.approx(0.5, abs=0.08)
    assert out["ratio_hmax"] == pytest.approx(0.5, abs=0.08)


def test_spread_ratio_reports_the_three_called_out_steps():
    rng = np.random.default_rng(2)
    n, h, m = 50, 30, 20
    history = np.full((n, 10), 100.0)
    obs = 100 * (1 + rng.normal(0, 0.02, size=(n, h)))
    ens = 100 * (1 + rng.normal(0, 0.02, size=(n, h, m)))

    out = spread_ratio_by_horizon(obs, ens, history)
    assert len(out["ratio_by_step"]) == 30
    assert out["ratio_h1"] == out["ratio_by_step"][0]
    assert out["ratio_hmid"] == out["ratio_by_step"][14]  # h = 15
    assert out["ratio_hmax"] == out["ratio_by_step"][29]  # h = 30


class FakeGrid:
    """Enough of EvalGrid for subsample_by_block."""

    def __init__(self, tickers, dates):
        from eval.windows import EvalGrid

        rows = [(t, d) for t in tickers for d in dates]
        n = len(rows)
        self.grid = EvalGrid(
            features={t: np.zeros((10, 6)) for t in tickers},
            timestamps={t: np.arange(10) for t in tickers},
            tickers=[r[0] for r in rows],
            offsets=list(range(n)),
            start_dates=[r[1] for r in rows],
            y_close=np.zeros((n, 3)),
            history_close=np.ones((n, 5)),
            atr_pct=np.linspace(0.01, 0.05, n),
            lookback=5,
            horizon=3,
        )
        self.grid.meta = {"n_tickers": len(tickers)}


def test_panel_is_balanced_across_every_block():
    """Every arm needs all date blocks, or the horizon curve reflects one period."""
    tickers = [f"T{i:02d}" for i in range(20)]
    dates = [f"2025-{m:02d}-01" for m in range(1, 13)]
    panel = FakeGrid(tickers, dates).grid.subsample_by_block(5, seed=1234)

    assert panel.meta["n_tickers"] == 5
    assert panel.meta["n_blocks"] == 12
    assert len(panel) == 60
    counts = {d: sum(1 for x in panel.start_dates if x == d) for d in dates}
    assert set(counts.values()) == {5}, "each block must carry the same five tickers"


def test_panel_selection_is_deterministic_in_the_seed():
    tickers = [f"T{i:02d}" for i in range(20)]
    dates = [f"2025-{m:02d}-01" for m in range(1, 7)]
    g = FakeGrid(tickers, dates).grid

    a = g.subsample_by_block(5, seed=1234).meta["panel_tickers"]
    b = g.subsample_by_block(5, seed=1234).meta["panel_tickers"]
    c = g.subsample_by_block(5, seed=99).meta["panel_tickers"]
    assert a == b
    assert a != c
    assert len(a) == 5


def test_panel_records_its_ticker_list():
    """The deliverable requires the panel be reproducible from results.json alone."""
    tickers = [f"T{i:02d}" for i in range(12)]
    panel = FakeGrid(tickers, ["2025-01-01", "2025-02-01"]).grid.subsample_by_block(4, seed=7)
    listed = panel.meta["panel_tickers"]
    assert sorted(listed) == listed
    assert set(listed) == set(panel.tickers)


def test_reference_is_chosen_by_crps_not_by_spread_ratio():
    """The pinned rule, stated as a test.

    Selecting on spread ratio would reward any arm that widens the cone with junk mass.
    CRPS decides; coverage@80 breaks ties.
    """
    rows = [
        {"temperature": 1.0, "crps_fair": 100.0, "coverage_80": 0.41, "spread_ratio_h30": 0.48},
        {"temperature": 1.5, "crps_fair": 140.0, "coverage_80": 0.62, "spread_ratio_h30": 0.90},
    ]
    best = min(rows, key=lambda r: (round(r["crps_fair"], 6), -r["coverage_80"]))
    assert best["temperature"] == 1.0, "the wider, worse-scoring arm must not win"


def test_tie_on_crps_is_broken_by_coverage():
    rows = [
        {"temperature": 1.0, "crps_fair": 100.0, "coverage_80": 0.41, "spread_ratio_h30": 0.48},
        {"temperature": 1.1, "crps_fair": 100.0, "coverage_80": 0.55, "spread_ratio_h30": 0.60},
    ]
    best = min(rows, key=lambda r: (round(r["crps_fair"], 6), -r["coverage_80"]))
    assert best["temperature"] == 1.1


def test_branch_fires_on_the_registered_threshold():
    from eval.calibrate import SPREAD_RATIO_TARGET

    assert SPREAD_RATIO_TARGET == 0.85
    below = [{"spread_ratio_h30": 0.84}]
    at = [{"spread_ratio_h30": 0.85}]
    assert not [r for r in below if r["spread_ratio_h30"] >= SPREAD_RATIO_TARGET]
    assert [r for r in at if r["spread_ratio_h30"] >= SPREAD_RATIO_TARGET]


def test_temperature_arms_are_the_registered_four():
    from eval.calibrate import TEMPERATURE_SWEEP

    assert TEMPERATURE_SWEEP == (1.0, 1.1, 1.3, 1.5)
    assert TEMPERATURE_SWEEP[0] == 1.0, "T=1.0 must be re-run, not borrowed from the grid"
