"""A5's analysis must calibrate on the val grid and never on the test grid.

The whole value of the pre-registration is that the test period is measured rather than
fitted. A bug that quietly let test residuals into the calibration set would produce
better-looking numbers and destroy the claim, without failing anything.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "phase-a")]

from eval.run_analysis import preregistered_conformal, shared_block_mask  # noqa: E402


class FakeNpz(dict):
    @property
    def files(self):
        return list(self)


def grid(n_tickers, n_blocks, horizon=30, m=400, seed=0, scale=1.0):
    """A grid whose ensemble is `scale` times as wide as the outcomes actually require.

    The outcome is a random walk from 100; the ensemble is centred on 100 with the spread
    a correctly specified forecaster would have, multiplied by `scale`. So scale below 1
    under-covers by construction, which is what gives the conformal fitter something real
    to correct.
    """
    rng = np.random.default_rng(seed)
    n = n_tickers * n_blocks
    last = np.full(n, 100.0)
    sigma = 0.02

    truth = last[:, None] * np.exp(np.cumsum(rng.normal(0, sigma, size=(n, horizon)), axis=1))
    step_sd = last[:, None, None] * sigma * np.sqrt(np.arange(1, horizon + 1))[None, :, None]
    ens = last[:, None, None] + rng.normal(0, 1, size=(n, horizon, m)) * step_sd * scale
    rw = last[:, None, None] + rng.normal(0, 1, size=(n, horizon, m)) * step_sd

    tickers = np.array([f"T{i:02d}" for i in range(n_tickers)] * n_blocks, dtype="<U10")
    blocks = np.repeat(np.repeat(np.arange(n_blocks), n_tickers)[:, None], horizon, axis=1)
    return FakeNpz(
        y_close=truth,
        history_close=np.tile(last[:, None], (1, 20)),
        atr_tercile=np.tile(np.arange(3), n // 3 + 1)[:n],
        block_ids=blocks,
        tickers=tickers,
        ens__kronos_zeroshot=ens,
        ens__random_walk_drift=rw,
    )


def test_orphan_blocks_are_dropped_from_the_calibration_set():
    """Section 17b: one ticker's private forecast dates are not exchangeable."""
    g = grid(n_tickers=10, n_blocks=4)
    tickers = g["tickers"].copy()
    blocks = g["block_ids"].copy()
    # T00 wanders onto dates nobody else shares, exactly as BAJAJ-AUTO does in 2024.
    orphan = (tickers == "T00") & (blocks[:, 0] > 0)
    blocks[orphan] = 99

    keep = shared_block_mask(blocks, tickers)
    assert not keep[orphan].any(), "orphan windows must be excluded"
    assert keep[~orphan].all(), "shared windows must be kept"


def test_calibration_comes_from_the_other_run_not_the_test_run():
    """The load-bearing property. Test residuals must never reach the fitter.

    Built so the two grids disagree: calibration is fitted where the ensemble is far too
    narrow, and evaluated where it is correctly sized. A leak would drag the correction
    toward the test grid's own residuals and land coverage near nominal; honest
    calibration overshoots, because it applies the val grid's much larger widening.
    """
    cal = grid(n_tickers=6, n_blocks=6, seed=1, scale=0.30)  # badly under-dispersed
    tst = grid(n_tickers=6, n_blocks=6, seed=2, scale=0.70)  # mildly under-dispersed

    out = preregistered_conformal(cal, tst)
    assert out["design"] == "preregistered_val_2024"
    assert out["n_calibration_windows"] == 36
    assert out["n_test_windows"] == 36

    marginal = out["arms"]["marginal_conformal"]["marginal"]
    raw = out["arms"]["raw_kronos"]["marginal"]
    assert marginal > raw, "the correction must widen the band"
    assert marginal > 0.95, (
        f"a scale fitted where the cone is 0.30x should over-widen a 0.70x cone; got "
        f"{marginal:.3f}. A value near 0.80 would mean the fit saw test residuals."
    )


def test_every_arm_is_reported_including_the_null():
    cal, tst = grid(6, 6, seed=3), grid(6, 6, seed=4)
    arms = preregistered_conformal(cal, tst)["arms"]
    assert set(arms) == {
        "raw_kronos",
        "marginal_conformal",
        "normalized_lookback_vol",
        "mondrian",
        "conformalized_rw_drift",
    }
    for a in arms.values():
        assert 0.0 <= a["marginal"] <= 1.0
        assert set(a["by_regime"]) == {"calm", "mid", "volatile"}


def test_a_calibration_run_without_the_model_is_refused():
    cal, tst = grid(6, 6, seed=5), grid(6, 6, seed=6)
    del cal["ens__kronos_zeroshot"]
    with pytest.raises(SystemExit, match="both runs need kronos_zeroshot"):
        preregistered_conformal(cal, tst)
