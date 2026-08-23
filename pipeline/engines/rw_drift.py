"""Gaussian random walk with drift -- the engine CandleCast actually serves.

Not a placeholder. On the full 708-window test grid it beat zero-shot Kronos-small by 81%
on fair CRPS and 111% on interval score, and after conformal correction it reached 0.796
coverage at 30% narrower bands than the best Kronos arm (Facts of Record F1, F5).

It is worth being precise about *why* it wins, because it is not that a random walk models
markets well. It is that the one property these forecasts need -- uncertainty that grows
correctly with horizon -- is the random walk's only real feature, and it is exactly the
property the foundation model lacks (F2, F9). The product's value is the calibrated cone,
so the forecaster that calibrates is the one that ships.

Deliberately analytic: no model download, no GPU, no torch import on this path. The
nightly job runs on a free Actions runner in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.engines.base import ENSEMBLE_SIZE, HORIZON, LOOKBACK, Engine

EPS = 1e-12


class RandomWalkDrift(Engine):
    """Log returns are Gaussian with drift and volatility estimated from the lookback."""

    name = "rw_drift"
    validated = True  # measured on the A3 grid; see report Part II

    def __init__(self, lookback: int = LOOKBACK):
        self.lookback = lookback

    def forecast(
        self,
        bars: pd.DataFrame,
        horizon: int = HORIZON,
        m: int = ENSEMBLE_SIZE,
        seed: int = 0,
    ) -> np.ndarray:
        close = self.check_bars(bars, self.lookback)

        log_ret = np.diff(np.log(np.maximum(close, EPS)))
        mu = float(log_ret.mean())
        sigma = float(log_ret.std(ddof=1))
        last = float(close[-1])

        # Same construction as phase-a/eval/baselines.py so the served engine and the
        # evaluated baseline are the same forecaster, not two implementations that agree
        # by inspection. tests/test_engine_parity.py asserts they match bit-for-bit.
        rng = np.random.default_rng(seed)
        shocks = rng.normal(size=(horizon, m))
        cumulative = np.cumsum(mu + sigma * shocks, axis=0)
        paths = last * np.exp(cumulative)
        return paths.T  # (m, horizon)

    def params(self, bars: pd.DataFrame) -> dict:
        """Drift and volatility actually used -- recorded in the run summary."""
        close = self.check_bars(bars, self.lookback)
        log_ret = np.diff(np.log(np.maximum(close, EPS)))
        return {
            "mu_per_session": float(log_ret.mean()),
            "sigma_per_session": float(log_ret.std(ddof=1)),
            "last_close": float(close[-1]),
            "lookback": self.lookback,
        }
