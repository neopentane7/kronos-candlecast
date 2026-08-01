"""Reference forecasters, emitting ensembles in the same shape as Kronos.

Both return ``(n_windows, horizon, n_samples)`` so every metric in ``metrics.py``
applies unchanged. A model that cannot beat these on CRPS is not earning its parameters.

Two baselines with different jobs:

* **last value** is a *point* forecast with no dispersion. Its CRPS collapses to MAE, so
  it sets the accuracy floor and, by construction, has near-zero band coverage -- which
  is the cleanest possible demonstration that accuracy and calibration are different
  claims.
* **random walk with drift** is the honest *probabilistic* baseline. Log returns are
  Gaussian with drift and volatility estimated from the same lookback the model sees, so
  its cone is the one a practitioner would draw without a model at all.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def last_value(last_close: np.ndarray, horizon: int, n_samples: int = 1) -> np.ndarray:
    """Flat forecast at the final observed close.

    ``n_samples`` members are emitted only so the array shape matches the other
    forecasters; they are identical, and the ensemble is deliberately degenerate.
    """
    last_close = np.asarray(last_close, dtype=float).reshape(-1, 1, 1)
    return np.repeat(np.repeat(last_close, horizon, axis=1), n_samples, axis=2)


def random_walk_drift(
    history_close: np.ndarray,
    horizon: int,
    n_samples: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian random walk with drift, fitted per window on the lookback.

    ``history_close`` is ``(n_windows, lookback)``. Drift and volatility come from that
    window's log returns, so the baseline sees exactly the information the model sees.
    """
    hist = np.asarray(history_close, dtype=float)
    if hist.ndim != 2:
        raise ValueError(f"history_close must be (n_windows, lookback), got {hist.shape}")

    log_ret = np.diff(np.log(np.maximum(hist, EPS)), axis=1)
    mu = log_ret.mean(axis=1, keepdims=True)[:, :, None]
    sigma = log_ret.std(axis=1, ddof=1, keepdims=True)[:, :, None]
    last = hist[:, -1][:, None, None]

    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(hist.shape[0], horizon, n_samples))
    cumulative = np.cumsum(mu + sigma * shocks, axis=1)
    return last * np.exp(cumulative)


def build_baselines(
    history_close: np.ndarray, horizon: int, n_samples: int = 30, seed: int = 0
) -> dict[str, np.ndarray]:
    """Every baseline ensemble for one evaluation grid, keyed by name."""
    return {
        "last_value": last_value(history_close[:, -1], horizon, n_samples),
        "random_walk_drift": random_walk_drift(history_close, horizon, n_samples, seed),
    }
