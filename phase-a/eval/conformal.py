"""Split-conformal calibration of sampled forecast cones.

The A3 measurements say the zero-shot cone has approximately the right shape and is
uniformly too tight, so the correction needed is a **multiplicative scale per horizon
step** rather than a reshaping. That is what these functions produce.

Three variants, differing only in what the nonconformity score is divided by:

* ``None``          -- score is ``|obs - centre| / ensemble_halfwidth``. The ensemble's own
                       spread is the normalizer, so the correction inherits whatever
                       conditional behaviour the model already has.
* an array          -- score is ``|obs - centre| / normalizer``. Substitutes an ex-ante
                       risk proxy for the model's spread. Only helps if that proxy predicts
                       future error scale better than the model does.
* Mondrian strata   -- a separate correction per stratum. Buys conditional coverage
                       directly at the cost of dividing the calibration set.

All use the finite-sample corrected quantile ``ceil((n+1)(1-alpha))/n``, the same ``n+1``
rank argument that governs the ensemble-size bias in ``metrics.py``.
"""

from __future__ import annotations

import numpy as np
from eval.metrics import QUANTILE_METHOD


def _centre_and_halfwidth(ens: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - level) / 2.0
    lo = np.quantile(ens, tail, axis=-1, method=QUANTILE_METHOD)
    hi = np.quantile(ens, 1.0 - tail, axis=-1, method=QUANTILE_METHOD)
    centre = np.quantile(ens, 0.5, axis=-1, method=QUANTILE_METHOD)
    return centre, np.maximum((hi - lo) / 2.0, 1e-12)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample corrected empirical quantile of the calibration scores.

    Uses ``ceil((n+1)(1-alpha))/n`` rather than the plain ``1-alpha`` quantile. Same
    exchangeability argument as the ensemble-size correction: with ``n`` calibration
    points, a fresh score's rank among ``n+1`` values is uniform.
    """
    s = np.sort(np.asarray(scores).reshape(-1))
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(s[-1])  # calibration set too small to certify this level
    return float(s[k - 1])


def fit_scale(
    cal_obs: np.ndarray,
    cal_ens: np.ndarray,
    level: float = 0.80,
    normalizer: np.ndarray | None = None,
) -> np.ndarray:
    """Per-horizon multiplicative correction, shaped ``(horizon,)``."""
    alpha = 1.0 - level
    centre, half = _centre_and_halfwidth(cal_ens, level)
    denom = half if normalizer is None else np.asarray(normalizer).reshape(-1, 1)
    scores = np.abs(cal_obs - centre) / np.maximum(denom, 1e-12)
    return np.array([conformal_quantile(scores[:, h], alpha) for h in range(scores.shape[1])])


def apply_scale(
    ens: np.ndarray,
    scale: np.ndarray,
    level: float = 0.80,
    normalizer: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformalized band bounds for ``ens``."""
    centre, half = _centre_and_halfwidth(ens, level)
    denom = half if normalizer is None else np.asarray(normalizer).reshape(-1, 1)
    width = scale[None, :] * denom
    return centre - width, centre + width


def fit_mondrian(
    cal_obs: np.ndarray,
    cal_ens: np.ndarray,
    cal_strata: np.ndarray,
    level: float = 0.80,
) -> dict[int, np.ndarray]:
    """A separate per-horizon correction for each stratum."""
    out = {}
    for s in np.unique(cal_strata):
        idx = np.flatnonzero(cal_strata == s)
        out[int(s)] = fit_scale(cal_obs[idx], cal_ens[idx], level)
    return out


def apply_mondrian(
    ens: np.ndarray,
    scales: dict[int, np.ndarray],
    strata: np.ndarray,
    level: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    centre, half = _centre_and_halfwidth(ens, level)
    fallback = np.mean(list(scales.values()), axis=0)
    per_row = np.stack([scales.get(int(s), fallback) for s in strata])
    width = per_row * half
    return centre - width, centre + width


def coverage_of(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((obs >= lo) & (obs <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray, obs: np.ndarray) -> float:
    """Mean band width relative to the observed level, so it compares across tickers."""
    return float(np.mean((hi - lo) / np.abs(obs)))
