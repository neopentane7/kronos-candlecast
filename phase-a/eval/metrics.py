"""Probabilistic forecast metrics.

Array convention throughout: observations are ``(n_windows, horizon)`` and ensembles are
``(n_windows, horizon, n_samples)`` -- the ensemble axis is last, matching
``scoringrules``.

Coverage is reported as an **estimate with an interval**, never as a bare number. The
evaluation panel has far fewer independent units than it has rows: windows overlap in
time and large-cap Indian equities move together, so a naive binomial interval would
badly overstate precision. ``block_bootstrap_ci`` resamples whole date blocks to keep
that dependence intact.
"""

from __future__ import annotations

import numpy as np
import scoringrules as sr

NOMINAL_LEVELS = (0.50, 0.80, 0.90)

# Plotting position i/(m+1) rather than numpy's default linear interpolation.
#
# This is not a cosmetic choice. The empirical p10/p90 of a small ensemble form a
# systematically NARROWER band than the true p10/p90, so a perfectly calibrated
# forecaster measures as over-confident. Simulated on an exactly calibrated standard
# normal, 200k observations, nominal 80%:
#
#     m       linear    weibull
#     10      0.6627
#     20      0.7253
#     30      0.7496     0.8020      <- A3's sample_count
#     50      0.7673
#    100      0.7837
#    200      0.7926
#   1000      0.7974
#
# At m=30 the default estimator costs 5.04pp of coverage from sampling alone -- 2.5x the
# width of A5's 78-82% acceptance band, before the model contributes any error at all.
# Weibull positions remove essentially all of it at no compute cost. Anything comparing
# nominal to empirical coverage must use this estimator, or it is measuring its own
# ensemble size.
QUANTILE_METHOD = "weibull"


def ensemble_quantile(ens: np.ndarray, q: float, method: str = QUANTILE_METHOD) -> np.ndarray:
    """Per-window, per-horizon quantile across ensemble members."""
    return np.quantile(ens, q, axis=-1, method=method)


def band_bounds(ens: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Central prediction band for a nominal ``level`` (e.g. 0.80 -> p10/p90)."""
    tail = (1.0 - level) / 2.0
    return ensemble_quantile(ens, tail), ensemble_quantile(ens, 1.0 - tail)


# --- point accuracy -------------------------------------------------------------


def point_metrics(obs: np.ndarray, ens: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / MAPE against the ensemble median."""
    median = ensemble_quantile(ens, 0.5)
    err = median - obs
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(err / obs)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.nanmean(np.where(np.isfinite(ape), ape, np.nan))),
    }


# --- proper scoring rules -------------------------------------------------------


def crps(obs: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """CRPS per (window, horizon). Lower is better; reduces to MAE for a point forecast."""
    return np.asarray(sr.crps_ensemble(obs, ens, m_axis=-1))


def interval_score(obs: np.ndarray, ens: np.ndarray, level: float = 0.80) -> np.ndarray:
    """Winkler interval score for the central band at ``level``.

    Rewards narrow bands but penalises misses in proportion to how far outside they fall,
    so it cannot be gamed by widening the cone.
    """
    lower, upper = band_bounds(ens, level)
    return np.asarray(sr.interval_score(obs, lower, upper, alpha=1.0 - level))


def pinball_loss(obs: np.ndarray, ens: np.ndarray, quantiles: tuple[float, ...]) -> float:
    """Mean quantile (pinball) loss averaged over the requested quantile levels."""
    losses = [
        np.asarray(sr.quantile_score(obs, ensemble_quantile(ens, q), alpha=q)) for q in quantiles
    ]
    return float(np.mean(losses))


# --- calibration ----------------------------------------------------------------


def coverage_indicator(obs: np.ndarray, ens: np.ndarray, level: float) -> np.ndarray:
    """Boolean array: did the observation fall inside the nominal band?"""
    lower, upper = band_bounds(ens, level)
    return (obs >= lower) & (obs <= upper)


def pit_values(
    obs: np.ndarray, ens: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Randomized PIT values, uniform on [0, 1] when the forecast is calibrated.

    Randomization breaks the ties a finite ensemble necessarily produces; without it the
    histogram shows spurious structure at ensemble-sized increments.
    """
    rng = rng or np.random.default_rng(0)
    obs_e = obs[..., None]
    below = np.sum(ens < obs_e, axis=-1)
    equal = np.sum(ens == obs_e, axis=-1)
    m = ens.shape[-1]
    u = rng.uniform(size=below.shape)
    return (below + u * equal) / m


def block_bootstrap_ci(
    indicator: np.ndarray,
    block_ids: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling whole blocks rather than observations.

    ``block_ids`` labels the dependence structure -- use the forecast start date, so all
    tickers and all horizon steps sharing a date move together, which is how they behave.
    """
    flat = np.asarray(indicator).reshape(-1).astype(float)
    blocks = np.asarray(block_ids).reshape(-1)
    if flat.shape != blocks.shape:
        raise ValueError(f"indicator {flat.shape} and block_ids {blocks.shape} must align")

    unique = np.unique(blocks)
    grouped = [flat[blocks == b] for b in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(grouped)
    for i in range(n_boot):
        pick = rng.integers(0, n, size=n)
        means[i] = np.concatenate([grouped[j] for j in pick]).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def effective_sample_size(block_ids: np.ndarray) -> int:
    """Number of independent blocks -- the honest denominator for coverage precision."""
    return int(len(np.unique(np.asarray(block_ids).reshape(-1))))


def calibration_report(
    obs: np.ndarray,
    ens: np.ndarray,
    block_ids: np.ndarray,
    levels: tuple[float, ...] = NOMINAL_LEVELS,
    seed: int = 0,
) -> dict:
    """Coverage at each nominal level, each with a block-bootstrap interval."""
    out = {}
    for level in levels:
        ind = coverage_indicator(obs, ens, level)
        lo, hi = block_bootstrap_ci(ind, block_ids, alpha=0.05, seed=seed)
        out[f"{int(level * 100)}"] = {
            "nominal": level,
            "empirical": float(np.mean(ind)),
            "ci95": [lo, hi],
            "covers_nominal": bool(lo <= level <= hi),
        }
    return out


def summarize(obs: np.ndarray, ens: np.ndarray, block_ids: np.ndarray, seed: int = 0) -> dict:
    """The full metric block for one model on one evaluation grid."""
    return {
        "n_windows": int(obs.shape[0]),
        "horizon": int(obs.shape[1]),
        "n_samples": int(ens.shape[-1]),
        "effective_blocks": effective_sample_size(block_ids),
        "point": point_metrics(obs, ens),
        "crps": float(np.mean(crps(obs, ens))),
        "interval_score_80": float(np.mean(interval_score(obs, ens, 0.80))),
        "pinball": pinball_loss(obs, ens, (0.1, 0.25, 0.5, 0.75, 0.9)),
        "coverage": calibration_report(obs, ens, block_ids, seed=seed),
    }
