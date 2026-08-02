"""Offline analysis over a completed evaluation run. CPU only.

Baselines are deterministic given the grid and seed, so they can be rebuilt exactly
without a GPU. Model ensembles must be loaded from a run's ``ensembles.npz``.
"""

from __future__ import annotations

import numpy as np
from eval.metrics import QUANTILE_METHOD, coverage_indicator, crps

try:  # scipy is not a declared dependency; fall back to a local inverse-CDF.
    from scipy.stats import norm

    def _z(p: float) -> float:
        return float(norm.ppf(p))
except ImportError:  # pragma: no cover - exercised only without scipy
    import math

    def _z(p: float) -> float:
        """Acklam's rational approximation to the standard normal quantile."""
        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
            )
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
            )
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )


def horizon_exponent(obs: np.ndarray, ens: np.ndarray, steps=(1, 5, 10, 20, 30)) -> dict:
    """Fit CRPS ~ h^k across horizon steps.

    A correctly specified random walk diffuses, giving k ~ 0.5. An exponent materially
    above that means error is compounding faster than diffusion alone -- systematic drift
    rather than widening variance.
    """
    steps = [s for s in steps if s <= obs.shape[1]]
    values = [float(np.mean(crps(obs[:, s - 1 : s], ens[:, s - 1 : s, :]))) for s in steps]
    k, log_a = np.polyfit(np.log(steps), np.log(values), 1)
    return {
        "steps": list(steps),
        "crps_by_step": values,
        "exponent": float(k),
        "intercept": float(np.exp(log_a)),
    }


def z_ratio_table(obs: np.ndarray, ens: np.ndarray, levels=(0.50, 0.80, 0.90)) -> dict:
    """Is miscalibration scale or shape?

    A pure scale error gives a constant ``z_achieved / z_nominal``. Location bias makes
    the ratio drift monotonically upward with nominal level. Constancy therefore *rejects*
    bias-dominated error but does not by itself confirm pure scale error -- a mixture can
    also look flat, so read this alongside the re-centering decomposition.
    """
    rows = []
    for level in levels:
        achieved = float(coverage_indicator(obs, ens, level).mean())
        z_nom = _z((1 + level) / 2)
        z_ach = _z((1 + min(max(achieved, 1e-6), 1 - 1e-6)) / 2)
        rows.append(
            {
                "nominal": level,
                "achieved": achieved,
                "z_nominal": z_nom,
                "z_achieved": z_ach,
                "ratio": z_ach / z_nom,
            }
        )
    ratios = np.array([r["ratio"] for r in rows])
    diffs = np.diff(ratios)
    return {
        "rows": rows,
        "ratio_mean": float(ratios.mean()),
        "ratio_sd": float(ratios.std(ddof=1)),
        "ratio_spread": float(ratios.max() - ratios.min()),
        "monotone_increasing": bool(np.all(diffs > 0)),
        "implied_scale_factor": float(1.0 / ratios.mean()),
    }


def oracle_recenter(obs: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """Subtract each window-step's OWN realized median error. **Degenerate — do not use.**

    Kept only to document the trap. Removing the realized per-window error centres the
    ensemble exactly on the observation, so the observation is inside every band by
    construction and coverage is 1.0 regardless of the forecaster. Verified: an unbiased
    forecast, a systematically biased one and an idiosyncratically biased one all return
    1.0000. It measures nothing about the model.

    Use :func:`remove_systematic_bias` instead.
    """
    median = np.quantile(ens, 0.5, axis=-1, method=QUANTILE_METHOD)
    return ens - (median - obs)[..., None]


def remove_systematic_bias(
    obs: np.ndarray, ens: np.ndarray, relative: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the *average* location error at each horizon step, pooled across windows.

    This is the non-degenerate decomposition. One bias value per horizon step is
    estimated across all windows and applied to all of them, so no window is told its own
    answer. It isolates the component of miscalibration attributable to systematic drift —
    which is what fine-tuning can remove — from dispersion error and idiosyncratic
    per-window error, which it largely cannot.

    Behaviour, verified on synthetic cases at nominal 0.80:

    ===================  ========  =======
    forecast             before    after
    ===================  ========  =======
    unbiased              0.4761   0.4746
    systematic bias       0.2424   0.7971
    idiosyncratic bias    0.2479   0.2471
    ===================  ========  =======

    ``relative=True`` estimates the bias multiplicatively, which is the right choice for
    price panels where levels differ by orders of magnitude across tickers and an additive
    mean would be dominated by the most expensive name.

    Returns ``(corrected_ensemble, bias_per_step)``.
    """
    median = np.quantile(ens, 0.5, axis=-1, method=QUANTILE_METHOD)
    if relative:
        bias = (median / obs - 1.0).mean(axis=0)
        return ens / (1.0 + bias)[None, :, None], bias
    bias = (median - obs).mean(axis=0)
    return ens - bias[None, :, None], bias


def tercile_table(obs, ens, terciles, block_ids, levels=(0.50, 0.80, 0.90)) -> dict:
    """Coverage by volatility regime, using pre-computed tercile labels.

    Labels are passed in rather than recomputed so two models can be compared on
    identical cuts.
    """
    out = {}
    for t, name in enumerate(("calm", "mid", "volatile")):
        idx = np.flatnonzero(terciles == t)
        if len(idx) == 0:
            continue
        out[name] = {
            "n_windows": int(len(idx)),
            "crps": float(np.mean(crps(obs[idx], ens[idx]))),
            **{
                f"coverage_{int(level * 100)}": float(
                    coverage_indicator(obs[idx], ens[idx], level).mean()
                )
                for level in levels
            },
        }
    if {"calm", "volatile"} <= set(out):
        out["calm_minus_volatile_80"] = out["calm"]["coverage_80"] - out["volatile"]["coverage_80"]
    return out


def paired_block_bootstrap_diff(
    per_obs_a: np.ndarray,
    per_obs_b: np.ndarray,
    block_ids: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI for mean(a) - mean(b) where a and b are scored on the SAME windows.

    Resampling the arms independently would ignore that they share windows and would
    overstate the uncertainty of their difference.
    """
    a = np.asarray(per_obs_a).reshape(-1)
    b = np.asarray(per_obs_b).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    if not (a.shape == b.shape == blocks.shape):
        raise ValueError(f"shapes must align: {a.shape}, {b.shape}, {blocks.shape}")

    diff = a - b
    unique = np.unique(blocks)
    grouped = [diff[blocks == u] for u in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(grouped), size=len(grouped))
        means[i] = np.concatenate([grouped[j] for j in pick]).mean()
    lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean_difference": float(diff.mean()),
        "ci95": [lo, hi],
        "includes_zero": bool(lo <= 0.0 <= hi),
    }
