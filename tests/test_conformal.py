"""Tests for the conformal calibration layer.

These pin the properties A5's conclusions will rest on, including two negative results:
marginal conformal does not deliver conditional coverage, and normalizing by a biased
risk proxy does not fix that either.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a"))

from eval.conformal import (  # noqa: E402
    apply_mondrian,
    apply_scale,
    conformal_quantile,
    coverage_of,
    fit_mondrian,
    fit_scale,
    mean_width,
)

LEVEL = 0.80


def regime_case(n_per=400, beta=0.5, tight=0.45, horizon=8, m=30, seed=0):
    """A forecaster whose cone is anchored to trailing vol, where vol mean-reverts.

    This reproduces the mechanism diagnosed on the real data: the calm stratum
    under-covers most because its future is relatively more volatile than its past.
    """
    rng = np.random.default_rng(seed)
    n = 3 * n_per
    regime = np.repeat([0, 1, 2], n_per)
    v_look = np.array([0.006, 0.012, 0.024])[regime] * np.exp(rng.normal(0, 0.12, n))
    v_fut = v_look**beta * 0.012 ** (1 - beta)

    steps = np.sqrt(np.arange(1, horizon + 1))[None, :]
    level0 = 1000.0 * np.exp(rng.normal(0, 0.3, n))[:, None]
    obs = level0 * (1 + rng.normal(0, 1, (n, horizon)) * v_fut[:, None] * steps)
    ens = level0[..., None] * (
        1 + rng.normal(0, 1, (n, horizon, m)) * (tight * v_look)[:, None, None] * steps[..., None]
    )
    return obs, ens, regime, v_look * level0[:, 0]


def regime_coverage(obs, lo, hi, regime):
    inside = (obs >= lo) & (obs <= hi)
    return np.array([inside[regime == r].mean() for r in (0, 1, 2)])


# --- the finite-sample quantile -------------------------------------------------


def test_conformal_quantile_uses_the_n_plus_one_correction():
    """With n=9 and alpha=0.2, ceil(10*0.8)=8 -> the 8th smallest, not the 7th."""
    scores = np.arange(1.0, 10.0)  # 1..9
    assert conformal_quantile(scores, 0.20) == 8.0


def test_conformal_quantile_falls_back_when_the_set_is_too_small():
    """n=3 cannot certify 90%: ceil(4*0.9)=4 > 3, so return the max and stay conservative."""
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.10) == 3.0


def test_conformal_quantile_rejects_an_empty_calibration_set():
    with pytest.raises(ValueError, match="empty calibration set"):
        conformal_quantile(np.array([]), 0.20)


# --- marginal calibration works -------------------------------------------------


def test_split_conformal_restores_marginal_coverage():
    cal = regime_case(seed=1)
    test = regime_case(seed=2)
    raw = coverage_of(test[0], *np.percentile(test[1], [10, 90], axis=-1))
    scale = fit_scale(cal[0], cal[1], LEVEL)
    lo, hi = apply_scale(test[1], scale, LEVEL)
    assert raw < 0.60, raw
    assert abs(coverage_of(test[0], lo, hi) - LEVEL) < 0.04


def test_conformal_scale_is_above_one_for_an_under_dispersed_forecast():
    cal = regime_case(seed=3)
    scale = fit_scale(cal[0], cal[1], LEVEL)
    assert np.all(scale > 1.0), scale


def test_conformal_leaves_an_already_calibrated_forecast_near_unchanged():
    rng = np.random.default_rng(4)
    obs_c, obs_t = rng.standard_normal((2000, 5)), rng.standard_normal((2000, 5))
    ens_c = rng.standard_normal((2000, 5, 60))
    ens_t = rng.standard_normal((2000, 5, 60))
    scale = fit_scale(obs_c, ens_c, LEVEL)
    lo, hi = apply_scale(ens_t, scale, LEVEL)
    assert abs(coverage_of(obs_t, lo, hi) - LEVEL) < 0.04
    np.testing.assert_allclose(scale, 1.0, atol=0.25)


# --- the negative results that shape A5 -----------------------------------------


def test_marginal_conformal_does_not_deliver_conditional_coverage():
    """Fixing the average leaves the regime spread intact -- honest on average only."""
    cal, test = regime_case(seed=5), regime_case(seed=6)
    raw_spread = np.ptp(
        regime_coverage(test[0], *np.percentile(test[1], [10, 90], axis=-1), test[2])
    )
    lo, hi = apply_scale(test[1], fit_scale(cal[0], cal[1], LEVEL), LEVEL)

    assert abs(coverage_of(test[0], lo, hi) - LEVEL) < 0.04  # marginal is fixed
    spread = np.ptp(regime_coverage(test[0], lo, hi, test[2]))
    assert spread > 0.20, spread  # conditional is not
    assert spread >= raw_spread - 0.05  # and is no better than before


def test_normalizing_by_a_biased_proxy_does_not_fix_conditional_coverage():
    """Normalizing by trailing vol cannot repair a defect caused by trailing vol.

    The proxy mis-predicts future error scale in exactly the stratum that under-covers,
    so dividing by it reproduces the bias rather than removing it.
    """
    cal, test = regime_case(seed=7), regime_case(seed=8)
    scale = fit_scale(cal[0], cal[1], LEVEL, normalizer=cal[3])
    lo, hi = apply_scale(test[1], scale, LEVEL, normalizer=test[3])
    assert abs(coverage_of(test[0], lo, hi) - LEVEL) < 0.05
    assert np.ptp(regime_coverage(test[0], lo, hi, test[2])) > 0.20


def test_mondrian_delivers_conditional_coverage():
    cal, test = regime_case(seed=9), regime_case(seed=10)
    scales = fit_mondrian(cal[0], cal[1], cal[2], LEVEL)
    lo, hi = apply_mondrian(test[1], scales, test[2], LEVEL)

    per_regime = regime_coverage(test[0], lo, hi, test[2])
    assert abs(coverage_of(test[0], lo, hi) - LEVEL) < 0.04
    assert np.ptp(per_regime) < 0.06, per_regime
    assert np.all(np.abs(per_regime - LEVEL) < 0.06), per_regime


def test_mondrian_is_not_wider_than_marginal_conformal():
    """The cost of stratification is calibration-set size, not interval width.

    Marginal conformal must over-widen the volatile stratum to lift the calm one, so
    per-stratum correction is narrower on average despite covering better.
    """
    cal, test = regime_case(seed=11), regime_case(seed=12)
    lo_m, hi_m = apply_scale(test[1], fit_scale(cal[0], cal[1], LEVEL), LEVEL)
    lo_d, hi_d = apply_mondrian(
        test[1], fit_mondrian(cal[0], cal[1], cal[2], LEVEL), test[2], LEVEL
    )
    assert mean_width(lo_d, hi_d, test[0]) < mean_width(lo_m, hi_m, test[0])


def test_mondrian_falls_back_for_an_unseen_stratum():
    """A regime absent from calibration must not crash serving."""
    cal, test = regime_case(seed=13), regime_case(seed=14)
    scales = fit_mondrian(cal[0], cal[1], cal[2], LEVEL)
    del scales[2]
    lo, hi = apply_mondrian(test[1], scales, test[2], LEVEL)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
