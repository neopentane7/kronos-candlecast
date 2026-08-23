"""Bands must be ordered, must respond to ACI state, and must not silently cross."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.aci import ACIState  # noqa: E402
from pipeline.calibration import (  # noqa: E402
    aci_band,
    build_bands,
    enforce_monotone,
    path_probabilities,
    raw_band,
    split_conformal_band,
)


def paths(m=2000, horizon=30, seed=0, last=100.0, sigma=0.02):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=(m, horizon))
    return last * np.exp(np.cumsum(steps, axis=1))


def test_raw_band_recovers_the_nominal_level():
    p = paths(m=20000)
    lo, hi = raw_band(p, 0.80)
    inside = ((p >= lo) & (p <= hi)).mean(axis=0)
    assert np.allclose(inside, 0.80, atol=0.02)


def test_weibull_is_used_not_the_numpy_default():
    """At m=30 the default estimator measures a calibrated forecaster ~5pp low (§1)."""
    p = paths(m=30, seed=1)
    lo_w, hi_w = raw_band(p, 0.80)
    lo_d = np.quantile(p, 0.10, axis=0)  # numpy default: linear / type 7
    assert np.all(lo_w <= lo_d + 1e-12), "weibull must sit at or outside the type-7 band"
    assert np.any(lo_w < lo_d), "the two estimators must actually differ at m=30"


def test_aci_band_widens_where_the_state_says_to():
    p = paths()
    s = ACIState(gamma=0.05)
    lo0, hi0 = aci_band(p, "X", 0.80, s)

    for _ in range(20):  # a run of misses at the far end only
        s.update("X", 0.80, 30, covered=False)
    lo1, hi1 = aci_band(p, "X", 0.80, s)

    assert (hi1 - lo1)[29] > (hi0 - lo0)[29], "misses at h=30 must widen h=30"
    assert (hi1 - lo1)[0] == pytest.approx((hi0 - lo0)[0]), "h=1 must be untouched"


def test_split_conformal_scale_widens_symmetrically_about_the_median():
    p = paths()
    lo1, hi1 = split_conformal_band(p, 0.50, scale=1.0)
    lo2, hi2 = split_conformal_band(p, 0.50, scale=2.0)
    centre = np.quantile(p, 0.5, axis=0, method="weibull")
    assert np.allclose((hi2 - lo2), 2 * (hi1 - lo1))
    assert np.allclose((lo2 + hi2) / 2, centre)


def test_build_bands_emits_an_ordered_ladder():
    b = build_bands(paths(), "X", ACIState(), split_scale=1.0)
    stacked = np.vstack([b[k] for k in ("p10", "p25", "p50", "p75", "p90")])
    assert np.all(np.diff(stacked, axis=0) >= -1e-9)
    assert b["method_50"] == "split_conformal"


def test_build_bands_falls_back_to_aci_when_no_split_state_exists():
    b = build_bands(paths(), "X", ACIState(), split_scale=None)
    assert b["method_50"] == "aci", "the 50% method must be reported honestly, not assumed"


def test_crossings_are_repaired_and_counted():
    """An inner band widened by a scale the outer one never saw can overtake it."""
    n = 5
    bands = {
        "p10": np.full(n, 100.0),
        "p25": np.full(n, 90.0),  # crosses below p10
        "p50": np.full(n, 100.0),
        "p75": np.full(n, 110.0),
        "p90": np.full(n, 105.0),  # crosses below p75
    }
    out = enforce_monotone(bands)
    assert out["crossings_repaired"] == 2 * n
    stacked = np.vstack([out[k] for k in ("p10", "p25", "p50", "p75", "p90")])
    assert np.all(np.diff(stacked, axis=0) >= -1e-9)


def test_clean_ladder_reports_no_crossings():
    b = build_bands(paths(), "X", ACIState(), split_scale=1.0)
    assert enforce_monotone(b)["crossings_repaired"] == 0


def test_path_probabilities_are_in_range_and_directionally_right():
    p = paths(m=4000, seed=2, last=100.0, sigma=0.02)
    out = path_probabilities(p, last_close=100.0, lookback_sigma=0.02)
    assert len(out["prob_above_last_close"]) == 30
    assert all(0.0 <= v <= 1.0 for v in out["prob_above_last_close"])
    assert 0.0 <= out["prob_vol_exceeds_recent"] <= 1.0
    # Driftless: roughly half the scenarios finish above the anchor.
    assert abs(out["prob_above_last_close"][-1] - 0.5) < 0.08


def test_prob_above_tracks_a_strong_drift():
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(rng.normal(0.01, 0.005, size=(2000, 30)), axis=1))
    out = path_probabilities(p, last_close=100.0, lookback_sigma=0.005)
    assert out["prob_above_last_close"][-1] > 0.95
