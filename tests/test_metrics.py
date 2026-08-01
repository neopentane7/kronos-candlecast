"""Tests for the probabilistic metric layer.

These verify the properties the A3/A5 conclusions will rest on: that a calibrated
ensemble reports calibrated, that a miscalibrated one is caught, and that the coverage
interval widens when the data are dependent rather than pretending to binomial precision.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a"))

from eval.baselines import build_baselines, last_value, random_walk_drift  # noqa: E402
from eval.metrics import (  # noqa: E402
    band_bounds,
    block_bootstrap_ci,
    calibration_report,
    coverage_indicator,
    crps,
    effective_sample_size,
    interval_score,
    pit_values,
    point_metrics,
    summarize,
)


def calibrated_case(n_windows=400, horizon=5, n_samples=200, seed=0):
    """Observations drawn from the same distribution the ensemble represents."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(100, 5, size=(n_windows, horizon))
    obs = rng.normal(mu, 1.0)
    ens = rng.normal(mu[..., None], 1.0, size=(n_windows, horizon, n_samples))
    return obs, ens


# --- scoring rules --------------------------------------------------------------


def test_crps_of_point_forecast_equals_absolute_error():
    """A degenerate ensemble reduces CRPS to MAE; this pins the units."""
    obs = np.array([[1.0, 2.0]])
    ens = np.full((1, 2, 50), 1.5)
    np.testing.assert_allclose(crps(obs, ens), np.array([[0.5, 0.5]]), atol=1e-9)


def test_crps_rewards_the_sharper_of_two_unbiased_forecasts():
    rng = np.random.default_rng(1)
    obs = np.zeros((200, 1))
    sharp = rng.normal(0, 1.0, size=(200, 1, 300))
    diffuse = rng.normal(0, 4.0, size=(200, 1, 300))
    assert crps(obs, sharp).mean() < crps(obs, diffuse).mean()


def test_interval_score_penalises_a_miss_more_than_a_wide_band():
    obs = np.array([[10.0]])
    covering = np.linspace(5, 15, 200).reshape(1, 1, 200)
    missing = np.linspace(20, 21, 200).reshape(1, 1, 200)
    assert interval_score(obs, missing, 0.80) > interval_score(obs, covering, 0.80)


def test_point_metrics_are_computed_off_the_median():
    obs = np.array([[10.0, 10.0]])
    ens = np.tile(np.linspace(8, 12, 101), (1, 2, 1))
    m = point_metrics(obs, ens)
    assert m["mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-9)


# --- calibration ----------------------------------------------------------------


def test_calibrated_ensemble_reports_near_nominal_coverage():
    obs, ens = calibrated_case()
    for level in (0.50, 0.80, 0.90):
        emp = coverage_indicator(obs, ens, level).mean()
        assert abs(emp - level) < 0.03, f"level {level} -> {emp}"


def test_small_ensemble_does_not_fake_overconfidence():
    """The measurement must not manufacture the very defect the project reports.

    With numpy's default interpolation, a perfectly calibrated 30-member ensemble
    measures ~0.75 at nominal 0.80 -- a 5pp artifact that would be indistinguishable
    from real model over-confidence. Weibull plotting positions remove it.
    """
    rng = np.random.default_rng(11)
    n = 60_000
    obs = rng.standard_normal((n, 1))
    ens = rng.standard_normal((n, 1, 30))

    corrected = coverage_indicator(obs, ens, 0.80).mean()
    assert abs(corrected - 0.80) < 0.01, f"finite-ensemble bias not corrected: {corrected}"

    naive_lo = np.quantile(ens, 0.10, axis=-1, method="linear")
    naive_hi = np.quantile(ens, 0.90, axis=-1, method="linear")
    naive = np.mean((obs >= naive_lo) & (obs <= naive_hi))
    assert naive < 0.77, "expected the default estimator to under-cover at m=30"


def test_overconfident_ensemble_is_detected():
    """The failure mode the whole project exists to measure: too-narrow bands."""
    rng = np.random.default_rng(2)
    mu = np.zeros((500, 3))
    obs = rng.normal(mu, 1.0)
    ens = rng.normal(mu[..., None], 0.3, size=(500, 3, 200))  # far too sharp
    assert coverage_indicator(obs, ens, 0.80).mean() < 0.45


def test_band_bounds_are_ordered_and_nested():
    _, ens = calibrated_case(n_windows=20)
    lo50, hi50 = band_bounds(ens, 0.50)
    lo90, hi90 = band_bounds(ens, 0.90)
    assert np.all(lo50 <= hi50)
    assert np.all(lo90 <= lo50) and np.all(hi50 <= hi90)


def test_pit_is_uniform_when_calibrated():
    obs, ens = calibrated_case(n_windows=800, horizon=2)
    pit = pit_values(obs, ens).reshape(-1)
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    # Deciles of a uniform sample should each hold ~10% of the mass.
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    frac = hist / hist.sum()
    assert np.all(np.abs(frac - 0.1) < 0.025), frac


def test_pit_is_skewed_when_forecast_is_biased():
    rng = np.random.default_rng(3)
    obs = rng.normal(5.0, 1.0, size=(600, 2))
    ens = rng.normal(0.0, 1.0, size=(600, 2, 200))  # forecast far below observations
    assert pit_values(obs, ens).mean() > 0.9


# --- uncertainty about coverage itself ------------------------------------------


def test_block_bootstrap_interval_contains_the_truth_when_calibrated():
    obs, ens = calibrated_case(n_windows=300, horizon=4)
    blocks = np.repeat(np.arange(300), 4)  # each window is its own date block
    ind = coverage_indicator(obs, ens, 0.80)
    lo, hi = block_bootstrap_ci(ind, blocks, n_boot=500, seed=0)
    assert lo <= 0.80 <= hi


def test_dependence_widens_the_interval():
    """Correlated units carry less information, and the interval must say so.

    The dependence has to be real, not just asserted by the labelling: rows that share a
    date genuinely succeed or fail together, which is how a market panel behaves when one
    day is calm and the next is a crash.
    """
    rng = np.random.default_rng(9)
    n_blocks, per_block = 12, 100
    block_rate = rng.uniform(0.5, 1.0, size=n_blocks)  # each date has its own hit rate
    ind = rng.uniform(size=(n_blocks, per_block)) < block_rate[:, None]

    clustered = np.repeat(np.arange(n_blocks), per_block)
    independent = np.arange(ind.size)

    lo_c, hi_c = block_bootstrap_ci(ind, clustered, n_boot=400, seed=0)
    lo_i, hi_i = block_bootstrap_ci(ind, independent, n_boot=400, seed=0)
    assert (hi_c - lo_c) > 3 * (hi_i - lo_i)


def test_labelling_iid_data_as_clustered_does_not_change_the_interval():
    """The converse guard: blocking must not manufacture uncertainty that isn't there."""
    obs, ens = calibrated_case(n_windows=240, horizon=5)
    ind = coverage_indicator(obs, ens, 0.80)

    lo_i, hi_i = block_bootstrap_ci(ind, np.arange(ind.size), n_boot=400, seed=0)
    lo_c, hi_c = block_bootstrap_ci(
        ind, np.repeat(np.arange(12), ind.size // 12), n_boot=400, seed=0
    )
    assert abs((hi_c - lo_c) - (hi_i - lo_i)) < 0.02


def test_effective_sample_size_counts_blocks_not_rows():
    blocks = np.repeat(np.arange(12), 100)
    assert effective_sample_size(blocks) == 12
    assert blocks.size == 1200


def test_calibration_report_flags_whether_nominal_is_covered():
    obs, ens = calibrated_case(n_windows=300, horizon=3)
    blocks = np.repeat(np.arange(300), 3)
    rep = calibration_report(obs, ens, blocks)
    assert set(rep) == {"50", "80", "90"}
    assert rep["80"]["covers_nominal"] is True
    assert rep["80"]["ci95"][0] <= rep["80"]["empirical"] <= rep["80"]["ci95"][1]


# --- baselines ------------------------------------------------------------------


def test_last_value_is_flat_and_degenerate():
    ens = last_value(np.array([100.0, 50.0]), horizon=4, n_samples=7)
    assert ens.shape == (2, 4, 7)
    assert np.all(ens[0] == 100.0) and np.all(ens[1] == 50.0)
    assert ens.std(axis=-1).max() == 0.0


def test_last_value_has_no_coverage():
    """Accuracy and calibration are different claims; this baseline proves it."""
    obs = np.array([[101.0, 102.0]])
    ens = last_value(np.array([100.0]), horizon=2, n_samples=30)
    assert coverage_indicator(obs, ens, 0.80).mean() == 0.0


def test_random_walk_drift_widens_with_horizon():
    rng = np.random.default_rng(4)
    hist = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(5, 400)), axis=1))
    ens = random_walk_drift(hist, horizon=30, n_samples=200, seed=0)
    assert ens.shape == (5, 30, 200)
    spread = ens.std(axis=-1)
    assert np.all(spread[:, -1] > spread[:, 0])


def test_random_walk_drift_starts_at_the_last_close():
    hist = np.tile(np.linspace(90, 100, 400), (3, 1))
    ens = random_walk_drift(hist, horizon=5, n_samples=500, seed=1)
    assert np.allclose(np.median(ens[:, 0, :], axis=-1), hist[:, -1], rtol=0.02)


def test_build_baselines_shapes_agree():
    rng = np.random.default_rng(5)
    hist = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(6, 400)), axis=1))
    out = build_baselines(hist, horizon=30, n_samples=20)
    assert set(out) == {"last_value", "random_walk_drift"}
    for ens in out.values():
        assert ens.shape == (6, 30, 20)


def test_summarize_produces_the_full_metric_block():
    obs, ens = calibrated_case(n_windows=60, horizon=3, n_samples=50)
    blocks = np.repeat(np.arange(60), 3)
    s = summarize(obs, ens, blocks)
    assert s["n_windows"] == 60 and s["horizon"] == 3 and s["n_samples"] == 50
    assert s["effective_blocks"] == 60
    assert set(s["point"]) == {"mae", "rmse", "mape"}
    assert s["crps"] > 0 and s["interval_score_80"] > 0 and s["pinball"] > 0
