"""Tests for the offline analysis layer.

The claims these functions support are load-bearing for A5, so each is pinned against a
case whose answer is known analytically or by construction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a"))

from eval.analysis import (  # noqa: E402
    horizon_exponent,
    oracle_recenter,
    paired_block_bootstrap_diff,
    remove_systematic_bias,
    tercile_table,
    z_ratio_table,
)
from eval.metrics import coverage_indicator  # noqa: E402

# --- horizon exponent -----------------------------------------------------------


def test_horizon_exponent_recovers_diffusion_scaling():
    """A random walk widens as sqrt(h); the fit must return ~0.5."""
    rng = np.random.default_rng(0)
    n, h, m = 4000, 30, 60
    steps = np.arange(1, h + 1)
    sigma = np.sqrt(steps)[None, :, None]
    obs = rng.normal(0, np.sqrt(steps), size=(n, h))
    ens = rng.normal(0, 1, size=(n, h, m)) * sigma
    k = horizon_exponent(obs, ens)["exponent"]
    assert 0.45 <= k <= 0.55, k


def test_horizon_exponent_detects_compounding_drift():
    """Adding a drift that grows linearly must push the exponent above 0.5."""
    rng = np.random.default_rng(1)
    n, h, m = 4000, 30, 60
    steps = np.arange(1, h + 1)
    obs = rng.normal(0.3 * steps, np.sqrt(steps), size=(n, h))
    ens = rng.normal(0, 1, size=(n, h, m)) * np.sqrt(steps)[None, :, None]
    assert horizon_exponent(obs, ens)["exponent"] > 0.6


# --- z-ratio --------------------------------------------------------------------


def test_z_ratio_is_flat_for_a_pure_scale_error():
    rng = np.random.default_rng(2)
    n, m = 60_000, 200
    obs = rng.standard_normal((n, 1))
    ens = rng.standard_normal((n, 1, m)) * 0.5  # exactly half the correct width
    z = z_ratio_table(obs, ens)
    assert z["ratio_spread"] < 0.02, z
    assert abs(z["ratio_mean"] - 0.5) < 0.03
    assert abs(z["implied_scale_factor"] - 2.0) < 0.15


def test_z_ratio_drifts_upward_for_a_pure_location_bias():
    """The discriminating signature: bias makes the ratio climb with nominal level."""
    rng = np.random.default_rng(3)
    n, m = 60_000, 200
    bias = rng.choice([-1.1, 1.1], size=(n, 1))
    obs = rng.standard_normal((n, 1)) + bias
    ens = rng.standard_normal((n, 1, m))
    z = z_ratio_table(obs, ens)
    assert z["monotone_increasing"], z["rows"]
    assert z["ratio_spread"] > 0.04, z


def test_z_ratio_flatness_does_not_prove_pure_scale():
    """A bias-plus-scale mixture can also look flat, so flatness is one-sided evidence."""
    rng = np.random.default_rng(4)
    n, m = 60_000, 200
    bias = rng.choice([-0.75, 0.75], size=(n, 1))
    obs = rng.standard_normal((n, 1)) + bias
    ens = rng.standard_normal((n, 1, m)) * 0.75
    z = z_ratio_table(obs, ens)
    assert z["ratio_spread"] < 0.05, z  # indistinguishable from a scale error


# --- re-centering ---------------------------------------------------------------


def test_oracle_recentering_is_degenerate():
    """Guard the trap: per-window oracle re-centering returns 1.0 for any forecaster.

    Subtracting a window's own realized error centres the ensemble on the observation, so
    containment is guaranteed. This test exists so nobody reintroduces it as a diagnostic.
    """
    rng = np.random.default_rng(5)
    obs = rng.standard_normal((2000, 3))
    for ens in (
        rng.standard_normal((2000, 3, 80)) * 0.5,  # under-dispersed
        rng.standard_normal((2000, 3, 80)) + 3.0,  # badly biased
    ):
        assert coverage_indicator(obs, oracle_recenter(obs, ens), 0.80).mean() == 1.0


def test_systematic_bias_removal_leaves_an_unbiased_forecast_alone():
    rng = np.random.default_rng(5)
    obs = rng.standard_normal((4000, 3))
    ens = rng.standard_normal((4000, 3, 80)) * 0.5
    before = coverage_indicator(obs, ens, 0.80).mean()
    after = coverage_indicator(obs, remove_systematic_bias(obs, ens)[0], 0.80).mean()
    assert abs(after - before) < 0.02, (before, after)


def test_systematic_bias_removal_restores_a_systematically_biased_forecast():
    rng = np.random.default_rng(6)
    obs = rng.standard_normal((4000, 3)) + 2.0
    ens = rng.standard_normal((4000, 3, 80))
    before = coverage_indicator(obs, ens, 0.80).mean()
    corrected, bias = remove_systematic_bias(obs, ens)
    after = coverage_indicator(obs, corrected, 0.80).mean()
    assert before < 0.30
    assert after > 0.75, after
    np.testing.assert_allclose(bias, -2.0, atol=0.1)


def test_systematic_bias_removal_does_not_help_idiosyncratic_error():
    """The distinction that decides what fine-tuning can buy."""
    rng = np.random.default_rng(6)
    obs = rng.standard_normal((4000, 3)) + rng.choice([-2.0, 2.0], size=(4000, 1))
    ens = rng.standard_normal((4000, 3, 80))
    before = coverage_indicator(obs, ens, 0.80).mean()
    after = coverage_indicator(obs, remove_systematic_bias(obs, ens)[0], 0.80).mean()
    assert abs(after - before) < 0.02, (before, after)


def test_systematic_bias_removal_preserves_ensemble_spread():
    """It must move location only, never dispersion."""
    rng = np.random.default_rng(7)
    obs = rng.standard_normal((200, 3))
    ens = rng.standard_normal((200, 3, 40)) * 1.7
    corrected, _ = remove_systematic_bias(obs, ens)
    np.testing.assert_allclose(corrected.std(axis=-1), ens.std(axis=-1), rtol=1e-10)


def test_relative_mode_is_scale_free():
    """Price panels span orders of magnitude; an additive mean would follow the largest."""
    rng = np.random.default_rng(8)
    truth = np.array([100.0, 10_000.0])[:, None] * np.ones((2, 3))
    obs = truth
    ens = (truth * 1.05)[..., None] + rng.standard_normal((2, 3, 60)) * truth[..., None] * 0.01
    _, bias = remove_systematic_bias(obs, ens, relative=True)
    np.testing.assert_allclose(bias, 0.05, atol=0.01)


# --- tercile table --------------------------------------------------------------


def test_tercile_table_uses_supplied_cuts():
    """Cuts are passed in so two models can be compared on identical strata."""
    rng = np.random.default_rng(8)
    obs = rng.standard_normal((30, 2))
    ens = rng.standard_normal((30, 2, 40))
    ter = np.repeat([0, 1, 2], 10)
    blocks = np.repeat(np.arange(30), 2).reshape(30, 2)
    t = tercile_table(obs, ens, ter, blocks)
    assert {"calm", "mid", "volatile"} <= set(t)
    assert all(t[k]["n_windows"] == 10 for k in ("calm", "mid", "volatile"))
    assert "calm_minus_volatile_80" in t


# --- paired bootstrap -----------------------------------------------------------


def test_paired_bootstrap_is_tighter_than_treating_arms_independently():
    """Arms scored on the same windows share noise; pairing must exploit that."""
    rng = np.random.default_rng(9)
    n_blocks, per_block = 12, 40
    shared = rng.normal(0, 5, size=(n_blocks, 1))  # large common component
    a = shared + rng.normal(0.5, 0.2, size=(n_blocks, per_block))
    b = shared + rng.normal(0.0, 0.2, size=(n_blocks, per_block))
    blocks = np.repeat(np.arange(n_blocks), per_block).reshape(n_blocks, per_block)

    paired = paired_block_bootstrap_diff(a, b, blocks, n_boot=500, seed=0)
    width = paired["ci95"][1] - paired["ci95"][0]

    assert abs(paired["mean_difference"] - 0.5) < 0.15
    assert width < 1.0, "pairing should cancel the shared component"
    assert not paired["includes_zero"]


def test_paired_bootstrap_flags_a_null_difference():
    rng = np.random.default_rng(10)
    n_blocks, per_block = 12, 40
    a = rng.normal(0, 1, size=(n_blocks, per_block))
    b = rng.normal(0, 1, size=(n_blocks, per_block))
    blocks = np.repeat(np.arange(n_blocks), per_block).reshape(n_blocks, per_block)
    assert paired_block_bootstrap_diff(a, b, blocks, n_boot=500, seed=0)["includes_zero"]


def test_paired_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="shapes must align"):
        paired_block_bootstrap_diff(np.zeros(10), np.zeros(9), np.zeros(10))
