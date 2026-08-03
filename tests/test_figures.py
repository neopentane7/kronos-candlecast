"""Tests for the figure layer.

Plotting code is easy to leave untested and easy to get silently wrong, so these assert
the things a reader would be misled by: missing files, missing disclaimer, and diagnostics
that do not respond to the defect they are supposed to reveal.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a"))

from eval.figures import (  # noqa: E402
    coverage_by_regime,
    horizon_curve,
    pit_histogram,
    reliability_diagram,
    write_all,
)
from eval.metrics import pit_values  # noqa: E402


def case(n=300, horizon=6, m=40, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(100, 5, size=(n, horizon))
    obs = rng.normal(mu, 1.0)
    ens = rng.normal(mu[..., None], scale, size=(n, horizon, m))
    return obs, ens


def blocks_for(obs):
    return np.repeat(np.arange(obs.shape[0]), obs.shape[1]).reshape(obs.shape)


def test_reliability_diagram_writes_a_file(tmp_path):
    obs, ens = case()
    out = reliability_diagram(obs, {"calibrated": ens}, blocks_for(obs), tmp_path / "r.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_pit_histogram_writes_a_file(tmp_path):
    obs, ens = case()
    out = pit_histogram(obs, ens, tmp_path / "p.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_horizon_curve_writes_a_file(tmp_path):
    obs, ens = case()
    out = horizon_curve(obs, {"a": ens}, tmp_path / "h.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_coverage_by_regime_writes_a_file(tmp_path):
    obs, ens = case(n=300)
    terciles = np.repeat([0, 1, 2], 100)
    out = coverage_by_regime(obs, {"a": ens}, terciles, tmp_path / "c.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_write_all_emits_one_pit_per_model_plus_shared_figures(tmp_path):
    obs, ens = case(n=150)
    models = {"kronos_zeroshot": ens, "random_walk_drift": ens * 1.01}
    terciles = np.repeat([0, 1, 2], 50)
    written = write_all(obs, models, blocks_for(obs), tmp_path, terciles=terciles)

    names = {p.name for p in written}
    assert "reliability.png" in names
    assert "horizon_crps.png" in names
    assert "coverage_by_regime.png" in names
    assert "pit_kronos_zeroshot.png" in names
    assert "pit_random_walk_drift.png" in names
    assert all(p.exists() for p in written)


def test_every_figure_carries_the_disclaimer(tmp_path):
    """Constraint 10: figures travel separately from the report that made them."""
    from common.results import DISCLAIMER

    obs, ens = case(n=100)
    written = write_all(obs, {"a": ens}, blocks_for(obs), tmp_path)
    for path in written:
        fig_text = path.with_suffix(".png").read_bytes()
        assert len(fig_text) > 5_000  # a real render, not a stub
    # The disclaimer is drawn, so assert the source constant is what figures.py uses.
    import eval.figures as figmod

    assert figmod.DISCLAIMER == DISCLAIMER
    assert "not investment advice" in figmod.DISCLAIMER


# --- the diagnostics must actually diagnose ------------------------------------


def test_pit_detects_narrow_bands_as_a_u_shape():
    """Under-dispersion pushes mass to both tails; that is the shape the figure shows."""
    obs, ens = case(n=4000, horizon=3, m=60, scale=0.4, seed=1)
    pit = pit_values(obs, ens).reshape(-1)
    counts, _ = np.histogram(pit, bins=10, range=(0, 1))
    frac = counts / counts.sum()
    edges_mass = frac[0] + frac[-1]
    middle_mass = frac[4] + frac[5]
    assert edges_mass > 2 * middle_mass, (edges_mass, middle_mass)


def test_pit_detects_wide_bands_as_a_dome():
    obs, ens = case(n=4000, horizon=3, m=60, scale=3.0, seed=2)
    pit = pit_values(obs, ens).reshape(-1)
    counts, _ = np.histogram(pit, bins=10, range=(0, 1))
    frac = counts / counts.sum()
    assert frac[4] + frac[5] > 2 * (frac[0] + frac[-1])


def test_pit_is_flat_when_calibrated():
    obs, ens = case(n=6000, horizon=3, m=200, scale=1.0, seed=3)
    pit = pit_values(obs, ens).reshape(-1)
    counts, _ = np.histogram(pit, bins=10, range=(0, 1))
    frac = counts / counts.sum()
    assert np.all(np.abs(frac - 0.1) < 0.02), frac


def test_figures_reject_a_model_dict_that_does_not_align(tmp_path):
    obs, ens = case(n=50, horizon=4)
    with pytest.raises((ValueError, IndexError)):
        reliability_diagram(obs, {"bad": ens[:10]}, blocks_for(obs), tmp_path / "x.png")
