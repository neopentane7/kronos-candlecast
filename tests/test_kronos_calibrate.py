"""kronos-calibrate is the published surface, so its contract is what needs pinning.

The metric implementations already have their own tests. What is untested until here is
everything the extraction added: the file validation, the feasibility arithmetic, and the
claim the README leads with. A reader who trusts the README is trusting these.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "phase-a"), str(REPO / "kronos-calibrate")]

import kronos_calibrate as kc  # noqa: E402
import make_example  # noqa: E402


# --------------------------------------------------------------- fixtures
def write_npz(tmp_path, **arrays) -> Path:
    p = tmp_path / "e.npz"
    np.savez_compressed(p, **arrays)
    return p


def minimal(n=24, h=6, m=30, blocks=4, seed=0):
    """A small well-formed file: n windows spread evenly over `blocks` dates."""
    rng = np.random.default_rng(seed)
    obs = 100 + rng.normal(0, 3, size=(n, h))
    ens = obs[:, :, None] + rng.normal(0, 3, size=(n, h, m))
    block_ids = np.repeat(np.arange(n) % blocks, h).reshape(n, h)
    return {"y_close": obs, "block_ids": block_ids, "ens__a": ens}


# --------------------------------------------------------------- validation
def test_a_file_without_ensembles_is_refused(tmp_path):
    d = minimal()
    d.pop("ens__a")
    with pytest.raises(ValueError, match="no ens__"):
        kc.Ensembles(write_npz(tmp_path, **d))


def test_a_file_without_block_ids_is_refused(tmp_path):
    """Without blocks there is no honest denominator, so there is no report."""
    d = minimal()
    d.pop("block_ids")
    with pytest.raises(ValueError, match="missing required arrays"):
        kc.Ensembles(write_npz(tmp_path, **d))


def test_an_ensemble_of_the_wrong_shape_is_refused(tmp_path):
    """The failure this catches is silent: mismatched arrays still produce numbers."""
    d = minimal()
    d["ens__a"] = d["ens__a"][:, :-1, :]
    with pytest.raises(ValueError, match=r"ens__a must be"):
        kc.Ensembles(write_npz(tmp_path, **d))


def test_block_ids_of_the_wrong_shape_are_refused(tmp_path):
    d = minimal()
    d["block_ids"] = d["block_ids"][:, 0]
    with pytest.raises(ValueError, match="block_ids must be"):
        kc.Ensembles(write_npz(tmp_path, **d))


def test_the_ens_prefix_is_stripped_to_name_the_arm(tmp_path):
    d = minimal()
    d["ens__my_model"] = d.pop("ens__a")
    ens = kc.Ensembles(write_npz(tmp_path, **d))
    assert set(ens.models) == {"my_model"}


def test_blocks_are_counted_as_dates_not_windows(tmp_path):
    """The 59x denominator error the block bootstrap exists to prevent."""
    ens = kc.Ensembles(write_npz(tmp_path, **minimal(n=24, blocks=4)))
    assert ens.n_windows == 24
    assert ens.blocks == 4


# --------------------------------------------------------------- feasibility
def test_feasibility_reproduces_the_registered_floors():
    """Fact of Record F4: at a 30-step horizon, 80% needs 18 dates and 90% needs 38."""
    f = kc.feasibility(blocks=12)["levels"]
    assert f["50"]["blocks_needed"] == 6
    assert f["80"]["blocks_needed"] == 18
    assert f["90"]["blocks_needed"] == 38


def test_twelve_blocks_permit_only_the_fifty_percent_band():
    """The Phase A corpus. 50% is the only level it can certify."""
    f = kc.feasibility(blocks=12)["levels"]
    assert f["50"]["feasible"]
    assert not f["80"]["feasible"]
    assert not f["90"]["feasible"]


def test_feasibility_turns_on_the_block_count_alone():
    assert kc.feasibility(blocks=18)["levels"]["80"]["feasible"]
    assert not kc.feasibility(blocks=17)["levels"]["80"]["feasible"]


# --------------------------------------------------------------- the README's claim
def test_the_estimator_artifact_is_real_and_of_the_advertised_size():
    """The README leads on this. If it stops being true, the README is wrong.

    A calibrated ensemble, the same outcomes, the same nominal level -- only the quantile
    estimator changes. NumPy's default reads several points lower, which is the artifact
    that would have been read as a model defect.
    """
    rng = np.random.default_rng(11)
    n, h, m = 4000, 1, 30
    obs = rng.normal(0, 1, size=(n, h))
    ens = rng.normal(0, 1, size=(n, h, m))

    def cov(method):
        lo = np.quantile(ens, 0.10, axis=-1, method=method)
        hi = np.quantile(ens, 0.90, axis=-1, method=method)
        return float(((obs >= lo) & (obs <= hi)).mean())

    weibull, linear = cov("weibull"), cov("linear")
    assert abs(weibull - 0.80) < 0.02, f"the corrected estimator should land on nominal: {weibull}"
    assert 0.80 - linear > 0.03, f"the default estimator should under-report: {linear}"
    assert kc.QUANTILE_METHOD == "weibull", "the harness must ship the corrected estimator"


def test_the_fair_crps_correction_is_applied_and_smaller_than_naive():
    """Ferro's estimator removes an inflation of order 1/(m-1); it never adds one."""
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, size=(200, 4))
    ens = rng.normal(0, 1, size=(200, 4, 30))
    assert kc.CRPS_ESTIMATOR == "fair"
    assert kc.crps(obs, ens).mean() < kc.crps(obs, ens, estimator="qd").mean()


# --------------------------------------------------------------- the worked example
@pytest.fixture(scope="module")
def example(tmp_path_factory):
    p = tmp_path_factory.mktemp("kc") / "example_ensembles.npz"
    np.savez_compressed(p, **make_example.build(seed=7))
    return kc.Ensembles(p)


def test_the_example_matches_the_documented_file_format(example):
    """The README's table is the integration contract; the example must satisfy it."""
    assert example.horizon == make_example.HORIZON
    assert example.m == make_example.M
    assert example.blocks == make_example.N_DATES
    assert example.n_windows == make_example.N_TICKERS * make_example.N_DATES
    assert set(example.models) == {"oracle", "too_tight", "flat"}
    assert example.history is not None and example.terciles is not None


def test_every_block_id_is_constant_across_the_horizon(example):
    """A row is one forecast made on one date. Varying ids inside a row is the bug."""
    assert (example.block_ids == example.block_ids[:, :1]).all()


def test_the_oracle_covers_at_nominal(example):
    """Calibrated by construction, so this is a test of the instrument, not the arm."""
    rep = kc.report(example, baseline="oracle")
    c = rep["models"]["oracle"]["coverage"]["80"]
    assert c["covers_nominal"], c


def test_the_under_dispersed_arm_is_caught_and_the_flat_one_is_caught_harder(example):
    rep = kc.report(example)
    cov = {k: v["coverage"]["80"]["empirical"] for k, v in rep["models"].items()}
    assert cov["too_tight"] < 0.75
    assert cov["flat"] == 0.0
    assert cov["oracle"] > cov["too_tight"] > cov["flat"]


def test_under_propagation_shows_as_a_falling_spread_ratio_not_a_low_one(example):
    """The signature is the slope. A uniformly narrow cone is a different defect."""
    rep = kc.report(example)
    tight = rep["models"]["too_tight"]["spread_ratio"]
    oracle = rep["models"]["oracle"]["spread_ratio"]
    assert tight["ratio_hmax"] < tight["ratio_h1"] * 0.75, tight
    assert oracle["ratio_hmax"] > oracle["ratio_h1"] * 0.9, oracle


def test_the_paired_comparison_is_only_computed_against_a_named_baseline(example):
    assert "paired_vs_baseline" not in kc.report(example)
    rep = kc.report(example, baseline="oracle")
    assert set(rep["paired_vs_baseline"]["diffs"]) == {"too_tight", "flat"}
    assert rep["paired_vs_baseline"]["diffs"]["flat"]["mean_difference"] > 0


def test_an_unknown_baseline_fails_loudly_rather_than_being_ignored(example):
    """Silently skipping the comparison would leave a report that looks complete."""
    with pytest.raises(ValueError, match="not in the file"):
        kc.report(example, baseline="typo_model")


# --------------------------------------------------------------- the CLI
def test_the_cli_runs_end_to_end_and_writes_its_report(tmp_path):
    npz = write_npz(tmp_path, **minimal())
    out = tmp_path / "report.json"
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "kronos-calibrate" / "kronos_calibrate.py"),
            str(npz),
            "--baseline",
            "a",
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    assert "independent forecast dates" in r.stdout
    assert "not investment advice" in r.stdout, "hard constraint 10 / working rule 12"
