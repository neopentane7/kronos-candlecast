"""ACI state is data the nightly job commits, so it must round-trip and it must converge.

The convergence test is the one that matters: ACI is being used because §17d rules out
split conformal at these horizons, so if the adaptation does not actually track the target
under a misspecified forecaster, the served 80% and 90% bands mean nothing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.aci import ALPHA_MAX, ALPHA_MIN, DEFAULT_GAMMA, ACIState  # noqa: E402


def test_defaults_to_the_nominal_miss_rate_before_any_observation():
    s = ACIState()
    assert s.alpha("RELIANCE.NS", 0.80, 1) == pytest.approx(0.20)
    assert s.effective_level("RELIANCE.NS", 0.80, 1) == pytest.approx(0.80)


def test_a_miss_widens_and_a_hit_narrows():
    s = ACIState(gamma=0.01)
    start = s.alpha("X", 0.80, 5)
    after_miss = s.update("X", 0.80, 5, covered=False)
    assert after_miss < start, "a miss must lower alpha, which widens the next band"
    after_hit = s.update("X", 0.80, 5, covered=True)
    assert after_hit > after_miss


def test_state_is_per_ticker_level_and_step():
    """The design point: a 30-step forecast is not one prediction problem (F2)."""
    s = ACIState(gamma=0.05)
    s.update("A", 0.80, 30, covered=False)
    assert s.alpha("A", 0.80, 30) < 0.20  # moved
    assert s.alpha("A", 0.80, 1) == pytest.approx(0.20)  # other step untouched
    assert s.alpha("A", 0.90, 30) == pytest.approx(0.10)  # other level untouched
    assert s.alpha("B", 0.80, 30) == pytest.approx(0.20)  # other ticker untouched


@pytest.mark.parametrize("scale", [0.5, 0.8, 1.2])
def test_long_run_coverage_converges_to_target_under_misspecification(scale):
    """The property ACI is here for, tested against the actual miscalibration mechanism.

    Report §6 found the zero-shot model shape-correct and scale-wrong: its intervals are a
    roughly constant factor too tight at every level. So the forecaster here is Gaussian
    with the right centre and its sigma off by ``scale`` -- under-dispersed below 1,
    over-dispersed above. A band asked for at effective level ``eff`` then really covers

        2 * Phi(z(eff) * scale) - 1

    ACI never sees ``scale``; it only sees hits and misses, and must find the level that
    makes realized coverage 80% from either direction.
    """
    from math import erf, sqrt
    from statistics import NormalDist

    nd = NormalDist()
    rng = np.random.default_rng(0)
    s = ACIState(gamma=0.02)
    target = 0.80
    hits = []
    for _ in range(4000):
        eff = s.effective_level("T", target, 10)
        z = nd.inv_cdf(0.5 + eff / 2.0)
        p = erf(z * scale / sqrt(2.0))  # = 2 * Phi(z*scale) - 1
        covered = bool(rng.random() < p)
        hits.append(covered)
        s.update("T", target, 10, covered)

    realized = float(np.mean(hits[-1500:]))
    assert abs(realized - target) < 0.05, f"realized {realized:.3f} vs target {target}"


def test_convergence_is_not_vacuous_without_adaptation():
    """The control: freeze alpha and the same forecaster misses the target badly.

    Without this, the test above could pass because the simulated forecaster happened to
    sit near 80% already, and would tell us nothing about the adaptation.
    """
    from math import erf, sqrt
    from statistics import NormalDist

    nd = NormalDist()
    rng = np.random.default_rng(0)
    z = nd.inv_cdf(0.5 + 0.80 / 2.0)
    p = erf(z * 0.5 / sqrt(2.0))  # scale=0.5, alpha never adapts
    realized = float(np.mean(rng.random(4000) < p))
    assert abs(realized - 0.80) > 0.20, f"static band already covers {realized:.3f}"


def test_alpha_is_clamped_and_the_clamp_is_counted():
    """An unbounded update could drive alpha negative on a long adverse run."""
    s = ACIState(gamma=0.5)
    for _ in range(50):
        s.update("X", 0.80, 1, covered=False)
    assert s.alpha("X", 0.80, 1) == pytest.approx(ALPHA_MIN)
    assert s.clamped > 0, "hitting the clamp must be recorded, not silently absorbed"

    s2 = ACIState(gamma=0.5)
    for _ in range(50):
        s2.update("Y", 0.80, 1, covered=True)
    assert s2.alpha("Y", 0.80, 1) == pytest.approx(ALPHA_MAX)


def test_state_round_trips_through_disk(tmp_path):
    """It is committed by the job, so a lossy save silently resets calibration."""
    s = ACIState(gamma=0.007, provisional=False)
    for step in (1, 15, 30):
        s.update("RELIANCE.NS", 0.80, step, covered=False)
        s.update("TCS.NS", 0.90, step, covered=True)

    p = s.save(tmp_path / "aci_state.json")
    back = ACIState.load(p)

    assert back.gamma == pytest.approx(0.007)
    assert back.provisional is False
    assert back.updates == s.updates
    assert back.alphas == s.alphas
    for step in (1, 15, 30):
        assert back.alpha("RELIANCE.NS", 0.80, step) == pytest.approx(
            s.alpha("RELIANCE.NS", 0.80, step)
        )


def test_missing_state_file_starts_clean_rather_than_crashing(tmp_path):
    s = ACIState.load(tmp_path / "does-not-exist.json")
    assert s.gamma == pytest.approx(DEFAULT_GAMMA)
    assert s.alphas == {}
    assert s.provisional is True, (
        "a state that has adapted to nothing has had nothing measured about it; only a "
        "state checked against realized outcomes may call its gamma measured"
    )


def test_a_verified_state_keeps_its_label_across_a_round_trip(tmp_path):
    """The label lives in the file, not the constant, so loading must not re-provisionalise.

    The deployed state was checked on 2026-08-30 and records provisional=False. If load()
    fell back to the module default it would silently relabel a measured gamma as untested
    on every nightly run.
    """
    path = tmp_path / "state.json"
    s = ACIState()
    s.provisional = False
    s.save(path)

    assert ACIState.load(path).provisional is False


def test_saved_state_is_sorted_so_diffs_are_readable(tmp_path):
    """The job commits this file daily; an unstable key order makes every diff noise."""
    s = ACIState()
    for t in ("ZZZ", "AAA", "MMM"):
        s.update(t, 0.80, 1, covered=True)
    d = json.loads(s.save(tmp_path / "s.json").read_text())
    assert list(d["alphas"]) == sorted(d["alphas"])
