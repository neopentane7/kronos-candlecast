"""The G4 bar has to be able to fail, and the first version of it could not.

Scored on the zero-shot ensemble itself -- the model the gate exists to reject -- condition
2 originally returned h=30 of 0.481 against a 0.481 threshold and flatness of 0.530 against
0.529, and passed. That is working rule 6 exactly: a check that cannot fail is not evidence.
The word "materially" in the bar is what carries the weight, and it now means the 95%
block-bootstrap interval must lie entirely above the zero-shot value.

The first test here is the known-answer case that caught it, and it is the reason the file
exists.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "phase-a")]

from eval.analysis import spread_ratio_by_horizon  # noqa: E402
from eval.g4_verdict import (  # noqa: E402
    FINETUNED,
    NULL,
    ZEROSHOT_FLATNESS,
    ZEROSHOT_H30,
    condition_1,
    condition_2,
    period_drift_readout,
)

N, H, M, BLOCKS = 240, 30, 30, 12


def grid(spread_at_h, seed=0, n=N):
    """A grid whose predicted spread at step h is `spread_at_h(h)` times the realized one.

    Outcomes are a random walk with unit daily vol, so realized spread at step h is sqrt(h).
    Building the ensemble to a chosen multiple of that makes the dispersion ratio directly
    controllable, which is what condition 2 reads.
    """
    rng = np.random.default_rng(seed)
    last = np.full(n, 100.0)
    steps = np.arange(1, H + 1)

    obs = last[:, None] + rng.normal(0, 1, size=(n, H)).cumsum(axis=1)
    width = np.array([spread_at_h(h) for h in steps]) * np.sqrt(steps)
    ens = last[:, None, None] + rng.normal(0, 1, size=(n, H, M)) * width[None, :, None]

    blocks = np.repeat(np.arange(n) % BLOCKS, H).reshape(n, H)
    history = np.tile(last[:, None], (1, 400))
    return {
        "obs": obs,
        "history": history,
        "blocks": blocks,
        "models": {FINETUNED: ens, NULL: ens.copy()},
        "dir": Path("synthetic"),
    }


# ------------------------------------------------------- the known-answer case
def test_a_finetune_that_changed_nothing_fails_condition_2():
    """The bug this file was written for.

    A model reproducing zero-shot's dispersion curve exactly must not clear a bar that
    zero-shot failed. Before "materially" was operationalised, it did.
    """
    # Reproduce the published curve: 0.909 at h=1 decaying to 0.481 at h=30.
    d = grid(lambda h: 0.909 + (ZEROSHOT_H30 - 0.909) * (h - 1) / (H - 1))

    c2 = condition_2(d, seed=0)
    assert not c2["passes"], c2
    assert not c2["rose_at_h30"], "h=30 sitting on the reference is not materially above it"


def test_the_materiality_rule_is_an_interval_not_a_comparison():
    """A point estimate a hair above the reference must not pass on its own.

    The multiplier is calibrated against a measured grid rather than assumed: an
    M-member ensemble's spread is not exactly the multiplier it was built from, and a
    test that guessed would be asserting its own arithmetic instead of the rule.
    """
    probe = grid(lambda h: 1.0)
    measured = spread_ratio_by_horizon(probe["obs"], probe["models"][FINETUNED], probe["history"])
    k = (ZEROSHOT_H30 * 1.03) / measured["ratio_hmax"]

    d = grid(lambda h: k)
    c2 = condition_2(d, seed=0)

    assert c2["ratio_h30"] > ZEROSHOT_H30, "the point estimate clears the reference"
    assert c2["h30_ci95"][0] < ZEROSHOT_H30, "but the interval does not"
    assert not c2["rose_at_h30"], "so 'materially above' is not satisfied"
    assert not c2["passes"]


# ------------------------------------------------------- the conditions can pass
def test_a_correctly_dispersed_forecaster_clears_condition_2():
    """A flat curve at ratio 1.0 is what fixing the defect would look like."""
    d = grid(lambda h: 1.0)
    c2 = condition_2(d, seed=0)
    assert c2["rose_at_h30"] and c2["curve_flattened"], c2
    assert c2["passes"]
    assert c2["flatness_h30_over_h1"] == pytest.approx(1.0, abs=0.15)


def test_both_legs_are_required_so_a_uniform_lift_is_not_enough():
    """F9's failure mode: every arm raised h=30 while h30/h1 fell. That must not pass.

    Scaling the whole curve by a constant lifts h=30 without changing h30/h1, so a model
    that merely widens everywhere clears the first leg and fails the second.
    """
    d = grid(lambda h: 2.2 * (0.909 + (ZEROSHOT_H30 - 0.909) * (h - 1) / (H - 1)))
    c2 = condition_2(d, seed=0)
    assert c2["rose_at_h30"], "a uniform scale-up does raise h=30"
    assert not c2["curve_flattened"], "but the shape is unchanged"
    assert not c2["passes"], "so the condition must fail"


def test_an_identical_forecaster_reaches_parity_on_condition_1():
    d = grid(lambda h: 1.0)
    c1 = condition_1(d, seed=0)
    assert c1["mean_difference"] == pytest.approx(0.0, abs=1e-9)
    assert c1["passes"]


def test_a_worse_forecaster_fails_condition_1():
    d = grid(lambda h: 1.0)
    rng = np.random.default_rng(5)
    d["models"][FINETUNED] = d["models"][NULL] + rng.normal(40, 1, size=d["models"][NULL].shape)
    c1 = condition_1(d, seed=0)
    assert not c1["passes"]
    assert c1["ci95"][0] > 0, "the whole interval sits above zero"


# ------------------------------------------------------- it refuses the wrong inputs
def test_scoring_zero_shot_weights_by_mistake_is_refused():
    """calibrate.py renames the row on --checkpoint; absence of it means the wrong run."""
    d = grid(lambda h: 1.0)
    del d["models"][FINETUNED]
    with pytest.raises(SystemExit, match="Score the checkpoint"):
        condition_1(d, seed=0)


def test_the_null_must_be_present_because_the_bar_is_defined_against_it():
    d = grid(lambda h: 1.0)
    del d["models"][NULL]
    with pytest.raises(SystemExit, match="the bar is defined against it"):
        condition_1(d, seed=0)


# ------------------------------------------------------- the readout stays a readout
def test_the_period_drift_readout_is_absent_without_a_val_run():
    assert period_drift_readout(grid(lambda h: 1.0), None)["available"] is False


def test_the_period_drift_readout_is_marked_as_not_a_bar():
    """It must never be mistaken for a third pass condition; the bar was frozen 2026-08-09."""
    r = period_drift_readout(grid(lambda h: 1.0, seed=1), grid(lambda h: 1.0, seed=2))
    assert r["not_a_bar"] is True
    assert "passes" not in r
    assert set(r[FINETUNED]) == {"cov80_val", "cov80_test", "drift", "f11_reference"}


def test_the_readout_compares_against_the_recorded_f11_drift():
    val = grid(lambda h: 1.0, seed=3)
    test = grid(lambda h: 1.0, seed=4)
    r = period_drift_readout(test, val)
    assert r["zeroshot_drift"] == pytest.approx(0.1715)
    assert isinstance(r["shrank_vs_zeroshot"], bool)


def test_the_reference_constants_match_the_published_facts():
    """F2's curve and the flatness derived from it. If these drift, the bar has moved."""
    assert ZEROSHOT_H30 == 0.481
    assert pytest.approx(0.481 / 0.909, abs=1e-6) == ZEROSHOT_FLATNESS
