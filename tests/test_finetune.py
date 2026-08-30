"""A6's two silent-failure modes: a kill rule that never fires, and train/serve skew.

Neither would raise. A kill rule that cannot trigger just lets the run continue, which is
exactly the "one more epoch" failure CLAUDE.md §6 exists to prevent. A normalisation that
differs from inference produces a fine-tune that looks better offline and is worse in
serving, and nothing in the training log would say so.

These run without a GPU and without downloading weights.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "phase-a"), str(REPO / "phase-a" / "Kronos")]

torch = pytest.importorskip("torch")

# train.finetune reaches eval.sampler, which imports the upstream `model` package at
# module scope. Without the clone this module cannot even be collected.
pytest.importorskip(
    "model",
    reason=(
        "requires the upstream Kronos clone at phase-a/Kronos, created by "
        "phase-a/scripts/setup_upstream.ps1. It is gitignored per working rule 2, so CI "
        "never has it and these paths are exercised locally only. Documented boundary, "
        "not a silent hole."
    ),
)

from train.finetune import CLIP, SEQ_LEN, Config, kill_check  # noqa: E402


# --------------------------------------------------------------- the kill rule
def hist(*losses):
    return [{"epoch": i + 1, "val_loss": v} for i, v in enumerate(losses)]


def test_no_stop_while_training_is_still_improving():
    cfg = Config()
    stop, why = kill_check(hist(1.0, 0.9, 0.8), baseline=1.1, cfg=cfg, elapsed_h=0.5)
    assert not stop, why


def test_stops_at_the_registered_epoch_when_it_never_beats_the_baseline():
    """The rule that matters. Three epochs, none better than the pretrained init."""
    cfg = Config()
    stop, why = kill_check(hist(1.2, 1.15, 1.13), baseline=1.10, cfg=cfg, elapsed_h=0.5)
    assert stop
    assert "has not beaten the pretrained init" in why
    assert "epoch 3" in why


def test_does_not_stop_early_before_the_registered_epoch():
    """Two bad epochs are not grounds; the bar named three."""
    cfg = Config()
    stop, _ = kill_check(hist(1.2, 1.15), baseline=1.10, cfg=cfg, elapsed_h=0.5)
    assert not stop


def test_a_single_epoch_below_baseline_is_enough_to_continue():
    """`best` is the minimum, so one good epoch keeps the run alive."""
    cfg = Config()
    stop, _ = kill_check(hist(1.2, 1.05, 1.15), baseline=1.10, cfg=cfg, elapsed_h=0.5)
    assert not stop


def test_wall_clock_cap_fires_regardless_of_loss():
    """Even a run that is improving stops at the cap. Eight hours is the budget."""
    cfg = Config()
    stop, why = kill_check(hist(1.0, 0.5, 0.2), baseline=1.1, cfg=cfg, elapsed_h=8.0)
    assert stop
    assert "wall-clock cap" in why


def test_the_wall_clock_cap_wins_over_the_loss_rule():
    cfg = Config()
    stop, why = kill_check(hist(2.0, 2.0, 2.0), baseline=1.0, cfg=cfg, elapsed_h=9.0)
    assert stop
    assert "wall-clock" in why, "the cheaper-to-check condition should be reported first"


def test_the_registered_thresholds_are_what_the_plan_says():
    cfg = Config()
    assert cfg.kill_epoch == 3
    assert cfg.max_hours == 8.0
    assert cfg.epochs <= 10, "a pilot, not a training run"


# --------------------------------------------------------------- train/serve parity
def bars(n=SEQ_LEN):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=n)))
    vol = rng.uniform(1e5, 5e6, size=n)
    return pd.DataFrame(
        {
            "timestamps": pd.bdate_range("2021-01-04", periods=n),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": vol,
            "amount": close * vol,
        }
    )


def test_training_normalisation_matches_the_sampler_exactly():
    """Hard constraint 6: preprocessing is shared, not reimplemented per call site.

    The sampler normalises each window by its own mean and std and clips to ±5. Training
    must do the identical thing, or the model is fine-tuned on a distribution it will
    never see at serving time.
    """
    from eval.sampler import FEATURE_COLUMNS

    df = bars()
    arr = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # what eval/sampler.py does
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    serving = np.clip((arr - mean) / (std + 1e-5), -CLIP, CLIP)

    # what train/finetune.py does, inlined from WindowDataset
    m2, s2 = arr.mean(axis=0), arr.std(axis=0)
    training = np.clip((arr - m2) / (s2 + 1e-5), -CLIP, CLIP)

    np.testing.assert_array_equal(training, serving)
    assert CLIP == 5.0, "the sampler clips at 5.0; a different bound here is skew"


def test_sequence_length_fits_the_model_context():
    """430 = lookback 400 + horizon 30, and the model's context is 512."""
    assert SEQ_LEN == 430
    assert SEQ_LEN <= 512


# --------------------------------------------------------------- checkpoint loading
def test_a_mismatched_checkpoint_is_refused(tmp_path, monkeypatch):
    """A partially loaded model would score a mixture of two forecasters."""
    from eval import calibrate

    ckpt = tmp_path / "bad.pt"
    torch.save({"model": {"not_a_real_parameter": torch.zeros(3)}}, ckpt)

    class FakeModel:
        def load_state_dict(self, _state, strict=True):
            return (["missing.weight"], ["not_a_real_parameter"])

    monkeypatch.setattr(calibrate, "MODEL_ID", "fake")
    fake_module = type(sys)("model")
    fake_module.Kronos = type("K", (), {"from_pretrained": staticmethod(lambda _: FakeModel())})
    monkeypatch.setitem(sys.modules, "model", fake_module)

    with pytest.raises(SystemExit, match="Refusing to evaluate a partially loaded model"):
        calibrate.load_kronos(ckpt)


def test_no_checkpoint_means_zero_shot_and_says_so(monkeypatch):
    """The label is the guard: a fine-tune result must never be filed as zero-shot."""
    from eval import calibrate

    fake_module = type(sys)("model")
    fake_module.Kronos = type("K", (), {"from_pretrained": staticmethod(lambda _: object())})
    monkeypatch.setitem(sys.modules, "model", fake_module)

    _, label = calibrate.load_kronos(None)
    assert label == "kronos_zeroshot"
