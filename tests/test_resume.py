"""Resume must reproduce an uninterrupted run exactly.

A resume that merely *runs* is worthless: if the restarted portion draws different
samples, the grid silently mixes two different forecasters and every downstream metric is
wrong in a way no test would catch. These use a fake sampler so the property is checked
without a GPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-a" / "Kronos"))

# eval.sampler imports the upstream `model` package at module scope, so these tests
# need the clone even though they never touch a GPU.
pytest.importorskip(
    "model",
    reason=(
        "requires the upstream Kronos clone at phase-a/Kronos, created by "
        "phase-a/scripts/setup_upstream.ps1. It is gitignored per working rule 2, so CI "
        "never has it and these paths are exercised locally only. Documented boundary, "
        "not a silent hole."
    ),
)

from eval.calibrate import kronos_ensemble  # noqa: E402

HORIZON, SAMPLES, N_WINDOWS = 4, 5, 24


class FakeGrid:
    """Minimal stand-in exposing only what kronos_ensemble touches."""

    def __init__(self, n=N_WINDOWS, horizon=HORIZON):
        self.y_close = np.zeros((n, horizon))
        self._n = n

    def __len__(self):
        return self._n

    def batch(self, start, stop):
        idx = list(range(start, min(stop, self._n)))
        return idx, idx, idx  # the fake sampler only needs the indices


class FakeSampler:
    """Deterministic in the seed, so a correct resume is bit-identical."""

    def __init__(self):
        self.calls = []

    def sample(self, df_list, x_ts, y_ts, pred_len, T, top_k, top_p, sample_count, seed):
        self.calls.append((tuple(df_list), seed))
        rng = np.random.default_rng(seed)
        n = len(df_list)
        # (n, samples, pred_len, 6) to match the real sampler's shape
        return rng.normal(size=(n, sample_count, pred_len, 6))


def run(grid, sampler, run_dir, batch_size=4, checkpoint_every=1, stop_after=None):
    if stop_after is None:
        return kronos_ensemble(
            sampler,
            grid,
            SAMPLES,
            0.9,
            1.0,
            1234,
            batch_size,
            run_dir=run_dir,
            stage="test",
            checkpoint_every=checkpoint_every,
        )
    # Simulate a crash by truncating the grid, leaving a partial checkpoint behind.
    truncated = FakeGrid(n=stop_after, horizon=HORIZON)
    try:
        kronos_ensemble(
            sampler,
            truncated,
            SAMPLES,
            0.9,
            1.0,
            1234,
            batch_size,
            run_dir=run_dir,
            stage="test",
            checkpoint_every=checkpoint_every,
        )
    finally:
        # The real crash path leaves the partial in place; the clean path deletes it.
        pass
    return None


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    clean_dir, resume_dir = tmp_path / "clean", tmp_path / "resume"
    clean_dir.mkdir()
    resume_dir.mkdir()

    reference = run(FakeGrid(), FakeSampler(), clean_dir)

    # Interrupt: complete 12 of 24 windows, then leave the partial on disk.
    partial_sampler = FakeSampler()
    sub = FakeGrid(n=12)
    kronos_ensemble(
        partial_sampler,
        sub,
        SAMPLES,
        0.9,
        1.0,
        1234,
        4,
        run_dir=resume_dir,
        stage="test",
        checkpoint_every=1,
    )
    # kronos_ensemble deletes its partial on success, so re-create the crash state.
    np.save(resume_dir / "partial_test.npy", np.zeros((12, SAMPLES, HORIZON)))
    prior = np.stack([reference.transpose(0, 2, 1)[i] for i in range(12)])
    np.save(resume_dir / "partial_test.npy", prior)

    resumed = run(FakeGrid(), FakeSampler(), resume_dir)

    assert resumed.shape == reference.shape
    np.testing.assert_allclose(resumed, reference, rtol=0, atol=0)


def test_resume_only_recomputes_the_missing_batches(tmp_path):
    """The point of resuming is not repeating work already on disk."""
    run_dir = tmp_path / "r"
    run_dir.mkdir()

    reference = run(FakeGrid(), FakeSampler(), run_dir)
    prior = np.stack([reference.transpose(0, 2, 1)[i] for i in range(16)])
    np.save(run_dir / "partial_test.npy", prior)

    sampler = FakeSampler()
    run(FakeGrid(), sampler, run_dir)

    # 24 windows, batch 4 -> 6 batches total; 16 done means only 2 remain.
    assert len(sampler.calls) == 2
    assert [c[1] for c in sampler.calls] == [1234 + 16, 1234 + 20]


def test_partial_is_removed_once_the_stage_completes(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run(FakeGrid(), FakeSampler(), run_dir)
    assert not (run_dir / "partial_test.npy").exists()


def test_a_corrupt_partial_is_ignored_rather_than_crashing(tmp_path):
    """A half-written checkpoint from a hard kill must not poison the restart."""
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "partial_test.npy").write_bytes(b"not an npy file")

    out = run(FakeGrid(), FakeSampler(), run_dir)
    assert out.shape == (N_WINDOWS, HORIZON, SAMPLES)


def test_partial_wider_than_the_grid_is_ignored(tmp_path):
    """Guards against resuming into a grid that was rebuilt smaller."""
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    np.save(run_dir / "partial_test.npy", np.zeros((N_WINDOWS + 10, SAMPLES, HORIZON)))

    out = run(FakeGrid(), FakeSampler(), run_dir)
    assert out.shape == (N_WINDOWS, HORIZON, SAMPLES)


def test_resuming_with_a_different_batch_size_is_refused(tmp_path):
    """The subtle one: seeds are `seed + start`, so batch boundaries decide the draws.

    Resuming at a different batch size would reseed every remaining window and splice
    two different forecasters into one grid, with no downstream signal that it happened.
    """
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    reference = run(FakeGrid(), FakeSampler(), run_dir, batch_size=4)
    prior = np.stack([reference.transpose(0, 2, 1)[i] for i in range(12)])
    np.save(run_dir / "partial_test.npy", prior)
    (run_dir / "partial_test.meta.json").write_text(
        json.dumps(
            {
                "batch_size": 4,
                "sample_count": SAMPLES,
                "top_p": 0.9,
                "temperature": 1.0,
                "seed": 1234,
                "n_windows": N_WINDOWS,
                "horizon": HORIZON,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="batch_size"):
        run(FakeGrid(), FakeSampler(), run_dir, batch_size=6)


def test_resuming_with_matching_parameters_is_allowed(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    reference = run(FakeGrid(), FakeSampler(), run_dir, batch_size=4)

    prior = np.stack([reference.transpose(0, 2, 1)[i] for i in range(12)])
    np.save(run_dir / "partial_test.npy", prior)
    (run_dir / "partial_test.meta.json").write_text(
        json.dumps(
            {
                "batch_size": 4,
                "sample_count": SAMPLES,
                "top_p": 0.9,
                "temperature": 1.0,
                "seed": 1234,
                "n_windows": N_WINDOWS,
                "horizon": HORIZON,
            }
        ),
        encoding="utf-8",
    )
    resumed = run(FakeGrid(), FakeSampler(), run_dir, batch_size=4)
    np.testing.assert_allclose(resumed, reference, rtol=0, atol=0)


def test_a_partial_without_a_signature_still_resumes(tmp_path):
    """Backward compatibility: partials written before signatures existed."""
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    reference = run(FakeGrid(), FakeSampler(), run_dir)
    prior = np.stack([reference.transpose(0, 2, 1)[i] for i in range(12)])
    np.save(run_dir / "partial_test.npy", prior)  # no .meta.json alongside

    resumed = run(FakeGrid(), FakeSampler(), run_dir)
    np.testing.assert_allclose(resumed, reference, rtol=0, atol=0)


def test_signature_is_written_and_cleaned_up(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run(FakeGrid(), FakeSampler(), run_dir)
    assert not (run_dir / "partial_test.meta.json").exists()
    assert not (run_dir / "partial_test.npy").exists()


def test_progress_heartbeat_is_written(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run(FakeGrid(), FakeSampler(), run_dir)

    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["stage"] == "test"
    assert progress["done"] == N_WINDOWS
    assert progress["total"] == N_WINDOWS
