"""golden.json is the single source of the port-check numbers; keep it honest.

The Kaggle notebook and this suite read the same file, so the two cannot drift apart. What
these tests guard is the file itself: that it is well formed, that it still describes the
corpus on this machine, and -- when the corpus is present -- that the numbers in it are
what the harness actually produces.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "phase-a"))

GOLDEN = REPO / "phase-a" / "eval" / "golden.json"
PARQUET = REPO / "data" / "parquet"

needs_corpus = pytest.mark.skipif(
    not PARQUET.exists(), reason="corpus is gitignored; present only on a provisioned machine"
)


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_golden_file_is_well_formed(golden):
    for key in ("corpus", "models", "effective_blocks", "n_windows", "tolerances"):
        assert key in golden, f"golden.json is missing {key}"
    assert golden["corpus"]["digest"]
    assert set(golden["models"]) == {"last_value", "random_walk_drift"}
    for m in golden["models"].values():
        assert {"crps", "interval_score_80", "coverage_80"} <= set(m)


def test_block_count_is_the_corrected_one(golden):
    """22 was the pre-correction count, inflated by one ticker's orphan sessions.

    Pinned because the number is an input to the A5 power analysis (report §17c), not
    merely a reporting detail: at 22 the +-2pp criterion looks 34% more attainable than
    it is.
    """
    assert golden["effective_blocks"] == 12
    assert golden["n_windows"] == 708


@needs_corpus
def test_fingerprint_matches_the_corpus_on_this_machine(golden):
    """Catches a corpus that was refetched or restored from a stale archive."""
    from common.corpus import check, fingerprint

    got = fingerprint(PARQUET)
    assert got["digest"] == golden["corpus"]["digest"], (
        f"corpus digest {got['digest']} != golden {golden['corpus']['digest']}; "
        "regenerate with phase-a/scripts/make_golden.py or restore the corpus"
    )
    check(PARQUET, golden["corpus"])  # must not raise


@needs_corpus
def test_corpus_has_no_orphan_sessions():
    """The defect that started this: one ticker trading on a date the universe does not.

    Left in place it splits the evaluation grid's blocks and inflates the reported
    effective sample size, silently and without any schema check firing.
    """
    sys.path.insert(0, str(REPO / "phase-a" / "scripts"))
    from audit_calendar import audit, load_dates

    rep = audit(load_dates(f"{PARQUET.as_posix()}/*/*.parquet"))
    assert rep["n_orphan_rows"] == 0, (
        f"orphan sessions present: {rep['tickers_with_disagreements']}. "
        "Run: uv run python phase-a/scripts/audit_calendar.py --fix"
    )


@needs_corpus
def test_stale_corpus_is_refused_with_an_actionable_message(golden):
    """The failure this gate exists for: a cloud runner holding the pre-correction zip.

    Without it the stale corpus reproduces the *superseded* numbers exactly and the port
    check fails looking like a harness bug, hours into a paid session.
    """
    from common.corpus import check

    stale = {**golden["corpus"], "digest": "0000000000000000"}
    with pytest.raises(SystemExit) as exc:
        check(PARQUET, stale)
    message = str(exc.value)
    assert "fingerprint mismatch" in message
    assert "re-zip" in message  # tells the reader what to do, not just that it broke
    assert "fetch_nse" in message  # and what not to do


def test_fingerprint_is_structural_not_a_byte_hash(tmp_path):
    """Two parquet files holding the same table must fingerprint identically.

    Parquet encodes the same rows differently across writer versions and compression
    settings. A byte hash would report a corpus change on a rewrite that changed nothing,
    and the gate would cry wolf until someone disabled it.
    """
    pd = pytest.importorskip("pandas")
    from common.corpus import fingerprint

    df = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
            "amount": [10.0, 40.0],
        }
    )
    a, b = tmp_path / "a" / "ticker=X", tmp_path / "b" / "ticker=X"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    df.to_parquet(a / "data.parquet", index=False, compression="snappy")
    df.to_parquet(b / "data.parquet", index=False, compression="gzip")

    assert fingerprint(a.parent)["digest"] == fingerprint(b.parent)["digest"]

    df.iloc[:1].to_parquet(b / "data.parquet", index=False)  # a real change
    assert fingerprint(a.parent)["digest"] != fingerprint(b.parent)["digest"]


@needs_corpus
@pytest.mark.slow
def test_harness_reproduces_the_golden_numbers(golden, tmp_path):
    """The end-to-end claim the Kaggle port check makes, asserted locally too.

    Roughly a minute: baselines only, no model, no figures.
    """
    # Write into tmp_path rather than results/: the harness stamps a directory per run,
    # and a test that ran on every commit would silt up the repo with runs nobody cites.
    env = {**os.environ, "KRONOS_RESULTS_ROOT": str(tmp_path)}
    proc = subprocess.run(
        [
            sys.executable,
            "phase-a/eval/calibrate.py",
            "--split",
            "test",
            "--skip-model",
            "--no-figures",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    runs = list(tmp_path.glob("*/results.json"))
    assert len(runs) == 1, f"expected one run directory in {tmp_path}, got {runs}"
    res = json.loads(runs[0].read_text(encoding="utf-8"))

    tol = golden["tolerances"]
    for name, want in golden["models"].items():
        got = res["models"][name]
        assert abs(got["crps"] - want["crps"]) <= tol["crps"], f"{name} CRPS"
        assert abs(got["coverage"]["80"]["empirical"] - want["coverage_80"]) <= tol["coverage"], (
            f"{name} coverage@80"
        )
    assert res["models"]["last_value"]["effective_blocks"] == golden["effective_blocks"]
