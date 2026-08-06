"""Regenerate ``phase-a/eval/golden.json`` -- the one place the port-check numbers live.

The Kaggle notebook and the local test suite both assert the same baseline numbers. Held
in two places they drift, and the drift shows up as a failed cloud session rather than a
failed test. Both now read this file, and it is generated from a run rather than typed.

It carries a corpus fingerprint as well as the metrics, because the numbers are a property
of the corpus and the corpus is gitignored: a runner with a stale dataset would otherwise
reproduce superseded numbers exactly and look correct.

Usage (PowerShell):
    uv run python phase-a/scripts/make_golden.py results/<run-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.corpus import fingerprint  # noqa: E402
from common.results import DISCLAIMER, git_sha  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "phase-a" / "eval" / "golden.json"
BASELINES = ("last_value", "random_walk_drift")

# Baselines are pure NumPy and deterministic from the grid and the seed, so these should
# reproduce bit-for-bit on any machine. The tolerances are for float summation order
# across BLAS builds, not for genuine disagreement -- if a number moves more than this,
# the corpus or the harness changed and that is the thing to investigate.
CRPS_TOL = 1e-3
COVERAGE_TOL = 1e-4


def build(run_dir: Path, parquet_root: Path) -> dict:
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    models = {}
    for name in BASELINES:
        m = res["models"][name]
        models[name] = {
            "crps": round(m["crps"], 4),
            "interval_score_80": round(m["interval_score_80"], 4),
            "coverage_80": round(m["coverage"]["80"]["empirical"], 4),
        }
    blocks = res["models"][BASELINES[0]]["effective_blocks"]
    return {
        "disclaimer": DISCLAIMER,
        "generated_from_run": run_dir.name,
        "git_sha": git_sha(),
        "split": res.get("split", "test"),
        "n_windows": res["models"][BASELINES[0]]["n_windows"],
        "effective_blocks": blocks,
        "corpus": fingerprint(parquet_root),
        "tolerances": {"crps": CRPS_TOL, "coverage": COVERAGE_TOL},
        "models": models,
        "note": (
            "Regenerate with phase-a/scripts/make_golden.py after any corpus correction. "
            "Both the Kaggle notebook port check and tests/test_golden.py read this file."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--parquet", type=Path, default=REPO_ROOT / "data" / "parquet")
    ap.add_argument("--out", type=Path, default=GOLDEN_PATH)
    args = ap.parse_args()

    payload = build(args.run_dir, args.parquet)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(
        f"  corpus  {payload['corpus']['digest']}  "
        f"({payload['corpus']['n_rows']} rows, {payload['corpus']['n_tickers']} tickers)"
    )
    print(f"  blocks  {payload['effective_blocks']}   windows {payload['n_windows']}")
    for name, m in payload["models"].items():
        print(
            f"  {name:<20} crps={m['crps']:<10} cov80={m['coverage_80']:<8} "
            f"IS={m['interval_score_80']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
