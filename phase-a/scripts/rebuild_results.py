"""Rebuild a run's ``results.json`` from its ``ensembles.npz``.

Recovery tool. ``run_analysis.py`` used to call ``write_results`` without a filename,
so the offline pass overwrote the grid metrics of the run it was analysing. Two runs
were damaged that way before it was caught.

Everything scored in ``results.json`` is a deterministic function of the saved arrays,
so the metric block is fully recoverable with no GPU. What is *not* recoverable is the
run's own bookkeeping -- wall clock, peak VRAM, the sampling configuration -- because
those were never in the arrays. Rebuilt files are marked ``rebuilt: true`` and carry
the fields that could not be restored, so a reader can never mistake one for the
original artifact.

Usage (PowerShell):
    uv run python phase-a/scripts/rebuild_results.py results/<run-dir>
    uv run python phase-a/scripts/rebuild_results.py results/<run-dir> --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))

from eval.metrics import CRPS_ESTIMATOR, QUANTILE_METHOD, summarize  # noqa: E402

from common.results import DISCLAIMER, write_results  # noqa: E402

UNRECOVERABLE = [
    "wall_clock (kronos_grid_seconds, seconds_per_window)",
    "peak_vram_mb",
    "sampling policy actually used (sample_count, top_p, temperature, seed)",
    "sampling_policy_sweep, if the run performed one",
]


def rebuild(run_dir: Path, split: str, seed: int) -> dict:
    npz = np.load(run_dir / "ensembles.npz", allow_pickle=True)
    obs, blocks = npz["y_close"], npz["block_ids"]
    models = {k[len("ens__") :]: npz[k] for k in npz.files if k.startswith("ens__")}
    if not models:
        raise SystemExit(f"no ens__* arrays in {run_dir / 'ensembles.npz'}")

    return {
        "run": "A3_zero_shot_gate",
        "split": split,
        "rebuilt": True,
        "rebuilt_from": "ensembles.npz",
        "rebuilt_because": (
            "the original results.json was overwritten by run_analysis.py, which called "
            "write_results without a filename"
        ),
        "not_recoverable": UNRECOVERABLE,
        "quantile_method": QUANTILE_METHOD,
        "crps_estimator": CRPS_ESTIMATOR,
        "models": {
            name: summarize(obs, ens, blocks, seed=seed) for name, ens in sorted(models.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed; harness default is 0")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results.json")
    args = ap.parse_args()

    target = args.run_dir / "results.json"
    if target.exists() and not args.force:
        import json

        existing = json.loads(target.read_text(encoding="utf-8"))
        if "models" in existing and not existing.get("rebuilt"):
            raise SystemExit(
                f"{target} already holds grid metrics -- refusing to overwrite an intact "
                "artifact. Pass --force only if you know it is damaged."
            )

    payload = rebuild(args.run_dir, args.split, args.seed)
    path = write_results(args.run_dir, payload)

    print(f"rebuilt {path}")
    print(f"{'model':<22}{'CRPS(fair)':>12}{'cov@80':>9}{'IS@80':>11}{'blocks':>8}")
    for name, m in payload["models"].items():
        print(
            f"{name:<22}{m['crps']:>12.4f}{m['coverage']['80']['empirical']:>9.4f}"
            f"{m['interval_score_80']:>11.3f}{m['effective_blocks']:>8}"
        )
    print("\nnot recovered (never lived in the arrays):")
    for item in UNRECOVERABLE:
        print(f"  - {item}")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
