"""Run-directory helper enforcing hard constraint 9.

Every eval/smoke run writes ``results/<timestamp>_<git-sha>/results.json`` plus
figures, so any number in a report is traceable back to a commit.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

DISCLAIMER = "Research/education tool - scenario visualization, not investment advice."


def git_sha() -> str:
    """Short SHA of HEAD, with a ``-dirty`` suffix if the tree has changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"

    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    return f"{sha}-dirty" if dirty else sha


def new_run_dir() -> Path:
    """Create and return ``results/<UTC timestamp>_<git-sha>/``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_ROOT / f"{stamp}_{git_sha()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_results(run_dir: Path, payload: dict[str, Any]) -> Path:
    """Write ``results.json``, always stamped with the SHA and the disclaimer."""
    path = run_dir / "results.json"
    payload = {"git_sha": git_sha(), "disclaimer": DISCLAIMER, **payload}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
