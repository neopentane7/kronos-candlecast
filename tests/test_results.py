"""Tests for the rule-9 results-directory contract."""

import json
import re

from common.results import DISCLAIMER, RESULTS_ROOT, git_sha, new_run_dir, write_results

RUN_DIR_PATTERN = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{7,}(-dirty)?$|^\d{8}T\d{6}Z_nogit$")


def test_git_sha_is_resolvable():
    sha = git_sha()
    assert sha
    assert " " not in sha


def test_new_run_dir_matches_rule9_naming():
    run_dir = new_run_dir()
    try:
        assert run_dir.is_dir()
        assert run_dir.parent == RESULTS_ROOT
        assert RUN_DIR_PATTERN.match(run_dir.name), run_dir.name
    finally:
        run_dir.rmdir()


def test_write_results_stamps_sha_and_disclaimer(tmp_path):
    path = write_results(tmp_path, {"run": "unit_test", "metric": 1.23})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "results.json"
    assert payload["run"] == "unit_test"
    assert payload["metric"] == 1.23
    # Hard constraints 9 and 10: traceable to a commit, and never advice.
    assert payload["git_sha"] == git_sha()
    assert payload["disclaimer"] == DISCLAIMER
    assert "not investment advice" in payload["disclaimer"]
