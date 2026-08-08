"""An offline pass must not overwrite the measurement it was run to analyse.

run_analysis.py called write_results without a filename, so it wrote results.json --
replacing the grid metrics with an analysis payload. It destroyed two runs before it
was noticed, and neither failure produced an error: the analysis printed its own
results and exited zero.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "phase-a"))

from common.results import write_results  # noqa: E402


def _calls_write_results_with_filename(path: Path) -> list[int]:
    """Line numbers of write_results calls that omit `filename`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "write_results":
            continue
        if not any(kw.arg == "filename" for kw in node.keywords) and len(node.args) < 3:
            bad.append(node.lineno)
    return bad


def test_offline_analysis_names_its_output_file():
    """The regression. Any analysis writing results.json is destroying a measurement."""
    analysis = REPO / "phase-a" / "eval" / "run_analysis.py"
    bad = _calls_write_results_with_filename(analysis)
    assert not bad, (
        f"{analysis.name} calls write_results without filename at line(s) {bad}; "
        "it will overwrite the run's results.json"
    )


def test_diagnostics_names_its_output_file():
    diagnose = REPO / "phase-a" / "eval" / "diagnose.py"
    assert not _calls_write_results_with_filename(diagnose)


def test_write_results_defaults_to_results_json(tmp_path):
    """The default is fine -- it is the harness's own name. Callers must opt out."""
    p = write_results(tmp_path, {"run": "x"})
    assert p.name == "results.json"
    assert json.loads(p.read_text())["run"] == "x"


def test_write_results_honours_filename(tmp_path):
    p = write_results(tmp_path, {"run": "y"}, filename="analysis.json")
    assert p.name == "analysis.json"
    assert not (tmp_path / "results.json").exists()


def test_write_results_stamps_every_file(tmp_path):
    """Rule 9 applies to the analysis artifact too, not only the measurement."""
    p = write_results(tmp_path, {"run": "z"}, filename="analysis.json")
    payload = json.loads(p.read_text())
    assert payload["git_sha"]
    assert "not investment advice" in payload["disclaimer"]


def test_rebuild_reproduces_metrics_and_flags_itself(tmp_path):
    """Recovery must be honest about what it could not recover."""
    sys.path.insert(0, str(REPO / "phase-a" / "scripts"))
    from rebuild_results import rebuild

    rng = np.random.default_rng(0)
    n, h, m = 20, 5, 30
    obs = 100 + rng.normal(0, 1, size=(n, h))
    ens = obs[:, :, None] + rng.normal(0, 1, size=(n, h, m))
    np.savez(
        tmp_path / "ensembles.npz",
        y_close=obs,
        block_ids=np.repeat(np.arange(n)[:, None] % 4, h, axis=1),
        ens__model=ens,
    )

    payload = rebuild(tmp_path, "test", seed=0)
    assert payload["rebuilt"] is True
    assert payload["not_recoverable"], "must say what was lost"
    assert "crps" in payload["models"]["model"]
    assert payload["models"]["model"]["effective_blocks"] == 4


def test_rebuild_refuses_to_clobber_an_intact_artifact(tmp_path, monkeypatch):
    """The recovery tool must not become a second way to lose the same file."""
    sys.path.insert(0, str(REPO / "phase-a" / "scripts"))
    import rebuild_results

    (tmp_path / "results.json").write_text(
        json.dumps({"run": "A3_zero_shot_gate", "models": {"a": {}}}), encoding="utf-8"
    )
    np.savez(tmp_path / "ensembles.npz", y_close=np.zeros((2, 2)))

    monkeypatch.setattr(sys, "argv", ["rebuild_results.py", str(tmp_path)])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        rebuild_results.main()
