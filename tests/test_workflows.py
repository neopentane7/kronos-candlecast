"""The nightly job's output is only real if the workflow commits it and the site redeploys.

Two failures already happened here, and neither raised anything. The commit step named
state files one at a time, so `track_record_live.json` was written on the runner for seven
nights and discarded every time. And the Pages deploy waited on a push trigger that a
GITHUB_TOKEN push cannot fire, so the published site served a 2026-08-27 forecast while the
archive ran to 09-08.

Both are workflow-shaped, so no Python test would have caught them. These read the YAML.
"""

import sys
from pathlib import Path

# Imported outright rather than through importorskip: both failures below were silent,
# and a guard that turns a missing parser into a skip would be a third.
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


WORKFLOWS = REPO / ".github" / "workflows"
NIGHTLY = WORKFLOWS / "nightly-forecast.yml"
PAGES = WORKFLOWS / "pages.yml"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def steps(wf: dict) -> list[dict]:
    return [s for job in wf["jobs"].values() for s in job.get("steps", [])]


def script(wf: dict) -> str:
    return "\n".join(s.get("run", "") for s in steps(wf))


# ------------------------------------------------- everything written is committed
def test_the_commit_step_covers_the_whole_state_directory():
    """Naming state files individually is what lost seven nights of the live ledger.

    A directory cannot omit a file added later; a path list silently can, and did.
    """
    body = script(load(NIGHTLY))
    add = next(ln.strip() for ln in body.splitlines() if ln.strip().startswith("git add"))
    assert "pipeline/state" in add, add
    assert "pipeline/state/aci_state.json" not in add, (
        "a named file here means the next state file the pipeline writes is dropped silently"
    )


def test_every_path_the_nightly_job_writes_is_staged():
    """Derived from the module's own constants, so adding an output updates this test."""
    from pipeline import run_nightly

    body = script(load(NIGHTLY))
    add = next(ln.strip() for ln in body.splitlines() if ln.strip().startswith("git add"))
    staged = [Path(p) for p in add.replace("git add", "").split()]

    written = [run_nightly.ACI_PATH, run_nightly.LIVE_LEDGER, run_nightly.SITE_DATA]
    for target in written:
        rel = target.resolve().relative_to(REPO.resolve())
        assert any(rel == p or p in rel.parents for p in staged), (
            f"{rel} is written by the nightly job but is not staged by: {add}"
        )


# ------------------------------------------------- the deploy can actually be triggered
def test_the_nightly_job_dispatches_the_deploy_itself():
    """A GITHUB_TOKEN push does not fire workflows, so the push trigger is not enough.

    Without this the site freezes at the last human push and nothing reports it: every
    workflow run is green, the data is committed, and the published page is stale.
    """
    wf = load(NIGHTLY)
    body = script(wf)
    assert "gh workflow run pages.yml" in body, (
        "nothing redeploys the site after the nightly commit"
    )
    # gh needs actions:write to dispatch, and a token in the environment.
    assert wf.get("permissions", {}).get("actions") == "write"
    assert any("GH_TOKEN" in str(s.get("env", {})) for s in steps(wf))


def test_the_dispatch_runs_after_the_push_not_before():
    """Dispatching first would deploy the previous day's data."""
    body = script(load(NIGHTLY))
    assert body.index("git push origin") < body.index("gh workflow run pages.yml")


def test_pages_still_accepts_a_manual_dispatch():
    """The dispatch path must exist on the receiving end, or the call is a no-op."""
    # PyYAML parses a bare `on:` key as the boolean True, which is the YAML 1.1 rule.
    triggers = load(PAGES).get("on") or load(PAGES).get(True)
    assert "workflow_dispatch" in triggers


def test_pages_still_deploys_on_a_human_push_to_the_data():
    triggers = load(PAGES).get("on") or load(PAGES).get(True)
    paths = triggers["push"]["paths"]
    assert "site/data/**" in paths
    assert "candlecast-web/**" in paths


# ------------------------------------------------- the deploy refuses to ship nothing
def test_the_build_fails_rather_than_publishing_an_empty_site():
    """A staging step that silently copied nothing would deploy a blank page."""
    body = script(load(PAGES))
    assert 'test "$count" -gt 0' in body or "test $count -gt 0" in body
    assert "dist/data/index.json" in body
