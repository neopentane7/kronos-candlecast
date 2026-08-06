"""The encoding guard, and a sweep of every tracked text file.

The hook runs on changed files at commit time; this runs on all of them, so a file that
was damaged before the hook existed cannot sit in the tree indefinitely.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "phase-a" / "scripts"))

from check_encoding import TEXT_SUFFIXES, check  # noqa: E402


def test_valid_utf8_passes(tmp_path):
    p = tmp_path / "ok.md"
    p.write_text("clean — en-dash, § section, café\n", encoding="utf-8")
    assert check(p) == []


def test_legitimate_accented_prose_is_not_flagged(tmp_path):
    """The reason the check matches pairs and not single characters.

    A blocklist containing a bare circumflex-a would reject ordinary French, Portuguese
    and Vietnamese text. A guard that fires on correct input gets switched off, and then
    it protects nothing.
    """
    p = tmp_path / "prose.md"
    p.write_text(
        "château, râle, fenêtre, São Paulo, ação, Ângela, mangiò, ¿cómo?, ±3%, 90° C\n",
        encoding="utf-8",
    )
    assert check(p) == []


def test_bom_is_rejected(tmp_path):
    p = tmp_path / "bom.md"
    p.write_bytes(b"\xef\xbb\xbfhello\n")
    assert any("BOM" in m for m in check(p))


def test_cp1252_bytes_are_rejected(tmp_path):
    """A file written wholesale in the system codepage."""
    p = tmp_path / "ansi.md"
    p.write_bytes("en-dash – here\n".encode("cp1252"))
    assert any("not valid UTF-8" in m for m in check(p))


def test_double_encoded_text_is_rejected(tmp_path):
    """The subtle case: valid UTF-8 whose *contents* are mangled.

    No decoder can flag this, because the file really is well-formed UTF-8 -- it just
    says the wrong thing. This is exactly what the two PowerShell round-trips produced.
    """
    damaged = "en-dash – here".encode().decode("cp1252")
    p = tmp_path / "mojibake.md"
    p.write_text(damaged + "\n", encoding="utf-8")

    assert p.read_bytes().decode("utf-8")  # well-formed, so decoding alone won't catch it
    problems = check(p)
    assert any("double-encoding" in m for m in problems)
    assert any("line 1" in m for m in problems)


def test_every_tracked_text_file_is_clean():
    """The sweep. Catches damage that predates the hook."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\n")

    problems = {}
    for name in filter(None, out):
        path = REPO / name
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        found = check(path)
        if found:
            problems[name] = found

    assert not problems, "encoding damage in tracked files:\n" + "\n".join(
        f"  {k}: {v}" for k, v in problems.items()
    )


@pytest.mark.parametrize("suffix", [".md", ".py", ".json", ".ipynb", ".yaml"])
def test_the_suffixes_that_matter_are_covered(suffix):
    assert suffix in TEXT_SUFFIXES
