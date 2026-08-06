"""Reject text files that are not clean UTF-8, or that carry double-encoding damage.

Twice now a PowerShell read/write round-trip has rewritten a Markdown file in the system
codepage, turning en-dashes and section signs into sequences no reader wants. Both times
it was caught by eye, which is not a control. This is the documentation equivalent of
``test_corpus_has_no_orphan_sessions``: make the failure mechanically impossible instead
of relying on someone noticing.

Three checks:

* **decodable as UTF-8** -- catches a file written in cp1252 outright;
* **no BOM** -- a UTF-8 BOM breaks shebangs, some YAML parsers, and diffs badly;
* **no mojibake signature** -- catches the subtler case where the file is *valid* UTF-8
  but its contents are UTF-8 bytes that were previously decoded as cp1252 and re-encoded.
  No decoder can flag that for you: the file is well formed, it just says the wrong thing.

Usage:
    uv run python phase-a/scripts/check_encoding.py FILE [FILE ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Structural signature rather than a list of known-bad examples.
#
# When UTF-8 bytes are decoded as cp1252 and re-encoded, one multi-byte character becomes
# a *lead* lookalike -- U+00C2, U+00C3 or U+00E2, from lead bytes 0xC2/0xC3/0xE2 --
# followed by whatever cp1252 maps its continuation bytes to. Continuation bytes are
# always 0x80..0xBF, which cp1252 renders as its C1 punctuation block plus U+00A0..U+00BF.
#
# Matching the *pair* is what keeps this usable. "â" alone occurs in ordinary words
# (chateau, rale, fenetre with circumflexes), so a bare-character blocklist would reject
# legitimate prose and promptly get switched off. A lead followed by a continuation
# lookalike does not arise naturally in any language.
_LEAD = "ÂÃâ"
_CONT = (
    # cp1252's printable renderings of bytes 0x80-0x9F
    "€‚ƒ„…†‡ˆ‰Š‹ŒŽ"
    "‘’“”•–—˜™š›œž"
    "Ÿ"
    # and bytes 0xA0-0xBF, which map to themselves
    " -¿"
)
MOJIBAKE_RE = re.compile(f"[{_LEAD}][{_CONT}]")

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".ipynb",
    ".ps1",
    ".sh",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".java",
    ".xml",
}


def check(path: Path) -> list[str]:
    problems = []
    raw = path.read_bytes()

    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("has a UTF-8 BOM")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"is not valid UTF-8 at byte {exc.start} ({exc.reason})")
        return problems

    for lineno, line in enumerate(text.splitlines(), 1):
        match = MOJIBAKE_RE.search(line)
        if match:
            problems.append(
                f"contains a double-encoding sequence {match.group()!r} at line {lineno} "
                "-- the file was written through a non-UTF-8 codepage"
            )
            break
    return problems


def main(argv: list[str]) -> int:
    failed = False
    for name in argv:
        path = Path(name)
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        for problem in check(path):
            print(f"{path}: {problem}")
            failed = True

    if failed:
        print(
            "\nFix: rewrite the file as UTF-8 without a BOM. If a PowerShell round-trip "
            "caused it, restore with `git checkout -- <file>` and edit the file directly "
            "rather than via Get-Content/Set-Content."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
