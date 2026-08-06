"""Build the corpus archive for upload to a cloud runner.

Not a one-liner because PowerShell's ``Compress-Archive`` writes **backslash** path
separators into the archive. The ZIP specification requires forward slashes, and Kaggle
rejects the upload with "contains a forbidden character in name ('\\')" for every entry.
Python's ``zipfile`` writes conforming names, so the archive is built here instead.

The fingerprint of the archive's contents is printed and checked against
``phase-a/eval/golden.json``, so an archive built from a stale or partially corrected
corpus is caught here rather than by the gate on the runner.

Usage (PowerShell):
    uv run python phase-a/scripts/make_corpus_zip.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.corpus import fingerprint  # noqa: E402

DEFAULT_SRC = REPO_ROOT / "data" / "parquet"
DEFAULT_OUT = REPO_ROOT / "kronos-nse-corpus.zip"
GOLDEN = REPO_ROOT / "phase-a" / "eval" / "golden.json"


def build(src: Path, out: Path) -> list[str]:
    """Zip ``src`` under a top-level ``parquet/`` so the notebook's autodetect finds it."""
    files = sorted(p for p in src.rglob("*.parquet") if p.is_file())
    if not files:
        raise SystemExit(f"no parquet files under {src}")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            # as_posix() is the whole point: forward slashes, per the ZIP spec.
            z.write(f, arcname=(Path(src.name) / f.relative_to(src)).as_posix())
    return [i.filename for i in zipfile.ZipFile(out).infolist()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    names = build(args.src, args.out)
    bad = [n for n in names if "\\" in n]
    if bad:
        raise SystemExit(f"archive contains backslash separators: {bad[:3]}")

    size_mb = args.out.stat().st_size / 2**20
    print(f"{args.out.name}  {len(names)} entries  {size_mb:.2f} MB")
    print(f"  top level: {sorted({n.split('/')[0] for n in names})}")
    print(f"  sample:    {names[0]}")

    # Round-trip: extract to a temp dir and fingerprint what a runner would actually see.
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(args.out).extractall(tmp)
        got = fingerprint(Path(tmp) / args.src.name)
    print(f"  corpus:    {got['digest']}  ({got['n_rows']} rows, {got['n_tickers']} tickers)")

    if GOLDEN.exists():
        want = json.loads(GOLDEN.read_text(encoding="utf-8"))["corpus"]
        if got["digest"] != want["digest"]:
            raise SystemExit(
                f"archive fingerprint {got['digest']} != golden {want['digest']}.\n"
                "The archive was built from a corpus the golden numbers do not describe. "
                "Run phase-a/scripts/audit_calendar.py, then make_golden.py, before "
                "uploading this."
            )
        print("  MATCHES golden.json -- safe to upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
