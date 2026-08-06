"""Corpus fingerprint: identify *which* corpus a number was computed from.

Hard constraint 9 makes every result traceable to a commit, but the corpus is gitignored
and travels separately -- as a zip uploaded to a cloud runner. A git SHA therefore says
nothing about which prices were used. When the corpus was corrected (one orphan session
dropped) the committed golden numbers changed while the SHA that produced them did not,
and a runner holding the previous zip would have reproduced the *old* numbers perfectly.

The fingerprint closes that gap. It is checked before a long run starts, so a stale corpus
fails in the first seconds with "re-upload the dataset" rather than in the port check with
an arithmetic mismatch that looks like a harness bug.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb


def fingerprint(parquet_root: Path | str) -> dict:
    """Row counts and date bounds per ticker, hashed into one short digest.

    Deliberately structural rather than a hash of the file bytes: Parquet encodes the same
    table differently depending on writer version and compression, so byte hashes report
    differences that do not exist. This changes only when the data changes.
    """
    root = Path(parquet_root)
    con = duckdb.connect()
    rows = con.sql(f"""
        SELECT ticker,
               count(*) AS n,
               min(CAST(timestamps AS DATE)) AS first,
               max(CAST(timestamps AS DATE)) AS last
        FROM read_parquet('{root.as_posix()}/*/*.parquet', hive_partitioning=1)
        GROUP BY ticker ORDER BY ticker
    """).fetchall()

    payload = [[str(t), int(n), str(a), str(b)] for t, n, a, b in rows]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()[
        :16
    ]

    return {
        "digest": digest,
        "n_tickers": len(payload),
        "n_rows": sum(p[1] for p in payload),
        "first": min(p[2] for p in payload) if payload else None,
        "last": max(p[3] for p in payload) if payload else None,
    }


def check(parquet_root: Path | str, expected: dict) -> None:
    """Raise with an actionable message if the corpus is not the one ``expected`` names."""
    got = fingerprint(parquet_root)
    if got["digest"] == expected.get("digest"):
        return
    raise SystemExit(
        "corpus fingerprint mismatch -- this is not the corpus the golden numbers were "
        f"measured on.\n  expected {expected.get('digest')} "
        f"({expected.get('n_rows')} rows, {expected.get('n_tickers')} tickers)\n"
        f"  found    {got['digest']} ({got['n_rows']} rows, {got['n_tickers']} tickers)\n\n"
        "If you are on a cloud runner, the attached dataset predates the corpus "
        "correction: re-zip data/ locally, upload it as a new dataset version, and "
        "re-attach it. Do not re-run fetch_nse.py -- back-adjustment would rewrite history "
        "and produce a third corpus."
    )
