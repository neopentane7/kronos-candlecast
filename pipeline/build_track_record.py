"""Rebuild the backtest half of the published track record.

Split from the nightly job on purpose. This half is derived from the archive and the
committed corpus, and the corpus is gitignored, so the Actions runner cannot produce it.
The nightly job only ever appends to the `live` half; if it regenerated this one it would
erase it on the first scheduled run.

    uv run python pipeline/build_track_record.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline import archive, track_record  # noqa: E402
from pipeline.run_nightly import (  # noqa: E402
    ARCHIVE_ROOT,
    LIVE_LEDGER,
    SITE_DATA,
    load_corpus_bars,
)


def main() -> int:
    corpus = load_corpus_bars(REPO_ROOT / "data" / "parquet")
    if not corpus:
        raise SystemExit("the backtest series needs the local corpus at data/parquet")

    payload = track_record.build(archive.read_all(ARCHIVE_ROOT), corpus)
    # Every date in this half came from a replayed forecast, and the panel has to say so.
    payload["backtest"] = payload.pop("levels")
    payload["live"] = (
        json.loads(LIVE_LEDGER.read_text(encoding="utf-8")).get("days", [])
        if LIVE_LEDGER.exists()
        else []
    )

    out = SITE_DATA / "track_record.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{out}: {payload['days']} backtest dates, {len(payload['live'])} live")
    print(f"{'level':>7}{'trailing':>11}{'days':>7}{'cumulative':>13}{'gap':>10}")
    for key, lvl in payload["backtest"].items():
        t = lvl["trailing"]
        if not t:
            continue
        gap = (t["coverage"] - lvl["nominal"]) * 100
        print(
            f"{key:>7}{t['coverage']:>11.4f}{t['days']:>7}{lvl['cumulative']:>13.4f}{gap:>+9.1f}pp"
        )
    print("\ncumulative is published but never headlined: it averages the ACI warm-up")
    print("against the steady state and can read near-nominal by cancellation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
