"""Export adapter: partitioned Parquet -> the CSV layout upstream training consumes.

Parquet is the source of truth (hard constraint 8); this is a one-way export. Every
partition is re-validated against the canonical schema on the way out, so a corpus that
was written by an older version of the schema cannot quietly reach training.

Why per-ticker, per-split files
-------------------------------
Upstream's ``CustomKlineDataset`` (finetune_csv/finetune_base_model.py) reads ONE csv,
sorts it globally by ``timestamps``, and slides ``iloc`` windows across the whole frame,
splitting by *ratio*. Handing it 54 concatenated tickers would interleave them by date
and splice windows across ticker boundaries, and a single global ratio cannot express
the date splits of hard constraint 7. So we emit one file per ticker per split and keep
the boundaries explicit. A4 supplies an overlay dataset that reads this layout, builds
windows within a ticker, and never crosses a split edge.

Column order is upstream's (``timestamps, open, close, high, low, volume, amount``),
which puts close BEFORE high -- not our canonical OHLC order.

Usage (PowerShell):
    uv run python phase-a/scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.preprocess import UPSTREAM_CSV_COLUMNS, validate  # noqa: E402
from common.results import DISCLAIMER  # noqa: E402
from common.splits import SPLITS, slice_split  # noqa: E402

PARQUET_ROOT = REPO_ROOT / "data" / "parquet"
CSV_ROOT = REPO_ROOT / "data" / "csv"
DATASET_MANIFEST = REPO_ROOT / "data" / "dataset_manifest.json"

LOOKBACK = 400  # hard constraint 2
HORIZON = 30  # contract v2
STRIDE = 30  # A3 eval grid

# A *training* window must fit entirely inside its split, so the split needs this many
# rows. Only `train` does; see EVAL_WINDOW_NOTE for why that is fine.
MIN_TRAIN_SPLIT_ROWS = LOOKBACK + HORIZON + 1

EVAL_WINDOW_NOTE = (
    "Splits partition forecast TARGETS, not input windows. val (~245 sessions/ticker) "
    "and test (~366) are each shorter than the 431 rows a self-contained window needs, "
    "so an eval window takes its 400-bar lookback from before the split boundary and "
    "only its 30-step target inside. That is not leakage -- at serving time the model "
    "always conditions on the most recent 400 bars regardless of period; what must not "
    "leak is TRAINING on target-period data, and train windows are fully contained. "
    "The eval harness therefore reads data/parquet (the source of truth) rather than "
    "these per-split CSVs, which exist for upstream training only."
)


def load_corpus(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every partition, in canonical column order, ordered by ticker then time."""
    glob = (PARQUET_ROOT / "*" / "*.parquet").as_posix()
    return con.sql(
        f"""
        SELECT ticker, timestamps, open, high, low, close, volume, amount
        FROM read_parquet('{glob}', hive_partitioning=true)
        ORDER BY ticker, timestamps
        """
    ).to_df()


def eval_window_count(df: pd.DataFrame, split: str) -> int:
    """How many rolling eval windows this ticker supports in ``split``.

    A window is admissible when its 30-step target lies inside the split AND at least
    ``LOOKBACK`` bars of history precede it anywhere in the series.
    """
    part = slice_split(df, split)
    if part.empty:
        return 0
    first_target = df.index[df["timestamps"] == part["timestamps"].iloc[0]][0]
    if first_target < LOOKBACK:
        return 0  # not enough history before the split to condition on
    return max(0, (len(part) - HORIZON) // STRIDE + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CSV_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out)

    if not PARQUET_ROOT.exists():
        print(f"No corpus at {PARQUET_ROOT}. Run fetch_nse.py first.")
        return 1

    con = duckdb.connect()
    corpus = load_corpus(con)
    tickers = sorted(corpus["ticker"].unique())
    print(f"corpus: {len(tickers)} tickers, {len(corpus)} rows")

    entries: list[dict] = []
    validation_failures: list[dict] = []
    split_totals = dict.fromkeys(SPLITS, 0)

    for ticker in tickers:
        df = corpus.loc[corpus["ticker"] == ticker].drop(columns="ticker").reset_index(drop=True)

        # Acceptance requires pandera to pass on every partition, so re-check here
        # rather than trusting whatever schema version wrote the Parquet.
        try:
            validate(df)
        except Exception as exc:  # noqa: BLE001 - recorded, and fails the run below
            validation_failures.append({"ticker": ticker, "reason": str(exc)[:400]})
            print(f"  {ticker}: FAILED validation, not exported")
            continue

        per_split = {}
        for split in SPLITS:
            part = slice_split(df, split)
            per_split[split] = int(len(part))
            split_totals[split] += len(part)
            if part.empty:
                continue
            split_dir = out_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            part[UPSTREAM_CSV_COLUMNS].to_csv(split_dir / f"{ticker}.csv", index=False)

        entries.append(
            {
                "ticker": ticker,
                "rows": int(len(df)),
                "rows_per_split": per_split,
                # Train windows must be self-contained; eval windows need only enough
                # history before the split to supply a lookback (see EVAL_WINDOW_NOTE).
                "train_windows_contained": bool(per_split["train"] >= MIN_TRAIN_SPLIT_ROWS),
                "eval_windows": {s: eval_window_count(df, s) for s in ("val", "test")},
            }
        )

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "disclaimer": DISCLAIMER,
        "source": "data/parquet (hard constraint 8: Parquet is the source of truth)",
        "csv_column_order": UPSTREAM_CSV_COLUMNS,
        "layout": "data/csv/<split>/<TICKER>.csv",
        "splits": SPLITS,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "eval_stride": STRIDE,
        "min_rows_for_a_contained_train_window": MIN_TRAIN_SPLIT_ROWS,
        "eval_window_note": EVAL_WINDOW_NOTE,
        "tickers_exported": len(entries),
        "validation_failures": validation_failures,
        "rows_per_split": split_totals,
        "tickers": entries,
    }
    DATASET_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n--- build summary ---")
    for split, total in split_totals.items():
        start, end = SPLITS[split]
        if split == "train":
            n = sum(1 for e in entries if e["train_windows_contained"])
            detail = f"tickers with contained train windows={n}"
        else:
            n = sum(e["eval_windows"][split] for e in entries)
            detail = f"eval windows (stride {STRIDE})={n}"
        print(f"{split:<6} {start} -> {end}  rows={total:>7}  {detail}")
    print(f"exported : {len(entries)} tickers -> {out_root}")
    print(f"failed   : {len(validation_failures)}")
    for f in validation_failures:
        print(f"    {f['ticker']}: {f['reason'][:120]}")
    print(f"manifest : {DATASET_MANIFEST}")
    return 1 if validation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
