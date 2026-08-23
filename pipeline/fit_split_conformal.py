"""Fit the 50% band's conformal scale on the 2024 validation split. Offline, no GPU.

§17d permits a finite-sample split-conformal guarantee at exactly one nominal level on
this geometry: 50% needs 3 exchangeable calibration residuals per side, which the 2024
split affords, while 80% needs 9 and 90% needs 19 -- 2.2 and 4.6 years consumed by the
calibration split alone. So this fits 50% and nothing else, deliberately.

The output is state, like the ACI file: fitted once here against the committed corpus and
read by the nightly job, which never sees the corpus. Refitting requires the corpus, so it
happens locally and the result is committed.

Usage (PowerShell):
    uv run python pipeline/fit_split_conformal.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration import QUANTILE_METHOD  # noqa: E402
from pipeline.engines import ENSEMBLE_SIZE, HORIZON, LOOKBACK, RandomWalkDrift  # noqa: E402

VAL_START, VAL_END = "2024-01-01", "2024-12-31"
STRIDE = 30
LEVEL = 0.50
OUT = REPO_ROOT / "pipeline" / "state" / "split_conformal.json"


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected quantile: the ceil((n+1)(1-alpha))/n order statistic.

    The (n+1) correction is what makes the guarantee hold at finite n. Dropping it is the
    single most common way a conformal implementation quietly under-covers.
    """
    n = scores.size
    if n == 0:
        raise ValueError("no calibration residuals")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        raise ValueError(
            f"{n} residuals cannot support a {1 - alpha:.0%} band; "
            f"need at least {int(np.ceil((1 - alpha) / alpha))}"
        )
    return float(np.sort(scores)[k - 1])


def collect_scores(parquet_root: Path, seed: int) -> tuple[dict[str, list[np.ndarray]], int]:
    """Nonconformity scores grouped by forecast date -- the exchangeable unit.

    Pooling every (ticker, step) residual would give ~16,000 numbers and a spuriously
    tight quantile. They are not exchangeable: 59 tickers on the same date share a market,
    and 30 steps of one forecast share a path. The unit conformal can actually treat as
    exchangeable here is the forecast date, which is the same argument report section 17d
    makes about the block count.
    """
    engine = RandomWalkDrift()
    by_date: dict[str, list[np.ndarray]] = {}
    tickers = 0

    for part in sorted(Path(parquet_root).glob("ticker=*/data.parquet")):
        df = pd.read_parquet(part).sort_values("timestamps").reset_index(drop=True)
        ts = pd.to_datetime(df["timestamps"])
        idx = np.flatnonzero((ts >= VAL_START) & (ts <= VAL_END))
        if idx.size == 0:
            continue
        used = False
        for start in range(idx[0], idx[-1] + 1, STRIDE):
            if start < LOOKBACK or start + HORIZON > len(df):
                continue
            bars = df.iloc[:start]
            actual = df["close"].to_numpy()[start : start + HORIZON]
            if len(bars) < LOOKBACK or actual.size < HORIZON:
                continue

            paths = engine.forecast(bars, horizon=HORIZON, m=ENSEMBLE_SIZE, seed=seed)
            tail = (1 - LEVEL) / 2
            lo = np.quantile(paths, tail, axis=0, method=QUANTILE_METHOD)
            hi = np.quantile(paths, 1 - tail, axis=0, method=QUANTILE_METHOD)
            centre = np.quantile(paths, 0.5, axis=0, method=QUANTILE_METHOD)
            half = np.maximum((hi - lo) / 2.0, 1e-9)

            date = pd.Timestamp(ts.iloc[start]).strftime("%Y-%m-%d")
            by_date.setdefault(date, []).append(np.abs(actual - centre) / half)
            used = True
        tickers += int(used)

    if not by_date:
        raise SystemExit("no calibration windows found in the 2024 split")
    return by_date, tickers


def date_level_scores(by_date: dict[str, list[np.ndarray]]) -> np.ndarray:
    """One score per date: the half-width multiplier that covers LEVEL within that date."""
    dates = sorted(by_date)
    return np.array([np.quantile(np.concatenate(by_date[d]), LEVEL) for d in dates])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=REPO_ROOT / "data" / "parquet")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    by_date, n_tickers = collect_scores(args.parquet, args.seed)
    per_date = date_level_scores(by_date)
    scale = conformal_quantile(per_date, alpha=1 - LEVEL)

    pooled = np.concatenate([a for arrs in by_date.values() for a in arrs])
    payload = {
        "level": LEVEL,
        "scale_50": scale,
        "n_calibration_dates": int(per_date.size),
        "n_tickers": n_tickers,
        "n_pooled_residuals": int(pooled.size),
        "pooled_scale_for_comparison": float(np.quantile(pooled, LEVEL)),
        "dates": sorted(by_date),
        "split": [VAL_START, VAL_END],
        "engine": "rw_drift",
        "quantile_method": QUANTILE_METHOD,
        "note": (
            "Finite-sample split conformal over forecast dates, which is the exchangeable "
            "unit: tickers share a market within a date and steps share a path within a "
            "forecast, so the ~16k pooled residuals are not 16k independent points. "
            "50% is the only level this geometry supports (report section 17d); 80% and "
            "90% are served by ACI."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"scale_50 = {scale:.4f}   from {per_date.size} calibration dates")
    print(f"  {n_tickers} tickers, {pooled.size} pooled residuals behind those dates")
    print(
        f"  pooled quantile would have said {payload['pooled_scale_for_comparison']:.4f} "
        f"-- reported for comparison, not used"
    )
    print("  a scale above 1 means the raw 50% band was too narrow on held-out data")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
