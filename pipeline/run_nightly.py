"""The nightly job: fetch, forecast, calibrate, publish, and learn from yesterday.

Runs on a free CPU Actions runner in seconds. There is no model download and no torch on
this path -- the served engine is analytic, because Phase A found the foundation model
loses to it by 81% on fair CRPS (F1) and no sampling change repairs that (F9).

Two modes:

* **live** (default) fetches recent bars from yfinance and forecasts from today.
* **as-of** replays a past session from the local corpus. Used once before first deploy to
  warm the ACI state and pre-fill the archive, so day-one visitors see bands that mean
  something rather than cold-start nominal. Every record it writes is flagged
  ``backfilled: true`` (resolved conflict #10).

Usage (PowerShell):
    uv run python pipeline/run_nightly.py
    uv run python pipeline/run_nightly.py --as-of 2026-06-02 --limit 5
    uv run python pipeline/run_nightly.py --seed-from-corpus 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.calendar import future_sessions  # noqa: E402
from pipeline import archive  # noqa: E402
from pipeline.aci import ACIState  # noqa: E402
from pipeline.calibration import build_bands, enforce_monotone, path_probabilities  # noqa: E402
from pipeline.contract import BandMethods, Forecast, Metadata, Quantiles, RawQuantiles  # noqa: E402
from pipeline.engines import HORIZON, LOOKBACK, SERVING_ENSEMBLE_SIZE, RandomWalkDrift  # noqa: E402

STATE_DIR = REPO_ROOT / "pipeline" / "state"
ACI_PATH = STATE_DIR / "aci_state.json"
SPLIT_PATH = STATE_DIR / "split_conformal.json"
ARCHIVE_ROOT = REPO_ROOT / "pipeline" / "archive_data"
SITE_DATA = REPO_ROOT / "site" / "data"
HISTORY_SESSIONS = 120
IST = "+05:30"


# ---------------------------------------------------------------- data access
def load_corpus_bars(parquet_root: Path) -> dict[str, pd.DataFrame]:
    """Every ticker's canonical bars from the committed corpus (as-of mode)."""
    out: dict[str, pd.DataFrame] = {}
    for part in sorted(Path(parquet_root).glob("ticker=*/data.parquet")):
        ticker = part.parent.name.split("=", 1)[1]
        df = pd.read_parquet(part).sort_values("timestamps").reset_index(drop=True)
        out[ticker] = df
    return out


def fetch_live_bars(ticker: str, sessions: int = LOOKBACK + 60) -> pd.DataFrame:
    """Recent bars from yfinance, canonicalised through the shared preprocessing.

    Uses the same ``common/preprocess`` path as training, which is the whole point of
    hard constraint 6: a serving-only cleaning rule is how train/serve skew starts.
    """
    import yfinance as yf

    from common.preprocess import from_yfinance

    raw = yf.Ticker(f"{ticker}.NS").history(period="3y", auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError("yfinance returned no rows")
    return from_yfinance(raw).tail(sessions).reset_index(drop=True)


# ---------------------------------------------------------------- forecasting
def forecast_one(
    ticker: str,
    bars: pd.DataFrame,
    aci_state: ACIState,
    split_scale: float | None,
    generated_at: str,
    backfilled: bool,
    seed: int,
) -> tuple[Forecast, pd.DataFrame]:
    engine = RandomWalkDrift()
    params = engine.params(bars)
    paths = engine.forecast(bars, horizon=HORIZON, m=SERVING_ENSEMBLE_SIZE, seed=seed)

    bands = enforce_monotone(build_bands(paths, ticker, aci_state, split_scale))
    probs = path_probabilities(paths, params["last_close"], params["sigma_per_session"])

    last_ts = pd.Timestamp(bars["timestamps"].iloc[-1])
    stamps = future_sessions(last_ts, HORIZON)
    iso = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in stamps]

    fc = Forecast(
        ticker=f"{ticker}.NS",
        generated_at=generated_at,
        engine=engine.name,
        calibration="aci",
        challenger=None,
        horizon=HORIZON,
        last_close=params["last_close"],
        backfilled=backfilled,
        timestamps=iso,
        quantiles=Quantiles(
            p10=[float(v) for v in bands["p10"]],
            p25=[float(v) for v in bands["p25"]],
            p50=[float(v) for v in bands["p50"]],
            p75=[float(v) for v in bands["p75"]],
            p90=[float(v) for v in bands["p90"]],
        ),
        raw_quantiles_p10_p90=RawQuantiles(
            p10=[float(v) for v in np.quantile(paths, 0.10, axis=0, method="weibull")],
            p90=[float(v) for v in np.quantile(paths, 0.90, axis=0, method="weibull")],
        ),
        prob_above_last_close=probs["prob_above_last_close"],
        prob_vol_exceeds_recent=probs["prob_vol_exceeds_recent"],
        metadata=Metadata(
            band_methods=BandMethods(band_50=bands["method_50"]),
            aci_gamma=aci_state.gamma,
            aci_provisional=aci_state.provisional,
            ensemble_size=SERVING_ENSEMBLE_SIZE,
            lookback=LOOKBACK,
            engine_validated=engine.validated,
            note=(
                "50% band carries a finite-sample split-conformal guarantee; 80/90 are "
                "ACI and converge in the long run only. See report section 17d."
            ),
        ),
    )

    rows = []
    for level, lo_k, hi_k in (
        (0.50, "lo_50", "hi_50"),
        (0.80, "lo_80", "hi_80"),
        (0.90, "lo_90", "hi_90"),
    ):
        for h in range(HORIZON):
            rows.append(
                {
                    "ticker": f"{ticker}.NS",
                    "forecast_date": last_ts.strftime("%Y-%m-%d"),
                    "target_date": iso[h],
                    "step": h + 1,
                    "level": level,
                    "lo": float(bands[lo_k][h]),
                    "hi": float(bands[hi_k][h]),
                    "p50": float(bands["p50"][h]),
                    "engine": engine.name,
                    "backfilled": backfilled,
                }
            )
    return fc, pd.DataFrame(rows)


def score_and_update(aci_state: ACIState, target_date: str, realized: dict[str, float]) -> dict:
    """Fold every forecast whose horizon lands on ``target_date`` into the ACI state."""
    due = archive.due_for_scoring(ARCHIVE_ROOT, target_date)
    if due.empty:
        return {"scored": 0, "covered": 0, "tickers": 0}

    covered = 0
    for row in due.itertuples():
        actual = realized.get(row.ticker)
        if actual is None:
            continue
        hit = bool(row.lo <= actual <= row.hi)
        covered += int(hit)
        aci_state.update(row.ticker, float(row.level), int(row.step), hit)
    # Pooled coverage across three different nominal levels is not interpretable; the
    # per-level breakdown is what says whether calibration is working.
    per_level = {}
    for lvl, grp in due.groupby("level"):
        actual = grp["ticker"].map(realized)
        hits = (grp["lo"] <= actual) & (actual <= grp["hi"])
        per_level[f"{float(lvl):.2f}"] = {
            "n": int(len(grp)),
            "empirical": round(float(hits.mean()), 4),
            "nominal": float(lvl),
        }
    return {
        "scored": int(len(due)),
        "covered": covered,
        "coverage": round(covered / max(len(due), 1), 4),
        "by_level": per_level,
        "tickers": int(due["ticker"].nunique()),
    }


# ---------------------------------------------------------------- publishing
def write_site_data(
    forecasts: list[Forecast], bars: dict[str, pd.DataFrame], summary: dict
) -> None:
    """Static JSON the PWA reads directly. No backend, per resolved conflict #5."""
    (SITE_DATA / "forecasts").mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "history").mkdir(parents=True, exist_ok=True)

    for fc in forecasts:
        fc.write(SITE_DATA / "forecasts" / f"{fc.ticker}.json")

    for fc in forecasts:
        base = fc.ticker.removesuffix(".NS")
        tail = bars[base].tail(HISTORY_SESSIONS)
        payload = {
            "ticker": fc.ticker,
            "sessions": [
                {
                    "t": pd.Timestamp(r.timestamps).strftime("%Y-%m-%d"),
                    "o": round(float(r.open), 4),
                    "h": round(float(r.high), 4),
                    "l": round(float(r.low), 4),
                    "c": round(float(r.close), 4),
                }
                for r in tail.itertuples()
            ],
        }
        path = SITE_DATA / "history" / f"{fc.ticker}.json"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    index = {
        "generated_at": summary["generated_at"],
        "engine": "rw_drift",
        "contract_version": 3,
        "horizon": HORIZON,
        "backfilled": summary["backfilled"],
        "tickers": sorted(fc.ticker for fc in forecasts),
        "skipped": summary["skipped"],
        "aci": summary["aci"],
        "disclaimer": forecasts[0].disclaimer if forecasts else "",
    }
    (SITE_DATA / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- orchestration
def run(as_of: str | None, limit: int | None, seed: int, backfilled: bool) -> dict:
    t0 = time.perf_counter()
    aci_state = ACIState.load(ACI_PATH)
    split_scale = None
    if SPLIT_PATH.exists():
        split_scale = float(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["scale_50"])

    live = as_of is None
    corpus = {} if live else load_corpus_bars(REPO_ROOT / "data" / "parquet")
    universe = sorted(corpus) if corpus else _live_universe()
    if limit:
        universe = universe[:limit]

    forecasts: list[Forecast] = []
    used_bars: dict[str, pd.DataFrame] = {}
    archive_rows: list[pd.DataFrame] = []
    skipped: list[dict] = []
    realized: dict[str, float] = {}
    forecast_date = None

    for ticker in universe:
        try:
            if live:
                bars = fetch_live_bars(ticker)
            else:
                bars = corpus[ticker]
                bars = bars[bars["timestamps"] <= pd.Timestamp(as_of)].reset_index(drop=True)
            if len(bars) < LOOKBACK:
                raise ValueError(f"only {len(bars)} sessions of history, need {LOOKBACK}")

            stamp = pd.Timestamp(bars["timestamps"].iloc[-1])
            forecast_date = forecast_date or stamp.strftime("%Y-%m-%d")
            generated_at = f"{stamp.strftime('%Y-%m-%d')}T18:00:00{IST}"

            fc, rows = forecast_one(
                ticker, bars, aci_state, split_scale, generated_at, backfilled, seed
            )
            forecasts.append(fc)
            used_bars[ticker] = bars
            archive_rows.append(rows)
            realized[f"{ticker}.NS"] = float(bars["close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001 -- a bad ticker is skipped, never substituted
            skipped.append({"ticker": ticker, "reason": f"{type(exc).__name__}: {exc}"})

    if not forecasts:
        raise SystemExit(f"no ticker produced a forecast; skipped: {skipped}")

    scoring = score_and_update(aci_state, forecast_date, realized)
    aci_state.save(ACI_PATH)
    archive.write_day(ARCHIVE_ROOT, forecast_date, pd.concat(archive_rows, ignore_index=True))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "forecast_date": forecast_date,
        "mode": "as-of" if as_of else "live",
        "backfilled": backfilled,
        "engine": "rw_drift",
        "forecast": len(forecasts),
        "skipped": skipped,
        "aci": {
            "gamma": aci_state.gamma,
            "provisional": aci_state.provisional,
            "updates": aci_state.updates,
            "clamped": aci_state.clamped,
            **scoring,
        },
        "split_conformal_50": split_scale,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    write_site_data(forecasts, used_bars, summary)
    (SITE_DATA / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _live_universe() -> list[str]:
    """The traded universe, read from the Phase A fetch script's constants.

    Read-only: the serving path imports the list rather than keeping a second copy that
    would drift. If the corpus is present locally its ticker directories win, because the
    corpus is what every Phase A number was measured on.
    """
    sys.path.insert(0, str(REPO_ROOT / "phase-a" / "scripts"))
    from fetch_nse import CURRENT_CONSTITUENTS, SURVIVORSHIP_ADDITIONS

    return sorted(set(CURRENT_CONSTITUENTS) | set(SURVIVORSHIP_ADDITIONS))


def seed_from_corpus(sessions: int, limit: int | None, seed: int) -> dict:
    """Warm the ACI state and pre-fill the archive by replaying the last N sessions.

    Cold-starting ACI means day-one visitors get bands at nominal alpha, which for this
    engine is not what the long run converges to -- the site would be showing an
    uncalibrated cone on its first morning and calling it calibrated. Replaying real past
    sessions gives the state something to have learned from.

    Every record written here carries ``backfilled: true`` (resolved conflict #10). The
    archive must never be ambiguous about which forecasts were actually made on the day
    they claim.
    """
    corpus = load_corpus_bars(REPO_ROOT / "data" / "parquet")
    if not corpus:
        raise SystemExit("seeding needs the local corpus at data/parquet")

    # The shared session index: dates the whole universe traded (constraint 5's authority
    # ordering -- the corpus is NSE, the calendar is a neighbouring exchange's guess).
    counts: dict[pd.Timestamp, int] = {}
    for df in corpus.values():
        for ts in df["timestamps"]:
            counts[pd.Timestamp(ts)] = counts.get(pd.Timestamp(ts), 0) + 1
    shared = sorted(d for d, n in counts.items() if n >= len(corpus) / 2)
    replay = shared[-sessions:]

    print(f"seeding ACI over {len(replay)} sessions: {replay[0].date()} -> {replay[-1].date()}")
    last = None
    for i, day in enumerate(replay, 1):
        last = run(day.strftime("%Y-%m-%d"), limit, seed, backfilled=True)
        if i % 20 == 0 or i == len(replay):
            print(f"  [{i:>3}/{len(replay)}] {day.date()}  updates={last['aci']['updates']}")

    cov = archive_coverage(corpus)
    print()
    print("  realized coverage across the seeded archive")
    print(
        f"    {cov.get('matured_rows', 0)} matured rows over "
        f"{cov.get('forecast_dates', 0)} forecast dates"
    )
    for lvl in ("0.50", "0.80", "0.90"):
        d = cov.get(lvl)
        if d:
            print(f"    nominal {lvl}   empirical {d['empirical']:.4f}   ({d['dates']} dates)")
    last["archive_coverage"] = cov
    return last


def archive_coverage(corpus: dict[str, pd.DataFrame]) -> dict:
    """Realized coverage over every matured forecast in the archive.

    A single day's coverage is one observation, not n: the tickers on one date share a
    market and the 30 steps share a path. This aggregates by forecast date -- the unit
    report section 17b argues is the honest denominator -- and reports the date count
    alongside so nobody reads the row count as a sample size.
    """
    df = archive.read_all(ARCHIVE_ROOT)
    if df.empty:
        return {}

    closes = {}
    for tk, bars in corpus.items():
        stamps = pd.to_datetime(bars["timestamps"]).dt.strftime("%Y-%m-%d")
        closes[f"{tk}.NS"] = dict(zip(stamps, bars["close"], strict=True))

    df["actual"] = [
        closes.get(tk, {}).get(d) for tk, d in zip(df["ticker"], df["target_date"], strict=True)
    ]
    matured = df.dropna(subset=["actual"])
    if matured.empty:
        return {}

    out = {
        "matured_rows": int(len(matured)),
        "forecast_dates": int(matured["forecast_date"].nunique()),
    }
    for lvl, grp in matured.groupby("level"):
        hit = (grp["lo"] <= grp["actual"]) & (grp["actual"] <= grp["hi"])
        per_date = hit.groupby(grp["forecast_date"]).mean()
        out[f"{float(lvl):.2f}"] = {
            "nominal": float(lvl),
            "empirical": round(float(per_date.mean()), 4),
            "dates": int(per_date.size),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=None, help="replay a past session from the local corpus")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--seed-from-corpus",
        type=int,
        default=None,
        metavar="N",
        help="replay the last N shared sessions to warm ACI before first deploy",
    )
    ap.add_argument("--backfilled", action="store_true", help="flag output as seeded history")
    args = ap.parse_args()

    if args.seed_from_corpus:
        s = seed_from_corpus(args.seed_from_corpus, args.limit, args.seed)
    else:
        s = run(args.as_of, args.limit, args.seed, args.backfilled or args.as_of is not None)

    print(f"\n=== nightly forecast ({s['mode']}) ===")
    print(f"  forecast date : {s['forecast_date']}")
    print(f"  engine        : {s['engine']}   backfilled={s['backfilled']}")
    print(f"  forecast      : {s['forecast']} tickers")
    print(f"  skipped       : {len(s['skipped'])}")
    for sk in s["skipped"][:10]:
        print(f"      {sk['ticker']:<14} {sk['reason'][:80]}")
    a = s["aci"]
    print(
        f"  ACI           : gamma={a['gamma']} provisional={a['provisional']} "
        f"updates={a['updates']} clamped={a['clamped']}"
    )
    print(f"  scored today  : {a['scored']} rows")
    for lvl, d in sorted(a.get("by_level", {}).items()):
        print(f"      nominal {lvl}  empirical {d['empirical']:.4f}  (n={d['n']})")
    print(f"  split 50%     : {s['split_conformal_50']}")
    print(f"  wall clock    : {s['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
