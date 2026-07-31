"""A1 smoke test: zero-shot Kronos-small forecast on NSE daily bars, on GPU.

This is our overlay of upstream ``examples/prediction_example.py`` (working rule 5 -
the upstream clone is read-only). It differs from upstream deliberately:

* upstream reads ``examples/data/XSHG_5min_600977.csv``, which is not shipped in the
  repo, so we pull real NSE daily bars instead;
* future timestamps come from the ``XNSE`` exchange calendar (hard constraint 5),
  never ``bdate_range``;
* ``amount`` is the ``close * volume`` proxy (hard constraint 4);
* it saves a figure instead of blocking on ``plt.show()``, and records peak VRAM
  and wall-clock into a rule-9 results directory.

Usage (PowerShell):
    uv run python phase-a/scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a" / "Kronos"))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

from common.calendar import (  # noqa: E402
    NSE_CALENDAR_CODE,
    future_sessions,
    sessions_in_range,
)
from common.results import DISCLAIMER, new_run_dir, write_results  # noqa: E402

TICKER = "RELIANCE.NS"
LOOKBACK = 400  # hard constraint 2
PRED_LEN = 30  # project horizon (contract v2)
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-small"


def fetch_bars(ticker: str, n_bars: int) -> tuple[pd.DataFrame, str]:
    """Return the last ``n_bars`` daily bars in canonical schema, plus the source used.

    Falls back to a seeded synthetic series so the smoke test still exercises the
    model when the machine is offline. The source is recorded in results.json so a
    synthetic run can never be mistaken for a real one.
    """
    try:
        import yfinance as yf

        raw = yf.Ticker(ticker).history(period="5y", auto_adjust=False)
        if len(raw) < n_bars:
            raise ValueError(f"only {len(raw)} bars returned, need {n_bars}")
        raw = raw.tail(n_bars)
        df = pd.DataFrame(
            {
                "timestamps": pd.to_datetime(raw.index).tz_localize(None),
                "open": raw["Open"].to_numpy(dtype=float),
                "high": raw["High"].to_numpy(dtype=float),
                "low": raw["Low"].to_numpy(dtype=float),
                "close": raw["Close"].to_numpy(dtype=float),
                "volume": raw["Volume"].to_numpy(dtype=float),
            }
        ).reset_index(drop=True)
        source = "yfinance"
    except Exception as exc:  # noqa: BLE001 - offline fallback is the whole point
        print(f"[warn] yfinance fetch failed ({exc}); using synthetic bars.")
        rng = np.random.default_rng(0)
        sessions = sessions_in_range("2020-01-01", pd.Timestamp.today().normalize())[-n_bars:]
        close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(sessions))))
        spread = close * rng.uniform(0.002, 0.02, len(sessions))
        df = pd.DataFrame(
            {
                "timestamps": sessions,
                "open": close + rng.normal(0, 1, len(sessions)) * spread * 0.3,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": rng.uniform(1e6, 5e6, len(sessions)),
            }
        )
        source = "synthetic"

    # Hard constraint 4: `amount` is the close * volume proxy, everywhere.
    df["amount"] = df["close"] * df["volume"]
    return df, source


def plot(hist: pd.DataFrame, pred: pd.DataFrame, out: Path, ticker: str, source: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    tail = hist.tail(120)

    ax1.plot(tail["timestamps"], tail["close"], color="#1f77b4", lw=1.4, label="History")
    ax1.plot(pred.index, pred["close"], color="#d62728", lw=1.4, label="Zero-shot forecast")
    ax1.axvline(hist["timestamps"].iloc[-1], color="grey", ls=":", lw=1)
    ax1.set_ylabel("Close")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(tail["timestamps"], tail["volume"], color="#1f77b4", lw=1.2, label="History")
    ax2.plot(pred.index, pred["volume"], color="#d62728", lw=1.2, label="Zero-shot forecast")
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"{ticker} - Kronos-small zero-shot, {PRED_LEN} {NSE_CALENDAR_CODE} sessions "
        f"({source} data)"
    )
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=TICKER)
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu; default auto-detect")
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | available={cuda_ok}")
    if cuda_ok:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df, source = fetch_bars(args.ticker, LOOKBACK)
    print(
        f"Loaded {len(df)} bars from {source}: "
        f"{df['timestamps'].iloc[0].date()} -> {df['timestamps'].iloc[-1].date()}"
    )

    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df["timestamps"]
    # Kronos calls `.dt` on the timestamp arguments, so they must be Series, not Index.
    y_timestamp = pd.Series(future_sessions(df["timestamps"].iloc[-1], PRED_LEN), name="timestamps")
    print(f"Forecast sessions: {y_timestamp.iloc[0].date()} -> {y_timestamp.iloc[-1].date()}")

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Predictor device: {predictor.device} | Kronos-small params: {n_params / 1e6:.1f}M")

    if cuda_ok:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )
    if cuda_ok:
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    peak_alloc_mb = torch.cuda.max_memory_allocated() / 2**20 if cuda_ok else 0.0
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 2**20 if cuda_ok else 0.0

    print("\nForecast head:")
    print(pred_df.head())

    run_dir = new_run_dir()
    fig_path = run_dir / "smoke_forecast.png"
    plot(df, pred_df, fig_path, args.ticker, source)

    vram_ok = cuda_ok and peak_alloc_mb < 2048
    payload = {
        "run": "A1_smoke_test",
        "ticker": args.ticker,
        "data_source": source,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_ok,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_ok else None,
        "device_used": str(predictor.device),
        "model": MODEL_ID,
        "tokenizer": TOKENIZER_ID,
        "model_params_millions": round(n_params / 1e6, 2),
        "lookback": LOOKBACK,
        "pred_len": PRED_LEN,
        "calendar": NSE_CALENDAR_CODE,
        "history_range": [str(df["timestamps"].iloc[0]), str(df["timestamps"].iloc[-1])],
        "forecast_range": [str(y_timestamp.iloc[0]), str(y_timestamp.iloc[-1])],
        "inference_wall_seconds": round(wall_s, 2),
        "peak_vram_allocated_mb": round(peak_alloc_mb, 1),
        "peak_vram_reserved_mb": round(peak_reserved_mb, 1),
        "last_close": float(df["close"].iloc[-1]),
        "forecast_close": [float(v) for v in pred_df["close"]],
        "accept_cuda_available": cuda_ok,
        "accept_forecast_rendered": fig_path.exists(),
        "accept_peak_vram_under_2gb": vram_ok,
    }
    results_path = write_results(run_dir, payload)

    print("\n--- A1 acceptance ---")
    print(f"CUDA available .............. {'PASS' if cuda_ok else 'FAIL'}")
    print(f"Forecast renders ............ {'PASS' if fig_path.exists() else 'FAIL'} ({fig_path})")
    print(
        f"Peak VRAM < 2GB ............. {'PASS' if vram_ok else 'FAIL'} "
        f"(allocated {peak_alloc_mb:.1f} MB, reserved {peak_reserved_mb:.1f} MB)"
    )
    print(f"Inference wall-clock ........ {wall_s:.2f}s")
    print(f"Results ..................... {results_path}")
    print(f"\n{DISCLAIMER}")
    return 0 if (cuda_ok and fig_path.exists() and vram_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
