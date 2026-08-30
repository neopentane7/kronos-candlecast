"""Write a synthetic ``ensembles.npz`` so the toolkit can be verified without a corpus.

Three forecasters, whose true calibration is known by construction:

``oracle``      draws from exactly the law that generated the outcome. Correctly
                calibrated by definition, so its measured coverage is a test of the
                *instrument*, not of a model: it has to land on nominal.
``too_tight``   the same draws with the horizon-dependent spread the zero-shot Kronos
                study found -- roughly right one step out, roughly half as wide thirty
                steps out (Fact of Record F2). This is what under-propagation looks like
                when you already know it is there.
``flat``        every member equal to the last close. Zero spread, near-zero coverage.
                Present because a degenerate arm is the cheapest check that a metric is
                pointing the right way.

Outcomes carry a shared market factor within each forecast date, which is the whole
reason ``block_ids`` exists: 40 tickers on one date are not 40 independent observations,
and a confidence interval computed as though they were is wrong by a factor you cannot
see from the number itself.

    python make_example.py                 # writes example_ensembles.npz
    python kronos_calibrate.py example_ensembles.npz --baseline oracle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_TICKERS = 40
N_DATES = 12
HORIZON = 30
LOOKBACK = 400
M = 30

DAILY_VOL = 0.018
MARKET_SHARE = 0.45  # fraction of each day's variance that is common across tickers


def build(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = N_TICKERS * N_DATES

    # Per-ticker volatility, so the ATR terciles below mean something.
    ticker_vol = DAILY_VOL * np.exp(rng.normal(0.0, 0.35, size=N_TICKERS))

    history = np.empty((n, LOOKBACK))
    future = np.empty((n, HORIZON))
    block_ids = np.empty((n, HORIZON), dtype=np.int64)
    tickers = np.empty(n, dtype="<U10")
    start_dates = np.empty(n, dtype="<U10")
    sigma = np.empty(n)

    # One market path per forecast date, shared by every ticker priced on that date.
    market = rng.normal(0.0, np.sqrt(MARKET_SHARE) * DAILY_VOL, size=(N_DATES, HORIZON))
    dates = [f"2025-{1 + d // 2:02d}-{1 + 14 * (d % 2):02d}" for d in range(N_DATES)]

    i = 0
    for d in range(N_DATES):
        for t in range(N_TICKERS):
            s = ticker_vol[t]
            sigma[i] = s
            start = 100.0 * np.exp(rng.normal(0.0, 0.5))
            history[i] = start * np.exp(np.cumsum(rng.normal(0.0, s, size=LOOKBACK)))

            idio = rng.normal(0.0, np.sqrt(1.0 - MARKET_SHARE) * s, size=HORIZON)
            scaled_market = market[d] * (s / DAILY_VOL)
            future[i] = history[i, -1] * np.exp(np.cumsum(scaled_market + idio))

            block_ids[i] = d
            tickers[i] = f"SYN{t:03d}"
            start_dates[i] = dates[d]
            i += 1

    last = history[:, -1]
    steps = np.arange(1, HORIZON + 1)

    # The oracle knows the generating law: same random walk, same sigma, drawn fresh.
    shocks = rng.normal(0.0, 1.0, size=(n, HORIZON, M)) * sigma[:, None, None]
    oracle = last[:, None, None] * np.exp(np.cumsum(shocks, axis=1))

    # Under-propagation: the per-step spread is shrunk by a factor falling from ~0.91 at
    # h=1 to ~0.48 at h=30, applied around each window's own median path.
    shrink = np.interp(steps, [1, HORIZON], [0.91, 0.48])
    centre = np.median(oracle, axis=-1, keepdims=True)
    too_tight = centre + (oracle - centre) * shrink[None, :, None]

    flat = np.repeat(last[:, None, None], HORIZON, axis=1).repeat(M, axis=2)

    # Terciles of realized history volatility, the stratifier the Mondrian arms use.
    realized = np.diff(np.log(history), axis=1)[:, -60:].std(axis=1)
    cuts = np.quantile(realized, [1 / 3, 2 / 3])
    terciles = np.digitize(realized, cuts)

    return {
        "y_close": future,
        "history_close": history,
        "atr_pct": realized,
        "atr_tercile": terciles.astype(np.int64),
        "block_ids": block_ids,
        "tickers": tickers,
        "start_dates": start_dates,
        "ens__oracle": oracle.astype(np.float32),
        "ens__too_tight": too_tight.astype(np.float32),
        "ens__flat": flat.astype(np.float32),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("example_ensembles.npz"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    data = build(args.seed)
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(
        f"  {data['y_close'].shape[0]} windows x {HORIZON} steps, m={M}, "
        f"{N_DATES} forecast dates, {N_TICKERS} tickers"
    )
    print("\nnext:")
    print(f"  python kronos_calibrate.py {args.out.name} --baseline oracle")
    print("\nThe oracle is calibrated by construction, so its measured coverage is a")
    print("check on the estimator. Anything but ~0.80 at the 80% level is the tool's")
    print("fault, not the forecaster's -- see README.md, 'Why weibull quantiles'.")

    # The instrument finding, run rather than asserted. Same ensemble, same outcomes,
    # same nominal level; only the quantile estimator changes.
    obs, ens = data["y_close"], data["ens__oracle"].astype(float)
    print("\nthe estimator artifact, on this calibrated ensemble (m=30, nominal 0.80):")
    covs = {}
    for method in ("weibull", "linear"):
        lo = np.quantile(ens, 0.10, axis=-1, method=method)
        hi = np.quantile(ens, 0.90, axis=-1, method=method)
        covs[method] = float(((obs >= lo) & (obs <= hi)).mean())
        note = "  <- numpy default" if method == "linear" else ""
        print(f"  {method:>8}: {covs[method]:.4f}{note}")
    print(
        f"  difference: {(covs['weibull'] - covs['linear']) * 100:.2f} pp of coverage, from"
        " the estimator alone. Phase A's acceptance band was +/-2pp."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
