"""Two offline diagnostics that need no GPU, run against saved ensembles.

Both answer questions that decide what gets written before the full grid runs.

1. ``blocks_per_stratum`` -- is Mondrian conformal *evaluable* at all? The project's own
   effective-sample-size argument says overlapping windows collapse to their distinct
   forecast start dates. Volatility clusters in time, so a volatility stratum's windows
   arrive in temporal clumps that share blocks: a stratum can hold hundreds of windows
   and still rest on a handful of independent ones. Windows-per-stratum is the wrong
   denominator for the band-feasibility floor; blocks-per-stratum is the right one.

2. ``spread_vs_input_vol`` -- *why* is the model's dispersion regime-blind? If per-window
   instance normalization compresses volatility differences at the input, the model cannot
   see the regime and regime-blind dispersion is the predicted symptom rather than a
   coincidence. The test correlates each forecast's predicted spread at h=1 against the
   realized volatility of its own input window, with two controls that make a null result
   interpretable rather than merely absent.

Usage (PowerShell):
    uv run python phase-a/eval/diagnose.py results/<run-dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))

from eval.metrics import QUANTILE_METHOD, min_samples_for_band  # noqa: E402

from common.results import DISCLAIMER, write_results  # noqa: E402

STRATUM_NAMES = {0: "calm", 1: "mid", 2: "volatile"}
VOL_LOOKBACK = 20  # trading days of input used for the realized-vol estimate


def _blocks(block_ids: np.ndarray) -> np.ndarray:
    """One block label per window.

    ``block_ids`` is (n, horizon) but labels the forecast start date, so every step of a
    window carries the same label. Collapsing to one label per window is what makes
    "how many independent units does this stratum have" a well-posed question.
    """
    per_window = block_ids[:, 0]
    if not np.all(block_ids == per_window[:, None]):
        raise ValueError("block_ids vary within a window; the collapse below is invalid")
    return per_window


def blocks_per_stratum(npz, levels=(0.5, 0.8, 0.9)) -> dict:
    """Windows vs *independent blocks* per volatility stratum, against the feasibility floor."""
    tercile = npz["atr_tercile"]
    blocks = _blocks(npz["block_ids"])
    total_blocks = int(len(np.unique(blocks)))

    floors = {f"{int(lvl * 100)}": min_samples_for_band(round(1 - lvl, 10)) for lvl in levels}
    out = {
        "n_windows": int(len(tercile)),
        "effective_blocks_overall": total_blocks,
        "band_feasibility_floor": floors,
        "strata": {},
    }

    for code, name in STRATUM_NAMES.items():
        sel = tercile == code
        n_win = int(sel.sum())
        uniq = np.unique(blocks[sel])
        n_blk = int(len(uniq))
        # How lumpy is the stratum? If volatility clustering is real, a stratum's windows
        # concentrate into few blocks and the per-block counts are large and uneven.
        counts = np.array([int((blocks[sel] == b).sum()) for b in uniq]) if n_blk else np.array([])
        out["strata"][name] = {
            "n_windows": n_win,
            "effective_blocks": n_blk,
            "windows_per_block_max": int(counts.max()) if counts.size else 0,
            "windows_per_block_median": float(np.median(counts)) if counts.size else 0.0,
            "share_of_all_blocks": round(n_blk / total_blocks, 4) if total_blocks else 0.0,
            "feasible_by_windows": {k: bool(n_win >= v) for k, v in floors.items()},
            "feasible_by_blocks": {k: bool(n_blk >= v) for k, v in floors.items()},
        }
    return out


def _chronological_blocks(blocks: np.ndarray, start_dates: np.ndarray) -> np.ndarray:
    """Block labels ordered by date, not by label.

    Block ids come from ``pd.factorize``, which numbers in order of first appearance.
    That is chronological only while every ticker shares one session index. It does not
    here -- see :func:`alignment_report` -- so sorting the labels silently produced a
    "temporal" split that was not temporal. Sort by the date each block represents.
    """
    uniq = np.unique(blocks)
    # start_dates is an array of ISO strings; sort them as text, which is chronological
    # for ISO-8601 and avoids a parse. np.min has no loop for fixed-width unicode.
    first_date = {b: sorted(start_dates[blocks == b].tolist())[0] for b in uniq}
    return np.array(sorted(uniq, key=lambda b: first_date[b]))


def alignment_report(npz) -> dict:
    """Do all tickers share one session index, or does the grid contain orphan dates?

    Windows are enumerated per ticker at a fixed stride. If one ticker's session index is
    offset -- one extra or one missing bar anywhere in its history -- every later window
    it produces lands on a date no other ticker uses. Those dates become their own blocks,
    and because blocks are the unit of the bootstrap, the reported effective sample size
    counts singleton "blocks" belonging to a single ticker as if they were independent
    forecast dates. The confidence intervals and any power calculation built on that count
    are then wrong, in the optimistic direction.
    """
    blocks, dates, tick = _blocks(npz["block_ids"]), npz["start_dates"], npz["tickers"]
    n_tickers = int(len(np.unique(tick)))

    sizes = {b: int((blocks == b).sum()) for b in np.unique(blocks)}
    orphan = {b: n for b, n in sizes.items() if n < n_tickers / 2}
    shared = {b: n for b, n in sizes.items() if n >= n_tickers / 2}

    culprits: dict[str, int] = {}
    for b in orphan:
        for t in np.unique(tick[blocks == b]):
            culprits[str(t)] = culprits.get(str(t), 0) + 1

    return {
        "n_tickers": n_tickers,
        "n_blocks_reported": len(sizes),
        "n_blocks_shared": len(shared),
        "n_blocks_orphan": len(orphan),
        "orphan_dates": sorted(str(dates[blocks == b][0]) for b in orphan),
        "tickers_causing_orphans": culprits,
        "aligned": not orphan,
        "effective_blocks_corrected": len(shared),
        "note": (
            "effective_blocks in results.json counts orphan blocks; the corrected count "
            "is the number of dates the whole universe shares"
        ),
    }


def split_feasibility(npz, level: float = 0.8) -> dict:
    """Per-stratum feasibility of every temporal calibration/test split of the blocks.

    A conformal study needs a calibration set *and* a test set, and both halves have to
    clear the band-feasibility floor in every stratum -- calibration to form the quantile
    at all, test to measure whether it worked. Splitting a grid that is feasible whole can
    leave both halves infeasible.

    This enumerates the splits so the pre-registered split rule can be chosen from what
    the data supports rather than from a round number like 60/40. Blocks are forecast
    start dates, ordered in time, so calibration always precedes test -- a random split
    would leak future information into the calibration quantile.
    """
    tercile, blocks = npz["atr_tercile"], _blocks(npz["block_ids"])
    order = _chronological_blocks(blocks, npz["start_dates"])
    floor = min_samples_for_band(round(1 - level, 10))

    rows = []
    for k in range(1, len(order)):
        cal_blocks, test_blocks = set(order[:k]), set(order[k:])
        in_cal = np.array([b in cal_blocks for b in blocks])
        row = {"n_cal_blocks": k, "n_test_blocks": len(test_blocks), "strata": {}}
        ok = True
        for code, name in STRATUM_NAMES.items():
            sel = tercile == code
            c = int(len(np.unique(blocks[sel & in_cal])))
            t = int(len(np.unique(blocks[sel & ~in_cal])))
            row["strata"][name] = {"cal_blocks": c, "test_blocks": t}
            ok = ok and c >= floor and t >= floor
        row["feasible"] = ok
        rows.append(row)

    workable = [r for r in rows if r["feasible"]]
    return {
        "level": level,
        "floor_blocks_per_side": floor,
        "n_blocks": int(len(order)),
        "splits": rows,
        "n_feasible_splits": len(workable),
        "feasible_cal_block_counts": [r["n_cal_blocks"] for r in workable],
        "verdict": (
            "no temporal split leaves both halves above the floor in every stratum"
            if not workable
            else f"{len(workable)} of {len(rows)} splits are workable"
        ),
    }


def feasibility_horizon(npz, levels=(0.5, 0.8, 0.9), stride: int = 30) -> dict:
    """How much data a split-conformal study needs, in years, at each nominal level.

    "12 dates exist, 18 are required" invites the obvious rebuttal: lengthen the test
    window. This converts the block count into the quantity that rebuttal has to argue
    with -- calendar years of data consumed by the conformal split alone -- so the
    infeasibility reads as a trade-off with a price attached rather than an accident of
    how this particular split was drawn.

    Sessions per year are measured from the grid rather than assumed: consecutive blocks
    are ``stride`` sessions apart, so the calendar span between the first and last block
    fixes the conversion.
    """
    blocks = _blocks(npz["block_ids"])
    dates = npz["start_dates"]
    order = _chronological_blocks(blocks, dates)
    first = np.datetime64(sorted(dates[blocks == order[0]].tolist())[0])
    last = np.datetime64(sorted(dates[blocks == order[-1]].tolist())[0])

    n_blocks = len(order)
    span_days = int((last - first).astype("timedelta64[D]").astype(int))
    sessions_spanned = (n_blocks - 1) * stride
    sessions_per_year = (
        sessions_spanned / span_days * 365.25 if span_days and n_blocks > 1 else float("nan")
    )

    rows = {}
    for lvl in levels:
        floor = min_samples_for_band(round(1 - lvl, 10))
        need_blocks = 2 * floor  # calibration side and test side
        need_sessions = need_blocks * stride
        rows[f"{int(lvl * 100)}"] = {
            "floor_per_side": floor,
            "blocks_needed": need_blocks,
            "sessions_needed": need_sessions,
            "years_needed": round(need_sessions / sessions_per_year, 2),
            "feasible_here": bool(n_blocks >= need_blocks),
        }

    return {
        "stride": stride,
        "n_blocks_available": n_blocks,
        "span_days": span_days,
        "sessions_per_year": round(sessions_per_year, 1),
        "years_available": round(span_days / 365.25, 2),
        "levels": rows,
    }


def _realized_vol(prices: np.ndarray, lookback: int = VOL_LOOKBACK) -> np.ndarray:
    """Std of daily log returns over the trailing ``lookback`` days of each row."""
    tail = prices[:, -(lookback + 1) :]
    return np.std(np.diff(np.log(tail), axis=1), axis=1, ddof=1)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def _elasticity(spread: np.ndarray, vol: np.ndarray) -> float:
    """OLS slope of log(spread) on log(vol) -- how hard the forecaster responds to regime.

    Rank correlation answers "does it see volatility at all"; this answers "by how much
    does it move when volatility doubles". They come apart, and the difference is the
    whole diagnosis: a normalizer that compresses rather than erases the input signal
    leaves rank correlation intact while driving this slope toward zero.

    A slope of 1 means proportional response. The reference is not 1 in the abstract but
    the slope realized volatility itself exhibits over the same windows.
    """
    x, y = np.log(vol), np.log(spread)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x**2).sum())


def _block_bootstrap(fn, a, b, blocks, n_boot=2000, seed=0) -> tuple[float, float]:
    """Percentile CI for any two-sample statistic, resampling whole blocks.

    45 windows are not 45 independent observations; they are 12 blocks. Resampling
    windows would report an interval several times too narrow.
    """
    uniq = np.unique(blocks)
    groups = [np.flatnonzero(blocks == u) for u in uniq]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(a[idx])) < 3:  # a degenerate resample carries no information
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            v = fn(a[idx], b[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def spread_vs_input_vol(npz, model_keys=None, lookback=VOL_LOOKBACK, seed=0) -> dict:
    """Does predicted h=1 spread track the input window's realized volatility?

    Three quantities, and only the comparison between them is informative:

    * **the model under test** -- Kronos;
    * **a positive control** -- RW-drift, whose spread is a deterministic function of the
      input vol by construction. It must correlate near 1. If it does not, the wiring is
      wrong and the Kronos number means nothing;
    * **a ceiling** -- realized *future* vol against realized *input* vol. Volatility
      persistence is what a model could in principle exploit, so this is the target a
      regime-aware forecaster would approach.
    """
    hist, y = npz["history_close"], npz["y_close"]
    blocks = _blocks(npz["block_ids"])
    last = hist[:, -1]

    input_vol = _realized_vol(hist, lookback)
    future_vol = np.std(np.diff(np.log(np.column_stack([last, y])), axis=1), axis=1, ddof=1)

    if model_keys is None:
        model_keys = [k[len("ens__") :] for k in npz.files if k.startswith("ens__")]

    out = {
        "vol_lookback_days": lookback,
        "n_windows": int(len(input_vol)),
        "effective_blocks": int(len(np.unique(blocks))),
        "quantile_method": QUANTILE_METHOD,
        "models": {},
    }

    for name in model_keys:
        ens = npz[f"ens__{name}"][:, 0, :]  # h = 1 only
        lo = np.quantile(ens, 0.10, axis=-1, method=QUANTILE_METHOD)
        hi = np.quantile(ens, 0.90, axis=-1, method=QUANTILE_METHOD)
        spread = (hi - lo) / last  # relative, so cross-ticker price levels do not drive it
        if np.allclose(spread, spread[0]):
            out["models"][name] = {"degenerate": True, "relative_spread_h1": float(spread[0])}
            continue
        rho = _spearman(spread, input_vol)
        ci = _block_bootstrap(_spearman, spread, input_vol, blocks, seed=seed)
        beta = _elasticity(spread, input_vol)
        bci = _block_bootstrap(_elasticity, spread, input_vol, blocks, seed=seed)
        out["models"][name] = {
            "spearman_spread_vs_input_vol": round(rho, 4),
            "ci95_block_bootstrap": [round(ci[0], 4), round(ci[1], 4)],
            "elasticity_log_spread_on_log_vol": round(beta, 4),
            "elasticity_ci95": [round(bci[0], 4), round(bci[1], 4)],
            "mean_relative_spread_h1": float(np.mean(spread)),
            "spread_dispersion_cv": float(np.std(spread) / np.mean(spread)),
        }

    rho_ceiling = _spearman(future_vol, input_vol)
    beta_ceiling = _elasticity(future_vol, input_vol)
    out["ceiling_future_vol_vs_input_vol"] = {
        "spearman": round(rho_ceiling, 4),
        "ci95_block_bootstrap": [
            round(v, 4)
            for v in _block_bootstrap(_spearman, future_vol, input_vol, blocks, seed=seed)
        ],
        "elasticity_log_spread_on_log_vol": round(beta_ceiling, 4),
        "elasticity_ci95": [
            round(v, 4)
            for v in _block_bootstrap(_elasticity, future_vol, input_vol, blocks, seed=seed)
        ],
        "note": "volatility persistence -- what a regime-aware forecaster could exploit",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="a results/ directory containing ensembles.npz")
    ap.add_argument("--lookback", type=int, default=VOL_LOOKBACK)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz_path = args.run_dir / "ensembles.npz"
    if not npz_path.exists():
        print(f"no ensembles.npz in {args.run_dir}")
        return 1
    npz = np.load(npz_path)

    strata = blocks_per_stratum(npz)
    print(f"\n=== blocks per stratum ({strata['n_windows']} windows) ===")
    print(f"effective blocks overall: {strata['effective_blocks_overall']}")
    print(f"feasibility floor (residuals needed): {strata['band_feasibility_floor']}")
    hdr = f"{'stratum':<10}{'windows':>9}{'blocks':>8}{'w/blk':>7}"
    hdr += "".join(f"{'80%blk':>9}{'90%blk':>9}")
    print("\n" + hdr)
    for name, s in strata["strata"].items():
        print(
            f"{name:<10}{s['n_windows']:>9}{s['effective_blocks']:>8}"
            f"{s['windows_per_block_max']:>7}"
            f"{str(s['feasible_by_blocks']['80']):>9}"
            f"{str(s['feasible_by_blocks']['90']):>9}"
        )

    align = alignment_report(npz)
    print("\n=== grid alignment ===")
    print(
        f"tickers={align['n_tickers']}  blocks reported={align['n_blocks_reported']}  "
        f"shared={align['n_blocks_shared']}  orphan={align['n_blocks_orphan']}"
    )
    if not align["aligned"]:
        print(f"  orphan dates: {align['orphan_dates']}")
        print(f"  caused by:    {align['tickers_causing_orphans']}")
        print(f"  CORRECTED effective blocks: {align['effective_blocks_corrected']}")

    feas = split_feasibility(npz, level=0.8)
    floor = feas["floor_blocks_per_side"]
    print(f"\n=== cal/test split feasibility at 80% (floor {floor} blocks/side) ===")
    print(feas["verdict"])
    for r in feas["splits"]:
        s = r["strata"]
        mark = "ok " if r["feasible"] else "   "
        print(
            f"  {mark}cal={r['n_cal_blocks']:>3} test={r['n_test_blocks']:>3}  "
            + "  ".join(
                f"{n}:{s[n]['cal_blocks']}/{s[n]['test_blocks']}" for n in STRATUM_NAMES.values()
            )
        )

    horizon = feasibility_horizon(npz)
    print(
        f"\n=== data required for a split-conformal study "
        f"(stride {horizon['stride']}, {horizon['sessions_per_year']} sessions/yr) ==="
    )
    print(f"available: {horizon['n_blocks_available']} blocks over {horizon['years_available']} yr")
    print(f"{'level':>7}{'floor/side':>12}{'blocks':>9}{'sessions':>10}{'years':>8}{'here?':>8}")
    for lvl, r in horizon["levels"].items():
        print(
            f"{lvl:>7}{r['floor_per_side']:>12}{r['blocks_needed']:>9}"
            f"{r['sessions_needed']:>10}{r['years_needed']:>8}{str(r['feasible_here']):>8}"
        )

    spread = spread_vs_input_vol(npz, lookback=args.lookback, seed=args.seed)
    print(f"\n=== h=1 predicted spread vs input realized vol ({args.lookback}d) ===")
    for name, m in spread["models"].items():
        if m.get("degenerate"):
            print(f"{name:<22} degenerate (zero spread)")
            continue
        ci, bci = m["ci95_block_bootstrap"], m["elasticity_ci95"]
        print(
            f"{name:<22} rho={m['spearman_spread_vs_input_vol']:>7.4f} "
            f"[{ci[0]:>6.3f},{ci[1]:>6.3f}]   "
            f"beta={m['elasticity_log_spread_on_log_vol']:>7.4f} [{bci[0]:>6.3f},{bci[1]:>6.3f}]"
        )
    c = spread["ceiling_future_vol_vs_input_vol"]
    cc, cb = c["ci95_block_bootstrap"], c["elasticity_ci95"]
    print(
        f"{'[ceiling] future vol':<22} rho={c['spearman']:>7.4f} "
        f"[{cc[0]:>6.3f},{cc[1]:>6.3f}]   "
        f"beta={c['elasticity_log_spread_on_log_vol']:>7.4f} [{cb[0]:>6.3f},{cb[1]:>6.3f}]"
    )

    payload = {
        "run": "A3_diagnostics",
        "source_run": args.run_dir.name,
        "alignment": align,
        "blocks_per_stratum": strata,
        "split_feasibility_80": feas,
        "feasibility_horizon": horizon,
        "spread_vs_input_vol": spread,
    }
    write_results(args.run_dir, payload, filename="diagnostics.json")
    print(f"\nwritten: {args.run_dir / 'diagnostics.json'}")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
