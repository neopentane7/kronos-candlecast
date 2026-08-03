"""A3 evaluation harness: one command, one rule-9 results directory.

Runs the baselines and zero-shot Kronos over a rolling window grid and writes every
metric, the sampling-policy sweep, and the regime slices to
``results/<timestamp>_<git-sha>/``.

Two measurement decisions are load-bearing and are recorded in every results file:

* quantile estimator -- Weibull plotting positions, without which a perfectly calibrated
  30-member ensemble measures ~0.75 at nominal 0.80;
* CRPS estimator -- Ferro's fair form, without which the score is inflated by ~1/(m-1).

Both the corrected and naive values are emitted so a reader can see the size of each
correction rather than taking it on trust.

Usage (PowerShell):
    uv run python phase-a/eval/calibrate.py --split test
    uv run python phase-a/eval/calibrate.py --split test --limit 60 --sweep-top-p
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))
sys.path.insert(0, str(REPO_ROOT / "phase-a" / "Kronos"))

from eval.baselines import build_baselines  # noqa: E402
from eval.metrics import (  # noqa: E402
    CRPS_ESTIMATOR,
    QUANTILE_METHOD,
    crps,
    interval_score,
    min_samples_for_band,
    summarize,
)
from eval.windows import HORIZON, LOOKBACK, STRIDE, build_grid, load_corpus  # noqa: E402

from common.results import DISCLAIMER, new_run_dir, write_results  # noqa: E402

PARQUET_ROOT = REPO_ROOT / "data" / "parquet"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-small"

# Nucleus sampling truncates the token distribution at every autoregressive step, so a
# lower top_p yields a narrower ensemble and under-coverage that no quantile estimator
# can repair. This axis measures that on real data; the production value is chosen from
# the resulting table, not by default.
TOP_P_SWEEP = (0.9, 0.99, 1.0)

# Production sampling policy, chosen from run 20260801T061559Z_7c8f164-dirty
# (test split, 15 stratified windows, 12 blocks, m=30, T=1.0, top_k=0):
#
#   top_p   coverage@80   rel. width   CRPS (fair)
#   0.90       0.4689       0.1094        56.15
#   0.99       0.4689       0.1233        60.46
#   1.00       0.4933       0.1284        60.63
#
# Truncation is real but is NOT the binding constraint. Removing it entirely widens the
# bands 17.4% while buying only 2.4pp of coverage -- inside the noise at 12 blocks -- and
# CRPS gets worse, not better. Closing the remaining gap from ~0.49 to 0.80 would need
# bands roughly 1.9x wider (z from 0.66 to 1.28 under a normal), which is an order of
# magnitude more than the sampling policy can supply. The under-coverage is therefore
# dominated by location error, not by band width.
#
# So: keep top_p at the specified 0.9, which also scores best on CRPS, and leave the
# widening to the conformal layer, which is what it is for.
PRODUCTION_TOP_P = 0.9


def kronos_ensemble(sampler, grid, sample_count, top_p, temperature, seed, batch_size):
    """Sampled close-price paths for the whole grid, shaped (n_windows, horizon, m)."""
    from eval.sampler import FEATURE_COLUMNS

    close_idx = FEATURE_COLUMNS.index("close")
    chunks = []
    for start in range(0, len(grid), batch_size):
        stop = min(start + batch_size, len(grid))
        paths = sampler.sample(
            grid.x_frames[start:stop],
            grid.x_timestamps[start:stop],
            grid.y_timestamps[start:stop],
            pred_len=grid.y_close.shape[1],
            T=temperature,
            top_k=0,
            top_p=top_p,
            sample_count=sample_count,
            seed=seed + start,
        )
        chunks.append(paths[:, :, :, close_idx])  # (n, m, horizon)
        print(f"    windows {stop}/{len(grid)}", end="\r", flush=True)
    print()
    return np.concatenate(chunks, axis=0).transpose(0, 2, 1)  # -> (n, horizon, m)


def sweep_top_p(sampler, grid, args, run_dir, payload=None):
    """Quantify the nucleus-truncation mechanism on a stratified subsample.

    Returns ``(rows, ensembles, subgrid)`` so the arms can be compared with a *paired*
    bootstrap later -- they share windows, so unpaired intervals would overstate the
    uncertainty of their differences.
    """
    sub = grid.subsample(args.sweep_size, seed=args.seed)
    print(f"\n=== sampling-policy sweep on {len(sub)} stratified windows ===")
    rows = []
    sweep_ens: dict[str, np.ndarray] = {}
    for top_p in TOP_P_SWEEP:
        t0 = time.perf_counter()
        ens = kronos_ensemble(
            sampler,
            sub,
            args.sample_count,
            top_p,
            args.temperature,
            args.seed,
            args.batch_size,
        )
        sweep_ens[f"top_p_{top_p}"] = ens
        s = summarize(sub.y_close, ens, sub.block_ids, seed=args.seed)
        lower = np.quantile(ens, 0.10, axis=-1, method=QUANTILE_METHOD)
        upper = np.quantile(ens, 0.90, axis=-1, method=QUANTILE_METHOD)
        cov = s["coverage"]["80"]
        rows.append(
            {
                "top_p": top_p,
                "coverage_80": cov["empirical"],
                "coverage_80_ci95": cov["ci95"],
                "ci_width": cov["ci95"][1] - cov["ci95"][0],
                "mean_interval_width": float(np.mean(upper - lower)),
                "mean_relative_width": float(np.mean((upper - lower) / sub.y_close)),
                "crps_fair": s["crps"],
                "crps_naive": s["crps_naive"],
                "interval_score_80": s["interval_score_80"],
                "wall_seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(
            f"  top_p={top_p:<5} coverage@80={cov['empirical']:.4f} "
            f"width={rows[-1]['mean_relative_width']:.4f} crps={s['crps']:.4f}"
        )
        # Each arm is ~9 minutes; checkpoint so a loss costs one arm, not all three.
        if payload is not None:
            payload["sampling_policy_sweep"] = rows
            payload["sweep_subsample_of"] = len(grid)
            checkpoint(
                run_dir,
                sub,
                sweep_ens,
                payload,
                f"sweep_top_p_{top_p}",
                filename="ensembles_sweep.npz",
            )
    return rows, sweep_ens, sub


def save_artifacts(
    run_dir: Path,
    grid,
    ensembles: dict[str, np.ndarray],
    extra=None,
    filename: str = "ensembles.npz",
) -> Path:
    """Persist the ensembles and grid alignment so the run can be re-analysed offline.

    Without this, every follow-up question about a run -- re-centering, regime controls,
    paired bootstraps, dispersion ratios -- needs the GPU again. Metrics alone are not a
    reproducible artifact; the forecasts are. Gitignored (``results/**/*.npz``) because
    they are derived data, and the run directory's sha ties them to the code that made
    them.
    """
    path = run_dir / filename
    payload = {
        "y_close": grid.y_close,
        "history_close": grid.history_close,
        "atr_pct": grid.atr_pct,
        "atr_tercile": grid.atr_tercile(),
        "block_ids": grid.block_ids,
        "tickers": np.array(grid.tickers),
        "start_dates": np.array([str(d.date()) for d in grid.start_dates]),
    }
    for name, ens in ensembles.items():
        payload[f"ens__{name}"] = ens.astype(np.float32)
    for name, arr in (extra or {}).items():
        payload[name] = arr
    np.savez_compressed(path, **payload)
    return path


def checkpoint(run_dir: Path, grid, ensembles, payload: dict, label: str, filename="ensembles.npz"):
    """Flush everything computed so far to disk.

    Long GPU runs on this hardware have twice died mid-flight and lost work that was
    already finished and sitting in memory — a completed 45-window grid in one case. The
    harness previously wrote nothing until the very end, so any failure cost the whole
    run. Checkpointing after each stage turns that into the loss of one stage.
    """
    save_artifacts(run_dir, grid, ensembles, filename=filename)
    payload["last_checkpoint"] = label
    write_results(run_dir, payload)
    print(f"    [checkpoint: {label}]", flush=True)


def regime_slices(grid, ens, seed) -> dict:
    """Metrics split by volatility regime and by horizon step."""
    out = {"by_atr_tercile": {}, "by_horizon_step": {}}
    terciles = grid.atr_tercile()
    for t, name in enumerate(("calm", "mid", "volatile")):
        idx = np.flatnonzero(terciles == t)
        if len(idx) == 0:
            continue
        out["by_atr_tercile"][name] = {
            "n_windows": int(len(idx)),
            **summarize(grid.y_close[idx], ens[idx], grid.block_ids[idx], seed=seed),
        }
    for step in (1, 5, 10, 20, grid.y_close.shape[1]):
        k = step - 1
        out["by_horizon_step"][str(step)] = {
            "crps_fair": float(np.mean(crps(grid.y_close[:, k : k + 1], ens[:, k : k + 1, :]))),
            "interval_score_80": float(
                np.mean(interval_score(grid.y_close[:, k : k + 1], ens[:, k : k + 1, :], 0.80))
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="subsample the grid")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--sample-count", type=int, default=30)
    ap.add_argument("--top-p", type=float, default=PRODUCTION_TOP_P)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sweep-top-p", action="store_true")
    ap.add_argument("--sweep-size", type=int, default=60)
    ap.add_argument("--skip-model", action="store_true", help="baselines only")
    ap.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    args = ap.parse_args()

    # Fail before spending GPU time if the requested bands are unconstructible.
    for level in (0.50, 0.80, 0.90):
        need = min_samples_for_band(1 - level)
        if args.sample_count < need:
            print(
                f"sample_count={args.sample_count} cannot express a {level:.0%} band "
                f"(needs {need})."
            )
            return 1

    if not PARQUET_ROOT.exists():
        print(f"No corpus at {PARQUET_ROOT}. Run fetch_nse.py first.")
        return 1

    # Long runs are usually watched through a pipe, where Python buffers stdout and the
    # job looks hung. Reconfigure once, here, rather than sprinkling flush=True.
    sys.stdout.reconfigure(line_buffering=True)

    print(f"loading corpus and building the {args.split} grid ...")
    grid = build_grid(load_corpus(PARQUET_ROOT), args.split, stride=args.stride)
    if len(grid) == 0:
        print("empty grid")
        return 1
    if args.limit:
        grid = grid.subsample(args.limit, seed=args.seed)
    print(
        f"grid: {len(grid)} windows, {grid.meta['n_tickers']} tickers, "
        f"{len(set(grid.start_dates))} distinct start dates"
    )

    run_dir = new_run_dir()
    payload = {
        "run": "A3_zero_shot_gate",
        "split": args.split,
        "grid": {**grid.meta, "n_windows_evaluated": len(grid)},
        "config": {
            "model": MODEL_ID,
            "tokenizer": TOKENIZER_ID,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "stride": args.stride,
            "sample_count": args.sample_count,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": 0,
            "seed": args.seed,
        },
        "measurement": {
            "quantile_method": QUANTILE_METHOD,
            "crps_estimator": CRPS_ESTIMATOR,
            "note": (
                "Coverage and CRPS are both ensemble-size dependent. Weibull plotting "
                "positions and Ferro's fair CRPS remove that dependence; naive values "
                "are reported alongside so the size of each correction is visible."
            ),
        },
        "models": {},
    }

    print("\n=== baselines ===")
    ensembles: dict[str, np.ndarray] = {}
    for name, ens in build_baselines(
        grid.history_close, grid.y_close.shape[1], args.sample_count, args.seed
    ).items():
        ensembles[name] = ens
        payload["models"][name] = summarize(grid.y_close, ens, grid.block_ids, seed=args.seed)
        m = payload["models"][name]
        print(f"  {name:<20} crps={m['crps']:.4f} cov@80={m['coverage']['80']['empirical']:.4f}")
    checkpoint(run_dir, grid, ensembles, payload, "baselines")

    if not args.skip_model:
        import torch
        from eval.sampler import KronosSampler
        from model import Kronos, KronosTokenizer

        print(f"\nloading {MODEL_ID} ...")
        sampler = KronosSampler(
            Kronos.from_pretrained(MODEL_ID),
            KronosTokenizer.from_pretrained(TOKENIZER_ID),
            max_context=512,
        )
        payload["config"]["device"] = str(sampler.device)

        print(f"\n=== zero-shot Kronos ({len(grid)} windows) ===")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ens = kronos_ensemble(
            sampler,
            grid,
            args.sample_count,
            args.top_p,
            args.temperature,
            args.seed,
            args.batch_size,
        )
        wall = time.perf_counter() - t0

        ensembles["kronos_zeroshot"] = ens
        payload["models"]["kronos_zeroshot"] = summarize(
            grid.y_close, ens, grid.block_ids, seed=args.seed
        )
        payload["slices"] = regime_slices(grid, ens, args.seed)
        payload["wall_clock"] = {
            "kronos_grid_seconds": round(wall, 1),
            "seconds_per_window": round(wall / len(grid), 3),
            "peak_vram_mb": (
                round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available()
                else None
            ),
        }
        m = payload["models"]["kronos_zeroshot"]
        print(
            f"  kronos_zeroshot      crps={m['crps']:.4f} "
            f"cov@80={m['coverage']['80']['empirical']:.4f} ({wall:.0f}s)"
        )
        # The expensive stage is done. Make it survivable before starting the sweep --
        # this is exactly the work a mid-run GPU loss discarded twice.
        checkpoint(run_dir, grid, ensembles, payload, "kronos_zeroshot")

        if args.sweep_top_p:
            rows, sweep_ens, sub = sweep_top_p(sampler, grid, args, run_dir, payload)
            payload["sampling_policy_sweep"] = rows

    artifact_path = save_artifacts(run_dir, grid, ensembles)
    payload["artifacts"] = {
        "ensembles": artifact_path.name,
        "note": "forecast paths and grid alignment, for offline re-analysis without a GPU",
    }

    if not args.no_figures:
        from eval.figures import write_all

        subtitle = (
            f"{args.split} split · {len(grid)} windows · {len(set(grid.start_dates))} blocks · "
            f"m={args.sample_count} · {QUANTILE_METHOD} quantiles"
        )
        figures = write_all(
            grid.y_close,
            ensembles,
            grid.block_ids,
            run_dir,
            terciles=grid.atr_tercile(),
            seed=args.seed,
            subtitle=subtitle,
        )
        payload["artifacts"]["figures"] = [p.name for p in figures]
        print(f"\nfigures: {len(figures)} written to {run_dir}")

    results_path = write_results(run_dir, payload)

    print("\n--- summary ---")
    print(f"{'model':<22} {'CRPS(fair)':>11} {'cov@80':>8} {'ci95':>18} {'IS@80':>10}")
    for name, m in payload["models"].items():
        c = m["coverage"]["80"]
        print(
            f"{name:<22} {m['crps']:>11.4f} {c['empirical']:>8.4f} "
            f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] {m['interval_score_80']:>10.3f}"
        )
    print(f"\neffective blocks: {payload['models']['last_value']['effective_blocks']}")
    print(f"results: {results_path}")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
