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
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))
sys.path.insert(0, str(REPO_ROOT / "phase-a" / "Kronos"))

from eval.analysis import spread_ratio_by_horizon  # noqa: E402
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


def write_progress(run_dir: Path | None, stage: str, done: int, total: int) -> None:
    """Heartbeat for ``watch_run.py``.

    A long run's only progress signal was a carriage-returned stdout line, which is
    invisible to anything but the terminal that launched it. This lets a status window
    attached to the run directory report real progress instead of an estimate.
    """
    if run_dir is None:
        return
    with contextlib.suppress(OSError):  # never let telemetry break the run
        (run_dir / "progress.json").write_text(
            json.dumps({"stage": stage, "done": done, "total": total, "at": time.time()}, indent=2),
            encoding="utf-8",
        )


def kronos_ensemble(
    sampler,
    grid,
    sample_count,
    top_p,
    temperature,
    seed,
    batch_size,
    run_dir=None,
    stage="kronos",
    checkpoint_every=10,
):
    """Sampled close-price paths for the whole grid, shaped (n_windows, horizon, m).

    Checkpoints *within* the stage. The grid stage is the multi-hour one, and this machine
    has twice lost its GPU part-way through it; stage-boundary checkpointing alone would
    still discard every completed window. Partial results are flushed every
    ``checkpoint_every`` batches and reloaded on restart.

    Resume is exact rather than approximate: each batch is seeded with ``seed + start``,
    which depends only on its own offset, so skipping completed batches reproduces the
    same draws a clean run would have made.
    """
    from eval.sampler import FEATURE_COLUMNS

    close_idx = FEATURE_COLUMNS.index("close")
    partial_path = (run_dir / f"partial_{stage}.npy") if run_dir else None
    meta_path = (run_dir / f"partial_{stage}.meta.json") if run_dir else None

    # Every field here changes which draws a window receives, so resuming with any of
    # them altered would splice two different forecasters together. batch_size is the
    # subtle one: seeds are `seed + start`, and start depends on the batch boundaries,
    # so changing the batch size silently reseeds every remaining window.
    signature = {
        "batch_size": batch_size,
        "sample_count": sample_count,
        "top_p": top_p,
        "temperature": temperature,
        "seed": seed,
        "n_windows": len(grid),
        "horizon": int(grid.y_close.shape[1]),
    }

    done = 0
    chunks = []
    if partial_path is not None and partial_path.exists():
        prior = None
        with contextlib.suppress(Exception):
            candidate = np.load(partial_path)
            if candidate.ndim == 3 and candidate.shape[0] <= len(grid):
                prior = candidate

        if prior is not None:
            old = {}
            if meta_path is not None and meta_path.exists():
                with contextlib.suppress(Exception):
                    old = json.loads(meta_path.read_text(encoding="utf-8"))
            changed = {k: (old.get(k), v) for k, v in signature.items() if old.get(k) != v}
            if old and changed:
                raise SystemExit(
                    "cannot resume: the run parameters changed since the partial was "
                    "written, so the remaining windows would be drawn differently.\n  "
                    + "\n  ".join(
                        f"{k}: was {was!r}, now {now!r}" for k, (was, now) in changed.items()
                    )
                    + f"\n\nEither restore the original settings or delete {partial_path.name} "
                    "and start the stage over."
                )
            chunks = [prior]
            done = prior.shape[0]
            print(f"    resuming from {done}/{len(grid)} completed windows")

    for start in range(0, len(grid), batch_size):
        stop = min(start + batch_size, len(grid))
        if stop <= done:
            continue  # already on disk from an earlier attempt
        # Built one batch at a time; holding all of them is what drove the machine
        # into paging on the full grid.
        df_list, x_ts, y_ts = grid.batch(start, stop)
        paths = sampler.sample(
            df_list,
            x_ts,
            y_ts,
            pred_len=grid.y_close.shape[1],
            T=temperature,
            top_k=0,
            top_p=top_p,
            sample_count=sample_count,
            seed=seed + start,
        )
        chunks.append(paths[:, :, :, close_idx])  # (n, m, horizon)
        print(f"    windows {stop}/{len(grid)}", end="\r", flush=True)
        write_progress(run_dir, stage, stop, len(grid))

        batches_done = (stop - 1) // batch_size + 1
        if partial_path is not None and batches_done % checkpoint_every == 0:
            with contextlib.suppress(OSError):
                np.save(partial_path, np.concatenate(chunks, axis=0))
                meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    print()
    full = np.concatenate(chunks, axis=0)
    if partial_path is not None:
        # Stage complete; the partial and its signature are now redundant.
        for path in (partial_path, meta_path):
            with contextlib.suppress(OSError):
                path.unlink()
    return full.transpose(0, 2, 1)  # -> (n, horizon, m)


# Milestone A4. Every measurement to date was taken at T = 1.0. The diagnosed defect --
# per-step token distributions mildly too concentrated, compounding through the
# autoregressive loop into a 0.909 -> 0.481 spread-ratio collapse by h = 30 -- is exactly
# what temperature scales. This axis asks whether sampling policy alone repairs it.
TEMPERATURE_SWEEP = (1.0, 1.1, 1.3, 1.5)

# Pre-registered 2026-08-09, before the run. At or above this at h = 30, the compression is
# substantially a sampler artifact and the A6 pilot bar becomes "beat the reference on fair
# CRPS". Below it, the compression is attributed provisionally to the tokenizer and
# normalization, and becomes a documented limitation.
SPREAD_RATIO_TARGET = 0.85


def sweep_temperature(sampler, grid, args, run_dir):
    """Score every temperature arm on one identical block-balanced panel.

    T = 1.0 is re-run here rather than compared against the full-grid aggregate. Different
    windows give a different level, so a sweep arm and a grid row are not comparable
    quantities; the only honest comparison is arms against arms on the same windows.
    """
    panel = grid.subsample_by_block(args.sweep_size, seed=args.seed)
    print(
        f"\n=== temperature sweep on {len(panel)} windows "
        f"({panel.meta['n_tickers']} tickers x {panel.meta['n_blocks']} blocks) ==="
    )
    print(f"    panel: {', '.join(panel.meta['panel_tickers'])}")

    rows, ensembles = [], {}
    for temperature in TEMPERATURE_SWEEP:
        t0 = time.perf_counter()
        ens = kronos_ensemble(
            sampler,
            panel,
            args.sample_count,
            PRODUCTION_TOP_P,
            temperature,
            args.seed,
            args.batch_size,
            run_dir=run_dir,
            stage=f"sweep_T_{temperature}",
            checkpoint_every=args.checkpoint_every,
        )
        ensembles[f"T_{temperature}"] = ens
        s = summarize(panel.y_close, ens, panel.block_ids, seed=args.seed)
        disp = spread_ratio_by_horizon(panel.y_close, ens, panel.history_close)
        rows.append(
            {
                "temperature": temperature,
                "crps_fair": s["crps"],
                "crps_naive": s["crps_naive"],
                "crps_ratio": s["crps_naive"] / s["crps"],
                "interval_score_80": s["interval_score_80"],
                "coverage_50": s["coverage"]["50"]["empirical"],
                "coverage_80": s["coverage"]["80"]["empirical"],
                "coverage_80_ci95": s["coverage"]["80"]["ci95"],
                "coverage_90": s["coverage"]["90"]["empirical"],
                "mape": s["point"]["mape"],
                "spread_ratio_h1": disp["ratio_h1"],
                "spread_ratio_h15": disp["ratio_hmid"],
                "spread_ratio_h30": disp["ratio_hmax"],
                "spread_ratio_by_step": disp["ratio_by_step"],
                "wall_seconds": round(time.perf_counter() - t0, 1),
            }
        )
        r = rows[-1]
        print(
            f"  T={temperature:<4} crps={r['crps_fair']:8.3f}  cov@80={r['coverage_80']:.4f}  "
            f"spread h1/h30={r['spread_ratio_h1']:.3f}/{r['spread_ratio_h30']:.3f}"
        )

    # Pre-registered rule: best fair CRPS wins, coverage@80 breaks ties. Deliberately not
    # the best spread ratio -- widening the cone with junk mass would win that and lose this.
    best = min(rows, key=lambda r: (round(r["crps_fair"], 6), -r["coverage_80"]))
    reached = [r for r in rows if r["spread_ratio_h30"] >= SPREAD_RATIO_TARGET]
    baseline = next(r for r in rows if r["temperature"] == 1.0)

    verdict = {
        "reference_temperature": best["temperature"],
        "reference_crps_fair": best["crps_fair"],
        "spread_ratio_target": SPREAD_RATIO_TARGET,
        "target_reached_by": [r["temperature"] for r in reached],
        "branch": ("sampler_artifact" if reached else "tokenizer_limitation"),
        "crps_change_vs_T1": best["crps_fair"] - baseline["crps_fair"],
        "spread_h30_change_vs_T1": best["spread_ratio_h30"] - baseline["spread_ratio_h30"],
        "tradeoff_present": bool(
            max(r["spread_ratio_h30"] for r in rows) > baseline["spread_ratio_h30"]
            and min(r["crps_fair"] for r in rows) >= baseline["crps_fair"]
        ),
    }
    return rows, ensembles, panel, verdict


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
            run_dir=run_dir,
            stage=f"sweep_top_p_{top_p}",
            checkpoint_every=args.checkpoint_every,
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
    already finished and sitting in memory --Â a completed 45-window grid in one case. The
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
    ap.add_argument(
        "--sweep-temperature",
        action="store_true",
        help="A4: temperature arms on a block-balanced panel; skips the full grid",
    )
    ap.add_argument("--sweep-size", type=int, default=60)
    ap.add_argument("--skip-model", action="store_true", help="baselines only")
    ap.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="flush partial results every N batches; lower it on unreliable hardware",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue an interrupted run directory instead of starting a new one",
    )
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

    if args.resume:
        if not args.resume.exists():
            print(f"cannot resume: {args.resume} does not exist")
            return 1
        run_dir = args.resume
        print(f"resuming into {run_dir}")
    else:
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

    if args.sweep_temperature:
        # A4 runs the sweep alone. The full-grid model stage is 88 minutes and would
        # measure nothing new; the baselines do not depend on temperature at all, so their
        # full-grid numbers stand and no baseline arm belongs in this table.
        from eval.sampler import KronosSampler
        from model import Kronos, KronosTokenizer

        print(f"\nloading {MODEL_ID} ...")
        sampler = KronosSampler(
            Kronos.from_pretrained(MODEL_ID),
            KronosTokenizer.from_pretrained(TOKENIZER_ID),
            max_context=512,
        )
        payload["run"] = "A4_temperature_sweep"
        payload["config"]["device"] = str(sampler.device)
        payload["config"]["top_p"] = PRODUCTION_TOP_P

        rows, sweep_ens, panel, verdict = sweep_temperature(sampler, grid, args, run_dir)
        payload["temperature_sweep"] = rows
        payload["temperature_sweep_panel"] = panel.meta
        payload["decision_rule"] = {
            "fixed": "2026-08-09, before the run",
            "reference_selected_by": "lowest fair CRPS; coverage@80 breaks ties",
            "spread_ratio_target_at_hmax": SPREAD_RATIO_TARGET,
            **verdict,
        }
        save_artifacts(run_dir, panel, sweep_ens, filename="ensembles_sweep.npz")
        payload["artifacts"] = {"ensembles": "ensembles_sweep.npz"}

        if not args.no_figures:
            from eval.figures import spread_ratio_sweep

            fig_path = spread_ratio_sweep(
                rows,
                run_dir / "spread_ratio_vs_temperature.png",
                target=SPREAD_RATIO_TARGET,
                subtitle=(
                    f"{len(panel)} windows - {panel.meta['n_tickers']} tickers x "
                    f"{panel.meta['n_blocks']} blocks - m={args.sample_count} - "
                    f"top_p={PRODUCTION_TOP_P}"
                ),
            )
            payload["artifacts"]["figures"] = [fig_path.name]
            print(f"\nfigure: {fig_path.name}")

        results_path = write_results(run_dir, payload)
        print("\n--- temperature sweep ---")
        print(
            f"{'T':>5}{'CRPS fair':>11}{'CRPS naive':>12}{'IS@80':>10}{'cov@50':>8}"
            f"{'cov@80':>8}{'cov@90':>8}{'MAPE':>8}{'h=1':>7}{'h=15':>7}{'h=30':>7}"
        )
        for r in rows:
            mark = " *" if r["temperature"] == verdict["reference_temperature"] else "  "
            print(
                f"{r['temperature']:>5}{r['crps_fair']:>11.3f}{r['crps_naive']:>12.3f}"
                f"{r['interval_score_80']:>10.2f}{r['coverage_50']:>8.4f}{r['coverage_80']:>8.4f}"
                f"{r['coverage_90']:>8.4f}{r['mape']:>8.4f}{r['spread_ratio_h1']:>7.3f}"
                f"{r['spread_ratio_h15']:>7.3f}{r['spread_ratio_h30']:>7.3f}{mark}"
            )
        print(f"\n* reference config: T = {verdict['reference_temperature']}")
        print(
            f"branch fired: {verdict['branch']} "
            f"(target {SPREAD_RATIO_TARGET} reached by {verdict['target_reached_by'] or 'no arm'})"
        )
        print(f"results: {results_path}")
        print(f"\n{DISCLAIMER}")
        return 0

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
            run_dir=run_dir,
            stage="kronos_zeroshot",
            checkpoint_every=args.checkpoint_every,
        )
        wall = time.perf_counter() - t0

        ensembles["kronos_zeroshot"] = ens
        payload["models"]["kronos_zeroshot"] = summarize(
            grid.y_close, ens, grid.block_ids, seed=args.seed
        )
        payload["slices"] = regime_slices(grid, ens, args.seed)
        # The upstream paper's own metric. Reported alongside CRPS so a poor probabilistic
        # score is not mistaken for a refutation of a cross-sectional ranking claim.
        from eval.analysis import cross_sectional_ic

        payload["cross_sectional_ic"] = {
            name: cross_sectional_ic(
                grid.y_close,
                e,
                grid.history_close,
                np.array([str(d.date()) for d in grid.start_dates]),
            )
            for name, e in {**ensembles, "kronos_zeroshot": ens}.items()
        }
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
            f"{args.split} split - {len(grid)} windows - {len(set(grid.start_dates))} blocks - "
            f"m={args.sample_count} - {QUANTILE_METHOD} quantiles"
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
