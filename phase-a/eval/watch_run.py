"""Live status window for a running evaluation.

Reads a run directory rather than the harness's stdout, so it works on a job started in
another terminal, in the background, or by someone else. Safe to start and stop at any
time -- it only reads.

Usage (PowerShell):
    uv run python phase-a/eval/watch_run.py                # latest run, refresh every 30s
    uv run python phase-a/eval/watch_run.py --once         # print one panel and exit
    uv run python phase-a/eval/watch_run.py --run results/20260803T... --every 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

# Measured on this machine at batch 8, m=30, lookback 400, horizon 30. Used only for the
# fallback estimate when a run predates the progress heartbeat.
SECONDS_PER_WINDOW = 30.2

STAGES = ["baselines", "kronos_zeroshot", "sweep_top_p_0.9", "sweep_top_p_0.99", "sweep_top_p_1.0"]


def latest_run() -> Path | None:
    dirs = [p for p in RESULTS.glob("*") if p.is_dir()]
    return max(dirs, key=lambda p: p.name) if dirs else None


def gpu_state() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if out.returncode != 0:
            return "unavailable — " + (out.stderr or out.stdout).strip().splitlines()[0][:60]
        t, util, used, total, watt = (v.strip() for v in out.stdout.strip().split(","))
        pct = 100 * float(used) / float(total)
        warn = "  <-- tight" if pct > 92 else ""
        return f"{t}C  {util}% util  {used}/{total} MiB ({pct:.0f}%){warn}  {watt} W"
    except Exception as exc:  # noqa: BLE001 - a watcher must never crash the terminal
        return f"unavailable ({type(exc).__name__})"


def human(seconds: float) -> str:
    if seconds < 0:
        return "--"
    return str(timedelta(seconds=int(seconds)))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a half-written checkpoint is expected, not an error
        return {}


def panel(run: Path) -> str:
    results = read_json(run / "results.json")
    progress = read_json(run / "progress.json")
    started = datetime.fromtimestamp(run.stat().st_ctime)
    elapsed = (datetime.now() - started).total_seconds()

    lines = [
        "=" * 74,
        f"  {run.name}",
        f"  started {started:%H:%M:%S}   elapsed {human(elapsed)}   "
        f"refreshed {datetime.now():%H:%M:%S}",
        "=" * 74,
        "",
        f"  GPU        {gpu_state()}",
    ]

    grid = results.get("grid", {})
    total = progress.get("total") or grid.get("n_windows_evaluated") or grid.get("n_windows")
    if grid:
        lines.append(
            f"  grid       {total} windows · {grid.get('n_tickers')} tickers · "
            f"{grid.get('n_distinct_start_dates')} blocks"
        )

    done = progress.get("done")
    if done and total:
        frac = done / total
        rate = elapsed / done if done else SECONDS_PER_WINDOW
        remaining = (total - done) * rate
        bar = "#" * int(30 * frac) + "." * (30 - int(30 * frac))
        lines += [
            f"  stage      {progress.get('stage', '?')}",
            f"  progress   [{bar}] {done}/{total} ({frac:.0%})",
            f"  rate       {rate:.1f} s/window",
            f"  eta        {human(remaining)}  ->  "
            f"{(datetime.now() + timedelta(seconds=remaining)):%H:%M:%S}",
        ]
    elif total:
        est = total * SECONDS_PER_WINDOW
        lines += [
            "  progress   (no heartbeat — this run predates progress.json)",
            f"  estimate   {human(est)} total at {SECONDS_PER_WINDOW} s/window, "
            f"{human(max(0, est - elapsed))} remaining",
        ]

    ck = results.get("last_checkpoint")
    if ck:
        marks = (
            " ".join(
                ("[x]" if STAGES.index(s) <= STAGES.index(ck) else "[ ]") + s
                for s in STAGES
                if s in STAGES
            )
            if ck in STAGES
            else ck
        )
        lines += ["", f"  checkpoint {ck}", f"             {marks}"]

    models = results.get("models", {})
    if models:
        lines += ["", "  " + f"{'model':<22}{'CRPS(fair)':>12}{'cov@80':>9}{'IS@80':>11}"]
        for name, m in models.items():
            cov = m.get("coverage", {}).get("80", {}).get("empirical")
            lines.append(
                f"  {name:<22}{m.get('crps', float('nan')):>12.4f}"
                f"{cov if cov is not None else float('nan'):>9.4f}"
                f"{m.get('interval_score_80', float('nan')):>11.3f}"
            )

    ic = results.get("cross_sectional_ic", {}).get("kronos_zeroshot")
    if ic:
        lines += ["", "  cross-sectional IC (paper reports IC 0.043 / RankIC 0.025)"]
        for step, v in ic.items():
            if v.get("ic") is None:
                continue
            lines.append(
                f"    h={step:<3} IC {v['ic']:+.4f}  RankIC {v['rank_ic']:+.4f}  "
                f"({v['n_dates_used']}/{v['n_dates_total']} dates)"
            )

    figs = sorted(p.name for p in run.glob("*.png"))
    npz = sorted(p.name for p in run.glob("*.npz"))
    lines += ["", f"  artifacts  {len(npz)} npz, {len(figs)} figures"]
    if results.get("wall_clock"):
        lines.append(
            f"  FINISHED   {results['wall_clock'].get('kronos_grid_seconds')}s on the grid"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--every", type=int, default=30, help="refresh seconds")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run = args.run or latest_run()
    if run is None or not run.exists():
        print("no run directory found")
        return 1

    try:
        while True:
            text = panel(run)
            if args.once:
                print(text)
                return 0
            os.system("cls" if os.name == "nt" else "clear")  # noqa: S605 - display only
            print(text)
            print("  Ctrl-C to stop watching (the run is unaffected)")
            if read_json(run / "results.json").get("wall_clock"):
                print("  run complete.")
                return 0
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nstopped watching; the run continues.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
