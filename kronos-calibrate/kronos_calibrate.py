"""kronos-calibrate: score any probabilistic forecaster the way Phase A scored Kronos.

The toolkit is deliberately model-agnostic. It never loads a model, never calls a
sampler, and does not know what produced the numbers it reads. It takes one file --
sampled forecast paths plus the outcomes they were trying to predict -- and returns the
calibration picture, using the estimators Phase A had to fix before any of its own
measurements meant anything.

**Two corrections you inherit by using this rather than rolling your own.**

*Finite-ensemble coverage bias.* Form an interval from the empirical quantiles of an
m-member ensemble with NumPy's default estimator and a perfectly calibrated forecaster
measures **5.04 points below nominal at m = 30**. The shortfall depends only on m and the
estimator, not on the model. Weibull plotting positions (Hyndman-Fan type 6) place the
p-th quantile at p(m+1), which is what the order-statistic rank argument implies, and the
bias disappears. Phase A's acceptance band was +/-2pp; the artifact was 2.5x that band.

*CRPS estimator bias.* The naive ensemble CRPS is inflated by roughly 1/(m-1). Ferro's
fair form removes it. Both values are reported side by side here so the size of the
correction is visible instead of taken on trust.

**And one thing it will refuse to let you claim.** Coverage is a mean over *independent
forecast dates*, not over windows: 59 tickers on one date share a market and 30 horizon
steps share a path. Every interval here is a block bootstrap over ``block_ids``, and the
block count is reported next to every figure, because a coverage number without its
effective sample size is not a measurement.

Usage:
    python kronos_calibrate.py path/to/ensembles.npz
    python kronos_calibrate.py path/to/ensembles.npz --json out.json

See README.md for the file format. If you can write that file from your own model, every
number below is available to you.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The implementations live in the evaluation harness they were developed in, so there is
# exactly one copy of each and the tests that pin them keep pinning them. This module is
# the documented entry point, not a second implementation.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "phase-a"))

from eval.analysis import (  # noqa: E402
    horizon_exponent,
    paired_block_bootstrap_diff,
    spread_ratio_by_horizon,
    tercile_table,
    z_ratio_table,
)
from eval.conformal import (  # noqa: E402
    apply_scale,
    conformal_quantile,
    coverage_of,
    fit_scale,
    mean_width,
)
from eval.metrics import (  # noqa: E402
    CRPS_ESTIMATOR,
    QUANTILE_METHOD,
    band_bounds,
    block_bootstrap_ci,
    coverage_indicator,
    crps,
    effective_sample_size,
    interval_score,
    min_samples_for_band,
    summarize,
)

__all__ = [
    "CRPS_ESTIMATOR",
    "QUANTILE_METHOD",
    "Ensembles",
    "apply_scale",
    "band_bounds",
    "block_bootstrap_ci",
    "conformal_quantile",
    "coverage_indicator",
    "coverage_of",
    "crps",
    "effective_sample_size",
    "feasibility",
    "fit_scale",
    "horizon_exponent",
    "interval_score",
    "mean_width",
    "min_samples_for_band",
    "paired_block_bootstrap_diff",
    "report",
    "spread_ratio_by_horizon",
    "summarize",
    "tercile_table",
    "z_ratio_table",
]

REQUIRED = ("y_close", "block_ids")
OPTIONAL = ("history_close", "atr_tercile", "tickers", "start_dates")


class Ensembles:
    """A validated view over one ``ensembles.npz``.

    Validation is not politeness. A file with mismatched shapes still produces numbers,
    and they are wrong in ways no downstream check would notice, so the shapes are
    asserted once here rather than trusted at twenty call sites.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        d = np.load(self.path, allow_pickle=False)
        missing = [k for k in REQUIRED if k not in d.files]
        if missing:
            raise ValueError(f"{self.path.name} is missing required arrays: {missing}")

        self.obs = np.asarray(d["y_close"], dtype=float)
        self.block_ids = np.asarray(d["block_ids"])
        self.models = {
            k[len("ens__") :]: np.asarray(d[k], dtype=float)
            for k in d.files
            if k.startswith("ens__")
        }
        if not self.models:
            raise ValueError(
                f"{self.path.name} contains no ens__* arrays. At least one forecaster's "
                "sampled paths are needed; see README.md."
            )

        self.history = (
            np.asarray(d["history_close"], dtype=float) if "history_close" in d.files else None
        )
        self.terciles = np.asarray(d["atr_tercile"]) if "atr_tercile" in d.files else None

        n, h = self.obs.shape
        if self.block_ids.shape != (n, h):
            raise ValueError(f"block_ids must be {(n, h)}, got {self.block_ids.shape}")
        for name, ens in self.models.items():
            if ens.shape[:2] != (n, h):
                raise ValueError(
                    f"ens__{name} must be (n_windows, horizon, m) = ({n}, {h}, m), got {ens.shape}"
                )

        self.n_windows, self.horizon = n, h
        self.m = next(iter(self.models.values())).shape[2]
        self.blocks = effective_sample_size(self.block_ids)

    def __repr__(self) -> str:
        return (
            f"Ensembles({self.path.name}: {self.n_windows} windows x {self.horizon} steps, "
            f"m={self.m}, {self.blocks} blocks, models={sorted(self.models)})"
        )


def feasibility(blocks: int, levels=(0.50, 0.80, 0.90)) -> dict:
    """Which nominal levels a split-conformal study can actually support here.

    Split conformal at level 1-alpha needs ceil((2-alpha)/alpha) exchangeable calibration
    residuals, and the same again to test the result on. The unit is the forecast date,
    not the window. On daily bars at a 30-step horizon this bites hard: Phase A's corpus
    had 12 independent dates against the 18 an 80% band needs.
    """
    out = {"blocks_available": blocks, "levels": {}}
    for lvl in levels:
        floor = min_samples_for_band(round(1 - lvl, 10))
        out["levels"][f"{int(lvl * 100)}"] = {
            "residuals_needed_per_side": floor,
            "blocks_needed": 2 * floor,
            "feasible": bool(blocks >= 2 * floor),
        }
    return out


def report(ens: Ensembles, seed: int = 0, baseline: str | None = None) -> dict:
    """The whole calibration picture for every forecaster in the file."""
    out = {
        "source": ens.path.name,
        "n_windows": ens.n_windows,
        "horizon": ens.horizon,
        "ensemble_size": ens.m,
        "effective_blocks": ens.blocks,
        "quantile_method": QUANTILE_METHOD,
        "crps_estimator": CRPS_ESTIMATOR,
        "feasibility": feasibility(ens.blocks),
        "models": {},
    }

    for name, paths in sorted(ens.models.items()):
        m = summarize(ens.obs, paths, ens.block_ids, seed=seed)
        m["z_ratio"] = z_ratio_table(ens.obs, paths)
        if ens.history is not None:
            m["spread_ratio"] = spread_ratio_by_horizon(ens.obs, paths, ens.history)
        if ens.terciles is not None:
            m["by_regime"] = tercile_table(ens.obs, paths, ens.terciles, ens.block_ids)
        with np.errstate(all="ignore"):
            m["horizon_exponent"] = horizon_exponent(ens.obs, paths)
        out["models"][name] = m

    # A score is only meaningful against something. If a baseline is named, every other
    # model gets a paired block-bootstrap interval for the CRPS difference -- paired
    # because the arms share windows, and unpaired intervals would overstate the
    # uncertainty of exactly the comparison being made.
    if baseline is not None:
        if baseline not in ens.models:
            raise ValueError(
                f"--baseline {baseline!r} is not in the file; available: {sorted(ens.models)}"
            )
        ref = crps(ens.obs, ens.models[baseline])
        out["paired_vs_baseline"] = {"baseline": baseline, "diffs": {}}
        for name, paths in sorted(ens.models.items()):
            if name == baseline:
                continue
            out["paired_vs_baseline"]["diffs"][name] = paired_block_bootstrap_diff(
                crps(ens.obs, paths), ref, ens.block_ids, seed=seed
            )
    return out


def _print(rep: dict) -> None:
    print(
        f"{rep['source']}: {rep['n_windows']} windows x {rep['horizon']} steps, "
        f"m={rep['ensemble_size']}, {rep['effective_blocks']} independent forecast dates"
    )
    print(f"estimators: {rep['quantile_method']} quantiles, {rep['crps_estimator']} CRPS")

    print("\nsplit-conformal feasibility at this block count")
    for lvl, d in rep["feasibility"]["levels"].items():
        mark = "yes" if d["feasible"] else "NO"
        print(f"  {lvl}%  needs {d['blocks_needed']:>3} blocks  -> {mark}")

    print(f"\n{'model':<24}{'CRPS':>10}{'CRPS naive':>12}{'IS@80':>11}{'cov@80':>9}{'95% CI':>18}")
    for name, m in rep["models"].items():
        c = m["coverage"]["80"]
        ci = f"[{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]"
        print(
            f"{name:<24}{m['crps']:>10.3f}{m['crps_naive']:>12.3f}"
            f"{m['interval_score_80']:>11.2f}{c['empirical']:>9.4f}{ci:>18}"
        )

    print("\nspread ratio (predicted / realized), where history was supplied")
    for name, m in rep["models"].items():
        s = m.get("spread_ratio")
        if s:
            print(f"  {name:<24} h=1 {s['ratio_h1']:.3f}   h=max {s['ratio_hmax']:.3f}")

    if "paired_vs_baseline" in rep:
        b = rep["paired_vs_baseline"]
        print(f"\npaired CRPS difference vs {b['baseline']} (negative = better)")
        for name, d in b["diffs"].items():
            lo, hi = d["ci95"]
            verdict = "indistinguishable" if d["includes_zero"] else "different"
            print(f"  {name:<24} {d['mean_difference']:>9.3f}  [{lo:.3f}, {hi:.3f}]  {verdict}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path, help="an ensembles.npz -- see README.md for the format")
    ap.add_argument("--baseline", default=None, help="model name to compare the others against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the full report here")
    args = ap.parse_args(argv)

    try:
        ens = Ensembles(args.npz)
        rep = report(ens, seed=args.seed, baseline=args.baseline)
    except ValueError as exc:
        # A malformed file or a mistyped arm name is a user error, not a crash. The
        # message already says what is wrong and what the file contains.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(rep)

    if args.json:
        args.json.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwritten: {args.json}")
    print("\nResearch/education tool - scenario visualization, not investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
