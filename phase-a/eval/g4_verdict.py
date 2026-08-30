"""Evaluate the G4 bar mechanically, against the wording fixed on 2026-08-09.

A bar that is read off a table by a human is not pre-registered in any useful sense: the
reader already knows which answer they want. This computes the verdict from the arrays and
prints PASS or FAIL per condition, so the only judgement left is whether this file
implements the bar — which is auditable in a way "it looks like parity" is not.

**The bar, verbatim.** A6 passes on both:

  (a) fair-CRPS **parity** with conformalized rw_drift — the paired block-bootstrap
      interval for the difference contains zero; and
  (b) the **h=30 dispersion ratio rises materially above 0.481** with the curve
      flattening, not merely shifting.

Regime spread is reported but is **not** a pass condition; R1 made that leg vacuous.

**One ambiguity in (a), surfaced rather than resolved quietly.** `conformal.apply_scale`
returns band bounds, not an ensemble, so "fair CRPS of conformalized rw_drift" names a
quantity that does not exist: CRPS scores a distribution and conformalization at a single
level produces an interval. Two readings were available:

  1. The bar names *which null* — the one Phase B ships — and asks for CRPS parity with
     that forecaster's predictive distribution, which is the rw_drift ensemble. Section 5's
     headline comparison (121.87 against 67.24) was always computed this way.
  2. Build a conformalized ensemble by rescaling members about the median.

**This implements reading 1**, because reading 2 requires inventing a construction after
the bar was fixed, and a bar you reinterpret once you have the data is not a bar. The
choice is recorded in the output so a reader can disagree with it explicitly.

**Operationalising "flattening, not merely shifting" in (b).** A6's dispersion curve must
rise at the far end relative to the near end, not lift uniformly. The measured quantity is
`h30/h1` — precisely the number F9 used to rule temperature out, where every arm raised
h=30 spread from 0.419 to 0.569 while h30/h1 *fell* from 0.407 to 0.363. Zero-shot's
full-grid value is 0.481/0.909 = 0.529, and flattening means exceeding it.

    uv run python phase-a/eval/g4_verdict.py results/<finetuned-test-run> \\
        --val results/<finetuned-val-run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))

from eval.analysis import paired_block_bootstrap_diff, spread_ratio_by_horizon  # noqa: E402
from eval.metrics import coverage_indicator, crps  # noqa: E402

from common.results import write_results  # noqa: E402

BAR_FIXED = "2026-08-09"
NULL = "random_walk_drift"
FINETUNED = "kronos_finetuned"

# Zero-shot reference values from the full grid, run 20260808T161433Z_d6602cd (F2).
ZEROSHOT_H30 = 0.481
ZEROSHOT_H1 = 0.909
ZEROSHOT_FLATNESS = ZEROSHOT_H30 / ZEROSHOT_H1  # 0.529

# F11's period drift, for the readout to sit beside. Not a bar.
F11_ZEROSHOT_DRIFT = 0.1715
F11_NULL_DRIFT = 0.0502


def load(run_dir: Path) -> dict:
    npz = np.load(run_dir / "ensembles.npz", allow_pickle=False)
    models = {
        k[len("ens__") :]: np.asarray(npz[k], float) for k in npz.files if k.startswith("ens__")
    }
    return {
        "obs": np.asarray(npz["y_close"], float),
        "history": np.asarray(npz["history_close"], float),
        "blocks": np.asarray(npz["block_ids"]),
        "models": models,
        "dir": run_dir,
    }


def condition_1(d: dict, seed: int) -> dict:
    """Fair-CRPS parity with the null, paired over shared windows."""
    if FINETUNED not in d["models"]:
        raise SystemExit(
            f"no ens__{FINETUNED} in {d['dir']}. Score the checkpoint with "
            "`calibrate.py --checkpoint <path>`, which renames the model row so a G4 "
            "comparison cannot silently read zero-shot numbers."
        )
    if NULL not in d["models"]:
        raise SystemExit(f"no ens__{NULL} in {d['dir']}; the bar is defined against it")

    ft = crps(d["obs"], d["models"][FINETUNED])
    null = crps(d["obs"], d["models"][NULL])
    diff = paired_block_bootstrap_diff(ft, null, d["blocks"], seed=seed)

    return {
        "statement": "fair-CRPS parity with the null: paired interval contains zero",
        "reading": "CRPS computed against the rw_drift ensemble; see module docstring",
        "crps_finetuned": float(ft.mean()),
        "crps_null": float(null.mean()),
        "mean_difference": diff["mean_difference"],
        "ci95": diff["ci95"],
        "passes": bool(diff["includes_zero"]),
    }


def _bootstrap_dispersion(d: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Block-bootstrap the h=30 spread ratio and the h30/h1 flatness.

    Resampling is over distinct forecast dates, per the metric canon. Both quantities are
    ratios whose denominator is a cross-window statistic, so each replicate recomputes the
    whole ratio rather than perturbing a point estimate.
    """
    obs, ens, hist = d["obs"], d["models"][FINETUNED], d["history"]
    blocks = np.asarray(d["blocks"])
    per_window = blocks[:, 0] if blocks.ndim > 1 else blocks
    unique = np.unique(per_window)
    idx_by_block = [np.flatnonzero(per_window == b) for b in unique]

    rng = np.random.default_rng(seed)
    h30 = np.empty(n_boot)
    flat = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(idx_by_block), size=len(idx_by_block))
        rows = np.concatenate([idx_by_block[j] for j in pick])
        sr = spread_ratio_by_horizon(obs[rows], ens[rows], hist[rows])
        h30[i] = sr["ratio_hmax"]
        flat[i] = sr["ratio_hmax"] / sr["ratio_h1"] if sr["ratio_h1"] else np.nan

    return {
        "h30_ci95": [float(np.quantile(h30, 0.025)), float(np.quantile(h30, 0.975))],
        "flatness_ci95": [float(np.nanquantile(flat, 0.025)), float(np.nanquantile(flat, 0.975))],
    }


def condition_2(d: dict, seed: int = 0) -> dict:
    """Dispersion at h=30 materially above 0.481, with the curve flattening.

    **"Materially" is the load-bearing word and it needs an operationalisation.** A strict
    comparison against 0.481 is vacuous: scored on the zero-shot ensemble itself this
    condition returns h=30 of 0.481 and flatness of 0.530 against a 0.529 reference, and
    passes. A bar that a model already known to fail can clear is not a bar.

    So "materially above" is read as **above the reference by more than the measurement's
    own uncertainty**: the 95% block-bootstrap interval for the quantity must lie entirely
    above the zero-shot value. That is the project's canonical uncertainty machinery
    (constraint 12, block bootstrap over distinct forecast dates) rather than an arbitrary
    margin, and it is recorded here because it was chosen after the bar was written.
    """
    sr = spread_ratio_by_horizon(d["obs"], d["models"][FINETUNED], d["history"])
    h1, h30 = sr["ratio_h1"], sr["ratio_hmax"]
    flatness = h30 / h1 if h1 else float("nan")
    boot = _bootstrap_dispersion(d, seed=seed)

    rose = boot["h30_ci95"][0] > ZEROSHOT_H30
    flattened = boot["flatness_ci95"][0] > ZEROSHOT_FLATNESS
    return {
        "statement": "h=30 dispersion materially above 0.481, curve flattening not shifting",
        "materially": "95% block-bootstrap interval lies entirely above the zero-shot value",
        "ratio_h1": h1,
        "ratio_h30": h30,
        "h30_ci95": boot["h30_ci95"],
        "zeroshot_h30": ZEROSHOT_H30,
        "flatness_h30_over_h1": flatness,
        "flatness_ci95": boot["flatness_ci95"],
        "zeroshot_flatness": ZEROSHOT_FLATNESS,
        "rose_at_h30": bool(rose),
        "curve_flattened": bool(flattened),
        # Both legs are required: F9 showed a curve that shifts without flattening, and
        # that is the failure mode this condition exists to exclude.
        "passes": bool(rose and flattened),
    }


def period_drift_readout(test: dict, val: dict | None) -> dict:
    """Does fine-tuning shrink the F11 period-drift?

    **Explicitly not a pass condition.** The bar was fixed on 2026-08-09 and is not being
    amended after the fact. This is the one quantity A6 can produce that no earlier run
    touched, so it is recorded beside the verdict and kept out of it.
    """
    if val is None:
        return {
            "available": False,
            "why": "no --val run supplied; score the checkpoint on both splits",
        }

    out = {"available": True, "not_a_bar": True}
    for name, ref in ((FINETUNED, None), (NULL, F11_NULL_DRIFT)):
        if name not in test["models"] or name not in val["models"]:
            continue
        cov_t = float(coverage_indicator(test["obs"], test["models"][name], 0.80).mean())
        cov_v = float(coverage_indicator(val["obs"], val["models"][name], 0.80).mean())
        out[name] = {
            "cov80_val": round(cov_v, 4),
            "cov80_test": round(cov_t, 4),
            "drift": round(cov_t - cov_v, 4),
            "f11_reference": ref,
        }
    if FINETUNED in out:
        d = abs(out[FINETUNED]["drift"])
        out["shrank_vs_zeroshot"] = bool(d < F11_ZEROSHOT_DRIFT)
        out["zeroshot_drift"] = F11_ZEROSHOT_DRIFT
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="results/ dir for the fine-tuned TEST grid")
    ap.add_argument(
        "--val", type=Path, default=None, help="results/ dir for the fine-tuned VAL grid"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    test = load(args.run)
    val = load(args.val) if args.val else None

    c1 = condition_1(test, args.seed)
    c2 = condition_2(test, args.seed)
    verdict = bool(c1["passes"] and c2["passes"])

    payload = {
        "gate": "G4",
        "bar_fixed": BAR_FIXED,
        "run": str(args.run),
        "val_run": str(args.val) if args.val else None,
        "condition_1": c1,
        "condition_2": c2,
        "verdict": "PASS" if verdict else "FAIL",
        "period_drift_readout": period_drift_readout(test, val),
        "note": (
            "Regime spread is reported elsewhere and is not a pass condition (R1). The "
            "period-drift readout is recorded, never promoted to a bar."
        ),
    }

    print(f"=== G4 verdict, bar fixed {BAR_FIXED} ===\n")
    print("condition 1 -- fair-CRPS parity with the null")
    print(f"  finetuned {c1['crps_finetuned']:.3f}   null {c1['crps_null']:.3f}")
    lo, hi = c1["ci95"]
    print(f"  difference {c1['mean_difference']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  -> {'PASS' if c1['passes'] else 'FAIL'} (interval must contain zero)\n")

    print("condition 2 -- dispersion rises at h=30 and the curve flattens")
    print(f"  h=1 {c2['ratio_h1']:.3f}   h=30 {c2['ratio_h30']:.3f}  (zero-shot {ZEROSHOT_H30})")
    print(
        f"    h=30 CI [{c2['h30_ci95'][0]:.3f}, {c2['h30_ci95'][1]:.3f}] must clear {ZEROSHOT_H30}"
    )
    print(f"  h30/h1 {c2['flatness_h30_over_h1']:.3f}  (zero-shot {ZEROSHOT_FLATNESS:.3f})")
    print(f"    flatness CI [{c2['flatness_ci95'][0]:.3f}, {c2['flatness_ci95'][1]:.3f}]")
    print(f"  rose {c2['rose_at_h30']}, flattened {c2['curve_flattened']}")
    print(f"  -> {'PASS' if c2['passes'] else 'FAIL'} (both legs required)\n")

    print(f"G4: {payload['verdict']}\n")

    r = payload["period_drift_readout"]
    print("readout (not a bar) -- does fine-tuning shrink the F11 period drift?")
    if r.get("available") and FINETUNED in r:
        f = r[FINETUNED]
        print(
            f"  finetuned  val {f['cov80_val']:.4f} -> test {f['cov80_test']:.4f}"
            f"   drift {f['drift']:+.4f}"
        )
        print(f"  zero-shot drift {F11_ZEROSHOT_DRIFT:+.4f}   null {F11_NULL_DRIFT:+.4f}")
        print(f"  shrank vs zero-shot: {r['shrank_vs_zeroshot']}")
    else:
        print(f"  unavailable: {r.get('why')}")

    write_results(args.run, payload, filename="g4_verdict.json")
    print(f"\nwritten: {args.run / 'g4_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
