"""Offline analysis of a completed run. CPU only — reads ``ensembles.npz``.

Answers three questions that need the saved forecast paths but no GPU:

1. **Re-centering decomposition** — how much of the miscalibration is *systematic* drift,
   which fine-tuning removes cheaply, versus dispersion and idiosyncratic per-window
   error, which it does not.
2. **Dispersion ratio by horizon** — whether the ensemble is already too narrow at h = 1,
   which would make part of the conformal budget a structural ceiling rather than
   something training can lift.
3. **Conformal variants on real data** — marginal, normalized and Mondrian, plus the
   conformalized random-walk null that A5 requires the model to beat.

Usage:
    uv run python phase-a/eval/run_analysis.py results/<run-dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))

from eval.analysis import remove_systematic_bias, z_ratio_table  # noqa: E402
from eval.conformal import (  # noqa: E402
    apply_mondrian,
    apply_scale,
    coverage_of,
    fit_mondrian,
    fit_scale,
    mean_width,
)
from eval.metrics import coverage_indicator  # noqa: E402

from common.results import DISCLAIMER, write_results  # noqa: E402

LEVEL = 0.80
REGIMES = ("calm", "mid", "volatile")


def regime_coverage(obs, lo, hi, terciles):
    inside = (obs >= lo) & (obs <= hi)
    return np.array([inside[terciles == r].mean() for r in (0, 1, 2)])


def recentering_decomposition(obs, ens) -> dict:
    """How much of the gap would ideal removal of *systematic* drift close?"""
    corrected, bias = remove_systematic_bias(obs, ens, relative=True)
    raw, rec = z_ratio_table(obs, ens), z_ratio_table(obs, corrected)
    denom = raw["implied_scale_factor"] - 1.0
    return {
        "bias_per_step_first": float(bias[0]),
        "bias_per_step_last": float(bias[-1]),
        "bias_per_step_mean": float(bias.mean()),
        "raw": {
            "coverage": [float(coverage_indicator(obs, ens, lv).mean()) for lv in (0.5, 0.8, 0.9)],
            "z_ratio": raw["ratio_mean"],
            "implied_scale": raw["implied_scale_factor"],
        },
        "recentered": {
            "coverage": [
                float(coverage_indicator(obs, corrected, lv).mean()) for lv in (0.5, 0.8, 0.9)
            ],
            "z_ratio": rec["ratio_mean"],
            "implied_scale": rec["implied_scale_factor"],
        },
        "share_of_gap_from_systematic_drift": float(
            1.0 - (rec["implied_scale_factor"] - 1.0) / denom
        )
        if denom > 0
        else None,
    }


def dispersion_by_horizon(obs, ens, history, terciles) -> dict:
    """Predicted ensemble spread against realized spread, per horizon step.

    Both are expressed relative to the last observed close so tickers at different price
    levels are comparable. The realized figure is the cross-window standard deviation of
    the actual relative move at that step.
    """
    last = history[:, -1]
    pred_rel = ens.std(axis=-1) / last[:, None]
    realized_rel = (obs / last[:, None] - 1.0).std(axis=0)
    ratio = pred_rel.mean(axis=0) / realized_rel

    by_regime = {}
    for t, name in enumerate(REGIMES):
        idx = terciles == t
        if idx.sum() < 2:
            continue
        rr = (obs[idx] / last[idx][:, None] - 1.0).std(axis=0)
        by_regime[name] = float(pred_rel[idx, 0].mean() / rr[0])

    return {
        "ratio_by_step": [float(v) for v in ratio],
        "ratio_h1": float(ratio[0]),
        "ratio_hmax": float(ratio[-1]),
        "h1_ratio_by_regime": by_regime,
    }


def shared_block_mask(blocks: np.ndarray, tickers: np.ndarray) -> np.ndarray:
    """Windows on dates the whole universe traded.

    Report §17b: one ticker with an offset session index produces forecast dates nobody
    else shares, and those orphan dates are not exchangeable with the rest. The 2024 split
    still carries seven of them (all BAJAJ-AUTO, displaced by its 2024-01-15 hole), so a
    calibration set built from every window would fit partly on one ticker's private
    calendar.
    """
    per_window = blocks[:, 0]
    n_tickers = len(np.unique(tickers))
    keep = np.zeros(len(per_window), dtype=bool)
    for b in np.unique(per_window):
        sel = per_window == b
        if len(np.unique(tickers[sel])) >= n_tickers / 2:
            keep |= sel
    return keep


def conformal_comparison(obs, ens_by_model, history, terciles, blocks) -> dict:
    """Fit on the earlier half of the date blocks, evaluate on the later half.

    **Exploratory.** The rule pinned in §17d calibrates on the 2024 validation split and
    never on the test period. This within-test variant is retained because it measures
    something the pinned analysis does not: how far the residual distribution shifts
    *inside* the test window, which is exchangeability failure observed directly.
    """
    uniq = np.unique(blocks[:, 0])
    cal_set = set(uniq[: len(uniq) // 2])
    cal = np.array([b in cal_set for b in blocks[:, 0]])
    return _arms(obs, ens_by_model, history, terciles, cal, ~cal, None, "within_test_split")


def preregistered_conformal(cal_npz, tst_npz) -> dict:
    """The A5 analysis: calibrate on the 2024 val grid, evaluate on the test grid.

    Fixed in writing on 2026-08-06, before either grid ran. Calibration and test come from
    different files here rather than different halves of one, which is the whole point --
    the test period is measured, never fitted.
    """
    cal_obs, cal_hist = cal_npz["y_close"], cal_npz["history_close"]
    tst_obs, tst_hist = tst_npz["y_close"], tst_npz["history_close"]

    keep = shared_block_mask(cal_npz["block_ids"], cal_npz["tickers"])
    cal_models = {
        k[len("ens__") :]: cal_npz[k][keep] for k in cal_npz.files if k.startswith("ens__")
    }
    tst_models = {k[len("ens__") :]: tst_npz[k] for k in tst_npz.files if k.startswith("ens__")}

    shared = sorted(set(cal_models) & set(tst_models))
    if "kronos_zeroshot" not in shared:
        raise SystemExit(f"both runs need kronos_zeroshot; shared models were {shared}")

    out = _arms(
        tst_obs,
        {m: tst_models[m] for m in shared},
        tst_hist,
        tst_npz["atr_tercile"],
        cal=None,
        tst=np.ones(len(tst_obs), dtype=bool),
        cal_source={
            "obs": cal_obs[keep],
            "history": cal_hist[keep],
            "terciles": cal_npz["atr_tercile"][keep],
            "models": cal_models,
        },
        label="preregistered_val_2024",
    )
    out["calibration_windows_dropped_as_orphans"] = int((~keep).sum())
    out["calibration_blocks"] = int(len(np.unique(cal_npz["block_ids"][keep][:, 0])))
    return out


def _arms(obs, ens_by_model, history, terciles, cal, tst, cal_source=None, label="") -> dict:
    """Shared arm construction, whether calibration comes from a mask or another run."""
    vol_all = (history.std(axis=1) / history.mean(axis=1)) * history[:, -1]

    if cal_source is None:
        c_obs, c_hist, c_terc = obs[cal], history[cal], terciles[cal]
        c_models = {m: e[cal] for m, e in ens_by_model.items()}
    else:
        c_obs, c_hist = cal_source["obs"], cal_source["history"]
        c_terc, c_models = cal_source["terciles"], cal_source["models"]
    c_vol = (c_hist.std(axis=1) / c_hist.mean(axis=1)) * c_hist[:, -1]

    kronos_c, kronos_t = c_models["kronos_zeroshot"], ens_by_model["kronos_zeroshot"][tst]
    arms: dict[str, tuple] = {}

    lo = np.quantile(kronos_t, 0.10, axis=-1, method="weibull")
    hi = np.quantile(kronos_t, 0.90, axis=-1, method="weibull")
    arms["raw_kronos"] = (lo, hi)

    arms["marginal_conformal"] = apply_scale(kronos_t, fit_scale(c_obs, kronos_c, LEVEL), LEVEL)
    arms["normalized_lookback_vol"] = apply_scale(
        kronos_t,
        fit_scale(c_obs, kronos_c, LEVEL, normalizer=c_vol),
        LEVEL,
        normalizer=vol_all[tst],
    )
    arms["mondrian"] = apply_mondrian(
        kronos_t, fit_mondrian(c_obs, kronos_c, c_terc, LEVEL), terciles[tst], LEVEL
    )
    if "random_walk_drift" in c_models:
        arms["conformalized_rw_drift"] = apply_scale(
            ens_by_model["random_walk_drift"][tst],
            fit_scale(c_obs, c_models["random_walk_drift"], LEVEL),
            LEVEL,
        )

    out = {
        "design": label,
        "n_calibration_windows": int(len(c_obs)),
        "n_test_windows": int(tst.sum()),
        "calibration_windows_per_stratum": [int((c_terc == t).sum()) for t in (0, 1, 2)],
        "arms": {},
    }
    for name, (lo, hi) in arms.items():
        reg = regime_coverage(obs[tst], lo, hi, terciles[tst])
        out["arms"][name] = {
            "marginal": coverage_of(obs[tst], lo, hi),
            "by_regime": {r: float(v) for r, v in zip(REGIMES, reg, strict=True)},
            "regime_spread": float(np.ptp(reg)),
            "relative_width": mean_width(lo, hi, obs[tst]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="the TEST-split run to evaluate")
    ap.add_argument(
        "--calibration-run",
        type=Path,
        default=None,
        help=(
            "a val-split run to calibrate on. With it, the conformal section is the "
            "pre-registered A5 analysis (section 17d); without it, an exploratory "
            "within-test split."
        ),
    )
    args = ap.parse_args()

    npz = args.run_dir / "ensembles.npz"
    if not npz.exists():
        print(f"no ensembles.npz in {args.run_dir}")
        return 1

    d = np.load(npz, allow_pickle=True)
    obs, history = d["y_close"], d["history_close"]
    terciles, blocks = d["atr_tercile"], d["block_ids"]
    models = {k[len("ens__") :]: d[k] for k in d.files if k.startswith("ens__")}
    if "kronos_zeroshot" not in models:
        print(f"no model ensemble in {npz}; found {sorted(models)}")
        return 1

    payload = {
        "run": "A3_offline_analysis",
        "source_run": args.run_dir.name,
        "n_windows": int(obs.shape[0]),
        "recentering": recentering_decomposition(obs, models["kronos_zeroshot"]),
        "dispersion": dispersion_by_horizon(obs, models["kronos_zeroshot"], history, terciles),
        "conformal": conformal_comparison(obs, models, history, terciles, blocks),
    }

    if args.calibration_run:
        cal_npz = args.calibration_run / "ensembles.npz"
        if not cal_npz.exists():
            print(f"no ensembles.npz in {args.calibration_run}")
            return 1
        payload["run"] = "A5_preregistered_conformal"
        payload["calibration_run"] = args.calibration_run.name
        payload["conformal_preregistered"] = preregistered_conformal(
            np.load(cal_npz, allow_pickle=True), d
        )
    # filename is not optional here. Without it write_results defaults to results.json
    # and this offline pass silently destroys the measurement it was run to analyse --
    # which is what happened to run 20260803T095254Z_397dbc9, whose grid metrics were
    # replaced by an analysis payload and had to be rebuilt from ensembles.npz.
    path = write_results(
        args.run_dir,
        payload | {"note": "written by run_analysis.py"},
        filename="analysis.json",
    )

    r, disp, cf = payload["recentering"], payload["dispersion"], payload["conformal"]
    print(
        f"re-centering: systematic drift explains "
        f"{r['share_of_gap_from_systematic_drift']:.1%} of the gap "
        f"({r['raw']['implied_scale']:.3f}x -> {r['recentered']['implied_scale']:.3f}x)"
    )
    print(f"dispersion  : h=1 ratio {disp['ratio_h1']:.3f}, h=max ratio {disp['ratio_hmax']:.3f}")
    print(f"\n{'arm':<28} {'marginal':>9} {'spread':>8} {'width':>8}")
    for name, a in cf["arms"].items():
        print(
            f"{name:<28} {a['marginal']:>9.3f} {a['regime_spread']:>8.3f} "
            f"{a['relative_width']:>8.4f}"
        )
    print(f"\ncalibration per stratum: {cf['calibration_windows_per_stratum']}")
    print(f"analysis: {path}")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
