"""Turning sample paths into the bands the site draws.

Two tiers, because report §17d says only one of them is affordable at this horizon:

* **50%** -- split conformal against a scale fitted offline on the 2024 validation split.
  The finite-sample guarantee needs 3 exchangeable calibration residuals per side; twelve
  independent forecast dates exist, so this level and only this level is honestly
  supportable.
* **80% and 90%** -- ACI. Split conformal needs 9 and 19 residuals per side, i.e. 2.2 and
  4.6 years of data consumed by the split alone. ACI is not a fallback here; it is the
  only method the data geometry permits.

Both are applied above the engine so every forecaster is calibrated by the same code and
their intervals remain comparable. Quantiles use Weibull plotting positions throughout
(metric canon, hard constraint 12) -- with m=30 the NumPy default measures a perfectly
calibrated forecaster ~5pp low at nominal 80%, which is the instrument bias report §1
exists to document.
"""

from __future__ import annotations

import numpy as np

QUANTILE_METHOD = "weibull"
LEVELS = (0.50, 0.80, 0.90)


def raw_band(paths: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Uncalibrated empirical band from the ensemble, shaped ``(horizon,)`` each."""
    tail = (1.0 - level) / 2.0
    lo = np.quantile(paths, tail, axis=0, method=QUANTILE_METHOD)
    hi = np.quantile(paths, 1.0 - tail, axis=0, method=QUANTILE_METHOD)
    return lo, hi


def aci_band(
    paths: np.ndarray, ticker: str, level: float, aci_state
) -> tuple[np.ndarray, np.ndarray]:
    """Band drawn at the per-step *effective* level ACI has learned.

    The level varies along the horizon on purpose. F2 found the cone nearly correctly
    sized at h=1 and less than half wide enough by h=30, so a single correction for the
    whole path would over-widen the near end to rescue the far end -- which is exactly
    what the A4 temperature sweep demonstrated does not work (F9).
    """
    horizon = paths.shape[1]
    lo = np.empty(horizon)
    hi = np.empty(horizon)
    for h in range(horizon):
        eff = aci_state.effective_level(ticker, level, h + 1)
        tail = (1.0 - eff) / 2.0
        lo[h] = np.quantile(paths[:, h], tail, method=QUANTILE_METHOD)
        hi[h] = np.quantile(paths[:, h], 1.0 - tail, method=QUANTILE_METHOD)
    return lo, hi


def split_conformal_band(
    paths: np.ndarray, level: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Raw band with its half-width multiplied by a scale fitted on held-out data."""
    lo, hi = raw_band(paths, level)
    centre = np.quantile(paths, 0.5, axis=0, method=QUANTILE_METHOD)
    half = (hi - lo) / 2.0 * float(scale)
    return centre - half, centre + half


def build_bands(paths: np.ndarray, ticker: str, aci_state, split_scale: float | None) -> dict:
    """All three bands plus the median, each by the method its guarantee permits."""
    out: dict[str, np.ndarray] = {}

    if split_scale is None:
        out["lo_50"], out["hi_50"] = aci_band(paths, ticker, 0.50, aci_state)
        method_50 = "aci"
    else:
        out["lo_50"], out["hi_50"] = split_conformal_band(paths, 0.50, split_scale)
        method_50 = "split_conformal"

    out["lo_80"], out["hi_80"] = aci_band(paths, ticker, 0.80, aci_state)
    out["lo_90"], out["hi_90"] = aci_band(paths, ticker, 0.90, aci_state)
    out["p50"] = np.quantile(paths, 0.5, axis=0, method=QUANTILE_METHOD)

    # The site plots p10/p25/p50/p75/p90; 25/75 is the 50% band, 10/90 the 80% band.
    out["p10"], out["p90"] = out["lo_80"], out["hi_80"]
    out["p25"], out["p75"] = out["lo_50"], out["hi_50"]
    out["method_50"] = method_50
    return out


def enforce_monotone(bands: dict) -> dict:
    """Guarantee p10 <= p25 <= p50 <= p75 <= p90 at every step.

    Two independently calibrated bands can cross when the inner one is widened by a scale
    the outer one did not receive. The contract rejects crossed quantiles, so rather than
    emit something the schema will refuse, the ladder is sorted per step. Crossing is rare
    and small; a run that hits it a lot is a calibration problem worth seeing, so the
    count is returned.
    """
    keys = ["p10", "p25", "p50", "p75", "p90"]
    stacked = np.vstack([bands[k] for k in keys])
    crossed = int(np.sum(np.diff(stacked, axis=0) < 0))
    if crossed:
        stacked = np.sort(stacked, axis=0)
        for i, k in enumerate(keys):
            bands[k] = stacked[i]
    bands["crossings_repaired"] = crossed
    return bands


def path_probabilities(paths: np.ndarray, last_close: float, lookback_sigma: float) -> dict:
    """The two scalar summaries the contract carries, both sample-based.

    Labelled as such on the site: these are fractions of 30 sampled scenarios, not
    calibrated probabilities, and 30 samples resolve about 3 percentage points.
    """
    above = (paths > last_close).mean(axis=0)
    anchored = np.column_stack([np.full(paths.shape[0], last_close), paths])
    realized = np.diff(np.log(anchored), axis=1)
    horizon_sigma = realized.std(axis=1, ddof=1)
    return {
        "prob_above_last_close": [float(v) for v in above],
        "prob_vol_exceeds_recent": float((horizon_sigma > lookback_sigma).mean()),
    }
