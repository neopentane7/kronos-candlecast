"""Contract v3 -- the only interface between the nightly job and the site.

**Delta from the frozen v2 contract**, all of it forced by Phase A's outcome:

* ``model`` + ``model_variant`` -> ``engine``. v2 assumed a Kronos checkpoint with a
  variant label. What ships is an analytic forecaster, and "model: Kronos-small-NSE,
  model_variant: finetuned" would be a false claim on every response.
* ``challenger`` added, always ``null`` today. If A6 ever passes its bar, the site can
  show two cones without another contract bump.
* ``last_close`` added. The site anchors the cone to it; without it every render needs a
  second file just to find the number the forecast starts from.
* ``backfilled`` added, top-level. Resolved conflict #10 requires seeded history to be
  distinguishable from forecasts actually made that morning, and burying that in metadata
  invites it being missed.
* ``metadata`` added: which method produced which band, the ACI gamma, and whether that
  gamma is still provisional. §17d permits a finite-sample guarantee at 50% only, so a
  response that presents all three bands identically is misleading about two of them.
  ``aci_provisional`` went false on 2026-08-30, when the served bands were first checked
  against realized outcomes rather than assumed; the field is unchanged, only its value.

Everything else is v2 unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CONTRACT_VERSION = 3
DISCLAIMER = "Research/education tool - scenario visualization, not investment advice."
QUANTILE_KEYS = ("p10", "p25", "p50", "p75", "p90")


class Quantiles(BaseModel):
    p10: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p90: list[float]

    @model_validator(mode="after")
    def _ordered_and_equal_length(self):
        lengths = {len(getattr(self, k)) for k in QUANTILE_KEYS}
        if len(lengths) != 1:
            raise ValueError(f"quantile series have differing lengths: {lengths}")
        # A crossed quantile is not a rounding artifact; it means the calibration layer
        # produced an interval whose lower edge is above its upper edge, and the chart
        # would render a cone inside out.
        for i in range(len(self.p10)):
            row = [getattr(self, k)[i] for k in QUANTILE_KEYS]
            if any(row[j] > row[j + 1] + 1e-9 for j in range(len(row) - 1)):
                raise ValueError(f"quantiles cross at step {i + 1}: {row}")
        return self


class RawQuantiles(BaseModel):
    p10: list[float]
    p25: list[float] | None = None
    p50: list[float] | None = None
    p75: list[float] | None = None
    p90: list[float]


class BandMethods(BaseModel):
    """Which machinery produced each band. Not decoration -- they differ in guarantee."""

    band_50: Literal["split_conformal", "aci", "none"] = "split_conformal"
    band_80: Literal["split_conformal", "aci", "none"] = "aci"
    band_90: Literal["split_conformal", "aci", "none"] = "aci"


class Metadata(BaseModel):
    band_methods: BandMethods = Field(default_factory=BandMethods)
    aci_gamma: float
    aci_provisional: bool
    ensemble_size: int
    lookback: int
    engine_validated: bool
    note: str | None = None


class Forecast(BaseModel):
    contract_version: Literal[3] = CONTRACT_VERSION
    ticker: str
    generated_at: str
    engine: Literal["rw_drift", "kronos"]
    calibration: Literal["aci", "split_conformal", "none"]
    challenger: dict | None = None
    horizon: int
    last_close: float
    backfilled: bool = False
    timestamps: list[str]
    quantiles: Quantiles
    raw_quantiles_p10_p90: RawQuantiles
    prob_above_last_close: list[float]
    prob_vol_exceeds_recent: float
    metadata: Metadata
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _lengths_agree(self):
        n = self.horizon
        if len(self.timestamps) != n:
            raise ValueError(f"timestamps has {len(self.timestamps)} entries, horizon is {n}")
        if len(self.quantiles.p50) != n:
            raise ValueError(f"quantiles have {len(self.quantiles.p50)} steps, horizon is {n}")
        if len(self.prob_above_last_close) != n:
            raise ValueError("prob_above_last_close must have one value per horizon step")
        if not all(0.0 <= p <= 1.0 for p in self.prob_above_last_close):
            raise ValueError("prob_above_last_close must lie in [0, 1]")
        if not 0.0 <= self.prob_vol_exceeds_recent <= 1.0:
            raise ValueError("prob_vol_exceeds_recent must lie in [0, 1]")
        if self.last_close <= 0:
            raise ValueError("last_close must be positive")
        if self.disclaimer != DISCLAIMER:
            raise ValueError("the disclaimer is not editable (working rule 12)")
        return self

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


def validate_file(path: Path) -> Forecast:
    """Parse and validate a committed forecast. Used by tests and by the site's CI."""
    return Forecast.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
