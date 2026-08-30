"""Placeholder for the fine-tuned engine, if milestone A6 ever earns one.

Importing this module is cheap and pulls in no torch. Instantiating it raises, because a
half-wired model path that silently produces something is worse than one that refuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.engines.base import ENSEMBLE_SIZE, HORIZON, Engine

REASON = (
    "No Kronos engine is served. Zero-shot lost the A3 gate by 81% on fair CRPS "
    "(Fact of Record F1) and no temperature arm repaired the horizon under-propagation "
    "that causes it (F9). A checkpoint ships only if the A6 pilot passes the bar fixed "
    "by gate G4: fair-CRPS parity with conformalized rw_drift, and an h=30 "
    "dispersion ratio materially above 0.481 with the curve flattening rather than "
    "merely shifting."
)


class KronosEngine(Engine):
    name = "kronos"
    validated = False

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(REASON)

    def forecast(
        self,
        bars: pd.DataFrame,
        horizon: int = HORIZON,
        m: int = ENSEMBLE_SIZE,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError(REASON)
