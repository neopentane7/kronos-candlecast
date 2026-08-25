"""The forecasting interface the serving path depends on.

Phase A's verdict (Facts of Record F1, F5, F9) is that the foundation model loses to the
random walk and that no sampling-policy change repairs it, so ``rw_drift`` is what ships.
The interface exists anyway, for one reason: if the A6 pilot passes its bar, the engine
should be swappable without touching the nightly job, the contract, or the site.

An engine is deliberately narrow. It sees one ticker's recent bars and returns sample
paths; it does not know about calibration, contracts, archives, or which quantiles anyone
intends to take. Conformal correction happens above it, so every engine is calibrated by
the same code and their intervals stay comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

LOOKBACK = 400
HORIZON = 30
ENSEMBLE_SIZE = 30

# Serving draws far more paths than the evaluation did, and the difference is deliberate.
# m=30 was the *comparison* size: Phase A scored Kronos and the baseline at equal ensemble
# size so neither gained from the finite-ensemble coverage bias (report section 1). Nothing
# about serving needs that constraint, and rw_drift is analytic -- 5000 paths cost 3.4ms per
# ticker. At m=30 the 10th percentile rests on about three order statistics, so the cone
# visibly wiggles instead of widening; at m=5000 the 80% band is monotone at every step.
# Weibull plotting positions are unbiased at any m, so this removes variance without
# introducing bias.
SERVING_ENSEMBLE_SIZE = 5000


class Engine(ABC):
    """A probabilistic forecaster over a single ticker's close prices."""

    #: Value written to the contract's ``engine`` field. Stable; the site keys off it.
    name: str

    #: False for anything that has not passed a pre-registered gate. The site badges it.
    validated: bool = False

    @abstractmethod
    def forecast(
        self,
        bars: pd.DataFrame,
        horizon: int = HORIZON,
        m: int = ENSEMBLE_SIZE,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample paths of the close price, shaped ``(m, horizon)``.

        ``bars`` is canonical-schema OHLCV in ascending time order, at least ``LOOKBACK``
        rows. Returns absolute prices, not returns: the calibration layer above works in
        price space and the site plots price.
        """

    def check_bars(self, bars: pd.DataFrame, lookback: int = LOOKBACK) -> np.ndarray:
        """Shared precondition check, returning the close series an engine should use.

        Every engine needs the same guarantees and none of them should re-derive the
        checks. A short or gappy history is a skip, not a forecast on partial data --
        silently forecasting from 40 sessions because a listing is recent is exactly the
        kind of plausible-but-wrong output the pipeline must not emit.
        """
        if "close" not in bars.columns:
            raise ValueError("bars must carry a 'close' column (common/preprocess schema)")
        close = bars["close"].to_numpy(dtype=float)
        if close.size < lookback:
            raise ValueError(f"need {lookback} sessions of history, got {close.size}")
        window = close[-lookback:]
        if not np.all(np.isfinite(window)) or np.any(window <= 0):
            raise ValueError("lookback window contains non-finite or non-positive closes")
        return window
