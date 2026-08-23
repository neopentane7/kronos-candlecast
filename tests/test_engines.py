"""The served engine must be the forecaster Phase A measured, not a lookalike.

Every number CandleCast shows rests on the A3 grid having scored *this* engine. If the
serving implementation drifts from the evaluated baseline -- a different RNG order, a
different sigma convention -- the site inherits calibration claims it never earned. The
parity test below is the load-bearing one; the rest guard the preconditions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "phase-a")]

from pipeline.engines import RandomWalkDrift  # noqa: E402
from pipeline.engines.base import LOOKBACK  # noqa: E402
from pipeline.engines.kronos import KronosEngine  # noqa: E402


def bars(n=LOOKBACK, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=n)))
    return pd.DataFrame(
        {
            "timestamps": pd.bdate_range("2020-01-01", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
            "amount": close * 1e6,
        }
    )


def test_served_engine_matches_the_evaluated_baseline_bit_for_bit():
    """The claim the whole product rests on.

    phase-a/eval/baselines.py is what the A3 grid scored. pipeline/engines/rw_drift.py is
    what the nightly job serves. If these ever diverge, the site is quoting coverage
    numbers measured on a different forecaster.
    """
    from eval.baselines import random_walk_drift

    df = bars()
    close = df["close"].to_numpy()[-LOOKBACK:]

    evaluated = random_walk_drift(close[None, :], horizon=30, n_samples=30, seed=1234)
    served = RandomWalkDrift().forecast(df, horizon=30, m=30, seed=1234)

    assert evaluated.shape == (1, 30, 30)
    assert served.shape == (30, 30)
    np.testing.assert_array_equal(served, evaluated[0].T)


def test_forecast_is_deterministic_in_the_seed():
    df = bars()
    e = RandomWalkDrift()
    np.testing.assert_array_equal(e.forecast(df, seed=7), e.forecast(df, seed=7))
    assert not np.array_equal(e.forecast(df, seed=7), e.forecast(df, seed=8))


def test_paths_start_near_the_last_close_and_stay_positive():
    df = bars()
    paths = RandomWalkDrift().forecast(df, horizon=30, m=200, seed=3)
    last = float(df["close"].iloc[-1])
    assert (paths > 0).all(), "a price path must never go non-positive"
    # One session of drift+noise cannot move the median far from the anchor.
    assert abs(float(np.median(paths[:, 0])) / last - 1.0) < 0.05


def test_uncertainty_grows_with_horizon():
    """The property the foundation model lacks, and the reason this engine ships.

    Under a random walk the spread of the path ensemble scales with sqrt(h). This is not
    a tight test of the exponent -- it asserts the direction that Fact of Record F2 found
    missing in Kronos.
    """
    paths = RandomWalkDrift().forecast(bars(), horizon=30, m=2000, seed=5)
    spread = paths.std(axis=0)
    assert spread[0] < spread[14] < spread[29]
    ratio = spread[29] / spread[0]
    assert 4.0 < ratio < 7.0, f"sqrt(30) is 5.48; got {ratio:.2f}"


def test_short_history_is_refused_rather_than_forecast():
    """A recent listing is a skip. Forecasting from 40 sessions would look fine."""
    with pytest.raises(ValueError, match="need 400 sessions"):
        RandomWalkDrift().forecast(bars(n=40))


def test_non_positive_or_missing_closes_are_refused():
    df = bars()
    df.loc[df.index[-5], "close"] = 0.0
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        RandomWalkDrift().forecast(df)

    df2 = bars()
    df2.loc[df2.index[-5], "close"] = np.nan
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        RandomWalkDrift().forecast(df2)


def test_missing_close_column_is_refused():
    with pytest.raises(ValueError, match="'close' column"):
        RandomWalkDrift().forecast(bars().drop(columns=["close"]))


def test_params_report_what_was_actually_used():
    df = bars()
    p = RandomWalkDrift().params(df)
    assert p["last_close"] == pytest.approx(float(df["close"].iloc[-1]))
    assert p["lookback"] == LOOKBACK
    assert p["sigma_per_session"] > 0


def test_kronos_engine_refuses_to_instantiate():
    """A half-wired model path that silently produces something is the worse failure."""
    with pytest.raises(NotImplementedError, match="A6 pilot passes"):
        KronosEngine()


def test_serving_path_imports_no_torch():
    """The nightly job runs on a free CPU runner; a torch import would be a 2.5GB wheel."""
    import subprocess

    code = (
        "import sys; import pipeline.engines as e; "
        "assert e.RandomWalkDrift; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "importing the serving engines pulled in torch"
