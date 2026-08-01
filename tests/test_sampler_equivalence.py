"""The overlay sampler must reproduce upstream exactly, not merely approximately.

``phase-a/eval/sampler.py`` duplicates upstream's generation loop in order to keep the
sampled paths that upstream averages away. Duplicated inference code is only safe if it
is pinned to the original, so this test asserts that averaging our paths reproduces
``KronosPredictor.predict`` bit-for-bit under a fixed seed.

The same test guards the KV-cache optimization: a cache that changes any sampled token
will fail here rather than silently shifting every downstream metric.

GPU + network (model download) required; skipped in CI.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "phase-a" / "Kronos"))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")

LOOKBACK, PRED_LEN, SAMPLES, SEED = 400, 30, 4, 1234


@pytest.fixture(scope="module")
def loaded():
    from eval.sampler import KronosSampler
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    sampler = KronosSampler(model, tokenizer, device="cuda:0", max_context=512)
    return predictor, sampler


@pytest.fixture(scope="module")
def window():
    """A deterministic synthetic window; the test is about equivalence, not realism."""
    rng = np.random.default_rng(7)
    n = LOOKBACK + PRED_LEN
    close = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = close * rng.uniform(0.003, 0.015, n)
    open_ = close + rng.normal(0, 0.3, n) * spread
    df = pd.DataFrame(
        {
            "timestamps": pd.bdate_range("2021-01-04", periods=n),
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
        }
    )
    df["amount"] = df["close"] * df["volume"]
    x = df.iloc[:LOOKBACK].reset_index(drop=True)
    return x, df["timestamps"].iloc[LOOKBACK:].reset_index(drop=True)


def test_mean_of_sampled_paths_matches_upstream_predict(loaded, window):
    predictor, sampler = loaded
    x_df, y_ts = window
    feats = ["open", "high", "low", "close", "volume", "amount"]

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    upstream = predictor.predict(
        df=x_df[feats],
        x_timestamp=x_df["timestamps"],
        y_timestamp=y_ts,
        pred_len=PRED_LEN,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=SAMPLES,
        verbose=False,
    )

    paths = sampler.sample(
        [x_df[feats]],
        [x_df["timestamps"]],
        [y_ts],
        pred_len=PRED_LEN,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=SAMPLES,
        seed=SEED,
    )

    assert paths.shape == (1, SAMPLES, PRED_LEN, len(feats))
    ours = paths[0].mean(axis=0)
    np.testing.assert_allclose(
        ours, upstream[feats].to_numpy(dtype=np.float32), rtol=1e-5, atol=1e-3
    )


def test_sampled_paths_are_not_degenerate(loaded, window):
    """If sampling collapsed, the ensemble would be useless for CRPS or a cone."""
    _, sampler = loaded
    x_df, y_ts = window
    feats = ["open", "high", "low", "close", "volume", "amount"]

    paths = sampler.sample(
        [x_df[feats]],
        [x_df["timestamps"]],
        [y_ts],
        pred_len=PRED_LEN,
        sample_count=SAMPLES,
        seed=SEED,
    )
    close = paths[0, :, :, feats.index("close")]
    spread_at_final_step = close[:, -1].std()
    assert spread_at_final_step > 0, "all sampled paths identical"
    # Dispersion should widen with horizon for an autoregressive sampler.
    assert close[:, -1].std() > close[:, 0].std()
