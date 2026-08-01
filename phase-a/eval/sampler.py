"""Sampled forecast paths from Kronos.

Upstream's ``KronosPredictor.predict`` averages the ``sample_count`` sampled paths
before returning (``kronos.py:467``), which discards exactly the ensemble a
probabilistic evaluation needs: CRPS, quantile cones, and the contract's
``prob_above_last_close`` / ``prob_vol_exceeds_recent`` fields are all functions of the
individual paths.

This module is an overlay (the upstream clone is read-only). It reproduces upstream's
generation loop and normalization, and returns the paths un-averaged. The duplication is
deliberate and is guarded by ``tests/test_sampler_equivalence.py``: averaging our paths
must reproduce upstream's own output bit-for-bit under a fixed seed. That test is also
what will protect the KV-cache optimization when it lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from model.kronos import calc_time_stamps, sample_from_logits

FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


class KronosSampler:
    """Wraps a Kronos model + tokenizer and yields sampled paths rather than their mean."""

    def __init__(
        self, model, tokenizer, device: str | None = None, max_context: int = 512, clip: float = 5.0
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer.to(device).eval()
        self.max_context = max_context
        self.clip = clip

    # -- generation ---------------------------------------------------------------

    @torch.no_grad()
    def _generate_tokens(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count):
        """Mirror of upstream ``auto_regressive_inference`` that keeps every sample.

        Returns a tensor shaped ``(n_series, sample_count, context, n_features)``.
        """
        device = self.device
        x = torch.clip(x, -self.clip, self.clip)
        n_series = x.size(0)

        # Interleave samples exactly as upstream does, so a fixed seed draws the same
        # multinomial values in the same order and equivalence is testable.
        x = (
            x.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x.size(1), x.size(2))
            .to(device)
        )
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
            .to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
            .to(device)
        )

        x_token = self.tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch, pred_len)
        generated_post = x_token[1].new_empty(batch, pred_len)

        pre_buffer = x_token[0].new_zeros(batch, self.max_context)
        post_buffer = x_token[1].new_zeros(batch, self.max_context)
        buffer_len = min(initial_seq_len, self.max_context)
        if buffer_len > 0:
            start = max(0, initial_seq_len - self.max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start : start + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start : start + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, self.max_context)
            if current_seq_len <= self.max_context:
                tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                tokens = [pre_buffer, post_buffer]

            context_start = max(0, current_seq_len - self.max_context)
            stamp = full_stamp[:, context_start:current_seq_len, :].contiguous()

            s1_logits, context = self.model.decode_s1(tokens[0], tokens[1], stamp)
            sample_pre = sample_from_logits(
                s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )
            s2_logits = self.model.decode_s2(context, sample_pre)
            sample_post = sample_from_logits(
                s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < self.max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - self.max_context)
        decoded = self.tokenizer.decode(
            [
                full_pre[:, context_start:total_seq_len].contiguous(),
                full_post[:, context_start:total_seq_len].contiguous(),
            ],
            half=True,
        )
        return decoded.reshape(n_series, sample_count, decoded.size(1), decoded.size(2))

    # -- public API ---------------------------------------------------------------

    def sample(
        self,
        df_list: list[pd.DataFrame],
        x_timestamp_list: list[pd.Series],
        y_timestamp_list: list[pd.Series],
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 30,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sampled paths, shaped ``(n_series, sample_count, pred_len, 6)``.

        Each series is normalized by its own window statistics, exactly as upstream does,
        and the outputs are mapped back to price space with the same constants.
        """
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list and y_timestamp_list must be equal length")
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

        xs, x_stamps, y_stamps, means, stds = [], [], [], [], []
        for df, x_ts, y_ts in zip(df_list, x_timestamp_list, y_timestamp_list, strict=True):
            if len(y_ts) != pred_len:
                raise ValueError(f"y_timestamp has {len(y_ts)} entries, expected {pred_len}")
            arr = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            xs.append((arr - mean) / (std + 1e-5))
            x_stamps.append(calc_time_stamps(x_ts).to_numpy(dtype=np.float32))
            y_stamps.append(calc_time_stamps(y_ts).to_numpy(dtype=np.float32))
            means.append(mean)
            stds.append(std)

        x = torch.from_numpy(np.stack(xs)).to(self.device)
        x_stamp = torch.from_numpy(np.stack(x_stamps)).to(self.device)
        y_stamp = torch.from_numpy(np.stack(y_stamps)).to(self.device)

        decoded = self._generate_tokens(
            x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count
        )
        paths = decoded[:, :, -pred_len:, :].float().cpu().numpy()

        mean_arr = np.stack(means)[:, None, None, :]
        std_arr = np.stack(stds)[:, None, None, :]
        return paths * (std_arr + 1e-5) + mean_arr


def paths_to_quantiles(
    paths: np.ndarray, quantiles: dict[str, float], column: str = "close"
) -> dict[str, np.ndarray]:
    """Per-horizon quantiles of one feature across the sampled paths.

    ``paths`` is ``(n_series, n_samples, pred_len, 6)``; each output entry is
    ``(n_series, pred_len)``.
    """
    idx = FEATURE_COLUMNS.index(column)
    series = paths[:, :, :, idx]
    return {name: np.quantile(series, q, axis=1) for name, q in quantiles.items()}
