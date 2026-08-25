"""A6: the bounded fine-tuning pilot. One config, one run, a kill rule that fires itself.

Milestone A6 exists to complete a story, not to rescue a model. Phase A established that
zero-shot Kronos-small loses to a random walk by 81% on fair CRPS (F1), that no sampling
temperature repairs the horizon under-propagation causing it (F9), and that conformal
correction fixes the coverage only by inflating the band 4.6x (F10). The pilot asks the one
remaining cheap question: does *training* move the defect?

**The bar was fixed on 2026-08-09, before any of this existed** (CLAUDE.md G4). A6 passes on
both of:

1. fair-CRPS **parity** with conformalized rw_drift -- the paired block-bootstrap interval
   for the difference contains zero; and
2. the **h=30 dispersion ratio rises materially above 0.481**, with the ratio curve
   *flattening* rather than merely shifting.

Condition 2 has a measured negative example: A4 lifted h=30 spread 52% by raising
temperature and left h30/h1 completely unchanged (F9). That is what "merely shifting" looks
like, and the pilot has to beat the shape, not the number.

**The kill rule is enforced here rather than remembered.** One config; no sweeps; stop if
validation loss has not beaten the pretrained initialisation by the end of epoch 3, or at 8
hours wall-clock, whichever comes first. "One more epoch" is how weeks disappear, so the
decision is code.

Nothing is published on a fail. Non-goal 4: a checkpoint ships only if G4 passes.

Usage (PowerShell):
    uv run python phase-a/train/finetune.py --smoke          # 2 min, wiring check
    uv run python phase-a/train/finetune.py                  # the pilot
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase-a"))
sys.path.insert(0, str(REPO_ROOT / "phase-a" / "Kronos"))

from eval.sampler import FEATURE_COLUMNS  # noqa: E402
from eval.windows import HORIZON, LOOKBACK, load_corpus  # noqa: E402

from common.results import DISCLAIMER, new_run_dir, write_results  # noqa: E402
from common.splits import SPLITS  # noqa: E402

TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-small"
SEQ_LEN = LOOKBACK + HORIZON  # 430, inside the model's 512 context
CLIP = 5.0  # identical to eval/sampler.py -- see PARITY below


@dataclass(frozen=True)
class Config:
    """The one configuration. Non-goal 2 forbids a search over these."""

    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 8
    stride: int = 5
    warmup_steps: int = 100
    grad_clip: float = 3.0
    seed: int = 1234

    # Kill rule, per CLAUDE.md §6. Both are hard stops, not guidance.
    kill_epoch: int = 3
    max_hours: float = 8.0


class WindowDataset(torch.utils.data.Dataset):
    """Contiguous windows, normalised exactly the way inference normalises them.

    **PARITY.** Hard constraint 6 says preprocessing is mechanical and shared. The
    per-window mean/std and the ±5 clip here are copied from ``eval/sampler.py`` rather
    than reinvented, and ``tests/test_finetune.py`` asserts the two agree on real bars. A
    training-only normalisation is the classic way a fine-tune looks good offline and
    fails in serving.
    """

    def __init__(self, corpus: pd.DataFrame, split: str, stride: int, limit=None):
        from model.kronos import calc_time_stamps

        lo, hi = SPLITS[split]
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        # load_corpus returns one long frame, ticker-major and time-ordered. Grouping here
        # rather than expecting a dict keeps this the same corpus object the evaluation
        # harness reads, which is the point of hard constraint 6.
        for _ticker, df in corpus.groupby("ticker", sort=True):
            ts = pd.to_datetime(df["timestamps"])
            feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            # A window belongs to the split if it *ends* inside it, so no window straddles
            # a boundary and leaks later data into an earlier split.
            ends = np.flatnonzero((ts >= lo) & (ts <= hi))
            for end in range(ends[0] if ends.size else 0, ends[-1] + 1 if ends.size else 0, stride):
                start = end - SEQ_LEN + 1
                if start < 0:
                    continue
                arr = feats[start : end + 1]
                if arr.shape[0] != SEQ_LEN or not np.isfinite(arr).all():
                    continue
                mean, std = arr.mean(axis=0), arr.std(axis=0)
                x = np.clip((arr - mean) / (std + 1e-5), -CLIP, CLIP)
                stamp = calc_time_stamps(ts.iloc[start : end + 1]).to_numpy(dtype=np.float32)
                self.samples.append((x.astype(np.float32), stamp))
                if limit and len(self.samples) >= limit:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        x, stamp = self.samples[i]
        return torch.from_numpy(x), torch.from_numpy(stamp)


def evaluate(model, tokenizer, loader, device, amp_dtype) -> float:
    """Mean next-token loss over a loader. Used for the baseline and every epoch."""
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for x, stamp in loader:
            x, stamp = x.to(device, non_blocking=True), stamp.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                s1, s2 = tokenizer.encode(x, half=True)
                logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
            total += float(loss.item())
            batches += 1
    model.train()
    return total / max(batches, 1)


def kill_check(history: list[dict], baseline: float, cfg: Config, elapsed_h: float):
    """The pre-registered stop conditions, evaluated after every epoch.

    Returns ``(stop, reason)``. Kept as a pure function so the rule can be tested without
    a GPU -- a kill rule that only exists inside a training loop is a kill rule nobody
    ever checks.
    """
    if elapsed_h >= cfg.max_hours:
        return True, f"wall-clock cap: {elapsed_h:.2f}h >= {cfg.max_hours}h"
    if len(history) >= cfg.kill_epoch:
        best = min(h["val_loss"] for h in history)
        if best >= baseline:
            return True, (
                f"val loss has not beaten the pretrained init by epoch {cfg.kill_epoch}: "
                f"best {best:.5f} vs baseline {baseline:.5f}"
            )
    return False, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny run that only checks the wiring")
    ap.add_argument("--batch-size", type=int, default=None, help="lower it on OOM")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = Config(
        batch_size=args.batch_size or Config.batch_size,
        epochs=args.epochs or (1 if args.smoke else Config.epochs),
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else None

    from model import Kronos, KronosTokenizer

    print(f"loading {MODEL_ID} onto {device} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID).to(device).eval()
    for p in tokenizer.parameters():  # the tokenizer is frozen; only the predictor trains
        p.requires_grad_(False)
    model = Kronos.from_pretrained(MODEL_ID).to(device)

    corpus = load_corpus(REPO_ROOT / "data" / "parquet")
    lim = 64 if args.smoke else None
    train_ds = WindowDataset(corpus, "train", cfg.stride, limit=lim)
    val_ds = WindowDataset(corpus, "val", cfg.stride * 4, limit=lim)
    print(f"train windows: {len(train_ds)}   val windows: {len(val_ds)}")
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("no windows built; is data/parquet present?")

    dl = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, **dl)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **dl)

    run_dir = new_run_dir()
    payload = {
        "run": "A6_finetune_pilot",
        "config": asdict(cfg),
        "device": str(device),
        "amp": str(amp_dtype),
        "seq_len": SEQ_LEN,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "bar": {
            "fixed": "2026-08-09, before the pilot existed (CLAUDE.md G4)",
            "condition_1": "fair-CRPS parity with conformalized rw_drift",
            "condition_2": "h=30 dispersion ratio materially above 0.481, curve flattening",
            "kill_rule": f"stop at epoch {cfg.kill_epoch} if val loss has not beaten init, "
            f"or at {cfg.max_hours}h",
        },
    }

    # The baseline is the pretrained model's own validation loss, measured before a single
    # gradient step. Without it "did training help" has no referent and the kill rule at
    # epoch 3 has nothing to compare against.
    print("measuring the pretrained baseline ...")
    baseline = evaluate(model, tokenizer, val_loader, device, amp_dtype)
    payload["pretrained_val_loss"] = baseline
    print(f"  pretrained val loss: {baseline:.5f}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=amp_dtype is not None)
    steps_per_epoch = max(len(train_loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1
    )

    history: list[dict] = []
    best = float("inf")
    t0 = time.perf_counter()
    stop_reason = "completed all epochs"

    for epoch in range(1, cfg.epochs + 1):
        run_loss, seen = 0.0, 0
        for i, (x, stamp) in enumerate(train_loader, 1):
            x, stamp = x.to(device, non_blocking=True), stamp.to(device, non_blocking=True)
            with torch.no_grad():
                s1, s2 = tokenizer.encode(x, half=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += float(loss.item())
            seen += 1
            if i % 50 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss {run_loss / seen:.5f}")

        val = evaluate(model, tokenizer, val_loader, device, amp_dtype)
        elapsed_h = (time.perf_counter() - t0) / 3600
        history.append(
            {
                "epoch": epoch,
                "train_loss": run_loss / max(seen, 1),
                "val_loss": val,
                "beats_baseline": bool(val < baseline),
                "elapsed_hours": round(elapsed_h, 3),
            }
        )
        print(
            f"epoch {epoch}: train {history[-1]['train_loss']:.5f}  val {val:.5f}  "
            f"baseline {baseline:.5f}  {'BEATS' if val < baseline else 'no'}  "
            f"[{elapsed_h:.2f}h]"
        )

        if val < best:
            best = val
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val,
                    "config": asdict(cfg),
                },
                run_dir / "checkpoint_best.pt",
            )

        stop, reason = kill_check(history, baseline, cfg, elapsed_h)
        if stop:
            stop_reason = reason
            print(f"\nKILL RULE FIRED: {reason}")
            break

    payload["history"] = history
    payload["best_val_loss"] = best
    payload["stopped_because"] = stop_reason
    payload["improved_over_pretrained"] = bool(best < baseline)
    payload["wall_hours"] = round((time.perf_counter() - t0) / 3600, 3)
    payload["checkpoint"] = (
        "checkpoint_best.pt" if (run_dir / "checkpoint_best.pt").exists() else None
    )
    payload["note"] = (
        "Training loss is not the G4 bar. The bar is fair-CRPS parity with conformalized "
        "rw_drift and a flattening h=30 dispersion curve, both measured by running "
        "calibrate.py --checkpoint against this run. A lower val loss is necessary, not "
        "sufficient, and no checkpoint is published unless G4 passes (non-goal 4)."
    )
    path = write_results(run_dir, payload)

    print("\n--- A6 pilot ---")
    print(f"  pretrained val loss : {baseline:.5f}")
    verdict = "improved" if best < baseline else "NO improvement"
    print(f"  best val loss       : {best:.5f}  ({verdict})")
    print(f"  epochs run          : {len(history)} of {cfg.epochs}")
    print(f"  stopped because     : {stop_reason}")
    print(f"  wall clock          : {payload['wall_hours']}h")
    print(f"  results             : {path}")
    print("\n  Next: evaluate the checkpoint against the G4 bar --")
    print("    uv run python phase-a/eval/calibrate.py --split test --batch-size 24 \\")
    print(f"        --checkpoint {run_dir / 'checkpoint_best.pt'}")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
