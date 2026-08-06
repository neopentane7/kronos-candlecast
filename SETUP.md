# Running this project on another machine

Written after three GPU losses on the original laptop (an RTX 4050 that drops off the
PCIe bus under sustained CUDA load, ~30–60 windows in). The evaluation grid needs roughly
4–6 hours of stable GPU, so moving to different hardware is the practical fix.

> Research/education tool — scenario visualization, not investment advice.

---

## The short version

```bash
git clone <repo-url> kronos-candlecast && cd kronos-candlecast
uv sync --all-groups                      # env + Python 3.10, ~5 GB
git clone https://github.com/shiyu-coder/Kronos.git phase-a/Kronos
git -C phase-a/Kronos checkout --detach 67b630e67f6a18c9e9be918d9b4337c960db1e9a
# ...then copy data/ across (see below) and run:
uv run pytest -q
uv run python phase-a/eval/calibrate.py --split test --batch-size 6
```

---

## 1. What must be copied — `data/` only, 17.6 MB

| path | size | why it cannot be regenerated |
|---|---|---|
| `data/parquet/` | **6.1 MB** | **the corpus. Copy this.** |
| `data/manifest.json` | 63 KB | records `downloaded_at`, universe composition, per-ticker cleaning |
| `data/dataset_manifest.json` | 18 KB | split row counts, eval-window accounting |
| `data/csv/` | 11.4 MB | optional — regenerate with `build_dataset.py` |

`data/` is gitignored, so it will not arrive with the clone. Copy the whole directory; at
17.6 MB there is no reason to be selective.

### Do NOT re-run `fetch_nse.py` on the new machine

This is the one thing that will silently invalidate the work.

Canonical prices are **split/bonus back-adjusted**, and back-adjustment rewrites history:
a corporate action occurring after a fetch changes every earlier bar for that ticker. Two
corpora fetched on different dates are therefore **not interchangeable**, which is why the
manifest records `downloaded_at` alongside the commit sha.

Re-fetching would produce a corpus that no longer matches any measurement taken so far —
the zero-shot results, the decomposition, the conformal comparison — and nothing would
error. The numbers would simply stop being comparable.

Copy `data/`. Re-fetch only when deliberately starting a new corpus generation.

---

## 2. What regenerates automatically

| thing | how | notes |
|---|---|---|
| Python env | `uv sync --all-groups` | fetches Python 3.10 itself; ~5 GB with torch |
| Upstream Kronos | clone at pinned SHA (below) | gitignored by design; read-only overlay target |
| Model weights | first run | ~120 MB from Hugging Face, cached |
| `data/csv/` | `uv run python phase-a/scripts/build_dataset.py` | one-way export from Parquet |
| `results/**/*.npz` | re-running the harness | derived data |

### Upstream clone

`phase-a/scripts/setup_upstream.ps1` does this on Windows. On Linux/macOS:

```bash
git clone https://github.com/shiyu-coder/Kronos.git phase-a/Kronos
git -C phase-a/Kronos checkout --detach 67b630e67f6a18c9e9be918d9b4337c960db1e9a
```

The SHA is pinned deliberately — the harness overlays upstream's generation loop and the
equivalence test asserts bit-identical output against it.

---

## 3. Hardware requirements

| | minimum | comfortable |
|---|---|---|
| VRAM | ~2.5 GB at `--batch-size 2` | 6 GB+ at `--batch-size 6` |
| Peak VRAM measured | 5.9 GB at batch 6, m=30 | scales roughly linearly with batch |
| Disk | ~6 GB (env + weights + data) | |
| CUDA | 12.x driver (torch 2.5.1+cu121) | |
| **Stable GPU time** | **~4–6 h for the full 708-window grid** | the actual constraint |

Throughput on the original 4050 was 22–30 s/window. A T4 or better should be
substantially faster and, more importantly, will not drop off the bus.

Non-NVIDIA or CPU-only will technically run but is not viable at this grid size.

---

## 4. Verify the port before trusting it

Run these in order. Each is cheap and catches a different class of transfer error.

```bash
# 1. environment + everything that needs no GPU
uv run pytest -q                        # expect 141 passed (2 skip without CUDA)

# 2. corpus arrived intact — expect 59 tickers, 123,479 bars, 0 zero-volume,
#    amount proxy error exactly 0.0
uv run python phase-a/scripts/build_dataset.py

# 3. the corpus is the one the golden numbers were measured on, and no ticker
#    trades on a date the rest of the universe does not
uv run python phase-a/scripts/audit_calendar.py     # expect 0 orphan rows

# 4. the overlay still matches upstream bit-for-bit (needs GPU)
uv run pytest tests/test_sampler_equivalence.py -q

# 5. numbers reproduce. Baselines are deterministic from the grid and seed, so
#    these must match. They are NOT restated here — they live in
#    phase-a/eval/golden.json and this test reads them:
uv run pytest tests/test_golden.py -q
```

**Steps 3 and 5 are the real test of the port.** If the fingerprint matches and the
numbers reproduce, the corpus, the window enumeration, the metric layer and the seeding
all survived the move.

The expected values deliberately do **not** appear in this document. They lived in three
places once — here, the Kaggle notebook, and a comment in the harness — and a corpus
correction on 2026-08-06 changed all of them. They now live only in
`phase-a/eval/golden.json`, which also carries a structural fingerprint of the corpus that
produced them, so a mismatch tells you *which* of the two drifted.

> **Corpus correction, 2026-08-06.** One orphan session (`ITC`, 2025-03-18) was dropped;
> the corpus is 123,479 rows, not 123,480, and the test split has 12 effective blocks, not
> 22. If you are restoring an archive made before that date, the fingerprint check will
> refuse it. Re-zip from a corrected `data/` rather than re-running `fetch_nse.py` —
> refetching re-runs back-adjustment and produces a third corpus. See report §17b.

---

## 5. Running the grid

```bash
uv run python phase-a/eval/calibrate.py --split test --batch-size 6
```

Watch it from another terminal:

```bash
uv run python phase-a/eval/watch_run.py
```

If it dies, nothing is lost — partials flush every 10 batches:

```bash
uv run python phase-a/eval/run_analysis.py results/<run-dir>   # analyse what exists
uv run python phase-a/eval/calibrate.py --split test --batch-size 6 \
    --resume results/<run-dir>                                  # continue
```

Resume is bit-identical to an uninterrupted run — each batch is seeded from its own
offset, and `tests/test_resume.py` asserts equality at `rtol=0, atol=0`.

### On unreliable hardware, flush more often

The partial is written every `--checkpoint-every` batches (default 10). At
`--batch-size 6` that is a flush every 60 windows — and the original laptop died at
30–60 windows, i.e. **usually before the first flush**, which is why an entire 42-window
attempt was lost.

```bash
uv run python phase-a/eval/calibrate.py --split test --batch-size 6 --checkpoint-every 2
```

That flushes every 12 windows, capping any loss at ~4 minutes of compute. The cost is a
few extra `.npy` writes of a couple of MB — negligible against re-running.

On a machine that is actually stable, leave it at the default.

---

## 6. Cloud alternative

Given the grid is ~5 hours of pure GPU with no interactivity, a rented or free cloud GPU
is a reasonable substitute for buying hardware reliability.

What changes:

- The repo is private — use a deploy key, a token, or upload an archive.
- `data/` still has to be uploaded; it is 17.6 MB.
- `setup_upstream.ps1` is PowerShell; use the two git commands from §2 instead.
- Everything else is platform-neutral Python. No Windows-specific code is in the harness.
- A 16 GB card comfortably takes `--batch-size 24` or more, which should cut wall-clock
  substantially below the 4–6 h measured here.

Free tiers usually enforce idle timeouts and session limits well under 5 hours, so plan
to use `--resume` across sessions rather than assuming one uninterrupted run — which is
exactly what the checkpointing was built for.

---

## 7. What is deliberately not in the repo

`CLAUDE.md` and `.claude/` are gitignored, as is `data/`. The project's engineering
constraints are cited inline at each point they are enforced in the code, so the codebase
is self-describing without them.
