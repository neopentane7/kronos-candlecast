# Kronos-NSE + CandleCast

Fine-tuning and **conformally calibrating** the open-source
[Kronos](https://github.com/shiyu-coder/Kronos) financial foundation model on Indian
NSE equities, then shipping a forecast product on top of it.

> **Research/education tool — scenario visualization, not investment advice.**

The project works to a fixed set of engineering constraints — a single canonical data
schema validated at both training and serving time, date-based splits with the test
period held back until final evaluation, `amount` defined as the `close * volume` proxy
everywhere, and all trading sessions sourced from an exchange calendar. Each is cited at
the point it is enforced in the code.

## Status

| Milestone | State |
| --- | --- |
| A1 — Environment + smoke test | done |
| A2 — Data pipeline | done — 54 tickers, 113,068 bars, 2018-01 → 2026-06 |
| A3 — Eval harness + zero-shot gate | not started |

## Data pipeline

```powershell
uv run python phase-a/scripts/fetch_nse.py        # -> data/parquet + data/manifest.json
uv run python phase-a/scripts/build_dataset.py    # -> data/csv + data/dataset_manifest.json
```

Prices are **split/bonus back-adjusted** (yfinance) in both training and serving, so the
two paths cannot drift apart across corporate actions; jugaad-data is an independent
cross-check (`--cross-check`) and the source of real turnover for the constraint-4
ablation. Rationale and measurements are in [common/preprocess.py](common/preprocess.py).

**Moving to another machine?** See [SETUP.md](SETUP.md) — the corpus in `data/` (17.6 MB)
must be copied rather than re-fetched, because back-adjusted prices are rewritten by later
corporate actions and a re-fetch would silently produce a non-comparable corpus.

## Setup (Windows / PowerShell)

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU (developed on an RTX 4050, 6GB).

```powershell
# 1. Clone the read-only upstream Kronos at its pinned commit
./phase-a/scripts/setup_upstream.ps1

# 2. Create the environment (uv fetches Python 3.10 itself)
uv sync --all-groups

# 3. Install git hooks
uv run pre-commit install

# 4. Smoke test: zero-shot Kronos-small forecast on NSE daily bars
uv run python phase-a/scripts/smoke_test.py
```

Torch comes from the CUDA 12.1 wheel index (`[[tool.uv.index]]` in
[pyproject.toml](pyproject.toml)); PyPI would install the CPU-only build on Windows.

## Checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

## Layout

```
common/          # shared preprocessing, schema, results helpers (train/serve parity)
phase-a/
  Kronos/        # upstream clone, read-only and gitignored (see setup_upstream.ps1)
  scripts/       # data fetch, dataset build, smoke test
  configs/       # 6GB fine-tuning config
  eval/          # baselines, calibration harness, conformal prediction
  report/        # mini-paper
pipeline/        # nightly forecast job (local GPU + GitHub Actions CPU)
results/         # <timestamp>_<git-sha>/results.json + figures for every run
tests/
```

## Attribution

Upstream Kronos: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739), MIT licensed.
