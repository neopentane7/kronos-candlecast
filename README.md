# Kronos-NSE

**Does an open-source financial foundation model produce trustworthy forecast intervals on
Indian equities? Measured against a pre-registered bar: no.**

A calibration study of [Kronos-small](https://github.com/shiyu-coder/Kronos) (24.7M
parameters, pretrained on ~12B K-line records) on NSE daily bars at a 30-session horizon,
plus the evaluation harness it was measured with.

> **Research/education tool — scenario visualization, not investment advice.**

---

## Headline

Full test grid: 708 rolling windows, 59 tickers, **12 independent forecast dates**,
30-session horizon, 30-member ensembles.

| model | CRPS (fair) | interval score @80 | coverage @80 | 95% CI |
|---|---|---|---|---|
| random walk + drift | **67.24** | **462.8** | 0.837 | [0.800, 0.876] |
| flat line (`last_value`) | 89.05 | 890.5 | 0.001 | — |
| **Kronos-small, zero-shot** | **121.87** | **978.0** | **0.413** | [0.370, 0.456] |

Zero-shot Kronos is **81% worse than a Gaussian random walk on CRPS** and **111% worse on
interval score**. It is also worse than a flat line. Its 80% intervals contain the outcome
41% of the time.

The acceptance rule was written and committed on **2026-08-03**, five days before the grid
ran. It was not adjusted afterwards.

📄 **[Full technical report](phase-a/report/README.md)** — instrument validation,
decomposition, conformal study, and every correction made along the way.

---

## What was actually found

**1. The defect is uncertainty that stops growing.** Predicted-to-realized spread ratio
falls from **0.909 at h=1 to 0.481 at h=30**. One step ahead the cone is nearly the right
width; thirty steps ahead it is less than half. A random walk beats it precisely because
accumulating variance correctly is the one thing a random walk does.

**2. Measuring this correctly required fixing the instrument first.** At 30-member
ensembles, a *perfectly calibrated* forecaster measures **−5.04pp** below nominal at 80%
using NumPy's default quantile estimator. That artifact is 2.5× the acceptance band it
would have been judged against. Corrected with Hyndman–Fan type-6 quantiles and Ferro's
fair CRPS, both pinned by two-sided regression tests.

**3. Conformal calibration cannot repair it.** The best Kronos arm reaches 0.681 coverage
at relative width 0.211; the conformalized random walk reaches 0.796 at 0.146 — better
calibrated *and* 30% sharper.

**4. Regime-stratified conformal calibration is structurally infeasible at this horizon.**
Split conformal at 80% needs 9 exchangeable calibration residuals and 9 more to test on.
At a 30-session horizon that is 18 independent forecast dates — **2.2 years of data
consumed by the split alone**, and 4.64 years at 90%. Twelve dates exist. This is a
property of the horizon and the calendar, not of the corpus, and it forces adaptive
conformal inference rather than merely recommending it.

**5. Three preliminary findings were overturned by the full grid.** A 45-window probe
suggested Kronos was distinctively regime-blind. At 16× the sample its regime spread is
**0.117 against the random walk's 0.116** — identical — and its dispersion tracks input
volatility at ρ=0.3551 against a persistence ceiling of ρ=0.3552. Each retired claim is
recorded in place with the superseded reasoning intact.

---

## Reproduce it

The corpus is not committed (back-adjusted prices are rewritten by later corporate actions,
so a re-fetch produces a different corpus). Everything else is.

```bash
uv sync --all-groups
./phase-a/scripts/setup_upstream.ps1        # pinned upstream commit, read-only
uv run pytest -q                            # 169 tests

# baselines are pure NumPy and must reproduce exactly on any machine
uv run pytest tests/test_golden.py -q
```

Expected values live in [`phase-a/eval/golden.json`](phase-a/eval/golden.json) — one file,
read by both the test suite and the cloud notebook, carrying a structural fingerprint of
the corpus that produced them. See [SETUP.md](SETUP.md) to run it elsewhere.

```bash
# the full grid (~90 min on a T4)
uv run python phase-a/eval/calibrate.py --split test --batch-size 24 --checkpoint-every 5

# offline analysis, no GPU
uv run python phase-a/eval/run_analysis.py results/<run-dir>
uv run python phase-a/eval/diagnose.py    results/<run-dir>
```

---

## Engineering notes

Things that turned out to matter more than expected:

- **Bit-exact resume, verified on the target GPU.** Two identical runs and one
  killed-and-resumed run produce byte-identical ensembles on a T4 at the production batch
  size (`max |A−B| = 0.000e+00`). Kernel selection is shape-dependent, so this is checked
  at batch 24, not at a smaller smoke size.
- **Corpus fingerprinting.** A structural digest of row counts and date bounds per ticker,
  checked before a cloud run downloads model weights — a stale dataset fails in seconds
  rather than as an arithmetic mismatch hours later.
- **A calendar audit that found a real defect.** One ticker carried a session no other
  ticker had, which split the bootstrap blocks and inflated the reported effective sample
  size from 12 to 22 — silently, with no error and no failing schema check.
- **Verification that asserts its own preconditions.** Three checks in this repo could once
  have passed without testing anything. See §4a of the report.

---

## Layout

```
common/          shared preprocessing, schema, corpus fingerprint, results helpers
phase-a/
  eval/          harness, metrics, conformal prediction, diagnostics, figures
  scripts/       data fetch, dataset build, calendar audit, golden regeneration
  report/        the technical report
cloud/           Kaggle notebook for the GPU grid
pipeline/        serving design (Phase B — not yet built)
results/         <timestamp>_<git-sha>/ per run: metrics, ensembles, figures
tests/           169 tests
```

## Status

| | |
|---|---|
| A1 environment, A2 data pipeline | done — 59 tickers, 123,479 bars, 2018-01 → 2026-06 |
| A3 zero-shot gate | **done — FAIL**, see headline |
| A4 temperature sweep | harness built, run pending |
| A5 pre-registered conformal study | needs a validation-split grid |
| A6 bounded fine-tuning pilot | pending, kill rule fixed in advance |
| Phase B — nightly job + PWA | not built |

## Attribution

Upstream Kronos: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739), MIT licensed.
This repository is MIT licensed — see [LICENSE](LICENSE).
