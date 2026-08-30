# kronos-calibrate

**An evaluation harness for probabilistic time-series forecasters, extracted from the
[Kronos-NSE](../README.md) calibration study so it can be pointed at something else.**

It does not load a model, call a sampler, or know what produced the numbers it reads. It
takes one file — sampled forecast paths plus the outcomes they were predicting — and
returns the calibration picture: coverage with honest confidence intervals, proper scores
with their bias removed, the horizon decomposition, regime stratification, and a paired
comparison against whatever baseline you name.

> **Research/education tool — scenario visualization, not investment advice.**

---

## Why this exists

Phase A set out to measure a foundation model and spent its first week discovering the
instrument was wrong. Two of those corrections are why this is packaged rather than left
in the study.

**1. Small ensembles under-report coverage, and the shortfall looks like a model defect.**
Form an 80% interval from the empirical quantiles of a 30-member ensemble using NumPy's
default estimator, and a *perfectly calibrated* forecaster measures **5.04 points below
nominal**. The bias depends only on the ensemble size and the estimator — not on the
model, the data, or the horizon. Phase A's acceptance band was ±2pp, so the artifact was
2.5× the criterion it would have been judged against, in the direction that convicts.

The fix is Weibull plotting positions (Hyndman–Fan type 6), which place the p-th quantile
at rank p(m+1) — what the order-statistic argument actually implies. You can watch it
happen on a forecaster whose calibration is known by construction:

```
the estimator artifact, on this calibrated ensemble (m=30, nominal 0.80):
   weibull: 0.8251
    linear: 0.7786  <- numpy default
  difference: 4.65 pp of coverage, from the estimator alone.
```

There is a second trap behind it. `np.quantile` will happily return a "90% interval" from
a 12-member ensemble; it clamps to the min and max and reports (m−1)/(m+1) as though it
were an estimate. `assert_band_feasible` refuses instead.

**2. Naive ensemble CRPS is inflated by roughly 1/(m−1).** Ferro's fair estimator removes
it. Both numbers are printed side by side here, so the size of the correction is visible
rather than taken on trust — on the Phase A grid it is 123.16 → 121.87 for one arm and
69.50 → 67.24 for another, enough to reorder close comparisons.

**3. And the thing the tool will not let you claim.** Coverage is a mean over
*independent forecast dates*, not over windows. Fifty-nine tickers priced on one date
share a market; thirty horizon steps share a path. Phase A's grid has 708 windows and
**12** independent dates — a 59× difference in the denominator. Every interval here is a
block bootstrap over `block_ids`, and the block count is printed next to every figure,
because a coverage number without its effective sample size is not a measurement.

---

## Quickstart

No corpus required. The example is synthetic and seeded.

```bash
python make_example.py
python kronos_calibrate.py example_ensembles.npz --baseline oracle
```

`make_example.py` builds three forecasters whose true calibration is known:

| arm | construction | what it should measure |
|---|---|---|
| `oracle` | draws from exactly the law that generated the outcome | coverage on nominal — a test of the instrument, not of a model |
| `too_tight` | the same draws, spread shrunk 0.91 → 0.48 across the horizon | the under-propagation signature Phase A found in Kronos |
| `flat` | every member equal to the last close | zero spread, zero coverage — a degenerate arm to check the metrics point the right way |

Outcomes carry a market factor shared within each forecast date, so the block bootstrap
has something real to account for.

```
example_ensembles.npz: 480 windows x 30 steps, m=30, 12 independent forecast dates
estimators: weibull quantiles, fair CRPS

split-conformal feasibility at this block count
  50%  needs   6 blocks  -> yes
  80%  needs  18 blocks  -> NO
  90%  needs  38 blocks  -> NO

model                         CRPS  CRPS naive      IS@80   cov@80            95% CI
flat                         5.232       5.232      52.32   0.0000     [0.000,0.000]
oracle                       3.697       3.830      24.17   0.8251     [0.795,0.855]
too_tight                    3.879       3.965      26.27   0.6610     [0.618,0.702]

spread ratio (predicted / realized), where history was supplied
  oracle                   h=1 0.926   h=max 1.105
  too_tight                h=1 0.842   h=max 0.530
```

The oracle's interval contains 0.80. `too_tight` falls with horizon where the oracle does
not — that shape, not the coverage number, is what identifies under-propagation as
opposed to a location error.

---

## The file format

This is the whole integration surface. **If you can write this file from your own model,
every number above is available to you** — the toolkit never asks where it came from.

`n` = windows, `h` = horizon steps, `m` = ensemble members, `L` = lookback length.

| array | shape | dtype | required | meaning |
|---|---|---|---|---|
| `y_close` | `(n, h)` | float | **yes** | what actually happened |
| `block_ids` | `(n, h)` | int | **yes** | which independent forecast date each window belongs to |
| `ens__<name>` | `(n, h, m)` | float | **yes**, at least one | sampled paths for the forecaster called `<name>` |
| `history_close` | `(n, L)` | float | no | the lookback; enables the spread-ratio decomposition |
| `atr_tercile` | `(n,)` | int | no | volatility stratum; enables the regime table |
| `atr_pct` | `(n,)` | float | no | the underlying stratifier, carried for reference |
| `tickers` | `(n,)` | str | no | provenance |
| `start_dates` | `(n,)` | str | no | provenance |

Three things worth getting right:

- **`block_ids` is per-window, broadcast across the horizon.** All `h` entries of a row
  must be equal — the row is one forecast, made on one date. The toolkit counts unique
  ids as its sample size, so a per-*window* id here silently restores the
  over-optimistic denominator this whole design exists to avoid.
- **The ensemble is *paths*, not per-step marginals.** Member `k` should be a coherent
  trajectory. Marginals are enough for coverage and CRPS but wrong for anything that
  reads across the horizon.
- **The `ens__` prefix names the arm.** `ens__my_model` reports as `my_model`. Add as
  many as you like; they are scored on identical windows, which is what makes the paired
  bootstrap valid.

Shapes are validated on load. A mismatched file still produces numbers, and they are
wrong in ways nothing downstream would notice.

---

## Feasibility, before you spend the data

The report starts with a question most conformal studies never ask: *can this dataset
support the band I want?*

Split conformal at level 1−α needs ⌈(2−α)/α⌉ exchangeable calibration residuals, and as
many again to test the result on. In the unit that matters — independent forecast dates —
an 80% band needs 18 and a 90% band needs 38. On daily bars at a 30-session horizon that
is **2.2 and 4.6 years of data consumed by the split alone**. Phase A had 12 dates. Its
80% conformal result therefore cannot carry a finite-sample guarantee, and no amount of
extra tickers fixes it, because tickers are not the denominator.

This is a property of the horizon and the calendar, not of any corpus. Running it first
costs nothing and occasionally saves a study.

---

## API

```python
import kronos_calibrate as kc

ens = kc.Ensembles("ensembles.npz")  # validated on load
rep = kc.report(ens, baseline="random_walk_drift")
```

| function | what it answers |
|---|---|
| `summarize(obs, ens, block_ids)` | the full metric block: CRPS (fair and naive), interval score, pinball, point errors, coverage at 50/80/90 with block-bootstrap CIs |
| `crps`, `interval_score`, `coverage_indicator` | the individual proper scores, per observation |
| `band_bounds(ens, level)` | a central band, refusing levels the ensemble size cannot express |
| `block_bootstrap_ci(values, block_ids)` | a CI that resamples dates, not rows |
| `paired_block_bootstrap_diff(a, b, block_ids)` | is arm A's score different from arm B's, on shared windows |
| `spread_ratio_by_horizon(obs, ens, history)` | predicted width over realized width, per step — the under-propagation curve |
| `z_ratio_table(obs, ens)` | is the miscalibration scale or shape? Constancy across levels rejects bias-dominated error |
| `horizon_exponent(obs, ens)` | fits CRPS ~ h^k; a diffusing forecaster gives k ≈ 0.5 |
| `tercile_table(obs, ens, terciles, block_ids)` | coverage by volatility regime |
| `fit_scale` / `apply_scale` / `conformal_quantile` | split conformal, with the finite-sample ⌈(n+1)(1−α)⌉/n quantile |
| `min_samples_for_band(alpha)` / `feasibility(blocks)` | what this dataset can and cannot certify |

Implementations live in [`phase-a/eval/`](../phase-a/eval/) and are re-exported here, so
there is exactly one copy of each and the study's own regression tests keep pinning them.
[`tests/`](../tests/) covers them: `test_metrics.py`, `test_conformal.py`,
`test_analysis.py`, `test_diagnose.py`, `test_figures.py`, `test_golden.py`, plus
`test_kronos_calibrate.py` for this entry point.

---

## Provenance

Pointed at the Phase A test grid, the tool reproduces the study's headline exactly:

```
ensembles.npz: 708 windows x 30 steps, m=30, 12 independent forecast dates

model                         CRPS  CRPS naive      IS@80   cov@80            95% CI
kronos_zeroshot            121.869     123.163     978.04   0.4126     [0.371,0.455]
last_value                  89.045      89.045     890.45   0.0008     [0.000,0.001]
random_walk_drift           67.241      69.496     462.84   0.8370     [0.799,0.876]

spread ratio (predicted / realized)
  kronos_zeroshot          h=1 0.909   h=max 0.481
  random_walk_drift        h=1 0.984   h=max 1.130
```

Those point estimates are pinned by `phase-a/eval/golden.json` against corpus fingerprint
`0a990f9881418544` (59 tickers, 123,479 rows). The CI endpoints are bootstrap estimates
and move in the third decimal with `--seed`.

Full findings — what the failure decomposes into, and what conformal calibration could
and could not repair: [`phase-a/report/`](../phase-a/report/README.md).

---

## Requirements

`numpy` and `scoringrules`, both already in the parent project's lockfile:

```bash
uv run python kronos-calibrate/kronos_calibrate.py <file.npz>
```
