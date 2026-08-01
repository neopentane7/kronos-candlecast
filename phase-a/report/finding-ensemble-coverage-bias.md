# Finite-ensemble bias in empirical coverage

**Status:** measured and reproduced; mechanism derived and validated against simulation.
**Date:** 2026-07-31 · **Discovered during:** A3 metric-layer development

> Research/education tool — scenario visualization, not investment advice.

---

## 1. Statement

When prediction intervals are formed from the empirical quantiles of a finite ensemble,
a **perfectly calibrated** forecaster measures as **over-confident**. The measured
coverage falls below nominal by an amount that depends only on the ensemble size `m` and
the quantile estimator — not on the model.

At `m = 30` (the ensemble size specified for this project's evaluation) the shortfall at
a nominal 80% band is **−5.04 percentage points**. The intended acceptance band for A5 is
78–82%, i.e. ±2pp. **The measurement artifact is 2.5× the width of the acceptance
criterion it would have been judged against.**

---

## 2. Mechanism

Let `F` be continuous, `X₁…X_m` an i.i.d. ensemble from `F`, and `Y ~ F` a fresh,
independent observation.

`Y` and the ensemble members are **exchangeable**, so the rank of `Y` among the `m+1`
values is uniform on `{1, …, m+1}`. Therefore, exactly:

```
P(Y ≤ X₍ₖ₎) = k / (m + 1)
```

An interval spanning order statistics `j` through `k` covers `(k − j)/(m + 1)`. The
question is only which positions a given quantile estimator selects.

**NumPy's default** (`method="linear"`, Hyndman–Fan **type 7**) places quantile `p` at
position `p(m − 1) + 1`. A central `(1 − α)` band therefore spans `(1 − α)(m − 1)`
positions:

```
coverage_linear = (1 − α) · (m − 1) / (m + 1)
bias            = −2 (1 − α) / (m + 1)
```

**Weibull plotting positions** (`method="weibull"`, Hyndman–Fan **type 6**) place `p` at
`p(m + 1)`, spanning `(1 − α)(m + 1)` positions — giving coverage exactly `(1 − α)`.
That is precisely the estimator the rank argument implies.

---

## 3. Validation

Simulation: exactly calibrated standard normal, 400,000 observations per cell.

| m | nominal | predicted `(1−α)(m−1)/(m+1)` | simulated | Δ | weibull |
|---|---|---|---|---|---|
| 5 | 0.80 | 0.5333 | 0.5645 | +0.031 | 0.6672 |
| 10 | 0.80 | 0.6545 | 0.6625 | +0.008 | 0.8067 |
| 30 | 0.80 | 0.7484 | 0.7505 | +0.002 | 0.8013 |
| 100 | 0.80 | 0.7842 | 0.7842 | +0.000 | 0.8009 |
| 1000 | 0.80 | 0.7984 | 0.7978 | −0.001 | 0.7996 |
| 30 | 0.50 | 0.4677 | 0.4698 | +0.002 | 0.5027 |
| 30 | 0.90 | 0.8419 | 0.8458 | +0.004 | 0.9087 |

The closed form is accurate to <0.5pp for `m ≥ 30` and degrades at very small `m`, where
interpolation between order statistics departs from the pure rank argument.

---

## 4. The harder result: a ceiling on achievable coverage

The widest possible distribution-free central band from `m` samples runs between the
minimum and maximum, covering `(m − 1)/(m + 1)`. This is a **hard information limit**,
independent of model quality or estimator:

| m | max achievable coverage | 80% band? | 90% band? |
|---|---|---|---|
| 5 | 0.667 | **impossible** | **impossible** |
| 10 | 0.818 | barely | **impossible** |
| 19 | 0.900 | yes | exactly at the limit |
| 30 | 0.935 | yes | yes |
| 100 | 0.980 | yes | yes |

This explains the weibull column above: at `m = 5` weibull returns 0.6672, which is not
a defect of the estimator — it is the ceiling. **An 80% interval cannot be constructed
from 5 samples at all.** For a 90% band, `m ≥ 19` is a hard minimum, and for the band to
be estimable rather than merely attainable, substantially more.

---

## 5. Impact on this project

- **Fixed.** `QUANTILE_METHOD = "weibull"` is now the default in
  `phase-a/eval/metrics.py`, recovering 0.8013 at `m = 30` with no additional compute.
- **Two-sided regression test** in `tests/test_metrics.py`: the corrected estimate must
  sit within 1pp of nominal, *and* the naive estimator must still under-cover, so the
  correction cannot be silently dropped.
- **A5 becomes feasible.** Both `m = 30` and `m = 50` land outside the 78–82% band on
  sampling alone under the default estimator. With the correction, the band measures what
  it was meant to measure.
- **Averted a false result.** Uncorrected, the zero-shot gate would have reported
  spurious over-confidence, and the conformal layer in A5 would have "corrected" an
  artifact of our own instrument — producing a headline reliability diagram that looked
  like a success and was measuring nothing.

---

## 6. Upstream issue #254 — revised against the retrieved issue text

**Superseded.** An earlier draft of this section noted that upstream's
`auto_regressive_inference` defaults to `sample_count=5`
(`phase-a/Kronos/model/kronos.py:389`) and observed that a measured 52% at nominal 80%
inverts to an implied `m ≈ 4.7`. That coincidence is **dead**: the issue was retrieved and
the reporter states their configuration.

**Reporter's stated configuration** (retrieved 2026-08-01): `sample_count: 8`,
Kronos-small + Kronos-Tokenizer-base, 1-minute NIFTY bars, lookback 240, horizons 3/5/10,
`T=1.0`, `top_p=0.9`, six trading days (2026-03-24 → 2026-04-10), offline evaluation for
an intraday options workflow.

At **m = 8**, nominal 80%, default type-7 quantiles:

```
closed form  (0.8)(7)/(9) = 0.622
simulated                   0.643
hard ceiling (m-1)/(m+1)  = 0.778
```

So the instrument accounts for roughly **17pp of the 28pp gap** — *if* the 52% refers to
an 80% central interval. It does not account for all of it, and the residual is either
genuine miscalibration or a difference in definition.

**Still unknown, and still worth asking:** the band definition and the quantile function.
The surrounding analysis in the issue is directional (hit-rates, keep/filter decisions),
so "coverage" may mean directional hit-rate rather than interval coverage. **If it is
directional, none of this applies** — and ~52% on 1-minute bars is approximately the
chance-level null, which would be a different and much less interesting finding.

Note also the ceiling: at m = 8 a 90% band is **unconstructible**, since (m−1)/(m+1) =
0.778 < 0.90. Any 90% interval reported at that ensemble size is a clamped min/max.

| measured coverage @ nominal 80% | implied m |
|---|---|
| 0.52 | ~4.7 |
| 0.60 | ~7.0 |
| 0.70 | ~15.0 |
| 0.75 | ~31.0 |

### Public-entry-point defaults are worse than the internal ones

| entry point | `sample_count` | `top_p` |
|---|---|---|
| `auto_regressive_inference` (internal) | 5 | 0.99 |
| `KronosPredictor.predict` / `predict_batch` (public) | **1** | 0.9 |

At `sample_count=1` no interval exists at all. A user reaching for the documented API and
asking about uncertainty gets a single path by default.

---

## 6a. A second mechanism: nucleus-sampling truncation

Order-statistic bias is not the only route to under-coverage, and it is the only one a
quantile estimator can fix.

`top_p < 1.0` truncates the token distribution's tail at **every autoregressive step**.
The sampled ensemble is therefore drawn from a narrower distribution than the model's own
predictive distribution, and the resulting bands under-cover **independently of `m` and
independently of the quantile estimator**. Over a 30-step horizon the truncations compound.

Issue #254 ran at `top_p=0.9`, so both mechanisms were active simultaneously.

**This extends the interpretability claim.** A coverage figure is uninterpretable unless
it states:

```
m  ·  quantile estimator  ·  T  ·  top_p  ·  top_k
```

Reporting coverage without the sampling policy is like reporting a temperature without a
scale. The effect is quantified on real data by the `--sweep-top-p` axis in
`phase-a/eval/calibrate.py`; the production sampling policy is chosen from that table
rather than inherited from a default.

---

## 7. Novelty assessment — be honest here

**The statistics is textbook, not novel.** Distribution-free tolerance intervals from
order statistics go back to Wilks (1941); plotting positions to Weibull (1939); the
`(m+1)` rank argument is standard. The ensemble-size dependence of verification scores is
well established in numerical weather prediction.

**What may be underappreciated** is that this contaminates coverage claims in the
time-series-foundation-model literature, where small `sample_count` defaults are common,
and published coverage numbers frequently omit both the ensemble size and the quantile
estimator — the two quantities needed to interpret them.

The defensible claim is therefore narrow and checkable: *a coverage figure reported
without stating `m` and the quantile estimator is not interpretable, and at small `m` may
be measuring ensemble size rather than calibration.*

---

## 8. Connection to conformal prediction

The same `(n+1)` correction is why split conformal prediction uses the
`⌈(n+1)(1−α)⌉`-th smallest calibration residual rather than the `n(1−α)`-th. It is the
identical exchangeability argument. The project's own conformal layer already embodies
the correction on the calibration side.

**This raises a question that matters for A5's three-way comparison.** If conformal
calibration is fitted using a biased quantile estimator on the calibration set, it will
learn to widen the bands to compensate — *absorbing* the artifact. The raw path would
still show it. That would make "conformal fixes the miscalibration" partly an artifact of
the estimator rather than a property of the method, and it would inflate the apparent
improvement in the headline reliability diagram.

**Action for A5:** run the three-way comparison under both estimators and report whether
the conformal improvement survives the correction.

---

## 9. Open questions

**Q1 — CRPS analogue. ANSWERED: yes, it matters.** The naive estimator divides the spread
term by `2m²` where the unbiased form uses `2m(m−1)`, inflating the score by roughly
`1/(m−1)`. Against the analytic value for a calibrated N(0,1), `1/√π = 0.56419`:

| m | naive | fair | inflation |
|---|---|---|---|
| 8 | 0.6347 | 0.5642 | 1.125 |
| 30 | 0.5830 | 0.5642 | 1.033 |
| 100 | 0.5709 | 0.5652 | 1.010 |

Fair recovers the analytic value at every `m`; naive does not. The inflation does not
cancel in a three-way comparison and makes CRPS incomparable across studies at different
ensemble sizes. Ferro's fair estimator is now the default in `metrics.py`, with the naive
form retained for the dual-estimator run.

**Q2 — Does the conformal layer absorb the artifact?** Open. See §8; encoded as the A5
dual-estimator protocol.

**Q3 — What did issue #254 measure? PARTIALLY ANSWERED.** `sample_count=8` and
`top_p=0.9` are now known (§6). The band definition and quantile function remain unknown,
and the directional-hit-rate reading remains live.

**Q4 — Do published TSFM calibration studies state `m` and the estimator?** Open, and the
most valuable of these to answer. If they routinely do not, the interpretability claim in
§6a generalises well beyond this one model.

**Q5 — Interaction with the block bootstrap. FIRST-ORDER ANSWER: no interaction
expected.** The bias shifts the mean of the coverage indicator, not its dependence
structure, so interval *width* should be unaffected. Confirmed empirically as a column in
the `--sweep-top-p` output rather than left as an assumption.

**Q6 — new.** Does nucleus truncation interact with conformal calibration? A conformal
layer fitted on truncated samples will widen bands to compensate, exactly as it would for
the order-statistic bias — so the same dual-run logic may be needed across the sampling
policy, not just the quantile estimator.

### Literature pointers

- Wilks (1941), *Determination of sample sizes for setting tolerance limits* —
  distribution-free tolerance intervals from order statistics.
- Hyndman & Fan (1996), *Sample quantiles in statistical packages* — the nine quantile
  types; NumPy's `linear` is type 7, `weibull` is type 6.
- Cunnane (1978), *Unbiased plotting positions — a review*, J. Hydrology.
- Hamill (2001), *Interpretation of rank histograms for verifying ensemble forecasts*,
  Mon. Wea. Rev. — rank histograms and finite-ensemble effects.
- Ferro, Richardson & Weigel (2008), *On the effect of ensemble size on the discrete and
  continuous ranked probability scores* — the CRPS analogue of this result.
- Zamo & Naveau (2018), *Estimation of the CRPS with limited information*.
- Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World* — the `(n+1)`
  correction in conformal prediction.
- Angelopoulos & Bates, *A gentle introduction to conformal prediction*.

### Search terms

`plotting position quantile estimator coverage bias` · `Hyndman Fan quantile types` ·
`distribution-free tolerance interval order statistics` · `fair CRPS ensemble size` ·
`rank histogram ensemble size bias` · `conformal prediction finite sample n+1 correction`

---

## 10. Reproduction

```powershell
uv run pytest tests/test_metrics.py -k ensemble -v
```

The simulation scripts behind the tables in §3 and §4 sweep `m` and the nominal level for
an exactly calibrated normal, comparing `numpy.quantile` methods against the closed form
in §2.
