# Kronos-NSE — Phase A technical report

**Calibrating a financial foundation model on Indian equities, and the measurement
problems that had to be solved first.**

**Status:** A1–A2 accepted · A3 in progress · A4–A7 not started
**Last revised:** 2026-08-01

> Research/education tool — scenario visualization, not investment advice.

---

## How to read this

This report unifies four working documents: the finite-ensemble measurement finding, the
zero-shot evaluation results, the calibration method study, and the pre-registered A5
protocol. It is organised so the **instrument is validated before any model claim is
made**, because two defects in the measurement layer were large enough to have produced
false headline results.

Every number is traceable to a committed run. Figures marked *preliminary* come from
45- and 15-window subsamples, not the full 708-window grid; they are adequate for the
decisions they feed and are **not** the A3 acceptance artifact.

Where an earlier conclusion has been overturned by later measurement, the correction is
stated in place rather than silently applied. There are three such corrections, marked
**CORRECTED**.

| Part | Question |
|---|---|
| [I](#part-i--validating-the-instrument) | Does the measurement layer measure what it claims? |
| [II](#part-ii--zero-shot-results) | How good is pretrained Kronos-small on NSE daily bars? |
| [III](#part-iii--can-calibration-repair-it) | Can conformal calibration repair what we found? |
| [IV](#part-iv--a5-protocol-pre-registered) | What will A5 do, decided before results exist? |
| [V](#part-v--upstream-contribution) | What is worth contributing back? |

---

# Part I — Validating the instrument

Two estimator defects were found during harness development. Each is large enough to have
inverted a conclusion. Both are corrected, both are pinned by two-sided regression tests.

## 1. Finite-ensemble coverage bias

When intervals are formed from the empirical quantiles of a finite ensemble, a **perfectly
calibrated** forecaster measures as over-confident. The shortfall depends only on ensemble
size `m` and the quantile estimator — **not on the model.**

At `m = 30`, the ensemble size this project specifies, the shortfall at nominal 80% is
**−5.04pp**. The A5 acceptance band is ±2pp. **The artifact is 2.5× the criterion it would
have been judged against.**

### Mechanism

Let `F` be continuous, `X₁…X_m` an i.i.d. ensemble from `F`, and `Y ~ F` fresh and
independent. `Y` and the ensemble are **exchangeable**, so `Y`'s rank among the `m+1`
values is uniform, and exactly:

```
P(Y ≤ X₍ₖ₎) = k / (m + 1)
```

An interval spanning order statistics `j…k` covers `(k − j)/(m + 1)`. Everything follows
from which positions the estimator selects.

NumPy's default (`linear`, Hyndman–Fan **type 7**) places `p` at `p(m − 1) + 1`:

```
coverage_linear = (1 − α)(m − 1)/(m + 1)
bias            = −2(1 − α)/(m + 1)
```

Weibull positions (`weibull`, **type 6**) place `p` at `p(m + 1)`, giving coverage exactly
`(1 − α)`. That is the estimator the rank argument implies.

### Validation — closed form against simulation

400,000 observations per cell, exactly calibrated standard normal:

| m | nominal | predicted | simulated | Δ | weibull |
|---|---|---|---|---|---|
| 5 | 0.80 | 0.5333 | 0.5645 | +0.031 | 0.6672 |
| 10 | 0.80 | 0.6545 | 0.6625 | +0.008 | 0.8067 |
| 30 | 0.80 | 0.7484 | 0.7505 | +0.002 | **0.8013** |
| 100 | 0.80 | 0.7842 | 0.7842 | +0.000 | 0.8009 |
| 1000 | 0.80 | 0.7984 | 0.7978 | −0.001 | 0.7996 |
| 30 | 0.50 | 0.4677 | 0.4698 | +0.002 | 0.5027 |
| 30 | 0.90 | 0.8419 | 0.8458 | +0.004 | 0.9087 |

Accurate to <0.5pp for `m ≥ 30`; degrades at very small `m` where interpolation departs
from the pure rank argument.

### A hard ceiling on achievable coverage

The widest distribution-free central band from `m` samples runs min-to-max, covering
`(m − 1)/(m + 1)`. This is an information limit, independent of model or estimator:

| m | max achievable | 80% band | 90% band |
|---|---|---|---|
| 5 | 0.667 | **impossible** | **impossible** |
| 10 | 0.818 | barely | **impossible** |
| 19 | 0.900 | yes | at the limit |
| 30 | 0.935 | yes | yes |
| 100 | 0.980 | yes | yes |

This explains the weibull column at `m = 5`: 0.6672 is not an estimator defect, it is the
ceiling. **An 80% interval cannot be constructed from 5 samples at all.** `np.quantile`
clamps silently past this point, returning the ceiling as though it were a measurement —
guarded by `assert_band_feasible`.

## 2. CRPS estimator bias

The same disease in a different form. The naive estimator divides the spread term by `2m²`
where the unbiased form uses `2m(m−1)`, inflating the score by roughly `1/(m−1)`. Against
the analytic value for a calibrated N(0,1), `1/√π = 0.56419`:

| m | naive | fair | inflation |
|---|---|---|---|
| 8 | 0.6347 | 0.5642 | 1.125 |
| 30 | 0.5830 | 0.5642 | 1.033 |
| 100 | 0.5709 | 0.5652 | 1.010 |

Fair recovers the analytic value at every `m`. The inflation does not cancel in a
multi-arm comparison and makes CRPS incomparable across studies at different ensemble
sizes. Ferro's fair estimator is the default; the naive form stays reachable for the
dual-estimator run.

**Externally confirmed.** Ferro et al. (2008) define the adjusted CRPS with exactly the
`2N(N−1)` denominator, and it is "fair and unbiased with ensemble size" under
exchangeability.

## 3. Harness validation

The theoretical `h^0.5` diffusion benchmark is not assumed — it is measured on our own
random-walk arm, through the same code path, on the same windows:

```
random_walk_drift exponent = 0.4983     band [0.45, 0.55]     PASS
kronos_zeroshot   exponent = 0.639      excess = +0.141
```

The harness recovers the known answer where one exists, which is what licenses citing the
Kronos figure. Baseline reproduction from the grid and seed alone is bit-exact against the
recorded run (CRPS 39.9271, coverage 0.8807).

## 4. What this averted

Uncorrected, the zero-shot gate would have reported spurious over-confidence, and the A5
conformal layer would have "corrected" an artifact of our own instrument — producing a
headline reliability diagram that looked like success and measured nothing.

**Implication beyond this project:** a coverage figure is uninterpretable without
`m · quantile estimator · T · top_p · top_k`. The statistics here is textbook — Wilks
(1941), Weibull (1939), the `(m+1)` rank argument — and the ensemble-size dependence of
verification scores is well established in numerical weather prediction. What may be
underappreciated is that it **contaminates coverage claims in the time-series-foundation-
model literature**, where small `sample_count` defaults are common and published coverage
numbers routinely omit both quantities needed to interpret them.

---

# Part II — Zero-shot results

**Preliminary.** Test split, 45 windows subsampled from 708, 32 tickers, **12 distinct
forecast start dates** (the bootstrap blocks). Kronos-small, lookback 400 → horizon 30,
m = 30, T = 1.0, top_k = 0, top_p = 0.9. Weibull quantiles, fair CRPS. Cost 33.5 s/window,
3349 MB peak VRAM at batch 8 × 30.

## 5. Headline

| model | CRPS (fair) | MAPE | cov@50 | cov@80 | cov@90 |
|---|---|---|---|---|---|
| random_walk_drift | **39.93** | 0.0399 | 0.608 | **0.881** | 0.953 |
| last_value | 50.44 | 0.0365 | 0.002 | 0.002 | 0.002 |
| **kronos_zeroshot** | **67.57** | **0.0692** | **0.240** | **0.422** | **0.531** |

**Zero-shot Kronos is worse than a flat line on CRPS and worse than a Gaussian random walk
on every metric.** MAPE 6.9% against 4.0%, so the point forecast is also worse — this is
not only a calibration failure.

Verified as a real result, not a harness fault: the median forecast starts within **0.71%**
of the last observed close, confirming the sampler and normalisation round-trip correctly.

The naive/fair CRPS ratios independently corroborate under-dispersion: 1.045 for the random
walk (against the 1.034 a calibrated m = 30 forecaster would show), 1.017 for Kronos (its
spread term is shrunk), 1.000 for the degenerate baseline (no spread term to correct).

## 6. Is the miscalibration shape or scale?

This decides whether conformal calibration is the right instrument. A pure scale error
gives a constant `z_achieved / z_nominal`; a wrong shape does not.

| nominal | achieved | z_nominal | z_achieved | ratio |
|---|---|---|---|---|
| 0.50 | 0.2400 | 0.6745 | 0.3055 | 0.4529 |
| 0.80 | 0.4222 | 1.2816 | 0.5566 | 0.4343 |
| 0.90 | 0.5311 | 1.6449 | 0.7243 | 0.4403 |

**Mean 0.4425, sd 0.0095, spread 0.0186.** The distribution is approximately the right
shape and **uniformly too tight by 2.26×**. The random-walk control gives 1.27 / 1.22 /
1.21 — near-constant, slightly over-wide, as a correctly-specified Gaussian forecaster
should look.

This is the most encouraging result in the run: a uniform per-horizon scale correction is
exactly what split conformal produces.

### The test is one-sided — read it as rejection, not confirmation

Constancy **rejects** bias-dominated error but does **not confirm** pure scale error. A
bias-plus-scale mixture also reads flat:

| forecaster | ratios @ 50/80/90 | spread | vs our 0.0186 |
|---|---|---|---|
| pure scale, σ = 0.44 | 0.440 / 0.440 / 0.440 | 0.000 | consistent |
| random-sign bias 0.75 + scale 0.75 | 0.570 / 0.578 / 0.584 | 0.014 | **also consistent — not excluded** |
| pure bias 1.1, correct scale | 0.564 / 0.609 / 0.636 | 0.072 | excluded |

The discriminating signature is **monotonicity** — location bias makes the ratio climb
with nominal level. Our sequence (0.4529, 0.4343, 0.4403) is **non-monotone**, i.e. noise
around a constant. Encouraging, but underpowered at 12 blocks. `z_ratio_table` returns a
`monotone_increasing` flag; check it on the full grid.

## 7. Regime dependence

Terciles by ATR over the **lookback**; coverage measured on the **following** 30 sessions.

| regime | n | CRPS | IS@80 | MAPE | cov@50 | cov@80 | cov@90 |
|---|---|---|---|---|---|---|---|
| calm | 15 | 102.25 | 857.8 | 0.0735 | 0.122 | **0.260** | 0.356 |
| mid | 15 | 68.89 | 545.0 | 0.0605 | 0.264 | 0.456 | 0.544 |
| volatile | 15 | 31.58 | 205.3 | 0.0735 | 0.333 | **0.551** | 0.693 |

**Coverage at nominal 80% spans 0.260 to 0.551 — 29 points.** Ignore the CRPS column when
comparing regimes: it is in price units while MAPE is roughly flat, so its ordering
reflects price level, not skill. Coverage is scale-free and the pattern is real.

**Mechanism.** Kronos normalises each window by its own statistics, so cone width is
anchored to **trailing** volatility. Volatility mean-reverts. A calm lookback is therefore
disproportionately followed by a relatively more volatile future, and a cone sized for the
calm past under-covers. **The model extrapolates the recent past rather than anticipating
regime change.**

### Control: general mechanism, Kronos-specific magnitude

The mechanism punishes **any** forecaster whose dispersion comes from the lookback,
including our baseline. Same windows, same tercile cuts:

| forecaster | calm | mid | volatile | calm − volatile |
|---|---|---|---|---|
| random_walk_drift | 0.853 | 0.887 | 0.902 | **−0.049** |
| kronos_zeroshot | 0.260 | 0.456 | 0.551 | **−0.291** |

Same sign, **16.8%** of the magnitude. General in kind, Kronos-specific in degree.

## 8. Error growth with horizon

| step | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| CRPS | 11.14 | 38.50 | 57.30 | 78.42 | 99.61 |
| IS@80 | 79.0 | 304.7 | 426.5 | 634.8 | 822.6 |

**CRPS ∝ h^0.639** against the h^0.5 of pure diffusion (validated at 0.4983 on our own
baseline, §3). The excess indicates **drift error compounding across steps** rather than
variance merely widening — corroborated by per-window diagnostics where the model made
large directional calls that missed (GODREJCP predicted +4.3% against −7.8% actual;
ULTRACEMCO −7.6% against +0.6%).

## 9. Sampling-policy sweep

15 stratified windows, identical except `top_p`:

| top_p | cov@80 | 95% CI | CI width | rel. width | CRPS (fair) | IS@80 |
|---|---|---|---|---|---|---|
| 0.90 | 0.4689 | [0.290, 0.658] | 0.367 | 0.1094 | **56.15** | **428.3** |
| 0.99 | 0.4689 | [0.265, 0.695] | 0.430 | 0.1233 | 60.46 | 464.4 |
| 1.00 | 0.4933 | [0.289, 0.710] | 0.421 | 0.1284 | 60.63 | 459.7 |

**Read the intervals before the point estimates.** They are 0.37–0.43 wide and overlap
almost completely: the three policies are **statistically indistinguishable on coverage**
here. Only band width — an average rather than a proportion — is estimated precisely
enough to compare, and it moves monotonically.

The decisive argument is arithmetic, not statistical:

```
band widening required   2.260×      (§6)
available from top_p     1.174×      (0.1094 → 0.1284)
shortfall                1.926×
```

Nucleus truncation cannot close the gap even if free, and widening makes CRPS and interval
score **worse** — a wider band around a wrong location is penalised by proper scoring
rules. **Decision: hold `top_p` at 0.9**, recorded as `PRODUCTION_TOP_P`. The A5 comparison
runs once rather than twice.

*Caveat: 15 windows over 12 blocks. The claim is "not the dominant mechanism", not "no
effect".*

---

# Part III — Can calibration repair it?

Tested on CPU before committing GPU time to A5, using a synthetic forecaster tuned to the
measured pathology (marginal 0.456 vs 0.422; regimes 0.322/0.454/0.593 vs 0.260/0.456/0.551;
z-ratio 0.482 vs 0.443).

## 10. Method comparison

Nominal 80%:

| arm | marginal | calm | mid | volatile | spread | rel. width |
|---|---|---|---|---|---|---|
| raw | 0.459 | 0.333 | 0.443 | 0.602 | 0.269 | 0.064 |
| marginal conformal | 0.794 | 0.648 | 0.805 | 0.929 | **0.281** | 0.142 |
| **Mondrian (per regime)** | 0.796 | **0.799** | **0.790** | **0.798** | **0.009** | **0.123** |

**Conformal calibration works** — marginal coverage 0.459 → 0.794, confirming the
shape-versus-scale diagnosis.

**Marginal conformal fixes the average and nothing else.** The regime spread goes
0.269 → 0.281.

**CORRECTED — Mondrian is *narrower*, not wider.** An earlier revision asserted that
stratifying "widens every interval". The width half was wrong: 0.123 against 0.142, because
marginal conformal must over-widen the volatile stratum in order to lift the calm one. The
real cost of stratification is **calibration-set size only** — which matches the published
description of Mondrian CP as reducing calibration-set size and increasing variance around
the target, not as widening intervals.

## 11. Choosing a normalizer

**CORRECTED — normalized conformal was wrongly rejected.** An earlier revision recorded it
as "does not work, and cannot", on the strength of a single badly chosen normalizer. The
standard construction uses an **auxiliary** difficulty estimator; I had used the model's
own anchor, which is the degenerate case. Re-tested:

| normalizer | marginal | calm | volatile | spread | rel. width |
|---|---|---|---|---|---|
| the model's own anchor (trailing vol) | 0.789 | 0.635 | 0.928 | 0.293 | 0.075 |
| **mean-reversion aware blend** | 0.792 | **0.794** | **0.783** | **0.015** | **0.065** |
| long-run volatility only | 0.795 | 0.936 | 0.635 | 0.301 | 0.065 |
| Mondrian, for comparison | 0.794 | 0.797 | 0.797 | 0.011 | 0.067 |

**Failure is two-sided**, which is the more useful finding. Normalizing by the model's own
anchor reproduces the bias — it is the signal whose mis-prediction caused the defect.
Discarding local information entirely over-corrects and **inverts** the spread: calm now
over-covers at 0.936 while volatile under-covers at 0.635. The normalizer must be *right*,
not merely different.

With a normalizer that anticipates reversion the method matches Mondrian, is slightly
narrower, and **keeps the calibration set whole** — solving the affordability problem.

**Caveat keeping Mondrian primary:** the blend uses the generator's own reversion exponent.
It bounds what an auxiliary volatility model *could* achieve, not what a real EWMA or GARCH
estimator *will*. Mondrian needs no auxiliary model and cannot be undermined by a bad one.

## 12. Affordability

Mondrian degradation as the calibration set thins:

| windows/stratum | marginal | spread | width |
|---|---|---|---|
| 400 | 0.798 | 0.010 | 0.124 |
| 100 | 0.802 | 0.006 | 0.125 |
| 30 | 0.810 | 0.036 | 0.131 |
| 12 | **0.861** | 0.024 | 0.157 |
| 6 | 0.840 | 0.048 | 0.153 |

Usable to about **30 windows per stratum on independent data**. Ours are correlated within
a forecast date, so the real threshold is higher and must be sized from the full grid
before Mondrian is trusted.

## 13. Terminology

Exact conditional coverage is **impossible** distribution-free (Foygel Barber, Candès,
Ramdas & Tibshirani, 2021). **Group-conditional** is the honest description of what either
Mondrian or the normalized route delivers — coverage conditional on strata or on the
normalizer's level sets, never on the individual window.

---

# Part IV — A5 protocol (pre-registered)

Written before the results exist so the decision rules cannot be chosen after seeing them.

## 14. Arms and the pass bar

| arm | what it is | why it is here |
|---|---|---|
| raw | fine-tuned, uncalibrated | the shortfall to be closed |
| **Mondrian conformal** | per-regime correction | **primary** (§10–12) |
| marginal conformal | one pooled correction | the honest ablation |
| normalized conformal | auxiliary volatility normalizer | contender if the calibration set proves thin |
| **conformalized RW-drift** | random walk + same calibration | **the null that must be beaten** |

**The fourth arm is the important addition.** RW-drift is already near-calibrated (0.881)
and scores **41% better CRPS** than zero-shot Kronos. A calibrated cone a random walk also
produces is not a product.

> **Pass bar, fixed before the first A4 training run:** fine-tuned + calibrated Kronos must
> beat conformalized RW-drift on **fair CRPS and interval score**, not merely on coverage.

Coverage cannot separate them — conformal gives *both* arms nominal coverage by
construction. Only sharpness at equal coverage can.

## 15. Dual-estimator protocol

Every arm runs under both quantile estimators:

| run | estimator | purpose |
|---|---|---|
| A | `weibull` (type 6) | the honest measurement |
| B | `linear` (type 7) | what a default-settings study would report |

**Decision rule:** the conformal improvement is claimed **only for the component that
survives run A**. Present in both → real. Present in B only → the layer was absorbing our
own artifact; report it as such. Present in A only → investigate; that ordering is not
expected.

Rationale: a conformal layer fitted on a biased calibration set learns to widen bands to
compensate, absorbing the instrument error and inflating the apparent improvement with
nothing visible in the figure.

## 16. Reporting requirements

Every coverage number carries `m · quantile estimator · T · top_p · top_k`. Fair and naive
CRPS are reported side by side so the size of each correction is visible.

**Conditional coverage is the headline, not marginal.** §10 shows marginal coverage can
read 0.794 while strata underneath are 0.648 / 0.805 / 0.929. Reporting the marginal number
alone would be technically true and substantively misleading — the exact failure this
project exists to avoid. A run hitting nominal marginally while leaving a regime spread
above **0.10** is reported as **not calibrated**.

## 17. Acceptance criterion — change pending approval

The original criterion is conformalized 80% bands achieving 78–82% empirical coverage. Two
facts bear on attainability:

- The **instrument bias at m = 30 was 5.04pp**, 2.5× the band. Corrected, this is no longer
  an obstacle; uncorrected it made the criterion unreachable regardless of method.
- **Effective sample size is the count of distinct forecast start dates**, not windows —
  12 on current probes.

**Proposed:** assess acceptance on whether the **block-bootstrap interval covers the
nominal level**, not on whether a point estimate lands inside ±2pp. *Awaiting explicit
approval; this is a change to a documented milestone.*

---

# Part V — Upstream contribution

## 18. Issue #254 analysis

**Reporter's stated configuration** (retrieved 2026-08-01): `sample_count: 8`,
Kronos-small + Kronos-Tokenizer-base, 1-minute NIFTY bars, lookback 240, horizons 3/5/10,
`T=1.0`, `top_p=0.9`, six trading days. Reports *"forecast coverage was still only around
~52%"*.

At m = 8, nominal 80%, default type-7 quantiles:

```
closed form  (0.8)(7)/(9) = 0.622
simulated                   0.643
hard ceiling (m-1)/(m+1)  = 0.778
```

The instrument accounts for roughly **17pp of the 28pp gap** — *if* the 52% refers to an
80% central interval. Not all of it. At m = 8 a **90% band is unconstructible**.

**Still unknown:** the band definition and quantile function. The surrounding analysis is
directional, so "coverage" may mean directional hit-rate — under which reading none of this
applies and ~52% on 1-minute bars is approximately the chance-level null.

| measured @ nominal 80% | implied m |
|---|---|
| 0.52 | ~4.7 |
| 0.60 | ~7.0 |
| 0.70 | ~15.0 |
| 0.75 | ~31.0 |

### Public entry points are worse than internal ones

| entry point | `sample_count` | `top_p` |
|---|---|---|
| `auto_regressive_inference` (internal) | 5 | 0.99 |
| `KronosPredictor.predict` / `predict_batch` (public) | **1** | 0.9 |

At `sample_count=1` no interval exists at all. A user reaching for the documented API and
asking about uncertainty gets a single path by default.

## 19. Draft comment — NOT POSTED

Held for human review. Repo link is a placeholder; drop the closing line if the repository
stays private. Reviewer checklist at the end.

> On your question 3 — how to read the confidence/uncertainty as a secondary signal — two
> things in the sampling config affect a coverage number before the model does.
>
> **1. Ensemble size and the quantile estimator.** You mention `sample_count: 8`. A fresh
> observation and `m` ensemble members are exchangeable, so the rank of the observation
> among the `m+1` values is uniform, and an interval spanning positions `j..k` covers
> exactly `(k-j)/(m+1)`. NumPy's default quantile (`method="linear"`, Hyndman–Fan type 7)
> places `p` at position `p(m-1)+1`, so a central `(1-α)` band covers `(1-α)(m-1)/(m+1)`.
> At `m=8`, a nominal 80% band measures **~62%** for a *perfectly calibrated* forecaster.
> There is also a hard ceiling: the widest possible band runs min-to-max and covers
> `(m-1)/(m+1) = 77.8%` at `m=8`, so a 90% interval is not constructible at that ensemble
> size — `np.quantile` clamps silently rather than erroring. Fix is free: Weibull plotting
> positions (`method="weibull"`, type 6), which place `p` at `p(m+1)` and recover nominal
> coverage. Or raise `m` — at `m=30` the default estimator still costs ~5pp.
>
> **2. Nucleus truncation.** `top_p=0.9` truncates the token tail at every autoregressive
> step, so the sampled ensemble is narrower than the model's own predictive distribution.
> That under-covers independently of `m` and of the quantile estimator, and compounds over
> multi-step horizons. Worth re-measuring at `top_p=1.0` to separate sampling policy from
> model.
>
> **What I'd need to know to say more:** does the ~52% refer to a central prediction
> interval, or to directional hit-rate? If directional, none of the above applies, and ~52%
> on 1-minute bars is roughly the chance-level null. [implied-m table]
>
> More generally, a coverage figure needs `m`, the quantile estimator, and
> `T`/`top_p`/`top_k` alongside it to be interpretable.

**Reviewer checklist:** replace the repo link · confirm the repo is public or drop the last
line · re-read for tone, no implication their evaluation was careless · verify `m=8` and
`top_p=0.9` against the current issue text · ≤400 words.

---

# Part VI — Caveats, open questions, reproduction

## 20. Caveats

- Zero-shot results are **45 windows over 12 blocks**; tercile slices have 15 each. Every
  interval is wide.
- The regime finding is the most valuable **and** the most sample-hungry. It needs the full
  grid before it can carry weight.
- All runs to date are subsamples and several executed from a dirty tree (recorded in the
  directory name). **None is the A3 acceptance artifact**, which must be the full 708-window
  grid from a clean tree.
- Terciles are cut within each subsample, so boundaries are not comparable across runs.
- The universe fix for survivorship is an approximation, not a point-in-time
  reconstruction; `point_in_time: false` is recorded in the manifest and must be disclosed
  wherever coverage is published.

## 21. Open questions

| # | Question | Status |
|---|---|---|
| 1 | Does the constant z-ratio hold on the full grid and after fine-tuning? | open — check `monotone_increasing` |
| 2 | Does the regime spread survive conformalization? | answered in simulation; unconfirmed on real data |
| 3 | Is the h^0.639 exponent stable across regimes and tickers? | open |
| 4 | What is the random walk's exponent? | **answered: 0.4983** |
| 5 | Is under-dispersion partly an artifact of window normalisation? | **open, most consequential** |
| 6 | Does the `top_p=1.0` CRPS worsening indicate junk mass in the untruncated tail? | open |
| 7 | Do published TSFM calibration studies state `m` and the estimator? | open — the contribution hinges on this |

**Q5 is the one that changes plans.** If dispersion is already compressed at h = 1, part of
the 2.26× is a tokenizer/normalisation ceiling that fine-tuning the transformer cannot
lift — which would materially change what A4 is worth. Requires the saved forecast paths
(now persisted; needs a regeneration run).

## 22. Reproduction

```powershell
uv run pytest -q                                        # 92 tests
uv run python phase-a/eval/calibrate.py --split test     # full grid, ~6.6 h
uv run python phase-a/eval/calibrate.py --split test --limit 45 --sweep-top-p
```

Every run writes `results/<timestamp>_<git-sha>/` containing `results.json` and
`ensembles.npz` — the forecast paths and grid alignment, so any run can be re-analysed on
CPU without a GPU.

## 23. Literature

- **Order statistics / plotting positions:** Wilks (1941), *Determination of sample sizes
  for setting tolerance limits*; Weibull (1939); Cunnane (1978), *Unbiased plotting
  positions — a review*. Hyndman & Fan (1996), *Sample quantiles in statistical packages* —
  note they recommend **type 8** for general quantile estimation, while **type 6** is what
  the rank argument implies for *coverage* specifically. Our simulation confirms type 6
  recovers nominal at m = 30 (0.8020) where type 8 variants do not (median-unbiased 0.7858,
  normal-unbiased 0.7848). The choice is application-specific.
- **Ensemble-size effects on scores:** Ferro, Richardson & Weigel (2008); Zamo & Naveau
  (2018), *Estimation of the CRPS with limited information*; Hamill (2001), *Interpretation
  of rank histograms*. Also [*Ensemble-size-dependence of deep-learning post-processing
  methods that minimize an (un)fair score*](https://arxiv.org/html/2602.15830) — the same
  pathology in a deep-learning setting.
- **Conformal prediction:** Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random
  World* — the `(n+1)` correction. Angelopoulos & Bates, *A gentle introduction to conformal
  prediction*. [Foygel Barber, Candès, Ramdas & Tibshirani
  (2021)](https://arxiv.org/abs/1903.04684), *The limits of distribution-free conditional
  predictive inference*. Boström & Johansson on Mondrian regression. Normalized
  nonconformity scores achieve adaptivity **through auxiliary models** — the qualifier our
  first negative result missed.
- **Conformal for time series:** [benchmark of CP algorithms for time-series
  forecasting](https://arxiv.org/html/2601.18509v2) — MSCP best on Winkler among methods
  reaching nominal; ACI lowest of the valid set; EnbPI and SPCI fail coverage.

### Search terms

`plotting position quantile estimator coverage bias` · `Hyndman Fan quantile types` ·
`distribution-free tolerance interval order statistics` · `fair CRPS ensemble size` ·
`rank histogram ensemble size bias` · `conformal prediction finite sample n+1 correction` ·
`Mondrian conformal group-conditional coverage` · `normalized nonconformity difficulty
estimator`
