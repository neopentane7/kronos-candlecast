# Zero-shot miscalibration: shape, scale, and regime

**Run:** `results/20260801T061559Z_7c8f164-dirty/results.json`
**Date:** 2026-08-01 · **Milestone:** A3 (preliminary — not the acceptance run)

> Research/education tool — scenario visualization, not investment advice.

**Provenance warning.** This run executed from a dirty working tree, correctly recorded in
the directory name. It is adequate for the decisions it feeds; it is **not** the A3
acceptance artifact, which must run from a clean tree on the full 708-window grid.

---

## 1. Setup

| | |
|---|---|
| Grid | test split, 45 windows subsampled from 708, stratified by ATR tercile |
| Coverage | 32 tickers, **12 distinct forecast start dates** (the bootstrap blocks) |
| Model | Kronos-small zero-shot, lookback 400 → horizon 30 |
| Sampling | m = 30, T = 1.0, top_k = 0, top_p per row |
| Measurement | Weibull quantiles (type 6), Ferro fair CRPS |
| Cost | 33.5 s/window, 3349 MB peak VRAM at batch 8 × 30 samples |

---

## 2. Headline results

| model | CRPS (fair) | MAPE | cov@50 | cov@80 | cov@90 |
|---|---|---|---|---|---|
| random_walk_drift | **39.93** | 0.0399 | 0.608 | **0.881** | 0.953 |
| last_value | 50.44 | 0.0365 | 0.002 | 0.002 | 0.002 |
| **kronos_zeroshot** | **67.57** | **0.0692** | **0.240** | **0.422** | **0.531** |

Zero-shot Kronos is worse than a flat line on CRPS and worse than a Gaussian random walk
on every metric. MAPE is 6.9% against the random walk's 4.0%, so this is not only a
calibration failure — the point forecast is also worse.

Note the CRPS naive/fair ratio confirms the estimator work: 1.017 for Kronos, 1.045 for
the random walk, 1.000 for the degenerate baseline (a point forecast has no spread term to
correct).

---

## 3. Is the miscalibration shape or scale?

This is the question that decides whether conformal calibration is the right instrument.
If the predictive distribution has the **right shape but the wrong scale**, the ratio
`z_achieved / z_nominal` is constant across nominal levels and a single multiplicative
correction per horizon fixes it. If the **shape** is wrong, no scalar correction will.

| nominal | achieved | z_nominal | z_achieved | ratio |
|---|---|---|---|---|
| 0.50 | 0.2400 | 0.6745 | 0.3055 | 0.4529 |
| 0.80 | 0.4222 | 1.2816 | 0.5566 | 0.4343 |
| 0.90 | 0.5311 | 1.6449 | 0.7243 | 0.4403 |

**Ratio: mean 0.4425, sd 0.0095, total spread 0.0186 across the whole range.**

That is remarkably constant. **The predictive distribution is approximately the right
shape and uniformly too tight by a factor of 2.26×.** The same test on the random walk
gives ratios of 1.27 / 1.22 / 1.21 — also near-constant, slightly over-wide, exactly as a
correctly-specified Gaussian forecaster should look.

**This is the most encouraging result in the run.** A uniform per-horizon scale correction
is precisely what split conformal and MSCP produce. The failure mode is the one conformal
calibration is designed for.

### The test is one-sided — read it as rejection, not confirmation

Constancy **rejects** bias-dominated error but does **not confirm** pure scale error. A
bias-plus-scale mixture also reads flat. Simulated, 400k observations each:

| forecaster | ratios @ 50/80/90 | spread | verdict against our 0.0186 |
|---|---|---|---|
| pure scale, σ = 0.44 | 0.440 / 0.440 / 0.440 | 0.000 | consistent |
| random-sign bias 0.75 + scale 0.75 | 0.570 / 0.578 / 0.584 | 0.014 | **also consistent — not excluded** |
| pure bias 1.1, correct scale | 0.564 / 0.609 / 0.636 | 0.072 | excluded |

The discriminating signature is **monotonicity**: location bias makes the ratio climb with
nominal level; pure scale is flat. Our measured sequence (0.4529, 0.4343, 0.4403) is
**non-monotone**, i.e. noise around a constant — encouraging, but underpowered at 12
blocks. `z_ratio_table` returns a `monotone_increasing` flag; check it on the full grid
before treating the scale reading as settled.

---

## 4. Sampling-policy sweep

15 stratified windows, same configuration except `top_p`:

| top_p | cov@80 | 95% CI | CI width | rel. band width | CRPS (fair) | IS@80 |
|---|---|---|---|---|---|---|
| 0.90 | 0.4689 | [0.290, 0.658] | 0.367 | 0.1094 | **56.15** | **428.3** |
| 0.99 | 0.4689 | [0.265, 0.695] | 0.430 | 0.1233 | 60.46 | 464.4 |
| 1.00 | 0.4933 | [0.289, 0.710] | 0.421 | 0.1284 | 60.63 | 459.7 |

**Read the confidence intervals before the point estimates.** They are 0.37–0.43 wide and
overlap almost completely. On coverage, the three sampling policies are **statistically
indistinguishable** at this sample size. Only band width — an average rather than a
proportion — is estimated precisely enough to compare, and it moves monotonically.

So the mechanism is confirmed (truncation narrows the ensemble) while its effect on
coverage is unresolvable here. The decisive argument is arithmetic rather than statistical:

```
band widening required   2.260×      (from §3)
available from top_p     1.174×      (0.1094 → 0.1284)
shortfall                1.926×
```

Nucleus truncation cannot close the gap even if it were free. And widening makes both CRPS
and the interval score **worse**, because a wider band around a wrong location is penalised
by proper scoring rules.

**Decision:** hold `top_p` at 0.9 — best CRPS, best interval score — and leave the widening
to the conformal layer.

---

## 5. Regime dependence — the most consequential finding

Terciles by ATR over the **lookback** window; coverage measured on the **following** 30
sessions.

| regime | n | CRPS | IS@80 | MAPE | cov@50 | cov@80 | cov@90 |
|---|---|---|---|---|---|---|---|
| calm | 15 | 102.25 | 857.8 | 0.0735 | 0.122 | **0.260** | 0.356 |
| mid | 15 | 68.89 | 545.0 | 0.0605 | 0.264 | 0.456 | 0.544 |
| volatile | 15 | 31.58 | 205.3 | 0.0735 | 0.333 | **0.551** | 0.693 |

**Coverage at nominal 80% ranges from 0.260 to 0.551 — a spread of 29 points.**

Ignore the CRPS column when comparing regimes: it is in price units and MAPE is roughly
flat (0.074 / 0.061 / 0.074), so the CRPS ordering reflects price level, not skill.
Coverage is scale-free and the pattern in it is real.

**Mechanism.** Kronos normalises each window by its own statistics, so the width of its
cone is anchored to **trailing** volatility. Volatility mean-reverts. A calm lookback is
therefore disproportionately followed by a relatively more volatile future, and the cone —
sized for the calm past — under-covers badly. A volatile lookback produces a wide cone and
a future that tends to settle, so coverage looks better. **The model does not anticipate
regime change; it extrapolates the recent past.**

### Control: how much of this is Kronos, and how much is any lookback-based forecaster?

The mechanism is not Kronos-specific in kind — it punishes **any** forecaster whose
dispersion is estimated from the lookback, including our random-walk baseline. The same
tercile cuts, same windows:

| forecaster | calm | mid | volatile | calm − volatile |
|---|---|---|---|---|
| random_walk_drift | 0.853 | 0.887 | 0.902 | **−0.049** |
| kronos_zeroshot | 0.260 | 0.456 | 0.551 | **−0.291** |

The random walk shows the **same sign** — so the general mechanism is real — at **16.8%**
of the magnitude. The effect is therefore general in kind and **Kronos-specific in
degree**, and the excess is what fine-tuning and calibration have to address.

### Why this matters for A5 — tested, not assumed

A CPU viability experiment on a synthetic forecaster tuned to this pathology
(`tests/test_conformal.py`) settles the method question. Nominal 80%:

| arm | marginal | calm | mid | volatile | spread | rel. width |
|---|---|---|---|---|---|---|
| raw | 0.459 | 0.333 | 0.443 | 0.602 | 0.269 | 0.064 |
| marginal conformal | 0.794 | 0.648 | 0.805 | 0.929 | 0.281 | 0.142 |
| normalized by lookback vol | 0.793 | 0.643 | 0.804 | 0.932 | 0.289 | 0.138 |
| **Mondrian (per regime)** | 0.796 | 0.799 | 0.790 | 0.798 | **0.009** | **0.123** |

**Marginal conformal fixes the average and nothing else** — the spread goes 0.269 → 0.281.
The prediction in the previous revision of this section is confirmed.

**Normalizing by an ex-ante volatility proxy does not rescue it either** (0.289). It cannot:
the defect is trailing volatility mis-predicting future volatility, so dividing the
nonconformity score by trailing volatility reproduces the bias rather than removing it.

**Correction to the previous revision.** This section previously asserted that stratifying
"divides an already-thin calibration set, widening every interval". **The width half of
that is wrong.** Mondrian is *narrower* — 0.123 against 0.142 — because marginal conformal
must over-widen the volatile stratum in order to lift the calm one. The real cost of
stratification is **calibration-set size only**. Measured degradation, windows per stratum
→ marginal / spread: 400 → 0.798/0.010 · 100 → 0.802/0.006 · 30 → 0.810/0.036 ·
12 → 0.861/0.024 · 6 → 0.840/0.048. Usable to roughly 30 per stratum on independent
windows; ours are correlated within a date, so the real threshold is higher and must be
sized from the full grid.

---

## 6. Error growth with horizon

| step | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| CRPS | 11.14 | 38.50 | 57.30 | 78.42 | 99.61 |
| IS@80 | 79.0 | 304.7 | 426.5 | 634.8 | 822.6 |

Fitting a power law: **CRPS ∝ h^0.639**.

Pure diffusion — a correctly-specified random walk — gives **h^0.5**. An exponent above
that indicates error accumulating faster than diffusion alone, i.e. **systematic drift
error compounding across steps** rather than variance simply widening. This corroborates
the per-window diagnostics, where the model made large directional calls that missed
(GODREJCP predicted +4.3% against −7.8% actual; ULTRACEMCO predicted −7.6% against +0.6%).

**Validated against the baseline.** The theoretical 0.5 is not assumed — it is measured on
our own random-walk arm, through the same code path, on the same windows:

```
random_walk_drift exponent = 0.4983      band [0.45, 0.55]      PASS
kronos_zeroshot   exponent = 0.639       excess = +0.141
```

The harness therefore recovers the known answer where one exists, which is what licenses
citing the Kronos figure. The baseline reproduction was bit-exact against the recorded run
(CRPS 39.9271, coverage 0.8807), confirming the grid rebuild is deterministic.

---

## 7. What this implies

1. **Fine-tuning has a clear target.** The h^0.639 exponent and the 6.9% MAPE say the
   pretrained model's *location* is wrong on NSE daily bars, not merely its uncertainty.
   Fixing location should reduce the required conformal inflation below 2.26×.
2. **Conformal calibration is the right instrument for what remains.** The constant z-ratio
   means a per-horizon scale correction can work.
3. **Marginal coverage will not be sufficient.** §5 shows conditional coverage failing
   badly while marginal coverage could be made to look fine.
4. **The sampling policy is settled** and is not worth further compute.

---

## 8. Caveats

- 45 windows over **12 blocks**; the tercile slices have 15 windows each. Every interval
  here is wide, and the negative result on `top_p` is "not dominant", not "no effect".
- The regime finding is the most sample-hungry and the most valuable. It needs the full
  grid before it can carry weight.
- Run executed from a dirty tree (see header).
- Terciles are cut on the ATR distribution *within this subsample*, so the boundaries are
  not comparable to another run's.

---

## 9. Open questions

1. **Does the constant z-ratio hold on the full grid and after fine-tuning?** Open, and now
   with a sharper form: check the `monotone_increasing` flag, since flatness is only
   one-sided evidence (§3).
2. **Does the regime spread survive conformalization? ANSWERED in simulation.** Marginal
   conformal leaves it (0.269 → 0.281); Mondrian collapses it (→ 0.009). Still to confirm
   on real data.
3. **Is the h^0.639 exponent stable across regimes and tickers?** Open. A drift signature
   that varies by regime would argue for horizon-*and*-regime-conditional calibration.
4. **What does the random walk's exponent look like? ANSWERED: 0.4983**, inside the
   [0.45, 0.55] band. The harness recovers the known answer, so the Kronos figure stands.
5. **Is the under-dispersion partly an artifact of window normalisation?** Open, and the
   most consequential of the remaining questions. If dispersion is already compressed at
   h = 1, part of the 2.26× is a tokenizer/normalisation ceiling that fine-tuning the
   transformer cannot lift — which would materially change §7's expectations. Requires the
   saved forecast paths (now persisted; needs a regeneration run).
6. **Does the top_p = 1.0 CRPS worsening indicate junk mass in the untruncated tail?**
   Widening an under-dispersed forecast toward truth should improve CRPS. That it got worse
   points at the quantizer's edge bins and the upstream `clip=5` input clipping distorting
   predictive support. Related to Q5 and testable the same way.

### Literature pointers

- Mondrian / class-conditional conformal prediction: Vovk et al., *Algorithmic Learning in
  a Random World*; Boström & Johansson on Mondrian regression forests.
- Volatility mean reversion and forecast-horizon scaling: standard GARCH literature; the
  h^0.5 diffusion benchmark.
- Ferro, Richardson & Weigel (2008) for the fair CRPS estimator used throughout.
- Conditional vs marginal coverage: Foygel Barber et al., *The limits of distribution-free
  conditional predictive inference*.
