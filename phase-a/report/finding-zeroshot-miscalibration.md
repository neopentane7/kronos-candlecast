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

### Why this matters for A5

A **marginal** conformal correction sized to hit 80% on average would leave conditional
coverage badly wrong — roughly 0.6 in calm regimes and 0.9 in volatile ones, preserving
most of the 29-point spread. The cone would be honest on average and dishonest exactly
when a user is most exposed.

This is the empirical case for **regime-conditional (Mondrian) conformal calibration**,
stratifying the calibration set by an observable volatility proxy. It was previously a
speculative suggestion; it is now motivated by measurement.

**The counter-pressure is real:** stratifying three ways divides an already-thin
calibration set, widening every interval. With 12 effective blocks this is not currently
affordable. It becomes affordable only with the finer stride the KV cache unlocks.

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

1. **Does the constant z-ratio hold on the full grid and after fine-tuning?** If it does,
   it is a strong argument that a scalar conformal correction suffices — a cleaner claim
   than "conformal helps".
2. **Does the regime spread survive conformalization?** The A5 three-way comparison should
   report conditional coverage per regime, not just marginal.
3. **Is the h^0.639 exponent stable across regimes and tickers?** A drift-error signature
   that varies by regime would argue for horizon-and-regime-conditional calibration.
4. **What does the random walk's exponent look like?** It should be ~0.5 by construction;
   measuring it validates the harness and calibrates the comparison. Not yet computed.
5. **Is the under-dispersion partly an artifact of window normalisation?** Kronos
   standardises each window by its own mean and sd, which may systematically compress
   predicted variance relative to realised.

### Literature pointers

- Mondrian / class-conditional conformal prediction: Vovk et al., *Algorithmic Learning in
  a Random World*; Boström & Johansson on Mondrian regression forests.
- Volatility mean reversion and forecast-horizon scaling: standard GARCH literature; the
  h^0.5 diffusion benchmark.
- Ferro, Richardson & Weigel (2008) for the fair CRPS estimator used throughout.
- Conditional vs marginal coverage: Foygel Barber et al., *The limits of distribution-free
  conditional predictive inference*.
