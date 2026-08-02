# A5 protocol: conformal calibration study

**Status:** encoded now, executed at A5. Written before the results exist so the decision
rule cannot be chosen after seeing them.

> Research/education tool — scenario visualization, not investment advice.

---

## 1. Why this protocol exists

The headline A5 figure is a three-way reliability diagram — zero-shot vs fine-tuned vs
fine-tuned + conformal — in which the conformal line is expected to hug the diagonal.

That figure is at risk of being **partly self-inflicted**. Two instrument effects push
measured coverage below nominal independently of the model
(see [finding-ensemble-coverage-bias.md](finding-ensemble-coverage-bias.md)):

1. **Order-statistic bias** — empirical quantiles of a finite ensemble form a band
   narrower than the true one; −5.04pp at m = 30 under numpy's default estimator.
2. **Nucleus truncation** — `top_p < 1.0` narrows the sampled distribution at every
   autoregressive step, independently of m and of the quantile estimator.

A conformal layer fitted on a calibration set that carries either effect will learn to
**widen the bands to compensate**. The raw path still shows the shortfall. The diagram
would then show conformal "fixing" a defect that was substantially ours, and the apparent
improvement would be inflated by an amount nobody could see from the figure.

---

## 1a. Arms — revised on evidence

The comparison has **four** arms, not three. Two changes from the original plan, both
forced by measurement rather than preference.

| arm | what it is | why it is here |
|---|---|---|
| raw | fine-tuned, uncalibrated | the shortfall to be closed |
| **Mondrian conformal** | per-volatility-regime correction | **primary** — see §1b |
| marginal conformal | one correction pooled | the honest ablation against Mondrian |
| normalized conformal | auxiliary volatility normalizer | contender if the calibration set proves thin |
| **conformalized RW-drift** | random walk + the same calibration | **the null that must be beaten** |

**The fourth arm is the important addition.** Random-walk-with-drift is already
near-calibrated out of the box (0.881 at nominal 80%) and scores **41% better CRPS than
zero-shot Kronos** (39.93 against 67.57). A calibrated cone that a random walk also
produces is not a product. Conformalizing the baseline too is the only way to know whether
the model is contributing anything.

### Pre-registered pass bar

> Fine-tuned + calibrated Kronos must beat conformalized RW-drift on **fair CRPS and
> interval score**, not merely on coverage.

Coverage alone cannot distinguish them, because conformal calibration will give *both*
arms nominal coverage by construction — that is what it does. Only sharpness at equal
coverage separates a model from a random walk. **This bar is fixed before the first A4
training run**, so it cannot be relaxed after seeing results.

---

## 1b. Method selection — Mondrian, on measured evidence

A CPU viability experiment (`tests/test_conformal.py`, synthetic forecaster tuned to the
measured A3 pathology) settles which variant to use. Nominal 80%:

| arm | marginal | calm | mid | volatile | spread | rel. width |
|---|---|---|---|---|---|---|
| raw | 0.459 | 0.333 | 0.443 | 0.602 | 0.269 | 0.064 |
| marginal conformal | 0.794 | 0.648 | 0.805 | 0.929 | **0.281** | 0.142 |
| normalized by lookback vol | 0.793 | 0.643 | 0.804 | 0.932 | **0.289** | 0.138 |
| **Mondrian** | 0.796 | **0.799** | **0.790** | **0.798** | **0.009** | **0.123** |

**Marginal conformal fixes the average and nothing else.** The regime spread is 0.269
before and 0.281 after — it does not shrink.

**Normalized conformal: the first verdict was wrong, and is corrected here.** An earlier
revision recorded it as "tested and rejected" on the strength of the row above. That
conclusion over-generalised from one badly chosen normalizer. The standard construction
uses an **auxiliary** difficulty estimator; I had used the model's own anchor, which is the
degenerate case. Re-tested with three normalizers:

| normalizer | marginal | calm | volatile | spread | rel. width |
|---|---|---|---|---|---|
| the model's own anchor (trailing vol) | 0.789 | 0.635 | 0.928 | 0.293 | 0.075 |
| **mean-reversion aware blend** | 0.792 | **0.794** | **0.783** | **0.015** | **0.065** |
| long-run volatility only | 0.795 | 0.936 | 0.635 | 0.301 | 0.065 |
| Mondrian, for comparison | 0.794 | 0.797 | 0.797 | 0.011 | 0.067 |

**Failure is two-sided.** Normalizing by the model's anchor reproduces the bias;
discarding local information entirely over-corrects and *inverts* the spread — calm now
over-covers at 0.936 while volatile under-covers at 0.635. The normalizer must be right,
not merely different.

With a normalizer that anticipates reversion, the method **matches Mondrian on conditional
coverage (0.015 vs 0.011), is slightly narrower, and keeps the calibration set whole** —
which is exactly the affordability problem that makes Mondrian risky at 12 blocks.

**Caveat that keeps Mondrian primary:** the blend above uses the data generator's own
reversion exponent. It bounds what an auxiliary volatility model *could* achieve; it does
not predict what a real EWMA or GARCH estimator *will*. Mondrian needs no auxiliary model
and cannot be undermined by a bad one.

**Mondrian is also narrower, which inverts the expected trade-off.** 0.123 against 0.142.
Marginal conformal has to over-widen the volatile stratum in order to lift the calm one.
So the cost of stratification is **calibration-set size, not interval width** — the
opposite of what this protocol previously assumed.

Degradation as the calibration set thins (windows per stratum → marginal coverage /
spread): 400 → 0.798/0.010 · 100 → 0.802/0.006 · 30 → 0.810/0.036 · 12 → 0.861/0.024 ·
6 → 0.840/0.048. Usable to about **30 windows per stratum** on independent data; ours are
correlated within a forecast date, so the real threshold is higher and must be sized from
the full grid before Mondrian is trusted.

---

## 2. Protocol

Run every arm in §1a under **both** quantile estimators:

| run | quantile estimator | purpose |
|---|---|---|
| A | `weibull` (type 6) | the honest measurement |
| B | `linear` (type 7) | reproduces what a default-settings study would report |

Produce paired reliability diagrams. The **difference between A and B isolates the
instrument from the method.**

### Decision rule (fixed in advance)

> The conformal improvement is claimed **only for the component that survives run A.**

Concretely:

- Improvement present in **both** A and B → real, attributable to the conformal method.
- Improvement present in **B only** → the conformal layer was absorbing our own
  measurement artifact. Report it as such; do not claim it.
- Improvement present in **A only** → investigate before claiming; this ordering is not
  expected and suggests something else is wrong.

### Reporting requirements

Every coverage number in the A5 report carries its full provenance:

```
m · quantile estimator · T · top_p · top_k
```

Both fair and naive CRPS are reported side by side, as `summarize()` already emits, so
the size of each correction is visible rather than asserted.

**Conditional coverage is the headline, not marginal coverage.** The viability experiment
shows marginal coverage can be made to look correct while every stratum is wrong — 0.794
overall with 0.648 / 0.805 / 0.929 underneath. Reporting the marginal number alone would
be technically true and substantively misleading, which is the exact failure this project
exists to avoid. Every reliability diagram is therefore accompanied by a per-regime
coverage table, and a run that hits nominal marginally while leaving a regime spread above
0.10 is reported as **not calibrated**.

---

## 3. Extension to sampling policy — RESOLVED: hold `top_p` fixed

This section was written as a conditional on the A3 sweep. The sweep has run and the
conditional resolves to **no repetition required**.

Run `20260801T061559Z_7c8f164-dirty`, test split, 15 stratified windows, 12 blocks,
m = 30, T = 1.0, top_k = 0:

| top_p | coverage @80% | rel. band width | CRPS (fair) |
|---|---|---|---|
| 0.90 | 0.4689 | 0.1094 | **56.15** |
| 0.99 | 0.4689 | 0.1233 | 60.46 |
| 1.00 | **0.4933** | **0.1284** | 60.63 |

Truncation is real and measurable — removing it widens the bands by **17.4%** — but it is
**not the binding constraint**. That widening buys only **2.4pp** of coverage, which is
inside the noise at 12 blocks, and CRPS gets *worse* rather than better.

The arithmetic makes the point sharper. Closing the residual gap from ~0.49 to 0.80 needs
bands roughly **1.9× wider** (under a normal, z must move from 0.66 to 1.28). The entire
sampling-policy axis supplies 1.17×. **The under-coverage is dominated by location error,
not band width**, which matches the per-window diagnostics: the model makes large
directional calls that miss, rather than correct calls with tight bands.

**Decision:** hold `top_p` at 0.9 (the specified value, and the best CRPS of the three),
run the A5 comparison once rather than twice, and leave the widening to the conformal
layer, which is exactly the job it exists to do. Recorded as `PRODUCTION_TOP_P` in
`phase-a/eval/calibrate.py` with this table cited inline.

**Caveat:** 15 windows over 12 blocks is a small basis for a negative result. The
conclusion is "not the dominant mechanism", not "no effect". If the full-grid run shows
materially different band behaviour, revisit.

**Consequence for §2, which is unaffected:** the dual-*estimator* run still stands. The
order-statistic bias is a property of the measurement, not of the sampling policy, and
nothing here bears on it.

---

## 4. Interaction with the acceptance criterion

The original A5 acceptance — conformalized 80% bands achieving 78–82% empirical coverage —
is a ±2pp band. Two facts bear on whether that is attainable:

- The **instrument bias at m = 30 is 5.04pp**, 2.5× the band width. Corrected, this is no
  longer an obstacle; uncorrected, it made the criterion unreachable regardless of method.
- The **effective sample size** is the number of distinct forecast start dates, not the
  number of windows. Coverage is therefore reported as an estimate with a block-bootstrap
  interval, and acceptance is assessed on whether **the interval covers the nominal
  level**, not on whether the point estimate lands inside ±2pp.

The second point is a change to the criterion as originally written and is flagged for
explicit approval before A5 runs.
