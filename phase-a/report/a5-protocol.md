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

## 2. Protocol

Run the full three-way comparison — **raw / ACI / MSCP** — under **both** quantile
estimators:

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
