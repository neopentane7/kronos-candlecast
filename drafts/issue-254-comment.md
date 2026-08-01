# DRAFT — comment for upstream issue #254

**Not posted.** For human review and posting. Repo link is a placeholder.

Target: https://github.com/shiyu-coder/Kronos/issues/254

Constraints applied: answers their question first; no claims about their model quality;
every number reproducible from the closed form; one self-reference, last line; ≤400 words.

---

On your question 3 — how to read the confidence/uncertainty as a secondary signal — two
things in the sampling config affect a coverage number before the model does. Both are
worth ruling out before concluding anything about calibration.

**1. Ensemble size and the quantile estimator.** You mention `sample_count: 8`. A fresh
observation and `m` ensemble members are exchangeable, so the rank of the observation
among the `m+1` values is uniform, and an interval spanning positions `j..k` covers
exactly `(k-j)/(m+1)`. NumPy's default quantile (`method="linear"`, Hyndman–Fan type 7)
places `p` at position `p(m-1)+1`, so a central `(1-α)` band covers:

```
(1 - α)(m - 1)/(m + 1)
```

At `m=8`, a nominal 80% band measures **~62%** for a *perfectly calibrated* forecaster.
There is also a hard ceiling: the widest possible band runs min-to-max and covers
`(m-1)/(m+1) = 77.8%` at `m=8`, so a 90% interval is not constructible at that ensemble
size at all — `np.quantile` clamps silently rather than erroring.

Fix is free: use Weibull plotting positions (`method="weibull"`, type 6), which place `p`
at `p(m+1)` and recover nominal coverage. Or raise `m` — at `m=30` the default estimator
still costs ~5pp.

**2. Nucleus truncation.** `top_p=0.9` truncates the token tail at every autoregressive
step, so the sampled ensemble is narrower than the model's own predictive distribution.
That under-covers independently of `m` and of the quantile estimator, and it compounds
over multi-step horizons. Worth re-measuring at `top_p=1.0` to separate the sampling
policy from the model.

**What I'd need to know to say more:** does the ~52% refer to a central prediction
interval, or to directional hit-rate? If directional, none of the above applies, and ~52%
on 1-minute bars is roughly the chance-level null you'd expect. If it is an 80% interval,
here's where different ensemble sizes land:

| measured @ nominal 80% | implied m |
|---|---|
| 0.52 | ~4.7 |
| 0.60 | ~7.0 |
| 0.70 | ~15 |
| 0.75 | ~31 |

More generally, a coverage figure needs `m`, the quantile estimator, and `T`/`top_p`/
`top_k` alongside it to be interpretable.

I'm running a daily-horizon NSE calibration study on Kronos-small at
`<REPO_LINK_PLACEHOLDER>` and am happy to share the measurement harness.

---

## Reviewer checklist before posting

- [ ] Replace `<REPO_LINK_PLACEHOLDER>`.
- [ ] Confirm the repo is public, or drop the last line entirely.
- [ ] Re-read for tone: no implication their evaluation was careless.
- [ ] Verify `m=8` and `top_p=0.9` against the current issue text (retrieved 2026-08-01).
- [ ] Word count ≤400 (currently ~370 excluding this checklist).
