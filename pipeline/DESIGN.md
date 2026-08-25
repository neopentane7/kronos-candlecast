# CandleCast serving design — decisions inherited from Phase A

> Research/education tool — scenario visualization, not investment advice.

Phase A measured two things that constrain Phase B before any serving code is written.
Both are recorded here so they are design inputs rather than discoveries made later, in
production, on a day the market was open.

Nothing in this document is built yet. It exists to make two decisions cheap now that
would be expensive to reverse.

---

## 1. Calibration tiers — stratified at 50%, ACI above it

Phase A report §17d establishes a feasibility theorem for split conformal at a 30-step
horizon on daily bars. Measured on this corpus at 245.5 sessions/year:

| nominal | calibration points needed per side | blocks | sessions | **years of data** | feasible |
|---|---|---|---|---|---|
| **50%** | 3 | 6 | 180 | **0.73** | **yes** |
| 80% | 9 | 18 | 540 | 2.20 | no |
| 90% | 19 | 38 | 1140 | 4.64 | no |

The corpus is 8.5 years, of which 2018–2023 is spent on training (rule 7). An 80% band
needs 2.2 years dedicated to the conformal split alone; 90% needs 4.64, which is 55% of
everything. Neither is affordable without starving training or stretching calibration
across a span where exchangeability — the assumption split conformal actually requires —
is what is being given up to obtain it.

**So exactly one nominal level supports honest per-regime calibration on this geometry.**
That yields a two-tier serving story:

| band | method | guarantee |
|---|---|---|
| **50% (p25–p75)** | **Mondrian split conformal, per volatility regime** | finite-sample, conditional on regime |
| **80% (p10–p90)** | **ACI** | asymptotic in update steps, marginal |

ACI is not a fallback of convenience here. Its guarantee is asymptotic in the number of
online updates and requires neither exchangeability nor a minimum calibration set, which
makes it **the only available method at the levels the theorem forbids**. The Phase A
decision to use ACI on the serving path (CLAUDE.md B1) is therefore forced by the data
geometry rather than chosen for operational tidiness — a better sentence for both the
paper and this document.

**Product consequence, deferred:** whether the UI distinguishes the two tiers is a product
call, not a measurement one. If it does, the honest framing is *"the inner band is
calibrated per volatility regime; the outer band adapts online"* — and shipping stratified
calibration at the one level the feasibility theorem permits is what demonstrates the
theorem is load-bearing rather than ornamental. If the UI shows only one band, it should
be the 50% one, because it is the only one carrying a conditional guarantee.

**Contract impact:** none. API contract v2 already carries `p25`/`p75` alongside
`p10`/`p90` and a `calibration` field. A per-band method label would require a
`contract_version` bump and explicit approval; a decision on that is not needed until B2.

---

## 1a. What the served calibration actually is, and is not

Measured on the seeded archive: 59 forecast dates, 236k matured rows, aggregated by date
because a single day is one observation and not 5,310.

| nominal | pooled | first 29 dates | last 30 dates |
|---|---|---|---|
| 0.50 | 0.5007 | 0.4520 | 0.5478 |
| 0.80 | 0.7978 | 0.7449 | 0.8488 |
| 0.90 | 0.8907 | 0.8554 | 0.9248 |

**Do not quote the pooled column on its own.** ACI adapts online: early dates run on a cold
state and under-cover, then the correction overshoots. Pooling averages those two errors
into a figure that reads as near-perfect calibration and is really a warm-up transient and
an overshoot cancelling out.

The defensible claim is **"converges toward nominal, with a warm-up transient and some
overshoot"** — not "calibrated to within half a point". `archive_coverage` now emits
`warmup` and `recent` beside `empirical`, so the pooled number cannot be read alone by
accident.

Two things follow:

* **Seeding was necessary, and this is the evidence.** The cold-state half sits 3–5 points
  below nominal at every level. Day-one visitors to an unseeded site would have seen that.
* **ACI's guarantee is asymptotic in update steps**, so a transient is expected behaviour,
  not a defect. What would be a defect is reporting the average as though it were the
  steady state.

---

## 2. Muhurat is a scheduled future defect, not a historical footnote

### The problem

`exchange_calendars` has no NSE calendar. `XNSE` does not exist, so the project uses
**`XBOM` — the BSE calendar — as a stand-in** (`common/calendar.py`). Hard constraint 5
gives that calendar authority over **future** sessions, correctly: when generating
forecast timestamps there is no price data to appeal to, so the calendar is the only
source available.

The Phase A corpus audit (`phase-a/scripts/audit_calendar.py`, report §17b) found **7
dates where the entire universe traded and XBOM lists no session.** All seven are Diwali
**Muhurat** sessions:

```
2018-11-07  2019-10-27  2020-11-14  2021-11-04  2022-10-24  2024-11-01  2025-10-21
```

Muhurat is a special one-hour ceremonial session NSE announces weeks in advance. It
recurs every Diwali. XBOM misses it systematically — 7 for 7 across the corpus.

**So the nightly pipeline, as currently specified, will believe the market is closed on a
day NSE trades.** It will skip the fetch, and the corpus will acquire exactly the kind of
single-ticker-free universe-wide hole that cost a session to find and a re-baseline to
fix. Next Diwali, not hypothetically.

### The decision

> **Ingest treats an off-calendar bar as a calendar-override event, logged, not an anomaly
> dropped.** If validated data arrives for a date the calendar does not list, the date is
> accepted and recorded in the run manifest as `calendar_override`. The calendar's
> authority over *future* timestamps is unchanged; this governs only the case where prices
> already exist and the calendar disagrees with them.

This is the same authority ordering Phase A settled on and for the same reason: **the
corpus is NSE, the calendar is a neighbouring exchange's approximation of it.** Where
prices exist, they outrank the approximation. Where they do not, the approximation is all
there is.

Chosen over the alternative — maintaining a hand-curated override list of announced
special sessions — because a list has to be updated every year by someone who remembers
to, and the failure mode of forgetting is silent. The data-driven rule needs no
maintenance and degrades safely: if no data arrives, nothing is overridden.

**What still needs building (B1):**

- the nightly job must attempt a fetch on Muhurat dates rather than short-circuiting on
  `calendar.is_session()`;
- `calendar_override` events must appear in the run manifest so an unexpected one is
  visible rather than merely tolerated;
- `audit_calendar.py` should run in the nightly job, not only ad hoc, so a universe-wide
  hole is caught the next morning instead of at the next full re-baseline.

**What must not change:** `common/calendar.py:future_sessions()` still comes from XBOM.
Forecast timestamps are generated before any prices exist, so there is nothing to
override them with. A Muhurat session inside a 30-day forecast horizon will be missing
from the predicted timestamps; that is a known and accepted one-session-per-year
inaccuracy in the forward calendar, and it is the price of not having an NSE calendar at
all.

---

## Open, not decided here

- Whether the two calibration tiers are surfaced separately in the UI (B3 product call).
- Whether `contract_version` bumps to label per-band methods (needs approval, B2).
- Whether ACI state is per-ticker×horizon only, or also per-regime (B1; per-regime
  multiplies state by 3 and each stream updates a third as often — likely not worth it,
  but it is a measurement question, not a taste one).
