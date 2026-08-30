import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchForecast, fetchHistory, fetchIndex, fetchTrackRecord } from "./api";
import { Cone } from "./components/Cone";
import { TrackRecord } from "./components/TrackRecord";

const DEFAULT = "RELIANCE.NS";

export default function App() {
  const [ticker, setTicker] = useState(DEFAULT);

  const index = useQuery({ queryKey: ["index"], queryFn: fetchIndex });
  const forecast = useQuery({
    queryKey: ["forecast", ticker],
    queryFn: () => fetchForecast(ticker),
    enabled: !!ticker,
  });
  const history = useQuery({
    queryKey: ["history", ticker],
    queryFn: () => fetchHistory(ticker),
    enabled: !!ticker,
  });

  // Site-wide, not per-ticker: coverage is a property of the engine, and slicing it by
  // ticker would put one name's 20 days where 60 names' 20 days belong.
  const track = useQuery({ queryKey: ["track"], queryFn: fetchTrackRecord, retry: false });

  const f = forecast.data;
  const tickers = index.data?.tickers ?? [];

  return (
    <div className="app">
      <header className="bar">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <div>
            <h1>CandleCast</h1>
            <p className="sub">Calibrated forecast cones — NSE daily</p>
          </div>
        </div>

        <label className="picker">
          <span className="visually-hidden">Ticker</span>
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            disabled={!tickers.length}
          >
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t.replace(".NS", "")}
              </option>
            ))}
          </select>
        </label>
      </header>

      <main>
        {index.isError && <Problem what="the ticker list" error={index.error} />}
        {(forecast.isError || history.isError) && (
          <Problem what={ticker} error={forecast.error ?? history.error} />
        )}

        {f && history.data ? (
          <>
            <section className="panel">
              <Cone history={history.data.sessions} forecast={f} />
            </section>
            <Readout f={f} />
            {track.data && <TrackRecord tr={track.data} />}
          </>
        ) : (
          !forecast.isError && <p className="loading">Loading {ticker.replace(".NS", "")}…</p>
        )}
      </main>

      {/* Always rendered, never dismissible (working rule 12). */}
      <footer className="disclaimer" role="note">
        {f?.disclaimer ?? "Research/education tool - scenario visualization, not investment advice."}
      </footer>
    </div>
  );
}

function Readout({ f }: { f: import("./types").Forecast }) {
  const last = f.horizon - 1;
  const above = Math.round(f.prob_above_last_close[last] * 100);
  const lo = f.quantiles.p10[last];
  const hi = f.quantiles.p90[last];
  const pct = (v: number) => ((v / f.last_close - 1) * 100).toFixed(1);

  return (
    <section className="readout">
      <div className="stat">
        <dt>In {f.horizon} sessions, 80% of scenarios land between</dt>
        <dd>
          {lo.toFixed(2)} <span className="delta">({pct(lo)}%)</span>
          <span className="dash">–</span>
          {hi.toFixed(2)} <span className="delta">(+{pct(hi)}%)</span>
        </dd>
      </div>

      <div className="stat">
        <dt>Scenarios finishing above today's close</dt>
        <dd>
          {above}% <span className="note">of {f.metadata.ensemble_size} sampled paths</span>
        </dd>
      </div>

      <div className="badges">
        <Badge label="engine" value={f.engine} tone={f.metadata.engine_validated ? "ok" : "warn"} />
        <Badge label="50% band" value={f.metadata.band_methods.band_50.replace("_", " ")} tone="ok" />
        <Badge label="80/90% band" value={f.metadata.band_methods.band_80} tone="ok" />
        {f.metadata.aci_provisional && <Badge label="gamma" value="provisional" tone="warn" />}
        {f.backfilled && <Badge label="data" value="backfilled" tone="warn" />}
      </div>

      <p className="fineprint">
        Bands are calibrated, not predicted: the 50% band carries a finite-sample guarantee
        from held-out 2024 data, and the 80/90% bands adapt online. Percentages are fractions
        of {f.metadata.ensemble_size} sampled paths, so they resolve to roughly 3 points.
        Generated {f.generated_at.slice(0, 10)}.
      </p>
    </section>
  );
}

function Badge({ label, value, tone }: { label: string; value: string; tone: "ok" | "warn" }) {
  return (
    <span className={`badge ${tone}`}>
      <span className="badge-label">{label}</span>
      {value}
    </span>
  );
}

function Problem({ what, error }: { what: string; error: unknown }) {
  return (
    <div className="problem" role="alert">
      <strong>Could not load {what}.</strong>
      <p>{error instanceof Error ? error.message : "Unknown error"}</p>
      <p className="hint">
        If this is the first load after a deploy, the nightly job may not have published yet.
        Try again in a few minutes.
      </p>
    </div>
  );
}
