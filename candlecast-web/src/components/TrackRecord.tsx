import type { TrackRecord as TR, LevelRecord, DayEntry } from "../types";

/**
 * Did the bands actually contain the outcome?
 *
 * Three decisions here are load-bearing, and each exists to stop the panel reporting
 * something the data does not support.
 *
 * The denominator is **trading days, not observations**. Sixty tickers priced on one date
 * share a market and thirty horizon steps share a path, so a day is one observation. The
 * underlying row count is roughly 1,800x larger and quoting it would overstate the
 * evidence by that factor.
 *
 * Only **fully matured dates** are averaged. The newest dates have one or two steps
 * resolved, short horizons are easier to cover, and mixing them into the window reads as
 * improvement that has not happened -- it manufactured a +4.6pp overshoot here before it
 * was caught. Incomplete dates still draw in the sparkline; they never enter the mean.
 *
 * The headline is a **trailing window, not the cumulative mean**. ACI adapts online, so
 * the archive opens on a cold state that under-covered badly. Cumulative, this engine
 * reads 0.4554 / 0.7492 / 0.8584 against nominal 0.50 / 0.80 / 0.90 -- five points light
 * at 80% -- because the running average carries a deficit the engine has already
 * corrected. The last twenty comparable dates read 0.5071 / 0.8086 / 0.9078.
 */
export function TrackRecord({ tr }: { tr: TR }) {
  const levels = Object.entries(tr.backtest ?? {});
  if (!levels.length) return null;

  const liveDays = tr.live?.length ?? 0;

  return (
    <section className="track">
      <header className="track-head">
        <h2>Did the bands hold?</h2>
        <p>
          Coverage over the last {tr.window} forecast dates whose full horizon has resolved. One date is one observation, so these are means over <strong>days</strong>,
          never over individual forecasts.
        </p>
      </header>

      <div className="track-rows">
        {levels.map(([key, lvl]) => (
          <LevelRow key={key} lvl={lvl} window={tr.window} />
        ))}
      </div>

      <p className="track-note">
        {tr.days} replayed forecast dates, {tr.first} to {tr.last}.{" "}
        {liveDays > 0
          ? `${liveDays} live ${liveDays === 1 ? "day has" : "days have"} been scored since; live
             days are recorded separately and never pooled with replayed ones.`
          : "No live day has matured yet. Live results are recorded separately and never pooled with replayed ones."}
      </p>
    </section>
  );
}

function LevelRow({ lvl, window }: { lvl: LevelRecord; window: number }) {
  const t = lvl.trailing;
  if (!t) return null;

  const gap = t.coverage - lvl.nominal;
  const pp = (gap * 100).toFixed(1);
  // Within a point of nominal is calibrated; beyond three is a plateau worth naming. The
  // middle band is honest uncertainty, not a pass.
  const tone = Math.abs(gap) <= 0.01 ? "on" : Math.abs(gap) <= 0.03 ? "near" : "off";
  const verdict = tone === "on" ? "on target" : gap > 0 ? "wider than needed" : "too narrow";

  return (
    <div className={`track-row ${tone}`}>
      <div className="track-label">
        <span className="track-nominal">{Math.round(lvl.nominal * 100)}% band</span>
        <span className="track-verdict">{verdict}</span>
      </div>

      <Sparkline series={lvl.series} nominal={lvl.nominal} window={window} />

      <div className="track-figures">
        <span className="track-actual">{(t.coverage * 100).toFixed(1)}%</span>
        <span className="track-gap">
          {gap > 0 ? "+" : ""}
          {pp} pp
        </span>
        <span className="track-days">over {t.days} days</span>
      </div>
    </div>
  );
}

/**
 * The full per-date series, with nominal as a reference line.
 *
 * The emphasised segment must be the same dates the headline number averages -- the last
 * `window` *complete* ones -- or the picture and the figure disagree. Dates still maturing
 * are drawn faintly at the tail so the shape stays continuous without implying they count.
 */
function Sparkline({
  series,
  nominal,
  window,
}: {
  series: DayEntry[];
  nominal: number;
  window: number;
}) {
  const W = 240;
  const H = 40;
  if (series.length < 2) return <div className="spark-empty">not enough days yet</div>;

  const x = (i: number) => (i / (series.length - 1)) * W;
  const y = (v: number) => H - v * H;
  const path = (pts: [number, number][]) =>
    pts.map(([i, v], k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  const all: [number, number][] = series.map((d, i) => [i, d.coverage]);
  const complete: [number, number][] = series
    .map((d, i) => [i, d.coverage, d.complete] as [number, number, boolean])
    .filter(([, , c]) => c)
    .map(([i, v]) => [i, v]);
  const counted = complete.slice(-window);
  const end = counted[counted.length - 1];

  return (
    <svg
      className="spark"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`Coverage per forecast date against a nominal ${Math.round(nominal * 100)} percent`}
    >
      <line className="spark-nominal" x1="0" y1={y(nominal)} x2={W} y2={y(nominal)} />
      <path className="spark-all" d={path(all)} />
      {counted.length > 1 && <path className="spark-recent" d={path(counted)} />}
      {end && <circle className="spark-end" cx={x(end[0])} cy={y(end[1])} r="2.5" />}
    </svg>
  );
}
