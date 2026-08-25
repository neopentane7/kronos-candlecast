import { useEffect, useRef } from "react";
import { createChart, type IChartApi, ColorType, LineStyle } from "lightweight-charts";
import type { Forecast, Session } from "../types";

const MUTED = "#7f9794";
const GROUND = "#0d1b1e";
const GRID = "#16292d";
const UP = "#3f9e79";
const DOWN = "#c2603f";
const CONE_FAINT = "rgba(79, 176, 184, 0.16)";
const CONE_STRONG = "rgba(79, 176, 184, 0.34)";
const MEDIAN = "#4fb0b8";

/**
 * Candles plus the forecast cone.
 *
 * lightweight-charts has no band series, so each band is drawn as an area filled to the
 * bottom and then partly masked by the next one. Painting order is faint, strong, faint,
 * ground: p90 faint, p75 strong over it, p25 faint again, p10 in the background colour.
 * What survives is p25-p75 strong, p10-p25 and p75-p90 faint, and nothing below p10 --
 * which is the nesting the contract describes.
 */
export function Cone({ history, forecast }: { history: Session[]; forecast: Forecast }) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!box.current) return;

    const c = createChart(box.current, {
      layout: { background: { type: ColorType.Solid, color: GROUND }, textColor: MUTED },
      grid: { vertLines: { color: GRID }, horzLines: { color: GRID } },
      rightPriceScale: { borderColor: GRID },
      timeScale: { borderColor: GRID, timeVisible: false, rightOffset: 4 },
      crosshair: { mode: 0 },
      handleScale: { axisPressedMouseMove: false },
      autoSize: true,
    });
    chart.current = c;

    // The cone starts at the last observed close so it visibly grows out of the candles
    // rather than floating detached from them.
    const anchor = history[history.length - 1];
    const band = (key: keyof Forecast["quantiles"]) => [
      { time: anchor.t, value: anchor.c },
      ...forecast.timestamps.map((t, i) => ({ time: t, value: forecast.quantiles[key][i] })),
    ];

    const area = (key: keyof Forecast["quantiles"], fill: string) => {
      const s = c.addAreaSeries({
        topColor: fill,
        bottomColor: fill,
        lineColor: "rgba(0,0,0,0)",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData(band(key) as never);
    };

    area("p90", CONE_FAINT);
    area("p75", CONE_STRONG);
    area("p25", CONE_FAINT);
    area("p10", GROUND);

    const median = c.addLineSeries({
      color: MEDIAN,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    median.setData(band("p50") as never);

    const candles = c.addCandlestickSeries({
      upColor: UP,
      downColor: DOWN,
      wickUpColor: UP,
      wickDownColor: DOWN,
      borderVisible: false,
    });
    candles.setData(
      history.map((s) => ({ time: s.t, open: s.o, high: s.h, low: s.l, close: s.c })) as never,
    );

    candles.createPriceLine({
      price: forecast.last_close,
      color: MUTED,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      axisLabelVisible: true,
      title: "last close",
    });

    c.timeScale().fitContent();
    return () => {
      c.remove();
      chart.current = null;
    };
  }, [history, forecast]);

  return (
    <div className="chart" ref={box} role="img" aria-label={coneSummary(forecast)} />
  );
}

/** Screen readers get the number, not a canvas they cannot see. */
function coneSummary(f: Forecast): string {
  const last = f.horizon - 1;
  const lo = f.quantiles.p10[last].toFixed(0);
  const hi = f.quantiles.p90[last].toFixed(0);
  return (
    `${f.ticker}: last close ${f.last_close.toFixed(2)}. In ${f.horizon} sessions the 80% ` +
    `band spans ${lo} to ${hi}.`
  );
}
