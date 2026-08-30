import type { Forecast, History, Index, TrackRecord } from "./types";

// Vite rewrites BASE_URL to the deployed prefix, so the same code works on
// localhost:5173 and on github.io/kronos-candlecast/ without a build-time branch.
const DATA = `${import.meta.env.BASE_URL}data`;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
  return (await res.json()) as T;
}

export const fetchIndex = () => getJSON<Index>(`${DATA}/index.json`);
export const fetchForecast = (t: string) => getJSON<Forecast>(`${DATA}/forecasts/${t}.json`);
export const fetchHistory = (t: string) => getJSON<History>(`${DATA}/history/${t}.json`);
// Published by the nightly job; absent on a fresh deploy before the first run, which
// the caller treats as "no track record yet" rather than as an error.
export const fetchTrackRecord = () => getJSON<TrackRecord>(`${DATA}/track_record.json`);
