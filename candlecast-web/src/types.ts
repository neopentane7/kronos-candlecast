/** Contract v3, mirrored from pipeline/contract.py. Any change there needs one here. */
export interface Quantiles {
  p10: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p90: number[];
}

export interface BandMethods {
  band_50: "split_conformal" | "aci" | "none";
  band_80: "split_conformal" | "aci" | "none";
  band_90: "split_conformal" | "aci" | "none";
}

export interface Forecast {
  contract_version: 3;
  ticker: string;
  generated_at: string;
  engine: "rw_drift" | "kronos";
  calibration: "aci" | "split_conformal" | "none";
  challenger: unknown | null;
  horizon: number;
  last_close: number;
  backfilled: boolean;
  timestamps: string[];
  quantiles: Quantiles;
  prob_above_last_close: number[];
  prob_vol_exceeds_recent: number;
  metadata: {
    band_methods: BandMethods;
    aci_gamma: number;
    aci_provisional: boolean;
    ensemble_size: number;
    lookback: number;
    engine_validated: boolean;
    note: string | null;
  };
  disclaimer: string;
}

export interface Session {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
}

export interface History {
  ticker: string;
  sessions: Session[];
}

export interface Index {
  generated_at: string;
  engine: string;
  contract_version: number;
  horizon: number;
  backfilled: boolean;
  tickers: string[];
  skipped: { ticker: string; reason: string }[];
  disclaimer: string;
}

export interface DayEntry {
  date: string;
  coverage: number;
  rows: number;
  tickers: number;
  /** Deepest horizon step resolved on this date. */
  steps: number;
  /** True once the full horizon has matured. Only these enter any average. */
  complete: boolean;
  live: boolean;
}

export interface LevelRecord {
  nominal: number;
  series: DayEntry[];
  trailing: {
    coverage: number;
    days: number;
    window: number;
    excluded_incomplete: number;
    from: string;
    to: string;
  } | null;
  days: number;
  live_days: number;
  cumulative: number | null;
}

export interface TrackRecord {
  window: number;
  unit: string;
  days: number;
  live_days: number;
  first: string | null;
  last: string | null;
  note: string;
  backtest: Record<string, LevelRecord>;
  live: { date: string; by_level: Record<string, { n: number; hits: number; empirical: number }> }[];
}
