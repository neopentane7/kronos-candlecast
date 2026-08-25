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
