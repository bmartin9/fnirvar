# backtest_statistics.py – compute intraday back‑test statistics
# -----------------------------------------------------------------------------
# Reads:
#   backtest_outputs/predictions.parquet       – produced by backtest.py
#   data_bars/30m/<TICKER>/<YYYY‑MM>.parquet   – realised excess returns
# Writes:
#   backtest_outputs/metrics.json              – aggregate stats (hit, corr, pnl)
#   backtest_outputs/pnl_bar.parquet           – per‑bar gross & cost‑adjusted PnL
#
# Usage (single core):
#   python evaluate_predictions.py  --transaction_cost 2  --quantile 0.2
# -----------------------------------------------------------------------------
from __future__ import annotations
import json, pathlib, datetime as dt, argparse
import numpy as np
import polars as pl
from scipy.stats import spearmanr, rankdata

ROOT = pathlib.Path("~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns").expanduser()
BARS = ROOT / "data_bars" / "30m"
PRED = pathlib.Path(__file__).resolve().parent / "backtest_outputs" / "predictions.parquet"
OUT  = pathlib.Path(__file__).resolve().parent / "backtest_outputs"

# --------------------------- benchmark class (vectorised) --------------------
class IntradayBenchmark:
    def __init__(self, preds: pl.DataFrame, rets: pl.DataFrame, tcost_bps: float = 0.0):
        # both DataFrames share the same ts index & asset columns order
        self.ts   = preds["ts"].to_numpy()
        self.cols = preds.columns[1:]
        self.p = preds.select(self.cols).to_numpy().astype(np.float32)
        self.r = rets.select(self.cols).to_numpy().astype(np.float32)
        assert self.p.shape == self.r.shape, "Predictions/returns mismatch"
        self.T, self.N = self.p.shape
        self.tcost = tcost_bps / 10_000
        self.sign_pred = np.sign(self.p)
        self.sign_rets = np.sign(self.r)

    # ---- ratios -------------------------------------------------------------
    def hit_ratio(self):
        return float(((self.sign_pred * self.sign_rets) > 0).mean())

    def long_ratio(self):
        return float((self.sign_pred > 0).mean())

    # ---- bar‑wise Spearman avg ---------------------------------------------
    def spearman(self):
        rho = [spearmanr(self.p[t], self.r[t], nan_policy='omit')[0] for t in range(self.T)]
        return float(np.nanmean(rho))

    # ---- equal‑weight PnL with turnover cost -------------------------------
    def pnl(self, q: float = 1.0):
        w_prev = np.zeros(self.N, dtype=np.float32)
        gross, net = [], []
        p_abs = np.abs(self.p)
        for t in range(self.T):
            cutoff = np.quantile(p_abs[t], 1 - q) if q < 1 else -np.inf
            mask   = p_abs[t] >= cutoff
            w      = np.zeros(self.N, dtype=np.float32)
            longs, shorts = (self.p[t] > 0) & mask, (self.p[t] <= 0) & mask
            if longs.any():  w[longs]  =  1 / longs.sum()
            if shorts.any(): w[shorts] = -1 / shorts.sum()
            turnover = np.sum(np.abs(w - w_prev))
            g = w @ self.r[t]
            n = g - self.tcost * turnover
            gross.append(g); net.append(n)
            w_prev = w
        return np.array(gross), np.array(net)

# --------------------------- build realised returns -------------------------

def load_realised(pred_df: pl.DataFrame) -> pl.DataFrame:
    cols = pred_df.columns[1:]
    realised_rows = []
    for ts in pred_df["ts"].to_numpy():
        ts_dt = dt.datetime.fromisoformat(ts)
        ym    = ts_dt.strftime("%Y-%m")
        row = []
        for tkr in cols:
            val = pl.read_parquet(BARS / tkr / f"{ym}.parquet") \
                    .with_columns(pl.col("ts").dt.replace_time_zone(None)) \
                    .filter(pl.col("ts") == ts_dt)["log_ret"].item()
            row.append(val)
        realised_rows.append([ts, *row])
    return pl.DataFrame(realised_rows, schema=pred_df.columns)

# --------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transaction_cost", type=float, default=0.0, help="cost in bps")
    ap.add_argument("--quantile", type=float, default=1.0, help="top‑x%% magnitude to trade")
    args = ap.parse_args()

    preds = pl.read_parquet(PRED)
    rets  = load_realised(preds)

    bench = IntradayBenchmark(preds, rets, tcost_bps=args.transaction_cost)
    gross, net = bench.pnl(q=args.quantile)

    metrics = {
        "num_bars": int(bench.T),
        "assets": int(bench.N),
        "hit_ratio": bench.hit_ratio(),
        "long_ratio": bench.long_ratio(),
        "spearman": bench.spearman(),
        "gross_pnl_sum": float(gross.sum()),
        "net_pnl_sum":   float(net.sum()),
    }

    # write outputs
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pl.DataFrame({"ts": bench.ts, "gross": gross, "net": net}) \
        .write_parquet(OUT / "pnl_bar.parquet")
    print("saved metrics →", OUT / "metrics.json")

if __name__ == "__main__":
    main()
