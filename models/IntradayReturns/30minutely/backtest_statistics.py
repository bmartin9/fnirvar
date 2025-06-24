#!/usr/bin/env python
# backtest_statistics.py – intraday evaluation (30-minute bars)
# -----------------------------------------------------------------------------
from __future__ import annotations
import json, pathlib, argparse, datetime as dt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

# ───────── paths ────────────────────────────────────────────────────────────
ROOT = pathlib.Path(
    "~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns"
).expanduser()
PRED = pathlib.Path(__file__).resolve().parent / "backtest_outputs" / "pred_all.parquet"
REAL = pathlib.Path(__file__).resolve().parent / "realised_all.parquet"
OUT  = pathlib.Path(__file__).resolve().parent / "30minutely_results"
OUT.mkdir(exist_ok=True)

# ───────── benchmark class ──────────────────────────────────────────────────
class IntradayBenchmark:
    def __init__(self, preds: pl.DataFrame, rets: pl.DataFrame, tcost_bps=0.0):
        self.ts   = preds["ts"].to_numpy()
        self.cols = preds.columns[1:]
        self.p = preds.select(self.cols).to_numpy().astype(np.float32)
        self.r = rets .select(self.cols).to_numpy().astype(np.float32)
        print(self.p.shape, self.r.shape)
        assert self.p.shape == self.r.shape, "prediction / realised mis-align"

        self.T, self.N = self.p.shape
        self.tcost     = tcost_bps / 10_000.0
        self.sign_pred = np.sign(self.p)
        self.sign_rets = np.sign(self.r)

    # hit / long ratios ------------------------------------------------------
    def hit_ratio (self): return float(((self.sign_pred * self.sign_rets) > 0).mean())
    def long_ratio(self): return float((self.sign_pred > 0).mean())

    # average bar-wise Spearman ---------------------------------------------
    def spearman(self):
        return float(np.nanmean([
            spearmanr(self.p[t], self.r[t], nan_policy="omit")[0]
            for t in range(self.T)
        ]))

    # equal-weight PnL with turnover cost -----------------------------------
    import numpy as np

    def pnl(self, q: float = 1.0):
        """
        Returns
        -------
        gross : np.ndarray[float]
            Per-bar gross PnL.
        net   : np.ndarray[float]
            Per-bar net PnL after turnover cost.
        trades: np.ndarray[int]
            Number of round-trip trades executed in each bar.
        """
        w_prev = np.zeros(self.N, np.float32)

        gross, net, trades = [], [], []
        p_abs = np.abs(self.p)

        for t in range(self.T):
            # ---------- portfolio mask (top-|q|%) -----------------------------
            cutoff = np.quantile(p_abs[t][~np.isnan(p_abs[t])], 1 - q) if q < 1 else -np.inf
            mask   = p_abs[t] >= cutoff

            # ---------- target weights  --------------------------------------
            w = np.zeros(self.N, np.float32)
            longs, shorts = (self.p[t] > 0) & mask, (self.p[t] <= 0) & mask
            if longs.any():  w[longs]  =  1.0 / longs.sum()
            if shorts.any(): w[shorts] = -1.0 / shorts.sum()

            # ---------- turnover & trades ------------------------------------
            dw       = np.abs(w - w_prev)
            turnover = np.sum(dw)                    # ∑|Δw|
            n_trades = int(np.count_nonzero(dw))     # trades executed this bar

            # ---------- PnL ---------------------------------------------------
            g = np.nansum(w * self.r[t])             # ignore NaNs in returns panel
            n = g - self.tcost * turnover

            gross.append(g)
            net.append(n)
            trades.append(n_trades)

            w_prev = w

        return (np.asarray(gross, dtype=np.float32),
                np.asarray(net,   dtype=np.float32),
                np.asarray(trades, dtype=np.int32))


# ───────── main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transaction_cost", type=float, default=0.0, help="bps per round-trip")
    ap.add_argument("--quantile",         type=float, default=1.0, help="top-x%% magnitude to trade")
    args = ap.parse_args()

    # 1. load & shift predictions ------------------------------------------
    preds = pl.read_parquet(PRED).with_columns(
        (pl.col("ts") + dt.timedelta(minutes=30)).alias("ts")   # t  → t+1
    )

    # drop the synthetic 16:30 bar (predicts overnight)
    preds = preds.filter(pl.col("ts").dt.time() != dt.time(16, 30))

    # 2. load realised and discard the very first bar (10:00)
    reals = (pl.read_parquet(REAL)
               .filter(pl.col("ts").dt.time() != dt.time(10, 0)))

    # 3. inner-join on timestamp – keeps only rows with both pred & return
    reals = reals.join(preds.select("ts"), on="ts", how="inner")

    # align column order
    reals = reals.select(preds.columns)

    # ── after applying the filters ─────────────────────────────────────────
    pred_ts  = preds["ts"]
    real_ts  = reals["ts"]

    # timestamps that exist in preds but not in reals
    only_pred = pred_ts.filter(~pred_ts.is_in(real_ts))

    # timestamps that exist in reals but not in preds (should be empty now)
    only_real = real_ts.filter(~real_ts.is_in(pred_ts))

    print(f"\nTimestamps only in PREDICTIONS  ({len(only_pred)} rows)")
    print(only_pred.head(25))          # show the first few, adjust as you like

    print(f"\nTimestamps only in REALISED     ({len(only_real)} rows)")
    print(only_real.head(25))


    # ---- enforce identical timestamp set ---------------------------------
    common_ts = preds.join(reals.select("ts"), on="ts", how="inner")["ts"]

    preds = preds.filter(pl.col("ts").is_in(common_ts)).sort("ts")
    reals = reals.filter(pl.col("ts").is_in(common_ts)).sort("ts")


    assert preds.shape == reals.shape  


    # 4. metrics ------------------------------------------------------------
    bench = IntradayBenchmark(preds, reals, tcost_bps=args.transaction_cost)
    gross, net, trades = bench.pnl(q=args.quantile)

    # ------------------------------------------------------------------
    # 1. Sharpe ratios (annualised)          ───────────────────────────
    #    252 trading days × 12 day-bars for 30-minute frequency
    ann_fact = np.sqrt(252 * 12)
    sharpe_gross = float(np.nanmean(gross) / np.nanstd(gross, ddof=1) * ann_fact)
    sharpe_net   = float(np.nanmean(net)   / np.nanstd(net,   ddof=1) * ann_fact)

    # daily aggregation (handles half-days)
    pnl_df = pl.DataFrame({"ts": bench.ts, "net": net})
    day_pnl = (pnl_df
               .with_columns(pl.col("ts").dt.date().alias("trade_date"))
               .group_by("trade_date")
               .agg(pl.col("net").sum().alias("day_net"))
               .sort("trade_date"))["day_net"].to_numpy()

    sharpe_day = float(day_pnl.mean() / day_pnl.std(ddof=1) * np.sqrt(252))

    # ------------------------------------------------------------------
    # 2. Avg PnL per trade in bp             ───────────────────────────
    tot_trades = trades.sum()
    pnl_per_trade_bp = float(np.nansum(net) / tot_trades * 10_000)  # 1 bp = 1e-4


    metrics = {
    "num_bars":      bench.T,
    "assets":        bench.N,
    "hit_ratio":     bench.hit_ratio(),
    "long_ratio":    bench.long_ratio(),
    "spearman":      bench.spearman(),

    "gross_pnl_sum": float(np.nansum(gross)),
    "net_pnl_sum":   float(np.nansum(net)),
    "sharpe_gross":  sharpe_gross,
    "sharpe_net":    sharpe_net,
    "sharpe_day":    sharpe_day,
    "trades":        int(tot_trades),
    "pnl_per_trade_bp": pnl_per_trade_bp,
}


    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pl.DataFrame({"ts": bench.ts, "gross": gross, "net": net}) \
        .write_parquet(OUT / "pnl_bar.parquet")

    print("saved metrics →", OUT / "metrics.json")

if __name__ == "__main__":
    main()
