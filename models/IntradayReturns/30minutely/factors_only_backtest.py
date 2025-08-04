#!/usr/bin/env python
# backtest.py – generate out-of-sample 30-min predictions (fixed universe)
# --------------------------------------------------------------------------
# • For each snapshot month, forecast ˆX_{i,t+1} with factor model only.
# • Universe is already gap-free → never drop bars or intersect timestamps.
#
# Output per month:
#   models/…/backtest_outputs/<YYYY-MM-DD>/predictions.parquet
#
from __future__ import annotations
import json, yaml, pathlib, datetime as dt, argparse
from typing import List, Dict
import numpy as np
import polars as pl

cli = argparse.ArgumentParser()
cli.add_argument("--first", help="first snapshot YYYY-MM-DD (inclusive)")
cli.add_argument("--last",  help="last  snapshot YYYY-MM-DD (inclusive)")
cli.add_argument("--excess", action="store_true",help="use market-excess returns (subtract SPY)")
args  = cli.parse_args()
first = dt.date.fromisoformat(args.first) if args.first else None
last  = dt.date.fromisoformat(args.last)  if args.last  else None
use_excess = args.excess   

ROOT = pathlib.Path("~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns").expanduser()
BARS = ROOT / "data_bars" / "30m"
SNAP = ROOT / "snapshots"

THIS = pathlib.Path(__file__).resolve().parent
CFG  = yaml.safe_load((THIS / "hyperparameters.yaml").read_text())
OUT  = THIS / "backtest_outputs"; OUT.mkdir(exist_ok=True)

np.random.seed(CFG["SEED"])

class BarReader:
    """Memory-maps one month of 30-minute bars for a fixed ticker list."""
    def __init__(self, tickers: List[str], excess: bool = True):
        self.tickers = tickers
        self.excess  = excess
        self.cache: Dict[str, pl.DataFrame] = {}
        self.loaded_ym: str | None = None
        self.ts_vec: np.ndarray | None = None

    def _ensure_month(self, ts: dt.datetime):
        ym = ts.strftime("%Y-%m")
        if ym == self.loaded_ym:
            return
        self.cache.clear()
        needed = set(self.tickers) | ({"SPY"} if self.excess else set())
        for t in needed:
            pf = BARS / t / f"{ym}.parquet"
            self.cache[t] = (
                pl.read_parquet(pf)
                  .with_columns(pl.col("ts").dt.replace_time_zone(None))
            )
        # universe is gap-free → time-vector from any ticker
        self.ts_vec = self.cache[self.tickers[0]]["ts"].to_numpy()
        self.loaded_ym = ym

    def row(self, ts: dt.datetime) -> np.ndarray:
        """Return a vector of raw- or excess-returns for the given bar."""
        self._ensure_month(ts)

        # ---------- helper to fetch a single value safely -----------------
        def _value(tkr: str) -> float:
            ser = self.cache[tkr].filter(pl.col("ts") == ts)["log_ret"]
            if ser.len() == 1:
                val = ser.item()
                return float(val) if val is not None else np.nan   # ensure nan, never None
            return np.nan                                           # bar missing ⇒ nan


        if not self.excess:
            # ---- raw (simple) returns -----------------------------------
            return np.array([_value(t) for t in self.tickers], dtype=np.float32)

        # ---- market-excess returns --------------------------------------
        spy_ret = _value("SPY")          # may be NaN if SPY bar missing

        def _excess(tkr: str) -> float:
            val = _value(tkr)
            return val - spy_ret if np.isfinite(val) and np.isfinite(spy_ret) else np.nan

        return np.array([_excess(t) for t in self.tickers], dtype=np.float32)


class FactorBuf:
    def __init__(self, tail: np.ndarray):
        self.buf = tail.copy()            # shape (lF, k)
    def predict(self, P: np.ndarray) -> np.ndarray:
        return np.einsum("lkj,lj->k", P, self.buf)
    def roll(self, f_new: np.ndarray):
        self.buf[:-1] = self.buf[1:]
        self.buf[-1]  = f_new

# snapshot loader 
def load_snap(d: pathlib.Path):
    meta   = json.loads((d / "meta.json").read_text())
    k, lF  = meta["k_hat"], meta["lF"]
    L      = pl.read_parquet(d / "L.parquet").to_numpy().astype(np.float32)
    P_hat  = (pl.read_parquet(d / "P_hat.parquet")
                    .to_numpy().reshape(lF, k, k).astype(np.float32))
    Phi    = pl.read_parquet(d / "Phi.parquet").to_numpy().astype(np.float32)
    Fbuf   = FactorBuf(pl.read_parquet(d / "F.parquet")
                           .tail(lF).to_numpy().astype(np.float32))
    xi_prev = (pl.read_parquet(d / "Xi.parquet")
                  .tail(1).to_numpy().ravel().astype(np.float32))
    tickers = pl.read_csv(d / "coverage.csv")["ticker"].to_list()
    return L, P_hat, Phi, Fbuf, xi_prev, tickers

def run():
    snap_dirs = [d for d in sorted(SNAP.iterdir())
                 if (first is None or dt.date.fromisoformat(d.name) >= first)
                 and (last  is None or dt.date.fromisoformat(d.name) <= last)]

    for si, sdir in enumerate(snap_dirs):
        L, P_hat, Phi, Fbuf, xi_t, tickers = load_snap(sdir)
        reader  = BarReader(tickers, excess=use_excess)
        preds   = []

        # bar timeline for the forward month
        month_first   = dt.date.fromisoformat(sdir.name) + dt.timedelta(days=1)
        ts_start      = dt.datetime.combine(month_first, dt.time(10, 0))
        reader._ensure_month(ts_start)
        ts_vec = reader.ts_vec

        next_snap = snap_dirs[si+1] if si+1 < len(snap_dirs) else None
        next_dt   = dt.date.fromisoformat(next_snap.name) if next_snap else None

        ts_delta = dt.timedelta(minutes=30)

        for ts64 in ts_vec:
            ts = ts64.astype("datetime64[us]").astype(dt.datetime)
            if ts.date() < month_first:
                print(1)
                continue
            if next_dt and ts.date() >= next_dt:
                print(2)
                break

            #  observe r_t
            r_t      = reader.row(ts)

            # update state
            f_t      = L.T @ r_t
            Fbuf.roll(f_t)

            # forecast t+1
            f_next       = Fbuf.predict(P_hat)
            x_pred_next  = L @ f_next 
            ts_next      = ts + ts_delta           # 30-minute-ahead time stamp

            # skip the synthetic 16:30 bar
            if ts.time() != dt.time(16, 0):        # only append if current bar isn’t 16:00
                preds.append([ts_next.isoformat(), *x_pred_next])


        mdir = OUT / sdir.name                 # backtest_outputs/2007-10-31/
        mdir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(preds, schema=["ts", *tickers], orient="row") \
          .write_parquet(mdir / "predictions.parquet")

        print(f"[{sdir.name}]  {len(preds):,} bar predictions written")

if __name__ == "__main__":
    run()
