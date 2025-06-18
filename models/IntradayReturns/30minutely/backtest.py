# backtest.py – generate out-of-sample 30‑minute predictions 
# --------------------------------------------------------------------------
# • For each snapshot month, produce out‑of‑sample forecasts \hat X_{i,t+1}
#   using factors + idiosyncratic VAR 
#
# Folder layout assumed (unchanged):
#   ~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns/
#       ├─ data_bars/30m/<TICKER>/<YYYY-MM>.parquet   – realised returns
#       └─ snapshots/<YYYY-MM-DD>/                    – model params per month‑end
#
# Output:  backtest_outputs/predictions.parquet
#           columns: ts, <tickers…>

from __future__ import annotations
import json, yaml, pathlib, datetime as dt
from typing import List, Dict
import numpy as np
import polars as pl
import argparse

# -------------- Arguments --------------
cli = argparse.ArgumentParser()
cli.add_argument("--first", help="first snapshot YYYY-MM-DD (inclusive)")
cli.add_argument("--last",  help="last  snapshot YYYY-MM-DD (inclusive)")
args = cli.parse_args()
first = dt.date.fromisoformat(args.first) if args.first else None
last  = dt.date.fromisoformat(args.last)  if args.last  else None
# -------------------------------------

# ---------- paths ----------------------------------------------------------
ROOT = pathlib.Path("~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns").expanduser()
BARS = ROOT / "data_bars" / "30m"
SNAP = ROOT / "snapshots"
THIS = pathlib.Path(__file__).resolve().parent
CFG  = yaml.safe_load((THIS / "hyperparameters.yaml").read_text())
OUT  = THIS / "backtest_outputs"; OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(CFG["SEED"])

# ---------- helper classes -------------------------------------------------
class BarReader:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.cache: Dict[str, pl.DataFrame] = {}
        self.loaded_ym: str | None = None
        self.ts_vec: np.ndarray | None = None 

    def _ensure_month(self, ts: dt.datetime):
        ym = ts.strftime("%Y-%m")
        if ym == self.loaded_ym:
            return
        self.cache.clear()
        for t in self.tickers:
            self.cache[t] = pl.read_parquet(BARS / t / f"{ym}.parquet")
            pf = BARS / t / f"{ym}.parquet"
            self.cache[t] = (pl.read_parquet(pf).with_columns(pl.col("ts").dt.replace_time_zone(None)))
        self.ts_vec = self.cache[self.tickers[0]]["ts"].to_numpy()
        self.loaded_ym = ym

    def row(self, ts: dt.datetime) -> np.ndarray:
        self._ensure_month(ts)
        vec = []
        for t in self.tickers:
            ser = self.cache[t].filter(pl.col("ts") == ts)["log_ret"] 
            if ser.len() != 1:                       
                raise ValueError(f"{t} has {ser.len()} rows for {ts}") 
            vec.append(ser.to_numpy()[0])
        return np.array(vec, dtype=np.float32)
        
class FactorBuf:
    def __init__(self, tail: np.ndarray):
        self.buf = tail.copy()  # shape (lF, k)
    def predict(self, P: np.ndarray) -> np.ndarray:
        return np.einsum("lkj,lj->k", P, self.buf)
    def roll(self, f_new: np.ndarray):
        self.buf[:-1] = self.buf[1:]
        self.buf[-1]  = f_new

# ---------- load snapshot --------------------------------------------------

def load_snap(d: pathlib.Path):
    meta = json.loads((d/"meta.json").read_text())
    k, lF = meta["k_hat"], meta["lF"]
    L     = pl.read_parquet(d/"L.parquet").to_numpy().astype(np.float32)
    P_hat = pl.read_parquet(d/"P_hat.parquet").to_numpy().reshape(lF,k,k).astype(np.float32)
    Phi   = pl.read_parquet(d/"Phi.parquet").to_numpy().astype(np.float32)
    Fbuf  = FactorBuf(pl.read_parquet(d/"F.parquet").tail(lF).to_numpy().astype(np.float32))
    xi_prev = pl.read_parquet(d/"Xi.parquet").tail(1).to_numpy().ravel().astype(np.float32)
    tickers = pl.read_csv(d/"coverage.csv")["ticker"].to_list()
    return meta,L,P_hat,Phi,Fbuf,xi_prev,tickers

# ---------- main -----------------------------------------------------------

def run():
    preds = []  # rows: ts + predictions vector
    snap_dirs = [d for d in sorted(SNAP.iterdir())
             if (first is None or dt.date.fromisoformat(d.name) >= first)
             and (last  is None or dt.date.fromisoformat(d.name) <= last)]
    reader = None

    for si, sdir in enumerate(snap_dirs):
        meta,L,P_hat,Phi,Fbuf,xi_prev,tickers = load_snap(sdir)
        reader = BarReader(tickers)

        next_snap = snap_dirs[si+1] if si+1 < len(snap_dirs) else None
        next_snap_dt = dt.date.fromisoformat(next_snap.name) if next_snap else None

        month_first = dt.date.fromisoformat(sdir.name) + dt.timedelta(days=1)
        ts_start    = dt.datetime.combine(month_first, dt.time(9, 30))

        reader._ensure_month(ts_start)

        ts_vec = reader.ts_vec         # np.ndarray of datetime64
        have_pos = False               # no position for first bar


        for ts64 in ts_vec:
            ts = ts64.item()    
            if ts.date() < (dt.date.fromisoformat(sdir.name)+dt.timedelta(days=1)):
                continue
            if next_snap_dt and ts.date() >= next_snap_dt:
                break

            if have_pos:
                r_t = reader.row(ts)
                xi_prev = r_t - (L @ f_next)   # realised residual
                Fbuf.roll(f_next)

            # --- forecast for NEXT bar (out-of-sample) ---------------
            f_next = Fbuf.predict(P_hat)
            x_pred_next = L @ f_next + Phi @ xi_prev
            preds.append([ts.isoformat(), *x_pred_next])
            have_pos = True



    # save predictions ------------------------------------------------------
    cols = ["ts", *tickers]
    pl.DataFrame(preds, schema=cols, orient="row").write_parquet(OUT / "predictions.parquet")

    print(f"Saved {len(preds):,} bar predictions → {OUT/'predictions.parquet'}")

if __name__ == "__main__":
    run()
