#!/usr/bin/env python
# detect_bad_rows_all.py – scan every snapshot for NaN/Inf rows
# --------------------------------------------------------------------------
# Creates two CSVs:
#   bad_rows_any.csv  – at least one bad value in the row
#   bad_rows_all.csv  – every value in the row is bad
# Columns: snapshot,timestamp,bad_tickers   (bad_tickers = ";"-separated list)
# --------------------------------------------------------------------------
from __future__ import annotations
import pathlib, datetime as dt, csv, polars as pl, numpy as np

BAR_FREQ       = "30m"
LOOKBACK_DAYS  = 60
ROOT   = pathlib.Path(
    "~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns"
).expanduser()
SNAPS  = ROOT / "snapshots"
BARS   = ROOT / "data_bars" / BAR_FREQ

OUT_DIR  = pathlib.Path(__file__).resolve().parent
ANY_CSV  = OUT_DIR / "bad_rows_any.csv"
ALL_CSV  = OUT_DIR / "bad_rows_all.csv"

# --------------------------------------------------------------------------
def spy_ts_vector(month_end: dt.date) -> list[str]:
    """Return ISO timestamps for the 60-day window via SPY (gap-free)."""
    start = month_end - dt.timedelta(days=LOOKBACK_DAYS)
    spy_files = list((BARS / "SPY").glob("*.parquet"))
    return (
        pl.scan_parquet(spy_files)
          .filter((pl.col("ts") >= start) & (pl.col("ts") <= month_end))
          .select("ts").sort("ts").collect()["ts"]
          .to_list()
    )

# --------------------------------------------------------------------------
def inspect_snapshot(sdir: pathlib.Path,
                     w_any: csv.writer,
                     w_all: csv.writer):
    x_file = sdir / "X.parquet"
    if not x_file.exists():
        return

    X          = pl.read_parquet(x_file)
    finite_mat = X.select(pl.all().is_finite()).to_numpy()   # shape (T,N_bool)
    any_bad    = ~finite_mat.all(axis=1)        # True if at least one bad
    all_bad    = ~finite_mat.any(axis=1)        # True if all bad

    if not any_bad.any():
        return

    ts_vec  = spy_ts_vector(dt.date.fromisoformat(sdir.name))
    cols    = X.columns
    X_np    = X.to_numpy()

    for i, bad in enumerate(any_bad):
        if not bad:
            continue
        bad_cols = [cols[j] for j in np.where(~np.isfinite(X_np[i]))[0]]
        row = [sdir.name, ts_vec[i], ";".join(bad_cols)]
        w_any.writerow(row)
        if all_bad[i]:
            w_all.writerow(row)

# --------------------------------------------------------------------------
def main():
    with ANY_CSV.open("w", newline="") as fa, ALL_CSV.open("w", newline="") as fb:
        w_any = csv.writer(fa); w_all = csv.writer(fb)
        header = ["snapshot", "timestamp", "bad_tickers"]
        w_any.writerow(header); w_all.writerow(header)

        for sdir in sorted(SNAPS.iterdir()):
            if sdir.is_dir():
                inspect_snapshot(sdir, w_any, w_all)

    print("wrote", ANY_CSV, "and", ALL_CSV)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
