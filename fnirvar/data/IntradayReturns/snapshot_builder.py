#!/usr/bin/env python
"""
snapshot_builder.py
Build one 60-day design matrix X(T×N) per calendar month-end.

Usage
-----
# build a single month
python snapshot_builder.py  --month 2010-06-30

# build every month in parallel (simple local fan-out)
python snapshot_builder.py  --all
"""
from __future__ import annotations
import argparse, json, pathlib, datetime as dt, glob
import polars as pl

BAR_FREQ       = "30m"
LOOKBACK_DAYS  = 60
SRC_ROOT       = pathlib.Path("data_bars") / BAR_FREQ
DST_ROOT       = pathlib.Path("snapshots")


# ───────────────────────── CLI ──────────────────────────
def parse_cli():
    ap = argparse.ArgumentParser()
    g  = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--month", help="Single month-end YYYY-MM-DD")
    g.add_argument("--all", action="store_true", help="Build every month")
    ap.add_argument("--no-demean", action="store_true",
                    help="Skip column-wise de-meaning before saving X")
    ap.add_argument("--excess", action="store_true",
                    help="Work with market-excess returns (subtract SPY)")
    return ap.parse_args()


def month_ends(start="2007-08-31", stop="2021-12-31"):
    d = dt.date.fromisoformat(start)
    end = dt.date.fromisoformat(stop)
    while d <= end:
        # next month-end = last calendar day of current month
        month_end = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
        yield month_end
        d = (d + dt.timedelta(days=32)).replace(day=1)

# ───────────────────────── snapshot builder ─────────────
def build_snapshot(month_end: dt.date):
    out_dir = DST_ROOT / month_end.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    window_start = month_end - dt.timedelta(days=LOOKBACK_DAYS)

    # 1️⃣ build the reference index from SPY (always present) -------------
    spy_files = glob.glob(str(SRC_ROOT / "SPY" / "*.parquet"))
    spy_ts = (pl.scan_parquet(spy_files)
                .filter((pl.col("ts") >= window_start) & (pl.col("ts") <= month_end))
                .select("ts")
                .sort("ts")
                .collect()
                .get_column("ts"))
    T = len(spy_ts)                          # TRUE required row count

    # 2️⃣ scan every ticker and keep those with perfect match -------------
    good, matrices = [], []
    for tkr_path in SRC_ROOT.iterdir():
        tkr = tkr_path.name
        files = glob.glob(str(tkr_path / "*.parquet"))
        if not files:
            continue

        df = (pl.scan_parquet(files)
                .filter((pl.col("ts").is_in(spy_ts.implode())))
                .sort("ts")
                .select("log_ret")
                .collect())
        if len(df) == T:
            matrices.append(df["log_ret"])
            good.append(tkr)

    N = len(good)
    if N == 0:
        print(f"{month_end} → no assets with full coverage; skipping")
        return


    # 1⃣  build raw X
    X = pl.DataFrame({t: col for t, col in zip(good, matrices)})

    # 2⃣  optional market-excess
    if args.excess:
        if "SPY" not in X.columns:
            raise SystemExit("SPY column missing; cannot compute excess returns")
        spy_vec = X["SPY"].to_frame()
        X = X.select([pl.col(c) - spy_vec["SPY"] for c in X.columns])
        # drop SPY from the matrix and coverage list
        if "SPY" in X.columns:
            X = X.drop("SPY")
            good.remove("SPY")

    # 3⃣  optional de-mean
    if not args.no_demean:
        X = (X - X.mean(axis=0, keepdims=True))

    # 4⃣  cast & write
    X = X.with_columns(pl.all().cast(pl.Float32))
    X.write_parquet(out_dir / "X.parquet", compression="snappy")
    pl.DataFrame({"ticker": good}).write_csv(out_dir / "coverage.csv")

    # 5⃣  record metadata
    meta = dict(bar=BAR_FREQ, lookback_days=LOOKBACK_DAYS, T=X.height, N=N)
    meta.update({
    "demeaned": not args.no_demean,
    "excess": args.excess,
    "bars": BAR_FREQ,
    "lookback_days": LOOKBACK_DAYS,
    "T": X.height,
    "N": len(good)
})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"{month_end} → {N} assets, {T} bars  ✓")

# ───────────────────────── main ─────────────────────────
if __name__ == "__main__":
    args = parse_cli()
    months = ([dt.date.fromisoformat(args.month)] if args.month
              else list(month_ends()))
    for m in months:
        build_snapshot(m)
