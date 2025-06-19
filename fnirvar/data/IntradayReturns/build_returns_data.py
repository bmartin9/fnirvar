#!/usr/bin/env python
"""
build_returns_data.py
Convert 1-minute mid-prices (data_parquet/<TICKER>/<YYYY-MM>/…)
into 5- or 30-minute log returns.

• 30-minute variant:   first bar is 10:00, return = log(P10:00 / P09:30)
                       (overnight jump 16:00→09:30 is excluded)
• 5-minute variant  :  trims the opening/closing 15 min, so first bar
                       is 09:45–09:50, last bar is 15:45–15:50.

Usage
-----
# Single asset, 30-minute bars
python build_returns_data.py --bar 30m AAPL

# All assets, 5-minute bars
python build_returns_data.py --bar 5m
"""
from __future__ import annotations
import argparse, pathlib, glob, datetime as dt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import polars as pl

SRC_ROOT = pathlib.Path("data_parquet")   # minute layer
DST_ROOT = pathlib.Path("data_bars")      # output root

# ──────────────────────────────────────────────────────────────────
def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", choices=("5m", "30m"), required=True,
                    help="Bar width (5m trims opening/closing 15 min)")
    ap.add_argument("tickers", nargs="*",
                    help="Tickers (default: every folder in data_parquet)")
    return ap.parse_args()

def all_assets() -> list[str]:
    return sorted(p.name for p in SRC_ROOT.iterdir() if p.is_dir())

# ──────────────────────────────────────────────────────────────────
def resample_asset(asset: str, bar: str):
    # 1) gather minute files
    pattern = SRC_ROOT / asset / "*/*.parquet"
    minute_files = glob.glob(str(pattern))
    if not minute_files:
        print(f"[{asset}]  ⚠  no minute files — skipping")
        return

    # 2) lazy scan
    lf = pl.scan_parquet(minute_files).with_columns(
            pl.col("ts").dt.replace_time_zone(None)  # drop UTC
         )

    if bar == "5m":
        lf = lf.filter(
            (pl.col("ts").dt.time() >= dt.time(9, 45)) &
            (pl.col("ts").dt.time() <  dt.time(15, 45))
        )
        every = "5m"
    else:
        every = "30m"

    # 3) build bars: [start, end] inclusive, label by window end
    bars = (
        lf.sort("ts")
          .group_by_dynamic(
              index_column="ts",
              every=every,
              period=every,
              closed="both",     # include open & close rows
              label="right",     # stamp by window end (10:00, 10:30…)
          )
          .agg([
              pl.col("mid_px").first().alias("px_open"),
              pl.col("mid_px").last().alias("px_close"),
              pl.len().alias("n"),
          ])
          .with_columns(                       # log-return
              (pl.col("px_close").log() - pl.col("px_open").log())
              .alias("log_ret"),
              pl.col("ts").dt.strftime("%Y-%m").alias("ym"),
          )
          .filter(pl.col("n") > 1) 
          .drop("px_open", "px_close", "n")
          .collect()
    )
    if bars.height == 0:
        print(f"[{asset}]  ⚠  resample produced 0 rows")
        return

    # 4) write one parquet per YYYY-MM
    out_root = DST_ROOT / bar / asset
    out_root.mkdir(parents=True, exist_ok=True)
    for key, grp in bars.partition_by("ym", as_dict=True).items():
        ym = key[0] if isinstance(key, tuple) else key
        grp.drop("ym").write_parquet(out_root / f"{ym}.parquet",
                                     compression="snappy")

    print(f"[{asset}]  ✓  wrote {bars.height:,} bars")

# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args    = parse_cli()
    tickers = args.tickers or all_assets()

    with ProcessPoolExecutor() as ex:
        for _ in ex.map(resample_asset, tickers, repeat(args.bar)):
            pass   # forces iteration so exceptions surface
