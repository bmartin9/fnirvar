#!/usr/bin/env python
"""
build_returns_data.py
Convert 1-minute mid-prices (data_parquet/<TICKER>/<YYYY-MM>/...)
into 5- or 30-minute log returns.

Usage
-----
# Single asset, 30-minute bars
python build_returns_data.py --bar 30m AAPL

# All assets, 5-minute bars (drops first/last 15 min of day)
python build_returns_data.py --bar 5m
"""
from __future__ import annotations
import argparse, pathlib, sys, itertools, datetime as dt, glob
from concurrent.futures import ProcessPoolExecutor
import polars as pl
from itertools import repeat

SRC_ROOT = pathlib.Path("data_parquet")          # minute layer
DST_ROOT = pathlib.Path("data_bars")             # output root

# ──────────────────────────────────────────────────────────────────────
def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", choices=("5m", "30m"), required=True,
                    help="Bar width (5m trims opening/closing 15 min)")
    ap.add_argument("tickers", nargs="*",
                    help="Tickers (default: every folder in data_parquet)")
    return ap.parse_args()

def all_assets() -> list[str]:
    return sorted(p.name for p in SRC_ROOT.iterdir() if p.is_dir())

# trim helper for 5-minute variant
def trim_open_close(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(
        (pl.col("ts").dt.time() >= dt.time(9, 45)) &
        (pl.col("ts").dt.time() <  dt.time(15, 45))
    )

def resample_asset(asset: str, bar: str):
    # ── 1  find all minute-files first (Python glob) ───────────────────────
    pattern = SRC_ROOT / asset / "*/*.parquet"
    minute_files = glob.glob(str(pattern))
    if not minute_files:
        print(f"[{asset}]   ⚠  no minute files found — skipping")
        return

    # ── 2  build lazy scan from that explicit list  ────────────────────────
    lazy = pl.scan_parquet(minute_files).with_columns(pl.col("ts"))

    # drop first & last 15min only for 5-minute bars
    if bar == "5m":
        lazy = lazy.filter(
            (pl.col("ts").dt.time() >= dt.time(9, 45)) &
            (pl.col("ts").dt.time() <  dt.time(15, 45))
        )
        every = "5m"
    else:
        every = "30m"

    # ── 3  resample and compute log-return  ───────────────────────────────
    lazy = lazy.sort("ts")
    bars = (
        lazy.group_by_dynamic(
                index_column="ts",
                every=every,             # "5m" or "30m"
                closed="right",
                label="right",
        )
        .agg(pl.col("mid_px").last().alias("px"))
        .with_columns(
            pl.col("px").log().diff().alias("log_ret"),
            pl.col("ts").dt.strftime("%Y-%m").alias("ym"),
        )
        .drop_nulls("log_ret")
        .collect()                      # eager collect (no streaming flag)
    )
    if bars.height == 0:
        print(f"[{asset}]   ⚠  resample produced 0 rows — check data gaps")
        return

    # ── 4  write one Parquet per month  ────────────────────────────────────
    out_root = DST_ROOT / bar / asset
    out_root.mkdir(parents=True, exist_ok=True)

    for key, group in bars.partition_by("ym", as_dict=True).items():
        ym_str = key[0] if isinstance(key, tuple) else key    # unwrap tuple → "2007-06"
        file_path = out_root / f"{ym_str}.parquet"
        group.drop("ym").write_parquet(file_path, compression="snappy")

    print(f"[{asset}]   ✓  wrote {bars.height:,} bars")

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args    = parse_cli()
    tickers = args.tickers or all_assets()

    # pass two iterables: tickers  and an endless repeat of the bar-width
    with ProcessPoolExecutor() as ex:
        for _ in ex.map(resample_asset, tickers, repeat(args.bar)):
            pass                     # forces iteration so exceptions surface