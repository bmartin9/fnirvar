#!/usr/bin/env python
# concat_realised.py  <data_bars/30m>  <out_file.parquet> [--excess]
#
# Concatenate all monthly parquet files under data_bars/30m/*/
# into a single wide panel:
#   ts, AAPL, MSFT, …   (or excess returns if --excess is given)
#
from __future__ import annotations
import sys, argparse, pathlib
import polars as pl
from functools import reduce

# ─── CLI ──────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("root", help="root folder, e.g. data_bars/30m/")
ap.add_argument("outfile", help="output parquet file")
ap.add_argument("--excess", action="store_true",
                help="store market-excess returns (each ticker − SPY)")
args = ap.parse_args()

root = pathlib.Path(args.root).expanduser().resolve()
out  = pathlib.Path(args.outfile).expanduser().resolve()

# ─── build a LazyFrame per ticker ────────────────────────────────────────
lazy_frames: list[pl.LazyFrame] = []
for tdir in sorted(root.iterdir()):
    if not tdir.is_dir():
        continue
    tkr = tdir.name
    lf = (
        pl.scan_parquet(str(tdir / "*.parquet"))
          .select("ts", pl.col("log_ret").alias(tkr))
          .with_columns(pl.col("ts").dt.replace_time_zone(None))
    )
    lazy_frames.append(lf)

if not lazy_frames:
    sys.exit(f"no parquet files found under {root}")

# ─── full outer join on “ts” across all tickers ──────────────────────────
def join_two(a: pl.LazyFrame, b: pl.LazyFrame) -> pl.LazyFrame:
    joined = a.join(b, on="ts", how="full", suffix="_r")
    # merge dual timestamp columns created by the full join
    return (joined
            .with_columns(pl.coalesce([pl.col("ts"), pl.col("ts_r")]).alias("ts"))
            .drop("ts_r"))

wide = reduce(join_two, lazy_frames).collect().sort("ts")

# ─── convert to excess returns if requested ──────────────────────────────
if args.excess:
    if "SPY" not in wide.columns:
        sys.exit("cannot compute excess returns – column 'SPY' not found")
    spy_col = pl.col("SPY")
    adjust_cols = [c for c in wide.columns if c not in ("ts", "SPY")]
    wide = wide.with_columns([(pl.col(c) - spy_col).alias(c) for c in adjust_cols])

# ─── write out ───────────────────────────────────────────────────────────
out.parent.mkdir(parents=True, exist_ok=True)
wide.write_parquet(out, compression="snappy")
print(f"wrote {out}  ({wide.height:,} bars, {wide.width-1} tickers)")
