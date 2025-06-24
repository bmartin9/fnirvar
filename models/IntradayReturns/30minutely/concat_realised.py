#!/usr/bin/env python
# concat_realised.py  <data_bars/30m>  <out_file.parquet>
#
# Creates a single realised_all.parquet with columns: ts, AAPL, MSFT, …
#
from __future__ import annotations
import sys, pathlib, polars as pl
from functools import reduce

if len(sys.argv) != 3:
    sys.exit("usage: concat_realised.py <data_bars/30m> <out_file.parquet>")

root = pathlib.Path(sys.argv[1]).expanduser().resolve()
out  = pathlib.Path(sys.argv[2]).expanduser().resolve()

lazy_frames = []
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

def join_two(a: pl.LazyFrame, b: pl.LazyFrame) -> pl.LazyFrame:
    joined = a.join(b, on="ts", how="full", suffix="_r")     # ①
    # merge the two timestamp columns: take left if present else right
    joined = joined.with_columns(
        pl.coalesce([pl.col("ts"), pl.col("ts_r")]).alias("ts")
    ).drop("ts_r")                                           # ②
    return joined


wide = reduce(join_two, lazy_frames).collect().sort("ts")

out.parent.mkdir(parents=True, exist_ok=True)
wide.write_parquet(out, compression="snappy")
print(f"wrote {out}  ({wide.height:,} bars, {wide.width-1} tickers)")
