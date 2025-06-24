#!/usr/bin/env python
import sys, pathlib, polars as pl, numpy as np

if len(sys.argv) != 3:
    sys.exit("usage: concat_predictions.py <parent_dir> <out_file.parquet>")

parent  = pathlib.Path(sys.argv[1]).expanduser().resolve()
outfile = pathlib.Path(sys.argv[2]).expanduser().resolve()

print("Searching under", parent)
pred_files = list(parent.rglob("predictions.parquet"))
print("Found", len(pred_files), "files")
for p in pred_files[:10]:
    print("  ", p)                     # show first few

if not pred_files:
    sys.exit("No predictions.parquet found – check file names / depth.")

# ---- build superset schema -----------------------------------------------
lazy_frames = [pl.scan_parquet(p) for p in pred_files]
all_cols    = set().union(*(lf.collect_schema().names() for lf in lazy_frames))
all_cols    = ["ts", *sorted(c for c in all_cols if c != "ts")]
flt_dtype   = pl.Float32
ts_dtype    = pl.Datetime

# ---- pad and cast ---------------------------------------------------------
padded = []
for lf in lazy_frames:
    # cast existing cols
    exprs = []
    for c in lf.columns:
        exprs.append(pl.col(c).cast(ts_dtype if c == "ts" else flt_dtype))
    lf = lf.select(exprs)
    # add missing
    for c in all_cols:
        if c not in lf.columns:
            lf = lf.with_columns(pl.lit(None, dtype=flt_dtype).alias(c))
    lf = lf.select(all_cols)
    padded.append(lf)

merged = pl.concat(padded).collect().sort("ts")
outfile.parent.mkdir(parents=True, exist_ok=True)
merged.write_parquet(outfile, compression="snappy")
print(f"Written {outfile} with {merged.height:,} rows and {merged.width-1} tickers")
