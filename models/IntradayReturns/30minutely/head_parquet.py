#!/usr/bin/env python
# view_parquet.py  <file.parquet>  [rows | start end]
#
# Examples
#   view_parquet.py data.parquet            # first 10 rows   (default)
#   view_parquet.py data.parquet 15         # first 15 rows
#   view_parquet.py data.parquet -20        # last 20 rows
#   view_parquet.py data.parquet  12  15    # rows 12-15 (0-based, inclusive)
#
from __future__ import annotations
import sys, pathlib, polars as pl

def usage():
    sys.exit("usage: view_parquet.py <file.parquet> [n | start end]")

# ─── parse CLI ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    usage()

path = pathlib.Path(sys.argv[1]).expanduser().resolve()
if not path.exists():
    sys.exit(f"{path} not found")

args = sys.argv[2:]

# default: first 10 rows
if not args:
    mode = ("head", 10)
elif len(args) == 1:                       # single number
    n = int(args[0])
    mode = ("head", n) if n > 0 else ("tail", abs(n))
elif len(args) == 2:                       # start end
    start, end = map(int, args)
    if end < start:
        sys.exit("end index must be ≥ start index")
    mode = ("slice", start, end)
else:
    usage()

# ─── load lazily and slice --------------------------------------------------
lf = pl.scan_parquet(str(path))

if mode[0] == "head":        # first n rows
    df = lf.limit(mode[1]).collect()
elif mode[0] == "tail":      # last n rows
    df = lf.tail(mode[1]).collect()
else:                        # explicit slice
    start, end = mode[1], mode[2]
    df = lf.slice(start, end - start + 1).collect()

print(df)
