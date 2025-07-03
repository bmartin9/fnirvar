#!/usr/bin/env python3
# view_parquet.py  <file.parquet>  [ rows | start end ]
#
# Row-based examples
#   view_parquet.py data.parquet            → first 10 rows
#   view_parquet.py data.parquet 15         → first 15 rows
#   view_parquet.py data.parquet -20        → last  20 rows
#   view_parquet.py data.parquet 12 15      → rows 12-15  (inclusive)
#
# Date/time examples   (assumes timestamp column is named "ts")
#   view_parquet.py data.parquet 2024-01-02
#   view_parquet.py data.parquet 2024-01-02 2024-01-05T10:30
#
from __future__ import annotations
import sys, pathlib, datetime as dt
import numpy as np
import polars as pl

TS_COL      = "ts"    # change if your timestamp column has another name
DEFAULT_N   = 10      # default number of rows when no slice/limit given

# ─────────────────── helpers ──────────────────────────────────────────────
def usage() -> None:
    sys.exit(
        "usage: view_parquet.py <file.parquet> "
        "[ n | -n | start_iso [end_iso] | start_row end_row ]"
    )

def parse_iso(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return None

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False

# ─────────────────── CLI parsing ──────────────────────────────────────────
if len(sys.argv) < 2:
    usage()

path = pathlib.Path(sys.argv[1]).expanduser().resolve()
if not path.exists():
    sys.exit(f"{path} not found")

cli = sys.argv[2:]
mode: tuple

if not cli:                                        # default head 10
    mode = ("head", DEFAULT_N)

elif len(cli) == 1:                                # one argument
    arg = cli[0]
    if is_int(arg):
        n = int(arg)
        mode = ("head", n) if n > 0 else ("tail", abs(n))
    elif (dt0 := parse_iso(arg)):
        mode = ("time_head", dt0)                  # first DEFAULT_N from dt0
    else:
        usage()

elif len(cli) == 2:                                # two arguments
    a, b = cli
    if is_int(a) and is_int(b):                    # row slice
        start, end = int(a), int(b)
        if end < start:
            sys.exit("end index must be ≥ start index")
        mode = ("slice", start, end)
    elif (dt0 := parse_iso(a)) and (dt1 := parse_iso(b)):  # datetime range
        if dt1 < dt0:
            sys.exit("end datetime must be ≥ start datetime")
        mode = ("time_range", dt0, dt1)
    else:
        usage()
else:
    usage()

# ─────────────────── load & slice lazily ──────────────────────────────────
lf = pl.scan_parquet(str(path))

if mode[0] == "head":
    df = lf.limit(mode[1]).collect()

elif mode[0] == "tail":
    df = lf.tail(mode[1]).collect()

elif mode[0] == "slice":
    s, e = mode[1], mode[2]
    df = lf.slice(s, e - s + 1).collect()

elif mode[0] == "time_head":
    dt0 = mode[1]
    df = (lf.filter(pl.col(TS_COL) >= dt0)
            .limit(DEFAULT_N)
            .collect())

else:                                              # "time_range"
    dt0, dt1 = mode[1], mode[2]
    df = (lf.filter(
            (pl.col(TS_COL) >= dt0) &
            (pl.col(TS_COL) <= dt1))
            .collect())

print(df)

# ─────────────────── max-value report ─────────────────────────────────────
if df.height == 0:
    sys.exit("\n(empty frame)")

num_cols = [c for c in df.columns if c != TS_COL and df[c].dtype.is_numeric()]

if not num_cols:
    sys.exit("\n(no numeric columns to compute a maximum on)")

global_max  = -np.inf
max_col     = None
max_row_idx = None

for col in num_cols:
    col_vals = df[col].to_numpy()
    if col_vals.size == 0 or np.isnan(col_vals).all():
        continue
    idx = np.nanargmax(col_vals)
    val = col_vals[idx]
    if val > global_max:
        global_max  = val
        max_col     = col
        max_row_idx = int(idx)

ts_val   = df[TS_COL][max_row_idx] if TS_COL in df.columns else "—"
col_idx  = df.columns.index(max_col)

print(
    f"\nMAX  {global_max:.6g}  |  ts: {ts_val}  |  ticker/col: {max_col}  "
    f"|  row#: {max_row_idx}  |  col#: {col_idx}"
)
