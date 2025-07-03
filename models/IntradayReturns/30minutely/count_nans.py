#!/usr/bin/env python3
"""
count_nans.py  ── Count total and percentage of NaN values in a Parquet file.

Usage
-----
    python count_nans.py <path/to/file.parquet>

Notes
-----
• If the file fits comfortably in RAM, the default (pandas) path is simplest.
• For very large files, switch on the --stream flag to use a memory-friendly
  PyArrow streaming loop.
"""

import sys
import argparse
import pandas as pd

def pandas_count(path: str) -> None:
    df = pd.read_parquet(path)
    nan_total = df.isna().values.sum()
    cell_total = df.size                      # rows * columns
    pct = nan_total * 100 / cell_total
    print(f"Total NaNs   : {nan_total:,}")
    print(f"Percentage   : {pct:,.4f} %  (of {cell_total:,} cells)")

def arrow_stream_count(path: str, batch_size: int = 1_000_000) -> None:
    import pyarrow.parquet as pq

    pf          = pq.ParquetFile(path)
    nan_total   = 0
    cell_total  = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        df_chunk     = batch.to_pandas()            # quick conversion
        nan_total   += df_chunk.isna().values.sum()
        cell_total  += df_chunk.size

    pct = nan_total * 100 / cell_total
    print(f"Total NaNs   : {nan_total:,}")
    print(f"Percentage   : {pct:,.4f} %  (of {cell_total:,} cells)")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_file", help="Path to .parquet file")
    parser.add_argument(
        "--stream", action="store_true",
        help="Use PyArrow streaming for large files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1_000_000,
        help="Rows per batch when --stream is on (default: 1 000 000)"
    )
    args = parser.parse_args()

    try:
        if args.stream:
            arrow_stream_count(args.parquet_file, args.batch_size)
        else:
            pandas_count(args.parquet_file)
    except Exception as exc:
        sys.exit(f"Error: {exc}")

if __name__ == "__main__":
    main()

