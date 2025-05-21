#!/usr/bin/env python3
"""
make_excess_returns.py

Compute market‐excess log‐returns by subtracting SPY (last column)
from every other asset in a minutely CSV.

Input format:
    Date,Time,AssetA,AssetB,...,SPY
    2007-06-27,09:31:00,-0.0055,   0.0021,...,0.0013
    ...

Output format:
    Date,Time,AssetA_excess,AssetB_excess,...
    2007-06-27,09:31:00,-0.0068,   0.0008,...
    ...
"""

import argparse
import pandas as pd

def compute_excess_returns(infile: str,
                           outfile: str,
                           date_col: str,
                           time_col: str) -> pd.DataFrame:
    # 1. Load
    df = pd.read_csv(infile)

    # 2. Identify market (SPY) column
    market_col = df.columns[-1]

    # 3. Prepare output frame with Date & Time
    out = df[[date_col, time_col]].copy()

    # 4. Compute excess for each asset except the market itself
    return_cols = [c for c in df.columns if c not in (date_col, time_col, market_col)]
    for c in return_cols:
        out[f"{c}"] = df[c] - df[market_col]

    # 5. Save and return
    out.to_csv(outfile, index=False)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Subtract SPY (last column) from each asset to get excess returns"
    )
    p.add_argument('infile',
                   help="CSV file with Date, Time, asset returns…, SPY as last column")
    p.add_argument('-o', '--outfile', default='data_excess.csv',
                   help="Where to write excess‐returns CSV")
    p.add_argument('--date-col', default='Date',
                   help="Name of the Date column")
    p.add_argument('--time-col', default='Time',
                   help="Name of the Time column")
    args = p.parse_args()

    compute_excess_returns(args.infile,
                           args.outfile,
                           args.date_col,
                           args.time_col)
    print(f"✓ Market‐excess returns written to {args.outfile}")


if __name__ == '__main__':
    main()
