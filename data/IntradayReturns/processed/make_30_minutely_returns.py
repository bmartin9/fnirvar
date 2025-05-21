#!/usr/bin/env python3
"""
USAGE: python make_30_minutely_returns.py ../raw/data_1min.csv

Convert 1-minute log-returns in data_1min.csv to 30-minute log-returns,
but only during trading hours (09:30–16:00), never mixing days,
and output with separate Date & Time columns.

Input format:
    Date,Time,r1,r2,...
    2007-06-27,09:31:00,-0.0055,...

Output format:
    Date,Time,r1,r2,...
    2007-06-27,10:00:00,(sum 09:31–10:00),...
    ...
"""

import argparse
import pandas as pd

def log_30min_intraday(csv_in: str,
                       csv_out: str = 'data_30min.csv',
                       date_col: str = 'Date',
                       time_col: str = 'Time') -> pd.DataFrame:
    # 1. Load
    df = pd.read_csv(csv_in)

    # 2. Build a single datetime index
    df['timestamp'] = pd.to_datetime(
        df[date_col] + ' ' + df[time_col],
        format='%Y-%m-%d %H:%M:%S'
    )
    df = (
        df
        .drop(columns=[date_col, time_col])
        .set_index('timestamp')
        .sort_index()
    )

    # 3. Restrict to trading hours
    intraday = df.between_time('09:30', '16:00')

    # 4. Resample each day separately into 30-min bins
    def day_resample(day):
        return day.resample('30min', closed='right', label='right').sum()

    r30 = (
        intraday
        .groupby(intraday.index.date)
        .apply(day_resample)
    )

    # 5. Keep only the timestamp level of the MultiIndex
    r30.index = r30.index.get_level_values(1)
    r30.index.name = 'timestamp'

    # 6. Drop bins with all NaNs
    r30 = r30.dropna(how='all')

    # 7. Split timestamp back to Date & Time
    out = r30.reset_index()
    out['Date'] = out['timestamp'].dt.strftime('%Y-%m-%d')
    out['Time'] = out['timestamp'].dt.strftime('%H:%M:%S')

    # 8. Reorder and save
    cols = ['Date', 'Time'] + [c for c in out.columns if c not in ('timestamp', 'Date', 'Time')]
    out = out[cols]
    out.to_csv(csv_out, index=False)
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='1-min ➜ 30-min intraday log-returns with Date & Time cols'
    )
    p.add_argument('infile', help='CSV with Date, Time, and return columns')
    p.add_argument('-o', '--outfile', default='data_30min.csv',
                   help='Where to save the 30-min CSV')
    p.add_argument('--date-col', default='Date',
                   help='Name of the Date column in input/output')
    p.add_argument('--time-col', default='Time',
                   help='Name of the Time column in input/output')
    args = p.parse_args()

    log_30min_intraday(
        csv_in=args.infile,
        csv_out=args.outfile,
        date_col=args.date_col,
        time_col=args.time_col
    )
    print(f'✓ Intraday 30-minute log-returns written to {args.outfile}')
