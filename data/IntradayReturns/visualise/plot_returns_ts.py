#!/usr/bin/env python3
"""
plot_returns.py

Usage:
    python plot_returns.py data.csv --cols 5 8 --interval 5 --date-col Date --time-col Time
    python plot_returns.py data.csv --cols 5 8 --interval 5 --date-col Date --time-col Time --t0 0 --T 500 --verbose
"""

import argparse
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot selected return series from a minutely CSV"
    )
    p.add_argument('infile',
                   help="Path to CSV (header row must have asset names)")
    p.add_argument('--cols', nargs='+', type=int, required=True,
                   help="List of column indices (0-based) to plot")
    p.add_argument('--interval', type=int, required=True,
                   help="Interval in minutes (e.g. 1, 5, 30) for y-axis label")
    p.add_argument('--date-col', default=None,
                   help="Name of the Date column (if separate)")
    p.add_argument('--time-col', default=None,
                   help="Name of the Time column (if separate)")
    p.add_argument('--t0', type=int, default=None,
                   help="Start index of the time series (inclusive)")
    p.add_argument('--T',  type=int, default=None,
                   help="End index of the time series (exclusive)")
    p.add_argument('--verbose', action='store_true',
                   help="Print the shape of the sliced DataFrame")
    return p.parse_args()


def load_df(path, date_col=None, time_col=None):
    """
    Always read CSV and, if date_col & time_col provided,
    combine them into a DatetimeIndex. Otherwise assume the
    first column is a pre-existing timestamp.
    """
    df = pd.read_csv(path, header=0)
    if date_col and time_col:
        if date_col not in df.columns or time_col not in df.columns:
            raise ValueError(f"Cannot find columns '{date_col}' and '{time_col}' in {path}")
        # combine into datetime
        df['__dt'] = pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[time_col].astype(str),
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.set_index('__dt').sort_index()
    else:
        # assume first column is timestamp
        ts = df.columns[0]
        df[ts] = pd.to_datetime(df[ts])
        df = df.set_index(ts).sort_index()
    return df


def make_layout(interval, holiday_breaks):
    return go.Layout(
        yaxis=dict(
            title=f"{interval} minutely returns",
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True
        ),
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True,
            automargin=True,
            tickmode='auto',
            nticks=10,
            rangebreaks=[ dict(bounds=[16, 9.5], pattern="hour"), dict(bounds=["sat", "mon"]), dict(values=holiday_breaks) ]
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bordercolor='black',
            borderwidth=1
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_family="Serif",
        font_size=11,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=350
    )


def plot_selected_returns(df, cols, interval, t0, T, verbose=False):
    # keep only numeric return columns
    returns = df.select_dtypes(include=[np.number]).copy()

    # compute slice bounds
    start = t0 if t0 is not None else 0
    end   = T  if T  is not None else len(returns)
    if start < 0 or end > len(returns) or start >= end:
        raise ValueError(f"Invalid slice [{start}:{end}] for data length {len(returns)}")

    # slice once
    sl = returns.iloc[start:end]
    sl = sl.between_time('09:30', '16:00')
    if verbose:
        print(f"Sliced rows [{start}:{end}] → shape {sl.shape}")

    trading_days = sl.index.normalize().unique()
    full_calendar = pd.date_range(trading_days.min(), trading_days.max(), freq="D")
    non_trading_days = full_calendar.difference(trading_days)

    # build traces
    traces = []
    for idx in cols:
        if idx < 0 or idx >= sl.shape[1]:
            raise IndexError(f"Column index {idx} out of range (0–{sl.shape[1]-1})")
        name = sl.columns[idx]
        traces.append(go.Scatter(
            x=sl.index,
            y=sl.iloc[:, idx],
            mode='lines',
            name=name
        ))

    # plot and write out
    fig = go.Figure(data=traces, layout=make_layout(interval, holiday_breaks=non_trading_days))
    fig.write_image("returns_plot.pdf", format='pdf')
    time.sleep(1)
    fig.write_image("returns_plot.pdf", format='pdf')


def main():
    args = parse_args()
    df = load_df(args.infile, args.date_col, args.time_col)
    plot_selected_returns(df,
                          cols=args.cols,
                          interval=args.interval,
                          t0=args.t0,
                          T=args.T,
                          verbose=args.verbose)


if __name__ == '__main__':
    main()
