#!/usr/bin/env python
"""
plot_ticker_ts.py  <parquet_file>  <TICKER>
                   [--start YYYY-MM-DD]  [--end YYYY-MM-DD]

Examples
--------
# full history of AAPL
python plot_ticker_ts.py realised_all.parquet AAPL

# MSFT from 2015-01-01 to 2017-12-31
python plot_ticker_ts.py realised_all.parquet MSFT --start 2015-01-01 --end 2017-12-31
"""
from __future__ import annotations
import argparse, pathlib, datetime as dt

import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
import time

# ────────────────────── CLI ────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("file",   help="Parquet file with columns: ts, AAPL, MSFT, …")
ap.add_argument("ticker", help="Ticker column to plot")
ap.add_argument("--start", type=str, help="start date (YYYY-MM-DD)", default=None)
ap.add_argument("--end",   type=str, help="end   date (YYYY-MM-DD)", default=None)
args = ap.parse_args()

path = pathlib.Path(args.file).expanduser().resolve()
if not path.exists():
    raise SystemExit(f"❌ {path} not found")

# ────────────────────── load & slice ───────────────────────────────────────
cols = ["ts", args.ticker]
df   = pl.read_parquet(path, columns=cols)

if args.start:
    start_dt = dt.datetime.fromisoformat(args.start)
    df = df.filter(pl.col("ts") >= start_dt)

if args.end:
    end_dt = dt.datetime.fromisoformat(args.end)
    df = df.filter(pl.col("ts") <= end_dt)

if df.height == 0:
    raise SystemExit("❌ no data in the requested range")

# ────────────────────── plotly figure ──────────────────────────────────────
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["ts"].to_numpy(),
        y=df[args.ticker].to_numpy(),
        mode="lines",
        name=args.ticker,
    )
)

layout = go.Layout(
    yaxis=dict(
        title=f"{args.ticker} value",
        showline=True,
        linewidth=1,
        linecolor="black",
        ticks="outside",
        mirror=True,
    ),
    xaxis=dict(
        title="Date",
        showline=True,
        linewidth=1,
        linecolor="black",
        ticks="outside",
        mirror=True,
        automargin=True,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_family="Serif",
    font_size=11,
    margin=dict(l=5, r=5, t=5, b=5),
    width=700,
    height=350,
)

fig.update_layout(layout)

pio.write_image(fig, 'timeseries_plot.pdf')
time.sleep(1)
pio.write_image(fig, 'timeseries_plot.pdf')