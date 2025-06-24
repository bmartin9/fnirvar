#!/usr/bin/env python
# plot_cum_pnl.py   pnl_bar.parquet [pnl_bar2.parquet ...]
#
# Draw a Plotly line-chart of **cumulative net PnL (in bp)** for one or
# several `pnl_bar.parquet` files produced by back-test statistics.
#
# ---------------------------------------------------------------------------
from __future__ import annotations
import sys, pathlib, datetime as dt
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
import time

# -------- legends (edit to taste) ------------------------------------------
LEGENDS = [
    "Strategy-1",
    "Strategy-2",
    "Strategy-3",
]
# If you pass more files than names above, default to the file-stem.
# ---------------------------------------------------------------------------


def load_cum_pnl(path: pathlib.Path) -> tuple[list[dt.date], list[float]]:
    """Return trading-day vector and cumulative net PnL in basis-points."""
    df = pl.read_parquet(path)
    # convert to daily PnL then cumulative
    ser = (
        df.with_columns(pl.col("ts").dt.date().alias("day"))
          .group_by("day")
          .agg(pl.col("net").sum().alias("day_net"))
          .sort("day")
    )
    cum = (ser["day_net"].cum_sum() ).to_list()        # → bps
    days = ser["day"].to_list()
    return days, cum


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: plot_cum_pnl.py  pnl_bar.parquet  [more.parquet …]")

    paths = [pathlib.Path(p).expanduser().resolve() for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            sys.exit(f"{p} not found")

    fig = go.Figure()

    for i, path in enumerate(paths):
        days, cum = load_cum_pnl(path)
        name = LEGENDS[i] if i < len(LEGENDS) else path.stem
        fig.add_trace(go.Scatter(x=days, y=cum, mode="lines", name=name))

    # build yearly tick locations (first trading day each year)
    all_days = sorted({d for path in paths for d in load_cum_pnl(path)[0]})
    year_ticks = [d for i, d in enumerate(all_days) if i == 0 or d.year != all_days[i-1].year]

    layout = go.Layout(
        yaxis=dict(title="Cumulative PnL (bpts)",
                   showline=True, linewidth=1, linecolor="black",
                   ticks="outside", mirror=True),
        xaxis=dict(title="Day",
                   showline=True, linewidth=1, linecolor="black",
                   ticks="outside", mirror=True,
                   automargin=True,
                   tickmode="array", tickvals=year_ticks,
                   tickformat="%Y",
                   range=[all_days[0], all_days[-1]]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_family="Serif",
        font_size=11,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    fig.update_layout(layout)
    fig.show()

    pio.write_image(fig, 'CumPnL.pdf')
    time.sleep(1)
    pio.write_image(fig, 'CumPnL.pdf')



if __name__ == "__main__":
    main()
