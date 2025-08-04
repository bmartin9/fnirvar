#!/usr/bin/env python3
# plot_cum_pnl.py  pnl_bar.parquet [pnl_bar2.parquet ...]
#
# Draw a Plotly line-chart of cumulative net PnL (in bp).  Optional
# --start / --end flags let you focus on a sub-interval.
#
# ---------------------------------------------------------------------
from __future__ import annotations
import sys, pathlib, datetime as dt, argparse, time
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio

# ---------- legends (edit to taste) ----------------------------------
LEGENDS = [
    "FNIRVAR",
    "Factors + LASSO",
    "Factors Only",
]
# If you pass more files than names above, the file-stem is used.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot cumulative PnL in basis-points")
    ap.add_argument("files", nargs="+",
                    help="One or more pnl_bar.parquet files")
    ap.add_argument("--start", type=str, metavar="YYYY-MM-DD",
                    help="Start date (inclusive)")
    ap.add_argument("--end",   type=str, metavar="YYYY-MM-DD",
                    help="End date   (inclusive)")
    return ap.parse_args()


def to_date(s: str | None) -> dt.date | None:
    """Parse ISO date (YYYY-MM-DD) → datetime.date, or None if s is None."""
    return dt.date.fromisoformat(s) if s else None


# ---------------------------------------------------------------------
def load_cum_pnl(path: pathlib.Path,
                 start: dt.date | None,
                 end:   dt.date | None
                 ) -> tuple[list[dt.date], list[float]]:
    """
    Read bar-level PnL, down-sample to daily, filter [start, end],
    then return days[] and cumulative net PnL in bp.
    """
    df = pl.read_parquet(path)

    ser = (df.with_columns(pl.col("ts").dt.date().alias("day"))
             .group_by("day")
             .agg(pl.col("net").sum().alias("day_net"))
             .sort("day"))

    if start:
        ser = ser.filter(pl.col("day") >= start)
    if end:
        ser = ser.filter(pl.col("day") <= end)

    # Polars cum_sum starts at element 0, so cumulative PnL is re-based
    # to zero at the chosen start date.
    cum  = ser["day_net"].cum_sum().to_list()
    days = ser["day"].to_list()
    return days, cum


# ---------------------------------------------------------------------
def main() -> None:
    args  = parse_cli()
    start = to_date(args.start)
    end   = to_date(args.end)

    paths = [pathlib.Path(p).expanduser().resolve() for p in args.files]
    for p in paths:
        if not p.exists():
            sys.exit(f"{p} not found")

    fig = go.Figure()

    # keep track of the global day range for the x-axis
    all_days: set[dt.date] = set()

    for i, path in enumerate(paths):
        days, cum = load_cum_pnl(path, start, end)
        if not days:
            print(f"⚠︎  {path.name}: no data within requested window")
            continue

        all_days.update(days)
        name = LEGENDS[i] if i < len(LEGENDS) else path.stem
        fig.add_trace(go.Scatter(x=days, y=cum, mode="lines", name=name))

    if not all_days:
        sys.exit("No data to plot after applying date filters.")

    # ----- x-axis ticks: first trading day of each year -----------------
    all_days_sorted = sorted(all_days)
    year_ticks = [d for i, d in enumerate(all_days_sorted)
                  if i == 0 or d.year != all_days_sorted[i-1].year]
    
    # skip the first label so “2007” doesn’t sit on top of “2008”
    year_texts = [''] + [str(d.year) for d in year_ticks[1:]]

    layout = go.Layout(
        yaxis=dict(title="Cumulative PnL (bpts)", showline=True,
                   linewidth=1, linecolor="black", ticks="outside",
                   mirror=True),
        xaxis=dict(title="Day", showline=True, linewidth=1, linecolor="black",
                   ticks="outside", mirror=True, automargin=True,
                   tickmode="array",ticktext=year_texts, tickvals=year_ticks,
                   tickformat="%Y",
                   range=[all_days_sorted[0], all_days_sorted[-1]]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_family="Serif",
        font_size=11,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=250,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0),
    )

    fig.update_layout(layout)
    fig.show()

    # save to PDF twice (work-around for Plotly .pdf race condition)
    pio.write_image(fig, "CumPnL.pdf")
    time.sleep(1)
    pio.write_image(fig, "CumPnL.pdf")


if __name__ == "__main__":
    main()
