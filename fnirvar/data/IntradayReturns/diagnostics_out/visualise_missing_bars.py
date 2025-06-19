# visualize_missing_bars.py – heat‑map of missing 30‑minute bars
# ------------------------------------------------------------------
# Usage (from project root):
#   python visualize_missing_bars.py diagnostics_out/missing_bars.parquet
# Opens an interactive Plotly window (or saves HTML when --out-html).

import argparse, pathlib, polars as pl, plotly.graph_objects as go

# --------------------------- CLI -------------------------------------------
parser = argparse.ArgumentParser(description="Plot heat‑map of missing bars")
parser.add_argument("parquet", help="path to missing_bars.parquet")
parser.add_argument("--out-html", help="optional path to save standalone html")
args = parser.parse_args()

pfile = pathlib.Path(args.parquet)
if not pfile.exists():
    raise FileNotFoundError(pfile)

# --------------------------- load & pivot ----------------------------------
# DataFrame columns: month (YYYY-MM), ticker, missing_bars
mf = pl.read_parquet(pfile)

# pivot to matrix; absent entries → 0
heat_df = mf.pivot(index="month", columns="ticker", values="missing_bars", aggregate_function="first") \
            .fill_null(0) \
            .sort("month")

months  = heat_df["month"].to_list()
assets  = heat_df.columns[1:]              # first column is month
matrix  = heat_df.select(assets).to_numpy().astype(int)

# x‑axis range for layout (min & max inclusive)
x_min = 0
x_max = len(assets)-1

# --------------------------- plotly heat‑map -------------------------------
fig = go.Figure(data=go.Heatmap(z=matrix, x=assets, y=months,
                               colorscale="Reds", zmin=0, zmax=matrix.max()))

layout = go.Layout(
    yaxis=dict(title='', showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
    xaxis=dict(title='', showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True,
               automargin=True, range=[x_min, x_max]),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font_family="Serif",
    font_size=14,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)
fig.update_layout(layout)

if args.out_html:
    fig.write_html(args.out_html)
    print("saved", args.out_html)
else:
    fig.show()
