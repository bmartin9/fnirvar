#!/usr/bin/env python3
"""
Visualise the scaling of selected eigenvalues with N for NIRVAR / FNIRVAR.

Usage
-----
    python plot_eigenvalues.py <means_csv> <stds_csv> <hyperparams.yaml> <k1> <k2>

    * <means_csv>        CSV with shape (len(N_list), num_eigenvalues) – row i: means for N_list[i]
    * <stds_csv>         Same shape – per–entry standard deviations
    * <hyperparams.yaml> Must contain the key `N_list`
    * <k1>, <k2>         1‑based ranks of the eigenvalues to plot (e.g. 1 5 for the largest
                         and 5‑th largest)

Requires `plotly` with the Kaleido engine installed for PDF export.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml
import time   # (Plotly + kaleido sometimes needs a short pause on some systems)

# ──────────────────────────────────────────────────────────────────────────────
def parse_cli() -> tuple[Path, Path, Path, int, int]:
    if len(sys.argv) != 6:
        sys.exit(
            f"Usage: {sys.argv[0]} <means_csv> <stds_csv> <hyperparams.yaml> <k1> <k2>"
        )

    means_csv = Path(sys.argv[1]).expanduser().resolve()
    stds_csv  = Path(sys.argv[2]).expanduser().resolve()
    yaml_file = Path(sys.argv[3]).expanduser().resolve()
    try:
        k1 = int(sys.argv[4])
        k2 = int(sys.argv[5])
    except ValueError:
        sys.exit("k1 and k2 must be integers (1‑based indices of eigenvalues)")

    if k1 < 1 or k2 < 1:
        sys.exit("Eigenvalue indices must be positive (1‑based).")

    return means_csv, stds_csv, yaml_file, k1, k2


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    means_csv, stds_csv, yaml_file, k1, k2 = parse_cli()

    # ---------- load data ----------------------------------------------------
    with yaml_file.open("r") as fh:
        config = yaml.safe_load(fh)
    N_list = list(config["N_list"])

    df_means = pd.read_csv(means_csv, header=None)
    df_stds  = pd.read_csv(stds_csv,  header=None)

    num_N, num_evals = df_means.shape
    for k in (k1, k2):
        if k > num_evals:
            sys.exit(f"Requested eigenvalue index {k} exceeds num_eigenvalues = {num_evals}")

    # ---------- build traces -------------------------------------------------
    traces = []
    colors = ["#1f77b4", "#d62728"]  # two distinct default Plotly colours
    for idx, k in enumerate((k1, k2), start=0):
        col = k - 1  # 0‑based column in the CSV
        y_means = df_means.iloc[:, col]
        y_stds  = df_stds.iloc[:,  col]

        if k == 1:
            # legend_label = rf"$\lambda_{{\text{{max}}}}(\hat{{\Gamma}}_{{\xi}})$"
            legend_label = rf"$\Large \lambda_{{\text{{min}}}}(\hat{{\Gamma}}_{{\chi}})$"

        else:
            # legend_label = rf"$\lambda_{{\text{{min}}}}(\hat{{\Gamma}}_{{\chi}})$"
            legend_label = rf"$\Large \lambda_{{\text{{max}}}}(\hat{{\Gamma}}_{{\xi}})$"

        traces.append(
            go.Scatter(
                x=N_list,
                y=y_means,
                mode="lines+markers",
                name=legend_label,   # legend label
                error_y=dict(type="data", array=y_stds, visible=True),
                line=dict(width=2, color=colors[idx]),
                marker=dict(size=6, color=colors[idx]),
            )
        )

    # ---------- layout -------------------------------------------------------
    layout = go.Layout(
        showlegend=True,
        xaxis=dict(
            title="N",
            showline=True, linewidth=1, linecolor="black",
            ticks="outside", mirror=True,
            range=[min(N_list) - 5, max(N_list) + 5],
        ),
        yaxis=dict(
            title="Eigenvalue",
            showline=True, linewidth=1, linecolor="black",
            ticks="outside", mirror=True,
        ),
        legend=dict(
        x=0.02,          # >1 ⇒ to the right of the plotting area
        y=0.8,           # centred vertically (0 = bottom, 1 = top)
        xanchor="left",  # anchor legend box’s left edge at x position
        yanchor="middle",
        borderwidth=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=30, color="black"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_family="Serif",
        font_size=30,
        margin=dict(l=5, r=5, t=5, b=5),
        width=600,
        height=400,
    )

    fig = go.Figure(data=traces, layout=layout)
    

    # ---------- save ---------------------------------------------------------
    out_name = f"eigenvalues_{k1}_{k2}.pdf"
    fig.write_image(out_name)
    # Occasionally Kaleido needs a short wait when writing multiple files in succession
    time.sleep(1)
    fig.write_image(out_name)

    print(f"Saved: {out_name}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
