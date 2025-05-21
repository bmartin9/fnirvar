#!/usr/bin/env python3
"""
plot_eigenvalues.py

Usage:
    python plot_eigenvalues.py data.csv \
         --start 100 --end 500 \
         --eig-start 0 --eig-end 4

This will:
  • load all numeric return columns from `data.csv`
  • take rows 100 (inclusive) through 500 (exclusive) → T = 400
  • compute the N×N covariance matrix across those T points
  • find its eigenvalues, sort them descending,
    and plot indices 0–4 (the 5 largest) as a bar-chart.

Arguments
---------
infile       : path to CSV of returns (no date/time columns required)
--start      : integer, first row index to include (inclusive)
--end        : integer, last row index to include (exclusive)
--eig-start  : integer, zero-based index of first eigenvalue to plot
--eig-end    : integer, zero-based index of last  eigenvalue to plot

Requires: pandas, numpy, plotly
"""

import argparse

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time


def parse_args():
    p = argparse.ArgumentParser(description="Plot a selected range of eigenvalues of the returns covariance matrix")
    p.add_argument('infile', help="CSV file containing only numeric return columns")
    p.add_argument('--start',    type=int, required=True, help="Row index to start (inclusive)")
    p.add_argument('--end',      type=int, required=True, help="Row index to end   (exclusive)")
    p.add_argument('--eig-start',type=int, required=True, help="Zero-based index of first eigenvalue to plot")
    p.add_argument('--eig-end',  type=int, required=True, help="Zero-based index of last  eigenvalue to plot")
    return p.parse_args()


def load_numeric_returns(path):
    """Read CSV and return only its numeric columns as a DataFrame."""
    df = pd.read_csv(path)
    return df.select_dtypes(include=[np.number])


def compute_eigenvalues(returns: pd.DataFrame, i0: int, i1: int) -> np.ndarray:
    """
    Slice rows [i0:i1), compute covariance of columns,
    return descending-sorted eigenvalues.
    """
    window = returns.iloc[i0:i1]
    cov = window.cov()
    eigvals = np.linalg.eigvalsh(cov)         # symmetric → use eigvalsh
    return np.sort(eigvals)[::-1]             # largest first


def make_layout():
    return go.Layout(
        yaxis=dict(showline=True, linewidth=1, linecolor='black',
                   ticks='outside', mirror=True),
        xaxis=dict(showline=True, linewidth=1, linecolor='black',
                   ticks='outside', mirror=True, automargin=True),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_family="Serif",
        font_size=11,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=350
    )


def plot_eigenvalues(eigvals: np.ndarray, start: int, end: int):
    """
    Plot eigvals[start:end+1] as a bar chart,
    with x=their indices and y=their magnitudes.
    """
    if start < 0 or end >= len(eigvals) or start > end:
        raise IndexError(f"Invalid eigenvalue range: 0–{len(eigvals)-1}, you asked {start}–{end}")
    y = eigvals[start:end+1]
    x = list(range(start, end+1))
    trace = go.Bar(x=x, y=y)
    fig = go.Figure(data=[trace], layout=make_layout())
    fig.update_layout(xaxis_title="Eigenvalue index",
                      yaxis_title="Eigenvalue magnitude")
    fig.write_image("eigenvalues.pdf", format="pdf", width=500, height=350)
    time.sleep(1)
    fig.write_image("eigenvalues.pdf", format="pdf", width=500, height=350)


def main():
    args = parse_args()
    returns = load_numeric_returns(args.infile)
    eigvals = compute_eigenvalues(returns, args.start, args.end)
    plot_eigenvalues(eigvals, args.eig_start, args.eig_end)


if __name__ == '__main__':
    main()
