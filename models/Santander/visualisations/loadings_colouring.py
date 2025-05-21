#!/usr/bin/env python3
"""
backtest.py  ─ Visualise factor loadings on a London basemap
===========================================================
This script now **extracts and plots** the sensor‐level loading vector for a
chosen back‑test day *τ* and factor *j*.

Inputs
------
1. ``loadings_hat.csv`` – matrix with shape ``(n_backtest_days * N, r)`` (no header).
2. ``config.yaml``        – experiment parameters (must contain ``SEED``,
   ``num_factors``, ``n_backtest_days``).
3. ``sensors.csv``        – coordinates *in the same order* as the sensors in
   the loading matrix.  Must expose columns ``latitude`` and ``longitude``.
4. ``tau``                – 0‑based back‑test‑day index (``0 ≤ τ < n_backtest_days``).
5. ``j``                  – 0‑based factor/column index (``0 ≤ j < r``).

Key options
~~~~~~~~~~~
``--background``   PNG/JPG basemap for London (if omitted, points are plotted on
                   a blank axis with matching x/y limits).
``--out``          Output figure path (default: *loadings_map.png*).
``--size-range``   Two ints «min max» for the dot radius (default 30 150).
``--cmap``         Any Matplotlib diverging cmap (default *RdBu_r*).

Example
~~~~~~~
$ ./loadings_colouring.py loadings_hat.csv config.yaml cleaned_locations.csv 12 3 --background London.png --out factor3_day12.pdf 
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl

###############################################################################
# CLI parsing
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a (N,) loading vector and visualise it on a London map "
                    "with colour + size encoding (diverging cmap)."
    )
    p.add_argument("csv_loadings", type=pathlib.Path, help="CSV matrix of loadings")
    p.add_argument("config",       type=pathlib.Path, help="YAML config file")
    p.add_argument("csv_coords",   type=pathlib.Path, help="CSV with sensor lat/lon")
    p.add_argument("tau",          type=int,          help="0‑based day index τ")
    p.add_argument("j",            type=int,          help="0‑based factor index j")

    p.add_argument("--background", type=pathlib.Path, default=None,
                   help="PNG/JPG background basemap for London (optional)")
    p.add_argument("--out", type=pathlib.Path, default="loadings_map.png",
                   help="Output image (default: loadings_map.png)")
    p.add_argument("--size-range", nargs=2, type=int, default=[30, 150], metavar=("MIN", "MAX"),
                   help="Min and max marker radius (default 30 150)")
    p.add_argument("--cmap", default="RdBu_r",
                   help="Matplotlib diverging colourmap (default RdBu_r)")
    return p.parse_args()

###############################################################################
# Helpers
###############################################################################

def infer_N(n_rows: int, n_days: int) -> int:
    if n_rows % n_days:
        sys.exit("(rows, n_days) are incompatible → rows %% n_days ≠ 0 — cannot infer N.")
    return n_rows // n_days

###############################################################################
# Main
###############################################################################

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")
    cfg = yaml.safe_load(args.config.read_text())
    for key in ("SEED", "num_factors", "n_backtest_days"):
        if key not in cfg:
            sys.exit(f"Config missing '{key}'")
    SEED = int(cfg["SEED"])
    r_cfg = int(cfg["num_factors"])
    n_days_cfg = int(cfg["n_backtest_days"])
    random.seed(SEED)

    # ------------------------------------------------------------------
    # 2. Loadings matrix
    # ------------------------------------------------------------------
    loadings_2d = np.genfromtxt(args.csv_loadings, delimiter=',')
    if loadings_2d.ndim == 1:
        loadings_2d = loadings_2d.reshape(-1, 1)
    n_rows, r_csv = loadings_2d.shape
    if r_csv != r_cfg:
        sys.exit(f"CSV has r={r_csv} ≠ num_factors={r_cfg} (config)")
    N = infer_N(n_rows, n_days_cfg)

    # reshape to (n_days, N, r)
    loadings_3d = loadings_2d.reshape(n_days_cfg, N, r_csv)

    if not (0 <= args.tau < n_days_cfg):
        sys.exit(f"τ out of range 0‒{n_days_cfg-1}")
    if not (0 <= args.j < r_cfg):
        sys.exit(f"j out of range 0‒{r_cfg-1}")

    vector = loadings_3d[args.tau, :, args.j]
    lo, hi = np.percentile(vector, [3, 97])     # tweak percents as you like
    vector = np.clip(vector, lo, hi)

    # ------------------------------------------------------------------
    # 3. Sensor coordinates
    # ------------------------------------------------------------------
    coords_df = pd.read_csv(args.csv_coords)
    if len(coords_df) != N:
        sys.exit(
            f"coords file has {len(coords_df)} rows but N={N} inferred from matrix."
        )
    if not {"latitude", "longitude"}.issubset(coords_df.columns):
        sys.exit("coords CSV must contain 'latitude' and 'longitude' columns")
    coords_df = coords_df.assign(loading=vector)

    # ------------------------------------------------------------------
    # 4. Prepare colour + size
    # ------------------------------------------------------------------
    vmax = np.max(np.abs(vector)) or 1.0
    cmap = mpl.colormaps.get_cmap(args.cmap)
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    colours = cmap(norm(coords_df.loading))

    size_min, size_max = args.size_range
    sizes = size_min + (np.abs(vector) / vmax) * (size_max - size_min)

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 7))

    # Background basemap (optional)
    if args.background is not None and args.background.exists():
        img = plt.imread(args.background)
        # Default extent for London (approx). Adjust if needed.
        extent = [-0.25, 0.01, 51.45, 51.555]
        ax.imshow(img, extent=extent, aspect='auto', zorder=0)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    else:
        # blank axes limits derived from data
        ax.set_xlim(coords_df.longitude.min() - 0.01, coords_df.longitude.max() + 0.01)
        ax.set_ylim(coords_df.latitude.min() - 0.01,  coords_df.latitude.max() + 0.01)

    # Scatter
    ax.scatter(
        coords_df.longitude,
        coords_df.latitude,
        c=colours,
        s=30,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.4,
        zorder=1,
    )

    # Aesthetics
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Colourbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # dummy for colourbar
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    # cbar.set_label('Loading value')

    # Grid & border styling similar to your cluster script
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved map → {args.out} (N={N})")


if __name__ == "__main__":
    main()
