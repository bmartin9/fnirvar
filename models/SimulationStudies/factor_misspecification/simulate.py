#!/usr/bin/env python3
"""
Simulate a static factor model, run a rolling-window PCA backtest, plot targets vs. predictions,
and report MSPE statistics.

Usage:
    python factor_backtest.py --config hyperparameters.yaml

Example `hyperparameters.yaml`:

N: 50
T: 500
r: 4
sigma_eps: 0.5
num_replicas: 50
window: 120
r_hat: 3
seed: 12345
loading_low: 0.7
loading_high: 1.3
factor_forecast_method: ar1   # or 'last'

# Factor control
factor_mode: diag              # 'iid' | 'diag' | 'series'
factor_diag: [10, 4, 0.8, 0.1] # only if factor_mode == 'diag'
# factor_series:               # only if factor_mode == 'series'
#   - [0.1, 0.0,  ...]
#   - [...]
#   # exactly T rows, each of length r

# Rolling-centering toggle
center_window: true            # if false, skip centering

# Plotting options
save_plot: true
replica_to_plot: 0             # which replica to visualize (0-indexed)
series_to_plot: 0              # which series (0..N-1)
plot_pdf_path: pred_vs_target.pdf
"""

from __future__ import annotations

import argparse
import yaml
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import plotly.graph_objects as go

# ------------------------- Configuration -------------------------

@dataclass
class Config:
    N: int
    T: int
    r: int
    sigma_eps: float
    num_replicas: int
    window: int
    r_hat: int

    seed: Optional[int] = None
    loading_low: float = 0.7
    loading_high: float = 1.3
    factor_forecast_method: str = "last"  # 'ar1' or 'last'

    # Factor specification
    factor_mode: str = "iid"              # 'iid' | 'diag' | 'series'
    factor_diag: Optional[List[float]] = None
    factor_series: Optional[List[List[float]]] = None

    # Centering toggle
    center_window: bool = True

    # Plotting
    save_plot: bool = True 
    replica_to_plot: int = 0
    series_to_plot: int = 0
    plot_pdf_path: str = "pred_vs_target.pdf"

# ------------------------- Simulation -------------------------

def simulate_factor_model(cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Y_t = Lambda f_t + e_t with user-controlled factors.
    Returns:
        Y : (T, N)
        F : (T, K)
        L : (N, K)
    """
    # Loadings
    L = rng.uniform(cfg.loading_low, cfg.loading_high, size=(cfg.N, cfg.r))

    # Factors
    if cfg.factor_mode == "series":
        if cfg.factor_series is None:
            raise ValueError("factor_mode='series' but factor_series not provided.")
        F = np.asarray(cfg.factor_series, dtype=float)
        if F.shape != (cfg.T, cfg.r):
            raise ValueError(f"factor_series must be shape ({cfg.T}, {cfg.r}).")
    elif cfg.factor_mode == "diag":
        if cfg.factor_diag is None:
            raise ValueError("factor_mode='diag' but factor_diag not provided.")
        diag = np.asarray(cfg.factor_diag, dtype=float)
        if diag.shape[0] != cfg.r:
            raise ValueError("factor_diag length must equal r.")
        F = rng.normal(size=(cfg.T, cfg.r)) * np.sqrt(diag)
    else:  # iid standard normal
        F = rng.normal(size=(cfg.T, cfg.r))

    # Idiosyncratic errors
    eps = rng.normal(scale=cfg.sigma_eps, size=(cfg.T, cfg.N))

    Y = F @ L.T + eps
    return Y, F, L

# ------------------------- PCA utilities -------------------------

def pca_fit(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """PCA via SVD on time x series matrix X (columns are series). Assumes X is centered.
    Returns:
        F_hat : (T_win, k) factor scores (time dimension first)
        L_hat : (N, k) loadings
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    F_hat = U[:, :k] * S[:k]
    L_hat = Vt[:k, :].T
    return F_hat, L_hat


def ar1_forecast_last_row(F_window: np.ndarray) -> np.ndarray:
    """AR(1) (no intercept) forecast for each factor."""
    if F_window.shape[0] < 2:
        return F_window[-1, :]

    F_t = F_window[1:, :]
    F_tm1 = F_window[:-1, :]
    denom = np.sum(F_tm1 * F_tm1, axis=0)
    phi = np.divide(np.sum(F_t * F_tm1, axis=0), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    return phi * F_window[-1, :]


def last_value_forecast(F_window: np.ndarray) -> np.ndarray:
    return F_window[-1, :]

# ------------------------- Rolling backtest -------------------------

def rolling_pca_forecast(
    Y: np.ndarray,
    r_hat: int,
    window: int,
    method: str,
    center_window: bool = True,
) -> Tuple[float, np.ndarray]:
    """Perform rolling-window PCA and produce MSPE & predictions.
    Args:
        Y: (T, N) data
        r_hat: number of estimated factors
        window: rolling window length
        method: 'ar1' or 'last'
        center_window: whether to center each window
    Returns:
        mspe: float
        Y_hat: (T, N) predictions (NaN for first `window` rows)
    """
    T, N = Y.shape
    Y_hat = np.full_like(Y, np.nan)
    sq_errors = []

    for t in range(window, T):
        X_win = Y[t - window:t, :]
        if center_window:
            mu = X_win.mean(axis=0, keepdims=True)
            Xc = X_win - mu
        else:
            mu = np.zeros((1, N))
            Xc = X_win

        F_win, L_win = pca_fit(Xc, r_hat)

        if method.lower() == "ar1":
            f_fore = ar1_forecast_last_row(F_win)
        else:
            f_fore = last_value_forecast(F_win)

        y_hat = mu.ravel() + L_win @ f_fore  # (N,)
        Y_hat[t, :] = y_hat
        sq_errors.append((Y[t, :] - y_hat) ** 2)

    mspe = float(np.mean(np.vstack(sq_errors)))
    return mspe, Y_hat

# ------------------------- Plotting -------------------------

def build_layout() -> go.Layout:
    """Return the custom layout provided by the user."""
    layout = go.Layout(
        barmode='overlay',
        xaxis=dict(
            title='T',
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True
        ),
        yaxis=dict(
            title='Value',
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_family="Serif",
        font_size=16,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=350
    )
    return layout


def plot_targets_predictions(
    Y: np.ndarray,
    Y_hat: np.ndarray,
    window: int,
    series_idx: int,
    out_pdf: str
) -> None:
    """Create a line plot (targets vs predictions) for a single series and save to PDF."""
    T = Y.shape[0]
    t_axis = np.arange(window, T)

    targets = Y[window:, series_idx]
    preds = Y_hat[window:, series_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_axis, y=targets, mode='lines', name='targets'))
    fig.add_trace(go.Scatter(x=t_axis, y=preds, mode='lines', name='predictions'))

    fig.update_layout(build_layout())

    out_path = Path(out_pdf)
    try:
        fig.write_image(out_path)
        print(f"Saved plot to {out_path.resolve()}")
    except Exception as e:
        print("\n[Warning] Could not save PDF with plotly. Ensure 'kaleido' is installed: pip install -U kaleido")
        print(f"Error: {e}")

# ------------------------- Experiment runner -------------------------

def run_replicas(cfg: Config):
    rng_master = np.random.default_rng(cfg.seed)
    mspe_list = []

    Y_plot = None
    Yhat_plot = None

    for r in range(cfg.num_replicas):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        Y, _, _ = simulate_factor_model(cfg, rng)
        mspe, Y_hat = rolling_pca_forecast(
            Y, cfg.r_hat, cfg.window, cfg.factor_forecast_method, cfg.center_window
        )
        mspe_list.append(mspe)

        if cfg.save_plot and r == cfg.replica_to_plot:
            Y_plot, Yhat_plot = Y, Y_hat

    mspe_arr = np.array(mspe_list)
    return mspe_arr.mean(), mspe_arr.std(ddof=1), mspe_arr, Y_plot, Yhat_plot

# ------------------------- CLI -------------------------

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="hyperparameters.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    avg_mspe, std_mspe, mspe_arr, Y_plot, Yhat_plot = run_replicas(cfg)

    print(f"Average MSPE over {cfg.num_replicas} replicas: {avg_mspe:.6f}")
    print(f"Std. dev. of MSPEs: {std_mspe:.6f}")

    if cfg.save_plot and Y_plot is not None and Yhat_plot is not None:
        plot_targets_predictions(
            Y_plot, Yhat_plot, cfg.window, cfg.series_to_plot, cfg.plot_pdf_path
        )


if __name__ == "__main__":
    main()
