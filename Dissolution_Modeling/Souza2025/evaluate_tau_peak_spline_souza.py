#!/usr/bin/env python3
"""
demo_tau_prior_souza_grid_2col.py

Tau-prior diagnostics for Souza dataset in a compact grid:

Each ROW = up to MAX_CURVES_PER_ROW BatchIDs overlaid
Two columns per row:
  (0) data + final spline yhat(t) + vertical t_peak
  (1) df/dt from spline + vertical t_peak

Color-coordinated per curve within each row.

Requires tau_prior_tools.py with:
  compute_tau_lognormal_prior(..., include_arrays=True)
which provides diag["tg"], diag["yhat"], diag["dfdt"], diag["t_star"].

Deps:
  pip install numpy pandas matplotlib scipy
  optional for QP smoothing inside tau_prior_tools:
    pip install cvxpy osqp (or scs)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tau_prior_tools import compute_tau_lognormal_prior


# -----------------------------
# USER SETTINGS
# -----------------------------
CSV_PATH  = "Souza2025_TableS1_Final.csv"
BATCH_COL = "BatchID"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

OUT_DIR = "souza_tau_prior_demo"
OUT_PNG = os.path.join(OUT_DIR, "souza_tau_prior_grid_2col.png")

# Which batches to include
BATCHES = None  # None => all, truncated to MAX_ROWS*MAX_CURVES_PER_ROW

# Layout
MAX_CURVES_PER_ROW = 10
MAX_ROWS = 10
DPI = 180
USE_TEX = True

# tau0 = TAU_FACTOR * t_peak_repr (doesn't change plotting; kept for consistency)
TAU_FACTOR = 1.0

# Tau-prior estimator knobs
SMOOTH_LAMBDA      = 50.0
PEAK_FRAC          = 0.95
EXCLUDE_BOUNDARIES = False
DENSE              = 2000
BOUNDARY_EPS_FRAC  = 1e-4
BOUNDARY_EPS_ABS   = 1e-12
QP_SOLVER          = "OSQP"   # "OSQP" or "SCS"
ENFORCE_01         = True

# Flatness handling inside tau_prior_tools
FLAT_RATIO_THRESHOLD = 1.10
LOGSIG              = 0.50
FLAT_LOGSIG         = 1.00
EARLY_EPS_ABS        = 1e-12
EARLY_EPS_FRAC       = 1e-6

# Optional x-window cropping for ALL batches
T_MIN = None
T_MAX = None


# -----------------------------
# Matplotlib
# -----------------------------
if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
    })


def preprocess_one_curve(t, y):
    """Sort, average duplicate times, clip y to [0,1]."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    ut, inv = np.unique(t, return_inverse=True)
    if ut.size != t.size:
        y_sum = np.zeros_like(ut, dtype=float)
        cnt = np.zeros_like(ut, dtype=float)
        for i, g in enumerate(inv):
            y_sum[g] += y[i]
            cnt[g] += 1.0
        y = y_sum / np.maximum(cnt, 1.0)
        t = ut

    y = np.clip(y, 0.0, 1.0)
    return t, y


def crop_window(t, y, tmin, tmax):
    if tmin is None and tmax is None:
        return t, y
    tmin2 = -np.inf if tmin is None else float(tmin)
    tmax2 = +np.inf if tmax is None else float(tmax)
    m = (t >= tmin2) & (t <= tmax2)
    if np.sum(m) >= 3:
        return t[m], y[m]
    return t, y


def chunk_list(xs, k):
    for i in range(0, len(xs), k):
        yield xs[i:i+k]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df[BATCH_COL] = df[BATCH_COL].astype(str)

    bids = sorted(df[BATCH_COL].unique().tolist())
    if BATCHES is not None:
        keep = set(str(b) for b in BATCHES)
        bids = [b for b in bids if b in keep]

    if not bids:
        print("No BatchIDs found for selection. Check CSV_PATH and BATCHES.")
        return

    max_total = MAX_ROWS * MAX_CURVES_PER_ROW
    if len(bids) > max_total:
        print(f"[info] Plotting only first {max_total} batches "
              f"({MAX_ROWS} rows x {MAX_CURVES_PER_ROW} curves/row).")
        bids = bids[:max_total]

    rows = list(chunk_list(bids, MAX_CURVES_PER_ROW))
    rows = rows[:MAX_ROWS]

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=2,
        figsize=(14.0, 3.3 * len(rows)),
        dpi=DPI,
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    for r, bids_row in enumerate(rows, start=1):
        ax_fit = axes[r-1, 0]
        ax_d   = axes[r-1, 1]

        cmap = plt.get_cmap("tab20")
        row_tmax = 0.0
        row_dfdt_max = 0.0

        for j, bid in enumerate(bids_row):
            dfi = df[df[BATCH_COL] == str(bid)].sort_values(TIME_COL)
            t_raw = dfi[TIME_COL].to_numpy(float)
            y_raw = dfi[Y_COL].to_numpy(float)

            t, y = preprocess_one_curve(t_raw, y_raw)
            t, y = crop_window(t, y, T_MIN, T_MAX)
            if t.size < 4:
                continue

            row_tmax = max(row_tmax, float(np.max(t)))
            color = cmap(j % cmap.N)

            # --- compute spline diagnostics ---
            try:
                res = compute_tau_lognormal_prior(
                    t, y,
                    tau_factor=TAU_FACTOR,
                    tau_bounds=None,
                    smooth_lambda=SMOOTH_LAMBDA,
                    solver=QP_SOLVER,
                    enforce_01=ENFORCE_01,
                    dense=DENSE,
                    peak_frac=PEAK_FRAC,
                    exclude_boundaries=EXCLUDE_BOUNDARIES,
                    boundary_eps_frac=BOUNDARY_EPS_FRAC,
                    boundary_eps_abs=BOUNDARY_EPS_ABS,
                    flat_ratio_threshold=FLAT_RATIO_THRESHOLD,
                    early_eps_abs=EARLY_EPS_ABS,
                    early_eps_frac=EARLY_EPS_FRAC,
                    logsig=LOGSIG,
                    flat_logsig=FLAT_LOGSIG,
                    include_arrays=True,
                )
                diag = res.diag or {}

                tg   = np.asarray(diag.get("tg", np.array([], float)), float)
                yhat = np.asarray(diag.get("yhat", np.array([], float)), float)
                dfdt = np.asarray(diag.get("dfdt", np.array([], float)), float)

                t_peak = float(diag.get("t_star", np.nan))

            except Exception as e:
                print(f"[warn] {bid}: tau prior estimation failed ({type(e).__name__}: {e})")
                tg = np.array([], float)
                yhat = np.array([], float)
                dfdt = np.array([], float)
                t_peak = np.nan

            # --- Plot left: data + spline + t_peak ---
            ax_fit.plot(t, y, "o", ms=3.0, alpha=0.75, color=color, label=str(bid))
            if tg.size and yhat.size:
                ax_fit.plot(tg, yhat, "-", lw=2.0, alpha=0.95, color=color)
            if np.isfinite(t_peak):
                ax_fit.axvline(t_peak, ls="--", lw=1.5, alpha=0.8, color=color)

            # --- Plot right: df/dt + t_peak ---
            if tg.size and dfdt.size:
                ax_d.plot(tg, dfdt, "-", lw=2.0, alpha=0.95, color=color, label=str(bid))
                if np.isfinite(np.nanmax(dfdt)):
                    row_dfdt_max = max(row_dfdt_max, float(np.nanmax(dfdt)))
            if np.isfinite(t_peak):
                ax_d.axvline(t_peak, ls="--", lw=1.5, alpha=0.8, color=color)

        # --- Format row axes ---
        # Left: release
        ax_fit.set_ylim(-0.05, 1.05)
        ax_fit.set_xlim(0.0, max(1e-9, row_tmax))
        ax_fit.set_xlabel("time (min)")
        ax_fit.set_ylabel("release fraction")
        ax_fit.set_title(f"Row {r}: data + spline + $t_\\mathrm{{peak}}$")

        # Right: derivative
        ax_d.set_xlim(0.0, max(1e-9, row_tmax))
        if row_dfdt_max > 0:
            ax_d.set_ylim(0.0, 1.05 * row_dfdt_max)
        ax_d.set_xlabel("time (min)")
        ax_d.set_ylabel(r"$df/dt$")
        ax_d.set_title(f"Row {r}: spline derivative $df/dt$ + $t_\\mathrm{{peak}}$")

        # Legends (one per panel; both match colors)
        ax_fit.legend(frameon=True, fontsize=8, ncol=2, loc="best")
        ax_d.legend(frameon=True, fontsize=8, ncol=2, loc="best")

    fig.suptitle(
        rf"Souza dataset: monotone-spline diagnostics "
        rf"({MAX_CURVES_PER_ROW} curves/row, up to {MAX_ROWS} rows). "
        rf"$\lambda_s$={SMOOTH_LAMBDA:g}, peak\_frac={PEAK_FRAC}, solver={QP_SOLVER}",
        y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
