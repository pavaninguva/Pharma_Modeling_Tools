#!/usr/bin/env python3
"""
demo_tau_prior.py

Visualize tau prior estimation (t* from max df/dt of a MONOTONE-SMOOTHED curve).

This version matches the simplified tau_prior_tools.py (NO PAV):
  - monotone smoothing via convex QP (cvxpy)
  - PCHIP on the monotone-smoothed points
  - df/dt on a dense grid, t* from plateau median
  - tau0 = TAU_FACTOR * t*

Produces one row per group (Case or BatchID):
  Left: data + monotone-smoothed points + PCHIP fit (smooth curve)
  Right: df/dt vs t, marks t* and tau0

Run:
  python demo_tau_prior.py

Deps:
  pip install cvxpy osqp corner emcee scipy numpy pandas matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tau_prior_tools import compute_tau_prior_from_data


# -----------------------------
# USER SETTINGS
# -----------------------------
CSV_PATH   = "synthetic_bayes_demo/synthetic_data.csv"  # change if needed
GROUP_COL  = "Case"        # "Case" for synthetic; "BatchID" for real
TIME_COL   = "time_min"
Y_COL      = "release_frac"

GROUPS     = None          # None = all groups; or e.g. ["Zero","First","Sigmoid"]

OUT_DIR    = "synthetic_bayes_demo"
OUT_PNG    = os.path.join(OUT_DIR, "tau_prior_diagnostics.png")

# tau0 = TAU_FACTOR * t*
TAU_FACTOR = 1.0

# --- New estimator knobs (QP smoothing) ---
# Larger => smoother (try 0, 1e-2, 1e-1, 1, 10, 100 depending on scale/noise)
SMOOTH_LAMBDA      = 5.0

# plateau threshold for selecting t* (0.95–0.99 typical)
PEAK_FRAC          = 0.99

# avoid boundary spikes dominating (often True is safer)
EXCLUDE_BOUNDARIES = False

# dense grid for df/dt evaluation
DENSE              = 2000

# enforce 0<=f<=1 in the monotone smoothing
ENFORCE_01         = True

# QP solver: "OSQP" (fast) or "SCS" (more robust sometimes)
QP_SOLVER          = "OSQP"

# Optional manual t* search window (minutes)
# (If you set these, demo passes them through by slicing the dense grid window indirectly:
#  easiest is to crop your data before calling compute_tau_prior_from_data.)
TSTAR_MIN = None   # e.g. 5.0 or 10.0
TSTAR_MAX = None   # e.g. 180.0 or 400.0

DPI = 180
USE_TEX = True


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


def _crop_window(t, y, tmin, tmax):
    if tmin is None and tmax is None:
        return t, y
    tmin2 = -np.inf if tmin is None else float(tmin)
    tmax2 = +np.inf if tmax is None else float(tmax)
    m = (t >= tmin2) & (t <= tmax2)
    # keep at least 3 points if possible
    if np.sum(m) >= 3:
        return t[m], y[m]
    return t, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    groups = sorted(df[GROUP_COL].unique().tolist())
    if GROUPS is not None:
        keep = set(str(g) for g in GROUPS)
        groups = [g for g in groups if g in keep]

    if not groups:
        print(f"No groups found for {GROUP_COL}.")
        return

    fig, axes = plt.subplots(
        nrows=len(groups),
        ncols=2,
        figsize=(12.5, 3.4 * len(groups)),
        dpi=DPI,
        sharex=False,
        sharey=False
    )
    axes = np.atleast_2d(axes)

    for i, g in enumerate(groups):
        sub = df[df[GROUP_COL] == g].sort_values(TIME_COL)
        t = sub[TIME_COL].to_numpy(float)
        y = np.clip(sub[Y_COL].to_numpy(float), 0.0, 1.0)

        # optional crop of the *data* window (simple way to constrain t* search)
        t_use, y_use = _crop_window(t, y, TSTAR_MIN, TSTAR_MAX)

        tau0, t_star, dfdt_max, diag = compute_tau_prior_from_data(
            t_use, y_use,
            tau_factor=TAU_FACTOR,
            smooth_lambda=SMOOTH_LAMBDA,
            dense=DENSE,
            peak_frac=PEAK_FRAC,
            exclude_boundaries=EXCLUDE_BOUNDARIES,
            enforce_01=ENFORCE_01,
            solver=QP_SOLVER,
        )

        ax_fit = axes[i, 0]
        ax_d   = axes[i, 1]

        # ---- Left panel: fit ----
        tt    = diag["t"]
        y_obs = diag["y_obs"]
        y_mono = diag["y_mono"]
        tg    = diag["tg"]
        yhat  = diag["yhat"]

        ax_fit.plot(tt, y_obs, "o", ms=4.0, alpha=0.85, label="data")
        ax_fit.plot(tt, y_mono, "-", lw=2.0, alpha=0.9, label="monotone-smoothed (QP)")
        ax_fit.plot(tg, yhat, "-", lw=2.0, alpha=0.9, label="PCHIP(on smoothed)")

        if np.isfinite(t_star):
            ax_fit.axvline(t_star, ls="--", lw=1.6, alpha=0.8, label=rf"$t^*={t_star:.2g}$")
            ax_fit.axvline(tau0,   ls=":",  lw=1.6, alpha=0.8, label=rf"$\tau_0={tau0:.2g}$")

        ax_fit.set_ylim(-0.05, 1.05)
        ax_fit.set_xlabel("time (min)")
        ax_fit.set_ylabel("release fraction")
        ax_fit.set_title(
            rf"{g}: monotone QP smooth ($\lambda_s$={SMOOTH_LAMBDA:g}, solver={QP_SOLVER})"
        )
        ax_fit.legend(frameon=True, fontsize=8, loc="best")

        # ---- Right panel: derivative ----
        dfdt = diag["dfdt"]
        ax_d.plot(tg, dfdt, lw=2.0)

        if np.isfinite(t_star):
            ax_d.axvline(t_star, ls="--", lw=1.6, alpha=0.8)
            if np.isfinite(dfdt_max):
                ax_d.axhline(dfdt_max, ls=":", lw=1.2, alpha=0.7)
                ax_d.set_title(rf"$df/dt$ (max={dfdt_max:.2g} at $t^*={t_star:.2g}$)")
            else:
                ax_d.set_title(rf"$df/dt$ (at $t^*={t_star:.2g}$)")
        else:
            ax_d.set_title(r"$df/dt$ (t* failed)")

        ax_d.set_xlabel("time (min)")
        ax_d.set_ylabel(r"$df/dt$")

    fig.suptitle(
        rf"Tau prior diagnostics: $\tau_0={TAU_FACTOR}\,t^*$  "
        rf"(peak\_frac={PEAK_FRAC}, exclude\_boundaries={EXCLUDE_BOUNDARIES}, "
        rf"$\lambda_s$={SMOOTH_LAMBDA:g})",
        y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print("\nIf t* looks wrong, try:")
    print("  - increase SMOOTH_LAMBDA (e.g. 0 -> 1 -> 10 -> 100)")
    print("  - set EXCLUDE_BOUNDARIES=True (usually safer)")
    print("  - lower PEAK_FRAC (0.98 -> 0.95) to stabilize plateau selection")
    print("  - set TSTAR_MIN to ignore early-time artifacts (e.g. 5 or 10)")
    print("  - switch QP_SOLVER='SCS' if OSQP struggles")


if __name__ == "__main__":
    main()