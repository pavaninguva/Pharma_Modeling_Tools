#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# -------------------------
# USER SETTINGS
# -------------------------
CSV_PATH = "Souza2025_TableS1_Final.csv"
OUT_CSV  = "souza_fitted_params_per_batch.csv"

BATCH_COL = "BatchID"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# Plot settings
MAKE_PLOTS = True
FIT_OVERVIEW_PNG = "souza_fit_overview.png"
PLOT_MAX_PER_AX = 10          # at most 10 batches per subplot
PLOT_LIMIT = None             # set e.g. 40 to plot only first 40 batches (None = plot all)

# Fitting settings
N_STARTS = 20                 # multi-starts per batch
SEED = 42

# parameter bounds (positive); wide bounds
LAM_BOUNDS  = (1e-8, 1e3)
TAU_BOUNDS  = (1e-6, 1e12)
N_BOUNDS    = (0.02, 10.0)
BETA_BOUNDS = (1e-6, 1e3)

# solve_ivp options
SOLVER_METHOD = "RK45"         # "RK45" can also work
RTOL = 1e-7
ATOL = 1e-9
MAX_STEP = 1.0

# Resume / checkpoint behavior
RESUME = True
SKIP_IF_SUCCESS_ONLY = True   # True: skip only if previous success==True; False: skip if any record exists
CHECKPOINT_EVERY = 1


# -------------------------
# MODEL
# df/dt = λ * (t^n / (τ + t^n)) * (1 - f)^β
# -------------------------
def ode_rhs(t, y, lam, tau, n, beta):
    f = float(y[0])

    # numeric safety
    if f >= 1.0:
        return [0.0]
    if f < 0.0:
        f = 0.0

    if t <= 0.0:
        g = 0.0
    else:
        t_pow = t**n
        g = t_pow / (tau + t_pow + 1e-15)

    one_minus_f = max(1.0 - f, 0.0)
    return [lam * g * (one_minus_f ** beta)]


def solve_curve(t_eval, lam, tau, n, beta):
    """Solve ODE on [0, max(t_eval)] and return f(t_eval)."""
    t_eval = np.asarray(t_eval, dtype=float)
    tf = float(np.max(t_eval))

    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, lam, tau, n, beta),
        t_span=(0.0, tf),
        y0=[0.0],
        t_eval=t_eval,
        method=SOLVER_METHOD,
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
    )

    if (not sol.success) or sol.y is None or sol.y.shape[1] != len(t_eval):
        return None

    return np.clip(sol.y[0], 0.0, 1.0)


# -------------------------
# FITTING (log-parameter space)
# z = [log λ, log τ, log n, log β]
# -------------------------
def unpack_params(z):
    z = np.clip(np.asarray(z, dtype=float), -60.0, 60.0)  # avoid exp overflow
    lam, tau, n, beta = np.exp(z)
    return float(lam), float(tau), float(n), float(beta)


def residuals(z, t, y_obs):
    lam, tau, n, beta = unpack_params(z)
    y_pred = solve_curve(t, lam, tau, n, beta)
    if y_pred is None:
        return np.ones_like(y_obs) * 1e3  # penalize solver failures
    return (y_pred - y_obs)


def fit_one_batch(t, y, rng):
    lb = np.log([LAM_BOUNDS[0], TAU_BOUNDS[0], N_BOUNDS[0], BETA_BOUNDS[0]])
    ub = np.log([LAM_BOUNDS[1], TAU_BOUNDS[1], N_BOUNDS[1], BETA_BOUNDS[1]])

    best = None
    for _ in range(N_STARTS):
        z0 = rng.uniform(lb, ub)

        res = least_squares(
            fun=residuals,
            x0=z0,
            bounds=(lb, ub),
            args=(t, y),
            method="trf",
            loss="soft_l1",   # robust; use "linear" for plain least squares
            f_scale=0.05,
            max_nfev=200,
        )

        print(f"for start# {_}, residual = {res.cost}")

        if (best is None) or (res.cost < best.cost):
            best = res
            

    lam, tau, n, beta = unpack_params(best.x)
    y_pred = solve_curve(t, lam, tau, n, beta)
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))

    return {
        "lambda": lam,
        "tau": tau,
        "n": n,
        "beta": beta,
        "rmse": rmse,
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "message": str(best.message),
    }, y_pred


# -------------------------
# PLOTTING: one figure, multiple subplots, <=10 batches each
# -------------------------
def plot_overview(curves, out_png, max_per_ax=10):
    """
    curves: list of dicts with keys: bid, t, y, y_pred
    """
    n = len(curves)
    if n == 0:
        raise ValueError("No curves to plot.")

    n_panels = int(math.ceil(n / float(max_per_ax)))
    ncols = int(math.ceil(math.sqrt(n_panels)))
    nrows = int(math.ceil(n_panels / float(ncols)))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 4.0 * nrows),
        dpi=180,
        sharex=True,
        sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap("tab20")

    for p in range(n_panels):
        ax = axes[p]
        i0 = p * max_per_ax
        i1 = min((p + 1) * max_per_ax, n)

        for j, item in enumerate(curves[i0:i1]):
            c = cmap(j % cmap.N)
            bid = str(item["bid"])
            t = item["t"]
            y = item["y"]
            y_pred = item["y_pred"]

            # continuous fitted curve
            ax.plot(t, y_pred, color=c, lw=2.0, alpha=0.95, label=bid)
            # measured data overlay (no extra legend entry)
            ax.plot(t, y, "o", color=c, ms=3.2, alpha=0.75, label="_nolegend_")

        ax.set_title(f"Batches {i0+1}–{i1} of {n}")
        ax.set_xlim(0.0, float(np.max(curves[i0]["t"])))
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("release fraction")
        ax.legend(frameon=True, fontsize=8, ncol=1)

    # hide unused axes
    for p in range(n_panels, len(axes)):
        axes[p].axis("off")

    fig.suptitle("ODE fits per BatchID (line = fitted ODE, markers = measured data)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

#-------------------------
# CHECKPOINTING
# ------------------------

def load_done_batchids(out_csv):
    """
    Returns:
      done_set: set of BatchIDs already fitted (based on OUT_CSV)
      df_prev:  previously saved results (or empty df)
    """
    if (not RESUME) or (not os.path.exists(out_csv)):
        return set(), pd.DataFrame()

    df_prev = pd.read_csv(out_csv)
    if df_prev.empty or (BATCH_COL not in df_prev.columns):
        return set(), df_prev

    if SKIP_IF_SUCCESS_ONLY and ("success" in df_prev.columns):
        done = set(df_prev.loc[df_prev["success"] == True, BATCH_COL].astype(str))
    else:
        done = set(df_prev[BATCH_COL].astype(str))

    return done, df_prev


def append_checkpoint(out_csv, new_rows):
    """
    Append rows safely to OUT_CSV.
    If file doesn't exist, write header.
    """
    df_new = pd.DataFrame(new_rows)
    write_header = not os.path.exists(out_csv)
    df_new.to_csv(out_csv, mode="a", header=write_header, index=False)


# -------------------------
# MAIN
# -------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    rng = np.random.default_rng(SEED)

    batch_ids = sorted(df[BATCH_COL].astype(str).unique().tolist())
    done_set, df_prev = load_done_batchids(OUT_CSV)

    if RESUME and done_set:
        print(f"Resume enabled: skipping {len(done_set)} already-fitted BatchIDs from {OUT_CSV}")

    new_rows_buffer = []
    curves_for_plot = []

    for i, bid in enumerate(batch_ids, 1):
        if RESUME and (bid in done_set):
            continue

        dfi = df[df[BATCH_COL].astype(str) == bid].sort_values(TIME_COL)

        t = dfi[TIME_COL].to_numpy(float)
        y = dfi[Y_COL].to_numpy(float)

        # average duplicates in time, if any
        if len(np.unique(t)) != len(t):
            tmp = pd.DataFrame({"t": t, "y": y}).groupby("t", as_index=False).mean()
            t = tmp["t"].to_numpy(float)
            y = tmp["y"].to_numpy(float)

        if len(t) < 3:
            print(f"[skip] {bid}: too few points ({len(t)})")
            continue

        fit, y_pred = fit_one_batch(t, y, rng)

        row = {
            "BatchID": bid,
            "n_points": int(len(t)),
            "t_min": float(t.min()),
            "t_max": float(t.max()),
        }
        row.update(fit)
        new_rows_buffer.append(row)

        if MAKE_PLOTS:
            curves_for_plot.append({"bid": bid, "t": t, "y": y, "y_pred": y_pred})

        print(f"[fit] {bid}: rmse={fit['rmse']:.5f} success={fit['success']}")

        # checkpoint write
        if len(new_rows_buffer) >= CHECKPOINT_EVERY:
            append_checkpoint(OUT_CSV, new_rows_buffer)
            new_rows_buffer = []

    # flush remaining
    if new_rows_buffer:
        append_checkpoint(OUT_CSV, new_rows_buffer)

    print(f"\nSaved/updated: {OUT_CSV}")

    # Optional: make one overview plot
    if MAKE_PLOTS:
        curves = curves_for_plot

        # If RESUME is on and you want to plot *all* fitted batches (old + new),
        # you would need to re-solve curves using df_prev parameters.
        # For simplicity, we plot only the batches fitted in THIS run.
        if PLOT_LIMIT is not None:
            curves = curves[:int(PLOT_LIMIT)]

        if len(curves) == 0:
            print("No new fitted batches to plot this run (everything may already be done).")
        else:
            plot_overview(curves, FIT_OVERVIEW_PNG, max_per_ax=PLOT_MAX_PER_AX)
            print(f"Saved overview plot (new fits this run): {FIT_OVERVIEW_PNG}")


if __name__ == "__main__":
    main()