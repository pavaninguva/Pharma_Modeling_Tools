import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ============================================================
# USER SETTINGS
# ============================================================
CSV_PATH = "Souza2025_TableS1_Final.csv"

BATCH_COL = "BatchID"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# Which cases to profile
BATCHES = ["F14", "F11", "F25", "F37", "F54"]

# Output
OUT_DIR = "profiles_selected_nfix1_nll"
OUT_PROFILE_CSV = os.path.join(OUT_DIR, "profiles_points_selected.csv")
OUT_FIT_CSV     = os.path.join(OUT_DIR, "profiles_bestfits_selected.csv")
OUT_FIG_PNG     = os.path.join(OUT_DIR, "profiles_selected_grid.png")
OUT_FITCURVES_PNG = os.path.join(OUT_DIR, "profiles_bestfit_curves.png")

# Checkpoint behavior
FORCE_RECOMPUTE = True

# Fit settings (per batch)
N_STARTS = 20
SEED = 42
MAX_NFEV_FIT = 250

# Profile settings
# We'll build a symmetric grid around the optimum:
#   p_hat[j] +/- PROFILE_SPAN_LOG  (in log-space),
# with ~N_GRID//2 points on each side (plus the center).
N_GRID = 50
PROFILE_SPAN_LOG = 8.0
MAX_NFEV_PROFILE = 250

# ODE solver settings
SOLVER_METHOD = "RK45"
RTOL = 1e-7
ATOL = 1e-9
MAX_STEP = 1.0

# Fixed exponent
N_FIXED = 1.0

# Parameter bounds (natural space; wide)
LAM_BOUNDS  = (1e-5, 1e4)
TAU_BOUNDS  = (1e-5, 1e4)
BETA_BOUNDS = (1e-3, 1e1)

# Likelihood-ratio 95% threshold for 1 dof:
# 2*(NLL - NLL_min) <= 3.84  ->  NLL <= NLL_min + 1.92
DELTA_NLL_95 = 0.5 * 3.84

# ============================================================
# Matplotlib LaTeX settings
# NOTE: requires a working LaTeX installation on your system.
# ============================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
})

# ============================================================
# MODEL (c* term dropped): df/dt = λ * (t^n/(τ+t^n)) * (1-f)^β
# with n fixed to N_FIXED
# ============================================================
def ode_rhs(t, y, lam, tau, beta):
    f = float(y[0])

    if f >= 1.0:
        return [0.0]
    if f < 0.0:
        f = 0.0

    if t <= 0.0:
        g = 0.0
    else:
        t_pow = t ** N_FIXED
        g = t_pow / (tau + t_pow + 1e-15)

    one_minus_f = max(1.0 - f, 0.0)
    return [lam * g * (one_minus_f ** beta)]


def solve_curve(t_eval, lam, tau, beta):
    """
    Solve on [0, max(t_eval)] and return f(t_eval) aligned to t_eval.
    Assumes t_eval includes t=0 if your data includes it.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    tf = float(np.max(t_eval))

    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, lam, tau, beta),
        t_span=(0.0, tf),
        y0=[0.0],
        t_eval=t_eval,
        method=SOLVER_METHOD,
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
    )

    if (not sol.success) or (sol.y is None) or (sol.y.shape[1] != len(t_eval)):
        return None

    return np.clip(sol.y[0], 0.0, 1.0)

# ============================================================
# Log-parameterization helpers
# p = [loglam, logtau, logbeta]
# ============================================================
def bounds_log():
    lb = np.log([LAM_BOUNDS[0], TAU_BOUNDS[0], BETA_BOUNDS[0]])
    ub = np.log([LAM_BOUNDS[1], TAU_BOUNDS[1], BETA_BOUNDS[1]])
    return lb, ub

def unpack(p_log):
    p_log = np.clip(np.asarray(p_log, float), -60.0, 60.0)
    lam, tau, beta = np.exp(p_log)
    return float(lam), float(tau), float(beta)

def residuals_log(p_log, t, y_obs):
    lam, tau, beta = unpack(p_log)
    y_pred = solve_curve(t, lam, tau, beta)
    if y_pred is None:
        return np.ones_like(y_obs) * 1e3
    return (y_pred - y_obs)

# ============================================================
# NLL (concentrated) from SSE
# NLL = const + (N/2)*log(SSE/N)
# ============================================================
def nll_from_sse(sse, npts):
    sse = max(float(sse), 1e-300)
    npts = int(npts)
    sigma2_hat = sse / max(npts, 1)
    return 0.5 * npts * (np.log(2.0 * np.pi * sigma2_hat) + 1.0)

# ============================================================
# Fit best parameters for one batch (multi-start)
# ============================================================
def fit_one_batch(t, y, rng):
    lb, ub = bounds_log()
    best = None

    for _ in range(N_STARTS):
        p0 = rng.uniform(lb, ub)
        res = least_squares(
            fun=residuals_log,
            x0=p0,
            bounds=(lb, ub),
            args=(t, y),
            method="trf",
            loss="linear",
            max_nfev=MAX_NFEV_FIT,
        )
        if (best is None) or (res.cost < best.cost):
            best = res

    p_hat = best.x
    r_hat = residuals_log(p_hat, t, y)
    sse_min = float(np.sum(r_hat**2))
    nll_min = float(nll_from_sse(sse_min, len(y)))

    lam, tau, beta = unpack(p_hat)
    y_pred = solve_curve(t, lam, tau, beta)

    return {
        "p_hat": p_hat,
        "lam": lam, "tau": tau, "beta": beta,
        "n_fixed": float(N_FIXED),
        "sse_min": sse_min,
        "nll_min": nll_min,
        "nll_thresh_95": nll_min + DELTA_NLL_95,
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "message": str(best.message),
        "y_pred": y_pred,
    }

# ============================================================
# Build a symmetric log-grid around the optimum (k points each side)
# ============================================================
def make_symmetric_grid_log(p0, lb, ub, span, n_grid):
    """
    Returns a sorted, unique grid that includes p0 and extends +/- span (clipped to [lb, ub]),
    with roughly n_grid//2 points on each side.
    """
    K = max(int(n_grid // 2), 1)

    lo = max(lb, p0 - span)
    hi = min(ub, p0 + span)

    # Build symmetric offsets, then clip endpoints via lo/hi
    # Use the available side lengths after clipping so spacing stays sensible.
    left_span  = p0 - lo
    right_span = hi - p0

    # Ensure p0 included
    left  = p0 - np.linspace(left_span, 0.0, K + 1)   # includes p0 at end
    right = p0 + np.linspace(0.0, right_span, K + 1)  # includes p0 at start

    grid = np.concatenate([left[:-1], right])  # keep single p0
    grid = np.unique(np.sort(grid))
    return grid

# ============================================================
# Profile NLL for parameter j (0..2) using BIDIRECTIONAL continuation
# ============================================================
def profile_param_bidirectional(t, y, p_hat, j, grid_log):
    """
    Compute profile NLL on grid_log for parameter j by:
      - starting at the optimum (p_hat),
      - sweeping to the right (increasing grid index) with warm-start continuation,
      - sweeping to the left (decreasing grid index) with warm-start continuation.
    This avoids latching onto a bad solution branch.
    """
    lb, ub = bounds_log()

    free = [k for k in range(3) if k != j]
    lb_free = lb[free]
    ub_free = ub[free]

    grid_log = np.asarray(grid_log, float)
    nll_prof = np.full_like(grid_log, np.nan, dtype=float)

    # Find the center index (grid point closest to p_hat[j])
    k0 = int(np.argmin(np.abs(grid_log - p_hat[j])))

    # --- Evaluate at the center using p_hat exactly (forces curve through optimum) ---
    p_center = np.array(p_hat, float)
    p_center[j] = grid_log[k0]  # should be equal/very close to p_hat[j]
    r = residuals_log(p_center, t, y)
    nll_prof[k0] = nll_from_sse(np.sum(r**2), len(y))

    # Helper: solve constrained problem at fixed pj with warm start
    def solve_constrained(p_prev, pj):
        x0 = p_prev[free].copy()

        def fun_free(x_free):
            p = p_prev.copy()
            p[j] = pj
            p[free] = x_free
            return residuals_log(p, t, y)

        res = least_squares(
            fun_free,
            x0=x0,
            bounds=(lb_free, ub_free),
            method="trf",
            loss="linear",
            max_nfev=MAX_NFEV_PROFILE,
        )

        p_sol = p_prev.copy()
        p_sol[j] = pj
        p_sol[free] = res.x

        rr = residuals_log(p_sol, t, y)
        sse = float(np.sum(rr**2))
        return p_sol, float(nll_from_sse(sse, len(y)))

    # --- Sweep right (k0+1 .. end) ---
    p_prev = np.array(p_hat, float)
    p_prev[j] = grid_log[k0]
    for k in range(k0 + 1, len(grid_log)):
        pj = float(grid_log[k])
        p_prev, nllk = solve_constrained(p_prev, pj)
        nll_prof[k] = nllk

    # --- Sweep left (k0-1 .. 0) ---
    p_prev = np.array(p_hat, float)
    p_prev[j] = grid_log[k0]
    for k in range(k0 - 1, -1, -1):
        pj = float(grid_log[k])
        p_prev, nllk = solve_constrained(p_prev, pj)
        nll_prof[k] = nllk

    return nll_prof

# ============================================================
# Helper: compute RMSE (for printing/diagnostics if you want)
# ============================================================
def rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df[BATCH_COL] = df[BATCH_COL].astype(str)
    rng = np.random.default_rng(SEED)

    fit_rows = []
    prof_rows = []

    available = set(df[BATCH_COL].unique().tolist())
    missing = [b for b in BATCHES if b not in available]
    if missing:
        print(f"[warn] These BatchIDs are not present in {CSV_PATH}: {missing}")

    existing_prof = None
    existing_fit = None
    if (not FORCE_RECOMPUTE) and os.path.exists(OUT_PROFILE_CSV) and os.path.exists(OUT_FIT_CSV):
        existing_prof = pd.read_csv(OUT_PROFILE_CSV)
        existing_fit = pd.read_csv(OUT_FIT_CSV)

    # Parameter names (only 3 now)
    pnames = ["lambda", "tau", "beta"]

    for bid in BATCHES:
        if bid not in available:
            continue

        # checkpoint: if already done, skip
        if (not FORCE_RECOMPUTE) and (existing_fit is not None):
            if (existing_fit["BatchID"].astype(str) == str(bid)).any():
                print(f"[skip] {bid}: already in {OUT_FIT_CSV}")
                continue

        dfi = df[df[BATCH_COL] == bid].sort_values(TIME_COL)
        t = dfi[TIME_COL].to_numpy(float)
        y = dfi[Y_COL].to_numpy(float)

        # average duplicates at same time (if any)
        if len(np.unique(t)) != len(t):
            tmp = pd.DataFrame({"t": t, "y": y}).groupby("t", as_index=False).mean()
            t = tmp["t"].to_numpy(float)
            y = tmp["y"].to_numpy(float)

        if len(t) < 4:
            print(f"[skip] {bid}: too few points ({len(t)})")
            continue

        print(f"[fit] {bid} (n fixed = {N_FIXED:g}) ...")
        fit = fit_one_batch(t, y, rng)

        fit_rows.append({
            "BatchID": bid,
            "n_points": int(len(t)),
            "t_min": float(t.min()),
            "t_max": float(t.max()),
            "n_fixed": float(N_FIXED),
            "lambda": fit["lam"],
            "tau": fit["tau"],
            "beta": fit["beta"],
            "sse_min": fit["sse_min"],
            "nll_min": fit["nll_min"],
            "nll_thresh_95": fit["nll_thresh_95"],
            "success": fit["success"],
            "nfev": fit["nfev"],
            "message": fit["message"],
        })

        # Profile grids around optimum in log-space (symmetric) + bidirectional continuation
        lb, ub = bounds_log()
        p_hat = fit["p_hat"]

        for j, name in enumerate(pnames):
            grid_log = make_symmetric_grid_log(
                p0=float(p_hat[j]),
                lb=float(lb[j]),
                ub=float(ub[j]),
                span=float(PROFILE_SPAN_LOG),
                n_grid=int(N_GRID),
            )

            print(f"  [profile] {bid} param={name} (bidirectional) ...")
            nll_prof = profile_param_bidirectional(t, y, p_hat, j, grid_log)

            grid_nat = np.exp(grid_log)
            for k in range(len(grid_log)):
                prof_rows.append({
                    "BatchID": bid,
                    "param": name,
                    "grid_value": float(grid_nat[k]),
                    "grid_log": float(grid_log[k]),
                    "nll_prof": float(nll_prof[k]),
                    "nll_min": float(fit["nll_min"]),
                    "nll_thresh_95": float(fit["nll_thresh_95"]),
                    "sse_min": float(fit["sse_min"]),
                    "n_fixed": float(N_FIXED),
                })

        # checkpoint write after each batch
        pd.DataFrame(fit_rows).to_csv(OUT_FIT_CSV, index=False)
        pd.DataFrame(prof_rows).to_csv(OUT_PROFILE_CSV, index=False)
        print(f"  [saved] fits -> {OUT_FIT_CSV}")
        print(f"  [saved] profiles -> {OUT_PROFILE_CSV}")

    # Load final data to plot
    if os.path.exists(OUT_PROFILE_CSV) and os.path.exists(OUT_FIT_CSV):
        df_fit = pd.read_csv(OUT_FIT_CSV)
        df_prof = pd.read_csv(OUT_PROFILE_CSV)
    else:
        df_fit = pd.DataFrame(fit_rows)
        df_prof = pd.DataFrame(prof_rows)

    batches_present = [b for b in BATCHES if (df_fit["BatchID"].astype(str) == str(b)).any()]
    if len(batches_present) == 0:
        print("[done] No batches were profiled.")
        return

    # ============================================================
    # FIGURE 1: Profile likelihoods (NLL) + black point at optimum
    # ============================================================
    fig, axes = plt.subplots(
        nrows=len(batches_present), ncols=3,
        figsize=(13.5, 3.2 * len(batches_present)),
        dpi=180,
        sharey=False
    )
    axes = np.atleast_2d(axes)

    for i, bid in enumerate(batches_present):
        row_fit = df_fit[df_fit["BatchID"].astype(str) == str(bid)].iloc[0]
        nll_min = float(row_fit["nll_min"])
        nll_thr = float(row_fit["nll_thresh_95"])

        # optimum parameter values (black point)
        x_opt = {
            "lambda": float(row_fit["lambda"]),
            "tau": float(row_fit["tau"]),
            "beta": float(row_fit["beta"]),
        }

        for j, pname in enumerate(pnames):
            ax = axes[i, j]
            sub = df_prof[(df_prof["BatchID"].astype(str) == str(bid)) & (df_prof["param"] == pname)].copy()
            sub = sub.sort_values("grid_value")

            x = sub["grid_value"].to_numpy(float)
            y_nll = sub["nll_prof"].to_numpy(float)

            mask = np.isfinite(x) & np.isfinite(y_nll)
            x, y_nll = x[mask], y_nll[mask]

            ax.plot(x, y_nll, linewidth=1.8)
            ax.axhline(nll_thr, linestyle="--", linewidth=1.2)

            # black point at the optimum (x_opt[pname], nll_min)
            ax.plot([x_opt[pname]], [nll_min], "o", color="black", markersize=4.5, zorder=5)

            ax.set_xscale("log")
            ax.set_xlabel(pname)
            if j == 0:
                ax.set_ylabel(r"$-\log \mathcal{L}$")

            if i == 0:
                ax.set_title(rf"profile({pname})")

            if len(y_nll) > 0:
                y_lo = min(np.min(y_nll), nll_min)
                y_hi = max(np.max(y_nll), nll_thr)
                pad = 0.05 * max(1.0, abs(y_hi - y_lo))
                ax.set_ylim(y_lo - pad, y_hi + pad)

        # robust LaTeX row label (no "\n" issues)
        label = rf"\begin{{tabular}}{{c}}{bid}\\$n$ fixed = {N_FIXED:g}\end{{tabular}}"
        axes[i, 0].text(
            -0.42, 0.5,
            label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    fig.suptitle(
        r"Profile likelihood curves ($y=-\log\mathcal{L}$). "
        r"Dashed line: 95\% LR threshold ($\mathrm{NLL}_{\min}+1.92$). "
        r"Black dot: optimum.",
        y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # FIGURE 2: Best-fit ODE curve overlayed with data for each batch
    # ============================================================
    n_panels = len(batches_present)
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))

    fig2, axes2 = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5.4 * ncols, 3.6 * nrows),
        dpi=180,
        sharex=False,
        sharey=True
    )
    axes2 = np.atleast_1d(axes2).ravel()

    for k, bid in enumerate(batches_present):
        ax = axes2[k]
        row_fit = df_fit[df_fit["BatchID"].astype(str) == str(bid)].iloc[0]
        lam = float(row_fit["lambda"])
        tau = float(row_fit["tau"])
        beta = float(row_fit["beta"])

        dfi = df[df[BATCH_COL] == str(bid)].sort_values(TIME_COL)
        t = dfi[TIME_COL].to_numpy(float)
        y = np.clip(dfi[Y_COL].to_numpy(float), 0.0, 1.0)

        # average duplicates (if any)
        if len(np.unique(t)) != len(t):
            tmp = pd.DataFrame({"t": t, "y": y}).groupby("t", as_index=False).mean()
            t = tmp["t"].to_numpy(float)
            y = tmp["y"].to_numpy(float)

        y_pred = solve_curve(t, lam, tau, beta)
        if y_pred is None:
            ax.set_title(rf"{bid}: solve failed")
            ax.plot(t, y, "o", ms=3.5, alpha=0.8, label="data")
            ax.legend(frameon=True)
            continue

        r = rmse(y_pred, y)

        ax.plot(t, y_pred, linewidth=2.0, label="best-fit ODE")
        ax.plot(t, y, "o", ms=3.5, alpha=0.8, label="data")
        ax.set_title(rf"{bid}  (RMSE={r:.4f})")
        ax.set_xlabel(r"time (min)")
        ax.set_ylabel(r"release fraction")
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True)

    for k in range(n_panels, len(axes2)):
        axes2[k].axis("off")

    fig2.suptitle(r"Best-fit ODE curves (line) overlaid with data (markers)", y=1.01)
    fig2.tight_layout()
    fig2.savefig(OUT_FITCURVES_PNG, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nSaved profile figure: {OUT_FIG_PNG}")
    print(f"Saved fit-curves fig: {OUT_FITCURVES_PNG}")
    print(f"Saved fits:          {OUT_FIT_CSV}")
    print(f"Saved profiles:      {OUT_PROFILE_CSV}")


if __name__ == "__main__":
    main()
