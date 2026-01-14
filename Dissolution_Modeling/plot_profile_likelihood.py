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
OUT_DIR = "profiles_selected"
OUT_PROFILE_CSV = os.path.join(OUT_DIR, "profiles_points_selected.csv")
OUT_FIT_CSV     = os.path.join(OUT_DIR, "profiles_bestfits_selected.csv")
OUT_FIG_PNG     = os.path.join(OUT_DIR, "profiles_selected_grid.png")

# Checkpoint behavior
FORCE_RECOMPUTE = True

# Fit settings (per batch)
N_STARTS = 20             # multi-starts for initial fit
SEED = 42
MAX_NFEV_FIT = 250       # max function evals per start

# Profile settings
N_GRID = 30              # points per profile
PROFILE_SPAN_LOG = 5.0   # +/- span in log-space around optimum (factor exp(3)~20)
MAX_NFEV_PROFILE = 120   # per grid point constrained optimization

# ODE solver settings
SOLVER_METHOD = "RK45"    # "BDF" robust; try "RK45" if you like
RTOL = 1e-7
ATOL = 1e-9
MAX_STEP = 1.0

# Parameter bounds (in natural space; wide)
LAM_BOUNDS  = (1e-10, 1e4)
TAU_BOUNDS  = (1e-8,  1e12)
N_BOUNDS    = (0.01,  10.0)
BETA_BOUNDS = (1e-8,  1e3)

# ============================================================
# MODEL (c* term dropped): df/dt = λ * (t^n/(τ+t^n)) * (1-f)^β
# ============================================================
def ode_rhs(t, y, lam, tau, n, beta):
    f = float(y[0])

    # Numerically safe clamp
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
    """
    Solve on [0, max(t_eval)] and return f(t_eval) aligned to the given t_eval.
    Assumes t_eval includes t=0 if your data includes it (you said it does).
    """
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

    if (not sol.success) or (sol.y is None) or (sol.y.shape[1] != len(t_eval)):
        return None

    return np.clip(sol.y[0], 0.0, 1.0)


# ============================================================
# Log-parameterization helpers
# p = [loglam, logtau, logn, logbeta]
# ============================================================
def bounds_log():
    lb = np.log([LAM_BOUNDS[0], TAU_BOUNDS[0], N_BOUNDS[0], BETA_BOUNDS[0]])
    ub = np.log([LAM_BOUNDS[1], TAU_BOUNDS[1], N_BOUNDS[1], BETA_BOUNDS[1]])
    return lb, ub

def unpack(p_log):
    p_log = np.clip(np.asarray(p_log, float), -60.0, 60.0)
    lam, tau, n, beta = np.exp(p_log)
    return float(lam), float(tau), float(n), float(beta)

def residuals_log(p_log, t, y_obs):
    lam, tau, n, beta = unpack(p_log)
    y_pred = solve_curve(t, lam, tau, n, beta)
    if y_pred is None:
        return np.ones_like(y_obs) * 1e3
    return (y_pred - y_obs)


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
            loss="linear",      # keep linear for likelihood-style SSE interpretation
            max_nfev=MAX_NFEV_FIT,
        )
        if (best is None) or (res.cost < best.cost):
            best = res

    p_hat = best.x
    r_hat = residuals_log(p_hat, t, y)
    sse_min = float(np.sum(r_hat**2))
    nu = max(len(y) - 4, 1)           # dof
    sigma2 = sse_min / nu
    delta_star = 3.84 * sigma2        # chi^2_{1,0.95} * sigma^2

    lam, tau, n, beta = unpack(p_hat)
    y_pred = solve_curve(t, lam, tau, n, beta)

    return {
        "p_hat": p_hat,
        "lam": lam, "tau": tau, "n": n, "beta": beta,
        "sse_min": sse_min,
        "sigma2": sigma2,
        "delta_star_95": delta_star,
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "message": str(best.message),
        "y_pred": y_pred,
    }


# ============================================================
# Profile SSE for parameter j (0..3), warm-started
# ============================================================
def profile_param(t, y, p_hat, j, grid_log):
    lb, ub = bounds_log()

    free = [k for k in range(4) if k != j]
    lb_free = lb[free]
    ub_free = ub[free]

    sse_prof = np.zeros_like(grid_log, dtype=float)
    p_solutions = []

    # warm start from p_hat
    p_prev = np.array(p_hat, float)

    for k, pj in enumerate(grid_log):
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

        # warm-start next point from this solution
        p_prev = p_sol

        r = residuals_log(p_sol, t, y)
        sse_prof[k] = float(np.sum(r**2))
        p_solutions.append(p_sol)

    return sse_prof, p_solutions


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    rng = np.random.default_rng(SEED)

    # collect results
    fit_rows = []
    prof_rows = []

    # helpful: ensure batches exist
    available = set(df[BATCH_COL].astype(str).unique().tolist())
    missing = [b for b in BATCHES if b not in available]
    if missing:
        print(f"[warn] These BatchIDs are not present in {CSV_PATH}: {missing}")

    # If checkpoint exists, load it (unless forcing recompute)
    existing_prof = None
    existing_fit = None
    if (not FORCE_RECOMPUTE) and os.path.exists(OUT_PROFILE_CSV) and os.path.exists(OUT_FIT_CSV):
        existing_prof = pd.read_csv(OUT_PROFILE_CSV)
        existing_fit = pd.read_csv(OUT_FIT_CSV)

    # Parameter names
    pnames = ["lambda", "tau", "n", "beta"]

    for bid in BATCHES:
        if bid not in available:
            continue

        # checkpoint: if this BatchID already in fits CSV, skip
        if (not FORCE_RECOMPUTE) and (existing_fit is not None):
            if (existing_fit["BatchID"].astype(str) == str(bid)).any():
                print(f"[skip] {bid}: already in {OUT_FIT_CSV}")
                continue

        dfi = df[df[BATCH_COL].astype(str) == str(bid)].sort_values(TIME_COL)
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

        # Fit
        print(f"[fit] {bid} ...")
        fit = fit_one_batch(t, y, rng)

        fit_rows.append({
            "BatchID": bid,
            "n_points": int(len(t)),
            "t_min": float(t.min()),
            "t_max": float(t.max()),
            "lambda": fit["lam"],
            "tau": fit["tau"],
            "n": fit["n"],
            "beta": fit["beta"],
            "sse_min": fit["sse_min"],
            "sigma2": fit["sigma2"],
            "delta_star_95": fit["delta_star_95"],
            "success": fit["success"],
            "nfev": fit["nfev"],
            "message": fit["message"],
        })

        # Build profile grids around optimum in log-space
        lb, ub = bounds_log()
        p_hat = fit["p_hat"]

        for j, name in enumerate(pnames):
            lo = max(lb[j], p_hat[j] - PROFILE_SPAN_LOG)
            hi = min(ub[j], p_hat[j] + PROFILE_SPAN_LOG)
            grid_log = np.linspace(lo, hi, N_GRID)

            print(f"  [profile] {bid} param={name} ...")
            sse_prof, _ = profile_param(t, y, p_hat, j, grid_log)

            delta = sse_prof - fit["sse_min"]
            grid_nat = np.exp(grid_log)

            for k in range(len(grid_log)):
                prof_rows.append({
                    "BatchID": bid,
                    "param": name,
                    "grid_value": float(grid_nat[k]),
                    "grid_log": float(grid_log[k]),
                    "sse_prof": float(sse_prof[k]),
                    "delta_sse": float(delta[k]),
                    "sse_min": float(fit["sse_min"]),
                    "delta_star_95": float(fit["delta_star_95"]),
                })

        # checkpoint write after each batch
        pd.DataFrame(fit_rows).to_csv(OUT_FIT_CSV, index=False)
        pd.DataFrame(prof_rows).to_csv(OUT_PROFILE_CSV, index=False)
        print(f"  [saved] fits -> {OUT_FIT_CSV}")
        print(f"  [saved] profiles -> {OUT_PROFILE_CSV}")

    # If we skipped all due to checkpoint, still plot from existing outputs
    if os.path.exists(OUT_PROFILE_CSV) and os.path.exists(OUT_FIT_CSV):
        df_fit = pd.read_csv(OUT_FIT_CSV)
        df_prof = pd.read_csv(OUT_PROFILE_CSV)
    else:
        df_fit = pd.DataFrame(fit_rows)
        df_prof = pd.DataFrame(prof_rows)

    # Plot: rows=batches, cols=params
    batches_present = [b for b in BATCHES if (df_fit["BatchID"].astype(str) == str(b)).any()]
    if len(batches_present) == 0:
        print("[done] No batches were profiled.")
        return

    fig, axes = plt.subplots(
        nrows=len(batches_present), ncols=4,
        figsize=(16, 3.2 * len(batches_present)),
        dpi=180,
        sharey=False
    )
    axes = np.atleast_2d(axes)

    for i, bid in enumerate(batches_present):
        row_fit = df_fit[df_fit["BatchID"].astype(str) == str(bid)].iloc[0]
        delta_star = float(row_fit["delta_star_95"])

        for j, pname in enumerate(pnames):
            ax = axes[i, j]
            sub = df_prof[(df_prof["BatchID"].astype(str) == str(bid)) & (df_prof["param"] == pname)].copy()
            sub = sub.sort_values("grid_value")

            x = sub["grid_value"].to_numpy(float)
            y = sub["delta_sse"].to_numpy(float)

            ax.plot(x, y, linewidth=1.8)
            ax.axhline(delta_star, linestyle="--", linewidth=1.2)

            ax.set_xscale("log")
            ax.set_xlabel(pname)
            ax.set_ylabel(r"$\Delta \mathrm{SSE}$" if j == 0 else "")
            ax.set_title(f"{bid} – profile({pname})" if i == 0 else "")

            # keep y lower bound at 0 for readability
            ax.set_ylim(bottom=0.0)

        # add a row label on the left side
        axes[i, 0].annotate(
            f"{bid}",
            xy=(-0.45, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )

    fig.suptitle("Profile SSE curves (log-x). Dashed line: approx 95% threshold per batch.", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure: {OUT_FIG_PNG}")
    print(f"Saved fits:   {OUT_FIT_CSV}")
    print(f"Saved проф:   {OUT_PROFILE_CSV}")


if __name__ == "__main__":
    main()