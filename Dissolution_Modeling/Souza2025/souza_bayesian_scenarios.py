#!/usr/bin/env python3
"""
souza_bayes_three_scenarios.py

Bayesian parameter estimation per BatchID for the n=1 model:

    df/dt = lam * (t/(tau+t)) * (1-f)^beta,   f(0)=0

We sample in log-space.

This script runs (configurable) THREE scenarios:

S1) "uniform_all"
    - estimate (lambda, tau, beta)
    - uniform priors in NATURAL space for all 3 params (within bounds)

S2) "tau_prior"
    - estimate (lambda, tau, beta)
    - lambda, beta: uniform priors in NATURAL space (within bounds)
    - tau: lognormal prior on log(tau) estimated from the observed curve using
           tau_prior_tools.compute_tau_lognormal_prior() (monotone smoothing + df/dt peak)
      If tau-prior estimation fails, fallback to uniform(tau) within bounds.

S3) "tau_fixed"
    - fix tau = TAU_FIXED_VALUE (must be within TAU_BOUNDS)
    - estimate (lambda, beta) with uniform priors in NATURAL space (within bounds)

Outputs (per scenario):
  - CSV: posterior summaries (median + 95% CI + 16/84%) per BatchID
  - Overview plot: median fit curve + data (grid of subplots) for batches fitted in THIS run

Final output (across scenarios):
  - scatter plot of estimated parameters:
      * subplot 1 (3D): Scenario 1 (log10 lambda, log10 tau, log10 beta)
      * subplot 2 (3D): Scenario 2 (log10 lambda, log10 tau, log10 beta)
      * subplot 3 (2D): Scenario 3 (log10 lambda, log10 beta) [tau fixed]

Notes:
- "log" for sampling is NATURAL LOG because we use np.log / np.exp.
- Corner plots are not generated here (you didn’t request them for this multi-scenario version).
- This is compute-heavy if you have many batches. Adjust N_BURN/N_STEPS/N_WALKERS as needed.

Requires:
  pip install emcee numpy pandas matplotlib scipy
  and for scenario 2 tau prior:
  tau_prior_tools.py available + (optional) cvxpy osqp (or scs) depending on your implementation
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from tau_prior_tools import compute_tau_lognormal_prior


# ============================================================
# USER SETTINGS
# ============================================================
CSV_PATH = "Souza2025_TableS1_Final.csv"
OUT_DIR  = "souza_bayes_three_scenarios"

BATCH_COL = "BatchID"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# --- which scenarios to run
RUN_SCENARIO_1_UNIFORM_ALL = True
RUN_SCENARIO_2_TAU_PRIOR   = True
RUN_SCENARIO_3_TAU_FIXED   = True

# Scenario 3 fixed tau value (minutes)
TAU_FIXED_VALUE = 1.0

# Plot settings (overview grid style)
MAKE_PLOTS = True
PLOT_MAX_PER_AX = 10
PLOT_LIMIT = None          # None = plot all batches fitted in THIS run for each scenario
DPI = 300

# MCMC settings
SEED = 42
N_WALKERS = 48
N_BURN    = 1500
N_STEPS   = 3000
THIN      = 2

# MLE init for centering walkers
USE_MLE_INIT = True
N_STARTS_MLE = 20
MAX_NFEV_MLE = 300

# Bounds in natural space
LAM_BOUNDS  = (1e-6, 1e2)
TAU_BOUNDS  = (1e-3, 1e4)
BETA_BOUNDS = (1e-3, 1e1)

# Measurement noise assumed in likelihood
SIGMA_MAIN = 0.03     # user-adjustable
SIGMA_T0   = 1e-3     # tiny but nonzero (if t==0 exists)
T0_ATOL    = 1e-12

# Tau prior estimation knobs (Scenario 2 only)
TAU_PRIOR_FACTOR = 1.0
TAU_PRIOR_LOGSIG = 0.50
TAU_PRIOR_LOGSIG_FLAT = 1.00
TAU_EST_SMOOTH_LAMBDA = 5.0
TAU_EST_DENSE = 4000
TAU_EST_PEAK_FRAC = 0.99
TAU_EST_EXCLUDE_BOUNDARIES = False
TAU_EST_BOUNDARY_EPS_FRAC = 1e-4
TAU_EST_BOUNDARY_EPS_ABS  = 1e-12
TAU_EST_FLAT_RATIO_THRESHOLD = 1.10
EARLY_EPS_ABS  = 1e-12
EARLY_EPS_FRAC = 1e-6

# Resume / checkpoint behavior (per-scenario CSV)
RESUME = True
SKIP_IF_SUCCESS_ONLY = True
CHECKPOINT_EVERY = 1

# Optional: save posterior samples per batch (per-scenario; can be big)
SAVE_SAMPLES = False


# ============================================================
# Helpers: preprocessing for likelihood (DO NOT add fake t=0)
# ============================================================
def preprocess_for_likelihood(t, y):
    """
    - sort by t
    - average duplicates in t
    - clip y to [0,1]
    Returns strictly increasing t.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    uniq_t, inv = np.unique(t, return_inverse=True)
    if uniq_t.size != t.size:
        y_sum = np.zeros_like(uniq_t, dtype=float)
        cnt = np.zeros_like(uniq_t, dtype=float)
        for i, g in enumerate(inv):
            y_sum[g] += y[i]
            cnt[g] += 1.0
        y = y_sum / np.maximum(cnt, 1.0)
        t = uniq_t

    y = np.clip(y, 0.0, 1.0)

    if t.size >= 2 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing after preprocessing.")
    return t, y


def sigma_vector(t):
    t = np.asarray(t, float)
    is_t0 = np.isclose(t, 0.0, atol=T0_ATOL, rtol=0.0)
    return np.where(is_t0, float(SIGMA_T0), float(SIGMA_MAIN))


# ============================================================
# Closed-form solver for n=1
# I(t)=∫0^t s/(tau+s) ds = t - tau*log(1+t/tau)
# ============================================================
def solve_curve_closed_form(t_eval, lam, tau, beta):
    t = np.asarray(t_eval, dtype=float)
    t = np.maximum(t, 0.0)

    lam = float(lam)
    tau = float(tau)
    beta = float(beta)

    tau_safe = max(tau, 1e-300)
    I = t - tau_safe * np.log1p(t / tau_safe)

    A = lam * I
    u = np.empty_like(t)

    if abs(beta - 1.0) < 1e-10:
        u[:] = np.exp(-A)
    else:
        base = 1.0 - (1.0 - beta) * A
        u[:] = 0.0
        pos = base > 0.0
        u[pos] = np.exp(np.log(base[pos]) / (1.0 - beta))

    f = 1.0 - u
    return np.clip(f, 0.0, 1.0)


# ============================================================
# Log-parameterization bounds
# ============================================================
def bounds_log_full():
    lb = np.log([LAM_BOUNDS[0], TAU_BOUNDS[0], BETA_BOUNDS[0]])
    ub = np.log([LAM_BOUNDS[1], TAU_BOUNDS[1], BETA_BOUNDS[1]])
    return lb, ub


def bounds_log_lam_beta():
    lb = np.log([LAM_BOUNDS[0], BETA_BOUNDS[0]])
    ub = np.log([LAM_BOUNDS[1], BETA_BOUNDS[1]])
    return lb, ub


def unpack_full(z3):
    z3 = np.asarray(z3, float)
    z3 = np.clip(z3, -60.0, 60.0)
    lam, tau, beta = np.exp(z3)
    return float(lam), float(tau), float(beta)


def unpack_lam_beta(z2, tau_fixed):
    z2 = np.asarray(z2, float)
    z2 = np.clip(z2, -60.0, 60.0)
    lam, beta = np.exp(z2)
    return float(lam), float(tau_fixed), float(beta)


# ============================================================
# Tau prior estimation (Scenario 2 only)
# ============================================================
def estimate_tau_prior_from_curve(t, y):
    """
    Returns (mu_tau, sig_tau, t_peak, t_peak_repr, diag_dict).

    mu_tau, sig_tau are for: log(tau) ~ Normal(mu_tau, sig_tau^2)
    where log is natural log.
    """
    res = compute_tau_lognormal_prior(
        t, y,
        tau_factor=TAU_PRIOR_FACTOR,
        tau_bounds=TAU_BOUNDS,
        smooth_lambda=TAU_EST_SMOOTH_LAMBDA,
        dense=TAU_EST_DENSE,
        peak_frac=TAU_EST_PEAK_FRAC,
        exclude_boundaries=TAU_EST_EXCLUDE_BOUNDARIES,
        boundary_eps_frac=TAU_EST_BOUNDARY_EPS_FRAC,
        boundary_eps_abs=TAU_EST_BOUNDARY_EPS_ABS,
        flat_ratio_threshold=TAU_EST_FLAT_RATIO_THRESHOLD,
        early_eps_abs=EARLY_EPS_ABS,
        early_eps_frac=EARLY_EPS_FRAC,
        logsig=TAU_PRIOR_LOGSIG,
        flat_logsig=TAU_PRIOR_LOGSIG_FLAT,
        include_arrays=True,
        t0_atol=T0_ATOL,
    )
    diag = res.diag or {}
    mu_tau = float(res.mu)
    sig_tau = float(res.sig)

    t_peak = float(diag.get("t_star", np.nan))
    t_peak_repr = float(diag.get("t_star_repr", np.nan))
    return mu_tau, sig_tau, t_peak, t_peak_repr, diag


# ============================================================
# Likelihoods
# ============================================================
def log_likelihood_full(z3, t, y):
    lam, tau, beta = unpack_full(z3)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    sig = sigma_vector(t)
    r = (y - y_pred) / sig
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sig) + np.log(2.0 * np.pi)))


def log_likelihood_tau_fixed(z2, t, y, tau_fixed):
    lam, tau, beta = unpack_lam_beta(z2, tau_fixed=tau_fixed)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    sig = sigma_vector(t)
    r = (y - y_pred) / sig
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sig) + np.log(2.0 * np.pi)))


# ============================================================
# Priors (log-space)
# ============================================================
def log_prior_uniform_all(z3):
    """
    Scenario 1:
      lam, tau, beta uniform in natural space within bounds.
      In log-space => +z0 + z1 + z2 (Jacobian) within bounds.
    """
    lb, ub = bounds_log_full()
    if np.any(z3 < lb) or np.any(z3 > ub):
        return -np.inf
    return float(np.sum(z3))


def log_prior_tau_prior(z3, mu_tau, sig_tau):
    """
    Scenario 2:
      lam, beta: uniform in natural space => +z0 + z2
      tau: lognormal on z1 if mu_tau,sig_tau finite; else uniform => +z1
    """
    lb, ub = bounds_log_full()
    if np.any(z3 < lb) or np.any(z3 > ub):
        return -np.inf

    lp = float(z3[0] + z3[2])

    if np.isfinite(mu_tau) and np.isfinite(sig_tau) and sig_tau > 0:
        lp += float(-0.5 * ((z3[1] - float(mu_tau)) / float(sig_tau)) ** 2)  # constants dropped
    else:
        lp += float(z3[1])  # uniform(tau) fallback

    return lp


def log_prior_tau_fixed(z2):
    """
    Scenario 3:
      lam, beta uniform in natural space within bounds => +z0 + z1 within bounds.
    """
    lb, ub = bounds_log_lam_beta()
    if np.any(z2 < lb) or np.any(z2 > ub):
        return -np.inf
    return float(z2[0] + z2[1])


# ============================================================
# Posterior wrappers
# ============================================================
def log_posterior_s1(z3, t, y):
    lp = log_prior_uniform_all(z3)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_full(z3, t, y)
    return lp + ll if np.isfinite(ll) else -np.inf


def log_posterior_s2(z3, t, y, mu_tau, sig_tau):
    lp = log_prior_tau_prior(z3, mu_tau=mu_tau, sig_tau=sig_tau)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_full(z3, t, y)
    return lp + ll if np.isfinite(ll) else -np.inf


def log_posterior_s3(z2, t, y, tau_fixed):
    lp = log_prior_tau_fixed(z2)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_tau_fixed(z2, t, y, tau_fixed=tau_fixed)
    return lp + ll if np.isfinite(ll) else -np.inf


# ============================================================
# MLE init
# ============================================================
def residuals_s1(z3, t, y_obs):
    lam, tau, beta = unpack_full(z3)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    return (y_pred - y_obs)


def residuals_s3(z2, t, y_obs, tau_fixed):
    lam, tau, beta = unpack_lam_beta(z2, tau_fixed=tau_fixed)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    return (y_pred - y_obs)


def fit_mle_s1(t, y, rng):
    lb, ub = bounds_log_full()
    best = None
    for _ in range(N_STARTS_MLE):
        z0 = rng.uniform(lb, ub)
        res = least_squares(
            fun=residuals_s1,
            x0=z0,
            bounds=(lb, ub),
            args=(t, y),
            method="trf",
            loss="linear",
            max_nfev=MAX_NFEV_MLE,
        )
        if best is None or res.cost < best.cost:
            best = res
    return best.x


def fit_mle_s3(t, y, rng, tau_fixed):
    lb, ub = bounds_log_lam_beta()
    best = None
    for _ in range(N_STARTS_MLE):
        z0 = rng.uniform(lb, ub)
        res = least_squares(
            fun=residuals_s3,
            x0=z0,
            bounds=(lb, ub),
            args=(t, y, tau_fixed),
            method="trf",
            loss="linear",
            max_nfev=MAX_NFEV_MLE,
        )
        if best is None or res.cost < best.cost:
            best = res
    return best.x


# ============================================================
# MCMC runners
# ============================================================
def run_mcmc_s1(t, y, rng):
    import emcee

    lb, ub = bounds_log_full()
    ndim = 3

    if USE_MLE_INIT:
        z_hat = fit_mle_s1(t, y, rng)
        p0 = z_hat[None, :] + 0.25 * rng.standard_normal(size=(N_WALKERS, ndim))
        p0 = np.minimum(np.maximum(p0, lb), ub)
    else:
        p0 = rng.uniform(lb, ub, size=(N_WALKERS, ndim))

    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, log_posterior_s1, args=(t, y))
    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, N_STEPS, progress=True)

    chain = sampler.get_chain(flat=True, thin=THIN)  # (S,3)
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, acc


def run_mcmc_s2(t, y, rng, mu_tau, sig_tau):
    import emcee

    lb, ub = bounds_log_full()
    ndim = 3

    if USE_MLE_INIT:
        z_hat = fit_mle_s1(t, y, rng)  # MLE ignores tau prior; OK as init
        p0 = z_hat[None, :] + 0.25 * rng.standard_normal(size=(N_WALKERS, ndim))
        p0 = np.minimum(np.maximum(p0, lb), ub)
    else:
        p0 = rng.uniform(lb, ub, size=(N_WALKERS, ndim))

    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, log_posterior_s2, args=(t, y, mu_tau, sig_tau))
    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, N_STEPS, progress=True)

    chain = sampler.get_chain(flat=True, thin=THIN)  # (S,3)
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, acc


def run_mcmc_s3(t, y, rng, tau_fixed):
    import emcee

    lb, ub = bounds_log_lam_beta()
    ndim = 2

    if USE_MLE_INIT:
        z_hat = fit_mle_s3(t, y, rng, tau_fixed=tau_fixed)
        p0 = z_hat[None, :] + 0.25 * rng.standard_normal(size=(N_WALKERS, ndim))
        p0 = np.minimum(np.maximum(p0, lb), ub)
    else:
        p0 = rng.uniform(lb, ub, size=(N_WALKERS, ndim))

    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, log_posterior_s3, args=(t, y, tau_fixed))
    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, N_STEPS, progress=True)

    chain = sampler.get_chain(flat=True, thin=THIN)  # (S,2)
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, acc


# ============================================================
# Summaries
# ============================================================
def summarize_chain_full(chain3):
    """
    chain3: (S,3) in log-space for (lam, tau, beta)
    Returns dict of medians and credible intervals in natural space.
    """
    theta = np.exp(chain3)
    lam = theta[:, 0]
    tau = theta[:, 1]
    beta = theta[:, 2]

    def q(x):
        return np.quantile(x, [0.025, 0.16, 0.50, 0.84, 0.975])

    lam_q = q(lam)
    tau_q = q(tau)
    beta_q = q(beta)

    return {
        "lambda_med": float(lam_q[2]),
        "lambda_q025": float(lam_q[0]),
        "lambda_q975": float(lam_q[4]),
        "lambda_q16": float(lam_q[1]),
        "lambda_q84": float(lam_q[3]),

        "tau_med": float(tau_q[2]),
        "tau_q025": float(tau_q[0]),
        "tau_q975": float(tau_q[4]),
        "tau_q16": float(tau_q[1]),
        "tau_q84": float(tau_q[3]),

        "beta_med": float(beta_q[2]),
        "beta_q025": float(beta_q[0]),
        "beta_q975": float(beta_q[4]),
        "beta_q16": float(beta_q[1]),
        "beta_q84": float(beta_q[3]),
    }


def summarize_chain_tau_fixed(chain2, tau_fixed):
    """
    chain2: (S,2) in log-space for (lam, beta), tau fixed.
    Returns dict shaped like full summary, with tau_* set to fixed.
    """
    theta = np.exp(chain2)
    lam = theta[:, 0]
    beta = theta[:, 1]

    def q(x):
        return np.quantile(x, [0.025, 0.16, 0.50, 0.84, 0.975])

    lam_q = q(lam)
    beta_q = q(beta)

    tf = float(tau_fixed)

    return {
        "lambda_med": float(lam_q[2]),
        "lambda_q025": float(lam_q[0]),
        "lambda_q975": float(lam_q[4]),
        "lambda_q16": float(lam_q[1]),
        "lambda_q84": float(lam_q[3]),

        "tau_med": tf,
        "tau_q025": tf,
        "tau_q975": tf,
        "tau_q16": tf,
        "tau_q84": tf,

        "beta_med": float(beta_q[2]),
        "beta_q025": float(beta_q[0]),
        "beta_q975": float(beta_q[4]),
        "beta_q16": float(beta_q[1]),
        "beta_q84": float(beta_q[3]),
    }


def rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ============================================================
# Plotting: overview grid style
# ============================================================
def plot_overview(curves, out_png, max_per_ax=10, title=""):
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
        dpi=DPI,
        sharex=False,
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

            ax.plot(t, y_pred, color=c, lw=2.0, alpha=0.95, label=bid)
            ax.plot(t, y, "o", color=c, ms=3.2, alpha=0.75, label="_nolegend_")

        ax.set_title(f"Batches {i0+1}–{i1} of {n}")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("release fraction")
        ax.legend(frameon=True, fontsize=8, ncol=1)

    for p in range(n_panels, len(axes)):
        axes[p].axis("off")

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Checkpointing (per scenario)
# ============================================================
def load_done_batchids(out_csv):
    if (not RESUME) or (not os.path.exists(out_csv)):
        return set()

    df_prev = pd.read_csv(out_csv)
    if df_prev.empty or (BATCH_COL not in df_prev.columns):
        return set()

    df_prev[BATCH_COL] = df_prev[BATCH_COL].astype(str)

    if SKIP_IF_SUCCESS_ONLY and ("success" in df_prev.columns):
        done = set(df_prev.loc[df_prev["success"] == True, BATCH_COL].astype(str))
    else:
        done = set(df_prev[BATCH_COL].astype(str))

    return done


def append_checkpoint(out_csv, new_rows):
    df_new = pd.DataFrame(new_rows)
    write_header = not os.path.exists(out_csv)
    df_new.to_csv(out_csv, mode="a", header=write_header, index=False)


# ============================================================
# Core: run one scenario
# ============================================================
def run_scenario(df_all, rng, *, scenario_key, scenario_mode, tau_fixed=None):
    """
    scenario_mode in {"uniform_all", "tau_prior", "tau_fixed"}
    Returns:
      - out_csv path
      - overview plot path (or None)
      - list of fitted medians for plotting comparison: [{"BatchID", "lambda_med","tau_med","beta_med"}]
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # per-scenario outputs
    scen_dir = os.path.join(OUT_DIR, scenario_key)
    os.makedirs(scen_dir, exist_ok=True)

    out_csv = os.path.join(scen_dir, f"souza_bayes_params_{scenario_key}.csv")
    out_png = os.path.join(scen_dir, f"souza_fit_overview_{scenario_key}.png")

    done_set = load_done_batchids(out_csv)
    if RESUME and done_set:
        print(f"[{scenario_key}] Resume enabled: skipping {len(done_set)} already-fitted BatchIDs from {out_csv}")

    new_rows_buffer = []
    curves_for_plot = []
    fitted_medians = []

    # batches
    df_all = df_all.copy()
    df_all[BATCH_COL] = df_all[BATCH_COL].astype(str)
    batch_ids = sorted(df_all[BATCH_COL].unique().tolist())

    if scenario_mode == "tau_fixed":
        if tau_fixed is None:
            raise ValueError("tau_fixed must be provided for scenario_mode='tau_fixed'")
        if not (TAU_BOUNDS[0] <= float(tau_fixed) <= TAU_BOUNDS[1]):
            raise ValueError(f"TAU_FIXED_VALUE={tau_fixed} must lie within TAU_BOUNDS={TAU_BOUNDS}")

    for bid in batch_ids:
        if RESUME and (bid in done_set):
            continue

        dfi = df_all[df_all[BATCH_COL] == str(bid)].sort_values(TIME_COL)
        t_raw = dfi[TIME_COL].to_numpy(float)
        y_raw = dfi[Y_COL].to_numpy(float)

        # preprocess for likelihood
        try:
            t, y = preprocess_for_likelihood(t_raw, y_raw)
        except Exception as e:
            print(f"[{scenario_key}][skip] {bid}: preprocess failed ({e})")
            continue

        if len(t) < 4:
            print(f"[{scenario_key}][skip] {bid}: too few points ({len(t)})")
            continue

        # scenario-specific tau prior
        mu_tau = np.nan
        sig_tau = np.nan
        t_peak = np.nan
        t_peak_repr = np.nan
        tau_diag = {}

        if scenario_mode == "tau_prior":
            try:
                mu_tau, sig_tau, t_peak, t_peak_repr, tau_diag = estimate_tau_prior_from_curve(t, y)
            except Exception as e:
                # fallback to uniform(tau)
                mu_tau, sig_tau = np.nan, np.nan
                tau_diag = {"prior_mode": "failed", "error": str(e)}

        print(f"\n[{scenario_key}][MCMC] {bid}  points={len(t)}")

        # run MCMC
        try:
            if scenario_mode == "uniform_all":
                chain, acc = run_mcmc_s1(t, y, rng)
                summ = summarize_chain_full(chain)
                lam_med, tau_med, beta_med = summ["lambda_med"], summ["tau_med"], summ["beta_med"]

                if SAVE_SAMPLES:
                    theta = np.exp(chain)
                    sdf = pd.DataFrame(theta, columns=["lambda", "tau", "beta"])
                    sdf["log_lambda"] = chain[:, 0]
                    sdf["log_tau"]    = chain[:, 1]
                    sdf["log_beta"]   = chain[:, 2]
                    sdf["BatchID"]    = str(bid)
                    sdf["accept_frac"] = float(acc)
                    sdf.to_csv(os.path.join(scen_dir, f"posterior_samples_{bid}.csv"), index=False)

            elif scenario_mode == "tau_prior":
                chain, acc = run_mcmc_s2(t, y, rng, mu_tau=mu_tau, sig_tau=sig_tau)
                summ = summarize_chain_full(chain)
                lam_med, tau_med, beta_med = summ["lambda_med"], summ["tau_med"], summ["beta_med"]

                if SAVE_SAMPLES:
                    theta = np.exp(chain)
                    sdf = pd.DataFrame(theta, columns=["lambda", "tau", "beta"])
                    sdf["log_lambda"] = chain[:, 0]
                    sdf["log_tau"]    = chain[:, 1]
                    sdf["log_beta"]   = chain[:, 2]
                    sdf["BatchID"]    = str(bid)
                    sdf["accept_frac"] = float(acc)
                    sdf["mu_tau_prior"] = float(mu_tau) if np.isfinite(mu_tau) else np.nan
                    sdf["sig_tau_prior"] = float(sig_tau) if np.isfinite(sig_tau) else np.nan
                    sdf.to_csv(os.path.join(scen_dir, f"posterior_samples_{bid}.csv"), index=False)

            elif scenario_mode == "tau_fixed":
                chain, acc = run_mcmc_s3(t, y, rng, tau_fixed=float(tau_fixed))
                summ = summarize_chain_tau_fixed(chain, tau_fixed=float(tau_fixed))
                lam_med, tau_med, beta_med = summ["lambda_med"], summ["tau_med"], summ["beta_med"]

                if SAVE_SAMPLES:
                    theta = np.exp(chain)
                    sdf = pd.DataFrame(theta, columns=["lambda", "beta"])
                    sdf["log_lambda"] = chain[:, 0]
                    sdf["log_beta"]   = chain[:, 1]
                    sdf["tau_fixed"]  = float(tau_fixed)
                    sdf["BatchID"]    = str(bid)
                    sdf["accept_frac"] = float(acc)
                    sdf.to_csv(os.path.join(scen_dir, f"posterior_samples_{bid}.csv"), index=False)

            else:
                raise ValueError(f"Unknown scenario_mode={scenario_mode}")

        except Exception as e:
            print(f"[{scenario_key}][fail] {bid}: MCMC failed ({e})")
            row = {
                "Scenario": scenario_key,
                "BatchID": str(bid),
                "n_points": int(len(t)),
                "t_min": float(np.min(t)),
                "t_max": float(np.max(t)),
                "success": False,
                "error": str(e),
            }
            new_rows_buffer.append(row)
            if len(new_rows_buffer) >= CHECKPOINT_EVERY:
                append_checkpoint(out_csv, new_rows_buffer)
                new_rows_buffer = []
            continue

        # median fit + rmse on observed points
        y_pred_med = solve_curve_closed_form(t, lam_med, tau_med, beta_med)
        r = rmse(y_pred_med, y)

        # store row (scenario-labeled)
        row = {
            "Scenario": scenario_key,
            "BatchID": str(bid),
            "n_points": int(len(t)),
            "t_min": float(np.min(t)),
            "t_max": float(np.max(t)),
            "accept_frac": float(acc),
            "rmse_med_curve": float(r),
            "success": True,

            # Scenario 2 diagnostics (kept as NaN/empty for other scenarios)
            "tau_prior_mode": str(tau_diag.get("prior_mode", "")) if isinstance(tau_diag, dict) else "",
            "t_peak": float(t_peak) if np.isfinite(t_peak) else np.nan,
            "t_peak_repr": float(t_peak_repr) if np.isfinite(t_peak_repr) else np.nan,
            "mu_tau_prior": float(mu_tau) if np.isfinite(mu_tau) else np.nan,
            "sig_tau_prior": float(sig_tau) if np.isfinite(sig_tau) else np.nan,
            "flat_ratio": float(tau_diag.get("flat_ratio", np.nan)) if isinstance(tau_diag, dict) else np.nan,

            # Scenario 3 fixed tau (NaN for other scenarios)
            "tau_fixed": float(tau_fixed) if (scenario_mode == "tau_fixed") else np.nan,

            # posterior summaries
            **summ,
        }
        new_rows_buffer.append(row)

        # collect medians for final scatter
        fitted_medians.append({
            "Scenario": scenario_key,
            "BatchID": str(bid),
            "lambda_med": float(lam_med),
            "tau_med": float(tau_med),
            "beta_med": float(beta_med),
        })

        if MAKE_PLOTS:
            curves_for_plot.append({"bid": bid, "t": t, "y": y, "y_pred": y_pred_med})

        print(f"[{scenario_key}][done] {bid}: rmse={r:.4f} acc~{acc:.3f}  "
              f"lam~{lam_med:.3g} tau~{tau_med:.3g} beta~{beta_med:.3g}")

        # checkpoint write
        if len(new_rows_buffer) >= CHECKPOINT_EVERY:
            append_checkpoint(out_csv, new_rows_buffer)
            new_rows_buffer = []

    if new_rows_buffer:
        append_checkpoint(out_csv, new_rows_buffer)

    print(f"\n[{scenario_key}] Saved/updated: {out_csv}")

    # Plot (only batches fitted in THIS run, same behavior as your original)
    plot_path = None
    if MAKE_PLOTS:
        curves = curves_for_plot
        if PLOT_LIMIT is not None:
            curves = curves[:int(PLOT_LIMIT)]

        if len(curves) == 0:
            print(f"[{scenario_key}] No new fitted batches to plot.")
        else:
            if scenario_mode == "uniform_all":
                title = "Scenario 1: Uniform priors for $(\\lambda,\\tau,\\beta)$ (line = posterior median, markers = data)"
            elif scenario_mode == "tau_prior":
                title = "Scenario 2: Uniform $(\\lambda,\\beta)$ + spline-estimated lognormal prior on $\\tau$ (median fit)"
            else:
                title = f"Scenario 3: Fixed $\\tau={float(tau_fixed):g}$, estimate $(\\lambda,\\beta)$ (median fit)"

            plot_overview(curves, out_png, max_per_ax=PLOT_MAX_PER_AX, title=title)
            plot_path = out_png
            print(f"[{scenario_key}] Saved overview plot: {out_png}")

    return out_csv, plot_path, fitted_medians


# ============================================================
# Final comparison scatter plot
# ============================================================
def safe_log10(x):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan, dtype=float)
    pos = x > 0
    out[pos] = np.log10(x[pos])
    return out


def plot_parameter_scatter_all_scenarios(df_s1, df_s2, df_s3, out_png):
    """
    df_s1, df_s2: columns lambda_med, tau_med, beta_med
    df_s3: columns lambda_med, beta_med (tau fixed in df_s3['tau_fixed'] or tau_med constant)
    """
    fig = plt.figure(figsize=(15.5, 5.2), dpi=DPI)

    # --- subplot 1: Scenario 1 (3D)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    x1 = safe_log10(df_s1["lambda_med"].to_numpy(float))
    y1 = safe_log10(df_s1["tau_med"].to_numpy(float))
    z1 = safe_log10(df_s1["beta_med"].to_numpy(float))
    ax1.scatter(x1, y1, z1, s=18, alpha=0.85)
    ax1.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{med}})$")
    ax1.set_ylabel(r"$\log_{10}(\tau_{\mathrm{med}})$")
    ax1.set_zlabel(r"$\log_{10}(\beta_{\mathrm{med}})$")
    ax1.set_title("Scenario 1: uniform $(\\lambda,\\tau,\\beta)$")

    # --- subplot 2: Scenario 2 (3D)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    x2 = safe_log10(df_s2["lambda_med"].to_numpy(float))
    y2 = safe_log10(df_s2["tau_med"].to_numpy(float))
    z2 = safe_log10(df_s2["beta_med"].to_numpy(float))
    ax2.scatter(x2, y2, z2, s=18, alpha=0.85)
    ax2.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{med}})$")
    ax2.set_ylabel(r"$\log_{10}(\tau_{\mathrm{med}})$")
    ax2.set_zlabel(r"$\log_{10}(\beta_{\mathrm{med}})$")
    ax2.set_title("Scenario 2: spline prior on $\\tau$")

    # --- subplot 3: Scenario 3 (2D)
    ax3 = fig.add_subplot(1, 3, 3)
    x3 = safe_log10(df_s3["lambda_med"].to_numpy(float))
    y3 = safe_log10(df_s3["beta_med"].to_numpy(float))
    ax3.scatter(x3, y3, s=18, alpha=0.85)
    ax3.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{med}})$")
    ax3.set_ylabel(r"$\log_{10}(\beta_{\mathrm{med}})$")

    tau_fixed_vals = df_s3.get("tau_fixed", pd.Series(dtype=float))
    tau_fixed_val = float(np.nanmedian(tau_fixed_vals.to_numpy(float))) if len(tau_fixed_vals) else np.nan
    ttl = "Scenario 3: fixed $\\tau$"
    if np.isfinite(tau_fixed_val):
        ttl += rf" ($\tau={tau_fixed_val:g}$)"
    ax3.set_title(ttl)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    rng = np.random.default_rng(SEED)

    scenarios = []
    if RUN_SCENARIO_1_UNIFORM_ALL:
        scenarios.append(("S1_uniform_all", "uniform_all", None))
    if RUN_SCENARIO_2_TAU_PRIOR:
        scenarios.append(("S2_tau_prior", "tau_prior", None))
    if RUN_SCENARIO_3_TAU_FIXED:
        scenarios.append((f"S3_tau_fixed_{TAU_FIXED_VALUE:g}", "tau_fixed", float(TAU_FIXED_VALUE)))

    # Run selected scenarios
    scenario_csvs = {}
    for scen_key, scen_mode, tau_fixed in scenarios:
        print("\n" + "=" * 80)
        if scen_mode == "tau_fixed":
            print(f"Running {scen_key}: tau fixed at {tau_fixed:g}")
        else:
            print(f"Running {scen_key}")
        print("=" * 80)

        out_csv, out_plot, _ = run_scenario(
            df, rng,
            scenario_key=scen_key,
            scenario_mode=scen_mode,
            tau_fixed=tau_fixed,
        )
        scenario_csvs[scen_key] = out_csv

    # Final comparison scatter plot (needs all three scenarios)
    if RUN_SCENARIO_1_UNIFORM_ALL and RUN_SCENARIO_2_TAU_PRIOR and RUN_SCENARIO_3_TAU_FIXED:
        s1_key = "S1_uniform_all"
        s2_key = "S2_tau_prior"
        s3_key = f"S3_tau_fixed_{TAU_FIXED_VALUE:g}"

        df_s1 = pd.read_csv(scenario_csvs[s1_key])
        df_s2 = pd.read_csv(scenario_csvs[s2_key])
        df_s3 = pd.read_csv(scenario_csvs[s3_key])

        # Keep only successful rows
        df_s1 = df_s1[df_s1.get("success", True) == True].copy()
        df_s2 = df_s2[df_s2.get("success", True) == True].copy()
        df_s3 = df_s3[df_s3.get("success", True) == True].copy()

        # Ensure required columns exist (guard)
        need_cols_full = {"lambda_med", "tau_med", "beta_med"}
        need_cols_2d = {"lambda_med", "beta_med"}
        if not need_cols_full.issubset(df_s1.columns) or not need_cols_full.issubset(df_s2.columns) or not need_cols_2d.issubset(df_s3.columns):
            print("[warn] Missing expected columns for final scatter plot; skipping.")
        else:
            out_cmp = os.path.join(OUT_DIR, "souza_parameter_scatter_three_scenarios.png")
            plot_parameter_scatter_all_scenarios(df_s1, df_s2, df_s3, out_cmp)
            print(f"\nSaved final parameter scatter: {out_cmp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
