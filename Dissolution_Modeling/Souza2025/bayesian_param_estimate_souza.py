#!/usr/bin/env python3
"""
souza_bayesian_nfix1.py

Bayesian parameter estimation per BatchID for the n=1 model:

    df/dt = lam * (t/(tau+t)) * (1-f)^beta,   f(0)=0

We sample in log-space:
    z = [log lam, log tau, log beta]

Priors:
  - lam  ~ Uniform(LAM_BOUNDS) in natural space -> log-prior adds +z0 within bounds
  - beta ~ Uniform(BETA_BOUNDS) in natural space -> log-prior adds +z2 within bounds
  - tau  prior is lognormal in log-space:
        log(tau) ~ Normal(mu_tau, sig_tau^2), truncated to TAU_BOUNDS
    where (mu_tau, sig_tau) are estimated from the observed curve using
    tau_prior_tools.compute_tau_lognormal_prior() (monotone smoothing + PCHIP df/dt peak).

Likelihood:
  Gaussian with heteroscedastic SD:
    sigma(t)=SIGMA_T0 if t==0 else SIGMA_MAIN

Outputs:
  - OUT_CSV: per-batch posterior summaries (median + CI) + t_peak estimate
  - FIT_OVERVIEW_PNG: overview plot (median fit curve + data), many subplots
  - (optional) FIT_BANDS_PNG: overview plot with 95% credible bands (subset can be limited)
  - (optional) posterior_samples_<BatchID>.csv if SAVE_SAMPLES=True

Requires:
  pip install emcee numpy pandas matplotlib scipy
  (optional for QP smoothing in tau_prior_tools) pip install cvxpy osqp (or scs)
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
OUT_DIR  = "souza_bayes_nfix1"
OUT_CSV  = os.path.join(OUT_DIR, "souza_bayes_params_per_batch.csv")

BATCH_COL = "BatchID"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# Plot settings (same “overview grid” style as your MAP script)
MAKE_PLOTS = True
FIT_OVERVIEW_PNG = os.path.join(OUT_DIR, "souza_fit_overview_bayes_median.png")
PLOT_MAX_PER_AX = 10
PLOT_LIMIT = None          # None = plot all batches fitted in THIS run
DPI = 300

# Optional: uncertainty-band overview (can be heavy if many batches)
MAKE_BAND_PLOTS = False
FIT_BANDS_PNG = os.path.join(OUT_DIR, "souza_fit_overview_bayes_bands.png")
BAND_PLOT_LIMIT = 30       # plot only first N batches (for speed / clarity)
N_DRAWS_BANDS = 400        # posterior draws per batch for band plot
T_DENSE = 250              # dense time grid points per batch for band plot

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
TAU_BOUNDS  = (1e-6, 1e4)
BETA_BOUNDS = (1e-4, 1e2)

# Measurement noise assumed in likelihood
SIGMA_MAIN = 0.03     # user-adjustable
SIGMA_T0   = 1e-3     # tiny but nonzero (if t==0 exists)
T0_ATOL    = 1e-12

# Tau prior estimation knobs (passed to compute_tau_lognormal_prior)
USE_TAU_PRIOR = True
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

# Resume / checkpoint behavior
RESUME = True
SKIP_IF_SUCCESS_ONLY = True
CHECKPOINT_EVERY = 1

# Optional: save posterior samples per batch (large disk if many batches)
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
# Log-parameterization
# ============================================================
def bounds_log():
    lb = np.log([LAM_BOUNDS[0], TAU_BOUNDS[0], BETA_BOUNDS[0]])
    ub = np.log([LAM_BOUNDS[1], TAU_BOUNDS[1], BETA_BOUNDS[1]])
    return lb, ub


def unpack(z):
    z = np.asarray(z, float)
    z = np.clip(z, -60.0, 60.0)
    lam, tau, beta = np.exp(z)
    return float(lam), float(tau), float(beta)


# ============================================================
# Tau prior estimation (per batch)
# ============================================================
def estimate_tau_prior_from_curve(t, y):
    """
    Returns (mu_tau, sig_tau, t_peak, t_peak_repr, diag_dict).

    We use tau_prior_tools, which internally preprocesses and enforces t=0 for smoothing,
    but that does NOT alter the likelihood data (we keep them separate).
    """
    if not USE_TAU_PRIOR:
        return np.nan, np.nan, np.nan, np.nan, {"prior_mode": "disabled"}

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

    # "t_peak" we report as the peak time estimate from the spline workflow
    # Use diag["t_star"] as the raw peak and diag["t_star_repr"] as the representative value.
    t_peak = float(diag.get("t_star", np.nan))
    t_peak_repr = float(diag.get("t_star_repr", np.nan))

    return mu_tau, sig_tau, t_peak, t_peak_repr, diag


# ============================================================
# Prior / Likelihood / Posterior
# ============================================================
def log_prior(z, *, mu_tau, sig_tau):
    """
    lam, beta: Uniform in natural space => +z0 + z2 within bounds
    tau: lognormal prior on z1 if mu_tau,sig_tau finite; otherwise uniform => +z1
    """
    lb, ub = bounds_log()
    if np.any(z < lb) or np.any(z > ub):
        return -np.inf

    lp = float(z[0] + z[2])  # jacobians

    if np.isfinite(mu_tau) and np.isfinite(sig_tau) and sig_tau > 0:
        lp += float(-0.5 * ((z[1] - mu_tau) / sig_tau) ** 2)  # constants dropped
    else:
        lp += float(z[1])  # uniform(tau) fallback

    return lp


def log_likelihood(z, t, y):
    lam, tau, beta = unpack(z)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    sig = sigma_vector(t)
    r = (y - y_pred) / sig
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sig) + np.log(2.0 * np.pi)))


def log_posterior(z, t, y, mu_tau, sig_tau):
    lp = log_prior(z, mu_tau=mu_tau, sig_tau=sig_tau)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(z, t, y)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ============================================================
# MLE init (least squares on curve)
# ============================================================
def residuals_log(z, t, y_obs):
    lam, tau, beta = unpack(z)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    return (y_pred - y_obs)


def fit_mle_log(t, y, rng):
    lb, ub = bounds_log()
    best = None
    for _ in range(N_STARTS_MLE):
        z0 = rng.uniform(lb, ub)
        res = least_squares(
            fun=residuals_log,
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


# ============================================================
# MCMC per batch
# ============================================================
def run_mcmc_for_batch(t, y, rng, *, batch_id, mu_tau, sig_tau):
    import emcee

    lb, ub = bounds_log()
    ndim = 3

    if USE_MLE_INIT:
        z_hat = fit_mle_log(t, y, rng)
        p0 = z_hat[None, :] + 0.25 * rng.standard_normal(size=(N_WALKERS, ndim))
        p0 = np.minimum(np.maximum(p0, lb), ub)
    else:
        p0 = rng.uniform(lb, ub, size=(N_WALKERS, ndim))

    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim,
        log_posterior,
        args=(t, y, mu_tau, sig_tau),
    )

    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, N_STEPS, progress=True)

    chain = sampler.get_chain(flat=True, thin=THIN)  # shape (n_samples, 3)
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, acc


# ============================================================
# Summaries
# ============================================================
def summarize_posterior(chain):
    """
    chain: (S,3) in log-space
    Returns dict of medians and credible intervals in natural space.
    """
    theta = np.exp(chain)  # (S,3)
    lam = theta[:, 0]
    tau = theta[:, 1]
    beta = theta[:, 2]

    def q(x, qs):
        return np.quantile(x, qs)

    lam_q = q(lam, [0.025, 0.16, 0.50, 0.84, 0.975])
    tau_q = q(tau, [0.025, 0.16, 0.50, 0.84, 0.975])
    beta_q = q(beta, [0.025, 0.16, 0.50, 0.84, 0.975])

    out = {
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
    return out


def rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ============================================================
# Plotting: same overview grid style
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


def plot_overview_with_bands(curves, out_png, max_per_ax=10, title=""):
    """
    curves: list of dicts with keys:
      bid, t, y, t_dense, q_lo, q_med, q_hi
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
            td = item["t_dense"]
            q_lo = item["q_lo"]
            q_med = item["q_med"]
            q_hi = item["q_hi"]

            ax.fill_between(td, q_lo, q_hi, color=c, alpha=0.18)
            ax.plot(td, q_med, color=c, lw=2.0, alpha=0.95, label=bid)
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
# Checkpointing
# ============================================================
def load_done_batchids(out_csv):
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
    df_new = pd.DataFrame(new_rows)
    write_header = not os.path.exists(out_csv)
    df_new.to_csv(out_csv, mode="a", header=write_header, index=False)


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    rng = np.random.default_rng(SEED)

    df[BATCH_COL] = df[BATCH_COL].astype(str)
    batch_ids = sorted(df[BATCH_COL].unique().tolist())

    done_set, df_prev = load_done_batchids(OUT_CSV)
    if RESUME and done_set:
        print(f"Resume enabled: skipping {len(done_set)} already-fitted BatchIDs from {OUT_CSV}")

    new_rows_buffer = []
    curves_for_plot = []
    curves_for_bands = []

    for bid in batch_ids:
        if RESUME and (bid in done_set):
            continue

        dfi = df[df[BATCH_COL] == str(bid)].sort_values(TIME_COL)
        t_raw = dfi[TIME_COL].to_numpy(float)
        y_raw = dfi[Y_COL].to_numpy(float)

        # preprocess for likelihood (no synthetic t=0 insertion)
        try:
            t, y = preprocess_for_likelihood(t_raw, y_raw)
        except Exception as e:
            print(f"[skip] {bid}: preprocess failed ({e})")
            continue

        if len(t) < 4:
            print(f"[skip] {bid}: too few points ({len(t)})")
            continue

        # tau prior (uses its own preprocessing internally)
        mu_tau, sig_tau, t_peak, t_peak_repr, tau_diag = estimate_tau_prior_from_curve(t, y)

        print(f"\n[MCMC] {bid}  points={len(t)}  tau_prior_mode={tau_diag.get('prior_mode','')}")
        chain, acc = run_mcmc_for_batch(t, y, rng, batch_id=bid, mu_tau=mu_tau, sig_tau=sig_tau)

        summ = summarize_posterior(chain)
        lam_med = summ["lambda_med"]
        tau_med = summ["tau_med"]
        beta_med = summ["beta_med"]

        y_pred_med = solve_curve_closed_form(t, lam_med, tau_med, beta_med)
        r = rmse(y_pred_med, y)

        # save optional samples
        if SAVE_SAMPLES:
            theta = np.exp(chain)
            sdf = pd.DataFrame(theta, columns=["lambda", "tau", "beta"])
            sdf["log_lambda"] = chain[:, 0]
            sdf["log_tau"]    = chain[:, 1]
            sdf["log_beta"]   = chain[:, 2]
            sdf["BatchID"]    = str(bid)
            sdf["accept_frac"] = float(acc)
            sdf.to_csv(os.path.join(OUT_DIR, f"posterior_samples_{bid}.csv"), index=False)

        row = {
            "BatchID": str(bid),
            "n_points": int(len(t)),
            "t_min": float(np.min(t)),
            "t_max": float(np.max(t)),

            "accept_frac": float(acc),
            "rmse_med_curve": float(r),
            "success": True,

            # tau prior diagnostics
            "tau_prior_mode": str(tau_diag.get("prior_mode", "")),
            "t_peak": float(t_peak) if np.isfinite(t_peak) else np.nan,
            "t_peak_repr": float(t_peak_repr) if np.isfinite(t_peak_repr) else np.nan,
            "mu_tau_prior": float(mu_tau) if np.isfinite(mu_tau) else np.nan,
            "sig_tau_prior": float(sig_tau) if np.isfinite(sig_tau) else np.nan,
            "flat_ratio": float(tau_diag.get("flat_ratio", np.nan)),

            # posterior summaries
            **summ,
        }
        new_rows_buffer.append(row)

        if MAKE_PLOTS:
            curves_for_plot.append({"bid": bid, "t": t, "y": y, "y_pred": y_pred_med})

        if MAKE_BAND_PLOTS and (len(curves_for_bands) < int(BAND_PLOT_LIMIT)):
            # posterior predictive band on a dense grid
            t_dense = np.linspace(0.0, float(np.max(t)), T_DENSE)

            n_draws = min(N_DRAWS_BANDS, chain.shape[0])
            idx = np.random.default_rng(SEED + 123).choice(chain.shape[0], size=n_draws, replace=False)
            sub = chain[idx]
            thetas = np.exp(sub)

            Ypred = np.empty((n_draws, t_dense.size), dtype=float)
            for k in range(n_draws):
                Ypred[k, :] = solve_curve_closed_form(t_dense, thetas[k, 0], thetas[k, 1], thetas[k, 2])

            q_lo, q_med, q_hi = np.quantile(Ypred, [0.025, 0.5, 0.975], axis=0)

            curves_for_bands.append({
                "bid": bid,
                "t": t,
                "y": y,
                "t_dense": t_dense,
                "q_lo": q_lo,
                "q_med": q_med,
                "q_hi": q_hi,
            })

        print(f"[done] {bid}: rmse={r:.4f} acc~{acc:.3f}  "
              f"lam~{lam_med:.3g} tau~{tau_med:.3g} beta~{beta_med:.3g}  "
              f"t_peak_repr~{t_peak_repr:.3g}")

        # checkpoint write
        if len(new_rows_buffer) >= CHECKPOINT_EVERY:
            append_checkpoint(OUT_CSV, new_rows_buffer)
            new_rows_buffer = []

    if new_rows_buffer:
        append_checkpoint(OUT_CSV, new_rows_buffer)

    print(f"\nSaved/updated: {OUT_CSV}")

    # Plots (only for batches fitted in THIS run, matching your prior behavior)
    if MAKE_PLOTS:
        curves = curves_for_plot
        if PLOT_LIMIT is not None:
            curves = curves[:int(PLOT_LIMIT)]

        if len(curves) == 0:
            print("No new fitted batches to plot.")
        else:
            plot_overview(
                curves,
                FIT_OVERVIEW_PNG,
                max_per_ax=PLOT_MAX_PER_AX,
                title="Bayesian fits per BatchID (line = posterior median, markers = data)",
            )
            print(f"Saved overview plot: {FIT_OVERVIEW_PNG}")

    if MAKE_BAND_PLOTS:
        curves = curves_for_bands
        if len(curves) == 0:
            print("No batches available for band plot (or BAND_PLOT_LIMIT=0).")
        else:
            plot_overview_with_bands(
                curves,
                FIT_BANDS_PNG,
                max_per_ax=PLOT_MAX_PER_AX,
                title=f"Bayesian fits with 95% bands (subset up to {BAND_PLOT_LIMIT} batches)",
            )
            print(f"Saved band overview plot: {FIT_BANDS_PNG}")


if __name__ == "__main__":
    main()