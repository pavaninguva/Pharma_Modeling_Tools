#!/usr/bin/env python3
"""
Full Bayesian sampling (per selected BatchID) for the n-fixed ODE model (n=1),
with UNIFORM priors on the NATURAL parameters (lambda, tau, beta) within bounds,
and a FIXED, HETEROSCEDASTIC measurement SD:

Option B:
  - SD at t=0 is tiny (SIGMA_T0)
  - SD at t>0 is user-chosen (SIGMA_MAIN, e.g. 0.03)

Model:
  df/dt = lam * (t/(tau+t)) * (1-f)^beta ,  f(0)=0

We sample in log-space z = [log lam, log tau, log beta] for stability.
Uniform(theta) in natural space implies log-prior in z-space includes Jacobian:
  log p(z) ∝ sum(z)   within bounds.

Likelihood (Gaussian with known SD per timepoint):
  y_i ~ Normal(f(t_i;theta), sigma_i^2)
  logL = -0.5 * sum( ((y - f)/sigma)^2 + 2*log(sigma) + log(2*pi) )

Outputs (saved in OUT_DIR):
1) prior_vs_posterior_grid.png
   - k rows (each BatchID), 3 cols (lambda, tau, beta), plotted in log10-space
   - overlays: prior density in log10(theta) + posterior KDE

2) corner_grid.png
   - uses the `corner` module
   - creates one corner plot per batch and then stacks them into one image

3) fit_with_bands.png
   - per batch: data points + posterior median curve + 95% credible band
   - band reflects PARAMETER uncertainty (not additional random observation draws)

Also saves posterior samples per batch to:
   OUT_DIR/posterior_samples_<BatchID>.csv

Requires:
  pip install emcee corner scipy numpy pandas matplotlib
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.optimize import least_squares
from scipy.stats import gaussian_kde


# ============================================================
# USER SETTINGS
# ============================================================
CSV_PATH = "Souza2025_TableS1_Final.csv"

BATCH_COL = "BatchID"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# Which batches to sample
BATCHES = ["F14", "F11", "F25", "F37", "F54"]

# Output
OUT_DIR = "bayes_selected_nfix1_uniform_fixedsigma"
OUT_PRIOR_POST_PNG = os.path.join(OUT_DIR, "prior_vs_posterior_grid.png")
OUT_CORNER_PNG     = os.path.join(OUT_DIR, "corner_grid.png")
OUT_BANDS_PNG      = os.path.join(OUT_DIR, "fit_with_bands.png")

# Reproducibility
SEED = 42

# Fixed exponent
N_FIXED = 1.0

# Parameter bounds (natural space) — UNIFORM priors on these intervals
LAM_BOUNDS  = (1e-5, 1e5)
TAU_BOUNDS  = (1e-5, 1e3)
BETA_BOUNDS = (1e-4, 1e1)

# -------- Measurement noise (OPTION B) --------
# SD for t > 0 points (change this as needed)
SIGMA_MAIN = 0.03

# SD for t == 0 point (very certain). Keep small but not zero.
SIGMA_T0 = 1e-3

# Tolerance for "is t=0?" checks (handles floating point CSV artifacts)
T0_ATOL = 1e-12

# MCMC (emcee)
N_WALKERS = 48
N_BURN    = 1500
N_STEPS   = 3000
THIN      = 2

# Initialization (since uniform prior => MAP approx MLE, useful for centering walkers)
USE_MLE_INIT = True
N_STARTS_MLE = 20
MAX_NFEV_MLE = 250

# For uncertainty bands
N_DRAWS_BANDS = 600
T_DENSE = 250

# Corner plot settings
# Limit how many posterior samples we send to corner() to keep plots fast/clean
MAX_CORNER_SAMPLES = 5000
CORNER_LEVELS = (0.68, 0.95)  # contour probability levels
CORNER_QUANTILES = (0.16, 0.50, 0.84)  # for titles + vertical lines

# Plot settings
USE_TEX = True  # set True if you have a LaTeX install and want TeX rendering
DPI = 180


# ============================================================
# Matplotlib settings
# ============================================================
if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
    })


# ============================================================
# Closed-form model evaluation for n=1
# df/dt = lam * (t/(tau+t)) * (1-f)^beta, f(0)=0
# I(t) = ∫0^t s/(tau+s) ds = t - tau*log(1 + t/tau)
# ============================================================
def solve_curve_closed_form(t_eval, lam, tau, beta):
    t = np.asarray(t_eval, dtype=float)
    t = np.maximum(t, 0.0)

    tau = float(tau)
    lam = float(lam)
    beta = float(beta)

    tau_safe = max(tau, 1e-300)
    I = t - tau_safe * np.log1p(t / tau_safe)

    A = lam * I
    u = np.empty_like(t)

    if abs(beta - 1.0) < 1e-8:
        u[:] = np.exp(-A)
    else:
        base = 1.0 - (1.0 - beta) * A
        u[:] = 0.0
        pos = base > 0.0
        u[pos] = np.exp(np.log(base[pos]) / (1.0 - beta))

    f = 1.0 - u
    return np.clip(f, 0.0, 1.0)


# ============================================================
# Log-parameterization for sampling
# z = [log lam, log tau, log beta]
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

def residuals_log(z, t, y_obs):
    lam, tau, beta = unpack(z)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    return (y_pred - y_obs)


# ============================================================
# Heteroscedastic sigma(t): tiny at t=0, SIGMA_MAIN otherwise
# ============================================================
def sigma_vector(t):
    t = np.asarray(t, float)
    is_t0 = np.isclose(t, 0.0, atol=T0_ATOL, rtol=0.0)
    sig = np.where(is_t0, float(SIGMA_T0), float(SIGMA_MAIN))
    return sig


# ============================================================
# Bayesian pieces (uniform priors in NATURAL space)
# ============================================================
def log_prior(z):
    lb, ub = bounds_log()
    if np.any(z < lb) or np.any(z > ub):
        return -np.inf
    # Uniform(theta) prior in natural space => include Jacobian in z-space:
    # p(z) ∝ exp(z1+z2+z3)
    return float(np.sum(z))

def log_likelihood_fixedsigma(z, t, y):
    lam, tau, beta = unpack(z)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)

    sig = sigma_vector(t)
    r = (y - y_pred) / sig

    ll = -0.5 * np.sum(r * r + 2.0 * np.log(sig) + np.log(2.0 * np.pi))
    return float(ll)

def log_posterior(z, t, y):
    lp = log_prior(z)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_fixedsigma(z, t, y)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ============================================================
# Optional: MLE init (for centering walkers)
# ============================================================
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
        if (best is None) or (res.cost < best.cost):
            best = res
    return best.x


# ============================================================
# MCMC per batch (emcee)
# ============================================================
def run_mcmc_for_batch(t, y, rng, batch_id):
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
        args=(t, y)
    )

    # burn-in
    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()

    # main sampling
    sampler.run_mcmc(state, N_STEPS, progress=True)

    chain = sampler.get_chain(flat=True, thin=THIN)
    acc = float(np.mean(sampler.acceptance_fraction))

    theta = np.exp(chain)
    df_samples = pd.DataFrame(theta, columns=["lambda", "tau", "beta"])
    df_samples["log_lambda"]  = chain[:, 0]
    df_samples["log_tau"]     = chain[:, 1]
    df_samples["log_beta"]    = chain[:, 2]
    df_samples["BatchID"]     = str(batch_id)
    df_samples["accept_frac"] = acc

    return df_samples, acc


# ============================================================
# Plot helpers
# ============================================================
def kde_1d(x, grid):
    x = np.asarray(x, float)
    if len(x) < 5:
        return np.zeros_like(grid)
    kde = gaussian_kde(x)
    return kde(grid)

def prior_pdf_log10(u, a, b):
    """
    If theta ~ Uniform(a,b) in natural space, and u = log10(theta),
    then pdf(u) = ln(10) * 10^u / (b-a) for u in [log10 a, log10 b].
    """
    u = np.asarray(u, float)
    return (np.log(10.0) * (10.0 ** u)) / (b - a)

def make_corner_plot_for_batch(bid, sdf, out_path, rng, seed_offset=0):
    """
    Uses `corner.corner` to generate a standard corner plot in log10 space
    for (lambda, tau, beta), then saves it to out_path.
    """
    try:
        import corner
    except ImportError as e:
        raise RuntimeError(
            "corner is not installed. Install it with: pip install corner"
        ) from e

    X = np.vstack([
        np.log10(sdf["lambda"].to_numpy(float)),
        np.log10(sdf["tau"].to_numpy(float)),
        np.log10(sdf["beta"].to_numpy(float)),
    ]).T

    # subsample for speed/clarity
    if X.shape[0] > MAX_CORNER_SAMPLES:
        idx = rng.choice(X.shape[0], size=MAX_CORNER_SAMPLES, replace=False)
        Xp = X[idx]
    else:
        Xp = X

    labels = [
        r"$\log_{10}(\lambda)$",
        r"$\log_{10}(\tau)$",
        r"$\log_{10}(\beta)$",
    ]

    fig = corner.corner(
        Xp,
        labels=labels,
        quantiles=list(CORNER_QUANTILES),
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 10},
        levels=CORNER_LEVELS,
        plot_datapoints=True,
        fill_contours=True,
        smooth=1.0,
        bins=30,
    )

    fig.suptitle(f"{bid}  (log10 parameters)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def montage_images_vert(image_paths, out_path, title="Corner plots (stacked)"):
    """
    Creates a single figure that stacks images vertically (one per row).
    """
    if not image_paths:
        return

    imgs = [mpimg.imread(p) for p in image_paths]
    n = len(imgs)

    fig, axes = plt.subplots(
        nrows=n, ncols=1,
        figsize=(8.5, 8.5 * n),  # big enough for readability
        dpi=DPI
    )
    axes = np.atleast_1d(axes)

    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(CSV_PATH)
    df[BATCH_COL] = df[BATCH_COL].astype(str)

    available = set(df[BATCH_COL].unique().tolist())
    batches_present = [str(b) for b in BATCHES if str(b) in available]
    missing = [b for b in BATCHES if str(b) not in available]
    if missing:
        print(f"[warn] Missing BatchIDs in {CSV_PATH}: {missing}")
    if not batches_present:
        print("[done] No requested batches present.")
        return

    print(f"\nUsing fixed measurement SD: SIGMA_MAIN={SIGMA_MAIN}, SIGMA_T0={SIGMA_T0}\n")

    # Run MCMC per batch
    all_samples = {}
    all_data = {}
    accept = {}

    for bid in batches_present:
        dfi = df[df[BATCH_COL] == str(bid)].sort_values(TIME_COL)
        t = dfi[TIME_COL].to_numpy(float)
        y = np.clip(dfi[Y_COL].to_numpy(float), 0.0, 1.0)

        # average duplicates at same time
        if len(np.unique(t)) != len(t):
            tmp = pd.DataFrame({"t": t, "y": y}).groupby("t", as_index=False).mean()
            t = tmp["t"].to_numpy(float)
            y = tmp["y"].to_numpy(float)

        if len(t) < 4:
            print(f"[skip] {bid}: too few points ({len(t)})")
            continue

        print(f"\n[MCMC] {bid}  (n fixed={N_FIXED:g})  points={len(t)}")
        samples_df, acc = run_mcmc_for_batch(t, y, rng, bid)
        accept[bid] = acc
        all_samples[bid] = samples_df
        all_data[bid] = (t, y)

        out_csv = os.path.join(OUT_DIR, f"posterior_samples_{bid}.csv")
        samples_df.to_csv(out_csv, index=False)
        print(f"  saved samples -> {out_csv}")
        print(f"  acceptance fraction ~ {acc:.3f}")

    batches = [b for b in batches_present if b in all_samples]
    if not batches:
        print("[done] No batches sampled (all skipped).")
        return

    # ============================================================
    # PLOT 1: Prior vs Posterior (log10-space) grid: k rows x 3 cols
    # ============================================================
    param_info = [
        ("lambda", LAM_BOUNDS),
        ("tau",    TAU_BOUNDS),
        ("beta",   BETA_BOUNDS),
    ]

    fig1, axes1 = plt.subplots(
        nrows=len(batches), ncols=3,
        figsize=(13.5, 2.6 * len(batches)),
        dpi=DPI,
        sharey=False
    )
    axes1 = np.atleast_2d(axes1)

    for i, bid in enumerate(batches):
        sdf = all_samples[bid]
        for j, (pname, (a, b)) in enumerate(param_info):
            ax = axes1[i, j]
            s = sdf[pname].to_numpy(float)
            u = np.log10(s)

            umin = np.log10(a)
            umax = np.log10(b)
            grid = np.linspace(umin, umax, 400)

            post = kde_1d(u, grid)
            prior = prior_pdf_log10(grid, a, b)

            ax.plot(grid, prior, lw=1.6, ls="--", label="prior (Uniform)")
            ax.plot(grid, post,  lw=2.0, label="posterior (KDE)")

            ax.set_xlabel(rf"$\log_{{10}}({pname})$")
            if j == 0:
                ax.set_ylabel("density")
            if i == 0:
                ax.set_title(rf"Prior vs Posterior: {pname}")

        # row label
        axes1[i, 0].text(
            -0.35, 0.5, str(bid),
            transform=axes1[i, 0].transAxes,
            rotation=90,
            va="center", ha="center",
            fontsize=11, fontweight="bold"
        )

    axes1[0, 2].legend(frameon=True, fontsize=9, loc="best")
    fig1.suptitle("Prior vs Posterior (per batch; parameters in log10 space)", y=1.01)
    fig1.tight_layout()
    fig1.savefig(OUT_PRIOR_POST_PNG, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {OUT_PRIOR_POST_PNG}")

    # ============================================================
    # PLOT 2: Corner plots using the `corner` module (one per batch + montage)
    # ============================================================
    corner_paths = []
    for bi, bid in enumerate(batches):
        sdf = all_samples[bid]
        out_corner_i = os.path.join(OUT_DIR, f"corner_{bid}.png")
        make_corner_plot_for_batch(
            bid=bid,
            sdf=sdf,
            out_path=out_corner_i,
            rng=np.random.default_rng(SEED + 999 + bi),
            seed_offset=bi,
        )
        corner_paths.append(out_corner_i)
        print(f"Saved: {out_corner_i}")

    montage_images_vert(corner_paths, OUT_CORNER_PNG, title="Corner plots (per batch; log10 parameters)")
    print(f"Saved: {OUT_CORNER_PNG}")

    # ============================================================
    # PLOT 3: Data + posterior median curve + 95% band
    # ============================================================
    n_panels = len(batches)
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))

    fig3, axes3 = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5.6 * ncols, 3.8 * nrows),
        dpi=DPI,
        sharey=True
    )
    axes3 = np.atleast_1d(axes3).ravel()

    for k, bid in enumerate(batches):
        ax = axes3[k]
        t, y = all_data[bid]
        sdf = all_samples[bid]

        tmax = float(np.max(t))
        t_dense = np.linspace(0.0, tmax, T_DENSE)

        n_draws = min(N_DRAWS_BANDS, len(sdf))
        idx = np.random.default_rng(SEED + 100 + k).choice(len(sdf), size=n_draws, replace=False)
        sub = sdf.iloc[idx]

        Ypred = np.empty((n_draws, len(t_dense)), dtype=float)
        lam_vals  = sub["lambda"].to_numpy(dtype=float)
        tau_vals  = sub["tau"].to_numpy(dtype=float)
        beta_vals = sub["beta"].to_numpy(dtype=float)

        for i in range(n_draws):
            Ypred[i, :] = solve_curve_closed_form(t_dense, lam_vals[i], tau_vals[i], beta_vals[i])

        # pointwise posterior credible band for f(t) induced by parameter uncertainty
        q_lo, q_med, q_hi = np.quantile(Ypred, [0.025, 0.5, 0.975], axis=0)

        ax.fill_between(t_dense, q_lo, q_hi, alpha=0.25, label="95% band")
        ax.plot(t_dense, q_med, lw=2.2, label="posterior median")
        ax.plot(t, y, "o", ms=3.5, alpha=0.85, label="data")

        ax.set_title(f"{bid}  (acc~{accept[bid]:.2f})")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("release fraction")
        ax.set_ylim(0.0, 1.05)

        if k == 0:
            ax.legend(frameon=True, fontsize=9)

    for k in range(n_panels, len(axes3)):
        axes3[k].axis("off")

    fig3.suptitle(
        f"Posterior credible band (parameter uncertainty; fixed SD: t>0 {SIGMA_MAIN}, t=0 {SIGMA_T0})",
        y=1.01
    )
    fig3.tight_layout()
    fig3.savefig(OUT_BANDS_PNG, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {OUT_BANDS_PNG}")

    print("\nDone.")


if __name__ == "__main__":
    main()
