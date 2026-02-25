#!/usr/bin/env python3
"""
synthetic_bayesian_uniform_tau.py  (cleaned + updated)

Requested changes:
1) Fit-with-bands plot:
   - Put the CASE label (Zero/First/Sigmoid) + parameter values in the SUBPLOT TITLE
   - (No extra textbox annotation)

2) Corner plot montage (1 row):
   - Each subplot in the montage has a title with the CASE label

Notes:
- Individual corner plots themselves are kept clean (no suptitle; show_titles=False).
- Montage adds titles above each image panel.
- Uniform priors in NATURAL space for (lambda, tau, beta) within bounds.
  Sampling in z = log(theta) => log-prior = z0+z1+z2 inside bounds.

Outputs (OUT_DIR):
  synthetic_data.csv
  posterior_samples_<case>.csv
  corner_<case>.png
  corner_montage_row.png
  fit_with_bands.png

Requires:
  pip install emcee corner numpy pandas matplotlib scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.optimize import least_squares


# ============================================================
# USER SETTINGS
# ============================================================
OUT_DIR = "synthetic_bayes_demo_uniform_tau"

# ---- Synthetic sampling grid ----
T_END_MIN = 240.0
DT_SAMPLE = 15.0      # minutes; set None to use N_POINTS
N_POINTS  = None

# ---- Synthetic noise (Gaussian) ----
NOISE_SD_MAIN = 0.03   # added at t>0
NOISE_SD_T0   = 0.0    # added at t=0

# ---- Likelihood noise model (what inference assumes) ----
SIGMA_MAIN = NOISE_SD_MAIN
SIGMA_T0   = 1e-3
T0_ATOL    = 1e-12

SEED = 42

# ---- Parameter bounds (natural space) ----
LAM_BOUNDS  = (1e-6, 1e2)
TAU_BOUNDS  = (1e-6, 1e4)
BETA_BOUNDS = (1e-4, 1e2)

# ---- MCMC ----
N_WALKERS = 48
N_BURN    = 1500
N_STEPS   = 3000
THIN      = 2

# ---- Init via MLE ----
USE_MLE_INIT = True
N_STARTS_MLE = 20
MAX_NFEV_MLE = 300

# ---- Posterior predictive band ----
N_DRAWS_BANDS = 800
T_DENSE = 300

# ---- Corner plot ----
MAX_CORNER_SAMPLES = 6000
CORNER_LEVELS = (0.68, 0.95)
CORNER_QUANTILES = (0.16, 0.50, 0.84)

# ---- Plotting ----
USE_TEX = True
DPI = 180


# ============================================================
# True cases (order matters for montage + plots)
# ============================================================
TRUE_CASES = {
    "Zero":    dict(lam=0.025, tau=1.0,  beta=0.02),
    "First":   dict(lam=0.025, tau=1.0,  beta=1.0),
    "Sigmoid": dict(lam=0.025, tau=50.0, beta=1.0),
}


# ============================================================
# Matplotlib
# ============================================================
if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
    })


# ============================================================
# Model (closed-form, n=1)
# df/dt = lam * (t/(tau+t)) * (1-f)^beta, f(0)=0
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
# Synthetic data
# ============================================================
def make_time_grid():
    if DT_SAMPLE is not None:
        t = np.arange(0.0, T_END_MIN + 1e-12, float(DT_SAMPLE))
        if t.size and t[-1] < T_END_MIN - 1e-9:
            t = np.append(t, T_END_MIN)
        return t
    if N_POINTS is None or int(N_POINTS) < 2:
        raise ValueError("If DT_SAMPLE=None, set N_POINTS >= 2.")
    return np.linspace(0.0, T_END_MIN, int(N_POINTS))


def add_noise(t, y_true, rng):
    t = np.asarray(t, float)
    y = np.asarray(y_true, float).copy()
    is_t0 = np.isclose(t, 0.0, atol=T0_ATOL, rtol=0.0)

    if NOISE_SD_MAIN > 0:
        y[~is_t0] += rng.normal(0.0, float(NOISE_SD_MAIN), size=int(np.sum(~is_t0)))
    if NOISE_SD_T0 > 0:
        y[is_t0] += rng.normal(0.0, float(NOISE_SD_T0), size=int(np.sum(is_t0)))

    return np.clip(y, 0.0, 1.0)


def sigma_vector(t):
    t = np.asarray(t, float)
    is_t0 = np.isclose(t, 0.0, atol=T0_ATOL, rtol=0.0)
    return np.where(is_t0, float(SIGMA_T0), float(SIGMA_MAIN))


# ============================================================
# Sampling parameterization: z = [log lam, log tau, log beta]
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
# Prior: Uniform in natural space for ALL params
# ============================================================
def log_prior(z):
    lb, ub = bounds_log()
    if np.any(z < lb) or np.any(z > ub):
        return -np.inf
    # Jacobian for theta = exp(z): p(z) ∝ exp(z0+z1+z2)
    return float(np.sum(z))


# ============================================================
# Likelihood + posterior
# ============================================================
def log_likelihood(z, t, y):
    lam, tau, beta = unpack(z)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    sig = sigma_vector(t)
    r = (y - y_pred) / sig
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sig) + np.log(2.0 * np.pi)))


def log_posterior(z, t, y):
    lp = log_prior(z)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(z, t, y)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ============================================================
# MLE init (curve-only least squares)
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
# MCMC
# ============================================================
def run_mcmc(t, y, rng):
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
        args=(t, y),
    )

    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, N_STEPS, progress=True)

    chain = sampler.get_chain(flat=True, thin=THIN)
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, acc


# ============================================================
# Summaries + formatting
# ============================================================
def chain_to_df(chain, case_name, acc):
    theta = np.exp(chain)
    df = pd.DataFrame(theta, columns=["lambda", "tau", "beta"])
    df["log_lambda"] = chain[:, 0]
    df["log_tau"]    = chain[:, 1]
    df["log_beta"]   = chain[:, 2]
    df["Case"]       = str(case_name)
    df["accept_frac"] = float(acc)
    return df


def posterior_medians(samples_df):
    lam_hat = float(np.median(samples_df["lambda"].to_numpy(float)))
    tau_hat = float(np.median(samples_df["tau"].to_numpy(float)))
    beta_hat = float(np.median(samples_df["beta"].to_numpy(float)))
    return lam_hat, tau_hat, beta_hat


def fmt_g(x):
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3g}"


# ============================================================
# Plot helpers
# ============================================================
def make_corner_plot(samples_df, truth, out_path, rng):
    import corner

    X = np.vstack([
        np.log10(samples_df["lambda"].to_numpy(float)),
        np.log10(samples_df["tau"].to_numpy(float)),
        np.log10(samples_df["beta"].to_numpy(float)),
    ]).T

    if X.shape[0] > MAX_CORNER_SAMPLES:
        idx = rng.choice(X.shape[0], size=MAX_CORNER_SAMPLES, replace=False)
        Xp = X[idx]
    else:
        Xp = X

    truths = [
        np.log10(truth["lam"]),
        np.log10(truth["tau"]),
        np.log10(truth["beta"]),
    ]
    labels = [r"$\log_{10}(\lambda)$", r"$\log_{10}(\tau)$", r"$\log_{10}(\beta)$"]

    # Keep corner clean; montage will label each subplot
    fig = corner.corner(
        Xp,
        labels=labels,
        truths=truths,
        quantiles=list(CORNER_QUANTILES),
        show_titles=False,
        levels=CORNER_LEVELS,
        plot_datapoints=True,
        fill_contours=True,
        smooth=1.0,
        bins=30,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def montage_images_row(image_paths, titles, out_path):
    """
    Place all corner plot PNGs in ONE ROW (ncols = n_images).
    Adds a SUBPLOT TITLE for each case.
    """
    if not image_paths:
        return
    if len(image_paths) != len(titles):
        raise ValueError("image_paths and titles must have the same length")

    imgs = [mpimg.imread(p) for p in image_paths]
    n = len(imgs)

    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(8.5 * n, 8.5), dpi=DPI)
    axes = np.atleast_1d(axes)

    for ax, im, ttl in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(str(ttl), fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- Synthetic dataset ---
    t = make_time_grid()
    synthetic = {}
    rows = []

    for case_name, truth in TRUE_CASES.items():
        y_true = solve_curve_closed_form(t, truth["lam"], truth["tau"], truth["beta"])
        y_obs  = add_noise(t, y_true, rng)
        synthetic[case_name] = dict(t=t, y_true=y_true, y_obs=y_obs, truth=truth)

        for ti, yi in zip(t, y_obs):
            rows.append({"Case": case_name, "time_min": float(ti), "release_frac": float(yi)})

    syn_csv = os.path.join(OUT_DIR, "synthetic_data.csv")
    pd.DataFrame(rows).to_csv(syn_csv, index=False)
    print(f"Saved synthetic dataset: {syn_csv}")
    print(f"Noise: main SD={NOISE_SD_MAIN}, t=0 SD={NOISE_SD_T0}")
    print(f"Inference assumes: SIGMA_MAIN={SIGMA_MAIN}, SIGMA_T0={SIGMA_T0}\n")

    # --- MCMC per case ---
    all_samples = {}
    accept = {}
    corner_paths = []
    corner_titles = []

    for case_name in TRUE_CASES.keys():
        y_obs = synthetic[case_name]["y_obs"]

        print(f"[MCMC] {case_name}  points={len(t)}  |  tau prior: uniform in bounds")
        chain, acc = run_mcmc(t, y_obs, rng)
        accept[case_name] = acc

        sdf = chain_to_df(chain, case_name, acc)
        all_samples[case_name] = sdf

        out_samples = os.path.join(OUT_DIR, f"posterior_samples_{case_name}.csv")
        sdf.to_csv(out_samples, index=False)
        print(f"  saved samples -> {out_samples}")
        print(f"  acceptance fraction ~ {acc:.3f}")

        out_corner = os.path.join(OUT_DIR, f"corner_{case_name}.png")
        make_corner_plot(
            sdf,
            synthetic[case_name]["truth"],
            out_path=out_corner,
            rng=np.random.default_rng(SEED + 1000 + (hash(case_name) % 10000)),
        )
        corner_paths.append(out_corner)
        corner_titles.append(case_name)
        print(f"  saved corner -> {out_corner}\n")

    # Montage corner plots in ONE ROW with subplot titles
    out_montage = os.path.join(OUT_DIR, "corner_montage_row.png")
    montage_images_row(corner_paths, corner_titles, out_montage)
    print(f"Saved montage: {out_montage}")

    # --- Credible band plot: SUBPLOT TITLES contain case + params (estimated + true) ---
    fig, axes = plt.subplots(1, len(TRUE_CASES), figsize=(14.5, 4.2), dpi=DPI, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, case_name in zip(axes, TRUE_CASES.keys()):
        truth  = synthetic[case_name]["truth"]
        y_true = synthetic[case_name]["y_true"]
        y_obs  = synthetic[case_name]["y_obs"]
        sdf    = all_samples[case_name]

        # posterior median params (for title)
        lam_hat, tau_hat, beta_hat = posterior_medians(sdf)

        tmax = float(np.max(t))
        t_dense = np.linspace(0.0, tmax, T_DENSE)

        n_draws = min(N_DRAWS_BANDS, len(sdf))
        idx = np.random.default_rng(SEED + 200 + (hash(case_name) % 10000)).choice(
            len(sdf), size=n_draws, replace=False
        )
        sub = sdf.iloc[idx]

        lam_vals  = sub["lambda"].to_numpy(float)
        tau_vals  = sub["tau"].to_numpy(float)
        beta_vals = sub["beta"].to_numpy(float)

        Ypred = np.empty((n_draws, len(t_dense)), dtype=float)
        for i in range(n_draws):
            Ypred[i, :] = solve_curve_closed_form(t_dense, lam_vals[i], tau_vals[i], beta_vals[i])

        q_lo, q_med, q_hi = np.quantile(Ypred, [0.025, 0.5, 0.975], axis=0)

        # plot
        ax.fill_between(t_dense, q_lo, q_hi, alpha=0.25)
        ax.plot(t_dense, q_med, lw=2.2)
        ax.plot(t, y_obs, "o", ms=3.8, alpha=0.85)
        ax.plot(t, y_true, lw=1.8, ls="--")

        ax.set_xlabel("time (min)")
        ax.set_ylim(0.0, 1.05)

        # subplot title: label + parameter values
        title = (
            f"{case_name}\n"
            rf"$\hat\lambda={fmt_g(lam_hat)}$ (true {fmt_g(truth['lam'])}), "
            rf"$\hat\tau={fmt_g(tau_hat)}$ (true {fmt_g(truth['tau'])}), "
            rf"$\hat\beta={fmt_g(beta_hat)}$ (true {fmt_g(truth['beta'])})"
        )
        ax.set_title(title, fontsize=10)

    axes[0].set_ylabel("release fraction")

    # legend once (proxy artists) — keeps titles clean
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], lw=2.2),
        Line2D([0], [0], lw=1.8, ls="--"),
        Line2D([0], [0], marker="o", lw=0, ms=4),
    ]
    axes[0].legend(proxy, ["posterior median", "true curve", "data"], frameon=True, fontsize=9, loc="lower right")

    fig.tight_layout()
    out_bands = os.path.join(OUT_DIR, "fit_with_bands.png")
    fig.savefig(out_bands, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved credible band plot: {out_bands}")

    print("\nDone.")


if __name__ == "__main__":
    main()
