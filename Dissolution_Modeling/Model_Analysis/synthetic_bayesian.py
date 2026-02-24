#!/usr/bin/env python3
"""
synthetic_bayesian.py

Synthetic data generation + Bayesian parameter inference (emcee) for n=1 model:

    df/dt = lam * (t/(tau+t)) * (1-f)^beta ,  f(0)=0

This version integrates the UPDATED tau-prior machinery from tau_prior_tools.py:

Tau prior per curve via `compute_tau_prior_from_data(...)`, which supports:
  - monotone QP smoothing (cvxpy) + PCHIP derivative peak (prior_mode="spline_peak")
  - flat-derivative detection: if df/dt is essentially flat, t* is not identifiable and we
    switch to an EARLY-PEAK prior automatically (prior_mode="flat_early_uniform" by default)
  - optional explicit early-peak prior (prior_mode="early_uniform"):
        t1 = first strictly-positive sample time
        t* ~ Uniform(t_eps, t1)  =>  tau ~ Uniform(tau_factor*t_eps, tau_factor*t1)

Bayesian sampling uses z = [log lam, log tau, log beta]. Priors:
  - lambda ~ Uniform(LAM_BOUNDS) in natural space  -> log-prior adds +z0 (Jacobian)
  - beta   ~ Uniform(BETA_BOUNDS) in natural space -> log-prior adds +z2 (Jacobian)
  - tau prior chosen per case:
      * if tau_prior_tools returns tau_prior_bounds (early-uniform): tau ~ Uniform(bounds) -> +z1
      * else: log(tau) ~ Normal(log(tau0), TAU_PRIOR_LOGSIG^2), truncated to TAU_BOUNDS

Outputs (in OUT_DIR):
  synthetic_data.csv
  tau_prior_summary.csv
  posterior_samples_<case>.csv
  corner_<case>.png
  corner_montage.png
  fit_with_bands.png

Requires:
  pip install emcee corner numpy pandas matplotlib scipy
  pip install cvxpy osqp  (for tau_prior_tools monotone QP)
  (optional fallback solver) pip install scs

Place tau_prior_tools.py in the same directory (or on PYTHONPATH).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.optimize import least_squares

from tau_prior_tools import compute_tau_prior_from_data


# ============================================================
# USER SETTINGS
# ============================================================
OUT_DIR = "synthetic_bayes_demo"

# ---- Synthetic sampling grid ----
T_END_MIN = 240.0
DT_SAMPLE = 15.0      # one sample every DT_SAMPLE minutes (set None to use N_POINTS)
N_POINTS  = None

# ---- Synthetic noise (Gaussian) ----
NOISE_SD_MAIN = 0.03   # added at t>0
NOISE_SD_T0   = 0.0    # added at t=0 (often 0)

# ---- Likelihood noise model (what inference assumes) ----
SIGMA_MAIN = NOISE_SD_MAIN
SIGMA_T0   = 1e-3
T0_ATOL    = 1e-12

SEED = 42

# ---- Parameter bounds (natural space) ----
LAM_BOUNDS  = (1e-6, 1e2)
TAU_BOUNDS  = (1e-6, 1e4)
BETA_BOUNDS = (1e-4, 1e2)

# ---- Tau prior estimation knobs (passed to tau_prior_tools) ----
USE_TAU_PRIOR = True
TAU_PRIOR_FACTOR = 1.0        # tau0 = factor * t*
TAU_PRIOR_LOGSIG = 0.5        # used when prior is lognormal on log(tau)

TAU_EST_SMOOTH_LAMBDA = 5.0  # increase for more smoothing (try 10, 50, 200, ...)
TAU_EST_DENSE = 4000
TAU_EST_PEAK_FRAC = 0.99
TAU_EST_EXCLUDE_BOUNDARIES = False
TAU_EST_BOUNDARY_EPS_FRAC = 1e-4
TAU_EST_BOUNDARY_EPS_ABS  = 1e-12

# Flat-derivative detection (inside tau_prior_tools):
# If max(dfdt)/median(dfdt) < threshold, slope is "flat" => t* not identifiable.
# Recommended behavior: switch to early-uniform prior on t*.
TAU_EST_FLAT_RATIO_THRESHOLD = 1.10
TAU_EST_FLAT_SWITCH_TO_EARLY_UNIFORM = True
FLAT_EPS_ABS  = 1e-12
FLAT_EPS_FRAC = 1e-6

# Explicit "peak before first measured time" mode:
# If a case is listed here, we force early-uniform prior for that case
# (even if derivative is not detected as flat).
EARLY_UNIFORM_CASES = {}   # e.g. {"First"}; set to empty set() to disable
EARLY_EPS_ABS  = 1e-12            # lower bound t_eps >= EARLY_EPS_ABS
EARLY_EPS_FRAC = 1e-6             # and t_eps >= EARLY_EPS_FRAC * t1

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
# True cases (first of each family)
# ============================================================
TRUE_CASES = {
    "Zero":    dict(lam=0.025, tau=1.0,   beta=0.02),
    "First":   dict(lam=0.025, tau=1.0,   beta=1.0),
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
# Closed-form solution (n=1)
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
        if t[-1] < T_END_MIN - 1e-9:
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
        y[~is_t0] += rng.normal(0.0, float(NOISE_SD_MAIN), size=np.sum(~is_t0))
    if NOISE_SD_T0 > 0:
        y[is_t0] += rng.normal(0.0, float(NOISE_SD_T0), size=np.sum(is_t0))

    return np.clip(y, 0.0, 1.0)


# ============================================================
# Likelihood noise
# ============================================================
def sigma_vector(t):
    t = np.asarray(t, float)
    is_t0 = np.isclose(t, 0.0, atol=T0_ATOL, rtol=0.0)
    return np.where(is_t0, float(SIGMA_T0), float(SIGMA_MAIN))


# ============================================================
# Tau prior: compute per-case using tau_prior_tools
# ============================================================
def _intersect_bounds(a, b, c, d):
    lo = max(float(a), float(c))
    hi = min(float(b), float(d))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return None
    return (lo, hi)


def compute_tau_prior_config(t, y_obs, *, case_name):
    """
    Returns a dict describing the tau prior for this case.

    Prior types:
      - kind="uniform_bounds": tau ~ Uniform(tau_lo, tau_hi) in natural space
      - kind="lognormal":      log(tau) ~ Normal(mu, sig^2) in z-space (truncated to TAU_BOUNDS)
      - kind="uniform_global": tau ~ Uniform(TAU_BOUNDS) in natural space
    """
    if not USE_TAU_PRIOR:
        return dict(kind="uniform_global", tau0=np.nan, tau_bounds=None, mu=None, sig=None,
                    t_star=np.nan, dfdt_max=np.nan, diag={})

    force_early = (str(case_name) in set(EARLY_UNIFORM_CASES))

    tau0, t_star, dfdt_max, diag = compute_tau_prior_from_data(
        t, y_obs,
        tau_factor=TAU_PRIOR_FACTOR,
        tau_bounds=TAU_BOUNDS,
        smooth_lambda=TAU_EST_SMOOTH_LAMBDA,
        dense=TAU_EST_DENSE,
        peak_frac=TAU_EST_PEAK_FRAC,
        exclude_boundaries=TAU_EST_EXCLUDE_BOUNDARIES,
        boundary_eps_frac=TAU_EST_BOUNDARY_EPS_FRAC,
        boundary_eps_abs=TAU_EST_BOUNDARY_EPS_ABS,
        solver="OSQP",
        flat_ratio_threshold=TAU_EST_FLAT_RATIO_THRESHOLD,
        flat_switch_to_early_uniform=TAU_EST_FLAT_SWITCH_TO_EARLY_UNIFORM,
        flat_early_eps_abs=FLAT_EPS_ABS,
        flat_early_eps_frac=FLAT_EPS_FRAC,
        use_early_uniform_prior=force_early,
        early_eps_abs=EARLY_EPS_ABS,
        early_eps_frac=EARLY_EPS_FRAC,
    )

    prior_mode = str(diag.get("prior_mode", ""))

    # If tau_prior_tools produced uniform bounds (explicit early or flat-switched early), use them.
    tb = diag.get("tau_prior_bounds", None)
    if tb is not None and np.all(np.isfinite(tb)) and (tb[1] > tb[0]):
        ib = _intersect_bounds(tb[0], tb[1], TAU_BOUNDS[0], TAU_BOUNDS[1])
        if ib is not None:
            tau_lo, tau_hi = ib
            tau_mid = 0.5 * (tau_lo + tau_hi)
            return dict(
                kind="uniform_bounds",
                tau0=float(tau_mid),
                tau_bounds=(float(tau_lo), float(tau_hi)),
                mu=None,
                sig=None,
                t_star=float(t_star),
                dfdt_max=float(dfdt_max) if np.isfinite(dfdt_max) else np.nan,
                diag=diag,
            )

    # Otherwise: use lognormal on log(tau) centered at tau0
    tau0 = float(np.clip(tau0, TAU_BOUNDS[0], TAU_BOUNDS[1])) if np.isfinite(tau0) else np.nan
    if not np.isfinite(tau0) or tau0 <= 0.0:
        return dict(kind="uniform_global", tau0=tau0, tau_bounds=None, mu=None, sig=None,
                    t_star=float(t_star), dfdt_max=float(dfdt_max) if np.isfinite(dfdt_max) else np.nan, diag=diag)

    mu = float(np.log(tau0))
    sig = float(TAU_PRIOR_LOGSIG)
    return dict(
        kind="lognormal",
        tau0=float(tau0),
        tau_bounds=None,
        mu=mu,
        sig=sig,
        t_star=float(t_star),
        dfdt_max=float(dfdt_max) if np.isfinite(dfdt_max) else np.nan,
        diag=diag,
    )


# ============================================================
# Sampling parameterization: z = [log lam, log tau, log beta]
# ============================================================
def bounds_log(tau_bounds=None):
    tau_lo, tau_hi = TAU_BOUNDS
    if tau_bounds is not None:
        tau_lo, tau_hi = float(tau_bounds[0]), float(tau_bounds[1])
    lb = np.log([LAM_BOUNDS[0], tau_lo, BETA_BOUNDS[0]])
    ub = np.log([LAM_BOUNDS[1], tau_hi, BETA_BOUNDS[1]])
    return lb, ub


def unpack(z):
    z = np.asarray(z, float)
    z = np.clip(z, -60.0, 60.0)
    lam, tau, beta = np.exp(z)
    return float(lam), float(tau), float(beta)


# ============================================================
# Log prior (in z-space)
# ============================================================
def log_prior(z, *, tau_prior_cfg):
    """
    lam, beta: Uniform in natural space -> Jacobian terms +z0, +z2.
    tau:
      - uniform_global or uniform_bounds: Uniform in natural space -> +z1, plus bounds restriction
      - lognormal: Normal on z1 centered at mu with sd sig, plus global bounds restriction
    """
    kind = str(tau_prior_cfg["kind"])
    tau_bounds_local = tau_prior_cfg["tau_bounds"] if kind == "uniform_bounds" else None

    lb, ub = bounds_log(tau_bounds=tau_bounds_local)
    if np.any(z < lb) or np.any(z > ub):
        return -np.inf

    # Jacobian for uniform(lam) and uniform(beta) in natural space
    lp = float(z[0] + z[2])

    if kind in ("uniform_global", "uniform_bounds"):
        # uniform(tau) in natural space -> Jacobian +z1
        lp += float(z[1])
        return lp

    if kind == "lognormal":
        mu = float(tau_prior_cfg["mu"])
        sig = float(tau_prior_cfg["sig"])
        if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
            return -np.inf
        lp += float(-0.5 * ((z[1] - mu) / sig) ** 2)  # constants omitted
        return lp

    # fallback
    lp += float(z[1])
    return lp


# ============================================================
# Likelihood
# ============================================================
def log_likelihood(z, t, y):
    lam, tau, beta = unpack(z)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    sig = sigma_vector(t)
    r = (y - y_pred) / sig
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sig) + np.log(2.0 * np.pi)))


def log_posterior(z, t, y, tau_prior_cfg):
    lp = log_prior(z, tau_prior_cfg=tau_prior_cfg)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(z, t, y)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ============================================================
# MLE init
# ============================================================
def residuals_log(z, t, y_obs):
    lam, tau, beta = unpack(z)
    y_pred = solve_curve_closed_form(t, lam, tau, beta)
    return (y_pred - y_obs)


def fit_mle_log(t, y, rng, *, tau_bounds=None):
    lb, ub = bounds_log(tau_bounds=tau_bounds)
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
# MCMC per case
# ============================================================
def run_mcmc(t, y, rng, *, case_name, tau_prior_cfg):
    import emcee

    kind = str(tau_prior_cfg["kind"])
    tau_bounds_local = tau_prior_cfg["tau_bounds"] if kind == "uniform_bounds" else None

    lb, ub = bounds_log(tau_bounds=tau_bounds_local)
    ndim = 3

    if USE_MLE_INIT:
        z_hat = fit_mle_log(t, y, rng, tau_bounds=tau_bounds_local)
        p0 = z_hat[None, :] + 0.25 * rng.standard_normal(size=(N_WALKERS, ndim))
        p0 = np.minimum(np.maximum(p0, lb), ub)
    else:
        p0 = rng.uniform(lb, ub, size=(N_WALKERS, ndim))

    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim,
        log_posterior,
        args=(t, y, tau_prior_cfg)
    )

    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, N_STEPS, progress=True)

    chain = sampler.get_chain(flat=True, thin=THIN)
    acc = float(np.mean(sampler.acceptance_fraction))

    theta = np.exp(chain)
    df = pd.DataFrame(theta, columns=["lambda", "tau", "beta"])
    df["log_lambda"] = chain[:, 0]
    df["log_tau"]    = chain[:, 1]
    df["log_beta"]   = chain[:, 2]
    df["Case"]       = str(case_name)
    df["accept_frac"] = acc
    df["tau_prior_kind"] = str(tau_prior_cfg["kind"])
    return df, acc


# ============================================================
# Plot helpers
# ============================================================
def make_corner_plot(case_name, samples_df, truth, tau_prior_cfg, out_path, rng):
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

    fig = corner.corner(
        Xp,
        labels=labels,
        truths=truths,
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

    kind = str(tau_prior_cfg["kind"])
    if kind == "uniform_bounds":
        lo, hi = tau_prior_cfg["tau_bounds"]
        subtitle = rf"$\tau \sim \mathrm{{Unif}}({lo:.3g},{hi:.3g})$"
    elif kind == "lognormal":
        tau0 = tau_prior_cfg["tau0"]
        s = tau_prior_cfg["sig"]
        subtitle = rf"$\log(\tau)\sim \mathcal{{N}}(\log({tau0:.3g}), {s:.3g}^2)$"
    else:
        subtitle = rf"$\tau \sim \mathrm{{Unif}}({TAU_BOUNDS[0]:.1g},{TAU_BOUNDS[1]:.1g})$"

    fig.suptitle(f"{case_name} (truth lines)   |   {subtitle}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def montage_images_vert(image_paths, out_path, title="Corner plots (stacked)"):
    if not image_paths:
        return
    imgs = [mpimg.imread(p) for p in image_paths]
    n = len(imgs)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8.5, 8.5 * n), dpi=DPI)
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

    # --- Generate synthetic data ---
    t = make_time_grid()
    rows = []
    synthetic = {}

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

    # --- Tau prior config per case ---
    tau_prior_rows = []
    tau_prior_cfgs = {}

    for case_name in TRUE_CASES.keys():
        y = synthetic[case_name]["y_obs"]
        cfg = compute_tau_prior_config(t, y, case_name=case_name)
        tau_prior_cfgs[case_name] = cfg

        diag = cfg.get("diag", {}) or {}
        prior_mode = str(diag.get("prior_mode", ""))
        flat_ratio = diag.get("flat_ratio", np.nan)
        t1 = diag.get("t1", np.nan)

        tstar_lo = np.nan
        tstar_hi = np.nan
        if diag.get("tstar_prior_bounds", None) is not None:
            a, b = diag["tstar_prior_bounds"]
            tstar_lo, tstar_hi = float(a), float(b)

        tau_lo = np.nan
        tau_hi = np.nan
        if cfg["kind"] == "uniform_bounds" and cfg["tau_bounds"] is not None:
            tau_lo, tau_hi = cfg["tau_bounds"]

        tau_prior_rows.append({
            "Case": case_name,
            "tau_prior_kind": cfg["kind"],
            "tau0_repr": float(cfg.get("tau0", np.nan)),
            "tau_lo": float(tau_lo),
            "tau_hi": float(tau_hi),
            "t_star_repr": float(cfg.get("t_star", np.nan)),
            "dfdt_max": float(cfg.get("dfdt_max", np.nan)),
            "prior_mode": prior_mode,
            "t1_first_pos_time": float(t1) if np.isfinite(t1) else np.nan,
            "tstar_lo": float(tstar_lo) if np.isfinite(tstar_lo) else np.nan,
            "tstar_hi": float(tstar_hi) if np.isfinite(tstar_hi) else np.nan,
            "flat_ratio": float(flat_ratio) if np.isfinite(flat_ratio) else np.nan,
            "smooth_lambda": float(TAU_EST_SMOOTH_LAMBDA),
            "tau_prior_factor": float(TAU_PRIOR_FACTOR),
            "tau_prior_logsig": float(TAU_PRIOR_LOGSIG),
            "force_early_uniform": bool(case_name in EARLY_UNIFORM_CASES),
            "flat_switch_to_early_uniform": bool(TAU_EST_FLAT_SWITCH_TO_EARLY_UNIFORM),
            "flat_ratio_threshold": float(TAU_EST_FLAT_RATIO_THRESHOLD),
        })

        # pretty print
        if cfg["kind"] == "uniform_bounds":
            lo, hi = cfg["tau_bounds"]
            print(f"[tau prior] {case_name}: UNIFORM tau in [{lo:.3g}, {hi:.3g}]  (mode={prior_mode})")
        elif cfg["kind"] == "lognormal":
            print(f"[tau prior] {case_name}: LOGNORMAL log(tau) centered at tau0={cfg['tau0']:.3g}  (mode={prior_mode})")
        else:
            print(f"[tau prior] {case_name}: GLOBAL UNIFORM tau in TAU_BOUNDS  (mode={prior_mode})")

    tau_prior_csv = os.path.join(OUT_DIR, "tau_prior_summary.csv")
    pd.DataFrame(tau_prior_rows).to_csv(tau_prior_csv, index=False)
    print(f"\nSaved tau prior summary: {tau_prior_csv}\n")

    # --- Run MCMC per case ---
    all_samples = {}
    accept = {}
    corner_paths = []

    for case_name in TRUE_CASES.keys():
        y = synthetic[case_name]["y_obs"]
        cfg = tau_prior_cfgs[case_name]

        print(f"[MCMC] {case_name}  points={len(t)}  |  tau prior kind: {cfg['kind']}")
        sdf, acc = run_mcmc(t, y, rng, case_name=case_name, tau_prior_cfg=cfg)
        all_samples[case_name] = sdf
        accept[case_name] = acc

        out_samples = os.path.join(OUT_DIR, f"posterior_samples_{case_name}.csv")
        sdf.to_csv(out_samples, index=False)
        print(f"  saved samples -> {out_samples}")
        print(f"  acceptance fraction ~ {acc:.3f}")

        out_corner = os.path.join(OUT_DIR, f"corner_{case_name}.png")
        make_corner_plot(
            case_name,
            sdf,
            synthetic[case_name]["truth"],
            tau_prior_cfg=cfg,
            out_path=out_corner,
            rng=np.random.default_rng(SEED + 1000 + (hash(case_name) % 10000)),
        )
        corner_paths.append(out_corner)
        print(f"  saved corner -> {out_corner}\n")

    # Montage corner plots
    out_montage = os.path.join(OUT_DIR, "corner_montage.png")
    montage_images_vert(corner_paths, out_montage, title="Corner plots (truth shown as lines; tau prior annotated)")
    print(f"Saved montage: {out_montage}")

    # --- Credible band plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=DPI, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, case_name in zip(axes, TRUE_CASES.keys()):
        y_true = synthetic[case_name]["y_true"]
        y_obs  = synthetic[case_name]["y_obs"]
        sdf    = all_samples[case_name]

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

        ax.fill_between(t_dense, q_lo, q_hi, alpha=0.25, label="95% credible band")
        ax.plot(t_dense, q_med, lw=2.2, label="posterior median")

        ax.plot(t, y_obs, "o", ms=3.8, alpha=0.85, label="synthetic data")
        ax.plot(t, y_true, lw=1.8, ls="--", label="true curve")

        ax.set_title(f"{case_name} (acc~{accept[case_name]:.2f})")
        ax.set_xlabel("time (min)")
        ax.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("release fraction")
    axes[0].legend(frameon=True, fontsize=9, loc="best")
    fig.suptitle(
        rf"Posterior credible bands from parameter uncertainty "
        rf"(assumed SD: t>0 {SIGMA_MAIN}, t=0 {SIGMA_T0})",
        y=1.02
    )
    fig.tight_layout()
    out_bands = os.path.join(OUT_DIR, "fit_with_bands.png")
    fig.savefig(out_bands, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved credible band plot: {out_bands}")

    print("\nDone.")


if __name__ == "__main__":
    main()
