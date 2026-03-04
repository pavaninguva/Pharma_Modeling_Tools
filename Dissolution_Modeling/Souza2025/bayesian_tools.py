#!/usr/bin/env python3
"""
bayes_activation_framework.py  (UPDATED: drop t=0 from likelihood)

Reusable Bayesian inference utilities for the "activation" dissolution model:

    df/dt = lam * (t^n / (tau + t^n)) * (1 - f)^beta,   f(0)=0

Parameters: lam, tau, beta, n  (all assumed > 0)

Forward solve:
  - If n is FIXED to exactly 1 (or within a tiny tolerance), use a fast closed form:
      I1(t) = t - tau * log(1 + t/tau)
      A(t)  = lam * I1(t)
      u(t)  = 1-f(t)
      beta=1:  u = exp(-A)
      beta!=1: u = [1 - (1-beta)A]^(1/(1-beta))  (real-valued; saturate u=0 when bracket<=0)

  - Otherwise (n != 1), use scipy.integrate.solve_ivp.

Bayesian inference:
  - Sampling in z = log(theta) for positive params.
  - Priors are specified per-parameter with small dicts (uniform in natural space, or normal in log-space).
  - Supports fixing parameters by passing fixed_values dict.

Likelihood (UPDATED):
  - Treat f(0)=0 as an initial condition, NOT a noisy datapoint.
  - We DROP all t==0 points from the likelihood.
  - Therefore the likelihood uses a single sigma_main for all remaining timepoints.

Dependencies:
  pip install emcee numpy scipy
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Preprocessing for likelihood
# ---------------------------------------------------------------------
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


def drop_t0_points(t, y, *, t0_atol=1e-12):
    """
    Drops points with t==0 (within atol). Returns (t_keep, y_keep).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    keep = ~np.isclose(t, 0.0, atol=float(t0_atol), rtol=0.0)
    return t[keep], y[keep]


# ---------------------------------------------------------------------
# Forward solves
# ---------------------------------------------------------------------
def _I_n1(t, tau):
    tau_safe = max(float(tau), 1e-300)
    return t - tau_safe * np.log1p(t / tau_safe)


def forward_closed_form_n1(t_eval, lam, tau, beta, *, beta_eps=1e-10):
    """
    Closed form for n=1.

    Uses a Taylor-safe form near beta=1:
      delta = 1 - beta
      log u = log(1 - delta*A)/delta
      log u ~ -A - delta*A^2/2 - delta^2*A^3/3
    """
    t = np.asarray(t_eval, float)
    t = np.maximum(t, 0.0)

    lam = float(lam)
    tau = float(tau)
    beta = float(beta)

    A = lam * _I_n1(t, tau)
    delta = 1.0 - beta

    if abs(delta) < beta_eps:
        logu = -A - delta * (A * A) * 0.5 - (delta * delta) * (A * A * A) / 3.0
        u = np.exp(logu)
    else:
        base = 1.0 - delta * A
        u = np.zeros_like(base)
        pos = base > 0.0
        u[pos] = np.exp(np.log(base[pos]) / delta)
        u[~pos] = 0.0  # saturate to f=1 in real-valued regime

    f = 1.0 - u
    return np.clip(f, 0.0, 1.0)


def forward_solve_ivp(t_eval, lam, tau, beta, n, *, rtol=1e-6, atol=1e-8, method="RK45"):
    """
    Generic forward solve for n != 1 using solve_ivp.
    """
    t_eval = np.asarray(t_eval, float)
    t_eval = np.maximum(t_eval, 0.0)
    tmax = float(np.max(t_eval))
    if tmax <= 0.0:
        return np.zeros_like(t_eval)

    lam = float(lam)
    tau = float(tau)
    beta = float(beta)
    n = float(n)

    def rhs(t, f):
        f = float(np.clip(f[0], 0.0, 1.0))
        tpow = t ** n
        g = tpow / (tau + tpow + 1e-12)
        return [lam * g * max(1.0 - f, 0.0) ** beta]

    sol = solve_ivp(
        rhs,
        t_span=(0.0, tmax),
        y0=[0.0],
        t_eval=t_eval,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        vectorized=False,
    )
    if (not sol.success) or sol.y is None:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    f = sol.y[0]
    return np.clip(f, 0.0, 1.0)


def forward_model(t_eval, lam, tau, beta, n, *, n1_tol=0.0, ode_opts=None):
    """
    If |n-1| <= n1_tol -> use closed form; else use ODE solve_ivp.
    Recommended: set n1_tol=0.0 and only get closed form when n is FIXED to 1.
    """
    ode_opts = ode_opts or {}
    if abs(float(n) - 1.0) <= float(n1_tol):
        return forward_closed_form_n1(t_eval, lam, tau, beta)
    return forward_solve_ivp(t_eval, lam, tau, beta, n, **ode_opts)


# ---------------------------------------------------------------------
# Priors in z = log(theta)
# ---------------------------------------------------------------------
def log_prior_z(z, active_names, priors, bounds):
    """
    priors[name] supports:
      - {"type":"uniform"} -> uniform in natural space within bounds => log prior adds +z
      - {"type":"normal_log","mu":..,"sig":..} -> normal on z (lognormal on theta)
    bounds[name] = (lo, hi) in natural space.

    NOTE: We always enforce bounds as support (return -inf outside).
    """
    z = np.asarray(z, float)
    lp = 0.0
    for i, name in enumerate(active_names):
        lo, hi = bounds[name]
        lb, ub = np.log(lo), np.log(hi)
        zi = float(z[i])
        if zi < lb or zi > ub:
            return -np.inf

        spec = priors.get(name, {"type": "uniform"})
        typ = str(spec.get("type", "uniform")).lower()

        if typ == "uniform":
            lp += zi  # Jacobian term for uniform in natural space
        elif typ in ("normal_log", "lognormal"):
            mu = float(spec["mu"])
            sig = float(spec["sig"])
            if sig <= 0:
                return -np.inf
            lp += float(-0.5 * ((zi - mu) / sig) ** 2)  # constants dropped
        else:
            raise ValueError(f"Unknown prior type for {name}: {typ}")

    return float(lp)


# ---------------------------------------------------------------------
# Likelihood / posterior (UPDATED: t=0 dropped)
# ---------------------------------------------------------------------
def unpack_z(z, active_names, fixed_values):
    z = np.asarray(z, float)
    z = np.clip(z, -60.0, 60.0)
    theta = dict(fixed_values)
    for i, name in enumerate(active_names):
        theta[name] = float(np.exp(z[i]))
    return theta


def log_likelihood(
    z,
    t,
    y,
    *,
    active_names,
    fixed_values,
    priors,
    bounds,
    sigma_main=0.03,
    t0_atol=1e-12,
    n1_tol=0.0,
    ode_opts=None,
):
    """
    Gaussian log-likelihood with constant sigma_main, after DROPPING t==0 points.
    """
    theta = unpack_z(z, active_names, fixed_values)
    lam = theta["lambda"]
    tau = theta["tau"]
    beta = theta["beta"]
    n = theta["n"]

    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # UPDATED: drop t=0 points
    t_use, y_use = drop_t0_points(t, y, t0_atol=t0_atol)
    if t_use.size == 0:
        # no likelihood information left; treat as invalid
        return -np.inf

    try:
        y_pred = forward_model(t_use, lam, tau, beta, n, n1_tol=n1_tol, ode_opts=ode_opts)
    except Exception:
        return -np.inf

    sig = float(sigma_main)
    sig = max(sig, 1e-12)

    r = (y_use - y_pred) / sig
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sig) + np.log(2.0 * np.pi)))


def log_posterior(
    z,
    t,
    y,
    *,
    active_names,
    fixed_values,
    priors,
    bounds,
    sigma_main=0.03,
    t0_atol=1e-12,
    n1_tol=0.0,
    ode_opts=None,
):
    lp = log_prior_z(z, active_names, priors, bounds)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(
        z, t, y,
        active_names=active_names,
        fixed_values=fixed_values,
        priors=priors,
        bounds=bounds,
        sigma_main=sigma_main,
        t0_atol=t0_atol,
        n1_tol=n1_tol,
        ode_opts=ode_opts,
    )
    if not np.isfinite(ll):
        return -np.inf
    return float(lp + ll)


# ---------------------------------------------------------------------
# MLE init (optional; recommended mainly when n is fixed to 1)
# ---------------------------------------------------------------------
def _residuals_for_mle(z, t, y, *, active_names, fixed_values, n1_tol, ode_opts, t0_atol):
    theta = unpack_z(z, active_names, fixed_values)
    t_use, y_use = drop_t0_points(t, y, t0_atol=t0_atol)
    y_pred = forward_model(t_use, theta["lambda"], theta["tau"], theta["beta"], theta["n"], n1_tol=n1_tol, ode_opts=ode_opts)
    return (y_pred - y_use)


def fit_mle_init(
    t,
    y,
    rng,
    *,
    active_names,
    fixed_values,
    bounds,
    n_starts=10,
    max_nfev=200,
    n1_tol=0.0,
    ode_opts=None,
    t0_atol=1e-12,
):
    ode_opts = ode_opts or {}
    lb = np.array([np.log(bounds[nm][0]) for nm in active_names], float)
    ub = np.array([np.log(bounds[nm][1]) for nm in active_names], float)

    best = None
    for _ in range(int(n_starts)):
        z0 = rng.uniform(lb, ub)
        res = least_squares(
            fun=_residuals_for_mle,
            x0=z0,
            bounds=(lb, ub),
            args=(t, y),
            kwargs=dict(
                active_names=active_names,
                fixed_values=fixed_values,
                n1_tol=n1_tol,
                ode_opts=ode_opts,
                t0_atol=t0_atol,
            ),
            method="trf",
            loss="linear",
            max_nfev=int(max_nfev),
        )
        if best is None or res.cost < best.cost:
            best = res

    return np.asarray(best.x, float)


# ---------------------------------------------------------------------
# MCMC runner + summaries
# ---------------------------------------------------------------------
def run_mcmc_curve(
    t,
    y,
    *,
    priors,
    bounds,
    fixed_values=None,
    sigma_main=0.03,
    t0_atol=1e-12,
    n1_tol=0.0,
    ode_opts=None,
    # mcmc
    seed=0,
    n_walkers=48,
    n_burn=1500,
    n_steps=3000,
    thin=2,
    use_mle_init=True,
    mle_n_starts=10,
    mle_max_nfev=200,
    init_jitter=0.25,
):
    """
    Main reusable entry point. Returns (chain_z, accept_frac, active_names).

    NOTE: t==0 points are dropped from the likelihood automatically.
    """
    import emcee

    fixed_values = fixed_values or {}
    ode_opts = ode_opts or {}

    names_all = ["lambda", "tau", "beta", "n"]
    active_names = [nm for nm in names_all if nm not in fixed_values]

    rng = np.random.default_rng(int(seed))

    ndim = len(active_names)
    lb = np.array([np.log(bounds[nm][0]) for nm in active_names], float)
    ub = np.array([np.log(bounds[nm][1]) for nm in active_names], float)

    if use_mle_init:
        try:
            z_hat = fit_mle_init(
                t, y, rng,
                active_names=active_names,
                fixed_values=fixed_values,
                bounds=bounds,
                n_starts=mle_n_starts,
                max_nfev=mle_max_nfev,
                n1_tol=n1_tol,
                ode_opts=ode_opts,
                t0_atol=t0_atol,
            )
            p0 = z_hat[None, :] + float(init_jitter) * rng.standard_normal(size=(int(n_walkers), ndim))
            p0 = np.minimum(np.maximum(p0, lb), ub)
        except Exception:
            p0 = rng.uniform(lb, ub, size=(int(n_walkers), ndim))
    else:
        p0 = rng.uniform(lb, ub, size=(int(n_walkers), ndim))

    def _logpost(zvec):
        return log_posterior(
            zvec, t, y,
            active_names=active_names,
            fixed_values=fixed_values,
            priors=priors,
            bounds=bounds,
            sigma_main=sigma_main,
            t0_atol=t0_atol,
            n1_tol=n1_tol,
            ode_opts=ode_opts,
        )

    sampler = emcee.EnsembleSampler(int(n_walkers), ndim, _logpost)
    state = sampler.run_mcmc(p0, int(n_burn), progress=True)
    sampler.reset()
    sampler.run_mcmc(state, int(n_steps), progress=True)

    chain = sampler.get_chain(flat=True, thin=int(thin))
    acc = float(np.mean(sampler.acceptance_fraction))
    return chain, acc, active_names


def summarize_chain(chain_z, active_names, fixed_values):
    """
    Returns medians + (2.5,16,84,97.5) quantiles for each parameter in natural space.
    Fixed params are returned with all quantiles equal to the fixed value.
    """
    qs = [0.025, 0.16, 0.50, 0.84, 0.975]
    out = {}
    theta = np.exp(chain_z)

    for j, nm in enumerate(active_names):
        q = np.quantile(theta[:, j], qs)
        out[f"{nm}_q025"] = float(q[0])
        out[f"{nm}_q16"] = float(q[1])
        out[f"{nm}_med"] = float(q[2])
        out[f"{nm}_q84"] = float(q[3])
        out[f"{nm}_q975"] = float(q[4])

    for nm, val in (fixed_values or {}).items():
        v = float(val)
        out[f"{nm}_q025"] = v
        out[f"{nm}_q16"] = v
        out[f"{nm}_med"] = v
        out[f"{nm}_q84"] = v
        out[f"{nm}_q975"] = v

    return out