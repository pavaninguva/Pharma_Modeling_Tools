#!/usr/bin/env python3
"""
tau_prior_tools.py  (NO PAV)

Monotone smoothing via a small convex QP (cvxpy), then:
  - fit PCHIP on the smoothed monotone curve
  - estimate t* = argmax df/dt (robust plateau median)
  - set tau0 = tau_factor * t*

Key additions for robustness / identifiability:

1) FLAT-SLOPE SWITCH (for near-linear / zero-order-ish cases)
   If max(df/dt) / median(df/dt) < flat_ratio_threshold, then t* is not identifiable
   (the "maximum" is basically everywhere). In that case, we DO NOT pick a late random time.
   Instead, by default we switch to an EARLY-PEAK PRIOR on t*:

      t1 = first strictly-positive sample time (after preprocessing)
      t* ~ Uniform(t_eps, t1)

   This encodes: "peak is early, but data don't resolve where exactly."

2) EARLY-PEAK PRIOR MODE (explicit)
   You can also force the early-uniform prior regardless of flatness by setting:
      use_early_uniform_prior=True

Because downstream Bayesian code typically samples in log-space, we cannot use 0 exactly.
So we implement t* ~ Uniform(t_eps, t1) where:
      t_eps = max(early_eps_abs, early_eps_frac * t1)

Outputs:
  - compute_tau_prior_from_data(...) returns (tau0, t_star_repr, dfdt_max, diag)
  - If an early-uniform prior is active (explicitly or via flat switch),
    diag["tstar_prior_bounds"] and diag["tau_prior_bounds"] are populated.
    In that case, tau0/t_star_repr are just convenient representative midpoints.

Dependencies:
  pip install cvxpy osqp
(optional fallback)
  pip install scs

This module contains ONLY computational bits. Do diagnostics plotting elsewhere.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


# ============================================================
# Preprocess helpers
# ============================================================
def preprocess_xy(t, y, *, clip01=True, ensure_t0=True, t0_atol=1e-12):
    """
    - sort by t
    - average exact duplicate times
    - (optional) clip y to [0,1]
    - (optional) ensure a t=0 point exists and enforce y(0)=0 if present
    Returns (t, y) with strictly increasing t (after duplicate-averaging).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size != y.size:
        raise ValueError("t and y must have the same length")
    if t.size == 0:
        return t.copy(), y.copy()

    # sort
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    # average exact duplicates
    uniq_t, inv = np.unique(t, return_inverse=True)
    if uniq_t.size != t.size:
        y_sum = np.zeros_like(uniq_t, dtype=float)
        cnt = np.zeros_like(uniq_t, dtype=float)
        for i, g in enumerate(inv):
            y_sum[g] += y[i]
            cnt[g] += 1.0
        y = y_sum / np.maximum(cnt, 1.0)
        t = uniq_t

    if clip01:
        y = np.clip(y, 0.0, 1.0)

    if ensure_t0:
        has_t0 = np.any(np.isclose(t, 0.0, atol=t0_atol, rtol=0.0))
        if not has_t0:
            t = np.concatenate([[0.0], t])
            y = np.concatenate([[0.0], y])
        else:
            j0 = np.where(np.isclose(t, 0.0, atol=t0_atol, rtol=0.0))[0][0]
            y[j0] = 0.0

    if t.size >= 2 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing after preprocessing.")
    return t, y


def first_positive_time(t):
    """Return t1 = min{t_i : t_i > 0}, else np.nan."""
    t = np.asarray(t, float)
    pos = t[np.isfinite(t) & (t > 0.0)]
    return float(np.min(pos)) if pos.size else np.nan


# ============================================================
# Monotone smoothing QP (cvxpy)
#   min ||yhat - y||^2 + smooth_lambda * ||D2 yhat||^2
#   s.t. yhat nondecreasing, optional 0<=yhat<=1, yhat(t=0)=0
# ============================================================
def monotone_smooth_qp(
    t,
    y,
    *,
    smooth_lambda=0.0,
    enforce_01=True,
    enforce_t0=True,
    t0_atol=1e-12,
    solver="OSQP",
    verbose=False,
):
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("Need cvxpy. Install: pip install cvxpy osqp (or scs)") from e

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = y.size
    if n < 2:
        return y.copy()

    yhat = cp.Variable(n)

    obj = cp.sum_squares(yhat - y)

    lam = float(smooth_lambda)
    if lam > 0.0 and n >= 3:
        d2 = yhat[:-2] - 2.0 * yhat[1:-1] + yhat[2:]
        obj += lam * cp.sum_squares(d2)

    cons = [yhat[1:] >= yhat[:-1]]
    if enforce_01:
        cons += [yhat >= 0.0, yhat <= 1.0]
    if enforce_t0:
        idx0 = np.where(np.isclose(t, 0.0, atol=t0_atol, rtol=0.0))[0]
        if idx0.size:
            cons.append(yhat[int(idx0[0])] == 0.0)

    prob = cp.Problem(cp.Minimize(obj), cons)

    solver = str(solver).upper()
    try:
        if solver == "OSQP":
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=bool(verbose))
        elif solver == "SCS":
            prob.solve(solver=cp.SCS, warm_start=True, verbose=bool(verbose))
        else:
            prob.solve(warm_start=True, verbose=bool(verbose))
    except Exception:
        prob.solve(solver=cp.SCS, warm_start=True, verbose=bool(verbose))

    if yhat.value is None:
        raise RuntimeError(
            "Monotone QP failed. Try solver='SCS', reduce smooth_lambda, or set enforce_01=False."
        )
    return np.asarray(yhat.value, float)


# ============================================================
# Prior-bound helpers (early-uniform)
# ============================================================
def early_uniform_tstar_bounds_from_t1(
    t,
    *,
    t0_atol=1e-12,
    early_eps_abs=1e-12,
    early_eps_frac=1e-6,
):
    """
    Compute bounds for the belief "peak happens before first measured time":

        t1 = first strictly-positive sample time
        t* ~ Uniform(t_eps, t1)

    where:
        t_eps = max(early_eps_abs, early_eps_frac * t1)

    Returns (t_eps, t1). If t1 unavailable, returns (nan, nan).
    """
    t = np.asarray(t, float)
    # minimal cleanup so first_positive_time behaves well
    t_sorted = np.unique(np.sort(t[np.isfinite(t)]))
    if t_sorted.size == 0:
        return np.nan, np.nan
    # ensure t=0 present doesn't matter here, but keep consistent logic
    if not np.any(np.isclose(t_sorted, 0.0, atol=t0_atol, rtol=0.0)):
        t_sorted = np.concatenate([[0.0], t_sorted])

    t1 = first_positive_time(t_sorted)
    if not np.isfinite(t1) or t1 <= 0.0:
        return np.nan, np.nan

    t_eps = max(float(early_eps_abs), float(early_eps_frac) * float(t1))
    if t_eps >= t1:
        # degenerate; make eps safely below t1
        t_eps = 0.5 * t1
    return float(t_eps), float(t1)


# ============================================================
# Flatness helper
# ============================================================
def _flat_ratio(dfdt, *, eps=1e-12):
    """
    Compute max(dfdt)/median(dfdt) on positive finite dfdt values.
    Returns (ratio, med, mx). If insufficient positives, returns (nan, nan, nan).
    """
    dfdt = np.asarray(dfdt, float)
    pos = dfdt[np.isfinite(dfdt) & (dfdt > 0.0)]
    if pos.size < 5:
        return np.nan, np.nan, np.nan
    med = float(np.median(pos))
    mx = float(np.max(pos))
    ratio = mx / max(med, eps)
    return ratio, med, mx


# ============================================================
# Main estimator: monotone spline peak OR early-uniform prior
# ============================================================
def estimate_tstar_from_monotone_spline(
    t,
    y,
    *,
    smooth_lambda=0.0,
    dense=4000,
    peak_frac=0.98,
    exclude_boundaries=True,
    enforce_01=True,
    t0_atol=1e-12,
    solver="OSQP",
    # boundary handling: shrink by epsilon instead of jumping to t[1]
    boundary_eps_frac=1e-4,
    boundary_eps_abs=1e-12,
    # flat slope detection
    flat_ratio_threshold=1.20,
    # if flat, switch to early-uniform t* prior (recommended)
    flat_switch_to_early_uniform=True,
    flat_early_eps_abs=1e-12,
    flat_early_eps_frac=1e-6,
    # explicit early-uniform prior mode (always)
    use_early_uniform_prior=False,
    early_eps_abs=1e-12,
    early_eps_frac=1e-6,
):
    """
    Returns: t_star_repr, dfdt_max, diag

    diag includes:
      - t, y_obs, y_mono, tg, yhat, dfdt
      - t1
      - flat_ratio, flat_median, flat_max
      - prior_mode in {"spline_peak", "early_uniform", "flat_early_uniform"}
      - tstar_prior_bounds (if early_uniform or flat_early_uniform)
    """
    t, y_obs = preprocess_xy(t, y, ensure_t0=True, t0_atol=t0_atol)
    t1 = first_positive_time(t)

    # -------- Explicit early-uniform prior mode --------
    if bool(use_early_uniform_prior):
        t_eps, t1b = early_uniform_tstar_bounds_from_t1(
            t, t0_atol=t0_atol, early_eps_abs=early_eps_abs, early_eps_frac=early_eps_frac
        )
        if np.isfinite(t_eps) and np.isfinite(t1b) and (t1b > t_eps):
            t_star_repr = 0.5 * (t_eps + t1b)  # representative midpoint
        else:
            t_star_repr = np.nan

        diag = dict(
            t=t, y_obs=y_obs, y_mono=None,
            tg=np.array([], float), yhat=np.array([], float), dfdt=np.array([], float),
            t1=float(t1) if np.isfinite(t1) else np.nan,
            flat_ratio=np.nan, flat_median=np.nan, flat_max=np.nan,
            prior_mode="early_uniform",
            tstar_prior_bounds=(float(t_eps), float(t1b)) if np.isfinite(t_star_repr) else (np.nan, np.nan),
        )
        return float(t_star_repr), np.nan, diag

    # -------- Monotone smoothing + PCHIP derivative --------
    y_mono = monotone_smooth_qp(
        t,
        y_obs,
        smooth_lambda=smooth_lambda,
        enforce_01=enforce_01,
        enforce_t0=True,
        t0_atol=t0_atol,
        solver=solver,
    )

    pchip = PchipInterpolator(t, y_mono, extrapolate=False)
    dp = pchip.derivative(1)

    # search window
    t_min = float(t[0])
    t_max = float(t[-1])

    if exclude_boundaries and np.isfinite(t_max - t_min) and (t_max > t_min):
        horizon = t_max - t_min
        eps = max(float(boundary_eps_abs), float(boundary_eps_frac) * float(horizon))
        t_min2 = t_min + eps
        t_max2 = t_max - eps
        if t_max2 > t_min2:
            t_min, t_max = float(t_min2), float(t_max2)

    tg = np.linspace(t_min, t_max, int(dense))
    yhat = pchip(tg)
    dfdt = dp(tg)

    mask = np.isfinite(dfdt)
    if not np.any(mask):
        diag = dict(
            t=t, y_obs=y_obs, y_mono=y_mono,
            tg=tg, yhat=yhat, dfdt=dfdt,
            t1=float(t1) if np.isfinite(t1) else np.nan,
            flat_ratio=np.nan, flat_median=np.nan, flat_max=np.nan,
            prior_mode="spline_peak",
            tstar_prior_bounds=None,
        )
        return np.nan, np.nan, diag

    tg2 = tg[mask]
    y2 = yhat[mask]
    df2 = np.asarray(dfdt[mask], float)

    dfdt_max = float(np.max(df2)) if df2.size else np.nan

    # -------- Flatness check --------
    ratio, med, mx = _flat_ratio(df2)
    if np.isfinite(ratio) and ratio < float(flat_ratio_threshold):
        # slope is essentially flat => t* not identifiable from df/dt
        if bool(flat_switch_to_early_uniform):
            t_eps, t1b = early_uniform_tstar_bounds_from_t1(
                t,
                t0_atol=t0_atol,
                early_eps_abs=flat_early_eps_abs,
                early_eps_frac=flat_early_eps_frac,
            )
            if np.isfinite(t_eps) and np.isfinite(t1b) and (t1b > t_eps):
                t_star_repr = 0.5 * (t_eps + t1b)
                diag = dict(
                    t=t, y_obs=y_obs, y_mono=y_mono,
                    tg=tg2, yhat=y2, dfdt=df2,
                    t1=float(t1b),
                    flat_ratio=float(ratio), flat_median=float(med), flat_max=float(mx),
                    prior_mode="flat_early_uniform",
                    tstar_prior_bounds=(float(t_eps), float(t1b)),
                )
                return float(t_star_repr), float(dfdt_max), diag

        # deterministic fallback (still "small"): take earliest interior time
        if np.isfinite(t_max - t_min) and (t_max > t_min):
            horizon = t_max - t_min
            eps = max(float(boundary_eps_abs), float(boundary_eps_frac) * float(horizon))
            t_star = float(t_min + eps)
        else:
            t_star = float(t_min)

        diag = dict(
            t=t, y_obs=y_obs, y_mono=y_mono,
            tg=tg2, yhat=y2, dfdt=df2,
            t1=float(t1) if np.isfinite(t1) else np.nan,
            flat_ratio=float(ratio), flat_median=float(med), flat_max=float(mx),
            prior_mode="flat_point_fallback",
            tstar_prior_bounds=None,
        )
        return float(t_star), float(dfdt_max), diag

    # -------- Peak / plateau selection --------
    if not np.isfinite(dfdt_max) or dfdt_max <= 0.0:
        # weird case; just return argmax index
        j = int(np.argmax(df2))
        t_star = float(tg2[j])
        diag = dict(
            t=t, y_obs=y_obs, y_mono=y_mono,
            tg=tg2, yhat=y2, dfdt=df2,
            t1=float(t1) if np.isfinite(t1) else np.nan,
            flat_ratio=float(ratio) if np.isfinite(ratio) else np.nan,
            flat_median=float(med) if np.isfinite(med) else np.nan,
            flat_max=float(mx) if np.isfinite(mx) else np.nan,
            prior_mode="spline_peak",
            tstar_prior_bounds=None,
        )
        return float(t_star), float(dfdt_max), diag

    thr = float(peak_frac) * float(dfdt_max)
    plateau = tg2[df2 >= thr]
    t_star = float(np.median(plateau)) if plateau.size else float(tg2[int(np.argmax(df2))])

    diag = dict(
        t=t, y_obs=y_obs, y_mono=y_mono,
        tg=tg2, yhat=y2, dfdt=df2,
        t1=float(t1) if np.isfinite(t1) else np.nan,
        flat_ratio=float(ratio) if np.isfinite(ratio) else np.nan,
        flat_median=float(med) if np.isfinite(med) else np.nan,
        flat_max=float(mx) if np.isfinite(mx) else np.nan,
        prior_mode="spline_peak",
        tstar_prior_bounds=None,
    )
    return float(t_star), float(dfdt_max), diag


def compute_tau_prior_from_data(
    t,
    y_obs,
    *,
    tau_factor=0.5,
    tau_bounds=None,  # e.g. (1e-6, 1e4) if you want clipping
    smooth_lambda=0.0,
    dense=4000,
    peak_frac=0.98,
    exclude_boundaries=True,
    enforce_01=True,
    t0_atol=1e-12,
    solver="OSQP",
    boundary_eps_frac=1e-4,
    boundary_eps_abs=1e-12,
    # flatness switch
    flat_ratio_threshold=1.20,
    flat_switch_to_early_uniform=True,
    flat_early_eps_abs=1e-12,
    flat_early_eps_frac=1e-6,
    # explicit early-uniform mode
    use_early_uniform_prior=False,
    early_eps_abs=1e-12,
    early_eps_frac=1e-6,
):
    """
    Returns: tau0, t_star_repr, dfdt_max, diag

    If prior_mode in {"early_uniform", "flat_early_uniform"}:
      diag["tstar_prior_bounds"] = (t_eps, t1)
      diag["tau_prior_bounds"]   = (tau_factor*t_eps, tau_factor*t1)
      tau0 and t_star_repr are representative midpoints (useful defaults),
      but for Bayesian sampling you should use the BOUNDS as the prior.
    """
    t_star_repr, dfdt_max, diag = estimate_tstar_from_monotone_spline(
        t,
        y_obs,
        smooth_lambda=smooth_lambda,
        dense=dense,
        peak_frac=peak_frac,
        exclude_boundaries=exclude_boundaries,
        enforce_01=enforce_01,
        t0_atol=t0_atol,
        solver=solver,
        boundary_eps_frac=boundary_eps_frac,
        boundary_eps_abs=boundary_eps_abs,
        flat_ratio_threshold=flat_ratio_threshold,
        flat_switch_to_early_uniform=flat_switch_to_early_uniform,
        flat_early_eps_abs=flat_early_eps_abs,
        flat_early_eps_frac=flat_early_eps_frac,
        use_early_uniform_prior=use_early_uniform_prior,
        early_eps_abs=early_eps_abs,
        early_eps_frac=early_eps_frac,
    )

    tf = float(tau_factor)

    # early-uniform prior bounds
    if diag.get("prior_mode", "") in ("early_uniform", "flat_early_uniform"):
        tlo, thi = diag.get("tstar_prior_bounds", (np.nan, np.nan))
        tau_lo = tf * float(tlo) if np.isfinite(tlo) else np.nan
        tau_hi = tf * float(thi) if np.isfinite(thi) else np.nan

        if tau_bounds is not None and np.isfinite(tau_lo) and np.isfinite(tau_hi):
            lo_b, hi_b = float(tau_bounds[0]), float(tau_bounds[1])
            tau_lo = float(np.clip(tau_lo, lo_b, hi_b))
            tau_hi = float(np.clip(tau_hi, lo_b, hi_b))
            if tau_hi < tau_lo:
                tau_lo, tau_hi = tau_hi, tau_lo

        diag["tau_prior_bounds"] = (float(tau_lo), float(tau_hi))

        # representative midpoint
        tau0 = 0.5 * (tau_lo + tau_hi) if np.isfinite(tau_lo) and np.isfinite(tau_hi) else np.nan
        t_star_mid = 0.5 * (float(tlo) + float(thi)) if np.isfinite(tlo) and np.isfinite(thi) else np.nan
        return float(tau0), float(t_star_mid), float(dfdt_max), diag

    # deterministic tau0
    tau0 = tf * float(t_star_repr) if np.isfinite(t_star_repr) else np.nan

    if tau_bounds is not None and np.isfinite(tau0):
        lo_b, hi_b = float(tau_bounds[0]), float(tau_bounds[1])
        tau0 = float(np.clip(tau0, lo_b, hi_b))

    return float(tau0), float(t_star_repr), float(dfdt_max), diag
