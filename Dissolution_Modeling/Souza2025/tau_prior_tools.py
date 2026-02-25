#!/usr/bin/env python3
"""
tau_prior_tools.py (cleaned)

Goal:
  Given (t, y) for a monotone release curve (0..1), compute a robust lognormal prior
  for tau in the n=1 model:
      df/dt = lam * (t/(tau+t)) * (1-f)^beta

Approach:
  1) Preprocess: sort, average duplicate times, clip y, enforce (t=0,y=0)
  2) Monotone smoothing:
       - Prefer convex QP (cvxpy) with optional curvature penalty
       - Fallback to monotone cumulative-max if cvxpy unavailable or QP fails
  3) Fit PCHIP to smoothed y(t), evaluate df/dt on dense grid, pick t*:
       - t* = median of plateau {t: df/dt >= peak_frac * max(df/dt)}
  4) Flat-slope detection:
       - If max(df/dt)/median(df/dt) < flat_ratio_threshold, derivative peak is not identifiable.
       - In that case, choose an "early" representative t* in (t_eps, t1) where:
           t1 = first strictly-positive sample time
           t_eps = max(early_eps_abs, early_eps_frac * t1)
  5) Return lognormal prior for tau:
       log(tau) ~ Normal(mu=log(tau_factor * t_star_repr), sig^2)
       - If flat, use a larger sig (flat_logsig) for robustness.

Outputs:
  compute_tau_lognormal_prior(...) returns:
    (mu, sig, tau0, t_star_repr, dfdt_max, diag)

where:
  - mu, sig define log(tau) prior
  - tau0 = exp(mu) is a representative center
  - diag includes diagnostics + optional arrays if include_arrays=True

Dependencies:
  - numpy, scipy (PchipInterpolator)
  - optional: cvxpy + osqp (or scs) for QP smoothing

This module contains only computational bits (no plotting).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator


# ============================================================
# Data structures
# ============================================================
@dataclass(frozen=True)
class TauPriorResult:
    mu: float                 # mean of log(tau)
    sig: float                # std of log(tau)
    tau0: float               # exp(mu)
    t_star_repr: float        # representative peak-time used to build tau0
    dfdt_max: float           # max derivative (from PCHIP on smoothed curve)
    diag: dict                # diagnostics


# ============================================================
# Preprocess
# ============================================================
def preprocess_xy(
    t,
    y,
    *,
    clip01: bool = True,
    ensure_t0: bool = True,
    t0_atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    - sort by t
    - average exact duplicate times
    - optionally clip y to [0,1]
    - optionally ensure a t=0 point exists and enforce y(0)=0
    Returns strictly increasing t after duplicate averaging.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if t.size != y.size:
        raise ValueError("t and y must have the same length")
    if t.size == 0:
        return t.copy(), y.copy()

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


def first_positive_time(t: np.ndarray) -> float:
    """Return t1 = min{t_i : t_i > 0}, else nan."""
    t = np.asarray(t, float)
    pos = t[np.isfinite(t) & (t > 0.0)]
    return float(np.min(pos)) if pos.size else np.nan


# ============================================================
# Monotone smoothing
# ============================================================
def monotone_smooth(
    t: np.ndarray,
    y: np.ndarray,
    *,
    smooth_lambda: float = 0.0,
    enforce_01: bool = True,
    enforce_t0: bool = True,
    t0_atol: float = 1e-12,
    solver: str = "OSQP",
    verbose: bool = False,
    fallback: str = "cummax",
) -> tuple[np.ndarray, dict]:
    """
    Prefer cvxpy-QP smoothing; fallback to a simple monotone projection.

    Returns:
      y_mono, info
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = y.size
    info = {"method": None, "qp_ok": False, "qp_status": None}

    if n < 2:
        info["method"] = "none"
        return y.copy(), info

    # --- Try QP (cvxpy) ---
    try:
        import cvxpy as cp
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

        s = str(solver).upper()
        try:
            if s == "OSQP":
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=bool(verbose))
            elif s == "SCS":
                prob.solve(solver=cp.SCS, warm_start=True, verbose=bool(verbose))
            else:
                prob.solve(warm_start=True, verbose=bool(verbose))
        except Exception:
            prob.solve(solver=cp.SCS, warm_start=True, verbose=bool(verbose))

        info["qp_status"] = str(prob.status)
        if yhat.value is not None:
            info["method"] = "qp"
            info["qp_ok"] = True
            return np.asarray(yhat.value, float), info

    except Exception as e:
        info["qp_status"] = f"qp_failed: {type(e).__name__}"

    # --- Fallback: simple monotone projection ---
    if str(fallback).lower() == "cummax":
        y_mono = np.maximum.accumulate(y)
        if enforce_01:
            y_mono = np.clip(y_mono, 0.0, 1.0)
        if enforce_t0:
            idx0 = np.where(np.isclose(t, 0.0, atol=t0_atol, rtol=0.0))[0]
            if idx0.size:
                y_mono[int(idx0[0])] = 0.0
        info["method"] = "cummax"
        return y_mono, info

    raise RuntimeError("Monotone smoothing failed and no valid fallback specified.")


# ============================================================
# Peak + flatness
# ============================================================
def flat_ratio(dfdt: np.ndarray, *, eps: float = 1e-12) -> tuple[float, float, float]:
    """
    Compute ratio = max(dfdt)/median(dfdt) using positive finite dfdt values.
    Returns (ratio, median, max). If insufficient data, returns (nan, nan, nan).
    """
    dfdt = np.asarray(dfdt, float)
    pos = dfdt[np.isfinite(dfdt) & (dfdt > 0.0)]
    if pos.size < 5:
        return np.nan, np.nan, np.nan
    med = float(np.median(pos))
    mx = float(np.max(pos))
    return float(mx / max(med, eps)), med, mx


def estimate_tstar_from_smoothed(
    t: np.ndarray,
    y_mono: np.ndarray,
    *,
    dense: int = 4000,
    peak_frac: float = 0.98,
    exclude_boundaries: bool = True,
    boundary_eps_frac: float = 1e-4,
    boundary_eps_abs: float = 1e-12,
) -> tuple[float, float, dict]:
    """
    Fit PCHIP to (t, y_mono), evaluate df/dt on dense grid, return:
      t_star (plateau median), dfdt_max, arrays dict
    """
    t = np.asarray(t, float)
    y_mono = np.asarray(y_mono, float)

    p = PchipInterpolator(t, y_mono, extrapolate=False)
    dp = p.derivative(1)

    t_min = float(t[0])
    t_max = float(t[-1])

    if exclude_boundaries and (t_max > t_min):
        horizon = t_max - t_min
        eps = max(float(boundary_eps_abs), float(boundary_eps_frac) * horizon)
        t_min2 = t_min + eps
        t_max2 = t_max - eps
        if t_max2 > t_min2:
            t_min, t_max = t_min2, t_max2

    dense = int(max(50, dense))
    tg = np.linspace(t_min, t_max, dense)
    yhat = p(tg)
    dfdt = dp(tg)

    m = np.isfinite(dfdt)
    if not np.any(m):
        return np.nan, np.nan, {"tg": tg, "yhat": yhat, "dfdt": dfdt}

    tg2 = tg[m]
    y2 = yhat[m]
    df2 = np.asarray(dfdt[m], float)

    dfdt_max = float(np.max(df2)) if df2.size else np.nan
    if not np.isfinite(dfdt_max) or dfdt_max <= 0.0:
        j = int(np.argmax(df2))
        return float(tg2[j]), float(dfdt_max), {"tg": tg2, "yhat": y2, "dfdt": df2}

    thr = float(peak_frac) * dfdt_max
    plateau = tg2[df2 >= thr]
    t_star = float(np.median(plateau)) if plateau.size else float(tg2[int(np.argmax(df2))])
    return float(t_star), float(dfdt_max), {"tg": tg2, "yhat": y2, "dfdt": df2}


# ============================================================
# Main: compute lognormal prior for tau
# ============================================================
def compute_tau_lognormal_prior(
    t,
    y,
    *,
    tau_factor: float = 1.0,
    tau_bounds: tuple[float, float] | None = None,
    # smoothing + peak detection
    smooth_lambda: float = 0.0,
    solver: str = "OSQP",
    enforce_01: bool = True,
    dense: int = 4000,
    peak_frac: float = 0.98,
    exclude_boundaries: bool = True,
    boundary_eps_frac: float = 1e-4,
    boundary_eps_abs: float = 1e-12,
    # flatness detection
    flat_ratio_threshold: float = 1.15,
    # flat -> early representative time in (t_eps, t1)
    early_eps_abs: float = 1e-12,
    early_eps_frac: float = 1e-6,
    # prior widths
    logsig: float = 0.10,        # default (non-flat)
    flat_logsig: float = 0.80,   # inflated when flat
    # diagnostics
    include_arrays: bool = False,
    t0_atol: float = 1e-12,
) -> TauPriorResult:
    """
    Returns a lognormal prior for tau:
      log(tau) ~ Normal(mu, sig^2)
    where tau is anchored to tau_factor * t_star_repr.

    If derivative is flat, t_star_repr is chosen early (before first measured time)
    and sig is increased to flat_logsig.
    """
    t, y_obs = preprocess_xy(t, y, clip01=True, ensure_t0=True, t0_atol=t0_atol)
    t1 = first_positive_time(t)

    y_mono, smooth_info = monotone_smooth(
        t, y_obs,
        smooth_lambda=smooth_lambda,
        enforce_01=enforce_01,
        enforce_t0=True,
        t0_atol=t0_atol,
        solver=solver,
        verbose=False,
        fallback="cummax",
    )

    t_star, dfdt_max, arr = estimate_tstar_from_smoothed(
        t, y_mono,
        dense=dense,
        peak_frac=peak_frac,
        exclude_boundaries=exclude_boundaries,
        boundary_eps_frac=boundary_eps_frac,
        boundary_eps_abs=boundary_eps_abs,
    )

    # flatness ratio based on dfdt evaluated on grid
    ratio, med, mx = (np.nan, np.nan, np.nan)
    prior_mode = "spline_peak"
    t_star_repr = float(t_star)
    sig = float(logsig)

    if include_arrays and isinstance(arr, dict) and "dfdt" in arr:
        ratio, med, mx = flat_ratio(arr["dfdt"])
    else:
        # compute ratio without storing arrays: quickly re-evaluate derivative on grid
        # (cheap compared to MCMC; keeps diag lightweight by default)
        if np.isfinite(t_star) and np.isfinite(dfdt_max):
            ratio, med, mx = (np.nan, np.nan, np.nan)  # leave nan if not requested
        # If you want ratio always computed, set include_arrays=True.

    # If include_arrays=True, we can switch based on ratio.
    if include_arrays and np.isfinite(ratio) and ratio < float(flat_ratio_threshold):
        prior_mode = "flat_lognormal"
        sig = float(flat_logsig)

        # pick early representative time in (t_eps, t1)
        if np.isfinite(t1) and t1 > 0:
            t_eps = max(float(early_eps_abs), float(early_eps_frac) * float(t1))
            if t_eps >= t1:
                t_eps = 0.5 * t1
            t_star_repr = 0.5 * (t_eps + float(t1))
        else:
            # no positive times => fallback to t_star if we have it
            t_star_repr = float(t_star) if np.isfinite(t_star) else np.nan

    # build tau0 and mu
    tau0 = float(tau_factor) * float(t_star_repr) if np.isfinite(t_star_repr) else np.nan
    if tau_bounds is not None and np.isfinite(tau0):
        lo, hi = float(tau_bounds[0]), float(tau_bounds[1])
        tau0 = float(np.clip(tau0, lo, hi))

    mu = float(np.log(tau0)) if (np.isfinite(tau0) and tau0 > 0) else np.nan

    diag = {
        "prior_mode": prior_mode,
        "t1": float(t1) if np.isfinite(t1) else np.nan,
        "t_star": float(t_star) if np.isfinite(t_star) else np.nan,
        "t_star_repr": float(t_star_repr) if np.isfinite(t_star_repr) else np.nan,
        "tau_factor": float(tau_factor),
        "tau0": float(tau0) if np.isfinite(tau0) else np.nan,
        "mu": float(mu) if np.isfinite(mu) else np.nan,
        "sig": float(sig),
        "smooth_lambda": float(smooth_lambda),
        "smooth_method": smooth_info.get("method", None),
        "qp_ok": bool(smooth_info.get("qp_ok", False)),
        "qp_status": smooth_info.get("qp_status", None),
        "dfdt_max": float(dfdt_max) if np.isfinite(dfdt_max) else np.nan,
    }

    if include_arrays:
        diag.update({
            "t": t,
            "y_obs": y_obs,
            "y_mono": y_mono,
            "tg": arr.get("tg", np.array([], float)),
            "yhat": arr.get("yhat", np.array([], float)),
            "dfdt": arr.get("dfdt", np.array([], float)),
            "flat_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
            "flat_median": float(med) if np.isfinite(med) else np.nan,
            "flat_max": float(mx) if np.isfinite(mx) else np.nan,
        })

    return TauPriorResult(
        mu=float(mu),
        sig=float(sig),
        tau0=float(tau0) if np.isfinite(tau0) else np.nan,
        t_star_repr=float(t_star_repr) if np.isfinite(t_star_repr) else np.nan,
        dfdt_max=float(dfdt_max) if np.isfinite(dfdt_max) else np.nan,
        diag=diag,
    )