#!/usr/bin/env python3
"""
souza_bayes_three_scenarios_runner.py  (UPDATED: S4 lambda-only + 1D scatter)

Uses bayes_activation_framework.py where:
- t=0 is ALWAYS dropped from the likelihood (treated as IC).

Scenarios:
S1) uniform_all: estimate (lambda, tau, beta) with uniform natural priors; fix n=1
S2) tau_prior:  estimate (lambda, tau, beta) with uniform priors for (lambda,beta),
                tau gets a lognormal prior on log(tau) from spline estimator; fix n=1
S3) tau_fixed:  fix tau = TAU_FIXED_VALUE; estimate (lambda, beta); fix n=1
S4) lambda_only: fix tau = TAU_FIXED_VALUE and beta = 1.0; estimate lambda only; fix n=1

Final output:
  - parameter scatter across scenarios:
      * subplot 1 (3D): S1 (log10 lambda, log10 tau, log10 beta)
      * subplot 2 (3D): S2 (log10 lambda, log10 tau, log10 beta)
      * subplot 3 (2D): S3 (log10 lambda, log10 beta)
      * subplot 4 (1D): S4 (log10 lambda) strip plot
"""

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tau_prior_tools import compute_tau_lognormal_prior

from bayesian_tools import *

# ---------------------------
# USER SETTINGS
# ---------------------------
CSV_PATH = "Souza2025_TableS1_Final.csv"
OUT_DIR  = "souza_bayes_four_scenarios_modular_n"

BATCH_COL = "BatchID"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

RUN_S1 = True
RUN_S2 = True
RUN_S3 = True
RUN_S4 = True

TAU_FIXED_VALUE  = 1.0
BETA_FIXED_VALUE = 1.0  # for S4

# Likelihood (t=0 dropped inside framework)
SIGMA_MAIN = 0.03
T0_ATOL    = 1e-12

# Bounds (natural)
BOUNDS = {
    "lambda": (1e-6, 1e2),
    "tau":    (1e-3, 1e4),
    "beta":   (1e-3, 1e1),
    "n":      (1e-3, 5.0),
}

# MCMC
SEED = 42
N_WALKERS = 48
N_BURN    = 1500
N_STEPS   = 3000
THIN      = 2
USE_MLE_INIT = True
N_STARTS_MLE = 20
MAX_NFEV_MLE = 300
INIT_JITTER  = 0.25

# Plots
MAKE_PLOTS = True
PLOT_MAX_PER_AX = 10
PLOT_LIMIT = None
DPI = 250

# Tau-prior knobs (scenario 2)
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


# ---------------------------
# Plotting helpers
# ---------------------------
def rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def plot_overview(curves, out_png, max_per_ax=10, title=""):
    n = len(curves)
    if n == 0:
        return
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


def safe_log10(x):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan, dtype=float)
    m = x > 0
    out[m] = np.log10(x[m])
    return out


def plot_parameter_scatter(df_s1, df_s2, df_s3, df_s4, out_png):
    fig = plt.figure(figsize=(20.0, 5.2), dpi=DPI)

    # --- S1 (3D)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.scatter(
        safe_log10(df_s1["lambda_med"]),
        safe_log10(df_s1["tau_med"]),
        safe_log10(df_s1["beta_med"]),
        s=18, alpha=0.85
    )
    ax1.set_xlabel(r"$\log_{10}(\lambda)$")
    ax1.set_ylabel(r"$\log_{10}(\tau)$")
    ax1.set_zlabel(r"$\log_{10}(\beta)$")
    ax1.set_title("S1: uniform (λ,τ,β), n=1")

    # --- S2 (3D)
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2.scatter(
        safe_log10(df_s2["lambda_med"]),
        safe_log10(df_s2["tau_med"]),
        safe_log10(df_s2["beta_med"]),
        s=18, alpha=0.85
    )
    ax2.set_xlabel(r"$\log_{10}(\lambda)$")
    ax2.set_ylabel(r"$\log_{10}(\tau)$")
    ax2.set_zlabel(r"$\log_{10}(\beta)$")
    ax2.set_title("S2: τ prior (spline), n=1")

    # --- S3 (2D): tau fixed
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(
        safe_log10(df_s3["lambda_med"]),
        safe_log10(df_s3["beta_med"]),
        s=18, alpha=0.85
    )
    ax3.set_xlabel(r"$\log_{10}(\lambda)$")
    ax3.set_ylabel(r"$\log_{10}(\beta)$")
    ax3.set_title(f"S3: τ fixed={TAU_FIXED_VALUE:g}, n=1")

    # --- S4 (1D): lambda only (strip plot)
    ax4 = fig.add_subplot(1, 4, 4)
    x = safe_log10(df_s4["lambda_med"].to_numpy(float))
    rng = np.random.default_rng(0)
    yj = 0.03 * rng.standard_normal(size=x.size)  # tiny vertical jitter
    ax4.scatter(x, yj, s=18, alpha=0.85)
    ax4.set_xlabel(r"$\log_{10}(\lambda)$")
    ax4.set_yticks([])
    ax4.set_title(f"S4: only λ (τ={TAU_FIXED_VALUE:g}, β={BETA_FIXED_VALUE:g})")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Scenario 2 tau prior estimator
# ---------------------------
def estimate_tau_prior_from_curve(t, y):
    res = compute_tau_lognormal_prior(
        t, y,
        tau_factor=TAU_PRIOR_FACTOR,
        tau_bounds=BOUNDS["tau"],
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
    return float(res.mu), float(res.sig), diag


# ---------------------------
# Run one scenario (generic)
# ---------------------------
def run_scenario(df_all, *, scenario_key, priors, fixed_values):
    scen_dir = os.path.join(OUT_DIR, scenario_key)
    os.makedirs(scen_dir, exist_ok=True)

    out_csv = os.path.join(scen_dir, f"souza_bayes_params_{scenario_key}.csv")
    out_png = os.path.join(scen_dir, f"souza_fit_overview_{scenario_key}.png")

    df_all = df_all.copy()
    df_all[BATCH_COL] = df_all[BATCH_COL].astype(str)
    batch_ids = sorted(df_all[BATCH_COL].unique().tolist())

    rows = []
    curves_for_plot = []

    for bid in batch_ids:
        dfi = df_all[df_all[BATCH_COL] == bid].sort_values(TIME_COL)
        t_raw = dfi[TIME_COL].to_numpy(float)
        y_raw = dfi[Y_COL].to_numpy(float)

        try:
            t, y = preprocess_for_likelihood(t_raw, y_raw)
        except Exception as e:
            print(f"[{scenario_key}][skip] {bid}: preprocess failed ({e})")
            continue
        if len(t) < 4:
            print(f"[{scenario_key}][skip] {bid}: too few points ({len(t)})")
            continue

        print(f"\n[{scenario_key}][MCMC] {bid} points={len(t)}")

        chain, acc, active_names = run_mcmc_curve(
            t, y,
            priors=priors,
            bounds=BOUNDS,
            fixed_values=fixed_values,
            sigma_main=SIGMA_MAIN,
            t0_atol=T0_ATOL,
            n1_tol=0.0,
            ode_opts=None,
            seed=SEED + (hash(bid) % 100000),
            n_walkers=N_WALKERS,
            n_burn=N_BURN,
            n_steps=N_STEPS,
            thin=THIN,
            use_mle_init=USE_MLE_INIT,
            mle_n_starts=N_STARTS_MLE,
            mle_max_nfev=MAX_NFEV_MLE,
            init_jitter=INIT_JITTER,
        )
        summ = summarize_chain(chain, active_names, fixed_values)

        lam_med  = summ["lambda_med"]
        tau_med  = summ["tau_med"]
        beta_med = summ["beta_med"]
        n_med    = summ["n_med"]

        y_pred = forward_closed_form_n1(t, lam_med, tau_med, beta_med)
        r = rmse(y_pred, y)

        rows.append({
            "Scenario": scenario_key,
            "BatchID": bid,
            "n_points": int(len(t)),
            "t_min": float(np.min(t)),
            "t_max": float(np.max(t)),
            "accept_frac": float(acc),
            "rmse_med_curve": float(r),
            **summ
        })

        if MAKE_PLOTS:
            curves_for_plot.append({"bid": bid, "t": t, "y": y, "y_pred": y_pred})

        print(f"[{scenario_key}][done] {bid}: rmse={r:.4f} acc~{acc:.3f} "
              f"lam~{lam_med:.3g} tau~{tau_med:.3g} beta~{beta_med:.3g} n~{n_med:.3g}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[{scenario_key}] Saved: {out_csv}")

    if MAKE_PLOTS and len(curves_for_plot):
        curves = curves_for_plot[:int(PLOT_LIMIT)] if PLOT_LIMIT is not None else curves_for_plot
        plot_overview(curves, out_png, max_per_ax=PLOT_MAX_PER_AX, title=scenario_key)
        print(f"[{scenario_key}] Saved overview plot: {out_png}")

    return out_csv


# ---------------------------
# MAIN
# ---------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    if not (BOUNDS["tau"][0] <= float(TAU_FIXED_VALUE) <= BOUNDS["tau"][1]):
        raise ValueError(f"TAU_FIXED_VALUE={TAU_FIXED_VALUE} must be within BOUNDS['tau']={BOUNDS['tau']}")
    if not (BOUNDS["beta"][0] <= float(BETA_FIXED_VALUE) <= BOUNDS["beta"][1]):
        raise ValueError(f"BETA_FIXED_VALUE={BETA_FIXED_VALUE} must be within BOUNDS['beta']={BOUNDS['beta']}")

    fixed_n1 = {"n": 1.0}
    csv_s1 = csv_s2 = csv_s3 = csv_s4 = None

    # --- S1
    if RUN_S1:
        priors_s1 = {"lambda": {"type": "uniform"}, "tau": {"type": "uniform"}, "beta": {"type": "uniform"}}
        csv_s1 = run_scenario(df, scenario_key="S1_uniform_all_n1", priors=priors_s1, fixed_values=dict(fixed_n1))

    # --- S2 (per-curve tau prior)
    if RUN_S2:
        scen_key = "S2_tau_prior_n1"
        scen_dir = os.path.join(OUT_DIR, scen_key)
        os.makedirs(scen_dir, exist_ok=True)
        out_csv = os.path.join(scen_dir, f"souza_bayes_params_{scen_key}.csv")
        out_png = os.path.join(scen_dir, f"souza_fit_overview_{scen_key}.png")

        rows = []
        curves_for_plot = []
        df[BATCH_COL] = df[BATCH_COL].astype(str)

        for bid in sorted(df[BATCH_COL].unique().tolist()):
            dfi = df[df[BATCH_COL] == bid].sort_values(TIME_COL)
            t_raw = dfi[TIME_COL].to_numpy(float)
            y_raw = dfi[Y_COL].to_numpy(float)

            try:
                t, y = preprocess_for_likelihood(t_raw, y_raw)
            except Exception:
                continue
            if len(t) < 4:
                continue

            try:
                mu_tau, sig_tau, diag = estimate_tau_prior_from_curve(t, y)
                tau_prior = {"type": "normal_log", "mu": mu_tau, "sig": sig_tau}
                tau_mode = str(diag.get("prior_mode", ""))
            except Exception as e:
                tau_prior = {"type": "uniform"}
                mu_tau, sig_tau = np.nan, np.nan
                tau_mode = f"failed:{e}"

            priors_s2 = {"lambda": {"type": "uniform"}, "tau": tau_prior, "beta": {"type": "uniform"}}
            fixed_s2 = dict(fixed_n1)

            print(f"\n[{scen_key}][MCMC] {bid}  tau_prior_mode={tau_mode}")

            chain, acc, active_names = run_mcmc_curve(
                t, y,
                priors=priors_s2,
                bounds=BOUNDS,
                fixed_values=fixed_s2,
                sigma_main=SIGMA_MAIN,
                t0_atol=T0_ATOL,
                n1_tol=0.0,
                ode_opts=None,
                seed=SEED + (hash(bid) % 100000) + 999,
                n_walkers=N_WALKERS,
                n_burn=N_BURN,
                n_steps=N_STEPS,
                thin=THIN,
                use_mle_init=USE_MLE_INIT,
                mle_n_starts=N_STARTS_MLE,
                mle_max_nfev=MAX_NFEV_MLE,
                init_jitter=INIT_JITTER,
            )
            summ = summarize_chain(chain, active_names, fixed_s2)

            lam_med, tau_med, beta_med = summ["lambda_med"], summ["tau_med"], summ["beta_med"]
            y_pred = forward_closed_form_n1(t, lam_med, tau_med, beta_med)
            r = rmse(y_pred, y)

            rows.append({
                "Scenario": scen_key,
                "BatchID": bid,
                "accept_frac": float(acc),
                "rmse_med_curve": float(r),
                "tau_prior_mode": tau_mode,
                "mu_tau_prior": float(mu_tau) if np.isfinite(mu_tau) else np.nan,
                "sig_tau_prior": float(sig_tau) if np.isfinite(sig_tau) else np.nan,
                **summ
            })
            if MAKE_PLOTS:
                curves_for_plot.append({"bid": bid, "t": t, "y": y, "y_pred": y_pred})

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{scen_key}] Saved: {out_csv}")

        if MAKE_PLOTS and len(curves_for_plot):
            curves = curves_for_plot[:int(PLOT_LIMIT)] if PLOT_LIMIT is not None else curves_for_plot
            plot_overview(curves, out_png, max_per_ax=PLOT_MAX_PER_AX, title=scen_key)
            print(f"[{scen_key}] Saved overview plot: {out_png}")

        csv_s2 = out_csv

    # --- S3: tau fixed
    if RUN_S3:
        priors_s3 = {"lambda": {"type": "uniform"}, "beta": {"type": "uniform"}}
        fixed_s3 = {"n": 1.0, "tau": float(TAU_FIXED_VALUE)}
        csv_s3 = run_scenario(df, scenario_key=f"S3_tau_fixed_{TAU_FIXED_VALUE:g}_n1", priors=priors_s3, fixed_values=fixed_s3)

    # --- S4: lambda only (tau and beta fixed)
    if RUN_S4:
        priors_s4 = {"lambda": {"type": "uniform"}}
        fixed_s4 = {"n": 1.0, "tau": float(TAU_FIXED_VALUE), "beta": float(BETA_FIXED_VALUE)}
        csv_s4 = run_scenario(df, scenario_key=f"S4_lambda_only_tau{TAU_FIXED_VALUE:g}_beta{BETA_FIXED_VALUE:g}_n1",
                              priors=priors_s4, fixed_values=fixed_s4)

    # Final scatter plot if all ran
    if RUN_S1 and RUN_S2 and RUN_S3 and RUN_S4:
        df_s1 = pd.read_csv(csv_s1)
        df_s2 = pd.read_csv(csv_s2)
        df_s3 = pd.read_csv(csv_s3)
        df_s4 = pd.read_csv(csv_s4)

        out_cmp = os.path.join(OUT_DIR, "souza_parameter_scatter_four_scenarios.png")
        plot_parameter_scatter(df_s1, df_s2, df_s3, df_s4, out_cmp)
        print(f"\nSaved final parameter scatter: {out_cmp}")

    print("\nDone.")


if __name__ == "__main__":
    main()