#!/usr/bin/env python3
"""
souza_eval_test_forward_switch.py

Train the best hyperparameter configuration (from TRIALS_CSV produced by
souza_hpo_only_forward_switch.py) on the full training split, then evaluate on
the held-out test split.

- Uses the SAME:
  * Feature aggregation + DILUENT_ENCODING
  * stratified train/test split
  * ParamNet parameter mapping + PRIOR_APPROACH logic
  * Forward solver switch: FORWARD_METHOD in {"closed_form","odeint"}
  * Training objective: MAP loss = Gaussian NLL (heteroscedastic) + prior penalty

Outputs (OUT_DIR):
  - best_paramnet_checkpoint.pt
  - predicted_params_all.csv          (lambda,tau,beta per BatchID, with split tag)
  - test_parity.png
  - test_dynamics.png

Requires:
  pip install torch optuna numpy pandas matplotlib scipy
  If FORWARD_METHOD="odeint": pip install torchdiffeq
  If PRIOR_APPROACH="tau_spline_prior": tau_prior_tools.py (+ optional cvxpy/osqp)

"""

import os, re, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F

from tau_prior_tools import compute_tau_lognormal_prior


# =============================================================================
# CONFIG (match your HPO script)
# =============================================================================
CSV_PATH   = "./Souza2025_TableS1_Final_v2_diluent_continuous.csv"
TRIALS_CSV = "./souza_hpo_map_reg_t0ic/trials_hpo_map_reg_t0ic.csv"

OUT_DIR = "./souza_eval_outputs_forward_switch"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1
DEVICE = "cpu"  # "cuda" if available

# Choose which study slice to evaluate (must match how TRIALS_CSV was generated)
DILUENT_ENCODING = "continuous"    # "onehot" or "continuous"
PRIOR_APPROACH   = "tau_fixed"  # "uniform_all" | "tau_spline_prior" | "tau_fixed"
FORWARD_METHOD   = "closed_form"   # "closed_form" | "odeint"

# Used only if PRIOR_APPROACH="tau_fixed"
TAU_FIXED_VALUE = 1.0

# Likelihood noise model (as in HPO)
SIGMA_MAIN = 0.03
SIGMA_T0   = 1e-3
T0_ATOL    = 1e-12

# Forward solver settings
ODE_METHOD = "dopri5"
ODE_RTOL = 1e-6
ODE_ATOL = 1e-8

# closed-form numerical knobs
CLOSED_FORM_BETA_EPS = 1e-3
CLOSED_FORM_BLEND_K  = 6.0
CLOSED_FORM_BASE_SOFTPLUS = True

# Bounds (natural space) (must match HPO)
LAM_BOUNDS  = (1e-6, 1e2)
TAU_BOUNDS  = (1e-3, 1e4)
BETA_BOUNDS = (1e-3, 1e1)

# Training settings for final fit
MAX_EPOCHS = 400
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 25

# Defaults if trials CSV lacks columns
OPTIMIZER_DEFAULT = "adamw"
USE_SCHEDULER_DEFAULT = False
LOSS_MODE_DEFAULT = "absolute"    # "absolute" or "relative"
REL_EPS_DEFAULT = 1e-3

SCHED_FACTOR = 0.9
SCHED_PATIENCE = 50
SCHED_MIN_LR = 1e-6
SCHED_COOLDOWN = 0

# Split
TEST_FRAC = 0.20

# Plot outputs
PARITY_FIG   = os.path.join(OUT_DIR, "test_parity.png")
DYNAMICS_FIG = os.path.join(OUT_DIR, "test_dynamics.png")
MODEL_CKPT   = os.path.join(OUT_DIR, "best_paramnet_checkpoint.pt")
PARAMS_CSV   = os.path.join(OUT_DIR, "predicted_params_all.csv")

# Tau spline-prior knobs (only relevant when PRIOR_APPROACH="tau_spline_prior")
TAU_PRIOR_CACHE_TRAIN = os.path.join(OUT_DIR, "tau_spline_prior_cache_train.csv")
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

# Feature columns
BASE_CONT_COLS = ["PEO_N750_pct", "PEO_1105_pct", "PEO_N60K_pct", "PEO_303_pct"]
DILUENT_PCT_COL = "Diluent_pct"
ONEHOT_COLS = ["Diluent_G721", "Diluent_SMCC", "Diluent_MD_IT12"]
DILUENT_CONT_COLS = ["Diluent_G721_pct", "Diluent_SMCC_pct", "Diluent_MD_IT12_pct"]

# Dataset keys
BATCH_COL = "BatchID"
CAT_COL   = "Diluent_type"
TIME_COL  = "time_min"
Y_COL     = "release_frac"


# =============================================================================
# REPRO
# =============================================================================
def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)
torch.set_default_dtype(torch.float32)


def natural_key(batch_id: str):
    m = re.search(r"(\d+)$", str(batch_id))
    return int(m.group(1)) if m else 10**9


def get_feature_spec():
    enc = str(DILUENT_ENCODING).strip().lower()
    if enc == "onehot":
        cont_cols = BASE_CONT_COLS + [DILUENT_PCT_COL]
        cat_cols  = ONEHOT_COLS
        return cont_cols, cat_cols, "onehot"
    if enc in ("continuous", "cont", "gated"):
        cont_cols = BASE_CONT_COLS + DILUENT_CONT_COLS
        cat_cols  = []
        return cont_cols, cat_cols, "continuous"
    raise ValueError("DILUENT_ENCODING must be 'onehot' or 'continuous'.")


# =============================================================================
# DATA: aggregate to one row per BatchID
# =============================================================================
def load_aggregated(csv_path: str):
    df = pd.read_csv(csv_path)

    cont_cols, cat_cols, _ = get_feature_spec()
    needed = {BATCH_COL, CAT_COL, TIME_COL, Y_COL, *cont_cols, *cat_cols}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    t_eval = np.array(sorted(df[TIME_COL].unique()), dtype=float)
    batch_ids = sorted(df[BATCH_COL].unique(), key=natural_key)

    X_list, Y_list, labels, groups = [], [], [], []
    for bid in batch_ids:
        dfi = df[df[BATCH_COL] == bid].sort_values(TIME_COL)
        if not np.allclose(dfi[TIME_COL].to_numpy(dtype=float), t_eval):
            raise ValueError(f"Time grid mismatch for BatchID={bid}")

        x_cont = dfi.iloc[0][cont_cols].to_numpy(dtype=float)
        if len(cat_cols) > 0:
            x_cat = dfi.iloc[0][cat_cols].to_numpy(dtype=float)
            x = np.concatenate([x_cont, x_cat], axis=0)
        else:
            x = x_cont

        X_list.append(x)
        Y_list.append(dfi[Y_COL].to_numpy(dtype=float))
        labels.append(str(dfi.iloc[0][CAT_COL]))
        groups.append(str(bid))

    X_raw = np.vstack(X_list).astype(np.float32)
    Y     = np.vstack(Y_list).astype(np.float32)
    labels = np.array(labels)
    groups = np.array(groups)
    return X_raw, Y, labels, groups, t_eval


# =============================================================================
# SPLIT: formulation-level stratified by CAT_COL
# =============================================================================
def stratified_train_test_split(labels, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(labels))
    train_idx, test_idx = [], []

    for lab in np.unique(labels):
        idx = idx_all[labels == lab].copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * len(idx))))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx, dtype=int)
    test_idx  = np.array(test_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


# =============================================================================
# Standardization (continuous only) - global (train stats only)
# =============================================================================
def standardize_cont_global(X, cont_dim, train_idx):
    Xs = X.copy()
    mu = Xs[train_idx, :cont_dim].mean(axis=0, keepdims=True)
    sd = Xs[train_idx, :cont_dim].std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs[:, :cont_dim] = (Xs[:, :cont_dim] - mu) / sd
    return Xs, mu, sd


# =============================================================================
# Load best hyperparameters from trials CSV (filtered to your current settings)
# =============================================================================
def _as_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return default


def _as_loss_mode(x, default="absolute"):
    if x is None:
        return str(default).lower()
    s = str(x).strip().lower()
    if s in ("rel", "relative", "rel_mse", "rel-mse"):
        return "relative"
    if s in ("abs", "absolute", "abs_mse", "abs-mse"):
        return "absolute"
    return str(default).lower()


def load_best_config_filtered(trials_csv: str):
    df = pd.read_csv(trials_csv)
    if df.empty:
        raise ValueError(f"TRIALS_CSV is empty: {trials_csv}")

    # Back-compat defaults
    if "standardize_cont" not in df.columns:
        df["standardize_cont"] = True
    if "prior_mode" not in df.columns and "prior" in df.columns:
        df["prior_mode"] = df["prior"]
    if "forward_method" not in df.columns:
        df["forward_method"] = str(FORWARD_METHOD).strip().lower()
    if "diluent_encoding" not in df.columns:
        df["diluent_encoding"] = str(DILUENT_ENCODING).strip().lower()
    if "loss_mode" not in df.columns:
        df["loss_mode"] = str(LOSS_MODE_DEFAULT).strip().lower()

    required = ["hidden_size", "n_hidden_layers", "activation", "dropout", "standardize_cont", "lr", "mean_rmse"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Trials CSV missing columns: {missing}")

    # Filter to requested slice
    want_prior = str(PRIOR_APPROACH).strip().lower()
    want_fwd   = str(FORWARD_METHOD).strip().lower()
    want_enc   = str(DILUENT_ENCODING).strip().lower()

    m = (
        (df["prior_mode"].astype(str).str.lower() == want_prior) &
        (df["forward_method"].astype(str).str.lower() == want_fwd) &
        (df["diluent_encoding"].astype(str).str.lower() == want_enc)
    )

    # If loss_mode exists, filter to the script default unless you prefer "best regardless"
    # Here: keep only matching loss_mode if present.
    if "loss_mode" in df.columns:
        want_loss = str(LOSS_MODE_DEFAULT).strip().lower()
        m = m & (df["loss_mode"].astype(str).str.lower() == want_loss)

    df2 = df.loc[m].copy()
    if df2.empty:
        # fall back: ignore loss_mode filter
        m2 = (
            (df["prior_mode"].astype(str).str.lower() == want_prior) &
            (df["forward_method"].astype(str).str.lower() == want_fwd) &
            (df["diluent_encoding"].astype(str).str.lower() == want_enc)
        )
        df2 = df.loc[m2].copy()

    if df2.empty:
        raise ValueError(
            "No rows in TRIALS_CSV match "
            f"prior={want_prior}, forward={want_fwd}, diluent_encoding={want_enc}."
        )

    df2 = df2.sort_values("mean_rmse", ascending=True).reset_index(drop=True)
    best = df2.iloc[0]

    optimizer = str(best["optimizer"]).lower() if "optimizer" in df2.columns else OPTIMIZER_DEFAULT
    use_scheduler = _as_bool(best["use_scheduler"], default=USE_SCHEDULER_DEFAULT) if "use_scheduler" in df2.columns else USE_SCHEDULER_DEFAULT
    loss_mode = _as_loss_mode(best["loss_mode"], default=LOSS_MODE_DEFAULT) if "loss_mode" in df2.columns else LOSS_MODE_DEFAULT
    rel_eps = float(best["rel_eps"]) if ("rel_eps" in df2.columns and np.isfinite(best["rel_eps"])) else float(REL_EPS_DEFAULT)

    hp = {
        "hidden_size": int(best["hidden_size"]),
        "n_hidden_layers": int(best["n_hidden_layers"]),
        "activation": str(best["activation"]),
        "dropout": float(best["dropout"]),
        "standardize_cont": bool(best["standardize_cont"]),
        "lr": float(best["lr"]),
        "optimizer": optimizer,
        "use_scheduler": bool(use_scheduler),
        "loss_mode": str(loss_mode).lower(),
        "rel_eps": float(rel_eps),
    }

    # For tau_fixed mode, try to read tau_fixed_value from row if present
    tau_fixed_row = None
    if want_prior == "tau_fixed" and "tau_fixed_value" in df2.columns:
        v = best.get("tau_fixed_value", np.nan)
        if np.isfinite(v):
            tau_fixed_row = float(v)

    return hp, best.to_dict(), tau_fixed_row


# =============================================================================
# Forward solvers (same as HPO)
# =============================================================================
def solve_closed_form_batch_torch_smooth(lam, tau, beta, t_eval_torch,
                                        eps=CLOSED_FORM_BETA_EPS,
                                        blend_k=CLOSED_FORM_BLEND_K,
                                        base_softplus=CLOSED_FORM_BASE_SOFTPLUS):
    B = lam.shape[0]
    T = t_eval_torch.shape[0]
    t = t_eval_torch.view(1, T).expand(B, T)
    tau_bt = tau.view(B, 1).expand(B, T)
    lam_bt = lam.view(B, 1).expand(B, T)
    beta_b = beta.view(B, 1)

    tau_safe = torch.clamp(tau_bt, min=1e-30)
    I = t - tau_safe * torch.log1p(t / tau_safe)
    A = lam_bt * I

    delta = 1.0 - beta_b
    base = 1.0 - delta * A

    if base_softplus:
        sharp = 50.0
        base_safe = F.softplus(sharp * base) / sharp + 1e-12
    else:
        base_safe = base.clamp_min(1e-12)

    logu_gen = torch.log(base_safe) / torch.clamp(delta, min=-1e12, max=1e12)
    logu_tay = -A - delta * (A * A) * 0.5 - (delta * delta) * (A * A * A) / 3.0

    x = (torch.abs(delta) - float(eps)) / (float(eps) / float(blend_k))
    w = torch.sigmoid(x).expand_as(A)

    logu = (1.0 - w) * logu_tay + w * logu_gen
    u = torch.exp(logu)
    f = 1.0 - u
    return torch.clamp(f, 0.0, 1.0)


class DissolutionODEFunc(nn.Module):
    def __init__(self, lam, tau, beta):
        super().__init__()
        self.lam = lam
        self.tau = tau
        self.beta = beta

    def forward(self, t, f):
        f = torch.clamp(f, 0.0, 1.0)
        tt = t.expand_as(self.tau)
        g = tt / (self.tau + tt + 1e-12)
        one_minus = torch.clamp(1.0 - f, min=0.0)
        return self.lam * g * torch.pow(one_minus, self.beta)


def solve_odeint_batch_torch(lam, tau, beta, t_eval_np, device):
    try:
        from torchdiffeq import odeint
    except Exception as e:
        raise ImportError("FORWARD_METHOD='odeint' requires `pip install torchdiffeq`.") from e

    t = torch.tensor(np.asarray(t_eval_np, dtype=float), device=device, dtype=torch.float64)
    B = lam.shape[0]
    y0 = torch.zeros(B, device=device, dtype=torch.float64)

    lam64 = lam.to(dtype=torch.float64)
    tau64 = tau.to(dtype=torch.float64)
    beta64 = beta.to(dtype=torch.float64)

    func = DissolutionODEFunc(lam64, tau64, beta64)
    y = odeint(func, y0, t, method=str(ODE_METHOD), rtol=float(ODE_RTOL), atol=float(ODE_ATOL))  # [T,B]
    y = y.transpose(0, 1).to(dtype=torch.float32)
    return torch.clamp(y, 0.0, 1.0)


def forward_solve(lam, tau, beta, t_eval_torch, t_eval_np):
    m = str(FORWARD_METHOD).strip().lower()
    if m in ("closed_form", "closedform", "analytic", "analytical"):
        return solve_closed_form_batch_torch_smooth(lam, tau, beta, t_eval_torch)
    if m in ("odeint", "torchdiffeq"):
        return solve_odeint_batch_torch(lam, tau, beta, t_eval_np, device=lam.device)
    raise ValueError("FORWARD_METHOD must be 'closed_form' or 'odeint'.")


def sigma_vector_torch(t_eval_torch):
    is_t0 = torch.isclose(
        t_eval_torch,
        torch.tensor(0.0, device=t_eval_torch.device, dtype=t_eval_torch.dtype),
        atol=float(T0_ATOL), rtol=0.0
    )
    sig = torch.where(
        is_t0,
        torch.tensor(float(SIGMA_T0), device=t_eval_torch.device, dtype=t_eval_torch.dtype),
        torch.tensor(float(SIGMA_MAIN), device=t_eval_torch.device, dtype=t_eval_torch.dtype),
    )
    return sig


# =============================================================================
# Loss pieces (MAP objective)
# =============================================================================
def gaussian_nll_hetero(pred, target, sigma_t):
    sig = sigma_t.view(1, -1).to(dtype=pred.dtype)
    err = pred - target.to(dtype=pred.dtype)
    nll = 0.5 * ((err / sig) ** 2 + 2.0 * torch.log(sig) + math.log(2.0 * math.pi))
    return nll.sum(dim=1).mean()


def report_rmse(pred, target, *, loss_mode, rel_eps):
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    m = str(loss_mode).strip().lower()
    if m in ("absolute", "abs"):
        mse = ((pred - target) ** 2).mean(dim=1)
        return torch.sqrt(mse).mean()
    if m in ("relative", "rel"):
        denom = torch.clamp(target, min=float(rel_eps))
        rel_err = (pred - target) / denom
        mse = (rel_err ** 2).mean(dim=1)
        return torch.sqrt(mse).mean()
    raise ValueError("loss_mode must be 'relative' or 'absolute'.")


def prior_penalty(lam, tau, beta, *, prior_mode, mu_tau=None, sig_tau=None):
    pen = torch.zeros((), device=lam.device, dtype=lam.dtype)

    # optional global priors (kept OFF by default)
    # (If you later want these, you can add toggles here.)

    if prior_mode == "tau_spline_prior":
        if (mu_tau is not None) and (sig_tau is not None):
            log_tau = torch.log(torch.clamp(tau, min=1e-30))
            ok = torch.isfinite(mu_tau) & torch.isfinite(sig_tau) & (sig_tau > 0)
            if torch.any(ok):
                z = (log_tau[ok] - mu_tau[ok]) / sig_tau[ok]
                pen = pen + 0.5 * torch.mean(z ** 2)

    return pen


# =============================================================================
# ParamNet (same as HPO version)
# =============================================================================
def _inv_sigmoid(y):
    y = float(np.clip(y, 1e-6, 1.0 - 1e-6))
    return math.log(y / (1.0 - y))


class ParamNet(nn.Module):
    def __init__(self, in_dim, hidden_size, n_hidden_layers, activation, dropout=0.0, prior_mode="uniform_all"):
        super().__init__()
        activation = str(activation).lower()

        if activation in ("swish", "silu"):
            act = nn.SiLU
        elif activation == "leakyrelu":
            act = nn.LeakyReLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "mish":
            act = nn.Mish
        elif activation == "relu":
            act = nn.ReLU
        else:
            raise ValueError(f"Unknown activation='{activation}'.")

        self.prior_mode = str(prior_mode).strip().lower()

        layers = []
        d = in_dim
        for _ in range(int(n_hidden_layers)):
            layers += [nn.Linear(d, int(hidden_size)), act()]
            if dropout and float(dropout) > 0.0:
                layers += [nn.Dropout(p=float(dropout))]
            d = int(hidden_size)

        self.body = nn.Sequential(*layers)
        out_dim = 2 if self.prior_mode == "tau_fixed" else 3
        self.out = nn.Linear(d, out_dim)

        with torch.no_grad():
            loglam0 = math.log(2.5e-2)
            logtau0 = math.log(50.0)
            logbeta0 = math.log(1.0)

            loglam_lb, loglam_ub = math.log(LAM_BOUNDS[0]), math.log(LAM_BOUNDS[1])
            logtau_lb, logtau_ub = math.log(TAU_BOUNDS[0]), math.log(TAU_BOUNDS[1])
            logbeta_lb, logbeta_ub = math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1])

            def frac(logx, lb, ub):
                return float(np.clip((logx - lb) / (ub - lb), 1e-6, 1.0 - 1e-6))

            b0 = _inv_sigmoid(frac(loglam0, loglam_lb, loglam_ub))
            if out_dim == 2:
                b1 = _inv_sigmoid(frac(logbeta0, logbeta_lb, logbeta_ub))
                self.out.bias[:] = torch.tensor([b0, b1], dtype=torch.float32)
            else:
                b1 = _inv_sigmoid(frac(logtau0, logtau_lb, logtau_ub))
                b2 = _inv_sigmoid(frac(logbeta0, logbeta_lb, logbeta_ub))
                self.out.bias[:] = torch.tensor([b0, b1, b2], dtype=torch.float32)

    def forward(self, x, *, tau_fixed_value=None):
        h = self.out(self.body(x))

        loglam_lb, loglam_ub = math.log(LAM_BOUNDS[0]), math.log(LAM_BOUNDS[1])
        logtau_lb, logtau_ub = math.log(TAU_BOUNDS[0]), math.log(TAU_BOUNDS[1])
        logbeta_lb, logbeta_ub = math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1])

        s0 = torch.sigmoid(h[:, 0])
        log_lam = loglam_lb + (loglam_ub - loglam_lb) * s0
        lam = torch.exp(log_lam)

        if self.prior_mode == "tau_fixed":
            if tau_fixed_value is None:
                raise ValueError("tau_fixed_value must be provided for prior_mode='tau_fixed'")
            tau = torch.full_like(lam, float(tau_fixed_value))

            s1 = torch.sigmoid(h[:, 1])
            log_beta = logbeta_lb + (logbeta_ub - logbeta_lb) * s1
            beta = torch.exp(log_beta)
            return lam, tau, beta

        s1 = torch.sigmoid(h[:, 1])
        log_tau = logtau_lb + (logtau_ub - logtau_lb) * s1
        tau = torch.exp(log_tau)

        s2 = torch.sigmoid(h[:, 2])
        log_beta = logbeta_lb + (logbeta_ub - logbeta_lb) * s2
        beta = torch.exp(log_beta)

        return lam, tau, beta


# =============================================================================
# Optimizer / Scheduler
# =============================================================================
def build_optimizer(model: nn.Module, lr: float, optimizer_name: str):
    opt_name = str(optimizer_name).strip().lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))
    raise ValueError(f"Unknown optimizer '{optimizer_name}'.")


def build_scheduler(optimizer: torch.optim.Optimizer, use_scheduler: bool):
    if not bool(use_scheduler):
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(SCHED_FACTOR),
        patience=int(SCHED_PATIENCE),
        threshold=1e-6,
        threshold_mode="rel",
        cooldown=int(SCHED_COOLDOWN),
        min_lr=float(SCHED_MIN_LR),
        verbose=False,
    )


# =============================================================================
# Scenario 2: tau spline priors on TRAIN set (cache)
# =============================================================================
def precompute_tau_priors_train(Y_train, t_eval, groups_train):
    N = Y_train.shape[0]
    mu = np.full(N, np.nan, dtype=np.float32)
    sig = np.full(N, np.nan, dtype=np.float32)

    if str(PRIOR_APPROACH).strip().lower() != "tau_spline_prior":
        return mu, sig

    if os.path.exists(TAU_PRIOR_CACHE_TRAIN):
        dfc = pd.read_csv(TAU_PRIOR_CACHE_TRAIN)
        if set(dfc.columns) >= {"BatchID", "mu_tau", "sig_tau"}:
            dfc = dfc.set_index("BatchID")
            mu = np.array([dfc.loc[g, "mu_tau"] if g in dfc.index else np.nan for g in groups_train], dtype=np.float32)
            sig = np.array([dfc.loc[g, "sig_tau"] if g in dfc.index else np.nan for g in groups_train], dtype=np.float32)
            if len(mu) == N:
                return mu, sig

    rows = []
    print("\n[precompute] Computing tau spline priors for TRAIN curves...", flush=True)
    for i in range(N):
        bid = str(groups_train[i])
        y = np.asarray(Y_train[i], dtype=float)
        t = np.asarray(t_eval, dtype=float)
        try:
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
                include_arrays=False,
                t0_atol=T0_ATOL,
            )
            mu[i] = float(res.mu)
            sig[i] = float(res.sig)
            rows.append({"BatchID": bid, "mu_tau": float(res.mu), "sig_tau": float(res.sig), "ok": True})
        except Exception as e:
            rows.append({"BatchID": bid, "mu_tau": np.nan, "sig_tau": np.nan, "ok": False, "error": str(e)})

        if (i + 1) % 25 == 0 or (i + 1) == N:
            print(f"  computed {i+1}/{N}", flush=True)

    pd.DataFrame(rows).to_csv(TAU_PRIOR_CACHE_TRAIN, index=False)
    print(f"[precompute] Saved tau prior cache: {TAU_PRIOR_CACHE_TRAIN}", flush=True)
    return mu, sig


# =============================================================================
# Training (full train set)
# =============================================================================
def train_full(
    model,
    Xtr,
    Ytr,
    t_eval_np,
    *,
    hp,
    mu_tau_train=None,
    sig_tau_train=None,
    verbose=True,
):
    opt = build_optimizer(model, lr=float(hp["lr"]), optimizer_name=str(hp["optimizer"]))
    scheduler = build_scheduler(opt, use_scheduler=bool(hp["use_scheduler"]))

    t_torch = torch.tensor(np.asarray(t_eval_np, dtype=float), device=DEVICE, dtype=torch.float32)
    sigma_t = sigma_vector_torch(t_torch)

    n_train = Xtr.shape[0]
    best_loss = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)

        epoch_loss = 0.0
        n_batches = 0

        for k in range(0, n_train, BATCH_SIZE):
            idx = perm[k:k+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            mu_b = sg_b = None
            if str(PRIOR_APPROACH).strip().lower() == "tau_spline_prior":
                mu_b = mu_tau_train[idx]
                sg_b = sig_tau_train[idx]

            lam, tau, beta = model(xb, tau_fixed_value=TAU_FIXED_VALUE if str(PRIOR_APPROACH).strip().lower() == "tau_fixed" else None)

            try:
                pred = forward_solve(lam, tau, beta, t_torch, t_eval_np)
                nll = gaussian_nll_hetero(pred, yb, sigma_t)
                pen = prior_penalty(lam, tau, beta, prior_mode=str(PRIOR_APPROACH).strip().lower(), mu_tau=mu_b, sig_tau=sg_b)
                loss = nll + pen
            except Exception:
                # keep graph connected
                loss = torch.tensor(1e3, device=DEVICE, dtype=torch.float32) + 0.0 * (lam.mean() + tau.mean() + beta.mean())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_loss /= max(1, n_batches)
        if scheduler is not None:
            scheduler.step(epoch_loss)

        if epoch_loss < best_loss - 1e-12:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch == 1 or epoch % LOG_EVERY == 0):
            cur_lr = float(opt.param_groups[0]["lr"])
            print(
                f"epoch={epoch:04d} train_MAP_loss={epoch_loss:.6f} best={best_loss:.6f} "
                f"lr={cur_lr:.3g} opt={hp['optimizer']} sched={int(bool(hp['use_scheduler']))} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_dataset(model, X, t_eval_np):
    model.eval()
    t_torch = torch.tensor(np.asarray(t_eval_np, dtype=float), device=DEVICE, dtype=torch.float32)
    lam, tau, beta = model(X, tau_fixed_value=TAU_FIXED_VALUE if str(PRIOR_APPROACH).strip().lower() == "tau_fixed" else None)
    pred = forward_solve(lam, tau, beta, t_torch, t_eval_np)
    return pred, lam, tau, beta


# =============================================================================
# Plots
# =============================================================================
def make_parity_plot(y_true, y_pred, out_png, text_note=None):
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=250)
    ax.scatter(yt, yp, s=14, alpha=0.6)
    ax.plot([0, 1], [0, 1], linewidth=1.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("f (Test)")
    ax.set_ylabel("f (Predicted)")

    if text_note is not None:
        ax.text(
            0.03, 0.97, text_note,
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=4.0),
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_dynamics_plot(test_batch_ids, t_eval_np, y_true_np, model, X_test, out_png, curves_per_subplot=5):
    t_dense = np.linspace(float(np.min(t_eval_np)), float(np.max(t_eval_np)), 250).astype(np.float32)
    t_dense_torch = torch.tensor(t_dense, device=DEVICE, dtype=torch.float32)

    n = len(test_batch_ids)
    n_panels = 4
    n_per = int(curves_per_subplot)
    if n > n_panels * n_per:
        n_per = int(math.ceil(n / n_panels))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True, dpi=250)
    axes = axes.ravel()
    model.eval()

    for pi in range(n_panels):
        ax = axes[pi]
        start = pi * n_per
        end = min(n, (pi + 1) * n_per)
        if start >= n:
            ax.set_visible(False)
            continue

        for j in range(start, end):
            bid = test_batch_ids[j]
            short_bid = str(bid).split()[0]
            xj = X_test[j:j+1]

            with torch.no_grad():
                lam, tau, beta = model(xj, tau_fixed_value=TAU_FIXED_VALUE if str(PRIOR_APPROACH).strip().lower() == "tau_fixed" else None)
                if str(FORWARD_METHOD).strip().lower() == "closed_form":
                    pred_dense = solve_closed_form_batch_torch_smooth(lam, tau, beta, t_dense_torch).cpu().numpy().reshape(-1)
                else:
                    pred_dense = solve_odeint_batch_torch(lam, tau, beta, t_dense, device=DEVICE).cpu().numpy().reshape(-1)

            ax.plot(t_dense, pred_dense, linewidth=2.0, label=short_bid)
            ax.scatter(t_eval_np, y_true_np[j], s=16, alpha=0.85, label="_nolegend_")

        ax.set_ylabel("Release Fraction (f)")
        ax.legend(ncol=2, fontsize=8, frameon=False)

    for ax in axes[2:]:
        if ax.get_visible():
            ax.set_xlabel("Time (min)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Loading dataset...", flush=True)
    X_raw, Y, labels, batch_ids, t_eval = load_aggregated(CSV_PATH)
    cont_cols, cat_cols, enc_tag = get_feature_spec()
    cont_dim = len(cont_cols)

    print(f"  formulations={len(batch_ids)} | Xdim={X_raw.shape[1]} (cont_dim={cont_dim}, cat_dim={len(cat_cols)}) | T={Y.shape[1]}", flush=True)

    print("Loading best hyperparameters from trials CSV (filtered)...", flush=True)
    hp, best_row_dict, tau_fixed_row = load_best_config_filtered(TRIALS_CSV)

    # If tau_fixed and the best row has a tau_fixed_value, prefer it
    global TAU_FIXED_VALUE
    if str(PRIOR_APPROACH).strip().lower() == "tau_fixed" and tau_fixed_row is not None:
        TAU_FIXED_VALUE = float(tau_fixed_row)

    print("Best config (hp):", hp, flush=True)
    print("Best row summary:", best_row_dict, flush=True)

    if str(PRIOR_APPROACH).strip().lower() == "tau_fixed":
        if not (TAU_BOUNDS[0] <= float(TAU_FIXED_VALUE) <= TAU_BOUNDS[1]):
            raise ValueError(f"TAU_FIXED_VALUE={TAU_FIXED_VALUE} must be within TAU_BOUNDS={TAU_BOUNDS}")

    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)}", flush=True)

    mu = sd = None
    if bool(hp["standardize_cont"]):
        print("Applying standardization using TRAIN statistics only...", flush=True)
        X_all, mu, sd = standardize_cont_global(X_raw, cont_dim=cont_dim, train_idx=train_idx)
    else:
        X_all = X_raw
        print("Standardization disabled for best config.", flush=True)

    Xtr = torch.tensor(X_all[train_idx], device=DEVICE)
    Ytr = torch.tensor(Y[train_idx], device=DEVICE)
    Xte = torch.tensor(X_all[test_idx], device=DEVICE)
    Yte = torch.tensor(Y[test_idx], device=DEVICE)

    # Tau priors for training curves (only used in tau_spline_prior)
    mu_tau_train = sig_tau_train = None
    if str(PRIOR_APPROACH).strip().lower() == "tau_spline_prior":
        mu_np, sig_np = precompute_tau_priors_train(Y[train_idx], t_eval, batch_ids[train_idx])
        mu_tau_train = torch.tensor(mu_np, device=DEVICE, dtype=torch.float32)
        sig_tau_train = torch.tensor(sig_np, device=DEVICE, dtype=torch.float32)

    model = ParamNet(
        in_dim=Xtr.shape[1],
        hidden_size=hp["hidden_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        activation=hp["activation"],
        dropout=hp["dropout"],
        prior_mode=str(PRIOR_APPROACH).strip().lower(),
    ).to(DEVICE)

    print(
        "Training best model on FULL training set...\n"
        f"  prior={PRIOR_APPROACH}  forward={FORWARD_METHOD}  "
        f"loss_mode={hp['loss_mode']} (metric only)  optimizer={hp['optimizer']}  use_scheduler={int(bool(hp['use_scheduler']))}  rel_eps={hp['rel_eps']:g}",
        flush=True
    )

    train_full(
        model, Xtr, Ytr, t_eval,
        hp=hp,
        mu_tau_train=mu_tau_train,
        sig_tau_train=sig_tau_train,
        verbose=True
    )

    # Save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "hp": hp,
        "in_dim": int(Xtr.shape[1]),
        "standardize_cont": bool(hp["standardize_cont"]),
        "mu": (mu.astype(np.float32) if mu is not None else None),
        "sd": (sd.astype(np.float32) if sd is not None else None),
        "t_eval": np.asarray(t_eval, dtype=np.float32),
        "seed": int(SEED),
        "csv_path": str(CSV_PATH),
        "trials_csv": str(TRIALS_CSV),
        "prior_approach": str(PRIOR_APPROACH),
        "forward_method": str(FORWARD_METHOD),
        "tau_fixed_value": float(TAU_FIXED_VALUE) if str(PRIOR_APPROACH).strip().lower() == "tau_fixed" else None,
        "sigma_main": float(SIGMA_MAIN),
        "sigma_t0": float(SIGMA_T0),
        "t0_atol": float(T0_ATOL),
        "ode": {"method": str(ODE_METHOD), "rtol": float(ODE_RTOL), "atol": float(ODE_ATOL)},
        "closed_form": {"beta_eps": float(CLOSED_FORM_BETA_EPS), "blend_k": float(CLOSED_FORM_BLEND_K), "base_softplus": bool(CLOSED_FORM_BASE_SOFTPLUS)},
        "bounds": {"lambda": LAM_BOUNDS, "tau": TAU_BOUNDS, "beta": BETA_BOUNDS},
        "diluent_encoding": str(DILUENT_ENCODING),
    }
    # SECURITY NOTE: only torch.save/torch.load checkpoints you trust (pickle-based).
    torch.save(ckpt, MODEL_CKPT)
    print(f"Saved trained model checkpoint -> {MODEL_CKPT}", flush=True)

    print("Predicting on train and test sets...", flush=True)
    with torch.no_grad():
        Ypred_tr, lam_tr, tau_tr, beta_tr = predict_dataset(model, Xtr, t_eval)
        Ypred_te, lam_te, tau_te, beta_te = predict_dataset(model, Xte, t_eval)

    # Report metrics on test
    test_rmse = float(report_rmse(Ypred_te, Yte, loss_mode=hp["loss_mode"], rel_eps=hp["rel_eps"]).item())
    test_abs_rmse = float(report_rmse(Ypred_te, Yte, loss_mode="absolute", rel_eps=hp["rel_eps"]).item())
    test_rel_rmse = float(report_rmse(Ypred_te, Yte, loss_mode="relative", rel_eps=hp["rel_eps"]).item())

    print(f"Test RMSE (as hp['loss_mode']={hp['loss_mode']}) = {test_rmse:.6f}", flush=True)
    print(f"Test ABS-RMSE = {test_abs_rmse:.6f}", flush=True)
    print(f"Test REL-RMSE = {test_rel_rmse:.6f} (rel_eps={hp['rel_eps']:g})", flush=True)

    # Save parameters CSV for ALL formulations
    all_lam = torch.empty((len(batch_ids),), dtype=torch.float32, device=DEVICE)
    all_tau = torch.empty((len(batch_ids),), dtype=torch.float32, device=DEVICE)
    all_beta = torch.empty((len(batch_ids),), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        Xall = torch.tensor(X_all, device=DEVICE)
        lam_all, tau_all, beta_all = model(Xall, tau_fixed_value=TAU_FIXED_VALUE if str(PRIOR_APPROACH).strip().lower() == "tau_fixed" else None)
        all_lam[:] = lam_all
        all_tau[:] = tau_all
        all_beta[:] = beta_all

    split = np.array(["train"] * len(batch_ids), dtype=object)
    split[test_idx] = "test"

    df_params = pd.DataFrame({
        "BatchID": batch_ids,
        "split": split,
        "lambda_pred": all_lam.detach().cpu().numpy(),
        "tau_pred": all_tau.detach().cpu().numpy(),
        "beta_pred": all_beta.detach().cpu().numpy(),
    })
    df_params.to_csv(PARAMS_CSV, index=False)
    print(f"Saved predicted parameters -> {PARAMS_CSV}", flush=True)

    print(f"Saving parity plot -> {PARITY_FIG}", flush=True)
    note = (
        f"prior={PRIOR_APPROACH}, forward={FORWARD_METHOD}\n"
        f"loss_mode={hp['loss_mode']}  test_RMSE={test_rmse:.4f}\n"
        f"ABS-RMSE={test_abs_rmse:.4f}  REL-RMSE={test_rel_rmse:.4f}"
    )
    make_parity_plot(Yte.cpu().numpy(), Ypred_te.cpu().numpy(), PARITY_FIG, text_note=note)

    print(f"Saving dynamics plot -> {DYNAMICS_FIG}", flush=True)
    test_batch_ids = batch_ids[test_idx]
    make_dynamics_plot(test_batch_ids, t_eval, Yte.cpu().numpy(), model, Xte, DYNAMICS_FIG, curves_per_subplot=5)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()