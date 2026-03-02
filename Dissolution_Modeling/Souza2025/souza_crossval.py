#!/usr/bin/env python3
"""
souza_hpo_only_forward_switch.py

HPO-only (Optuna) hybrid mechanistic+NN dissolution model, with a switchable forward solver:

Forward solvers (choose via FORWARD_METHOD):
  - "closed_form": fast analytical closed-form solve (n=1) with a *smooth* beta≈1 treatment
  - "odeint"     : torchdiffeq odeint solve (Dopri5 etc.), same RHS as your earlier script

Model:
  df/dt = lam * (t/(tau+t)) * (1-f)^beta,  f(0)=0

Training objective (MAP-style):
  loss = heteroscedastic Gaussian NLL + prior penalty (optional)

Prior approaches:
  1) PRIOR_APPROACH="uniform_all"
     - learn (lam, tau, beta) with uniform-support bounds (enforced by NN output mapping)
  2) PRIOR_APPROACH="tau_spline_prior"
     - learn (lam, tau, beta); tau gets per-curve lognormal prior on log(tau)
       estimated from monotone spline workflow via compute_tau_lognormal_prior()
  3) PRIOR_APPROACH="tau_fixed"
     - fix tau = TAU_FIXED_VALUE; learn (lam, beta) only

What this script DOES:
- Runs Optuna CV on training split only, writes TRIALS_CSV incrementally (resume-safe)
- Generates a parallel-coordinates plot (PC_PNG) from TRIALS_CSV
- Can also run "plot-only" mode (RUN_HPO=False) to regenerate PC plot

What this script does NOT do:
- No final training on best hyperparameters
- No final evaluation/export of predicted parameters

Requires:
  pip install torch optuna numpy pandas matplotlib scipy
  If FORWARD_METHOD="odeint": pip install torchdiffeq
  If PRIOR_APPROACH="tau_spline_prior": tau_prior_tools.py (+ optional cvxpy/osqp depending on your implementation)
"""

import os, re, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
from optuna.samplers import TPESampler

from tau_prior_tools import compute_tau_lognormal_prior


# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH = "./Souza2025_TableS1_Final_v2_diluent_continuous.csv"

SEED = 1
DEVICE = "cpu"  # "cuda" if available

# -------------------------
# Forward solver switch
# -------------------------
# Choose: "closed_form" | "odeint"
FORWARD_METHOD = "closed_form"

# torchdiffeq settings (only used if FORWARD_METHOD="odeint")
ODE_METHOD = "dopri5"
ODE_RTOL = 1e-6
ODE_ATOL = 1e-8

# closed-form numerical knobs (only used if FORWARD_METHOD="closed_form")
CLOSED_FORM_BETA_EPS = 1e-3   # neighborhood around beta=1 to use Taylor stabilization
CLOSED_FORM_BLEND_K  = 6.0    # blend sharpness (bigger = sharper transition)
CLOSED_FORM_BASE_SOFTPLUS = True  # avoid base<=0 dead-grad region by smoothing positivity


# -------------------------
# Prior approach selection
# -------------------------
# Choose one:
#   "uniform_all" | "tau_spline_prior" | "tau_fixed"
PRIOR_APPROACH = "tau_spline_prior"
TAU_FIXED_VALUE = 1.0  # used only if PRIOR_APPROACH="tau_fixed"

# Split / CV
TEST_FRAC = 0.20
K_FOLDS = 5

# Likelihood noise model (heteroscedastic)
SIGMA_MAIN = 0.03
SIGMA_T0   = 1e-3
T0_ATOL    = 1e-12

# Training
MAX_EPOCHS = 700
PATIENCE = 100
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 10

# Scheduler
USE_SCHEDULER = False
SCHED_FACTOR = 0.9
SCHED_PATIENCE = 50
SCHED_MIN_LR = 1e-6
SCHED_COOLDOWN = 0

# Optimizer
OPTIMIZER_TYPE = "adamw"  # "adam" or "adamw"

# Report metric (Optuna objective). Training loss is NLL+prior.
LOSS_MODE = "absolute"   # "relative" or "absolute"
REL_EPS = 1e-3

# Optional global priors on log-lam and log-beta (default OFF)
USE_GLOBAL_LAM_BETA_PRIOR = False
MU_LOG_LAM  = math.log(2.5e-2)
SIG_LOG_LAM = 1.0
MU_LOG_BETA  = math.log(1.0)
SIG_LOG_BETA = 1.0

# Tau spline-prior knobs (Scenario 2)
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

# Bounds (natural space)
LAM_BOUNDS  = (1e-6, 1e2)
TAU_BOUNDS  = (1e-3, 1e4)
BETA_BOUNDS = (1e-3, 1e1)

# Diluent encoding
DILUENT_ENCODING = "continuous"

# HPO controls
RUN_HPO = False
N_TRIALS = 100
OPTUNA_STORAGE = "sqlite:///souza_optuna_hpo_forward_switch.db"
OPTUNA_STUDY_NAME = f"souza_hpo_{DILUENT_ENCODING}_{PRIOR_APPROACH}_{FORWARD_METHOD}"
USE_PRUNER = False

# Search space
HIDDEN_SIZE_MIN = 12
HIDDEN_SIZE_MAX = 64
HIDDEN_SIZE_STEP = 1
N_HIDDEN_LAYERS = [2, 3, 4, 5]
ACTIVATIONS = ["swish", "leakyrelu", "gelu", "mish"]
DROPOUTS = [0.0, 0.1, 0.2]
STANDARDIZE_OPTIONS = [True]
LR_MIN = 1e-3
LR_MAX = 5e-2

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

# Outputs
OUT_DIR = "souza_hpo_forward_switch"
os.makedirs(OUT_DIR, exist_ok=True)

TRIALS_CSV = os.path.join(OUT_DIR, "trials_hpo_forward_switch.csv")
PC_PNG     = os.path.join(OUT_DIR, "parallel_coords_hpo_forward_switch.png")

# Tau prior cache (Scenario 2 only; computed on TRAIN set)
TAU_PRIOR_CACHE_TRAIN = os.path.join(OUT_DIR, "tau_spline_prior_cache_train.csv")


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
# FOLDS: rare-class-safe stratified folds
# =============================================================================
def stratified_folds(labels, k=5, seed=0, max_tries=20000):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    N = len(labels)
    idx_all = np.arange(N)

    classes = sorted(np.unique(labels))
    idx_by = {c: idx_all[labels == c].copy().astype(int) for c in classes}
    counts = {c: len(idx_by[c]) for c in classes}

    n_val_target = max(len(classes), int(round(N / float(k))))
    val_use = np.zeros(N, dtype=int)

    def ok_split(tr, va):
        for c in classes:
            if counts[c] >= 1 and np.sum(labels[va] == c) < 1:
                return False
            if counts[c] >= 2 and np.sum(labels[tr] == c) < 1:
                return False
        return True

    splits = []
    tries = 0
    while len(splits) < k and tries < max_tries:
        tries += 1

        val = []
        remaining = set(idx_all.tolist())

        for c in classes:
            cand = idx_by[c]
            if len(cand) == 0:
                continue
            min_use = val_use[cand].min()
            cand_min = cand[val_use[cand] == min_use]
            pick = int(rng.choice(cand_min))
            if pick in remaining:
                val.append(pick)
                remaining.remove(pick)

        remaining_list = np.array(sorted(list(remaining)), dtype=int)
        rng.shuffle(remaining_list)
        need = max(0, n_val_target - len(val))
        if need > 0:
            val.extend(remaining_list[:need].tolist())

        val_idx = np.array(sorted(set(val)), dtype=int)
        tr_idx = np.array(sorted(list(set(idx_all) - set(val_idx))), dtype=int)

        if ok_split(tr_idx, val_idx):
            splits.append((tr_idx, val_idx))
            val_use[val_idx] += 1

    if len(splits) < k:
        raise RuntimeError(
            f"Could only generate {len(splits)}/{k} constrained splits after {tries} tries. "
            f"Try reducing k or increasing max_tries."
        )
    return splits


# =============================================================================
# Standardization (continuous only) - fold-safe
# =============================================================================
def standardize_cont_fold(X, cont_dim, train_idx):
    Xs = X.copy()
    mu = Xs[train_idx, :cont_dim].mean(axis=0, keepdims=True)
    sd = Xs[train_idx, :cont_dim].std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs[:, :cont_dim] = (Xs[:, :cont_dim] - mu) / sd
    return Xs


# =============================================================================
# Forward solvers
# =============================================================================
def solve_closed_form_batch_torch_smooth(lam, tau, beta, t_eval_torch,
                                        eps=CLOSED_FORM_BETA_EPS,
                                        blend_k=CLOSED_FORM_BLEND_K,
                                        base_softplus=CLOSED_FORM_BASE_SOFTPLUS):
    """
    Closed-form batched solve, differentiable around beta=1.

    lam,tau,beta: [B] float32
    t_eval_torch: [T] float32
    returns f: [B,T]
    """
    B = lam.shape[0]
    T = t_eval_torch.shape[0]
    t = t_eval_torch.view(1, T).expand(B, T)
    tau_bt = tau.view(B, 1).expand(B, T)
    lam_bt = lam.view(B, 1).expand(B, T)
    beta_b = beta.view(B, 1)

    tau_safe = torch.clamp(tau_bt, min=1e-30)
    I = t - tau_safe * torch.log1p(t / tau_safe)
    A = lam_bt * I

    delta = 1.0 - beta_b  # [B,1]
    base = 1.0 - delta * A  # [B,T]

    if base_softplus:
        sharp = 50.0
        base_safe = F.softplus(sharp * base) / sharp + 1e-12
    else:
        base_safe = base.clamp_min(1e-12)

    # general
    logu_gen = torch.log(base_safe) / torch.clamp(delta, min=-1e12, max=1e12)

    # taylor near delta=0: log u ≈ -A - delta*A^2/2 - delta^2*A^3/3
    logu_tay = -A - delta * (A * A) * 0.5 - (delta * delta) * (A * A * A) / 3.0

    # smooth blend weight
    x = (torch.abs(delta) - float(eps)) / (float(eps) / float(blend_k))
    w = torch.sigmoid(x).expand_as(A)

    logu = (1.0 - w) * logu_tay + w * logu_gen
    u = torch.exp(logu)

    f = 1.0 - u
    return torch.clamp(f, 0.0, 1.0)


class DissolutionODEFunc(nn.Module):
    """
    RHS for df/dt = lam * (t/(tau+t)) * (1-f)^beta  (n=1)
    lam,tau,beta: [B]
    f: [B]
    """
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
    """
    torchdiffeq odeint solve for a batch.
    returns [B,T] float32
    """
    try:
        from torchdiffeq import odeint
    except Exception as e:
        raise ImportError("FORWARD_METHOD='odeint' requires `pip install torchdiffeq`.") from e

    # do integration in float64 for stability, then cast back
    t = torch.tensor(np.asarray(t_eval_np, dtype=float), device=device, dtype=torch.float64)
    B = lam.shape[0]
    y0 = torch.zeros(B, device=device, dtype=torch.float64)

    lam64 = lam.to(dtype=torch.float64)
    tau64 = tau.to(dtype=torch.float64)
    beta64 = beta.to(dtype=torch.float64)

    func = DissolutionODEFunc(lam64, tau64, beta64)
    y = odeint(func, y0, t, method=str(ODE_METHOD), rtol=float(ODE_RTOL), atol=float(ODE_ATOL))  # [T,B]
    y = y.transpose(0, 1).to(dtype=torch.float32)  # [B,T]
    return torch.clamp(y, 0.0, 1.0)


def forward_solve(lam, tau, beta, t_eval_torch, t_eval_np):
    """
    Unified forward solver: returns pred [B,T] float32
    """
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
# Loss pieces
# =============================================================================
def gaussian_nll_hetero(pred, target, sigma_t):
    """
    pred,target: [B,T], sigma_t: [T]
    returns scalar mean NLL across batch
    """
    sig = sigma_t.view(1, -1).to(dtype=pred.dtype)
    err = pred - target.to(dtype=pred.dtype)
    nll = 0.5 * ((err / sig) ** 2 + 2.0 * torch.log(sig) + math.log(2.0 * math.pi))
    return nll.sum(dim=1).mean()


def report_rmse(pred, target):
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    mode = str(LOSS_MODE).strip().lower()
    if mode in ("absolute", "abs"):
        mse = ((pred - target) ** 2).mean(dim=1)
        return torch.sqrt(mse).mean()

    if mode in ("relative", "rel"):
        denom = torch.clamp(target, min=float(REL_EPS))
        rel_err = (pred - target) / denom
        mse = (rel_err ** 2).mean(dim=1)
        return torch.sqrt(mse).mean()

    raise ValueError("LOSS_MODE must be 'relative' or 'absolute'.")


def prior_penalty(lam, tau, beta, *, prior_mode, mu_tau=None, sig_tau=None):
    pen = torch.zeros((), device=lam.device, dtype=lam.dtype)

    log_lam = torch.log(torch.clamp(lam, min=1e-30))
    log_beta = torch.log(torch.clamp(beta, min=1e-30))

    if USE_GLOBAL_LAM_BETA_PRIOR:
        pen = pen + 0.5 * torch.mean(((log_lam - float(MU_LOG_LAM)) / float(SIG_LOG_LAM)) ** 2)
        pen = pen + 0.5 * torch.mean(((log_beta - float(MU_LOG_BETA)) / float(SIG_LOG_BETA)) ** 2)

    if prior_mode == "tau_spline_prior":
        if (mu_tau is not None) and (sig_tau is not None):
            log_tau = torch.log(torch.clamp(tau, min=1e-30))
            ok = torch.isfinite(mu_tau) & torch.isfinite(sig_tau) & (sig_tau > 0)
            if torch.any(ok):
                z = (log_tau[ok] - mu_tau[ok]) / sig_tau[ok]
                pen = pen + 0.5 * torch.mean(z ** 2)

    return pen


# =============================================================================
# ParamNet: MLP -> parameters; bounds enforced via sigmoid mapping in log-space
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
def build_optimizer(model: nn.Module, lr: float):
    opt_name = str(OPTIMIZER_TYPE).strip().lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))
    raise ValueError(f"Unknown OPTIMIZER_TYPE='{OPTIMIZER_TYPE}'.")


def build_scheduler(optimizer: torch.optim.Optimizer):
    if not bool(USE_SCHEDULER):
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
# Scenario 2: tau spline priors on TRAIN set
# =============================================================================
def precompute_tau_priors_train(Y_train, t_eval, groups_train):
    N = Y_train.shape[0]

    if PRIOR_APPROACH != "tau_spline_prior":
        return np.full(N, np.nan, dtype=np.float32), np.full(N, np.nan, dtype=np.float32)

    if os.path.exists(TAU_PRIOR_CACHE_TRAIN):
        dfc = pd.read_csv(TAU_PRIOR_CACHE_TRAIN)
        if set(dfc.columns) >= {"BatchID", "mu_tau", "sig_tau"}:
            dfc = dfc.set_index("BatchID")
            mu = np.array([dfc.loc[g, "mu_tau"] if g in dfc.index else np.nan for g in groups_train], dtype=np.float32)
            sig = np.array([dfc.loc[g, "sig_tau"] if g in dfc.index else np.nan for g in groups_train], dtype=np.float32)
            if len(mu) == N:
                return mu, sig

    rows = []
    mu = np.full(N, np.nan, dtype=np.float32)
    sig = np.full(N, np.nan, dtype=np.float32)

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
# TRAIN one fold (MAP objective). Returns validation RMSE for Optuna.
# =============================================================================
def train_one_fold(X_raw_train, Y_train, t_eval, train_idx, val_idx, hp, fold_id, config_id, cont_dim,
                   prior_mode, mu_tau_train=None, sig_tau_train=None):

    if bool(hp["standardize_cont"]):
        X_fold = standardize_cont_fold(X_raw_train, cont_dim=cont_dim, train_idx=train_idx)
    else:
        X_fold = X_raw_train

    Xtr = torch.tensor(X_fold[train_idx], device=DEVICE)
    Ytr = torch.tensor(Y_train[train_idx], device=DEVICE)
    Xva = torch.tensor(X_fold[val_idx], device=DEVICE)
    Yva = torch.tensor(Y_train[val_idx], device=DEVICE)

    if prior_mode == "tau_spline_prior":
        mu_tr = torch.tensor(mu_tau_train[train_idx], device=DEVICE)
        sg_tr = torch.tensor(sig_tau_train[train_idx], device=DEVICE)
        mu_va = torch.tensor(mu_tau_train[val_idx], device=DEVICE)
        sg_va = torch.tensor(sig_tau_train[val_idx], device=DEVICE)
    else:
        mu_tr = sg_tr = mu_va = sg_va = None

    model = ParamNet(
        in_dim=Xtr.shape[1],
        hidden_size=hp["hidden_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        activation=hp["activation"],
        dropout=hp["dropout"],
        prior_mode=prior_mode,
    ).to(DEVICE)

    opt = build_optimizer(model, lr=hp["lr"])
    scheduler = build_scheduler(opt)

    t_eval_np = np.asarray(t_eval, dtype=float)
    t_torch = torch.tensor(t_eval_np, device=DEVICE, dtype=torch.float32)
    sigma_t = sigma_vector_torch(t_torch)

    best_val = float("inf")
    best_state = None
    bad = 0

    n_train = Xtr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        last_train = None

        for k in range(0, n_train, BATCH_SIZE):
            idx = perm[k:k+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            mu_b = sg_b = None
            if prior_mode == "tau_spline_prior":
                mu_b = mu_tr[idx]
                sg_b = sg_tr[idx]

            lam, tau, beta = model(xb, tau_fixed_value=TAU_FIXED_VALUE if prior_mode == "tau_fixed" else None)
            pred = forward_solve(lam, tau, beta, t_torch, t_eval_np)

            nll = gaussian_nll_hetero(pred, yb, sigma_t)
            pen = prior_penalty(lam, tau, beta, prior_mode=prior_mode, mu_tau=mu_b, sig_tau=sg_b)
            loss = nll + pen

            last_train = float(loss.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # validate
        model.eval()
        with torch.no_grad():
            lam, tau, beta = model(Xva, tau_fixed_value=TAU_FIXED_VALUE if prior_mode == "tau_fixed" else None)
            pred = forward_solve(lam, tau, beta, t_torch, t_eval_np)

            nll_va = gaussian_nll_hetero(pred, Yva, sigma_t)
            pen_va = prior_penalty(lam, tau, beta, prior_mode=prior_mode, mu_tau=mu_va, sig_tau=sg_va)
            val_total = float((nll_va + pen_va).item())

            val_rmse = float(report_rmse(pred, Yva).item())

        if scheduler is not None:
            scheduler.step(val_total)

        cur_lr = float(opt.param_groups[0]["lr"])
        if (fold_id == 1) and (epoch == 1 or epoch % LOG_EVERY == 0):
            print(
                f"      cfg={config_id} fold={fold_id} epoch={epoch:03d} "
                f"train_loss~{last_train:.6f} val_loss={val_total:.6f} best_val={best_val:.6f} "
                f"val_rmse={val_rmse:.6f} lr={cur_lr:.3g}",
                flush=True
            )

        if val_total < best_val - 1e-12:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Return RMSE on val for Optuna score
    model.eval()
    with torch.no_grad():
        lam, tau, beta = model(Xva, tau_fixed_value=TAU_FIXED_VALUE if prior_mode == "tau_fixed" else None)
        pred = forward_solve(lam, tau, beta, t_torch, np.asarray(t_eval, dtype=float))
        return float(report_rmse(pred, Yva).item())


# =============================================================================
# Parallel coordinates plot (updated to include forward_method)
# =============================================================================
def parallel_coordinates_plot(df_trials, out_png):
    if df_trials.empty:
        raise ValueError("df_trials is empty.")

    dfp = df_trials.copy().reset_index(drop=True)

    required = {
        "hidden_size","n_hidden_layers","activation","dropout","standardize_cont","lr",
        "mean_rmse","std_rmse","prior_mode","forward_method"
    }
    missing = required - set(dfp.columns)
    if missing:
        raise ValueError(f"TRIALS_CSV missing required columns: {missing}")

    TOP_K_PLOT = min(50, len(dfp))
    dfp = dfp.sort_values("mean_rmse", ascending=True).head(TOP_K_PLOT).reset_index(drop=True)

    # categorical maps
    act_levels = sorted(dfp["activation"].astype(str).unique().tolist())
    act_to_y = {a: (0.5 if len(act_levels) == 1 else i/(len(act_levels)-1)) for i,a in enumerate(act_levels)}
    dfp["activation_y"] = dfp["activation"].astype(str).map(act_to_y).astype(float)

    prior_levels = sorted(dfp["prior_mode"].astype(str).str.lower().unique().tolist())
    prior_to_y = {p: (0.5 if len(prior_levels) == 1 else i/(len(prior_levels)-1)) for i,p in enumerate(prior_levels)}
    dfp["prior_y"] = dfp["prior_mode"].astype(str).str.lower().map(prior_to_y).astype(float)

    fwd_levels = sorted(dfp["forward_method"].astype(str).str.lower().unique().tolist())
    fwd_to_y = {m: (0.5 if len(fwd_levels) == 1 else i/(len(fwd_levels)-1)) for i,m in enumerate(fwd_levels)}
    dfp["forward_y"] = dfp["forward_method"].astype(str).str.lower().map(fwd_to_y).astype(float)

    dfp["standardize_y"] = dfp["standardize_cont"].astype(bool).map({False: 0.0, True: 1.0}).astype(float)

    dims = [
        ("prior_mode",        dfp["prior_y"].to_numpy(float), "categorical_prior"),
        ("forward_method",    dfp["forward_y"].to_numpy(float), "categorical_forward"),
        ("hidden_size",       dfp["hidden_size"].astype(float).to_numpy(), "linear"),
        ("n_hidden_layers",   dfp["n_hidden_layers"].astype(float).to_numpy(), "linear"),
        ("activation",        dfp["activation_y"].to_numpy(float), "categorical_activation"),
        ("dropout",           dfp["dropout"].astype(float).to_numpy(), "linear"),
        ("standardize_cont",  dfp["standardize_y"].to_numpy(float), "categorical_std"),
        ("learning_rate",     dfp["lr"].astype(float).to_numpy(), "log"),
        ("mean_rmse",         dfp["mean_rmse"].astype(float).to_numpy(), "linear"),
    ]

    meta = []
    YN = []
    for name, vals, kind in dims:
        vals = np.asarray(vals, dtype=float)
        if kind == "log":
            v = np.log10(vals)
            vmin, vmax = float(np.min(v)), float(np.max(v))
            yn = np.zeros_like(v) if abs(vmax - vmin) < 1e-12 else (v - vmin) / (vmax - vmin)
            meta.append({"name": name, "kind": kind, "vmin": vmin, "vmax": vmax})
            YN.append(yn)
        elif kind.startswith("categorical"):
            yn = np.clip(vals, 0.0, 1.0)
            meta.append({"name": name, "kind": kind, "vmin": 0.0, "vmax": 1.0})
            YN.append(yn)
        else:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            yn = np.zeros_like(vals) if abs(vmax - vmin) < 1e-12 else (vals - vmin) / (vmax - vmin)
            meta.append({"name": name, "kind": kind, "vmin": vmin, "vmax": vmax})
            YN.append(yn)

    std = dfp["std_rmse"].to_numpy(dtype=float)
    mean = dfp["mean_rmse"].to_numpy(dtype=float)

    cmap = mpl.cm.viridis
    norm_c = mpl.colors.Normalize(vmin=float(np.min(std)), vmax=float(np.max(std)))
    colors = cmap(norm_c(std))

    D = len(dims)
    x = np.arange(D, dtype=float)

    TOP_N_THICK = min(6, len(dfp))
    order_best = np.argsort(mean)
    idx_top = order_best[:TOP_N_THICK]
    idx_rest = np.array([i for i in range(len(dfp)) if i not in set(idx_top.tolist())], dtype=int)

    fig, ax = plt.subplots(figsize=(15.5, 6), dpi=220)
    ax.set_xlim(-0.5, D - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in meta], rotation=20, ha="right")

    for i in idx_rest:
        yline = [YN[j][i] for j in range(D)]
        ax.plot(x, yline, color=colors[i], alpha=0.65, linewidth=0.8)

    for i in idx_top:
        yline = [YN[j][i] for j in range(D)]
        ax.plot(x, yline, color=colors[i], alpha=1.0, linewidth=2.2)

    def draw_axis_ticks(j, m):
        kind, vmin, vmax = m["kind"], m["vmin"], m["vmax"]
        ax.vlines(j, 0.0, 1.0, color="k", linewidth=0.9, alpha=0.35)
        tick_x0, tick_x1 = j - 0.04, j + 0.04

        if kind == "categorical_activation":
            ys = [0.5] if len(act_levels) == 1 else np.linspace(0.0, 1.0, len(act_levels))
            for yv, lab in zip(ys, act_levels):
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, str(lab), ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

        elif kind == "categorical_prior":
            ys = [0.5] if len(prior_levels) == 1 else np.linspace(0.0, 1.0, len(prior_levels))
            for yv, lab in zip(ys, prior_levels):
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, str(lab), ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

        elif kind == "categorical_forward":
            ys = [0.5] if len(fwd_levels) == 1 else np.linspace(0.0, 1.0, len(fwd_levels))
            for yv, lab in zip(ys, fwd_levels):
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, str(lab), ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

        elif kind == "categorical_std":
            for yv, lab in zip([0.0, 1.0], ["False", "True"]):
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, lab, ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

        elif kind == "log":
            pmin, pmax = int(np.floor(vmin)), int(np.ceil(vmax))
            ticks = [10 ** p for p in range(pmin, pmax + 1)]
            for tval in ticks:
                lt = np.log10(tval)
                if lt < vmin - 1e-12 or lt > vmax + 1e-12:
                    continue
                yv = 0.0 if abs(vmax - vmin) < 1e-12 else (lt - vmin) / (vmax - vmin)
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, f"{tval:.0e}", ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))
        else:
            ticks = [vmin] if abs(vmax - vmin) < 1e-12 else np.linspace(vmin, vmax, 5)
            for tval in ticks:
                yv = 0.5 if abs(vmax - vmin) < 1e-12 else (tval - vmin) / (vmax - vmin)
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, f"{tval:.3g}", ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

    for j, m in enumerate(meta):
        draw_axis_ticks(j, m)

    ttl = f"Parallel coordinates | prior={PRIOR_APPROACH} | forward={FORWARD_METHOD} | color=std_rmse | thick=top-{TOP_N_THICK}"
    ax.set_title(ttl)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Std CV RMSE")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def regenerate_plot_only():
    if not os.path.exists(TRIALS_CSV):
        raise FileNotFoundError(f"TRIALS_CSV not found: {TRIALS_CSV}")
    df_trials = pd.read_csv(TRIALS_CSV)
    print("\nCreating parallel coordinates plot...", flush=True)
    parallel_coordinates_plot(df_trials, PC_PNG)
    print(f"Saved plot: {PC_PNG}", flush=True)

    print("\nTop 10 configs (by mean_rmse):", flush=True)
    cols_show = [c for c in ["prior_mode","forward_method","hidden_size","n_hidden_layers","activation","dropout","standardize_cont","lr","mean_rmse","std_rmse"]
                 if c in df_trials.columns]
    print(df_trials.sort_values("mean_rmse").head(10)[cols_show].to_string(index=False), flush=True)


# =============================================================================
# MAIN: HPO only
# =============================================================================
def main():
    if not bool(RUN_HPO):
        regenerate_plot_only()
        return

    cont_cols, cat_cols, dil_enc_tag = get_feature_spec()
    cont_dim = len(cont_cols)

    print("Loading and aggregating dataset...", flush=True)
    X_raw, Y, labels, groups, t_eval = load_aggregated(CSV_PATH)
    print(f"  Loaded {len(groups)} formulations; X dim={X_raw.shape[1]}, curve length T={Y.shape[1]}", flush=True)
    print(f"  Diluent encoding: {dil_enc_tag} | cont_dim={cont_dim} | cat_dim={len(cat_cols)}", flush=True)
    print(f"  Time range: {t_eval.min()}..{t_eval.max()} minutes | forward={FORWARD_METHOD}", flush=True)
    print(f"  Prior approach: {PRIOR_APPROACH}", flush=True)

    if PRIOR_APPROACH == "tau_fixed":
        if not (TAU_BOUNDS[0] <= float(TAU_FIXED_VALUE) <= TAU_BOUNDS[1]):
            raise ValueError(f"TAU_FIXED_VALUE={TAU_FIXED_VALUE} must be within TAU_BOUNDS={TAU_BOUNDS}")

    # outer split
    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)} (formulation-level)", flush=True)

    # CV only on training set
    X_train_raw = X_raw[train_idx]
    Y_train = Y[train_idx]
    labels_train = labels[train_idx]
    groups_train = groups[train_idx]

    splits = stratified_folds(labels_train, k=K_FOLDS, seed=SEED)
    print(f"Prepared {K_FOLDS}-fold CV splits on training set.", flush=True)

    # tau priors (scenario 2 only)
    mu_tau_train, sig_tau_train = precompute_tau_priors_train(Y_train, t_eval, groups_train)

    # Resume support via TRIALS_CSV
    done_keys = set()
    if os.path.exists(TRIALS_CSV):
        prev = pd.read_csv(TRIALS_CSV)
        for _, r in prev.iterrows():
            key = (
                str(r.get("prior_mode", "")).strip().lower(),
                str(r.get("forward_method", "")).strip().lower(),
                int(r["hidden_size"]),
                int(r["n_hidden_layers"]),
                str(r["activation"]),
                float(r["dropout"]),
                bool(r["standardize_cont"]),
                float(r["lr"]),
            )
            done_keys.add(key)
        print(f"Resuming: found {len(done_keys)} completed configs in {TRIALS_CSV}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        hs = trial.suggest_int("hidden_size", HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX, step=HIDDEN_SIZE_STEP)
        nl = trial.suggest_categorical("n_hidden_layers", N_HIDDEN_LAYERS)
        act = trial.suggest_categorical("activation", ACTIVATIONS)
        drop = trial.suggest_categorical("dropout", DROPOUTS)
        stdz = trial.suggest_categorical("standardize_cont", STANDARDIZE_OPTIONS)
        lr = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)

        key = (str(PRIOR_APPROACH).lower(), str(FORWARD_METHOD).lower(), int(hs), int(nl), str(act), float(drop), bool(stdz), float(lr))

        # reuse if already computed
        if key in done_keys and os.path.exists(TRIALS_CSV):
            df_prev = pd.read_csv(TRIALS_CSV)
            m = (
                (df_prev["prior_mode"].astype(str).str.lower() == str(PRIOR_APPROACH).lower()) &
                (df_prev["forward_method"].astype(str).str.lower() == str(FORWARD_METHOD).lower()) &
                (df_prev["hidden_size"].astype(int) == int(hs)) &
                (df_prev["n_hidden_layers"].astype(int) == int(nl)) &
                (df_prev["activation"].astype(str) == str(act)) &
                (df_prev["dropout"].astype(float) == float(drop)) &
                (df_prev["standardize_cont"].astype(bool) == bool(stdz)) &
                (np.isclose(df_prev["lr"].astype(float), float(lr)))
            )
            if m.any():
                return float(df_prev.loc[m, "mean_rmse"].iloc[0])

        config_id = f"trial{trial.number}"
        print(
            f"\n[TRIAL {trial.number}] prior={PRIOR_APPROACH} forward={FORWARD_METHOD} "
            f"hs={hs} nl={nl} act={act} drop={drop:g} std={int(bool(stdz))} lr={lr:.3g}",
            flush=True
        )

        hp = {
            "hidden_size": int(hs),
            "n_hidden_layers": int(nl),
            "activation": str(act),
            "dropout": float(drop),
            "standardize_cont": bool(stdz),
            "lr": float(lr),
        }

        fold_scores = []
        t0 = time.time()
        for fold_id, (tr, va) in enumerate(splits, 1):
            r = train_one_fold(
                X_train_raw, Y_train, t_eval, tr, va, hp,
                fold_id=fold_id,
                config_id=config_id,
                cont_dim=cont_dim,
                prior_mode=str(PRIOR_APPROACH).lower(),
                mu_tau_train=mu_tau_train,
                sig_tau_train=sig_tau_train,
            )
            fold_scores.append(r)

            if USE_PRUNER:
                trial.report(float(np.mean(fold_scores)), step=fold_id)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_rmse = float(np.mean(fold_scores))
        std_rmse  = float(np.std(fold_scores))
        print(f"[TRIAL DONE] {config_id} | mean_rmse={mean_rmse:.5f} ± {std_rmse:.5f} | elapsed={time.time()-t0:.1f}s", flush=True)

        row = {
            "prior_mode": str(PRIOR_APPROACH).strip().lower(),
            "forward_method": str(FORWARD_METHOD).strip().lower(),
            "tau_fixed_value": float(TAU_FIXED_VALUE) if str(PRIOR_APPROACH).strip().lower() == "tau_fixed" else np.nan,
            "loss_mode": str(LOSS_MODE).strip().lower(),
            "diluent_encoding": dil_enc_tag,
            "hidden_size": int(hs),
            "n_hidden_layers": int(nl),
            "activation": str(act),
            "dropout": float(drop),
            "standardize_cont": bool(stdz),
            "lr": float(lr),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "optimizer": str(OPTIMIZER_TYPE).strip().lower(),
            "use_scheduler": bool(USE_SCHEDULER),
            "rel_eps": float(REL_EPS) if str(LOSS_MODE).strip().lower() == "relative" else np.nan,
            "seed": int(SEED),
            # log forward config (helps later debugging)
            "ode_method": str(ODE_METHOD) if str(FORWARD_METHOD).lower() == "odeint" else "",
            "ode_rtol": float(ODE_RTOL) if str(FORWARD_METHOD).lower() == "odeint" else np.nan,
            "ode_atol": float(ODE_ATOL) if str(FORWARD_METHOD).lower() == "odeint" else np.nan,
            "cf_beta_eps": float(CLOSED_FORM_BETA_EPS) if str(FORWARD_METHOD).lower() == "closed_form" else np.nan,
            "cf_blend_k": float(CLOSED_FORM_BLEND_K) if str(FORWARD_METHOD).lower() == "closed_form" else np.nan,
            "cf_base_softplus": bool(CLOSED_FORM_BASE_SOFTPLUS) if str(FORWARD_METHOD).lower() == "closed_form" else False,
        }

        out_df = pd.DataFrame([row])
        if os.path.exists(TRIALS_CSV):
            prev = pd.read_csv(TRIALS_CSV)
            out_df = pd.concat([prev, out_df], ignore_index=True)

        out_df = out_df.drop_duplicates(
            subset=["prior_mode","forward_method","hidden_size","n_hidden_layers","activation","dropout","standardize_cont","lr"]
        )
        out_df.to_csv(TRIALS_CSV, index=False)

        done_keys.add(key)
        return mean_rmse

    sampler = TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2) if USE_PRUNER else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )

    print(f"\nRunning Optuna: n_trials={N_TRIALS} | storage={OPTUNA_STORAGE} | study={OPTUNA_STUDY_NAME}", flush=True)
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nOptuna done.", flush=True)
    print(f"Best mean_rmse = {study.best_value:.6f}", flush=True)
    print(f"Best params    = {study.best_params}", flush=True)

    regenerate_plot_only()


if __name__ == "__main__":
    main()
