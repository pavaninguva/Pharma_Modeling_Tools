#!/usr/bin/env python3
"""
souza_hpo_only_map_reg_t0ic.py

HPO-only (Optuna) cross-validation for hybrid mechanistic+NN dissolution model.

Fixed modeling choices:
  - FORWARD_METHOD = "closed_form"
  - PRIOR_APPROACH = "tau_fixed" with TAU_FIXED_VALUE

Mechanistic model:
  df/dt = lam * (t/(tau+t)) * (1-f)^beta

t=0 treated as an initial condition from the LOSS POV:
  - The datapoint(s) at t=0 are EXCLUDED from the curve likelihood and RMSE metrics.
  - The forward model still predicts at all provided times (including t=0); we just mask it out.

Training objective (single fixed MAP-style objective with parameter regularization):
  loss = L_curve_masked + gamma * R_param

Where:
  - L_curve_masked: heteroscedastic Gaussian NLL on curve points excluding t=0
  - R_param: per-curve regularization on log-parameters using precomputed (mu_log, sig_log):
        sum_k [ 0.5 * ((log theta_k - mu_k)/sig_k)^2 + log(sig_k) ]   (+ const)
    for active params. For tau_fixed, active params are (lambda, beta).
    Missing rows in the reg CSV are ignored (regularization term = 0).

What this script DOES:
  - Runs Optuna CV on training split only, writes TRIALS_CSV incrementally (resume-safe)
  - Adds median_best_epoch (median epoch of best validation MAP across folds) to TRIALS_CSV
  - Generates a parallel-coordinates plot (PC_PNG) from TRIALS_CSV
  - Can run "plot-only" mode (RUN_HPO=False) to regenerate PC plot

What this script does NOT do:
  - No final training on best hyperparameters
  - No final evaluation/export of predicted parameters

HPO hyperparameters:
  - hidden_size, n_hidden_layers, activation, dropout, lr
  - gamma (strength of parameter regularization)

Requires:
  pip install torch optuna numpy pandas matplotlib scipy
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


# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH = "./Souza2025_TableS1_Final_v2_diluent_continuous.csv"

# Parameter-regularization targets (one row per BatchID)
# Accepts columns either:
#   - log_lambda_med, log_lambda_sd, log_beta_med, log_beta_sd
# or
#   - lambda_med, lambda_q16, lambda_q84, beta_med, beta_q16, beta_q84
REG_SUMMARY_CSV = "./souza_bayes_four_scenarios_modular_n/S3_tau_fixed_1_n1/souza_bayes_params_S3_tau_fixed_1_n1.csv"

SEED = 1
DEVICE = "cpu"  # "cuda" if available

# Fixed forward/prior choices
FORWARD_METHOD = "closed_form"
PRIOR_APPROACH = "tau_fixed"
TAU_FIXED_VALUE = 1.0

# Split / CV
TEST_FRAC = 0.20
K_FOLDS = 5

# Likelihood noise model (heteroscedastic)
# NOTE: with t=0 masked out, only SIGMA_MAIN matters for training/metrics.
SIGMA_MAIN = 0.03
SIGMA_T0   = 1e-3
T0_ATOL    = 1e-12

# Treat t=0 as initial condition (mask out from likelihood/metrics)
T0_AS_INITIAL_CONDITION = True

# Training
MAX_EPOCHS = 700
PATIENCE = 100
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 10

# Optimizer (fixed)
OPTIMIZER_TYPE = "adamw"  # "adam" or "adamw"

# Parameter-regularization stability
PARAM_SIG_FLOOR = 0.15

# Bounds (natural space)
LAM_BOUNDS  = (1e-6, 1e2)
TAU_BOUNDS  = (1e-3, 1e4)
BETA_BOUNDS = (1e-3, 1e1)

# Diluent encoding
DILUENT_ENCODING = "continuous"

# HPO controls
RUN_HPO = True
N_TRIALS = 300
OPTUNA_STORAGE = "sqlite:///souza_optuna_hpo_map_reg.db"
OPTUNA_STUDY_NAME = f"souza_hpo_{DILUENT_ENCODING}_{PRIOR_APPROACH}_{FORWARD_METHOD}_t0IC"
USE_PRUNER = False

# Search space
HIDDEN_SIZE_MIN = 12
HIDDEN_SIZE_MAX = 64
HIDDEN_SIZE_STEP = 1
N_HIDDEN_LAYERS = [2, 3, 4, 5]
ACTIVATIONS = ["swish", "leakyrelu", "gelu", "mish"]
DROPOUTS = [0.0, 0.1, 0.2]
STANDARDIZE_OPTIONS = [True]   # kept, but removed from parallel coordinates plot
LR_MIN = 1e-3
LR_MAX = 5e-2
GAMMA_MIN = 1e-4
GAMMA_MAX = 1.0

# closed-form numerical knobs
CLOSED_FORM_BETA_EPS = 1e-3
CLOSED_FORM_BLEND_K  = 6.0
CLOSED_FORM_BASE_SOFTPLUS = True

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
OUT_DIR = "souza_hpo_map_reg_t0ic"
os.makedirs(OUT_DIR, exist_ok=True)

TRIALS_CSV = os.path.join(OUT_DIR, "trials_hpo_map_reg_t0ic.csv")
PC_PNG     = os.path.join(OUT_DIR, "parallel_coords_hpo_map_reg_t0ic.png")


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
# Time masks (t=0 as initial condition)
# =============================================================================
def make_time_mask_like(t_eval_np, device):
    t = torch.tensor(np.asarray(t_eval_np, dtype=float), device=device, dtype=torch.float32)
    is_t0 = torch.isclose(
        t,
        torch.tensor(0.0, device=device, dtype=t.dtype),
        atol=float(T0_ATOL),
        rtol=0.0
    )
    if bool(T0_AS_INITIAL_CONDITION):
        mask_like = ~is_t0
    else:
        mask_like = torch.ones_like(is_t0, dtype=torch.bool)
    return t, is_t0, mask_like


# =============================================================================
# Parameter-regularization targets loader
# =============================================================================
def _safe_log(x):
    x = np.asarray(x, float)
    x = np.clip(x, 1e-300, np.inf)
    return np.log(x)

def _sigma_from_q16_q84(q16, q84):
    q16 = np.asarray(q16, float)
    q84 = np.asarray(q84, float)
    return 0.5 * (q84 - q16)

def load_param_reg_targets(reg_csv: str, batch_ids):
    """
    For PRIOR_APPROACH='tau_fixed', active params are [lambda, beta].
    Returns:
      mu_log: (N,2), sig_log: (N,2), mask: (N,) bool
    """
    df = pd.read_csv(reg_csv)
    if BATCH_COL not in df.columns:
        raise ValueError(f"REG_SUMMARY_CSV must have column '{BATCH_COL}'")

    df[BATCH_COL] = df[BATCH_COL].astype(str)
    df = df.set_index(BATCH_COL)

    active = ["lambda", "beta"]
    mu_log = np.full((len(batch_ids), len(active)), np.nan, dtype=float)
    sig_log = np.full((len(batch_ids), len(active)), np.nan, dtype=float)

    for i, bid in enumerate(batch_ids):
        sid = str(bid)
        if sid not in df.index:
            continue
        row = df.loc[sid]

        for j, p in enumerate(active):
            col_mu = f"log_{p}_med"
            col_sd = f"log_{p}_sd"
            if (col_mu in df.columns) and (col_sd in df.columns):
                mu_log[i, j] = float(row[col_mu])
                sig_log[i, j] = float(row[col_sd])
                continue

            col_med = f"{p}_med"
            col_q16 = f"{p}_q16"
            col_q84 = f"{p}_q84"
            if (col_med in df.columns) and (col_q16 in df.columns) and (col_q84 in df.columns):
                med = float(row[col_med])
                q16 = float(row[col_q16])
                q84 = float(row[col_q84])
                mu_log[i, j] = float(_safe_log(med))
                sig_log[i, j] = float(_sigma_from_q16_q84(_safe_log(q16), _safe_log(q84)))
                continue

            if col_med in df.columns:
                med = float(row[col_med])
                mu_log[i, j] = float(_safe_log(med))
                sig_log[i, j] = 1.0

    sig_log = np.where(np.isfinite(sig_log), np.maximum(sig_log, float(PARAM_SIG_FLOOR)), np.nan)
    mask = np.all(np.isfinite(mu_log), axis=1) & np.all(np.isfinite(sig_log), axis=1)
    return {"mu_log": mu_log, "sig_log": sig_log, "mask": mask}


# =============================================================================
# Forward solver (closed-form only)
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

    # beta=1 limit: log u = -A
    logu_limit = -A

    delta_bt = delta.expand_as(A)
    delta_safe = torch.where(torch.abs(delta_bt) < float(eps), torch.ones_like(delta_bt), delta_bt)
    logu_gen = torch.log(base_safe) / delta_safe

    x = (torch.abs(delta_bt) - float(eps)) / (float(eps) / float(blend_k))
    w = torch.sigmoid(x)
    w = torch.where(torch.abs(delta_bt) < float(eps), torch.zeros_like(w), w)

    logu = (1.0 - w) * logu_limit + w * logu_gen
    u = torch.exp(logu)
    f = 1.0 - u
    return torch.clamp(f, 0.0, 1.0)


def forward_solve(lam, tau, beta, t_eval_torch):
    return solve_closed_form_batch_torch_smooth(lam, tau, beta, t_eval_torch)


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
# Losses
# =============================================================================
def gaussian_nll_hetero_masked(pred, target, sigma_t, time_mask):
    pred_m = pred[:, time_mask]
    targ_m = target[:, time_mask].to(dtype=pred.dtype)
    sig_m  = sigma_t[time_mask].view(1, -1).to(dtype=pred.dtype)

    err = pred_m - targ_m
    nll = 0.5 * ((err / sig_m) ** 2 + 2.0 * torch.log(sig_m) + math.log(2.0 * math.pi))
    return nll.sum(dim=1).mean()


def report_abs_rmse(pred, target, time_mask):
    pred = pred.to(dtype=torch.float32)[:, time_mask]
    target = target.to(dtype=torch.float32)[:, time_mask]
    mse = ((pred - target) ** 2).mean(dim=1)
    return torch.sqrt(mse).mean()


def param_reg_nll_batch(lam, beta, mu_log, sig_log, mask_valid):
    if mask_valid is None:
        mask_valid = torch.ones(lam.shape[0], device=lam.device, dtype=torch.bool)
    if not torch.any(mask_valid):
        return torch.zeros((), device=lam.device, dtype=torch.float32)

    lam_m = lam[mask_valid]
    beta_m = beta[mask_valid]
    mu_m = mu_log[mask_valid]
    sg_m = sig_log[mask_valid]

    pred_log = torch.stack(
        [
            torch.log(torch.clamp(lam_m, min=1e-30)),
            torch.log(torch.clamp(beta_m, min=1e-30)),
        ],
        dim=1
    )

    sg = torch.clamp(sg_m, min=float(PARAM_SIG_FLOOR))
    z = (pred_log - mu_m) / sg
    per = 0.5 * (z ** 2) + torch.log(sg)
    sub_mean = per.sum(dim=1).mean()

    frac = float(mask_valid.sum().item()) / float(mask_valid.numel())
    return sub_mean * frac


# =============================================================================
# ParamNet
# =============================================================================
def _inv_sigmoid(y):
    y = float(np.clip(y, 1e-6, 1.0 - 1e-6))
    return math.log(y / (1.0 - y))


class ParamNet(nn.Module):
    def __init__(self, in_dim, hidden_size, n_hidden_layers, activation, dropout=0.0):
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

        layers = []
        d = in_dim
        for _ in range(int(n_hidden_layers)):
            layers += [nn.Linear(d, int(hidden_size)), act()]
            if dropout and float(dropout) > 0.0:
                layers += [nn.Dropout(p=float(dropout))]
            d = int(hidden_size)

        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 2)  # outputs (lam, beta)

        with torch.no_grad():
            loglam0 = math.log(2.5e-2)
            logbeta0 = math.log(1.0)

            loglam_lb, loglam_ub = math.log(LAM_BOUNDS[0]), math.log(LAM_BOUNDS[1])
            logbeta_lb, logbeta_ub = math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1])

            def frac(logx, lb, ub):
                return float(np.clip((logx - lb) / (ub - lb), 1e-6, 1.0 - 1e-6))

            b0 = _inv_sigmoid(frac(loglam0, loglam_lb, loglam_ub))
            b1 = _inv_sigmoid(frac(logbeta0, logbeta_lb, logbeta_ub))
            self.out.bias[:] = torch.tensor([b0, b1], dtype=torch.float32)

    def forward(self, x):
        h = self.out(self.body(x))

        loglam_lb, loglam_ub = math.log(LAM_BOUNDS[0]), math.log(LAM_BOUNDS[1])
        logbeta_lb, logbeta_ub = math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1])

        s0 = torch.sigmoid(h[:, 0])
        log_lam = loglam_lb + (loglam_ub - loglam_lb) * s0
        lam = torch.exp(log_lam)

        s1 = torch.sigmoid(h[:, 1])
        log_beta = logbeta_lb + (logbeta_ub - logbeta_lb) * s1
        beta = torch.exp(log_beta)

        tau = torch.full_like(lam, float(TAU_FIXED_VALUE))
        return lam, tau, beta


# =============================================================================
# Optimizer
# =============================================================================
def build_optimizer(model: nn.Module, lr: float):
    opt_name = str(OPTIMIZER_TYPE).strip().lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))
    raise ValueError(f"Unknown OPTIMIZER_TYPE='{OPTIMIZER_TYPE}'.")


# =============================================================================
# TRAIN one fold (MAP objective).
# Returns:
#   (val_rmse_at_best, best_epoch, best_val_map)
# =============================================================================
def train_one_fold(
    X_raw_train, Y_train, t_eval_np, time_mask, sigma_t,
    train_idx, val_idx,
    hp, fold_id, config_id, cont_dim,
    mu_log_train=None, sig_log_train=None, mask_reg_train=None,
):
    # fold-safe standardization
    if bool(hp["standardize_cont"]):
        X_fold = standardize_cont_fold(X_raw_train, cont_dim=cont_dim, train_idx=train_idx)
    else:
        X_fold = X_raw_train

    Xtr = torch.tensor(X_fold[train_idx], device=DEVICE)
    Ytr = torch.tensor(Y_train[train_idx], device=DEVICE)
    Xva = torch.tensor(X_fold[val_idx], device=DEVICE)
    Yva = torch.tensor(Y_train[val_idx], device=DEVICE)

    # reg targets for this fold (aligned to TRAIN-set indices)
    if (mu_log_train is not None) and (sig_log_train is not None) and (mask_reg_train is not None):
        mu_tr = torch.tensor(mu_log_train[train_idx], device=DEVICE, dtype=torch.float32)
        sg_tr = torch.tensor(sig_log_train[train_idx], device=DEVICE, dtype=torch.float32)
        mk_tr = torch.tensor(mask_reg_train[train_idx].astype(bool), device=DEVICE)

        mu_va = torch.tensor(mu_log_train[val_idx], device=DEVICE, dtype=torch.float32)
        sg_va = torch.tensor(sig_log_train[val_idx], device=DEVICE, dtype=torch.float32)
        mk_va = torch.tensor(mask_reg_train[val_idx].astype(bool), device=DEVICE)
    else:
        mu_tr = sg_tr = mk_tr = None
        mu_va = sg_va = mk_va = None

    model = ParamNet(
        in_dim=Xtr.shape[1],
        hidden_size=hp["hidden_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        activation=hp["activation"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    opt = build_optimizer(model, lr=hp["lr"])
    t_torch = torch.tensor(np.asarray(t_eval_np, dtype=float), device=DEVICE, dtype=torch.float32)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    n_train = Xtr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        last_train = None

        for k in range(0, n_train, BATCH_SIZE):
            idx = perm[k:k+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            lam, tau, beta = model(xb)
            pred = forward_solve(lam, tau, beta, t_torch)

            nll = gaussian_nll_hetero_masked(pred, yb, sigma_t, time_mask)

            if (mu_tr is not None) and (sg_tr is not None) and (mk_tr is not None):
                reg = param_reg_nll_batch(lam, beta, mu_tr[idx], sg_tr[idx], mk_tr[idx])
            else:
                reg = torch.zeros((), device=DEVICE, dtype=torch.float32)

            loss = nll + float(hp["gamma"]) * reg
            last_train = float(loss.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # validate (MAP loss)
        model.eval()
        with torch.no_grad():
            lam, tau, beta = model(Xva)
            pred = forward_solve(lam, tau, beta, t_torch)

            nll_va = gaussian_nll_hetero_masked(pred, Yva, sigma_t, time_mask)
            if (mu_va is not None) and (sg_va is not None) and (mk_va is not None):
                reg_va = param_reg_nll_batch(lam, beta, mu_va, sg_va, mk_va)
            else:
                reg_va = torch.zeros((), device=DEVICE, dtype=torch.float32)

            val_map = float((nll_va + float(hp["gamma"]) * reg_va).item())
            val_rmse = float(report_abs_rmse(pred, Yva, time_mask).item())

        cur_lr = float(opt.param_groups[0]["lr"])
        if (fold_id == 1) and (epoch == 1 or epoch % LOG_EVERY == 0):
            print(
                f"      cfg={config_id} fold={fold_id} epoch={epoch:03d} "
                f"train_MAP~{last_train:.6f} val_MAP={val_map:.6f} best_val={best_val:.6f} "
                f"val_RMSE={val_rmse:.6f} lr={cur_lr:.3g}",
                flush=True
            )

        if val_map < best_val - 1e-12:
            best_val = val_map
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = int(epoch)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Return RMSE at best checkpoint + best_epoch (and best_val_map for logging)
    model.eval()
    with torch.no_grad():
        lam, tau, beta = model(Xva)
        pred = forward_solve(lam, tau, beta, torch.tensor(np.asarray(t_eval_np, dtype=float), device=DEVICE, dtype=torch.float32))
        rmse_best = float(report_abs_rmse(pred, Yva, time_mask).item())

    return rmse_best, int(best_epoch), float(best_val)


# =============================================================================
# Parallel coordinates plot
#   - scheduler removed
#   - standardize_cont removed
#   - plot title removed
# =============================================================================
def parallel_coordinates_plot(df_trials, out_png):
    if df_trials.empty:
        raise ValueError("df_trials is empty.")

    dfp = df_trials.copy().reset_index(drop=True)

    required = {
        "hidden_size","n_hidden_layers","activation","dropout","lr","gamma",
        "median_best_epoch","mean_rmse","std_rmse"
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

    dims = [
        ("hidden_size",         dfp["hidden_size"].astype(float).to_numpy(), "linear"),
        ("n_hidden_layers",     dfp["n_hidden_layers"].astype(float).to_numpy(), "linear"),
        ("activation",          dfp["activation_y"].to_numpy(float), "categorical_activation"),
        ("dropout",             dfp["dropout"].astype(float).to_numpy(), "linear"),
        ("learning_rate",       dfp["lr"].astype(float).to_numpy(), "log"),
        ("gamma",               dfp["gamma"].astype(float).to_numpy(), "log"),
        ("median_best_epoch",   dfp["median_best_epoch"].astype(float).to_numpy(), "linear"),
        ("mean_rmse",           dfp["mean_rmse"].astype(float).to_numpy(), "linear"),
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

    fig, ax = plt.subplots(figsize=(16.0, 6), dpi=220)
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
    cols_show = [c for c in ["hidden_size","n_hidden_layers","activation","dropout","lr","gamma","median_best_epoch","mean_rmse","std_rmse"]
                 if c in df_trials.columns]
    print(df_trials.sort_values("mean_rmse").head(10)[cols_show].to_string(index=False), flush=True)


# =============================================================================
# MAIN: HPO only
# =============================================================================
def main():
    if not bool(RUN_HPO):
        regenerate_plot_only()
        return

    # sanity: fixed configs
    if str(FORWARD_METHOD).strip().lower() != "closed_form":
        raise ValueError("This script expects FORWARD_METHOD='closed_form'.")
    if str(PRIOR_APPROACH).strip().lower() != "tau_fixed":
        raise ValueError("This script expects PRIOR_APPROACH='tau_fixed'.")

    if not (TAU_BOUNDS[0] <= float(TAU_FIXED_VALUE) <= TAU_BOUNDS[1]):
        raise ValueError(f"TAU_FIXED_VALUE={TAU_FIXED_VALUE} must be within TAU_BOUNDS={TAU_BOUNDS}")

    cont_cols, cat_cols, dil_enc_tag = get_feature_spec()
    cont_dim = len(cont_cols)

    print("Loading and aggregating dataset...", flush=True)
    X_raw, Y, labels, batch_ids, t_eval = load_aggregated(CSV_PATH)
    print(f"  Loaded {len(batch_ids)} formulations; X dim={X_raw.shape[1]}, curve length T={Y.shape[1]}", flush=True)
    print(f"  Diluent encoding: {dil_enc_tag} | cont_dim={cont_dim} | cat_dim={len(cat_cols)}", flush=True)
    print(f"  Time range: {t_eval.min()}..{t_eval.max()} minutes | fixed forward={FORWARD_METHOD}", flush=True)
    print(f"  Fixed prior: {PRIOR_APPROACH} (TAU_FIXED_VALUE={TAU_FIXED_VALUE})", flush=True)

    # time masks + sigma(t)
    t_eval_np = np.asarray(t_eval, dtype=float)
    t_torch, is_t0, time_mask = make_time_mask_like(t_eval_np, device=DEVICE)
    sigma_t = sigma_vector_torch(t_torch)

    if bool(T0_AS_INITIAL_CONDITION):
        print(f"[loss] t=0 treated as IC: excluding {int(is_t0.sum().item())} timepoint(s) from curve likelihood + RMSE.", flush=True)

    # outer split
    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)} (formulation-level)", flush=True)

    # CV only on training set
    X_train_raw = X_raw[train_idx]
    Y_train = Y[train_idx]
    labels_train = labels[train_idx]

    splits = stratified_folds(labels_train, k=K_FOLDS, seed=SEED)
    print(f"Prepared {K_FOLDS}-fold CV splits on training set.", flush=True)

    # load parameter-regularization targets aligned to ALL, then slice to training split
    print(f"Loading parameter-regularization targets: {REG_SUMMARY_CSV}", flush=True)
    reg = load_param_reg_targets(REG_SUMMARY_CSV, batch_ids=batch_ids)
    mu_log_all = reg["mu_log"].astype(np.float32)
    sig_log_all = reg["sig_log"].astype(np.float32)
    mask_all = reg["mask"].astype(bool)
    print(f"Reg targets valid: {int(mask_all.sum())}/{len(batch_ids)} formulations", flush=True)

    mu_log_train = mu_log_all[train_idx]
    sig_log_train = sig_log_all[train_idx]
    mask_train = mask_all[train_idx]

    # Resume support via TRIALS_CSV
    done_keys = set()
    if os.path.exists(TRIALS_CSV):
        prev = pd.read_csv(TRIALS_CSV)
        for _, r in prev.iterrows():
            key = (
                int(r["hidden_size"]),
                int(r["n_hidden_layers"]),
                str(r["activation"]),
                float(r["dropout"]),
                bool(r.get("standardize_cont", True)),
                float(r["lr"]),
                float(r["gamma"]),
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
        gamma = trial.suggest_float("gamma", GAMMA_MIN, GAMMA_MAX, log=True)

        key = (int(hs), int(nl), str(act), float(drop), bool(stdz), float(lr), float(gamma))

        # reuse if already computed
        if key in done_keys and os.path.exists(TRIALS_CSV):
            df_prev = pd.read_csv(TRIALS_CSV)
            m = (
                (df_prev["hidden_size"].astype(int) == int(hs)) &
                (df_prev["n_hidden_layers"].astype(int) == int(nl)) &
                (df_prev["activation"].astype(str) == str(act)) &
                (np.isclose(df_prev["dropout"].astype(float), float(drop))) &
                (df_prev.get("standardize_cont", True).astype(bool) == bool(stdz)) &
                (np.isclose(df_prev["lr"].astype(float), float(lr))) &
                (np.isclose(df_prev["gamma"].astype(float), float(gamma)))
            )
            if m.any():
                return float(df_prev.loc[m, "mean_rmse"].iloc[0])

        config_id = f"trial{trial.number}"
        print(
            f"\n[TRIAL {trial.number}] "
            f"hs={hs} nl={nl} act={act} drop={drop:g} std={int(bool(stdz))} lr={lr:.3g} gamma={gamma:.3g}",
            flush=True
        )

        hp = {
            "hidden_size": int(hs),
            "n_hidden_layers": int(nl),
            "activation": str(act),
            "dropout": float(drop),
            "standardize_cont": bool(stdz),
            "lr": float(lr),
            "gamma": float(gamma),
        }

        fold_scores = []
        fold_best_epochs = []
        fold_best_vals = []
        t0 = time.time()

        for fold_id, (tr, va) in enumerate(splits, 1):
            rmse, best_ep, best_val = train_one_fold(
                X_train_raw, Y_train, t_eval_np, time_mask, sigma_t,
                tr, va,
                hp,
                fold_id=fold_id,
                config_id=config_id,
                cont_dim=cont_dim,
                mu_log_train=mu_log_train,
                sig_log_train=sig_log_train,
                mask_reg_train=mask_train,
            )
            fold_scores.append(rmse)
            fold_best_epochs.append(best_ep)
            fold_best_vals.append(best_val)

            if USE_PRUNER:
                trial.report(float(np.mean(fold_scores)), step=fold_id)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_rmse = float(np.mean(fold_scores))
        std_rmse  = float(np.std(fold_scores))
        median_best_epoch = int(np.median(np.asarray(fold_best_epochs, dtype=float)))

        print(
            f"[TRIAL DONE] {config_id} | mean_rmse={mean_rmse:.5f} ± {std_rmse:.5f} | "
            f"median_best_epoch={median_best_epoch} | elapsed={time.time()-t0:.1f}s",
            flush=True
        )

        row = {
            "hidden_size": int(hs),
            "n_hidden_layers": int(nl),
            "activation": str(act),
            "dropout": float(drop),
            "standardize_cont": bool(stdz),  # kept for traceability; NOT plotted
            "lr": float(lr),
            "gamma": float(gamma),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "median_best_epoch": median_best_epoch,
            # traceability
            "optimizer": str(OPTIMIZER_TYPE).strip().lower(),
            "seed": int(SEED),
            "diluent_encoding": str(dil_enc_tag),
            "prior_mode": str(PRIOR_APPROACH),
            "tau_fixed_value": float(TAU_FIXED_VALUE),
            "forward_method": str(FORWARD_METHOD),
            "t0_as_initial_condition": bool(T0_AS_INITIAL_CONDITION),
            "sigma_main": float(SIGMA_MAIN),
            "t0_atol": float(T0_ATOL),
            "reg_csv": str(REG_SUMMARY_CSV),
        }

        out_df = pd.DataFrame([row])
        if os.path.exists(TRIALS_CSV):
            prev = pd.read_csv(TRIALS_CSV)
            out_df = pd.concat([prev, out_df], ignore_index=True)

        out_df = out_df.drop_duplicates(
            subset=["hidden_size","n_hidden_layers","activation","dropout","standardize_cont","lr","gamma"]
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