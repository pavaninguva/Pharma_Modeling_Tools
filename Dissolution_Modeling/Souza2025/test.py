#!/usr/bin/env python3
"""
souza_eval_test_map_reg_t0ic.py

Final train + test evaluation script aligned with the UPDATED CV/HPO script:
  - souza_hpo_only_map_reg_t0ic.py  (closed_form + tau_fixed + t0 masked + MAP reg)

Fixed modeling choices:
  - forward_method = closed_form
  - prior_mode     = tau_fixed (tau = TAU_FIXED_VALUE)

Key alignment points:
  1) t=0 treated as an initial condition from the LOSS POV:
     - t=0 point(s) are EXCLUDED from curve likelihood and RMSE metrics via a mask.
  2) Loss matches CV objective:
       loss = L_curve_masked + gamma * R_param
     - L_curve_masked: Gaussian NLL with sigma=SIGMA_MAIN on non-t0 points (t0 masked out)
     - R_param: log-normal/normal-in-log regularization on (log lambda, log beta) using REG_SUMMARY_CSV
       Missing reg rows are ignored (contribute 0).
  3) Epoch count:
     - Uses the best row's `median_best_epoch` from TRIALS_CSV as the fixed number of epochs.

Outputs (OUT_DIR):
  - best_paramnet_checkpoint.pt
  - predicted_params_all.csv          (lambda,tau,beta per BatchID, with split tag)
  - test_parity.png                   (RMSE computed with t=0 masked out)
  - test_dynamics.png

Requires:
  pip install torch numpy pandas matplotlib scipy
"""

import os, re, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIG (match your updated CV script)
# =============================================================================
CSV_PATH   = "./Souza2025_TableS1_Final_v2_diluent_continuous.csv"
TRIALS_CSV = "./souza_hpo_map_reg_t0ic/trials_hpo_map_reg_t0ic.csv"

# Parameter-regularization targets (one row per BatchID)
REG_SUMMARY_CSV = "./souza_bayes_four_scenarios_modular_n/S3_tau_fixed_1_n1/souza_bayes_params_S3_tau_fixed_1_n1.csv"

OUT_DIR = "./souza_eval_outputs_map_reg_t0ic"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1
DEVICE = "cpu"  # "cuda" if available

# Fixed modeling choices
DILUENT_ENCODING = "continuous"
FORWARD_METHOD = "closed_form"
PRIOR_APPROACH = "tau_fixed"
TAU_FIXED_VALUE_DEFAULT = 1.0

# Likelihood noise model
SIGMA_MAIN = 0.03
SIGMA_T0   = 1e-3   # irrelevant if t=0 is masked out, kept for completeness
T0_ATOL    = 1e-12

# Treat t=0 as initial condition (mask out from likelihood/metrics)
T0_AS_INITIAL_CONDITION = True

# Closed-form numerical knobs (match CV)
CLOSED_FORM_BETA_EPS = 1e-3
CLOSED_FORM_BLEND_K  = 6.0
CLOSED_FORM_BASE_SOFTPLUS = True

# Bounds (must match CV)
LAM_BOUNDS  = (1e-6, 1e2)
TAU_BOUNDS  = (1e-3, 1e4)
BETA_BOUNDS = (1e-3, 1e1)

# Training
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 25
EPOCH_FACTOR = 1.25

# Parameter-regularization stability
PARAM_SIG_FLOOR = 0.15

# Split
TEST_FRAC = 0.20

# Outputs
PARITY_FIG   = os.path.join(OUT_DIR, "test_parity.png")
DYNAMICS_FIG = os.path.join(OUT_DIR, "test_dynamics.png")
MODEL_CKPT   = os.path.join(OUT_DIR, "best_paramnet_checkpoint.pt")
PARAMS_CSV   = os.path.join(OUT_DIR, "predicted_params_all.csv")

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


def sigma_vector_torch(t_eval_torch):
    # Kept for completeness; t=0 points are masked out if T0_AS_INITIAL_CONDITION=True.
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
# Load best config (and median_best_epoch) from trials CSV
# =============================================================================
def load_best_config(trials_csv: str):
    df = pd.read_csv(trials_csv)
    if df.empty:
        raise ValueError(f"TRIALS_CSV is empty: {trials_csv}")

    required = [
        "hidden_size","n_hidden_layers","activation","dropout","lr","gamma",
        "mean_rmse","median_best_epoch"
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Trials CSV missing required columns: {missing}")

    # optional filtering if those columns exist
    df2 = df.copy()
    if "forward_method" in df2.columns:
        df2 = df2[df2["forward_method"].astype(str).str.lower() == str(FORWARD_METHOD).lower()]
    if "prior_mode" in df2.columns:
        df2 = df2[df2["prior_mode"].astype(str).str.lower() == str(PRIOR_APPROACH).lower()]
    if "diluent_encoding" in df2.columns:
        df2 = df2[df2["diluent_encoding"].astype(str).str.lower() == str(DILUENT_ENCODING).lower()]
    if "t0_as_initial_condition" in df2.columns:
        df2 = df2[df2["t0_as_initial_condition"].astype(bool) == bool(T0_AS_INITIAL_CONDITION)]
    if df2.empty:
        df2 = df.copy()

    df2 = df2.sort_values("mean_rmse", ascending=True).reset_index(drop=True)
    best = df2.iloc[0].to_dict()

    hp = {
        "hidden_size": int(best["hidden_size"]),
        "n_hidden_layers": int(best["n_hidden_layers"]),
        "activation": str(best["activation"]),
        "dropout": float(best["dropout"]),
        "lr": float(best["lr"]),
        "gamma": float(best["gamma"]),
        "median_best_epoch": int(best["median_best_epoch"]),
        # standardize_cont is kept (even if constant True)
        "standardize_cont": bool(best.get("standardize_cont", True)),
    }

    tau_fixed_row = None
    if "tau_fixed_value" in best and np.isfinite(best["tau_fixed_value"]):
        tau_fixed_row = float(best["tau_fixed_value"])

    return hp, best, tau_fixed_row


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
    return {"mu_log": mu_log.astype(np.float32), "sig_log": sig_log.astype(np.float32), "mask": mask.astype(bool)}


# =============================================================================
# Forward solver (closed-form only)
# =============================================================================
def solve_closed_form_batch_torch_smooth(lam, tau, beta, t_eval_torch,
                                        eps=CLOSED_FORM_BETA_EPS,
                                        blend_k=CLOSED_FORM_BLEND_K,
                                        base_softplus=CLOSED_FORM_BASE_SOFTPLUS):
    """
    Closed-form batched solve for IC f(0)=0, stabilized near beta=1.

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


# =============================================================================
# Losses / metrics (masked)
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
    """
    Regularization term on log-parameters:
      log(theta) ~ Normal(mu_log, sig_log^2)
    Returns scalar behaving like "mean over full batch with missing rows contributing 0".
    """
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
    per = 0.5 * (z ** 2) + torch.log(sg)  # (+ const omitted)
    sub_mean = per.sum(dim=1).mean()

    frac = float(mask_valid.sum().item()) / float(mask_valid.numel())
    return sub_mean * frac


# =============================================================================
# ParamNet (tau_fixed => outputs (lam, beta); tau injected as constant)
# =============================================================================
def _inv_sigmoid(y):
    y = float(np.clip(y, 1e-6, 1.0 - 1e-6))
    return math.log(y / (1.0 - y))


class ParamNet(nn.Module):
    def __init__(self, in_dim, hidden_size, n_hidden_layers, activation, dropout=0.0, tau_fixed_value=1.0):
        super().__init__()
        self.tau_fixed_value = float(tau_fixed_value)

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

        tau = torch.full_like(lam, self.tau_fixed_value)
        return lam, tau, beta


# =============================================================================
# Optimizer
# =============================================================================
def build_optimizer(model: nn.Module, lr: float):
    # match CV default (adamw)
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))


# =============================================================================
# Training (full train set) for a FIXED number of epochs (median_best_epoch)
# =============================================================================
def train_full_fixed_epochs(
    model,
    Xtr,
    Ytr,
    t_eval_np,
    time_mask,
    sigma_t,
    *,
    hp,
    mu_log_tr,
    sig_log_tr,
    mask_tr,
    n_epochs,
):
    opt = build_optimizer(model, lr=float(hp["lr"]))
    t_torch = torch.tensor(np.asarray(t_eval_np, dtype=float), device=DEVICE, dtype=torch.float32)

    n_train = Xtr.shape[0]
    t0 = time.time()

    for epoch in range(1, int(n_epochs) + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)

        epoch_loss = 0.0
        n_batches = 0

        for k in range(0, n_train, BATCH_SIZE):
            idx = perm[k:k+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            lam, tau, beta = model(xb)
            pred = forward_solve(lam, tau, beta, t_torch)

            nll = gaussian_nll_hetero_masked(pred, yb, sigma_t, time_mask)
            reg = param_reg_nll_batch(lam, beta, mu_log_tr[idx], sig_log_tr[idx], mask_tr[idx])

            loss = nll + float(hp["gamma"]) * reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_loss /= max(1, n_batches)

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == int(n_epochs):
            cur_lr = float(opt.param_groups[0]["lr"])
            print(
                f"epoch={epoch:04d}/{int(n_epochs)} train_MAP={epoch_loss:.6f} "
                f"lr={cur_lr:.3g} gamma={hp['gamma']:.3g} elapsed={time.time()-t0:.1f}s",
                flush=True
            )

    return model


@torch.no_grad()
def predict_dataset(model, X, t_eval_np):
    model.eval()
    t_torch = torch.tensor(np.asarray(t_eval_np, dtype=float), device=DEVICE, dtype=torch.float32)
    lam, tau, beta = model(X)
    pred = forward_solve(lam, tau, beta, t_torch)
    return pred, lam, tau, beta


# =============================================================================
# Plots
# =============================================================================
def make_parity_plot(y_true, y_pred, out_png, text_note=None, time_mask_np=None):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if time_mask_np is not None:
        yt = yt[:, time_mask_np]
        yp = yp[:, time_mask_np]
    yt = yt.reshape(-1)
    yp = yp.reshape(-1)

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
                lam, tau, beta = model(xj)
                pred_dense = solve_closed_form_batch_torch_smooth(lam, tau, beta, t_dense_torch).cpu().numpy().reshape(-1)

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
    # sanity: fixed choices
    if str(FORWARD_METHOD).strip().lower() != "closed_form":
        raise ValueError("This script expects FORWARD_METHOD='closed_form'.")
    if str(PRIOR_APPROACH).strip().lower() != "tau_fixed":
        raise ValueError("This script expects PRIOR_APPROACH='tau_fixed'.")

    print("Loading dataset...", flush=True)
    X_raw, Y, labels, batch_ids, t_eval = load_aggregated(CSV_PATH)
    cont_cols, cat_cols, _ = get_feature_spec()
    cont_dim = len(cont_cols)
    print(f"  formulations={len(batch_ids)} | Xdim={X_raw.shape[1]} | T={Y.shape[1]}", flush=True)

    print("Loading best hyperparameters from trials CSV...", flush=True)
    hp, best_row_dict, tau_fixed_row = load_best_config(TRIALS_CSV)

    tau_fixed_value = float(TAU_FIXED_VALUE_DEFAULT if tau_fixed_row is None else tau_fixed_row)
    if not (TAU_BOUNDS[0] <= tau_fixed_value <= TAU_BOUNDS[1]):
        raise ValueError(f"tau_fixed_value={tau_fixed_value} must be within TAU_BOUNDS={TAU_BOUNDS}")

    n_epochs = math.ceil(EPOCH_FACTOR*hp["median_best_epoch"])
    if n_epochs < 1:
        raise ValueError(f"median_best_epoch must be >= 1, got {n_epochs}")

    print("Best hp:", {k: hp[k] for k in hp.keys()}, flush=True)
    print("Best row summary (selected):", {k: best_row_dict.get(k) for k in ["mean_rmse","std_rmse","median_best_epoch"] if k in best_row_dict}, flush=True)
    print(f"Using tau_fixed_value={tau_fixed_value} | epochs={n_epochs}", flush=True)

    # split (must match CV)
    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)}", flush=True)

    # standardize (global train stats)
    mu = sd = None
    if bool(hp.get("standardize_cont", True)):
        print("Applying standardization using TRAIN statistics only...", flush=True)
        X_all, mu, sd = standardize_cont_global(X_raw, cont_dim=cont_dim, train_idx=train_idx)
    else:
        X_all = X_raw
        print("Standardization disabled.", flush=True)

    # time masks + sigma(t)
    t_eval_np = np.asarray(t_eval, dtype=float)
    t_torch, is_t0, time_mask = make_time_mask_like(t_eval_np, device=DEVICE)
    sigma_t = sigma_vector_torch(t_torch)
    time_mask_np = time_mask.detach().cpu().numpy().astype(bool)

    if bool(T0_AS_INITIAL_CONDITION):
        print(f"[loss] t=0 treated as IC: excluding {int(is_t0.sum().item())} timepoint(s) from curve likelihood + RMSE.", flush=True)

    # tensors
    Xtr = torch.tensor(X_all[train_idx], device=DEVICE)
    Ytr = torch.tensor(Y[train_idx], device=DEVICE)
    Xte = torch.tensor(X_all[test_idx], device=DEVICE)
    Yte = torch.tensor(Y[test_idx], device=DEVICE)

    # reg targets aligned to ALL, then slice to train/test split
    print(f"Loading parameter-regularization targets: {REG_SUMMARY_CSV}", flush=True)
    reg = load_param_reg_targets(REG_SUMMARY_CSV, batch_ids=batch_ids)
    mu_log_all = torch.tensor(reg["mu_log"], device=DEVICE, dtype=torch.float32)
    sig_log_all = torch.tensor(reg["sig_log"], device=DEVICE, dtype=torch.float32)
    mask_all = torch.tensor(reg["mask"], device=DEVICE, dtype=torch.bool)
    print(f"Reg targets valid: {int(mask_all.sum().item())}/{len(batch_ids)}", flush=True)

    mu_log_tr = mu_log_all[train_idx]
    sig_log_tr = sig_log_all[train_idx]
    mask_tr = mask_all[train_idx]

    # model
    model = ParamNet(
        in_dim=Xtr.shape[1],
        hidden_size=hp["hidden_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        activation=hp["activation"],
        dropout=hp["dropout"],
        tau_fixed_value=tau_fixed_value,
    ).to(DEVICE)

    print(
        "\nTraining best model on FULL training set...\n"
        f"  fixed: prior=tau_fixed (tau={tau_fixed_value}) | forward=closed_form | t0_as_IC={int(bool(T0_AS_INITIAL_CONDITION))}\n"
        f"  lr={hp['lr']:.3g} gamma={hp['gamma']:.3g} epochs={n_epochs}\n",
        flush=True
    )

    train_full_fixed_epochs(
        model, Xtr, Ytr, t_eval_np, time_mask, sigma_t,
        hp=hp,
        mu_log_tr=mu_log_tr,
        sig_log_tr=sig_log_tr,
        mask_tr=mask_tr,
        n_epochs=n_epochs,
    )

    # Save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "hp": hp,
        "selected_best_row": best_row_dict,
        "tau_fixed_value": float(tau_fixed_value),
        "n_epochs": int(n_epochs),
        "in_dim": int(Xtr.shape[1]),
        "standardize_cont": bool(hp.get("standardize_cont", True)),
        "mu": (mu.astype(np.float32) if mu is not None else None),
        "sd": (sd.astype(np.float32) if sd is not None else None),
        "t_eval": np.asarray(t_eval, dtype=np.float32),
        "seed": int(SEED),
        "csv_path": str(CSV_PATH),
        "trials_csv": str(TRIALS_CSV),
        "reg_summary_csv": str(REG_SUMMARY_CSV),
        "fixed": {"prior_approach": str(PRIOR_APPROACH), "forward_method": str(FORWARD_METHOD)},
        "loss": {
            "sigma_main": float(SIGMA_MAIN),
            "sigma_t0": float(SIGMA_T0),
            "t0_atol": float(T0_ATOL),
            "t0_as_initial_condition": bool(T0_AS_INITIAL_CONDITION),
        },
        "closed_form": {
            "beta_eps": float(CLOSED_FORM_BETA_EPS),
            "blend_k": float(CLOSED_FORM_BLEND_K),
            "base_softplus": bool(CLOSED_FORM_BASE_SOFTPLUS),
        },
        "bounds": {"lambda": LAM_BOUNDS, "tau": TAU_BOUNDS, "beta": BETA_BOUNDS},
        "diluent_encoding": str(DILUENT_ENCODING),
    }
    torch.save(ckpt, MODEL_CKPT)
    print(f"Saved trained model checkpoint -> {MODEL_CKPT}", flush=True)

    # Predict on test
    print("Predicting on test set...", flush=True)
    with torch.no_grad():
        Ypred_te, lam_te, tau_te, beta_te = predict_dataset(model, Xte, t_eval_np)

    test_abs_rmse = float(report_abs_rmse(Ypred_te, Yte, time_mask).item())
    test_curve_nll = float(gaussian_nll_hetero_masked(Ypred_te, Yte, sigma_t, time_mask).item())
    print(f"Test ABS-RMSE (masked t=0) = {test_abs_rmse:.6f}", flush=True)
    print(f"Test curve NLL (masked t=0) = {test_curve_nll:.6f}", flush=True)

    # Save predicted params for ALL formulations
    print("Saving predicted parameters for all formulations...", flush=True)
    with torch.no_grad():
        Xall = torch.tensor(X_all, device=DEVICE)
        lam_all, tau_all, beta_all = model(Xall)

    split = np.array(["train"] * len(batch_ids), dtype=object)
    split[test_idx] = "test"

    df_params = pd.DataFrame({
        "BatchID": batch_ids,
        "split": split,
        "lambda_pred": lam_all.detach().cpu().numpy(),
        "tau_pred": tau_all.detach().cpu().numpy(),
        "beta_pred": beta_all.detach().cpu().numpy(),
    })
    df_params.to_csv(PARAMS_CSV, index=False)
    print(f"Saved predicted parameters -> {PARAMS_CSV}", flush=True)

    # Plots
    print(f"Saving parity plot -> {PARITY_FIG}", flush=True)
    note = (
        f"tau_fixed (tau={tau_fixed_value}), closed_form, t0_as_IC={int(bool(T0_AS_INITIAL_CONDITION))}\n"
        f"epochs={n_epochs}  gamma={hp['gamma']:.3g}\n"
        f"test_ABS-RMSE(masked t0)={test_abs_rmse:.4f}"
    )
    make_parity_plot(Yte.cpu().numpy(), Ypred_te.cpu().numpy(), PARITY_FIG, text_note=note, time_mask_np=time_mask_np)

    print(f"Saving dynamics plot -> {DYNAMICS_FIG}", flush=True)
    test_batch_ids = np.asarray(batch_ids)[test_idx]
    make_dynamics_plot(test_batch_ids, t_eval_np, Yte.cpu().numpy(), model, Xte, DYNAMICS_FIG, curves_per_subplot=5)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()