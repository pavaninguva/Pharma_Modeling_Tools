#!/usr/bin/env python3
"""
Train the *best* hyperparameter configuration (from a grid-search/Optuna results CSV)
on the full training set, then evaluate on the held-out test set.

Outputs:
1) Parity plot: test data vs model predictions (all test points across all curves)
2) Dynamics plots: continuous predicted curves overlaid with test data points
3) Saves model checkpoint

Assumptions:
- TRIALS_CSV has columns:
  hidden_size, n_hidden_layers, activation, dropout, standardize_cont, lr, mean_rmse, std_rmse
  NOTE: if TRIALS_CSV was produced by your *relative* crossval run, mean_rmse/std_rmse are REL-RMSEs.

Alignment updates:
- Can train with RELATIVE squared error (REL-MSE) OR ABSOLUTE squared error (ABS-MSE).
- Can switch Adam / AdamW.
- Can switch ReduceLROnPlateau on/off (stepped on train loss since no val split).
- Optionally reads optimizer/scheduler/loss_mode from TRIALS_CSV if those columns exist.
- Adds odeint failure handling similar to your crossval script.
"""

import os, re, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# -------------------------
# CONFIG
# -------------------------
DATA_CSV   = "./Souza2025_TableS1_Final.csv"
TRIALS_CSV = "./souza_clean_gridsearch_reparam_trials.csv"

SEED = 1
DEVICE = "cpu"  # set "cuda" if desired and available

TEST_FRAC = 0.20

# ODE solver settings
ODE_METHOD = "dopri5"
ODE_RTOL = 1e-6
ODE_ATOL = 1e-8

# Fixed exponent n
N_FIXED = 1.0

# training settings for the final fit
MAX_EPOCHS = 700
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 25

# Optimizer / scheduler / loss switches
# If TRIALS_CSV contains these columns, they can override defaults:
#   optimizer: "adam" or "adamw"
#   use_scheduler: bool
#   loss_mode: "relative" or "absolute"
OPTIMIZER_DEFAULT = "adam"      # "adamw" or "adam"
USE_SCHEDULER_DEFAULT = True    # ReduceLROnPlateau
LOSS_MODE_DEFAULT = "absolute"  # "relative" or "absolute"

SCHED_FACTOR = 0.9
SCHED_PATIENCE = 50
SCHED_MIN_LR = 1e-6
SCHED_COOLDOWN = 0

# Relative error settings
REL_EPS = 1e-3

# Warm-start settings
WARM_START = False
WARM_START_CKPT = "./best_model_eval_outputs/best_paramnet_checkpoint.pt"  # or wherever
WARM_START_STRICT = True  # strict=True requires exact architecture match

# plot outputs
OUT_DIR = "./best_model_eval_outputs"
PARITY_FIG = os.path.join(OUT_DIR, "test_parity.png")
DYNAMICS_FIG = os.path.join(OUT_DIR, "test_dynamics.png")
MODEL_CKPT = os.path.join(OUT_DIR, "best_paramnet_checkpoint.pt")

# features in cleaned CSV
CONT_COLS = ["PEO_N750_pct", "PEO_1105_pct", "PEO_N60K_pct", "PEO_303_pct", "Diluent_pct"]
ONEHOT_COLS = ["Diluent_G721", "Diluent_SMCC", "Diluent_MD_IT12"]

# keys
BATCH_COL = "BatchID"
CAT_COL   = "Diluent_type"
TIME_COL  = "time_min"
Y_COL     = "release_frac"


# -------------------------
# REPRO
# -------------------------
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


# -------------------------
# DATA: aggregate to one row per BatchID
# -------------------------
def load_aggregated(csv_path: str):
    df = pd.read_csv(csv_path)

    needed = {BATCH_COL, CAT_COL, TIME_COL, Y_COL, *CONT_COLS, *ONEHOT_COLS}
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

        x_cont = dfi.iloc[0][CONT_COLS].to_numpy(dtype=float)
        x_oh   = dfi.iloc[0][ONEHOT_COLS].to_numpy(dtype=float)
        X_list.append(np.concatenate([x_cont, x_oh], axis=0))
        Y_list.append(dfi[Y_COL].to_numpy(dtype=float))
        labels.append(str(dfi.iloc[0][CAT_COL]))
        groups.append(bid)

    X_raw = np.vstack(X_list).astype(np.float32)
    Y     = np.vstack(Y_list).astype(np.float32)
    labels = np.array(labels)
    groups = np.array(groups)
    return X_raw, Y, labels, groups, t_eval


# -------------------------
# SPLIT: formulation-level stratified by label
# -------------------------
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


# -------------------------
# Standardization (continuous only) - global (train stats only)
# -------------------------
def standardize_cont_global(X, cont_dim, train_idx):
    """
    Returns (Xs, mu, sd) where mu/sd are computed using ONLY train_idx.
    Then applies that same transform to *all rows* of X (train + test).
    """
    Xs = X.copy()
    mu = Xs[train_idx, :cont_dim].mean(axis=0, keepdims=True)
    sd = Xs[train_idx, :cont_dim].std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs[:, :cont_dim] = (Xs[:, :cont_dim] - mu) / sd
    return Xs, mu, sd


# -------------------------
# Load best hyperparameters from trials CSV
# -------------------------
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


def _as_loss_mode(x, default="relative"):
    if x is None:
        return str(default).lower()
    s = str(x).strip().lower()
    if s in ("rel", "relative", "rel_mse", "rel-mse"):
        return "relative"
    if s in ("abs", "absolute", "abs_mse", "abs-mse"):
        return "absolute"
    return str(default).lower()


def load_best_config(trials_csv: str):
    df = pd.read_csv(trials_csv)

    # backward compatibility
    if "standardize_cont" not in df.columns:
        df["standardize_cont"] = True

    required = ["hidden_size", "n_hidden_layers", "activation", "dropout", "standardize_cont", "lr", "mean_rmse"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Trials CSV missing columns: {missing}")

    df = df.sort_values("mean_rmse", ascending=True).reset_index(drop=True)
    best = df.iloc[0]

    optimizer = str(best["optimizer"]).lower() if "optimizer" in df.columns else OPTIMIZER_DEFAULT
    use_scheduler = _as_bool(best["use_scheduler"], default=USE_SCHEDULER_DEFAULT) if "use_scheduler" in df.columns else USE_SCHEDULER_DEFAULT
    loss_mode = _as_loss_mode(best["loss_mode"], default=LOSS_MODE_DEFAULT) if "loss_mode" in df.columns else LOSS_MODE_DEFAULT

    hp = {
        "hidden_size": int(best["hidden_size"]),
        "n_hidden_layers": int(best["n_hidden_layers"]),
        "activation": str(best["activation"]),
        "dropout": float(best["dropout"]),
        "standardize_cont": bool(best["standardize_cont"]),
        "lr": float(best["lr"]),
        "optimizer": optimizer,                 # "adam" or "adamw"
        "use_scheduler": bool(use_scheduler),   # ReduceLROnPlateau on/off
        "loss_mode": str(loss_mode).lower(),    # "relative" or "absolute"
    }
    return hp, best


# -------------------------
# MODEL: MLP -> (λ, τ, β)
# -------------------------
class ParamNet(nn.Module):
    def __init__(self, in_dim, hidden_size, n_hidden_layers, activation, dropout=0.0):
        super().__init__()
        activation = str(activation).lower()

        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU
        elif activation == "softplus":
            act = lambda: nn.Softplus(beta=1.0, threshold=20.0)
        elif activation in ("swish", "silu"):
            act = nn.SiLU
        elif activation == "leakyrelu":
            act = nn.LeakyReLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "mish":
            act = nn.Mish
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
        self.out = nn.Linear(d, 3)  # (lam, tau, beta)

        with torch.no_grad():
            self.out.bias[:] = torch.tensor([-6.0, 2.0, 1.0], dtype=torch.float32)

    def forward(self, x):
        z = self.out(self.body(x))
        lam  = torch.exp(torch.clamp(z[:, 0], -10.0,  4.0))
        tau  = torch.exp(torch.clamp(z[:, 1], -10.0,  4.0))
        beta = F.softplus(torch.clamp(z[:, 2], -5.0,  2.0))
        return lam, tau, beta


# -------------------------
# ODE for dissolution
# -------------------------
class DissolutionODEFunc(nn.Module):
    """
    df/dt = lam * (t^n/(tau+t^n)) * (1-f)^beta, with n fixed.
    lam, tau, beta: [B]; f: [B]
    """
    def __init__(self, lam, tau, beta, n_fixed=1.0):
        super().__init__()
        self.lam = lam
        self.tau = tau
        self.beta = beta
        self.n_fixed = float(n_fixed)

    def forward(self, t, f):
        f = torch.clamp(f, 0.0, 1.0)
        tt = t.expand_as(self.tau)

        if self.n_fixed == 1.0:
            t_pow = tt
        else:
            t_pow = torch.pow(tt, self.n_fixed)

        g = t_pow / (self.tau + t_pow + 1e-12)
        one_minus = torch.clamp(1.0 - f, min=0.0)
        return self.lam * g * torch.pow(one_minus, self.beta)


def odeint_solve_batch(lam, tau, beta, t_eval, device,
                       method=ODE_METHOD, rtol=ODE_RTOL, atol=ODE_ATOL):
    t = torch.tensor(np.asarray(t_eval, dtype=float), device=device, dtype=torch.float64)
    B = lam.shape[0]
    y0 = torch.zeros(B, device=device, dtype=torch.float64)

    func = DissolutionODEFunc(lam, tau, beta, n_fixed=N_FIXED)
    y = odeint(func, y0, t, method=method, rtol=rtol, atol=atol)  # [T, B]
    y = y.transpose(0, 1)  # [B, T]
    return torch.clamp(y, 0.0, 1.0)


# -------------------------
# Loss/metrics
# -------------------------
def abs_mse(pred, target):
    """
    Absolute MSE pooled over (batch,time).
    """
    target = target.to(dtype=pred.dtype)
    return ((pred - target) ** 2).mean()


def rel_mse(pred, target, rel_eps=REL_EPS):
    """
    Relative MSE pooled over (batch,time):
        rel_err = (pred - y) / max(y, rel_eps)
        mse = mean(rel_err^2)
    """
    target = target.to(dtype=pred.dtype)
    denom = torch.clamp(target, min=float(rel_eps))
    rel_err = (pred - target) / denom
    return (rel_err ** 2).mean()


def choose_mse(loss_mode: str):
    """
    Returns a callable loss(pred, target) -> scalar tensor
    """
    m = str(loss_mode).strip().lower()
    if m == "relative":
        return lambda pred, target: rel_mse(pred, target, rel_eps=REL_EPS)
    if m == "absolute":
        return lambda pred, target: abs_mse(pred, target)
    raise ValueError("loss_mode must be 'relative' or 'absolute'.")


def rel_rmse(pred, target, rel_eps=REL_EPS):
    return torch.sqrt(rel_mse(pred, target, rel_eps=rel_eps))


def abs_rmse(pred, target):
    return torch.sqrt(abs_mse(pred, target))


# -------------------------
# Optimizer factory
# -------------------------

def warm_start_model_if_possible(model, hp, ckpt_path, device):
    if (ckpt_path is None) or (not os.path.exists(ckpt_path)):
        print(f"[warmstart] No checkpoint found at {ckpt_path}; training from scratch.", flush=True)
        return False

    # SECURITY NOTE: only torch.load checkpoints you trust (pickle-based).
    ckpt = torch.load(ckpt_path, map_location=device)

    # Optional sanity checks
    if "in_dim" in ckpt and int(ckpt["in_dim"]) != next(model.parameters()).shape[1]:
        print(f"[warmstart] Warning: ckpt in_dim={ckpt['in_dim']} vs current in_dim={next(model.parameters()).shape[1]}", flush=True)

    if "hp" in ckpt:
        old_hp = ckpt["hp"]
        keys_to_check = ["hidden_size", "n_hidden_layers", "activation", "dropout"]
        mismatch = {k: (old_hp.get(k), hp.get(k)) for k in keys_to_check if old_hp.get(k) != hp.get(k)}
        if mismatch:
            print(f"[warmstart] Warning: HP mismatch vs checkpoint: {mismatch}", flush=True)

    model.load_state_dict(ckpt["model_state_dict"], strict=bool(WARM_START_STRICT))
    print(f"[warmstart] Loaded model weights from {ckpt_path}", flush=True)
    return True


def make_optimizer(params, optimizer_name: str, lr: float, weight_decay: float):
    name = str(optimizer_name).strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Use 'adam' or 'adamw'.")


# -------------------------
# Train on full training set (no CV here)
# -------------------------
def train_full(model, Xtr, Ytr, t_eval, lr, optimizer_name, use_scheduler, loss_mode, verbose=True):
    opt = make_optimizer(model.parameters(), optimizer_name=optimizer_name, lr=lr, weight_decay=WEIGHT_DECAY)

    loss_fn = choose_mse(loss_mode)

    scheduler = None
    if bool(use_scheduler):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=SCHED_FACTOR,
            patience=SCHED_PATIENCE,
            threshold=1e-6,
            threshold_mode="rel",
            cooldown=SCHED_COOLDOWN,
            min_lr=SCHED_MIN_LR,
            verbose=False,
        )

    n_train = Xtr.shape[0]
    t0 = time.time()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)

        epoch_loss = 0.0
        n_batches = 0

        for k in range(0, n_train, BATCH_SIZE):
            idx = perm[k:k+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            lam, tau, beta = model(xb)

            try:
                pred = odeint_solve_batch(lam, tau, beta, t_eval, DEVICE)
                loss = loss_fn(pred, yb)
            except Exception:
                # keep graph connected so backward works
                loss = torch.tensor(1e3, device=DEVICE, dtype=torch.float32) + 0.0 * (
                    lam.mean() + tau.mean() + beta.mean()
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_loss /= max(1, n_batches)

        if scheduler is not None:
            scheduler.step(epoch_loss)

        cur_lr = float(opt.param_groups[0]["lr"])

        if epoch_loss < best_loss - 1e-12:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch == 1 or epoch % LOG_EVERY == 0):
            tag = "rel_mse" if str(loss_mode).lower() == "relative" else "abs_mse"
            print(
                f"epoch={epoch:04d} train_{tag}={epoch_loss:.6f} best_{tag}={best_loss:.6f} "
                f"lr={cur_lr:.3g} opt={optimizer_name} sched={int(bool(use_scheduler))} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# -------------------------
# Evaluate a dataset
# -------------------------
@torch.no_grad()
def predict_dataset(model, X, t_eval):
    model.eval()
    lam, tau, beta = model(X)
    pred = odeint_solve_batch(lam, tau, beta, t_eval, DEVICE)
    return pred


# -------------------------
# Plots
# -------------------------
def make_parity_plot(y_true, y_pred, out_png, text_note=None):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=14, alpha=0.6)
    ax.plot([0, 1], [0, 1], linewidth=1.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$f$ (Test)")
    ax.set_ylabel(r"$f$ (Predicted)")

    if text_note is not None:
        ax.text(
            0.03, 0.97, text_note,
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=4.0),
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_dynamics_plot(test_batch_ids, t_eval, y_true, model, X_test, out_png, curves_per_subplot=5):
    t_dense = np.linspace(float(np.min(t_eval)), float(np.max(t_eval)), 250).astype(np.float32)

    n = len(test_batch_ids)
    n_panels = 4

    n_per = int(curves_per_subplot)
    if n > n_panels * n_per:
        n_per = int(math.ceil(n / n_panels))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True)
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
            xj = X_test[j:j+1]  # [1, D]

            with torch.no_grad():
                lam, tau, beta = model(xj)
                pred_dense = odeint_solve_batch(lam, tau, beta, t_dense, DEVICE).cpu().numpy().reshape(-1)

            ax.plot(t_dense, pred_dense, linewidth=2.0, label=short_bid)
            ax.scatter(t_eval, y_true[j], s=16, alpha=0.85, label="_nolegend_")

        ax.set_ylabel(r"Release Fraction ($f$)")
        ax.legend(ncol=2, fontsize=8, frameon=False)

    for ax in axes[2:]:
        if ax.get_visible():
            ax.set_xlabel(r"Time (min)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


# -------------------------
# MAIN
# -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading dataset...", flush=True)
    X_raw, Y, labels, batch_ids, t_eval = load_aggregated(DATA_CSV)
    print(f"  formulations={len(batch_ids)} | Xdim={X_raw.shape[1]} | T={Y.shape[1]}", flush=True)

    print("Loading best hyperparameters from trials CSV...", flush=True)
    hp, best_row = load_best_config(TRIALS_CSV)
    print("Best config:", hp, flush=True)
    print("Best row summary:", best_row.to_dict(), flush=True)

    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)}", flush=True)

    mu, sd = None, None
    if bool(hp["standardize_cont"]):
        print("Applying standardization using TRAIN statistics only...", flush=True)
        X_all, mu, sd = standardize_cont_global(X_raw, cont_dim=len(CONT_COLS), train_idx=train_idx)
    else:
        print("Standardization disabled for best config.", flush=True)
        X_all = X_raw

    Xtr = torch.tensor(X_all[train_idx], device=DEVICE)
    Ytr = torch.tensor(Y[train_idx], device=DEVICE)
    Xte = torch.tensor(X_all[test_idx], device=DEVICE)
    Yte = torch.tensor(Y[test_idx], device=DEVICE)

    model = ParamNet(
        in_dim=Xtr.shape[1],
        hidden_size=hp["hidden_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        activation=hp["activation"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    if WARM_START:
        warm_start_model_if_possible(model, hp, WARM_START_CKPT, DEVICE)

    print(
        "Training best model on FULL training set...\n"
        f"  loss_mode={hp['loss_mode']}  optimizer={hp['optimizer']}  use_scheduler={int(bool(hp['use_scheduler']))}  REL_EPS={REL_EPS:g}",
        flush=True
    )

    train_full(
        model, Xtr, Ytr, t_eval,
        lr=hp["lr"],
        optimizer_name=hp["optimizer"],
        use_scheduler=hp["use_scheduler"],
        loss_mode=hp["loss_mode"],
        verbose=True
    )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "hp": hp,
        "in_dim": int(Xtr.shape[1]),
        "cont_cols": CONT_COLS,
        "onehot_cols": ONEHOT_COLS,
        "standardize_cont": bool(hp["standardize_cont"]),
        "mu": (mu.astype(np.float32) if mu is not None else None),
        "sd": (sd.astype(np.float32) if sd is not None else None),
        "t_eval": np.asarray(t_eval, dtype=np.float32),
        "N_FIXED": float(N_FIXED),
        "ODE_METHOD": ODE_METHOD,
        "ODE_RTOL": float(ODE_RTOL),
        "ODE_ATOL": float(ODE_ATOL),
        "seed": int(SEED),
        "REL_EPS": float(REL_EPS),
        "optimizer_used": str(hp["optimizer"]),
        "use_scheduler": bool(hp["use_scheduler"]),
        "loss_mode": str(hp["loss_mode"]),
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": float(SCHED_FACTOR),
            "patience": int(SCHED_PATIENCE),
            "min_lr": float(SCHED_MIN_LR),
            "cooldown": int(SCHED_COOLDOWN),
        } if bool(hp["use_scheduler"]) else None,
    }
    torch.save(ckpt, MODEL_CKPT)
    print(f"Saved trained model checkpoint -> {MODEL_CKPT}", flush=True)

    print("Predicting on test set...", flush=True)
    with torch.no_grad():
        Ypred = predict_dataset(model, Xte, t_eval)

    # Always report both (helps interpret even if you trained on one)
    test_rel_rmse = float(rel_rmse(Ypred, Yte, rel_eps=REL_EPS).item())
    test_abs_rmse = float(abs_rmse(Ypred, Yte).item())
    print(f"Test REL-RMSE (pooled points) = {test_rel_rmse:.6f}   (REL_EPS={REL_EPS:g})", flush=True)
    print(f"Test ABS-RMSE (pooled points) = {test_abs_rmse:.6f}", flush=True)

    print(f"Saving parity plot -> {PARITY_FIG}", flush=True)
    note = rf"trained={hp['loss_mode']}\quad REL-RMSE={test_rel_rmse:.4f}\quad ABS-RMSE={test_abs_rmse:.4f}"
    make_parity_plot(Yte.cpu().numpy(), Ypred.cpu().numpy(), PARITY_FIG, text_note=note)

    print(f"Saving dynamics plot -> {DYNAMICS_FIG}", flush=True)
    test_batch_ids = batch_ids[test_idx]
    make_dynamics_plot(test_batch_ids, t_eval, Yte.cpu().numpy(), model, Xte, DYNAMICS_FIG, curves_per_subplot=5)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
