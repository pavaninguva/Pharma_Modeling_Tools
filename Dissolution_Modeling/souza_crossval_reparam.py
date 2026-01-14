#!/usr/bin/env python3
"""
Neural ODE parameter learning for dissolution curves using torchdiffeq (Dopri5).

This version adds `standardize_cont` as a grid-search hyperparameter (True/False),
and includes it as an axis in the parallel-coordinates plot.

Key points:
- Fix n = 1 globally (NOT learned).
- NN predicts only (lambda, tau, beta).
- ODE solution computed with torchdiffeq.odeint using method="dopri5".
- Standardization (if enabled) is applied *within each CV fold* using ONLY that
  fold's training indices (avoids leakage into validation).

Requires:
    pip install torchdiffeq
"""

import os, re, math, time, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint


# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "./Souza2025_TableS1_Final.csv"

SEED = 42
DEVICE = "cpu"  # "cuda" if available

# leakage-safe outer split
TEST_FRAC = 0.20
K_FOLDS = 5

# ODE settings (torchdiffeq)
ODE_METHOD = "dopri5"
ODE_RTOL = 1e-6
ODE_ATOL = 1e-8

# Fixed exponent
N_FIXED = 1.0

# training
MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 10  # epoch logging frequency

# GRID SEARCH: hyperparameter ranges
HIDDEN_SIZES = list(range(11, 16))
N_HIDDEN_LAYERS = [3, 4, 5]
ACTIVATIONS = ["relu", "tanh", "swish", "softplus", "leakyrelu", "gelu"]
LEARNING_RATES = [1e-2, 2e-2, 3e-2]
DROPOUTS = [0.0, 0.1]
STANDARDIZE_OPTIONS = [True]  

# features already in cleaned CSV
CONT_COLS = ["PEO_N750_pct", "PEO_1105_pct", "PEO_N60K_pct", "PEO_303_pct", "Diluent_pct"]
ONEHOT_COLS = ["Diluent_G721", "Diluent_SMCC", "Diluent_MD_IT12"]

# keys
BATCH_COL = "BatchID"
CAT_COL   = "Diluent_type"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# outputs
TRIALS_CSV = "souza_clean_gridsearch_reparam_trials.csv"
PC_PNG     = "souza_clean_gridsearch_reparam_parallel_coords.png"


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
# SPLIT: 80/20 formulation-level stratified by CAT_COL
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
# FOLDS: rare-class-safe stratified folds
# -------------------------
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


# -------------------------
# Standardization (continuous only) - fold-safe
# -------------------------
def standardize_cont_fold(X, cont_dim, train_idx):
    """
    Returns (Xs, mu, sd) where mu/sd are computed using ONLY train_idx.
    """
    Xs = X.copy()
    mu = Xs[train_idx, :cont_dim].mean(axis=0, keepdims=True)
    sd = Xs[train_idx, :cont_dim].std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs[:, :cont_dim] = (Xs[:, :cont_dim] - mu) / sd
    return Xs, mu, sd


# -------------------------
# MODEL: MLP -> (λ, τ, β) with n fixed
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
            act = nn.SiLU  # Swish
        elif activation == "leakyrelu":
            act = nn.LeakyReLU
        elif activation == "gelu":
            act = nn.GELU
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
        lam  = torch.exp(torch.clamp(z[:, 0], -10.0,  4.0))  # >0
        tau  = torch.exp(torch.clamp(z[:, 1], -10.0,  4.0))  # >0
        beta = F.softplus(torch.clamp(z[:, 2], -5.0,  2.0))   # >0
        return lam, tau, beta


# -------------------------
# ODE (torchdiffeq) with n fixed to N_FIXED
# -------------------------
class DissolutionODEFunc(nn.Module):
    """
    Batched RHS for df/dt = lam * (t^n/(tau+t^n)) * (1-f)^beta, with n fixed.
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
    t = torch.tensor(np.asarray(t_eval, dtype=float), device=device, dtype=torch.float32)
    B = lam.shape[0]
    y0 = torch.zeros(B, device=device, dtype=torch.float32)

    func = DissolutionODEFunc(lam, tau, beta, n_fixed=N_FIXED)
    y = odeint(func, y0, t, method=method, rtol=rtol, atol=atol)  # [T, B]
    y = y.transpose(0, 1)  # [B, T]
    return torch.clamp(y, 0.0, 1.0)


def curve_mse(pred, target):
    return ((pred - target) ** 2).mean(dim=1).mean()


# -------------------------
# TRAIN one fold (with fold-safe standardization)
# -------------------------
def train_one_fold(X_raw_train, Y_train, t_eval, train_idx, val_idx, hp,
                   fold_id, config_id, verbose_epochs=False):
    # fold-safe standardization toggle
    if bool(hp["standardize_cont"]):
        X_fold, _, _ = standardize_cont_fold(
            X_raw_train, cont_dim=len(CONT_COLS), train_idx=train_idx
        )
    else:
        X_fold = X_raw_train

    Xtr = torch.tensor(X_fold[train_idx], device=DEVICE)
    Ytr = torch.tensor(Y_train[train_idx], device=DEVICE)
    Xva = torch.tensor(X_fold[val_idx], device=DEVICE)
    Yva = torch.tensor(Y_train[val_idx], device=DEVICE)

    model = ParamNet(
        in_dim=Xtr.shape[1],
        hidden_size=hp["hidden_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        activation=hp["activation"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()

    n_train = Xtr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)

        last_loss = None
        for k in range(0, n_train, BATCH_SIZE):
            idx = perm[k:k+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            lam, tau, beta = model(xb)

            try:
                pred = odeint_solve_batch(lam, tau, beta, t_eval, DEVICE)
                loss = curve_mse(pred, yb)
            except Exception as e:
                # IMPORTANT: keep a grad connection so backward() doesn't crash
                if verbose_epochs and (epoch == 1 or epoch % LOG_EVERY == 0):
                    print(f"      [warn] odeint failed (epoch {epoch}): {type(e).__name__}: {e}", flush=True)
                loss = torch.tensor(1e3, device=DEVICE, dtype=torch.float32) + 0.0 * (
                    lam.mean() + tau.mean() + beta.mean()
                )

            last_loss = float(loss.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # validate
        model.eval()
        with torch.no_grad():
            lam, tau, beta = model(Xva)
            try:
                pred = odeint_solve_batch(lam, tau, beta, t_eval, DEVICE)
                val = float(curve_mse(pred, Yva).item())
            except Exception:
                val = float("inf")

        if verbose_epochs and (epoch == 1 or epoch % LOG_EVERY == 0):
            print(
                f"      cfg={config_id} fold={fold_id} epoch={epoch:03d} "
                f"train_mse~{last_loss:.6f} val_mse={val:.6f} best_val={best_val:.6f} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True
            )

        if val < best_val - 1e-12:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                if verbose_epochs:
                    print(f"      cfg={config_id} fold={fold_id} early-stop at epoch={epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return float(math.sqrt(best_val))


# -------------------------
# PARALLEL COORDINATES PLOT
# -------------------------
def parallel_coordinates_plot(df_trials, out_png):
    dfp = df_trials.copy().reset_index(drop=True)

    INCLUDE_OBJECTIVE_AXIS = True
    MAX_LINES = None
    N_TICKS_LINEAR = 5

    LINE_ALPHA = 0.65
    LW_THIN = 0.8
    LW_THICK = 2.0
    TOP_N_THICK = 5

    TOP_K_PLOT = 100  # <-- set K here

    if "mean_rmse" not in dfp.columns:
        raise ValueError("df_trials must contain 'mean_rmse' to select top-K.")
    dfp = dfp.sort_values("mean_rmse", ascending=True).head(min(TOP_K_PLOT, len(dfp))).reset_index(drop=True)

    # Backward compatibility: older CSVs might not have this column.
    if "standardize_cont" not in dfp.columns:
        dfp["standardize_cont"] = True

    required_cols = {
        "hidden_size", "n_hidden_layers", "activation", "dropout",
        "standardize_cont", "lr", "mean_rmse", "std_rmse"
    }
    missing = required_cols - set(dfp.columns)
    if missing:
        raise ValueError(f"df_trials missing required columns: {missing}")

    # activation categorical mapping
    act_levels_present = sorted(dfp["activation"].astype(str).unique().tolist())
    if len(act_levels_present) == 1:
        act_to_y = {act_levels_present[0]: 0.5}
    else:
        act_to_y = {a: i / (len(act_levels_present) - 1) for i, a in enumerate(act_levels_present)}
    dfp["activation_y"] = dfp["activation"].astype(str).map(act_to_y).astype(float)

    # standardize categorical mapping (False->0, True->1)
    dfp["standardize_y"] = dfp["standardize_cont"].astype(bool).map({False: 0.0, True: 1.0}).astype(float)

    dims = [
        ("hidden_size",       dfp["hidden_size"].astype(float).to_numpy(), "int_discrete"),
        ("n_hidden_layers",   dfp["n_hidden_layers"].astype(float).to_numpy(), "int_discrete"),
        ("activation",        dfp["activation_y"].astype(float).to_numpy(), "categorical_activation"),
        ("dropout",           dfp["dropout"].astype(float).to_numpy(), "linear"),
        ("standardize_cont",  dfp["standardize_y"].astype(float).to_numpy(), "categorical_standardize"),
        ("learning_rate",     dfp["lr"].astype(float).to_numpy(), "log"),
    ]
    if INCLUDE_OBJECTIVE_AXIS:
        dims.append(("mean_rmse", dfp["mean_rmse"].astype(float).to_numpy(), "linear"))

    D = len(dims)
    x = np.arange(D, dtype=float)

    meta = []
    YN = []
    for name, vals, kind in dims:
        vals = np.asarray(vals, dtype=float)

        if kind == "log":
            v = np.log10(vals)
            vmin, vmax = float(v.min()), float(v.max())
            yn = np.zeros_like(v) if abs(vmax - vmin) < 1e-12 else (v - vmin) / (vmax - vmin)
            meta.append({"name": name, "kind": kind, "vmin": vmin, "vmax": vmax})
            YN.append(yn)

        elif kind.startswith("categorical"):
            yn = np.clip(vals, 0.0, 1.0)
            meta.append({"name": name, "kind": kind, "vmin": 0.0, "vmax": 1.0})
            YN.append(yn)

        else:
            vmin, vmax = float(vals.min()), float(vals.max())
            yn = np.zeros_like(vals) if abs(vmax - vmin) < 1e-12 else (vals - vmin) / (vmax - vmin)
            meta.append({"name": name, "kind": kind, "vmin": vmin, "vmax": vmax})
            YN.append(yn)

    std = dfp["std_rmse"].to_numpy(dtype=float)
    mean = dfp["mean_rmse"].to_numpy(dtype=float)

    cmap = mpl.cm.viridis
    norm_c = mpl.colors.Normalize(vmin=float(std.min()), vmax=float(std.max()))
    colors = cmap(norm_c(std))

    N = len(dfp)
    idx_plot = np.arange(N)
    if MAX_LINES is not None and N > MAX_LINES:
        order = np.argsort(mean)
        pick = np.linspace(0, N - 1, MAX_LINES).round().astype(int)
        idx_plot = np.sort(order[pick])

    idx_plot_set = set(idx_plot.tolist())
    order_best = np.argsort(mean)
    idx_top = np.array([i for i in order_best if i in idx_plot_set][:min(TOP_N_THICK, len(idx_plot))], dtype=int)
    idx_rest = np.array([i for i in idx_plot if i not in set(idx_top.tolist())], dtype=int)

    fig, ax = plt.subplots(figsize=(13.5, 6), dpi=200)
    ax.set_xlim(-0.5, D - 0.5)
    ax.set_ylim(-0.05, 1.05)

    ax.set_yticks([])
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in meta], rotation=20, ha="right")

    for i in idx_rest:
        yline = [YN[j][i] for j in range(D)]
        ax.plot(x, yline, color=colors[i], alpha=LINE_ALPHA, linewidth=LW_THIN)

    for i in idx_top:
        yline = [YN[j][i] for j in range(D)]
        ax.plot(x, yline, color=colors[i], alpha=1.0, linewidth=LW_THICK)

    def _draw_ticks_for_axis(j, m):
        kind, vmin, vmax = m["kind"], m["vmin"], m["vmax"]
        ax.vlines(j, 0.0, 1.0, color="k", linewidth=0.9, alpha=0.45)
        tick_x0, tick_x1 = j - 0.04, j + 0.04

        if kind == "categorical_activation":
            levels = act_levels_present
            if len(levels) == 1:
                ys = [0.5]
                labs = [levels[0]]
            else:
                ys = np.linspace(0.0, 1.0, len(levels))
                labs = levels
            for yv, lab in zip(ys, labs):
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, str(lab), ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

        elif kind == "categorical_standardize":
            ys = [0.0, 1.0]
            labs = ["False", "True"]
            for yv, lab in zip(ys, labs):
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, lab, ha="center", va="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

        elif kind == "int_discrete":
            raw = dfp[m["name"]].astype(int).to_numpy()
            uniq = np.unique(raw)
            ticks = uniq.tolist() if len(uniq) <= 8 else [int(uniq.min()), int(uniq.max())]
            for tval in ticks:
                yv = 0.5 if abs(vmax - vmin) < 1e-12 else (tval - vmin) / (vmax - vmin)
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, f"{tval}", ha="center", va="center", fontsize=8,
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
            ticks = [vmin] if abs(vmax - vmin) < 1e-12 else np.linspace(vmin, vmax, N_TICKS_LINEAR)
            for tval in ticks:
                yv = 0.5 if abs(vmax - vmin) < 1e-12 else (tval - vmin) / (vmax - vmin)
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(j, yv, f"{tval:.3g}", ha="center", va="center", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35))

    for j, m in enumerate(meta):
        _draw_ticks_for_axis(j, m)

    ax.set_title(f"Parallel coordinates (color = std CV RMSE; top {min(TOP_N_THICK, len(idx_plot))} thick by mean_rmse)")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Std CV RMSE")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# MAIN: full grid search with resume
# -------------------------
def main():
    print("Loading and aggregating dataset...", flush=True)
    X_raw, Y, labels, batch_ids, t_eval = load_aggregated(CSV_PATH)
    print(f"  Loaded {len(batch_ids)} formulations; X dim={X_raw.shape[1]}, curve length T={Y.shape[1]}", flush=True)
    print(f"  Time range: {t_eval.min()}..{t_eval.max()} minutes | solver=torchdiffeq({ODE_METHOD}) | n_fixed={N_FIXED:g}", flush=True)

    # outer split
    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)} (formulation-level)", flush=True)
    print("Train counts by category:\n" + pd.Series(labels[train_idx]).value_counts().to_string(), flush=True)
    print("Test counts by category:\n"  + pd.Series(labels[test_idx]).value_counts().to_string(), flush=True)

    # CV is done ONLY on the training set
    X_train_raw = X_raw[train_idx]
    Y_train = Y[train_idx]
    labels_train = labels[train_idx]

    splits = stratified_folds(labels_train, k=K_FOLDS, seed=SEED)
    print(f"Prepared {K_FOLDS}-fold CV splits on training set.", flush=True)

    grid = list(itertools.product(
        HIDDEN_SIZES, N_HIDDEN_LAYERS, ACTIVATIONS, DROPOUTS, STANDARDIZE_OPTIONS, LEARNING_RATES
    ))
    total_cfgs = len(grid)
    print(f"Full grid size = {total_cfgs} configs (each config runs {K_FOLDS} folds).", flush=True)

    # resume support
    done_keys = set()
    if os.path.exists(TRIALS_CSV):
        prev = pd.read_csv(TRIALS_CSV)
        if "standardize_cont" not in prev.columns:
            prev["standardize_cont"] = True
            prev.to_csv(TRIALS_CSV, index=False)
        for _, r in prev.iterrows():
            key = (
                int(r["hidden_size"]),
                int(r["n_hidden_layers"]),
                str(r["activation"]),
                float(r["dropout"]),
                bool(r["standardize_cont"]),
                float(r["lr"]),
            )
            done_keys.add(key)
        print(f"Resuming: found {len(done_keys)} completed configs in {TRIALS_CSV}", flush=True)

    rows = []
    cfg_counter = 0

    for (hs, nl, act, drop, stdz, lr) in grid:
        key = (hs, nl, act, float(drop), bool(stdz), float(lr))
        if key in done_keys:
            continue

        cfg_counter += 1
        config_id = f"{hs}-{nl}-{act}-drop{drop:g}-std{int(bool(stdz))}-lr{lr:g}"
        print(f"\n[CONFIG {cfg_counter}] {config_id}", flush=True)

        hp = {
            "hidden_size": int(hs),
            "n_hidden_layers": int(nl),
            "activation": str(act),
            "dropout": float(drop),
            "standardize_cont": bool(stdz),
            "lr": float(lr),
        }

        fold_rmses = []
        cfg_t0 = time.time()
        for fold_id, (tr, va) in enumerate(splits, 1):
            print(f"  -> fold {fold_id}/{K_FOLDS} start...", flush=True)
            fold_t0 = time.time()
            verbose_epochs = (fold_id == 1)
            r = train_one_fold(X_train_raw, Y_train, t_eval, tr, va, hp, fold_id, config_id, verbose_epochs=verbose_epochs)
            fold_rmses.append(r)
            print(f"  <- fold {fold_id}/{K_FOLDS} done | rmse={r:.5f} | fold_elapsed={time.time()-fold_t0:.1f}s", flush=True)

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse  = float(np.std(fold_rmses))
        print(f"[CONFIG DONE] {config_id} | mean_rmse={mean_rmse:.5f} ± {std_rmse:.5f} | cfg_elapsed={time.time()-cfg_t0:.1f}s", flush=True)

        rows.append({
            "hidden_size": int(hs),
            "n_hidden_layers": int(nl),
            "activation": str(act),
            "dropout": float(drop),
            "standardize_cont": bool(stdz),
            "lr": float(lr),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
        })

        # checkpoint append/update CSV
        out_df = pd.DataFrame(rows)
        if os.path.exists(TRIALS_CSV):
            prev = pd.read_csv(TRIALS_CSV)
            if "standardize_cont" not in prev.columns:
                prev["standardize_cont"] = True
            out_df = pd.concat([prev, out_df], ignore_index=True)

        out_df = out_df.drop_duplicates(
            subset=["hidden_size", "n_hidden_layers", "activation", "dropout", "standardize_cont", "lr"]
        )
        out_df.to_csv(TRIALS_CSV, index=False)
        rows = []
        print(f"Checkpoint saved to {TRIALS_CSV}", flush=True)

    # final plot
    if os.path.exists(TRIALS_CSV):
        df_trials = pd.read_csv(TRIALS_CSV).sort_values("mean_rmse").reset_index(drop=True)
        if "standardize_cont" not in df_trials.columns:
            df_trials["standardize_cont"] = True
        print("\nCreating parallel coordinates plot...", flush=True)
        parallel_coordinates_plot(df_trials, PC_PNG)
        print(f"Saved plot: {PC_PNG}", flush=True)
        print("\nTop 10 configs:", flush=True)
        print(df_trials.head(10).to_string(index=False), flush=True)
    else:
        print("No results CSV found; nothing to plot.", flush=True)


if __name__ == "__main__":
    main()
