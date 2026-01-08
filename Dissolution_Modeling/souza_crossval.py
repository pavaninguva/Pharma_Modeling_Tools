#!/usr/bin/env python3
import os, re, math, time, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "./Souza2025_TableS1_Final_Pruned.csv"

SEED = 42
DEVICE = "cpu"

# leakage-safe split
TEST_FRAC = 0.20
K_FOLDS = 5

# ODE solver
DT = 1.0
USE_FAST_SEPARABLE_SOLVER = False  # False = RK4 unroll, True = faster separable solver

# training
MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 10  # epoch logging frequency

# GRID SEARCH: hyperparameter ranges
HIDDEN_SIZES = list(range(10, 13))       # 5..20 inclusive
N_HIDDEN_LAYERS = [1, 2, 3]
ACTIVATIONS = ["relu","tanh"]
LEARNING_RATES = [1e-2]  # change if desired
DROPOUTS = [0.0]

# features already in cleaned CSV
CONT_COLS = ["PEO_N750_pct", "PEO_1105_pct", "PEO_N60K_pct", "PEO_303_pct", "Diluent_pct"]
ONEHOT_COLS = ["Diluent_G721", "Diluent_SMCC", "Diluent_MD_IT12"]

# keys
BATCH_COL = "BatchID"
CAT_COL   = "Diluent_type"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# Optional: still recommended for MLP training stability
STANDARDIZE_CONT = True

# outputs
TRIALS_CSV = "souza_clean_gridsearch_trials.csv"
PC_PNG     = "souza_clean_gridsearch_parallel_coords.png"


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
        raise ValueError(f"Missing columns in cleaned CSV: {missing}")

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
    """
    Create k (train_idx, val_idx) splits that are NOT disjoint, but satisfy:
      - each class appears at least once in each validation split (if possible)
      - each class appears at least once in training split when possible (count >= 2)

    Validation split size is chosen to mimic k-fold CV: ~N/k.
    This allows repeating rare-class samples across different validation splits.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    N = len(labels)
    idx_all = np.arange(N)

    classes = sorted(np.unique(labels))
    idx_by = {c: idx_all[labels == c].copy().astype(int) for c in classes}
    counts = {c: len(idx_by[c]) for c in classes}

    # Must have at least one sample per class in validation
    n_val_target = max(len(classes), int(round(N / float(k))))

    # Track how often each index is used in validation, to spread usage
    val_use = np.zeros(N, dtype=int)

    def ok_split(tr, va):
        for c in classes:
            # must appear in val if it exists
            if counts[c] >= 1 and np.sum(labels[va] == c) < 1:
                return False
            # must appear in train if possible (count >= 2)
            if counts[c] >= 2 and np.sum(labels[tr] == c) < 1:
                return False
        return True

    splits = []
    tries = 0

    while len(splits) < k and tries < max_tries:
        tries += 1

        val = []
        remaining = set(idx_all.tolist())

        # Force 1 sample from each class into validation
        # Choose the least-used sample for that class to avoid always picking same rare point
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

        # Fill remaining validation points to reach target size
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
# OPTIONAL standardization (continuous only)
# -------------------------
def standardize_cont(X, cont_dim, train_idx):
    Xs = X.copy()
    mu = Xs[train_idx, :cont_dim].mean(axis=0, keepdims=True)
    sd = Xs[train_idx, :cont_dim].std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs[:, :cont_dim] = (Xs[:, :cont_dim] - mu) / sd
    return Xs


# -------------------------
# MODEL: MLP -> (λ, τ, n, β)
# -------------------------
class ParamNet(nn.Module):
    def __init__(self, in_dim, hidden_size, n_hidden_layers, activation, dropout=0.0):
        super().__init__()
        act = nn.Tanh if activation == "tanh" else nn.ReLU

        layers = []
        d = in_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(d, hidden_size), act()]
            if dropout and float(dropout) > 0.0:
                layers += [nn.Dropout(p=float(dropout))]
            d = hidden_size

        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 4)

        # bias init helps keep parameters in a sane regime at epoch 1
        with torch.no_grad():
            self.out.bias[:] = torch.tensor([-6.0, 10.0, 2.0, 1.0], dtype=torch.float32)

    def forward(self, x):
        z = self.out(self.body(x))
        lam  = torch.exp(torch.clamp(z[:, 0], -12.0, 5.0))   # >0
        tau  = torch.exp(torch.clamp(z[:, 1], -12.0, 25.0))  # >0
        n    = F.softplus(z[:, 2]) + 1e-8                    # >0
        beta = F.softplus(z[:, 3])                           # >=0
        return lam, tau, n, beta


# -------------------------
# SOLVERS
# -------------------------
def rhs(t, f, lam, tau, n, beta):
    f = torch.clamp(f, 0.0, 1.0)
    t_pow = torch.pow(t, n)
    g = t_pow / (tau + t_pow + 1e-12)
    return lam * torch.pow(1.0 - f, beta) * g

def rk4_solve_batch(lam, tau, n, beta, t_eval, dt, device):
    t_eval = np.asarray(t_eval, dtype=float)
    eval_steps = np.rint(t_eval / dt).astype(int)
    max_step = int(eval_steps.max())
    step_to_col = {int(s): i for i, s in enumerate(eval_steps)}
    T = len(t_eval)

    B = lam.shape[0]
    f = torch.zeros(B, device=device, dtype=torch.float32)
    out = torch.zeros(B, T, device=device, dtype=torch.float32)

    for step in range(max_step + 1):
        if step in step_to_col:
            out[:, step_to_col[step]] = f
        if step == max_step:
            break

        t = torch.tensor(step * dt, device=device, dtype=torch.float32)
        k1 = rhs(t,          f,               lam, tau, n, beta)
        k2 = rhs(t + dt/2.0, f + dt*k1/2.0,   lam, tau, n, beta)
        k3 = rhs(t + dt/2.0, f + dt*k2/2.0,   lam, tau, n, beta)
        k4 = rhs(t + dt,     f + dt*k3,       lam, tau, n, beta)
        f = f + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return torch.clamp(out, 0.0, 1.0)

def fast_separable_solve(lam, tau, n, beta, t_eval, dt, device):
    t_eval = np.asarray(t_eval, dtype=float)
    t_max = float(t_eval.max())
    steps = int(round(t_max / dt))

    t_grid = torch.linspace(0.0, steps * dt, steps + 1, device=device, dtype=torch.float32)
    t = t_grid.unsqueeze(0)

    t_pow = torch.pow(t, n.unsqueeze(1))
    g = t_pow / (tau.unsqueeze(1) + t_pow + 1e-12)

    inc = 0.5 * (g[:, :-1] + g[:, 1:]) * dt
    I = torch.cat([torch.zeros(g.shape[0], 1, device=device), torch.cumsum(inc, dim=1)], dim=1)

    eval_steps = np.clip(np.rint(t_eval / dt).astype(int), 0, steps)
    I_eval = I[:, torch.tensor(eval_steps, device=device)]

    beta_e = beta.unsqueeze(1)
    lam_e  = lam.unsqueeze(1)
    close_to_1 = torch.abs(beta_e - 1.0) < 1e-4

    u1 = torch.exp(-lam_e * I_eval)
    one_minus_beta = 1.0 - beta_e
    inside = torch.clamp(1.0 - one_minus_beta * lam_e * I_eval, min=1e-12)
    u2 = torch.pow(inside, 1.0 / one_minus_beta)

    u = torch.where(close_to_1, u1, u2)
    return torch.clamp(1.0 - u, 0.0, 1.0)

def curve_mse(pred, target):
    return ((pred - target) ** 2).mean(dim=1).mean()

def curve_rmse(pred, target):
    return torch.sqrt(((pred - target) ** 2).mean())

# -------------------------
# TRAIN one fold (with logging)
# -------------------------
def train_one_fold(X, Y, t_eval, train_idx, val_idx, hp, fold_id, config_id, verbose_epochs=False):
    Xtr = torch.tensor(X[train_idx], device=DEVICE)
    Ytr = torch.tensor(Y[train_idx], device=DEVICE)
    Xva = torch.tensor(X[val_idx], device=DEVICE)
    Yva = torch.tensor(Y[val_idx], device=DEVICE)

    model = ParamNet(X.shape[1], hp["hidden_size"], hp["n_hidden_layers"], hp["activation"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()

    n_train = Xtr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)

        # train epoch
        last_loss = None
        for k in range(0, n_train, BATCH_SIZE):
            idx = perm[k:k+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            lam, tau, n, beta = model(xb)
            if USE_FAST_SEPARABLE_SOLVER:
                pred = fast_separable_solve(lam, tau, n, beta, t_eval, DT, DEVICE)
            else:
                pred = rk4_solve_batch(lam, tau, n, beta, t_eval, DT, DEVICE)

            loss = curve_mse(pred, yb)
            last_loss = float(loss.item())

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # validate
        model.eval()
        with torch.no_grad():
            lam, tau, n, beta = model(Xva)
            if USE_FAST_SEPARABLE_SOLVER:
                pred = fast_separable_solve(lam, tau, n, beta, t_eval, DT, DEVICE)
            else:
                pred = rk4_solve_batch(lam, tau, n, beta, t_eval, DT, DEVICE)
            val = float(curve_mse(pred, Yva).item())

        if verbose_epochs and (epoch == 1 or epoch % LOG_EVERY == 0):
            print(
                f"      cfg={config_id} fold={fold_id} epoch={epoch:03d} "
                f"train_mse~{last_loss:.6f} val_mse={val:.6f} best_val={best_val:.6f} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True
            )

        # early stopping
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

    return math.sqrt(best_val)


# -------------------------
# PARALLEL COORDINATES PLOT
# -------------------------
def parallel_coordinates_plot(df_trials, out_png):
    """
    Matplotlib parallel-coordinates plot:
      - Per-axis scales/ticks
      - color = std CV RMSE
      - constant alpha
      - Top-N (by lowest mean_rmse) drawn thicker on top of the rest
    """
    dfp = df_trials.copy().reset_index(drop=True)

    # -------------------------
    # USER-TWEAKABLE SETTINGS
    # -------------------------
    INCLUDE_OBJECTIVE_AXIS = True
    MAX_LINES = None              # None = plot all
    N_TICKS_LINEAR = 5

    LINE_ALPHA = 0.65
    LW_THIN = 0.8
    LW_THICK = 2.0
    TOP_N_THICK = 5              # top N configs (lowest mean_rmse) drawn thick

    required_cols = {"hidden_size", "n_hidden_layers", "activation", "dropout", "lr", "mean_rmse", "std_rmse"}
    missing = required_cols - set(dfp.columns)
    if missing:
        raise ValueError(f"df_trials missing required columns: {missing}")

    # -------------------------
    # Categorical mapping FROM DATA
    # -------------------------
    act_levels_present = sorted(dfp["activation"].astype(str).unique().tolist())
    if len(act_levels_present) == 1:
        act_to_y = {act_levels_present[0]: 0.5}
    else:
        act_to_y = {a: i / (len(act_levels_present) - 1) for i, a in enumerate(act_levels_present)}
    dfp["activation_y"] = dfp["activation"].astype(str).map(act_to_y).astype(float)

    # -------------------------
    # Dimensions to plot
    # -------------------------
    dims = [
    ("hidden_size",      dfp["hidden_size"].astype(float).to_numpy(), "int_discrete"),
    ("n_hidden_layers",  dfp["n_hidden_layers"].astype(float).to_numpy(), "int_discrete"),
    ("activation",       dfp["activation_y"].astype(float).to_numpy(), "categorical"),
    ("dropout",          dfp["dropout"].astype(float).to_numpy(), "linear"),
    ("learning_rate",    dfp["lr"].astype(float).to_numpy(), "log"),
    ]
    if INCLUDE_OBJECTIVE_AXIS:
        dims.append(("mean_rmse", dfp["mean_rmse"].astype(float).to_numpy(), "linear"))

    D = len(dims)
    x = np.arange(D, dtype=float)

    # -------------------------
    # Normalize for geometry
    # -------------------------
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

        elif kind == "categorical":
            yn = np.clip(vals, 0.0, 1.0)
            meta.append({"name": name, "kind": kind, "vmin": 0.0, "vmax": 1.0})
            YN.append(yn)

        else:
            vmin, vmax = float(vals.min()), float(vals.max())
            yn = np.zeros_like(vals) if abs(vmax - vmin) < 1e-12 else (vals - vmin) / (vmax - vmin)
            meta.append({"name": name, "kind": kind, "vmin": vmin, "vmax": vmax})
            YN.append(yn)

    # -------------------------
    # Color encoding = std_rmse
    # -------------------------
    std = dfp["std_rmse"].to_numpy(dtype=float)
    mean = dfp["mean_rmse"].to_numpy(dtype=float)

    cmap = mpl.cm.viridis
    norm_c = mpl.colors.Normalize(vmin=float(std.min()), vmax=float(std.max()))
    colors = cmap(norm_c(std))

    # Optional subsampling
    N = len(dfp)
    idx_plot = np.arange(N)
    if MAX_LINES is not None and N > MAX_LINES:
        # keep a spread across mean_rmse for interpretability
        order = np.argsort(mean)
        pick = np.linspace(0, N - 1, MAX_LINES).round().astype(int)
        idx_plot = order[pick]
        idx_plot = np.sort(idx_plot)

    # Identify top-N by mean_rmse among those being plotted
    idx_plot_set = set(idx_plot.tolist())
    order_best = np.argsort(mean)  # global best-first
    idx_top = [i for i in order_best if i in idx_plot_set][:min(TOP_N_THICK, len(idx_plot))]
    idx_top = np.array(idx_top, dtype=int)

    # Remaining indices
    idx_rest = np.array([i for i in idx_plot if i not in set(idx_top.tolist())], dtype=int)

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
    ax.set_xlim(-0.5, D - 0.5)
    ax.set_ylim(-0.05, 1.05)

    ax.set_yticks([])
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in meta], rotation=20, ha="right")

    # Draw thin lines first (background)
    for i in idx_rest:
        y = [YN[j][i] for j in range(D)]
        ax.plot(x, y, color=colors[i], alpha=LINE_ALPHA, linewidth=LW_THIN)

    # Draw top-N thick lines on top
    for i in idx_top:
        y = [YN[j][i] for j in range(D)]
        ax.plot(x, y, color=colors[i], alpha=1.0, linewidth=LW_THICK)

    # Per-axis ticks
    def _draw_ticks_for_axis(j, m):
        name, kind, vmin, vmax = m["name"], m["kind"], m["vmin"], m["vmax"]

        ax.vlines(j, 0.0, 1.0, color="k", linewidth=0.9, alpha=0.45)
        tick_x0, tick_x1 = j - 0.04, j + 0.04

        if kind == "categorical":
            levels = act_levels_present
            if len(levels) == 1:
                ys = [0.5]
                labs = [levels[0]]
            else:
                ys = np.linspace(0.0, 1.0, len(levels))
                labs = levels

            for yv, lab in zip(ys, labs):
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(
                    j, yv, str(lab),
                    ha="center", va="center", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35)
                )

        elif kind == "int_discrete":
            raw = dfp[name].astype(int).to_numpy()
            uniq = np.unique(raw)
            ticks = uniq.tolist() if len(uniq) <= 8 else [int(uniq.min()), int(uniq.max())]

            for t in ticks:
                yv = 0.0 if abs(vmax - vmin) < 1e-12 else (t - vmin) / (vmax - vmin)
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(
                    j, yv, f"{t}",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35)
                )

        elif kind == "log":
            pmin, pmax = int(np.floor(vmin)), int(np.ceil(vmax))
            ticks = [10 ** p for p in range(pmin, pmax + 1)]
            for t in ticks:
                lt = np.log10(t)
                if lt < vmin - 1e-12 or lt > vmax + 1e-12:
                    continue
                yv = 0.0 if abs(vmax - vmin) < 1e-12 else (lt - vmin) / (vmax - vmin)
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(
                    j, yv, f"{t:.0e}",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35)
                )

        else:
            if abs(vmax - vmin) < 1e-12:
                ticks = [vmin]
            else:
                ticks = np.linspace(vmin, vmax, N_TICKS_LINEAR)

            for t in ticks:
                yv = 0.5 if abs(vmax - vmin) < 1e-12 else (t - vmin) / (vmax - vmin)
                ax.hlines(yv, tick_x0, tick_x1, color="k", linewidth=0.9, alpha=0.8)
                ax.text(
                    j, yv, f"{t:.3g}",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.35)
                )

    for j, m in enumerate(meta):
        _draw_ticks_for_axis(j, m)

    ax.set_title(f"Parallel coordinates (color = std CV RMSE; top {min(TOP_N_THICK, len(idx_plot))} thick by mean_rmse)")

    # Colorbar for std_rmse
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Std CV RMSE")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# MAIN: full grid search with timely prints + resume
# -------------------------
def main():
    print("Loading and aggregating dataset...", flush=True)
    X_raw, Y, labels, batch_ids, t_eval = load_aggregated(CSV_PATH)
    print(f"  Loaded {len(batch_ids)} formulations; X dim={X_raw.shape[1]}, curve length T={Y.shape[1]}", flush=True)
    print(f"  Time range: {t_eval.min()}..{t_eval.max()} minutes | DT={DT} | solver={'fast' if USE_FAST_SEPARABLE_SOLVER else 'RK4'}", flush=True)

    # split 80/20 stratified by CAT_COL
    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)} (formulation-level)", flush=True)
    print("Train counts by category:\n" + pd.Series(labels[train_idx]).value_counts().to_string(), flush=True)
    print("Test counts by category:\n"  + pd.Series(labels[test_idx]).value_counts().to_string(), flush=True)

    # standardize continuous only (optional)
    if STANDARDIZE_CONT:
        print("Standardizing continuous inputs using train statistics...", flush=True)
        X = standardize_cont(X_raw, cont_dim=len(CONT_COLS), train_idx=train_idx)
    else:
        X = X_raw

    # training set only for CV / HPO
    X_train, Y_train, labels_train = X[train_idx], Y[train_idx], labels[train_idx]

    splits = stratified_folds(labels_train, k=K_FOLDS, seed=SEED)
    print(f"Prepared {K_FOLDS}-fold CV splits on training set.", flush=True)

    # build full grid
    grid = list(itertools.product(HIDDEN_SIZES, N_HIDDEN_LAYERS, ACTIVATIONS, DROPOUTS, LEARNING_RATES))
    total_cfgs = len(grid)
    print(f"Full grid size = {total_cfgs} configs (each config runs {K_FOLDS} folds).", flush=True)

    # resume support
    done_keys = set()
    if os.path.exists(TRIALS_CSV):
        prev = pd.read_csv(TRIALS_CSV)
        # if "dropout" not in prev.columns:
        #     prev["dropout"] = 0.0
        #     prev.to_csv(TRIALS_CSV, index=False)
        for _, r in prev.iterrows():
            key = (int(r["hidden_size"]), int(r["n_hidden_layers"]), str(r["activation"]), float(r["dropout"]), float(r["lr"]))
            done_keys.add(key)
        print(f"Resuming: found {len(done_keys)} completed configs in {TRIALS_CSV}", flush=True)
    

    rows = []
    cfg_counter = 0

    for (hs, nl, act, drop, lr) in grid:
        key = (hs, nl, act, float(drop), float(lr))
        if key in done_keys:
            continue

        cfg_counter += 1
        config_id = f"{hs}-{nl}-{act}-drop{drop:g}-lr{lr:g}"
        print(f"\n[CONFIG {cfg_counter}] {config_id}  ({cfg_counter}/{total_cfgs} remaining incl. skipped)", flush=True)
        hp = {"hidden_size": hs, "n_hidden_layers": nl, "activation": act, "dropout": float(drop), "lr": lr}

        fold_rmses = []
        cfg_t0 = time.time()
        for fold_id, (tr, va) in enumerate(splits, 1):
            print(f"  -> fold {fold_id}/{K_FOLDS} start...", flush=True)
            fold_t0 = time.time()

            # verbose epoch logging only on fold 1 (keeps terminal readable)
            verbose_epochs = (fold_id == 1)
            r = train_one_fold(X_train, Y_train, t_eval, tr, va, hp, fold_id, config_id, verbose_epochs=verbose_epochs)

            fold_rmses.append(r)
            print(f"  <- fold {fold_id}/{K_FOLDS} done | rmse={r:.5f} | fold_elapsed={time.time()-fold_t0:.1f}s", flush=True)

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse  = float(np.std(fold_rmses))
        print(f"[CONFIG DONE] {config_id} | mean_rmse={mean_rmse:.5f} ± {std_rmse:.5f} | cfg_elapsed={time.time()-cfg_t0:.1f}s", flush=True)

        rows.append({
            "hidden_size": hs,
            "n_hidden_layers": nl,
            "activation": act,
            "dropout": float(drop),
            "lr": float(lr),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
        })

        # checkpoint append/update CSV so you never lose progress
        out_df = pd.DataFrame(rows)
        if os.path.exists(TRIALS_CSV):
            prev = pd.read_csv(TRIALS_CSV)
            out_df = pd.concat([prev, out_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["hidden_size", "n_hidden_layers", "activation", "dropout", "lr"])
        out_df.to_csv(TRIALS_CSV, index=False)
        rows = []  # clear buffer after checkpoint

        print(f"Checkpoint saved to {TRIALS_CSV}", flush=True)

    # final plot
    if os.path.exists(TRIALS_CSV):
        df_trials = pd.read_csv(TRIALS_CSV).sort_values("mean_rmse").reset_index(drop=True)
        print("\nCreating parallel coordinates plot...", flush=True)
        parallel_coordinates_plot(df_trials, PC_PNG)
        print(f"Saved plot: {PC_PNG}", flush=True)
        print("\nTop 10 configs:", flush=True)
        print(df_trials.head(10).to_string(index=False), flush=True)
    else:
        print("No results CSV found; nothing to plot.", flush=True)


if __name__ == "__main__":
    import os
    main()