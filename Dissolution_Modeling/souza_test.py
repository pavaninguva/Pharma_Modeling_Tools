#!/usr/bin/env python3
"""
Train-best-and-test script.

What it does:
1) Reads your grid-search results CSV (TRIALS_CSV) and selects the best config
   by lowest mean_rmse (your CV objective).
2) Recreates the same leakage-safe stratified train/test split (same SEED).
3) Trains the ParamNet on the FULL training set using the best hyperparameters.
4) Evaluates on the held-out test set (curve RMSE over all test timepoints).
5) Saves two plots:
   - Test dynamics overlay (true vs predicted curves)
   - Parity plot (all test timepoints, predicted vs true)

Run:
  python train_best_and_test.py

Make sure:
- Souza2025_TableS1_long_fraction_norm_onehot.csv exists at CSV_PATH
- souza_clean_gridsearch_trials.csv exists at TRIALS_CSV
"""

import os, re, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# CONFIG (match your HPO script)
# -------------------------
CSV_PATH = "./Souza2025_TableS1_Final_Pruned.csv"
TRIALS_CSV = "souza_clean_gridsearch_trials.csv"

SEED = 42
DEVICE = "cpu"

TEST_FRAC = 0.20
DT = 1.0
USE_FAST_SEPARABLE_SOLVER = False  # keep consistent with HPO runs

MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
LOG_EVERY = 10

STANDARDIZE_CONT = False

# features already in cleaned CSV
CONT_COLS = ["PEO_N750_pct", "PEO_1105_pct", "PEO_N60K_pct", "PEO_303_pct", "Diluent_pct"]
ONEHOT_COLS = ["Diluent_G721", "Diluent_SMCC", "Diluent_MD_IT12"]

# keys
BATCH_COL = "BatchID"
CAT_COL   = "Diluent_type"
TIME_COL  = "time_min"
Y_COL     = "release_frac"

# outputs
BEST_MODEL_PT = "souza_best_model.pt"
TEST_CURVES_PNG = "souza_best_test_curves.png"
TEST_PARITY_PNG = "souza_best_test_parity.png"


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
# SOLVERS (same as your HPO script)
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


def predict_curves(model: nn.Module, X_np: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_np, device=DEVICE)
        lam, tau, n, beta = model(X)
        if USE_FAST_SEPARABLE_SOLVER:
            pred = fast_separable_solve(lam, tau, n, beta, t_eval, DT, DEVICE)
        else:
            pred = rk4_solve_batch(lam, tau, n, beta, t_eval, DT, DEVICE)
        return pred.detach().cpu().numpy()


# -------------------------
# Train on full training set (no validation available -> early stop on training loss)
# -------------------------
def train_full_training_set(X_train: np.ndarray, Y_train: np.ndarray, t_eval: np.ndarray, hp: dict):
    model = ParamNet(
        in_dim=X_train.shape[1],
        hidden_size=int(hp["hidden_size"]),
        n_hidden_layers=int(hp["n_hidden_layers"]),
        activation=str(hp["activation"]),
        dropout=float(hp.get("dropout", 0.0)),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=WEIGHT_DECAY)

    Xtr = torch.tensor(X_train, device=DEVICE)
    Ytr = torch.tensor(Y_train, device=DEVICE)

    best_loss = float("inf")
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

        # early stop on training loss (best snapshot)
        if last_loss is not None and last_loss < best_loss - 1e-12:
            best_loss = last_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(f"epoch={epoch:03d} train_mse~{last_loss:.6f} best_train_mse={best_loss:.6f} elapsed={time.time()-t0:.1f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# -------------------------
# Plots
# -------------------------
def plot_test_curves(t_eval, Y_true, Y_pred, batch_ids, out_png, max_per_axes=5):
    """
    Overlay plot: predicted (lines) and true (markers), grouped into subplots
    with at most `max_per_axes` curves per subplot.

    Legend entries are ONLY for true data, labeled by BatchID.
    """

    t_eval = np.asarray(t_eval, dtype=float)
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)
    batch_ids = [str(b) for b in batch_ids]

    n = Y_true.shape[0]
    if n == 0:
        raise ValueError("No test curves to plot.")

    n_panels = int(math.ceil(n / float(max_per_axes)))

    # Choose a near-square layout automatically
    ncols = int(math.ceil(math.sqrt(n_panels)))
    nrows = int(math.ceil(n_panels / float(ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), dpi=200, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap("tab20")

    for p in range(n_panels):
        ax = axes[p]
        i0 = p * max_per_axes
        i1 = min((p + 1) * max_per_axes, n)

        for i in range(i0, i1):
            c = cmap(i % 20)

            # Predicted curve: line (no legend entry)
            ax.plot(t_eval, Y_pred[i], color=c, linewidth=1.8, alpha=0.95, label="_nolegend_")

            # True curve: markers (legend entry uses BatchID)
            ax.plot(
                t_eval, Y_true[i],
                linestyle="None", marker="o", markersize=3.0,
                color=c, alpha=0.75,
                label=f"{batch_ids[i]}"
            )

        ax.set_title(f"Test curves {i0+1}–{i1} (of {n})")
        ax.set_xlim(float(t_eval.min()), float(t_eval.max()))
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("release fraction")
        ax.legend(frameon=True, ncol=1, fontsize=8)

    # Hide any unused axes
    for p in range(n_panels, len(axes)):
        axes[p].axis("off")

    fig.suptitle("Test set dynamics: predicted (lines) with true data (markers, labeled by BatchID)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_parity(Y_true, Y_pred, out_png):
    """
    Parity plot using all timepoints in the test set.
    """
    yt = Y_true.reshape(-1)
    yp = Y_pred.reshape(-1)

    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    # y in [0,1]
    lo, hi = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(yt, yp, s=10, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], linewidth=1.5)

    ax.set_xlabel("true release fraction")
    ax.set_ylabel("predicted release fraction")
    ax.set_title(f"Test parity plot (RMSE={rmse:.4f})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi = 300)
    plt.close(fig)

    return rmse


# -------------------------
# MAIN
# -------------------------
def main():
    if not os.path.exists(TRIALS_CSV):
        raise FileNotFoundError(f"Missing TRIALS_CSV: {TRIALS_CSV}")

    df_trials = pd.read_csv(TRIALS_CSV)

    required = {"hidden_size", "n_hidden_layers", "activation", "lr", "mean_rmse"}
    if not required.issubset(df_trials.columns):
        raise ValueError(f"TRIALS_CSV must contain columns {required}, got {set(df_trials.columns)}")

    # dropout might exist; if not, assume 0.0
    if "dropout" not in df_trials.columns:
        df_trials["dropout"] = 0.0

    # pick best by lowest mean_rmse (your CV objective)
    best = df_trials.sort_values("mean_rmse", ascending=True).iloc[0].to_dict()
    hp = {
        "hidden_size": int(best["hidden_size"]),
        "n_hidden_layers": int(best["n_hidden_layers"]),
        "activation": str(best["activation"]),
        "dropout": float(best.get("dropout", 0.0)),
        "lr": float(best["lr"]),
    }

    print("Best hyperparameters from CV (lowest mean_rmse):")
    print(hp)
    print(f"CV mean_rmse={float(best['mean_rmse']):.6f}  std_rmse={float(best.get('std_rmse', np.nan)):.6f}")

    print("\nLoading and aggregating dataset...", flush=True)
    X_raw, Y, labels, batch_ids, t_eval = load_aggregated(CSV_PATH)
    print(f"  Loaded {len(batch_ids)} formulations; X dim={X_raw.shape[1]}, curve length T={Y.shape[1]}", flush=True)

    # reproduce the same train/test split used during HPO
    train_idx, test_idx = stratified_train_test_split(labels, TEST_FRAC, SEED)
    print(f"Split: train={len(train_idx)} test={len(test_idx)} (formulation-level)", flush=True)

    # standardize if you did so in HPO
    if STANDARDIZE_CONT:
        X = standardize_cont(X_raw, cont_dim=len(CONT_COLS), train_idx=train_idx)
    else:
        X = X_raw

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    batch_ids_test = batch_ids[test_idx]

    # train model on full training set
    print("\nTraining best model on full training set...", flush=True)
    model = train_full_training_set(X_train, Y_train, t_eval, hp)
    torch.save({"hp": hp, "state_dict": model.state_dict()}, BEST_MODEL_PT)
    print(f"Saved model checkpoint: {BEST_MODEL_PT}", flush=True)

    # evaluate
    print("\nEvaluating on train/test sets...", flush=True)
    Ypred_train = predict_curves(model, X_train, t_eval)
    Ypred_test  = predict_curves(model, X_test,  t_eval)

    train_rmse = float(np.sqrt(np.mean((Ypred_train - Y_train) ** 2)))
    test_rmse  = float(np.sqrt(np.mean((Ypred_test  - Y_test)  ** 2)))

    print(f"Train RMSE (all train timepoints): {train_rmse:.6f}")
    print(f"Test  RMSE (all test  timepoints): {test_rmse:.6f}")

    # plots
    print("\nSaving plots...", flush=True)
    plot_test_curves(t_eval, Y_test, Ypred_test, batch_ids_test, TEST_CURVES_PNG, max_per_axes=5)
    parity_rmse = plot_parity(Y_test, Ypred_test, TEST_PARITY_PNG)

    print(f"Saved: {TEST_CURVES_PNG}")
    print(f"Saved: {TEST_PARITY_PNG}")
    print(f"Parity RMSE (check): {parity_rmse:.6f}")


if __name__ == "__main__":
    main()