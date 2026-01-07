import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# Matplotlib LaTeX (usetex)
# -----------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 7,
})

# -----------------------------
# Model
# df/dt = λ * (c* - f) * (t^n / (τ + t^n)) * (1 - f)^β
# Assumptions: M_tot/V = 1, c* fixed
# -----------------------------

def make_rhs(lam, tau, n, beta, cstar):
    lam = float(lam); tau = float(tau); n = float(n); beta = float(beta); cstar = float(cstar)

    def rhs(t, y):
        f = float(y[0])
        # enforce physical bounds in the dynamics
        if f <= 0.0:
            f = 0.0
        if f >= 1.0:
            return [0.0]  # once fully released, stay there

        if t <= 0.0:
            g = 0.0
        else:
            t_pow = t**n
            g = t_pow / (tau + t_pow + 1e-15)

        one_minus_f = max(1.0 - f, 0.0)  # safe

        return [lam * (cstar - f) * g * (one_minus_f ** beta)]
    return rhs

def solve_curve(params, t_end=250.0, n_eval=500, cstar=5.0):
    lam, tau, n, beta = params
    rhs = make_rhs(lam, tau, n, beta, cstar)
    t_eval = np.linspace(0.0, t_end, n_eval)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=[0.0],
        t_eval=t_eval,
        method="BDF",
        rtol=1e-7,
        atol=1e-9,
        max_step=1.0,
    )
    return sol.t, sol.y[0]

# -----------------------------
# Plot settings
# -----------------------------
CSTAR = 5.0
T_END = 180.0  # minutes -> 3 hours

zero_order_params = [
    (0.002, 1.0, 1.0, 0.02),
    (0.0016, 1.0, 1.0, 0.02),
    (0.0011, 1.0, 1.0, 0.02),
    (0.0007, 1.0, 1.0, 0.02),
]

first_order_params = [
    (0.012, 1.0, 1.0, 1.0),
    (0.01, 1.0, 1.0, 1.0),
    (0.007, 1.0, 1.0, 1.0),
    (0.004, 1.0, 1.0, 1.0),
]

sigmoid_params = [
    (0.030, 1.3e7, 4.0, 1.0),
    (0.020, 6.5e7, 4.0, 1.0),
    (0.020, 2.07e8, 4.0, 1.0),
    (0.010, 2.08e8, 4.0, 1.0),
]

# -----------------------------
# Make figure
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

families = [
    (r"\textbf{Zero-order}", zero_order_params),
    (r"\textbf{First-order}", first_order_params),
    (r"\textbf{Sigmoid}", sigmoid_params),
]

for ax, (title, plist) in zip(axes, families):
    for (lam, tau, n, beta) in plist:
        t, f = solve_curve((lam, tau, n, beta), t_end=T_END, cstar=CSTAR)
        lbl = rf"$\lambda={lam:g},\ \tau={tau:g},\ n={n:g},\ \beta={beta:g}$"
        ax.plot(t, f, linewidth=1.6, label=lbl)

    ax.set_title(title)
    ax.set_xlabel(r"$t\ \mathrm{(min)}$")
    ax.set_xlim(0, T_END)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, ncol=1)

axes[0].set_ylabel(r"$f\ \mathrm{(release\ fraction)}$")

fig.tight_layout()
plt.savefig("model_rep_plot.png",dpi=300)
plt.show()