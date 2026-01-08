import re
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User settings
# -----------------------------
CSV_PATH = "./Souza2025_TableS1_Final.csv"
OUT_PATH = "./souza2025_dissolution.png"

N_PANELS = 10
N_PER_PANEL = 10

FIGSIZE = (18, 12)
DPI = 300

# -----------------------------
# LaTeX rendering
# -----------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
})

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)

required = {"BatchID", "time_min", "release_frac"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

# Natural sort for formulation IDs like F1, F2, ..., F91
def natural_key(batch_id: str):
    m = re.search(r"(\d+)$", str(batch_id))
    return int(m.group(1)) if m else 10**9

batch_ids = sorted(df["BatchID"].unique(), key=natural_key)

# Ensure time ordering for each formulation
df = df.sort_values(["BatchID", "time_min"])

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 5, figsize=FIGSIZE, sharex=True, sharey=True)
axes = axes.ravel()

# Add y headroom for legends
ymin, ymax = 0.0, 1.0

for i in range(N_PANELS):
    ax = axes[i]
    start = i * N_PER_PANEL
    end = (i + 1) * N_PER_PANEL
    panel_ids = batch_ids[start:end]

    for bid in panel_ids:
        dfi = df[df["BatchID"] == bid]
        m = re.search(r"(\d+)$", str(bid))
        label = rf"$\mathrm{{F{int(m.group(1))}}}$" if m else rf"$\mathrm{{{bid}}}$"
        ax.plot(
            dfi["time_min"], dfi["release_frac"],
            marker="o", linewidth=1.2, markersize=3,
            label=label
        )

    ax.set_ylim(ymin, ymax)
    if panel_ids:
        ax.legend(fontsize=8, ncol=2, frameon=False, loc="best")

# Axis labels (no figure/subfigure titles)
for ax in axes[::5]:
    ax.set_ylabel(r"Fraction released, $f$")
for ax in axes[5:]:
    ax.set_xlabel(r"Time (min)")

fig.tight_layout(pad=1.2)

# Save + show
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
plt.show()