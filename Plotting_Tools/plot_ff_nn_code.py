import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def _layer_display_indices(N, max_show=5, head=2, tail=2):
    if N <= max_show:
        return list(range(N))
    head = min(head, N)
    tail = min(tail, N - head)
    return list(range(head)) + ["dots"] + list(range(N - tail, N))


def _layer_y_positions(display_items, spacing=1.0):
    L = len(display_items)
    ys = np.arange(L) * spacing
    return ys - ys.mean()


def draw_nn_schematic_matplotlib(
    K,
    M,
    output_node_labels=(r"$\lambda$", r"$\tau$", r"$\beta$"),
    n_hidden_layers=6,
    collapse_hidden_layers=True,

    # show 2, vdots, 2 in input/hidden
    max_show=5, head=2, tail=2,

    node_radius=0.24,

    # spacing control
    x_in=0.0,
    dx_io=2.8,          # Input->Hidden1 and HiddenJ->Output spacing (keep)
    dx_hidden=1.3,      # Hidden1->hdots and hdots->HiddenJ spacing (smaller => less whitespace)

    # whitespace around the ellipsis itself (smaller => tighter)
    hdots_gap=0.18,     # absolute x half-gap around the \cdots column

    # connection styling controls
    hidden_stub_frac=0.5,   # hidden stubs = half length of input/output connections (in x)
    conn_color="0.65",
    conn_lw=0.7,

    spacing=0.72,
    title_fontsize=12,
    node_label_fontsize=13,
    fig_size=(11.2, 5.2),
    vdots_fontsize=16,
    hdots_fontsize=18,
    ax=None,
    usetex=False,
):
    """
    Collapsed hidden layout (if enabled and n_hidden_layers > 2):
      Input -> Hidden 1 ->  ...  -> Hidden J -> Output

    FIXES:
    - Reduce whitespace between hidden columns using dx_hidden.
    - Hidden implied connections do NOT converge:
      we preserve angles of a full layer-to-layer connection and then truncate.
    """

    mpl.rcParams.update({
        "text.usetex": bool(usetex),
        "font.family": "serif",
        "mathtext.fontset": "stix",
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")

    J = int(n_hidden_layers)
    n_out = len(output_node_labels)

    # Hidden sizes list
    if isinstance(M, (list, tuple, np.ndarray)):
        hidden_sizes = list(M)
        if len(hidden_sizes) != J:
            raise ValueError("If M is a list/tuple, it must have length n_hidden_layers.")
    else:
        hidden_sizes = [int(M)] * J

    # Decide layout
    collapsed = bool(collapse_hidden_layers) and (J > 2)

    if collapsed:
        # 5 columns: Input, Hidden1, hdots, HiddenJ, Output
        x_in_pos = float(x_in)
        x_h1 = x_in_pos + float(dx_io)
        x_hdots = x_h1 + float(dx_hidden)
        x_hJ = x_hdots + float(dx_hidden)
        x_out_pos = x_hJ + float(dx_io)
        xs = [x_in_pos, x_h1, x_hdots, x_hJ, x_out_pos]
    else:
        # draw all hidden layers equally spaced by dx_io
        x_in_pos = float(x_in)
        x_hidden = [x_in_pos + dx_io * (i + 1) for i in range(J)]
        x_out_pos = x_in_pos + dx_io * (J + 1)
        xs = [x_in_pos] + x_hidden + [x_out_pos]

    # Items + y positions
    in_items = _layer_display_indices(K, max_show=max_show, head=head, tail=tail)
    y_in = _layer_y_positions(in_items, spacing=spacing)

    out_items = list(range(n_out))
    y_out = _layer_y_positions(out_items, spacing=spacing)
    out_labels = {i: str(lbl) for i, lbl in enumerate(output_node_labels)}

    def draw_layer(items, ys, x, labels=None):
        centers = {}
        for item, y in zip(items, ys):
            if item == "dots":
                ax.text(x, y, r"$\vdots$", ha="center", va="center",
                        fontsize=vdots_fontsize, color="k", zorder=6)
            else:
                ax.add_patch(Circle((x, y), radius=node_radius, fill=False,
                                    linewidth=1.4, edgecolor="k"))
                centers[item] = (x, y)
                if labels is not None and item in labels:
                    ax.text(x, y, labels[item], va="center", ha="center",
                            fontsize=node_label_fontsize)
        return centers

    # Draw input
    in_centers = draw_layer(in_items, y_in, x_in_pos)

    # Draw hidden(s)
    hid_centers = []
    hid_xs = []
    hid_ys = []

    if collapsed:
        # Hidden 1
        Mh1 = hidden_sizes[0]
        h1_items = _layer_display_indices(Mh1, max_show=max_show, head=head, tail=tail)
        y_h1 = _layer_y_positions(h1_items, spacing=spacing)
        h1_centers = draw_layer(h1_items, y_h1, x_h1)

        # hdots column
        ax.text(x_hdots, 0.0, r"$\cdots$", ha="center", va="center",
                fontsize=hdots_fontsize, color="k", zorder=6)

        # Hidden J
        MhJ = hidden_sizes[-1]
        hJ_items = _layer_display_indices(MhJ, max_show=max_show, head=head, tail=tail)
        y_hJ = _layer_y_positions(hJ_items, spacing=spacing)
        hJ_centers = draw_layer(hJ_items, y_hJ, x_hJ)

        hid_centers = [h1_centers, hJ_centers]
        hid_xs = [x_h1, x_hJ]
        hid_ys = [y_h1, y_hJ]
    else:
        # Draw all hidden layers
        for li in range(J):
            Mh = hidden_sizes[li]
            items = _layer_display_indices(Mh, max_show=max_show, head=head, tail=tail)
            ys = _layer_y_positions(items, spacing=spacing)
            centers = draw_layer(items, ys, xs[1 + li])
            hid_centers.append(centers)
            hid_xs.append(xs[1 + li])
            hid_ys.append(ys)

    # Draw output
    out_centers = draw_layer(out_items, y_out, x_out_pos, labels=out_labels)

    # ---- Connections helpers ----
    def connect_fully(A, B):
        for (_, (xa, ya)) in A.items():
            for (_, (xb, yb)) in B.items():
                ax.plot([xa + node_radius, xb - node_radius], [ya, yb],
                        linewidth=conn_lw, color=conn_color, zorder=1)

    def connect_truncated_preserve_angles_out(A, y_targets, x_end, full_dx):
        """
        From A nodes at (xA, yA) draw truncated segments that have the SAME angle
        as the full connection to (xA+full_dx, y_target), but stop at x_end.
        This avoids convergence because each (yA, yT) pair yields a different y_end.
        """
        # all nodes in A share same x
        for (_, (xa, ya)) in A.items():
            x0 = xa + node_radius
            frac = (x_end - x0) / max(1e-12, full_dx)
            for yt in y_targets:
                y_end = ya + frac * (yt - ya)
                ax.plot([x0, x_end], [ya, y_end],
                        linewidth=conn_lw, color=conn_color, zorder=1)

    def connect_truncated_preserve_angles_in(y_sources, B, x_start, full_dx):
        """
        Into B nodes at (xB, yB) draw truncated segments that have the SAME angle
        as the full connection from (xB-full_dx, y_source) to (xB, yB), but start at x_start.
        """
        for (_, (xb, yb)) in B.items():
            x1 = xb - node_radius
            frac = (x1 - x_start) / max(1e-12, full_dx)
            for ys in y_sources:
                y_start = yb + frac * (ys - yb)
                ax.plot([x_start, x1], [y_start, yb],
                        linewidth=conn_lw, color=conn_color, zorder=1)

    # ---- Build connections ----
    if collapsed:
        h1, hJ = hid_centers[0], hid_centers[1]

        # input -> hidden1 (full)
        connect_fully(in_centers, h1)

        # hiddenJ -> output (full)
        connect_fully(hJ, out_centers)

        # implied hidden connections near hdots:
        # define "reference full dx" to preserve angles like the IO connections
        # use the actual IO connection dx (minus radii) for consistent slopes
        full_dx_ref = (x_h1 - x_in_pos) - 2.0 * node_radius  # same as Input->Hidden1 connection x-span

        # stub length = hidden_stub_frac of IO connection x-span
        stub_dx = float(hidden_stub_frac) * float(full_dx_ref)

        # shrink whitespace by making hdots_gap small (already), and compute endpoints around hdots
        x_left_limit = x_hdots - float(hdots_gap)
        x_right_limit = x_hdots + float(hdots_gap)

        # where truncated stubs end/start (ensure they don't intrude into the ellipsis whitespace)
        x_end_left = min((x_h1 + node_radius) + stub_dx, x_left_limit)
        x_start_right = max((x_hJ - node_radius) - stub_dx, x_right_limit)

        # Use displayed node y-positions as "virtual targets/sources" to create angled fans without convergence
        y_virtual = np.array([y for (_, (_, y)) in h1.items()], dtype=float)
        y_virtual.sort()

        connect_truncated_preserve_angles_out(h1, y_virtual, x_end_left, full_dx_ref)
        connect_truncated_preserve_angles_in(y_virtual, hJ, x_start_right, full_dx_ref)

    else:
        # full network: connect sequentially
        if J == 0:
            connect_fully(in_centers, out_centers)
        else:
            connect_fully(in_centers, hid_centers[0])
            for i in range(J - 1):
                connect_fully(hid_centers[i], hid_centers[i + 1])
            connect_fully(hid_centers[-1], out_centers)

    # ---- Titles ----
    title_y_offset = spacing * 1.2

    ax.text(x_in_pos, float(np.max(y_in)) + title_y_offset,
            r"Input Layer$\in \mathbb{R}^{K}$", ha="center", va="bottom",
            fontsize=title_fontsize)

    if collapsed:
        ax.text(x_h1, float(np.max(hid_ys[0])) + title_y_offset,
                r"Hidden Layer $1$" + r"$\in \mathbb{R}^{M}$",
                ha="center", va="bottom", fontsize=title_fontsize)

        ax.text(x_hJ, float(np.max(hid_ys[1])) + title_y_offset,
                r"Hidden Layer $J$" + r"$\in \mathbb{R}^{M}$",
                ha="center", va="bottom", fontsize=title_fontsize)
    else:
        if J > 0:
            mid = J // 2
            ax.text(hid_xs[mid], float(np.max(hid_ys[mid])) + title_y_offset,
                    r"Hidden Layers $1 \dots J$" + r"$\in \mathbb{R}^{M}$",
                    ha="center", va="bottom", fontsize=title_fontsize)

    ax.text(x_out_pos, float(np.max(y_out)) + title_y_offset,
            rf"Output Layer$\in \mathbb{{R}}^{{{n_out}}}$",
            ha="center", va="bottom", fontsize=title_fontsize)

    # ---- Limits ----
    all_y = [y_in, y_out]
    if collapsed:
        all_y += hid_ys
    else:
        all_y += hid_ys
    all_y = np.concatenate(all_y)

    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(all_y.min() - 1.1, all_y.max() + 1.8)

    return fig, ax


if __name__ == "__main__":
    fig, ax = draw_nn_schematic_matplotlib(
        K=30,
        M=50,
        n_hidden_layers=6,
        collapse_hidden_layers=True,
        output_node_labels=(r"$\beta$", r"$\tau$", r"$\lambda$"),

        # tighten hidden region (less whitespace)
        dx_io=2.2,
        dx_hidden=1.2,     # smaller => much less whitespace between hidden layers
        hdots_gap=0.20,    # smaller => narrower blank gap around the \cdots

        # keep half-length hidden stubs, but now they don't converge
        hidden_stub_frac=0.5,

        usetex=False,  # set True if you have LaTeX installed
    )
    fig.tight_layout()
    fig.savefig("NN_Schematic.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
