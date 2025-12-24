import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

def _layer_display_indices(N, max_show=8, head=3, tail=3):
    if N <= max_show:
        return list(range(N))
    head = min(head, N)
    tail = min(tail, N - head)
    return list(range(head)) + ['dots'] + list(range(N - tail, N))

def _layer_y_positions(display_items, spacing=1.0):
    L = len(display_items)
    ys = np.arange(L) * spacing
    return ys - ys.mean()

def draw_nn_schematic_matplotlib(
    K, M,
    max_show=8, head=3, tail=3,
    node_radius=0.24,
    layer_x=(0.0, 3.2, 6.4),
    spacing=0.72,
    title_fontsize=12,
    node_label_fontsize=13,
    fig_size=(10.5, 5.2),
    vdots_fontsize=16,
    ax=None
):
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "stix",
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    ax.set_aspect('equal')
    ax.axis('off')

    x_in, x_hid, x_out = layer_x

    in_items  = _layer_display_indices(K, max_show=max_show, head=head, tail=tail)
    hid_items = _layer_display_indices(M, max_show=max_show, head=head, tail=tail)
    out_items = [0, 1, 2, 3]

    y_in  = _layer_y_positions(in_items,  spacing=spacing)
    y_hid = _layer_y_positions(hid_items, spacing=spacing)
    y_out = _layer_y_positions(out_items, spacing=spacing)

    out_labels = {0: r"$\lambda$", 1: r"$\tau$", 2: r"$n$", 3: r"$\beta$"}

    def draw_layer(items, ys, x, labels=None):
        centers = {}
        for item, y in zip(items, ys):
            if item == 'dots':
                ax.text(x, y, r"$\vdots$", ha="center", va="center",
                        fontsize=vdots_fontsize, color="k", zorder=6)
            else:
                ax.add_patch(Circle((x, y), radius=node_radius, fill=False,
                                    linewidth=1.4, edgecolor="k"))
                centers[item] = (x, y)
                if labels is not None and item in labels:
                    ax.text(x, y, labels[item], va='center', ha='center',
                            fontsize=node_label_fontsize)
        return centers

    in_centers  = draw_layer(in_items,  y_in,  x_in)
    hid_centers = draw_layer(hid_items, y_hid, x_hid)
    out_centers = draw_layer(out_items, y_out, x_out, labels=out_labels)

    # Grey connections
    conn_color = "0.65"
    for (_, (xi, yi)) in in_centers.items():
        for (_, (xh, yh)) in hid_centers.items():
            ax.plot([xi + node_radius, xh - node_radius], [yi, yh],
                    linewidth=0.7, color=conn_color, zorder=1)

    for (_, (xh, yh)) in hid_centers.items():
        for (_, (xo, yo)) in out_centers.items():
            ax.plot([xh + node_radius, xo - node_radius], [yh, yo],
                    linewidth=0.7, color=conn_color, zorder=1)

    # Titles (no whitespace between 'Layer' and '\in')
    y_in_top, y_hid_top, y_out_top = max(y_in), max(y_hid), max(y_out)
    title_y_offset = spacing * 1.2

    ax.text(x_in,  y_in_top  + title_y_offset, r"Input Layer$\in \mathbb{R}^{K}$",
            ha='center', va='bottom', fontsize=title_fontsize)
    ax.text(x_hid, y_hid_top + title_y_offset, r"Hidden Layer$\in \mathbb{R}^{M}$",
            ha='center', va='bottom', fontsize=title_fontsize)
    ax.text(x_out, y_out_top + title_y_offset, r"Output Layer$\in \mathbb{R}^{4}$",
            ha='center', va='bottom', fontsize=title_fontsize)

    # Limits
    all_y = np.concatenate([y_in, y_hid, y_out])
    ax.set_xlim(x_in - 1.2, x_out + 1.2)
    ax.set_ylim(all_y.min() - 1.1, all_y.max() + 1.8)

    return fig, ax

if __name__ == "__main__":
    fig, ax = draw_nn_schematic_matplotlib(K=30, M=50, vdots_fontsize=16)
    fig.tight_layout()
    # plt.show()
    fig.savefig("NN_Schematic.png",bbox_inches="tight",pad_inches=0.0,dpi=300)