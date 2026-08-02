"""
C51 — Output Layer for N = 3, 5, 10, 25 atoms
===============================================
Shows how the distribution matrix grows as N increases,
keeping |A| = 6 actions fixed across all panels.

author: Guilherme Alves Silveira
generated with AI tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── Config ─────────────────────────────────────────────────────────────────────
N_ACTIONS  = 6
ATOM_COUNTS = [3, 5, 10, 25]
HIDDEN     = 512

# ── Colors ─────────────────────────────────────────────────────────────────────
C_SHARED = "#B4B2A9"
C_C51    = "#BA7517"
C_TEXT   = "#2C2C2A"
C_MUTED  = "#5F5E5A"
C_BG     = "#F8F8F8"

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("white")

fig.suptitle(
    "C51 — Output layer as N grows\n"
    f"|A| = {N_ACTIONS} actions  ·  hidden = {HIDDEN}",
    fontsize=14, fontweight="500", y=0.99, color=C_TEXT
)

gs = GridSpec(1, 4, figure=fig,
              left=0.03, right=0.98,
              top=0.92,  bottom=0.04,
              wspace=0.10)

axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
for ax in axes:
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 10)
    ax.axis("off")


# ── Helpers ────────────────────────────────────────────────────────────────────
def rounded_box(ax, x, y, w, h, color, alpha=1.0, lw=0.8,
                edgecolor=None, zorder=2):
    ec = edgecolor if edgecolor else color
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=ec,
        alpha=alpha, linewidth=lw, zorder=zorder
    ))


def label(ax, x, y, text, fontsize=9, color=C_TEXT,
          ha="center", va="center", weight="normal"):
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha=ha, va=va, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, color=C_MUTED, lw=1.2):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10),
                zorder=3)


def neuron_row(ax, x_center, y, n, color, radius=0.18,
               spacing=0.52, alpha=0.5):
    total_w = (n - 1) * spacing
    x_start = x_center - total_w / 2
    for i in range(n):
        ax.add_patch(plt.Circle(
            (x_start + i * spacing, y), radius,
            color=color, alpha=alpha, zorder=3))


def shared_backbone(ax, ylim_top):
    """Draw the shared layers, returns y where the head starts."""
    top = ylim_top

    # input neurons
    neuron_row(ax, 5, top - 0.5, 5, C_SHARED)
    label(ax, 5, top - 1.05, "input  s", fontsize=8, color=C_MUTED)
    arrow(ax, 5, top - 1.15, 5, top - 1.75)

    # hidden 1
    rounded_box(ax, 1.5, top - 2.55, 7.0, 0.7,
                C_SHARED, alpha=0.25, edgecolor=C_SHARED)
    label(ax, 5, top - 2.2, "shared hidden  (512)",
          fontsize=8.5, color=C_MUTED)
    arrow(ax, 5, top - 2.55, 5, top - 3.15)

    # hidden 2
    rounded_box(ax, 1.5, top - 3.95, 7.0, 0.7,
                C_SHARED, alpha=0.25, edgecolor=C_SHARED)
    label(ax, 5, top - 3.6, "shared hidden  (512)",
          fontsize=8.5, color=C_MUTED)
    arrow(ax, 5, top - 3.95, 5, top - 4.55)

    label(ax, 5, top - 4.7, "feature vector",
          fontsize=7.5, color=C_MUTED)
    arrow(ax, 5, top - 4.85, 5, top - 5.45)

    return top - 5.45   # y where linear head begins


# ── Per-column layout ──────────────────────────────────────────────────────────
for col, N in enumerate(ATOM_COUNTS):
    ax  = axes[col]
    out = N_ACTIONS * N
    params = HIDDEN * out + out

    # dynamic ylim based on N
    # matrix width grows with N (columns = atoms); rows = actions, fixed at N_ACTIONS
    cell_h  = 0.55                  # fixed row height (one row per action)
    cell_w  = min(1.28, 8.5 / N)    # shrink cells as N grows (one column per atom)
    gap     = 0.06
    mat_h   = N_ACTIONS * (cell_h + gap)
    total_h = 5.6 + 1.8 + mat_h + 3.2   # backbone + head + matrix + info
    ylim    = total_h + 1.0

    ax.set_ylim(0, ylim)

    # title
    ax.text(5, ylim - 0.3, f"N = {N} atoms",
            fontsize=12, color=C_TEXT, ha="center",
            va="center", fontweight="500")

    # shared backbone
    y_head = shared_backbone(ax, ylim - 0.8)

    # linear head box
    rounded_box(ax, 1.0, y_head - 1.2, 8.0, 1.0,
                C_C51, alpha=0.15, edgecolor=C_C51, lw=1.2)
    label(ax, 5, y_head - 0.78,
          f"linear  ($512 \\rightarrow {N_ACTIONS} \\times {N} = {out}$)",
          fontsize=8.5, color="#633806", weight="500")
    label(ax, 5, y_head - 1.08,
          "one distribution per action",
          fontsize=7.5, color=C_MUTED)

    arrow(ax, 5, y_head - 1.2, 5, y_head - 1.75)
    label(ax, 5, y_head - 1.92,
          "softmax per action (row-wise, dim=1)",
          fontsize=7.2, color=C_MUTED)

    # distribution matrix — rows = actions (fixed), columns = atoms (grows with N)
    y_mat_top = y_head - 2.4
    x0 = 0.85

    # padding card around the matrix grid (~2px equivalent breathing room)
    pad = 0.14
    mat_x_left  = x0
    mat_x_right = x0 + N * (cell_w + gap) - gap
    mat_y_top   = y_mat_top - gap
    mat_y_bot   = y_mat_top - N_ACTIONS * (cell_h + gap)
    rounded_box(ax,
                mat_x_left - pad, mat_y_bot - pad,
                (mat_x_right - mat_x_left) + 2 * pad,
                (mat_y_top - mat_y_bot) + 2 * pad,
                "white", alpha=0.9, edgecolor="none", lw=0.6, zorder=1)

    # column headers (atoms)
    if N <= 10:
        for z_idx in range(N):
            cx = x0 + z_idx * (cell_w + gap)
            label(ax, cx + cell_w / 2, y_mat_top + 0.22,
                  f"$z_{z_idx}$", fontsize=6.5, color=C_MUTED)

    # action row labels + cells
    for a_idx in range(N_ACTIONS):
        cy = y_mat_top - (a_idx + 1) * (cell_h + gap)

        # row label
        label(ax, x0 - 0.28, cy + cell_h / 2,
              f"$a_{a_idx+1}$", fontsize=7.5, color="#633806",
              weight="500", ha="right", va="center")

        for z_idx in range(N):
            cx = x0 + z_idx * (cell_w + gap)
            # gaussian-ish intensity varying by atom and action
            mu_a = N * (0.3 + 0.07 * a_idx)
            intensity = np.exp(-0.5 * ((z_idx - mu_a) / (N * 0.22)) ** 2)
            color_val = plt.cm.YlOrBr(0.15 + 0.72 * intensity)
            rounded_box(ax, cx, cy, cell_w, cell_h,
                        color_val, alpha=0.92,
                        edgecolor="#BA7517", lw=0.25, zorder=2)

    # "actions" y-axis label
    ax.text(x0 - 0.6,
            y_mat_top - mat_h / 2,
            "actions",
            fontsize=7, color=C_MUTED,
            ha="center", va="center", rotation=90)

    y_below_mat = y_mat_top - mat_h - pad - 0.2
    label(ax, 5, y_below_mat,
          f"output: {N_ACTIONS}×{N} matrix  (rows=actions, cols=atoms)",
          fontsize=7.5, color=C_MUTED)
    label(ax, 5, y_below_mat - 0.28,
          f"atoms (N={N}) →",
          fontsize=7, color=C_MUTED)

    # params info box
    y_info = y_below_mat - 0.5
    rounded_box(ax, 0.8, y_info - 1.9, 8.4, 1.9,
                C_C51, alpha=0.06, edgecolor=C_C51, lw=0.8)

    label(ax, 5, y_info - 0.28,
          "Output layer params:", fontsize=8,
          color=C_MUTED, weight="500")
    label(ax, 5, y_info - 0.72,
          f"$512 \\times {out} + {out} = {params:,}$",
          fontsize=9, color="#633806", weight="500")

    ratio = params / (HIDDEN * N_ACTIONS + N_ACTIONS)
    label(ax, 5, y_info - 1.12,
          f"$= {ratio:.1f}\\times$ DQN final layer params",
          fontsize=8, color=C_MUTED)

    label(ax, 5, y_info - 1.55,
          f"$Q(s,a) = \\mathbb{{E}}[Z] = \\sum_{{i=0}}^{{{N} - 1}} z_i \\, p_i$",
          fontsize=8.5, color="#633806")

# ── Shared bottom annotation ───────────────────────────────────────────────────
fig.text(0.5, 0.005,
         "Shared backbone is identical across all panels — only the output head grows with N. "
         "DQN equivalent has 512×6+6 = 3,078 params in its final layer.",
         ha="center", fontsize=8.5, color=C_MUTED, style="italic")

OUT = "./graphics/c51_output_layer_atoms.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
