"""
Output Layer Comparison: DQN vs Dueling DQN vs C51
====================================================
Visualizes the structural difference between the final layers
of each architecture, showing how the output shape changes.

Assumes:
  |A| = 6 actions
  N   = 8 atoms  (simplified from 51 for visual clarity)
  Hidden layer size = 512

author: Guilherme Alves Silveira
generated with AI tools
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── Config ─────────────────────────────────────────────────────────────────────
N_ACTIONS = 6
N_ATOMS   = 8
HIDDEN    = 512

# ── Colors ─────────────────────────────────────────────────────────────────────
C_SHARED  = "#B4B2A9"   # gray  — shared layers
C_DQN     = "#534AB7"   # purple — DQN head
C_DUELING_V = "#1D9E75" # teal  — value stream
C_DUELING_A = "#D85A30" # coral — advantage stream
C_C51     = "#BA7517"   # amber — C51 head
C_SOFT    = "#085041"   # dark teal — softmax result
C_BG      = "#F8F8F8"
C_TEXT    = "#2C2C2A"
C_MUTED   = "#5F5E5A"

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("white")

fig.suptitle(
    "Output layer comparison: DQN  ·  Dueling DQN  ·  C51\n"
    f"|A| = {N_ACTIONS} actions,  N = {N_ATOMS} atoms",
    fontsize=14, fontweight="500", y=0.98, color=C_TEXT
)

gs = GridSpec(1, 3, figure=fig,
              left=0.04, right=0.97,
              top=0.90,  bottom=0.04,
              wspace=0.08)

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax in axes:
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")


# ── Helper functions ───────────────────────────────────────────────────────────
def rounded_box(ax, x, y, w, h, color, alpha=1.0, lw=0.8,
                edgecolor=None, zorder=2):
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=ec,
        alpha=alpha, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    return box


def label(ax, x, y, text, fontsize=9, color=C_TEXT,
          ha="center", va="center", weight="normal"):
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha=ha, va=va, fontweight=weight,
            fontfamily="monospace" if "$" not in text else "sans-serif")


def arrow(ax, x1, y1, x2, y2, color=C_MUTED, lw=1.2):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            mutation_scale=10
        ), zorder=3
    )


def neuron_row(ax, x_center, y, n, color, radius=0.18, spacing=0.5,
               label_text=None, alpha=1.0):
    """Draw a row of n neuron circles, centered at x_center."""
    total_w = (n - 1) * spacing
    x_start = x_center - total_w / 2
    for i in range(n):
        cx = x_start + i * spacing
        circle = plt.Circle((cx, y), radius,
                             color=color, alpha=alpha, zorder=3)
        ax.add_patch(circle)
    if label_text:
        ax.text(x_center, y - 0.55, label_text,
                fontsize=7.5, color=C_MUTED, ha="center", va="top")


def shared_layers(ax, title):
    """Draw the shared backbone present in all three architectures."""
    ax.text(5, 17.4, title, fontsize=12, color=C_TEXT,
            ha="center", va="center", fontweight="500")

    # input layer
    neuron_row(ax, 5, 16.2, 5, C_SHARED, alpha=0.5)
    ax.text(5, 15.72, "input  s", fontsize=7.5, color=C_MUTED,
            ha="center", va="top")

    arrow(ax, 5, 15.8, 5, 15.2)

    # hidden layer 1
    rounded_box(ax, 2.2, 14.4, 5.6, 0.7, C_SHARED, alpha=0.25)
    label(ax, 5, 14.75, "shared hidden  (512)", fontsize=8.5, color=C_MUTED)

    arrow(ax, 5, 14.4, 5, 13.8)

    # hidden layer 2
    rounded_box(ax, 2.2, 13.0, 5.6, 0.7, C_SHARED, alpha=0.25)
    label(ax, 5, 13.35, "shared hidden  (512)", fontsize=8.5, color=C_MUTED)

    arrow(ax, 5, 13.0, 5, 12.4)
    ax.text(5, 12.25, "feature vector", fontsize=7.5,
            color=C_MUTED, ha="center", va="top")


# ── Panel 1: DQN ──────────────────────────────────────────────────────────────
ax = axes[0]
shared_layers(ax, "DQN")

arrow(ax, 5, 11.9, 5, 11.2)

# single linear head
rounded_box(ax, 2.0, 10.0, 6.0, 1.0, C_DQN, alpha=0.15,
            edgecolor=C_DQN, lw=1.2)
label(ax, 5, 10.5, f"linear  (512 → {N_ACTIONS})",
      fontsize=8.5, color="#3C3489")

arrow(ax, 5, 10.0, 5, 9.3)

# output: one Q per action
y_out = 9.0
for i in range(N_ACTIONS):
    cx = 1.6 + i * 1.36
    rounded_box(ax, cx, y_out - 0.3, 1.1, 0.55,
                C_DQN, alpha=0.8, edgecolor="#3C3489", lw=0.8)
    label(ax, cx + 0.55, y_out - 0.02,
          f"Q(s,a{i+1})", fontsize=7, color="white")

ax.text(5, 8.45, f"output: {N_ACTIONS} scalars",
        fontsize=8, color=C_MUTED, ha="center")

# formula box
rounded_box(ax, 1.5, 5.8, 7.0, 1.8, C_DQN, alpha=0.06,
            edgecolor=C_DQN, lw=0.8)
label(ax, 5, 7.35, "Output layer params:", fontsize=8,
      color=C_MUTED, weight="500")
label(ax, 5, 6.85,
      f"512 × {N_ACTIONS}  +  {N_ACTIONS}  =  {512*N_ACTIONS + N_ACTIONS:,}",
      fontsize=9, color="#3C3489", weight="500")
label(ax, 5, 6.35, "one Q-value per action", fontsize=8, color=C_MUTED)
label(ax, 5, 5.95, "Q learned directly", fontsize=8, color=C_MUTED)

# intuition note
rounded_box(ax, 1.0, 3.0, 8.0, 2.5, C_DQN, alpha=0.04,
            edgecolor=C_DQN, lw=0.6)
label(ax, 5, 4.9, "How it works", fontsize=8.5,
      color="#3C3489", weight="500")
for i, line in enumerate([
    "Network maps state directly",
    "to one expected return",
    "per action — a scalar.",
    "Uncertainty is lost."
]):
    label(ax, 5, 4.35 - i * 0.42, line, fontsize=7.8, color=C_MUTED)


# ── Panel 2: Dueling DQN ──────────────────────────────────────────────────────
ax = axes[1]
shared_layers(ax, "Dueling DQN")

# split into two streams
arrow(ax, 3.5, 11.9, 2.5, 11.2)   # value stream
arrow(ax, 6.5, 11.9, 7.5, 11.2)   # advantage stream

ax.text(2.2, 11.85, "split", fontsize=7.5,
        color=C_MUTED, ha="center")

# value stream
rounded_box(ax, 0.5, 9.9, 3.8, 0.9, C_DUELING_V, alpha=0.15,
            edgecolor=C_DUELING_V, lw=1.0)
label(ax, 2.4, 10.35, "linear (512→1)", fontsize=8, color="#085041")
label(ax, 2.4, 9.98, "V(s) stream", fontsize=7.5, color=C_MUTED)

# advantage stream
rounded_box(ax, 5.7, 9.9, 3.8, 0.9, C_DUELING_A, alpha=0.15,
            edgecolor=C_DUELING_A, lw=1.0)
label(ax, 7.6, 10.35,
      f"linear (512→{N_ACTIONS})", fontsize=8, color="#712B13")
label(ax, 7.6, 9.98, "A(s,a) stream", fontsize=7.5, color=C_MUTED)

# outputs
# V(s) — single box
rounded_box(ax, 1.3, 8.7, 2.2, 0.55,
            C_DUELING_V, alpha=0.8, edgecolor="#085041", lw=0.8)
label(ax, 2.4, 8.98, "V(s)", fontsize=8, color="white")

# A(s,a) — one box per action
for i in range(N_ACTIONS):
    cx = 4.85 + i * 0.82
    rounded_box(ax, cx, 8.7, 0.7, 0.55,
                C_DUELING_A, alpha=0.75, edgecolor="#712B13", lw=0.6)
    label(ax, cx + 0.35, 8.98,
          f"A{i+1}", fontsize=6.5, color="white")

ax.text(5, 8.35, f"1 scalar  +  {N_ACTIONS} scalars",
        fontsize=8, color=C_MUTED, ha="center")

# combination arrows
arrow(ax, 2.4, 8.7, 5.0, 7.85, color=C_DUELING_V)
arrow(ax, 7.6, 8.7, 5.0, 7.85, color=C_DUELING_A)

# combination formula
rounded_box(ax, 1.8, 6.9, 6.4, 0.85, "#888780", alpha=0.12,
            edgecolor="#888780", lw=0.8)
label(ax, 5, 7.32, "Q(s,a) = V(s) + A(s,a) − mean[A(s,·)]",
      fontsize=8.2, color=C_TEXT)

arrow(ax, 5, 6.9, 5, 6.25)

# final Q output
for i in range(N_ACTIONS):
    cx = 1.6 + i * 1.36
    rounded_box(ax, cx, 5.7, 1.1, 0.5,
                C_DQN, alpha=0.7, edgecolor="#3C3489", lw=0.7)
    label(ax, cx + 0.55, 5.95,
          f"Q(s,a{i+1})", fontsize=7, color="white")

ax.text(5, 5.35, f"output: {N_ACTIONS} Q-values (combined)",
        fontsize=8, color=C_MUTED, ha="center")

# formula box
rounded_box(ax, 1.0, 3.5, 8.0, 1.6, C_DUELING_V, alpha=0.06,
            edgecolor=C_DUELING_V, lw=0.8)
label(ax, 5, 4.85, "Output layer params:", fontsize=8,
      color=C_MUTED, weight="500")
v_params = 512 * 1 + 1
a_params = 512 * N_ACTIONS + N_ACTIONS
label(ax, 5, 4.38,
      f"V: 512×1+1 = {v_params}   A: 512×{N_ACTIONS}+{N_ACTIONS} = {a_params}",
      fontsize=8.5, color="#085041", weight="500")
label(ax, 5, 3.92,
      f"total = {v_params + a_params:,}  (same scale as DQN)",
      fontsize=8, color=C_MUTED)

# intuition note
rounded_box(ax, 1.0, 0.9, 8.0, 2.4, C_DUELING_V, alpha=0.04,
            edgecolor=C_DUELING_V, lw=0.6)
label(ax, 5, 3.05, "How it works", fontsize=8.5,
      color="#085041", weight="500")
for i, line in enumerate([
    "Separates 'how good is",
    "this state' (V) from 'how",
    "much better is this action' (A).",
    "Still scalar outputs."
]):
    label(ax, 5, 2.55 - i * 0.42, line, fontsize=7.8, color=C_MUTED)


# ── Panel 3: C51 ──────────────────────────────────────────────────────────────
ax = axes[2]
shared_layers(ax, "C51")

arrow(ax, 5, 11.9, 5, 11.2)

# linear head
rounded_box(ax, 1.2, 10.0, 7.6, 1.0, C_C51, alpha=0.15,
            edgecolor=C_C51, lw=1.2)
label(ax, 5, 10.52,
      f"linear  (512 → {N_ATOMS}×{N_ACTIONS} = {N_ATOMS*N_ACTIONS})",
      fontsize=8.5, color="#633806")
label(ax, 5, 10.08, "one distribution per action", fontsize=7.5, color=C_MUTED)

arrow(ax, 5, 10.0, 5, 9.4)
ax.text(5, 9.28, "softmax per action (column-wise)",
        fontsize=7.5, color=C_MUTED, ha="center")

# matrix of distributions
cell_w = 0.98
cell_h = 0.38
x0 = 0.8
y0_top = 9.0

for j in range(N_ACTIONS):
    cx = x0 + j * (cell_w + 0.12)
    # column header
    ax.text(cx + cell_w/2, y0_top + 0.18,
            f"a{j+1}", fontsize=7, color="#633806",
            ha="center", va="center", fontweight="500")
    for i in range(N_ATOMS):
        cy = y0_top - (i + 0.5) * (cell_h + 0.04) - 0.12
        intensity = np.exp(-0.5 * ((i - N_ATOMS//2 + (j-2)*0.5) / 2.0)**2)
        color_val = plt.cm.YlOrBr(0.2 + 0.6 * intensity)
        rounded_box(ax, cx, cy, cell_w, cell_h,
                    color_val, alpha=0.9, lw=0.3,
                    edgecolor="#BA7517")
        if j == 0:
            ax.text(0.55, cy + cell_h/2,
                    f"z{i}", fontsize=6.2, color=C_MUTED,
                    ha="right", va="center")

# axis labels
ax.text(x0 - 0.05, y0_top - (N_ATOMS/2) * (cell_h + 0.04),
        "atoms\n(N)", fontsize=7, color=C_MUTED,
        ha="right", va="center", rotation=90)

ax.text(5, y0_top - N_ATOMS * (cell_h + 0.04) - 0.35,
        f"output: {N_ATOMS}×{N_ACTIONS} matrix  (p per atom per action)",
        fontsize=7.8, color=C_MUTED, ha="center")

# formula box
y_form = y0_top - N_ATOMS * (cell_h + 0.04) - 0.75
rounded_box(ax, 0.8, y_form - 1.9, 8.4, 2.0, C_C51, alpha=0.06,
            edgecolor=C_C51, lw=0.8)
label(ax, 5, y_form - 0.22, "Output layer params:", fontsize=8,
      color=C_MUTED, weight="500")
c51_params = 512 * N_ATOMS * N_ACTIONS + N_ATOMS * N_ACTIONS
label(ax, 5, y_form - 0.70,
      f"512 × ({N_ATOMS}×{N_ACTIONS})  +  {N_ATOMS}×{N_ACTIONS}  =  {c51_params:,}",
      fontsize=8.8, color="#633806", weight="500")
label(ax, 5, y_form - 1.12,
      f"= {N_ATOMS}× more params than DQN final layer",
      fontsize=8, color=C_MUTED)
label(ax, 5, y_form - 1.55,
      "Q emerges: E[Z] = Σ z_i · p_i", fontsize=8.5,
      color="#633806", weight="500")

# intuition note
y_note = y_form - 2.15
rounded_box(ax, 0.8, y_note - 2.5, 8.4, 2.5, C_C51, alpha=0.04,
            edgecolor=C_C51, lw=0.6)
label(ax, 5, y_note - 0.28, "How it works", fontsize=8.5,
      color="#633806", weight="500")
for i, line in enumerate([
    "Network maps state to a full",
    "return distribution per action.",
    "Softmax per column → valid p_i.",
    "Q is derived, not learned.",
    "More expressive, higher cost."
]):
    label(ax, 5, y_note - 0.78 - i * 0.42, line, fontsize=7.8, color=C_MUTED)


# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_SHARED,    alpha=0.5,  label="Shared layers (all)"),
    mpatches.Patch(color=C_DQN,       alpha=0.8,  label="DQN head"),
    mpatches.Patch(color=C_DUELING_V, alpha=0.8,  label="Dueling — value stream V(s)"),
    mpatches.Patch(color=C_DUELING_A, alpha=0.8,  label="Dueling — advantage stream A(s,a)"),
    mpatches.Patch(color=C_C51,       alpha=0.8,  label="C51 head (distribution matrix)"),
]
fig.legend(handles=legend_patches,
           loc="lower center", ncol=5,
           fontsize=8.5, framealpha=0.8,
           bbox_to_anchor=(0.5, 0.005))

OUT = "./graphics/output_layer_comparison.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
