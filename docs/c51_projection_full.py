"""
C51 — Bellman Projection: Before vs After
==========================================
For each atom count N, shows three panels:
  1. PMF before and after Bellman projection (overlaid bars)
  2. CDF before and after projection
  3. Support & spacing (lollipop chart for before and after)

Paper: "A Distributional Perspective on Reinforcement Learning"
        Bellemare, Dabney, Munos (2017)

author: Guilherme Alves Silveira
generated with AI tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# ── Global parameters ──────────────────────────────────────────────────────────
V_MIN       = -10.0
V_MAX       =  10.0
REWARD      =  1.5
GAMMA       =  0.9
ATOM_COUNTS = [5, 10, 25, 40, 51, 75, 100]

# ── Fixed colors: one per concept, consistent across every panel ───────────────
COLOR_BEFORE = "#534AB7"   # purple  — before projection
COLOR_AFTER  = "#1D9E75"   # teal    — after projection
COLOR_BEFORE_LIGHT = "#AFA9EC"
COLOR_AFTER_LIGHT  = "#5DCAA5"
CDF_FILL_BEFORE = "#9B99D4"
CDF_FILL_AFTER  = "#5DCAA5"

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_prior(N):
    """Gaussian prior centred slightly left of zero."""
    z     = np.linspace(V_MIN, V_MAX, N)
    raw   = np.exp(-0.5 * ((z - (-1.5)) / 4.0) ** 2)
    return z, raw / raw.sum()


def bellman_project(p, z, r, gamma, v_min, v_max):
    N  = len(z)
    dz = (v_max - v_min) / (N - 1)
    m  = np.zeros(N)
    for j in range(N):
        tz = np.clip(r + gamma * z[j], v_min, v_max)
        b  = (tz - v_min) / dz
        l  = int(np.floor(b))
        u  = int(np.ceil(b))
        if l == u:
            m[l] += p[j]
        else:
            m[l] += p[j] * (u - b)
            m[u] += p[j] * (b - l)
    return m


# ── Figure ─────────────────────────────────────────────────────────────────────
NROWS = len(ATOM_COUNTS)
fig   = plt.figure(figsize=(15, 4.2 * NROWS))
fig.patch.set_facecolor("white")

fig.suptitle(
    f"C51 — Bellman Projection: Before vs After\n"
    f"$r = {REWARD}$,  $\\gamma = {GAMMA}$,  "
    f"$[V_{{\\min}}, V_{{\\max}}] = [{int(V_MIN)}, {int(V_MAX)}]$",
    fontsize=14, fontweight="500", y=1.002
)

gs = GridSpec(
    NROWS, 3,
    figure=fig,
    hspace=0.65, wspace=0.32,
    left=0.07, right=0.97,
    top=0.98,   bottom=0.03
)

for row, N in enumerate(ATOM_COUNTS):
    z, p_before = make_prior(N)
    p_after     = bellman_project(p_before, z, REWARD, GAMMA, V_MIN, V_MAX)
    dz          = (V_MAX - V_MIN) / (N - 1)
    E_before    = float(np.dot(p_before, z))
    E_after     = float(np.dot(p_after,  z))

    ax_pmf  = fig.add_subplot(gs[row, 0])
    ax_cdf  = fig.add_subplot(gs[row, 1])
    ax_supp = fig.add_subplot(gs[row, 2])

    xlim = (V_MIN - 0.6, V_MAX + 0.6)
    ymax = max(p_before.max(), p_after.max()) * 1.28

    # ── 1. PMF ────────────────────────────────────────────────────────────────
    bw = dz * 0.36
    ax_pmf.bar(z - bw * 0.55, p_before,
               width=bw, color=COLOR_BEFORE, alpha=0.85,
               align="center", edgecolor="white", linewidth=0.5,
               label=f"Before  E[Z]={E_before:.1f}")
    ax_pmf.bar(z + bw * 0.55, p_after,
               width=bw, color=COLOR_AFTER, alpha=0.85,
               align="center", edgecolor="white", linewidth=0.5,
               hatch="//", label=f"After   E[Z]={E_after:.1f}")

    ax_pmf.axvline(E_before, color=COLOR_BEFORE, linestyle="--", linewidth=1.3)
    ax_pmf.axvline(E_after,  color=COLOR_AFTER,  linestyle="--", linewidth=1.3)

    ax_pmf.set_title(f"Probability Mass  (atoms={N})", fontsize=10, pad=5)
    ax_pmf.set_xlabel("Value (z)", fontsize=8)
    ax_pmf.set_ylabel("P(z)", fontsize=8)
    ax_pmf.set_xlim(xlim)
    ax_pmf.set_ylim(0, ymax)
    ax_pmf.tick_params(labelsize=7)
    ax_pmf.legend(fontsize=7, framealpha=0.85, loc="upper left")
    ax_pmf.set_facecolor("#F8F8F8")
    ax_pmf.grid(True, linewidth=0.4, color="#DDDDDD")

    # ── 2. CDF ────────────────────────────────────────────────────────────────
    cdf_before = np.cumsum(p_before)
    cdf_after  = np.cumsum(p_after)

    z_ext = np.concatenate([[V_MIN - dz], z, [V_MAX + dz]])

    c_b = np.concatenate([[0], cdf_before, [1]])
    ax_cdf.fill_between(z_ext, c_b, step="post",
                        color=CDF_FILL_BEFORE, alpha=0.40)
    ax_cdf.step(z_ext, c_b, where="post",
                color=COLOR_BEFORE, linewidth=1.1, label="Before")
    ax_cdf.scatter(z, cdf_before,
                   color=COLOR_BEFORE, s=18, zorder=5)

    c_a = np.concatenate([[0], cdf_after, [1]])
    ax_cdf.fill_between(z_ext, c_a, step="post",
                        color=CDF_FILL_AFTER, alpha=0.30)
    ax_cdf.step(z_ext, c_a, where="post",
                color=COLOR_AFTER, linewidth=1.1,
                linestyle="--", label="After")
    ax_cdf.scatter(z, cdf_after,
                   color=COLOR_AFTER, s=14, zorder=5, marker="D")

    ax_cdf.set_title(f"CDF  (atoms={N})", fontsize=10, pad=5)
    ax_cdf.set_xlabel("Value (z)", fontsize=8)
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=8)
    ax_cdf.set_xlim(xlim)
    ax_cdf.set_ylim(-0.02, 1.10)
    ax_cdf.tick_params(labelsize=7)
    ax_cdf.legend(fontsize=7, framealpha=0.85, loc="upper left")
    ax_cdf.set_facecolor("#F8F8F8")
    ax_cdf.grid(True, linewidth=0.4, color="#DDDDDD")

    # ── 3. Support & spacing (lollipop) ───────────────────────────────────────
    max_p = max(p_before.max(), p_after.max())

    ax_supp.vlines(z, 0, p_before / max_p,
                   color=COLOR_BEFORE, linewidth=1.0, alpha=0.9)
    ax_supp.scatter(z, p_before / max_p,
                    color=COLOR_BEFORE, s=30, zorder=4)

    ax_supp.vlines(z, 0, p_after / max_p,
                   color=COLOR_AFTER, linewidth=1.0,
                   linestyle="dashed", alpha=0.80)
    ax_supp.scatter(z, p_after / max_p,
                    color=COLOR_AFTER, s=22, zorder=5, marker="D")

    info = (
        f"Before  E[Z]={E_before:.2f}\n"
        f"After   E[Z]={E_after:.2f}\n"
        f"$\\Delta z={dz:.3f}$"
    )
    ax_supp.text(
        0.03, 0.97, info,
        transform=ax_supp.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#FFFFCC", edgecolor="#CCCC88", alpha=0.9)
    )

    leg_els = [
        Line2D([0], [0], color=COLOR_BEFORE, lw=1.2,
               marker="o", markersize=5, label="Before"),
        Line2D([0], [0], color=COLOR_AFTER,  lw=1.2,
               linestyle="dashed", marker="D", markersize=4, label="After"),
    ]
    ax_supp.legend(handles=leg_els, fontsize=7,
                   framealpha=0.85, loc="upper right")

    ax_supp.set_title(
        f"Support & Spacing  (atoms={N}, $\\Delta z={dz:.2f}$)",
        fontsize=10, pad=5
    )
    ax_supp.set_xlabel("Value (z)", fontsize=8)
    ax_supp.set_ylabel("Scaled Probability", fontsize=8)
    ax_supp.set_xlim(xlim)
    ax_supp.set_ylim(-0.05, 1.18)
    ax_supp.tick_params(labelsize=7)
    ax_supp.set_facecolor("#F8F8F8")
    ax_supp.grid(True, linewidth=0.4, color="#DDDDDD")

OUT = "./graphics/c51_projection_full.jpg"
plt.savefig(OUT, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
