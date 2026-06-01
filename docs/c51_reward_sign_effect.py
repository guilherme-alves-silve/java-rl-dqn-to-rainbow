"""
C51 — Bellman Projection: Effect of reward sign
=================================================
Compares the distributional shift for r > 0, r = 0, r < 0
using N = 25 atoms. Each panel shows PMF before/after and
the mass shift delta.

Paper: "A Distributional Perspective on Reinforcement Learning"
        Bellemare, Dabney, Munos (2017)

author: Guilherme Alves Silveira
generated with AI tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Config ─────────────────────────────────────────────────────────────────────
N      = 25
V_MIN  = -10.0
V_MAX  =  10.0
GAMMA  =  0.9

CASES = [
    dict(r=  2.0, label="r = +2.0  (positive reward)",  color_after="#1D9E75"),
    dict(r=  0.0, label="r =  0.0  (no reward, γ only)", color_after="#378ADD"),
    dict(r= -2.0, label="r = −2.0  (negative reward)",  color_after="#D85A30"),
]

COLOR_BEFORE = "#534AB7"
C_TEXT       = "#2C2C2A"
C_MUTED      = "#5F5E5A"

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_support(n, v_min, v_max):
    return np.linspace(v_min, v_max, n)


def make_prior(z):
    """Gaussian prior centred slightly left of zero."""
    raw = np.exp(-0.5 * ((z - (-1.0)) / 3.5) ** 2)
    return raw / raw.sum()


def bellman_project(p, z, r, gamma, v_min, v_max):
    n  = len(z)
    dz = (v_max - v_min) / (n - 1)
    m  = np.zeros(n)
    for j in range(n):
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


# ── Data ───────────────────────────────────────────────────────────────────────
z        = make_support(N, V_MIN, V_MAX)
dz       = (V_MAX - V_MIN) / (N - 1)
p_before = make_prior(z)
E_before = float(np.dot(p_before, z))

for case in CASES:
    case["p_after"]  = bellman_project(p_before, z, case["r"], GAMMA, V_MIN, V_MAX)
    case["E_after"]  = float(np.dot(case["p_after"], z))
    case["delta_m"]  = case["p_after"] - p_before

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor("white")

fig.suptitle(
    f"C51 — Effect of reward sign on Bellman projection\n"
    f"N = {N} atoms,  γ = {GAMMA},  "
    f"$[V_{{\\min}}, V_{{\\max}}] = [{int(V_MIN)}, {int(V_MAX)}]$",
    fontsize=14, fontweight="500", y=0.99, color=C_TEXT
)

gs = GridSpec(
    3, 3,
    figure=fig,
    hspace=0.55, wspace=0.32,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.05
)

xlim = (V_MIN - 0.5, V_MAX + 0.5)
bw   = dz * 0.36

for row, case in enumerate(CASES):
    p_after  = case["p_after"]
    delta_m  = case["delta_m"]
    E_after  = case["E_after"]
    c_after  = case["color_after"]
    r_val    = case["r"]

    ymax_pmf = max(p_before.max(), p_after.max()) * 1.28
    ymax_dlt = max(abs(delta_m).max() * 1.35, 0.01)

    ax_pmf  = fig.add_subplot(gs[row, 0])
    ax_cdf  = fig.add_subplot(gs[row, 1])
    ax_diff = fig.add_subplot(gs[row, 2])

    for ax in [ax_pmf, ax_cdf, ax_diff]:
        ax.set_facecolor("#F8F8F8")
        ax.grid(True, linewidth=0.4, color="#DDDDDD")
        ax.tick_params(labelsize=8)
        ax.set_xlim(xlim)

    # ── PMF ───────────────────────────────────────────────────────────────────
    ax_pmf.bar(z - bw * 0.55, p_before,
               width=bw, color=COLOR_BEFORE, alpha=0.85,
               align="center", edgecolor="white", linewidth=0.5,
               label=f"Before  E[Z]={E_before:.2f}")
    ax_pmf.bar(z + bw * 0.55, p_after,
               width=bw, color=c_after, alpha=0.85,
               align="center", edgecolor="white", linewidth=0.5,
               hatch="//", label=f"After   E[Z]={E_after:.2f}")

    ax_pmf.axvline(E_before, color=COLOR_BEFORE,
                   linestyle="--", linewidth=1.3, alpha=0.8)
    ax_pmf.axvline(E_after,  color=c_after,
                   linestyle="--", linewidth=1.3, alpha=0.8)

    # shift annotation arrow
    ax_pmf.annotate(
        "",
        xy=(E_after,  ymax_pmf * 0.88),
        xytext=(E_before, ymax_pmf * 0.88),
        arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.4)
    )
    shift = E_after - E_before
    sign  = "+" if shift >= 0 else ""
    ax_pmf.text(
        (E_before + E_after) / 2,
        ymax_pmf * 0.93,
        f"{sign}{shift:.2f}",
        ha="center", fontsize=8.5, color=C_TEXT, fontweight="500"
    )

    ax_pmf.set_title(case["label"], fontsize=10, pad=5, color=C_TEXT)
    ax_pmf.set_xlabel("Value z", fontsize=8.5)
    ax_pmf.set_ylabel("Probability p(z)", fontsize=8.5)
    ax_pmf.set_ylim(0, ymax_pmf)
    ax_pmf.legend(fontsize=7.5, framealpha=0.85, loc="upper left")

    # ── CDF ───────────────────────────────────────────────────────────────────
    cdf_before = np.cumsum(p_before)
    cdf_after  = np.cumsum(p_after)
    z_ext      = np.concatenate([[V_MIN - dz], z, [V_MAX + dz]])

    c_b = np.concatenate([[0], cdf_before, [1]])
    ax_cdf.fill_between(z_ext, c_b, step="post",
                        color=COLOR_BEFORE, alpha=0.20)
    ax_cdf.step(z_ext, c_b, where="post",
                color=COLOR_BEFORE, linewidth=1.2, label="Before")
    ax_cdf.scatter(z, cdf_before, color=COLOR_BEFORE, s=16, zorder=5)

    c_a = np.concatenate([[0], cdf_after, [1]])
    ax_cdf.fill_between(z_ext, c_a, step="post",
                        color=c_after, alpha=0.18)
    ax_cdf.step(z_ext, c_a, where="post",
                color=c_after, linewidth=1.2,
                linestyle="--", label="After")
    ax_cdf.scatter(z, cdf_after, color=c_after,
                   s=14, zorder=5, marker="D")

    ax_cdf.set_title(f"CDF  (N={N})", fontsize=10, pad=5)
    ax_cdf.set_xlabel("Value z", fontsize=8.5)
    ax_cdf.set_ylabel("Cumulative probability", fontsize=8.5)
    ax_cdf.set_ylim(-0.02, 1.10)
    ax_cdf.legend(fontsize=7.5, framealpha=0.85, loc="upper left")

    # ── Mass delta ────────────────────────────────────────────────────────────
    colors_delta = [c_after if d >= 0 else COLOR_BEFORE for d in delta_m]
    ax_diff.bar(z, delta_m,
                width=dz * 0.72, color=colors_delta,
                alpha=0.82, align="center",
                edgecolor="white", linewidth=0.4)
    ax_diff.axhline(0, color="#888", linewidth=0.8)
    ax_diff.axvline(E_before, color=COLOR_BEFORE,
                    linestyle="--", linewidth=1.0, alpha=0.6,
                    label=f"E[Z] before={E_before:.2f}")
    ax_diff.axvline(E_after,  color=c_after,
                    linestyle="--", linewidth=1.0, alpha=0.6,
                    label=f"E[Z] after={E_after:.2f}")

    ax_diff.set_title(r"Mass shift  $m_i - p_i$", fontsize=10, pad=5)
    ax_diff.set_xlabel("Value z", fontsize=8.5)
    ax_diff.set_ylabel(r"$\Delta$ probability", fontsize=8.5)
    ax_diff.set_ylim(-ymax_dlt, ymax_dlt)
    ax_diff.legend(fontsize=7.5, framealpha=0.85, loc="upper left")

    # gain/loss annotation
    gained = delta_m[delta_m > 0].sum()
    lost   = delta_m[delta_m < 0].sum()
    ax_diff.text(
        0.98, 0.97,
        f"gained: +{gained:.3f}\nlost:   {lost:.3f}",
        transform=ax_diff.transAxes,
        fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#FFFFCC", edgecolor="#CCCC88", alpha=0.9)
    )

# ── Row labels on the left ─────────────────────────────────────────────────────
row_labels = ["r > 0\n(positive)", "r = 0\n(no reward)", "r < 0\n(negative)"]
row_colors = [c["color_after"] for c in CASES]
for i, (lbl, col) in enumerate(zip(row_labels, row_colors)):
    fig.text(
        0.003,
        0.79 - i * 0.295,
        lbl,
        fontsize=9, color=col, fontweight="500",
        va="center", ha="left", rotation=90
    )

OUT = "./graphics/c51_reward_sign_effect.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
