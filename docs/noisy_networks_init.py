"""
Noisy Nets - Initialization

author: Guilherme Alves Silveira
generated with AI tools
"""
import numpy as np
import matplotlib.pyplot as plt


p = np.linspace(1, 500, 200)

# Fixed weight variance -> pre-activation variance grows linearly with p
var_fixed = 0.6 * p

# Weight variance scaled as 1/p -> pre-activation variance stays ~constant
var_scaled = np.full_like(p, 120.0)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(p, var_fixed, color="#E24B4A", linewidth=2.2, label="fixed weight variance")
ax.plot(p, var_scaled, color="#1D9E75", linewidth=2.2, label="weight variance ∝ 1/p")

ax.annotate("var(z) grows with p",
            xy=(400, var_fixed[np.searchsorted(p, 400)]),
            xytext=(250, 260),
            fontsize=10, color="#8a1f1f",
            arrowprops=dict(arrowstyle="->", color="#8a1f1f", lw=1))

ax.annotate("var(z) ≈ O(1)",
            xy=(300, 120),
            xytext=(300, 60),
            fontsize=10, color="#0f6e56",
            arrowprops=dict(arrowstyle="->", color="#0f6e56", lw=1))

ax.set_xlabel("Number of layer inputs (p)", fontsize=11)
ax.set_ylabel("Pre-activation variance z", fontsize=11)
ax.set_title("Effect of $\\frac{1}{p}$ scaling on pre-activation variance", fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.set_xlim(0, 500)
ax.set_ylim(0, 320)

fig.tight_layout()
OUT = "./graphics/noisy_net_variance_scaling.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
