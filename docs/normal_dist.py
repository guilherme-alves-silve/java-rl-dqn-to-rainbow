"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-4, 4, 1000)
y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(x, y, where=(x >= -1) & (x <= 1), alpha=0.35, label=r'$\mu \pm 1\sigma$ (68.2%)')
ax.fill_between(x, y, where=(x >= -2) & (x <= 2), alpha=0.25, label=r'$\mu \pm 2\sigma$ (95.4%)')
ax.fill_between(x, y, where=(x >= -3) & (x <= 3), alpha=0.15, label=r'$\mu \pm 3\sigma$ (99.7%)')

ax.plot(x, y, linewidth=2.5)
ax.axvline(0, linewidth=1.5, linestyle='--', alpha=0.6)
ax.text(0.05, 0.42, r'$\mu = 0$', fontsize=12, va='center')

for s, label in zip([1, 2, 3], [r'$\sigma$', r'$2\sigma$', r'$3\sigma$']):
    ax.annotate('', xy=(s, 0.02), xytext=(0, 0.02),
                arrowprops=dict(arrowstyle='<->',lw=1.5))
    ax.text(s / 2, 0.035, label, ha='center', fontsize=11)

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Probability Density', fontsize=13)
ax.set_title(r'Normal Distribution $\mathcal{N}(\mu=0,\ \sigma^2=1)$', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.45)

OUT = "./graphics/normal_distribution.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
