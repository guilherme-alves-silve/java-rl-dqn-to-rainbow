"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x = np.linspace(-6, 8, 1000)

curves = [
    (0.0, 1.0,  r'$\mu^w=0,\ \sigma^w=1$ (init)'),
    (2.5, 1.0,  r'$\mu^w=2.5,\ \sigma^w=1$ (shift right)'),
    (-2.0, 1.0, r'$\mu^w=-2,\ \sigma^w=1$ (shift left)'),
    (0.0, 0.4,  r'$\mu^w=0,\ \sigma^w=0.4$ (narrow)'),
    (0.0, 2.0,  r'$\mu^w=0,\ \sigma^w=2$ (wide)'),
]

fig, ax = plt.subplots(figsize=(11, 6))

for mu, sigma, label in curves:
    ax.plot(x, gaussian(x, mu, sigma), linewidth=2.2, label=label)
    ax.axvline(mu, linewidth=1.0, linestyle='--', alpha=0.4)

ax.set_xlabel('w', fontsize=13)
ax.set_ylabel('Probability Density', fontsize=13)
ax.set_title(r'NoisyNets — Weight Distribution $w \sim \mathcal{N}(\mu^w,\ (\sigma^w)^2)$',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-6, 8)
ax.set_ylim(0, 1.1)

plt.tight_layout()
OUT = "./graphics/noisy_nets_weight_dist.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
