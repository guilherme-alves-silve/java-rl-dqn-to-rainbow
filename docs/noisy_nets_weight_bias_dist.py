"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x = np.linspace(-6, 8, 1000)

curves_w = [
    (0.0, 1.0,  r'$\mu^w=0,\ \sigma^w=1$ (init)'),
    (2.5, 1.0,  r'$\mu^w=2.5,\ \sigma^w=1$ (shift right)'),
    (-2.0, 1.0, r'$\mu^w=-2,\ \sigma^w=1$ (shift left)'),
    (0.0, 0.4,  r'$\mu^w=0,\ \sigma^w=0.4$ (narrow)'),
    (0.0, 2.0,  r'$\mu^w=0,\ \sigma^w=2$ (wide)'),
]

curves_b = [
    (0.0, 1.0,  r'$\mu^b=0,\ \sigma^b=1$ (init)'),
    (2.8, 1.0,  r'$\mu^b=2.8,\ \sigma^b=1$ (shift right)'),
    (-2.3, 1.0, r'$\mu^b=-2.3,\ \sigma^b=1$ (shift left)'),
    (0.0, 0.36,  r'$\mu^b=0,\ \sigma^b=0.36$ (narrow)'),
    (0.0, 3.0,  r'$\mu^b=0,\ \sigma^b=3$ (wide)'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, curves, symbol, full in zip(axes,
                                    [curves_w, curves_b],
                                    ['w', 'b'],
                                    [r'$w \sim \mathcal{N}(\mu^w,\ (\sigma^w)^2)$',
                                     r'$b \sim \mathcal{N}(\mu^b,\ (\sigma^b)^2)$']):
    for mu, sigma, label in curves:
        ax.plot(x, gaussian(x, mu, sigma), linewidth=2.2, label=label)
        ax.axvline(mu, linewidth=1.0, linestyle='--', alpha=0.4)

    ax.set_xlabel(symbol, fontsize=13)
    ax.set_title(f'NoisyNets — {full}', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-6, 8)
    ax.set_ylim(0, 1.1)

axes[0].set_ylabel('Probability Density', fontsize=13)

plt.tight_layout()
OUT = "./graphics/noisy_nets_weight_bias_dist.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
