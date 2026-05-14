"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import matplotlib.pyplot as plt
import numpy as np

# Noisy Networks formula
# ε^w = f(ε^out) f(ε^in)^T
# ε^b = f(ε^out)
# f(x) = sign(x) * sqrt(|x|)

def f(x):
    return np.sign(x) * np.sqrt(np.abs(x))

# Sizes to test
sizes = [2, 4, 8, 16, 32, 64, 128]

fig, axes = plt.subplots(len(sizes), 6, figsize=(20, 4 * len(sizes)))
fig.suptitle('Noisy Networks: Factorized vs. Independent Sampling', fontsize=16, fontweight='bold')

for row, size in enumerate(sizes):
    # ===== FACTORIZED SAMPLING (Noisy Networks) =====
    eps_in = np.random.randn(size)
    eps_out = np.random.randn(size)
    f_in = f(eps_in)
    f_out = f(eps_out)
    eps_w_factorized = np.outer(f_out, f_in)
    eps_b_factorized = f_out

    # ===== INDEPENDENT SAMPLING (Standard) =====
    eps_w_independent = np.random.randn(size, size)
    eps_b_independent = np.random.randn(size)

    # Plot 1: ε^w heatmap - Factorized
    ax1 = axes[row, 0]
    im1 = ax1.imshow(eps_w_factorized, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax1.set_title(f'Factorized $\\epsilon^w$ ({size}×{size})')
    ax1.set_xlabel('$f(\\epsilon^{in})$ index')
    ax1.set_ylabel('$f(\\epsilon^{out})$ index')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Plot 2: Distribution of ε^w - Factorized
    ax2 = axes[row, 1]
    ax2.hist(eps_w_factorized.flatten(), bins=30, color='purple', edgecolor='black', alpha=0.7, range=(-3, 3))
    ax2.set_title(f'Factorized $\\epsilon^w$ dist (size={size})')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.axvline(x=0, color='black', linewidth=0.5)

    # Plot 3: ε^b bar - Factorized
    ax3 = axes[row, 2]
    colors_f = ['red' if v < 0 else 'blue' for v in eps_b_factorized]
    ax3.bar(range(size), eps_b_factorized, color=colors_f, edgecolor='black', alpha=0.8)
    ax3.set_title(f'Factorized $\\epsilon^b$ (size={size})')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Value')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylim(-2, 2)

    # Plot 4: ε^w heatmap - Independent
    ax4 = axes[row, 3]
    im4 = ax4.imshow(eps_w_independent, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax4.set_title(f'Independent $\\epsilon^w$ ({size}×{size})')
    ax4.set_xlabel('$\\epsilon$ index')
    ax4.set_ylabel('$\\epsilon$ index')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Plot 5: Distribution of ε^w - Independent
    ax5 = axes[row, 4]
    ax5.hist(eps_w_independent.flatten(), bins=30, color='darkgreen', edgecolor='black', alpha=0.7, range=(-3, 3))
    ax5.set_title(f'Independent $\\epsilon^w$ dist (size={size})')
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Frequency')
    ax5.axvline(x=0, color='black', linewidth=0.5)

    # Plot 6: ε^b bar - Independent
    ax6 = axes[row, 5]
    colors_i = ['red' if v < 0 else 'blue' for v in eps_b_independent]
    ax6.bar(range(size), eps_b_independent, color=colors_i, edgecolor='black', alpha=0.8)
    ax6.set_title(f'Independent $\\epsilon^b$ (size={size})')
    ax6.set_xlabel('Index')
    ax6.set_ylabel('Value')
    ax6.axhline(y=0, color='black', linewidth=0.5)
    ax6.set_ylim(-2, 2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('graphics/noisy_nets_full_comparison.jpg', dpi=150, bbox_inches='tight')
plt.show()