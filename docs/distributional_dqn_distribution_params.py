import matplotlib.pyplot as plt
import numpy as np

# Distributional DQN (C51) parameters
V_MIN = -10  # Minimum value of the support
V_MAX = 10   # Maximum value of the support

# Number of atoms to test (as in C51 paper and extensions)
atoms_list = [5, 10, 25, 40, 51, 75, 100]

fig, axes = plt.subplots(len(atoms_list), 3, figsize=(15, 4 * len(atoms_list)))
fig.suptitle('Distributional DQN (C51): Value Distribution with Different Numbers of Atoms',
             fontsize=14, fontweight='bold')

for row, num_atoms in enumerate(atoms_list):
    # Compute support (discretized value range)
    delta_z = (V_MAX - V_MIN) / (num_atoms - 1)
    support = np.linspace(V_MIN, V_MAX, num_atoms)

    # Simulate a probability distribution over the support
    # Using a Gaussian-like distribution centered at a random value
    center = np.random.uniform(-5, 5)
    std = np.random.uniform(1.5, 4.0)
    probs = np.exp(-0.5 * ((support - center) / std) ** 2)
    probs = probs / np.sum(probs)  # Normalize to sum to 1

    # Plot 1: Probability mass function (bar plot)
    ax1 = axes[row, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, num_atoms))
    bars = ax1.bar(support, probs, width=delta_z * 0.8, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title(f'Probability Mass (atoms={num_atoms})')
    ax1.set_xlabel('Value (z)')
    ax1.set_ylabel('Probability P(z)')
    ax1.set_xlim(V_MIN - 1, V_MAX + 1)
    ax1.axvline(x=center, color='red', linestyle='--', linewidth=2, label=f'Center={center:.1f}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative distribution function (CDF)
    ax2 = axes[row, 1]
    cdf = np.cumsum(probs)
    ax2.step(support, cdf, where='mid', color='darkblue', linewidth=2, label='CDF')
    ax2.fill_between(support, 0, cdf, step='mid', alpha=0.3, color='darkblue')
    ax2.scatter(support, cdf, color='red', s=30, zorder=5)
    ax2.set_title(f'CDF (atoms={num_atoms})')
    ax2.set_xlabel('Value (z)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_xlim(V_MIN - 1, V_MAX + 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Support visualization with atom spacing
    ax3 = axes[row, 2]
    ax3.scatter(support, np.zeros_like(support), c=colors, s=100, edgecolors='black', zorder=5)
    for i, (z, p) in enumerate(zip(support, probs)):
        ax3.vlines(z, 0, p * 3, colors=colors[i], linewidth=2, alpha=0.7)
    ax3.set_title(f'Support & Spacing (atoms={num_atoms}, $\\Delta z={delta_z:.2f}$)')
    ax3.set_xlabel('Value (z)')
    ax3.set_ylabel('Scaled Probability')
    ax3.set_xlim(V_MIN - 1, V_MAX + 1)
    ax3.set_ylim(-0.1, 1.0)
    ax3.grid(True, alpha=0.3)

    # Add text info
    expected_value = np.sum(support * probs)
    ax3.text(0.02, 0.95, f'E[Z]={expected_value:.2f}\n$\\Delta z={delta_z:.3f}$',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('distributional_dqn_atoms.jpg', dpi=150, bbox_inches='tight')
plt.show()
