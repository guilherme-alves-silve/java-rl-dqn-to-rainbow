"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import matplotlib.pyplot as plt
import numpy as np

V_MIN, V_MAX, N = -10, 10, 51
atoms = np.linspace(V_MIN, V_MAX, N)

# Jumping a pit: distribution skewed right (positive rewards)
mu_jump, sigma_jump = 5.5, 1.8
p_jump = np.exp(-0.5 * ((atoms - mu_jump) / sigma_jump) ** 2)
p_jump /= p_jump.sum()

# Going straight into the pit: distribution skewed hard left (death)
mu_death, sigma_death = -7.5, 1.4
p_death = np.exp(-0.5 * ((atoms - mu_death) / sigma_death) ** 2)
p_death /= p_death.sum()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
fig.patch.set_facecolor('#0f0f1a')

colors = {'jump': '#00e5ff', 'death': '#ff4757'}
bg = '#0f0f1a'
panel = '#1a1a2e'

for ax in axes:
    ax.set_facecolor(panel)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

# --- Left: Jumping the pit ---
ax1 = axes[0]
ax1.bar(atoms, p_jump, width=(V_MAX - V_MIN) / (N - 1) * 0.85,
        color=colors['jump'], alpha=0.85, zorder=3)
ax1.axvline(np.dot(atoms, p_jump), color='white', lw=1.5,
            linestyle='--', alpha=0.7, label=f'E[Z] = {np.dot(atoms, p_jump):.1f}')
ax1.set_title('Jumping the pit', color='white', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Return $z_i$', color='#aaaacc', fontsize=11)
ax1.set_ylabel('Probability $p_i$', color='#aaaacc', fontsize=11)
ax1.tick_params(colors='#aaaacc')
ax1.legend(framealpha=0, labelcolor='white', fontsize=10)
ax1.set_xlim(V_MIN - 0.5, V_MAX + 0.5)
ax1.grid(axis='y', color='#333355', linewidth=0.6, zorder=0)

# --- Right: Walking into the pit ---
ax2 = axes[1]
ax2.bar(atoms, p_death, width=(V_MAX - V_MIN) / (N - 1) * 0.85,
        color=colors['death'], alpha=0.85, zorder=3)
ax2.axvline(np.dot(atoms, p_death), color='white', lw=1.5,
            linestyle='--', alpha=0.7, label=f'E[Z] = {np.dot(atoms, p_death):.1f}')
ax2.set_title('Walking into the pit', color='white', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Return $z_i$', color='#aaaacc', fontsize=11)
ax2.set_ylabel('Probability $p_i$', color='#aaaacc', fontsize=11)
ax2.tick_params(colors='#aaaacc')
ax2.legend(framealpha=0, labelcolor='white', fontsize=10)
ax2.set_xlim(V_MIN - 0.5, V_MAX + 0.5)
ax2.grid(axis='y', color='#333355', linewidth=0.6, zorder=0)

fig.suptitle('C51 — Return Distributions per Action', color='white',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

OUT = "./graphics/c51_pit_distributions.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
