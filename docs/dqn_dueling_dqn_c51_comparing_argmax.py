import numpy as np
import matplotlib.pyplot as plt

"""
Comparison:
DQN vs Dueling DQN vs C51

Goal:
Show how each architecture represents action values.
Note: C51 uses 8 atoms in this example; standard C51 uses 51.

author: Guilherme Alves Silveira
generated with AI tools
"""

# Actions — 6 to match architecture diagram
actions = ["LEFT", "RIGHT", "JUMP", "ATTACK", "DEFEND", "SELECT"]

# ---------------------------------------------------------
# DQN
# Direct scalar Q-values
# Output shape: (6,)
# ---------------------------------------------------------

dqn_q = np.array([1.2, 2.8, 1.7, 0.9, 1.5, 2.1])

# ---------------------------------------------------------
# Dueling DQN
#
# Q(s,a) = V(s) + A(s,a)
#
# V(s) = 2.0
# ---------------------------------------------------------

V = 2.0
advantages = np.array([-0.6, 0.8, -0.2, -1.1, -0.4, 0.3])
dueling_q = V + advantages

# ---------------------------------------------------------
# C51
#
# Distribution over returns
# Q(s,a) = Σ z_i * p_i
# 8 atoms in this example (standard C51 uses 51)
# Output shape: (6, 8) → after E[Z]: (6,)
# ---------------------------------------------------------

z = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])

c51_distributions = {
    "LEFT":   np.array([0.25, 0.25, 0.20, 0.15, 0.08, 0.04, 0.02, 0.01]),
    "RIGHT":  np.array([0.01, 0.02, 0.04, 0.08, 0.15, 0.20, 0.25, 0.25]),
    "JUMP":   np.array([0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.03, 0.02]),
    "ATTACK": np.array([0.02, 0.05, 0.10, 0.15, 0.25, 0.25, 0.12, 0.06]),
    "DEFEND": np.array([0.20, 0.25, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]),
    "SELECT": np.array([0.05, 0.08, 0.15, 0.22, 0.22, 0.15, 0.08, 0.05]),
}

# Normalize
for a in actions:
    c51_distributions[a] /= c51_distributions[a].sum()

c51_q = np.array([np.sum(z * c51_distributions[a]) for a in actions])

# ---------------------------------------------------------
# Figure layout:
# Row 0: DQN (col 0) | Dueling DQN (col 1) | shape info (col 2)
# Row 1: C51 all 6 actions as grouped bar (spans full width)
# ---------------------------------------------------------

fig = plt.figure(figsize=(18, 11))

x = np.arange(len(actions))

# =========================================================
# DQN
# =========================================================

ax1 = plt.subplot2grid((2, 3), (0, 0))
bars = ax1.bar(x, dqn_q, color="#4C72B0", width=0.6)
best_dqn = np.argmax(dqn_q)

for i, q in enumerate(dqn_q):
    label = f"{q:.2f}"
    if i == best_dqn:
        label += "\n★"
    ax1.text(i, q + 0.05, label, ha='center', fontsize=8)

ax1.set_title("DQN — Direct Q-values\nOutput shape: (6,)", fontsize=15)
ax1.set_ylabel("Q(s,a)")
ax1.set_xticks(x)
ax1.set_xticklabels(actions, rotation=30, ha='right', fontsize=8)
ax1.set_ylim(0, 3.8)

# =========================================================
# Dueling DQN
# =========================================================

ax2 = plt.subplot2grid((2, 3), (0, 1))
ax2.bar(x, dueling_q, color="#DD8452", width=0.6)
best_dueling = np.argmax(dueling_q)

for i, q in enumerate(dueling_q):
    label = f"A={advantages[i]:+.1f}\nQ={q:.2f}"
    if i == best_dueling:
        label += "\n★"
    ax2.text(i, q + 0.05, label, ha='center', fontsize=8)

ax2.set_title(f"Dueling DQN — V(s)={V:.1f} + Advantage\nOutput shape: (6,)", fontsize=15)
ax2.set_ylabel("Q(s,a)")
ax2.set_xticks(x)
ax2.set_xticklabels(actions, rotation=30, ha='right', fontsize=8)
ax2.set_ylim(0, 3.8)

# =========================================================
# C51 — grouped bar: all 6 actions, 8 atoms each
# =========================================================

ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)

n_atoms = len(z)
n_actions = len(actions)
group_width = 0.8
bar_width = group_width / n_atoms

colors = plt.cm.tab10(np.linspace(0, 0.6, n_atoms))

for atom_idx in range(n_atoms):
    offsets = x + (atom_idx - n_atoms / 2 + 0.5) * bar_width
    probs = [c51_distributions[a][atom_idx] for a in actions]
    ax4.bar(offsets, probs, width=bar_width,
            color=colors[atom_idx],
            label=f"$z_{atom_idx}$={z[atom_idx]:.1f}")

best_c51 = np.argmax(c51_q)

for i, a in enumerate(actions):
    q = c51_q[i]
    label = f"Q={q:.2f}"
    if i == best_c51:
        label += " ★"
    ax4.text(i, 0.33, label, ha='center', fontsize=8,
             bbox=dict(boxstyle="round,pad=0.2",
                       facecolor="white", alpha=0.7))

ax4.set_title(
    "C51 — Categorical Distribution per Action\n"
    "Output shape: (6, 8)  →  after $\\mathbb{E}[Z] = \\sum z_i p_i$: (6,)  [inference only]",
    fontsize=15
)
ax4.set_ylabel("Probability")
ax4.set_xticks(x)
ax4.set_xticklabels(actions, fontsize=10)
ax4.set_ylim(0, 0.45)
ax4.legend(title="Atoms $z_i$", loc='upper right',
           fontsize=7, ncol=4)

# ---------------------------------------------------------
# Global title
# ---------------------------------------------------------

plt.suptitle(
    "DQN vs Dueling DQN vs C51",
    fontsize=17
)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.5)

OUT = "./graphics/dqn_dueling_dqn_c51_comparing_argmax.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
