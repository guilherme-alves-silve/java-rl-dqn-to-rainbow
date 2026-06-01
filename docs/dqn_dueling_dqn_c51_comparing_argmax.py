import numpy as np
import matplotlib.pyplot as plt

"""
Comparison:
DQN vs Dueling DQN vs C51

Goal:
Show how each architecture represents action values.

author: Guilherme Alves Silveira
generated with AI tools
"""

# Actions
actions = ["LEFT", "RIGHT", "JUMP"]

# ---------------------------------------------------------
# DQN
# Direct scalar Q-values
# ---------------------------------------------------------

dqn_q = np.array([1.2, 2.8, 1.7])

# ---------------------------------------------------------
# Dueling DQN
#
# Q(s,a) = V(s) + A(s,a)
#
# We'll simulate:
# V(s) = 2.0
# Advantages = [-0.6, +0.8, -0.2]
# ---------------------------------------------------------

V = 2.0
advantages = np.array([-0.6, 0.8, -0.2])

dueling_q = V + advantages

# ---------------------------------------------------------
# C51
#
# Distribution over returns
# Q(s,a) = Σ z_i * p_i
# ---------------------------------------------------------

z = np.array([-2, -1, 0, 1, 2])

c51_distributions = {
    "LEFT":  np.array([0.30, 0.30, 0.20, 0.15, 0.05]),
    "RIGHT": np.array([0.05, 0.10, 0.15, 0.30, 0.40]),
    "JUMP":  np.array([0.15, 0.20, 0.30, 0.20, 0.15]),
}

c51_q = []

for a in actions:
    q = np.sum(z * c51_distributions[a])
    c51_q.append(q)

c51_q = np.array(c51_q)

# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------

fig = plt.figure(figsize=(16, 10))

# =========================================================
# DQN
# =========================================================

ax1 = plt.subplot2grid((2, 3), (0, 0))

ax1.bar(actions, dqn_q)

best_dqn = np.argmax(dqn_q)

for i, q in enumerate(dqn_q):

    label = f"{q:.2f}"

    if i == best_dqn:
        label += "\nSELECTED ACTION"

    ax1.text(i, q + 0.05, label,
             ha='center',
             fontsize=10)

ax1.set_title("DQN\nDirect Q-values")
ax1.set_ylabel("Q(s,a)")
ax1.set_ylim(0, 3.5)

# =========================================================
# Dueling DQN
# =========================================================

ax2 = plt.subplot2grid((2, 3), (0, 1))

ax2.bar(actions, dueling_q)

best_dueling = np.argmax(dueling_q)

for i, q in enumerate(dueling_q):

    label = (
        f"V={V:.1f}\n"
        f"A={advantages[i]:+.1f}\n"
        f"Q={q:.2f}"
    )

    if i == best_dueling:
        label += "\nSELECTED ACTION"

    ax2.text(i, q + 0.05,
             label,
             ha='center',
             fontsize=9)

ax2.set_title("Dueling DQN\nValue + Advantage")
ax2.set_ylabel("Q(s,a)")
ax2.set_ylim(0, 3.5)

# =========================================================
# C51
# =========================================================

for idx, action in enumerate(actions):

    ax = plt.subplot2grid((2, 3), (1, idx))

    probs = c51_distributions[action]

    ax.bar(z, probs)

    q = c51_q[idx]

    title = (
        f"C51 - {action}\n"
        f"$Q = \\sum z_i p_i = {q:.2f}$"
    )

    if idx == np.argmax(c51_q):
        title += "\nSELECTED ACTION"

    ax.set_title(title, fontsize=10)

    ax.set_xlabel("Atoms $z_i$")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.5)

# ---------------------------------------------------------
# Global title
# ---------------------------------------------------------

plt.suptitle(
    "DQN vs Dueling DQN vs C51\n"
    "Different Representations of Action Values",
    fontsize=16
)

plt.tight_layout()
plt.subplots_adjust(
    wspace=0.2,   # horizontal spacing
    hspace=0.3    # vertical spacing
)

OUT = "./graphics/dqn_dueling_dqn_c51_comparing_argmax.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
