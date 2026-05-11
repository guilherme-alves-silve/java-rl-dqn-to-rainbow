"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import matplotlib.pyplot as plt

gamma = 0.9
rewards = [1, 1, 1, 1, 10]
bootstrap_q = 5

td_1 = rewards[0] + gamma * bootstrap_q

n2 = rewards[0] + gamma * rewards[1] + (gamma**2) * bootstrap_q

n3 = (
    rewards[0]
    + gamma * rewards[1]
    + (gamma**2) * rewards[2]
    + (gamma**3) * bootstrap_q
)

n4 = (
    rewards[0]
    + gamma * rewards[1]
    + (gamma**2) * rewards[2]
    + (gamma**3) * rewards[3]
    + (gamma**4) * bootstrap_q
)

mc = (
    rewards[0]
    + gamma * rewards[1]
    + (gamma**2) * rewards[2]
    + (gamma**3) * rewards[3]
    + (gamma**4) * rewards[4]
)

methods = ["TD(1)", "2-step", "3-step", "4-step", "Monte Carlo"]
returns = [td_1, n2, n3, n4, mc]

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(methods, returns, marker='o')

ax.set_title("Return Target: TD vs n-step vs Monte Carlo")
ax.set_xlabel("Method")
ax.set_ylabel(r"Return Target $G_t$")
ax.grid(True)

for x, y in zip(methods, returns):
    ax.annotate(f"{y:.2f}", (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center')

plt.tight_layout()

plt.savefig(
    'td_nstep_montecarlo.jpg',
    dpi=150,
    bbox_inches='tight'
)
plt.show()
