import numpy as np
import matplotlib.pyplot as plt

"""
C51 greedy action selection visualization
===============================================
a* = argmax_a Σ_i z_i * p_i(s', a)

This script shows:
1. The categorical distributions for each action
2. The expected value computation
3. Which action is selected by argmax

author: Guilherme Alves Silveira
generated with AI tools
"""

# Fixed support atoms (z_i)
z = np.array([-2, -1, 0, 1, 2])

# Example probability distributions for 3 actions
# Each row = p_i(s', a)

p_action_0 = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
p_action_1 = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
p_action_2 = np.array([0.05, 0.10, 0.15, 0.30, 0.40])

distributions = [
    ("Action 0", p_action_0),
    ("Action 1", p_action_1),
    ("Action 2", p_action_2),
]

# ---------------------------------------------------------
# Compute expected values
# Q(s,a) = Σ z_i * p_i
# ---------------------------------------------------------

q_values = []

for name, p in distributions:
    q = np.sum(z * p)
    q_values.append(q)

# Greedy action
best_idx = np.argmax(q_values)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (ax, (name, p)) in enumerate(zip(axes, distributions)):

    q = q_values[idx]

    # Bar plot of categorical distribution
    ax.bar(z, p)

    # Title with expectation
    title = (
        f"{name}\n"
        f"$Q(s,a)=\\sum z_i p_i = {q:.2f}$"
    )

    # Mark best action
    if idx == best_idx:
        title += "\nSELECTED ACTION"

    ax.set_title(title, fontsize=12)

    ax.set_xlabel("Atom $z_i$")
    ax.set_ylabel("Probability $p_i$")
    ax.set_xticks(z)
    ax.set_ylim(0, 0.5)

OUT = "./graphics/c51_argmax_visual.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")

# ---------------------------------------------------------
# Print exact calculations
# ---------------------------------------------------------

print("\nExpected value calculations:\n")

for idx, (name, p) in enumerate(distributions):

    terms = [f"({z_i} × {p_i:.2f})"
             for z_i, p_i in zip(z, p)]

    expression = " + ".join(terms)

    print(f"{name}:")
    print(f"Q = {expression}")
    print(f"Q = {q_values[idx]:.2f}")

    if idx == best_idx:
        print("-> Selected by argmax")

    print()