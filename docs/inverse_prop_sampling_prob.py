"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PER IMPORTANCE SAMPLING WEIGHT DEMONSTRATION
# ==========================================================

# Replay buffer size
N = 1000

# Full IS correction
beta = 1.0

# ----------------------------------------------------------
# Generate probability range
# ----------------------------------------------------------
# Avoid zero because of inverse function

p_values = np.linspace(0.0005, 0.1, 1000)

# IS weights
weights = (N * p_values) ** (-beta)

# ----------------------------------------------------------
# Example probabilities
# ----------------------------------------------------------

example_probs = np.array([
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.10
])

example_weights = (N * example_probs) ** (-beta)

# ----------------------------------------------------------
# Create figure
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

# Main curve
ax.plot(
    p_values,
    weights,
    linewidth=3,
    label=r'$w_i^{IS} = (N \cdot P(i))^{-\beta}$'
)

# Example points
ax.scatter(
    example_probs,
    example_weights,
    s=120,
    zorder=10
)

# ----------------------------------------------------------
# Annotate examples
# ----------------------------------------------------------

labels = [
    "small prob",
    "small prob",
    "medium prob",
    "medium prob",
    "high prob",
    "high prob"
]

for p, w, label in zip(example_probs, example_weights, labels):

    annotation = (
        f"P(i)={p:.4f}\n"
        f"w={w:.3f}\n"
        f"{label}"
    )

    ax.annotate(
        annotation,
        xy=(p, w),
        xytext=(15, 10),
        textcoords='offset points',
        fontsize=9,
        bbox=dict(
            boxstyle='round',
            alpha=0.85
        )
    )

# ----------------------------------------------------------
# Styling
# ----------------------------------------------------------

ax.set_title(
    'PER Importance Sampling Weights\n'
    'Smaller Sampling Probability → Larger IS Weight',
    fontsize=15,
    fontweight='bold'
)

ax.set_xlabel(
    r'Sampling Probability $P(i)$',
    fontsize=12
)

ax.set_ylabel(
    r'Importance Sampling Weight $w_i^{IS}$',
    fontsize=12
)

ax.grid(True, alpha=0.3)

ax.legend(fontsize=11)

# Log scale helps visualize inverse behavior clearly
ax.set_yscale('log')

# ----------------------------------------------------------
# Extra explanatory text
# ----------------------------------------------------------

explanation = (
    r"$w_i^{IS} = (N \cdot P(i))^{-\beta}$" "\n\n"
    "High replay probability:\n"
    "→ sampled frequently\n"
    "→ SMALL correction weight\n\n"
    "Low replay probability:\n"
    "→ sampled rarely\n"
    "→ LARGE correction weight"
)

ax.text(
    0.065,
    max(weights) * 0.7,
    explanation,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.9
    )
)

# ----------------------------------------------------------
# Show values in terminal
# ----------------------------------------------------------

print("=" * 60)
print("PER Importance Sampling Examples")
print("=" * 60)

for p, w in zip(example_probs, example_weights):

    print(
        f"P(i) = {p:<8.4f} "
        f"-> "
        f"w_i^IS = {w:.6f}"
    )

print("=" * 60)

# ----------------------------------------------------------
# Save + show
# ----------------------------------------------------------

plt.tight_layout()

plt.savefig(
    'graphics/per_is_inverse_relationship.jpg',
    dpi=150,
    bbox_inches='tight'
)

plt.show()
