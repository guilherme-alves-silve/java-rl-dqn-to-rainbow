import numpy as np
import matplotlib.pyplot as plt

"""
C51 + PER Training Visualization

Goal:
Show the relationship between:

  m_i  -> target distribution
  p_i  -> online prediction

and how cross-entropy / KL-like error emerges.

author: Guilherme Alves Silveira
generated with AI tools
"""

# ---------------------------------------------------------
# Fixed atom support
# ---------------------------------------------------------

z = np.array([-2, -1, 0, 1, 2])

# ---------------------------------------------------------
# Target distribution m
#
# Constructed from:
# target network
# + Bellman update
# + projection
# ---------------------------------------------------------

m = np.array([0.05, 0.10, 0.20, 0.35, 0.30])

# ---------------------------------------------------------
# Online prediction p_theta(s,a)
#
# Current network prediction
# ---------------------------------------------------------

p = np.array([0.25, 0.30, 0.20, 0.15, 0.10])

# ---------------------------------------------------------
# Cross-entropy terms
#
# L = -Σ m_i ln(p_i)
# ---------------------------------------------------------

cross_entropy_terms = -m * np.log(p)

total_loss = np.sum(cross_entropy_terms)

# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# =========================================================
# Plot 1 — Target distribution m
# =========================================================

axes[0].bar(z, m)

axes[0].set_title(
    "Target Distribution $m_i$\n"
    "(from target network + projection)",
    fontsize=12
)

axes[0].set_xlabel("Atoms $z_i$")
axes[0].set_ylabel("Probability")
axes[0].set_ylim(0, 0.5)

# Show values
for x, y in zip(z, m):
    axes[0].text(x, y + 0.02,
                 f"{y:.2f}",
                 ha='center')

# =========================================================
# Plot 2 — Online prediction p
# =========================================================

axes[1].bar(z, p)

axes[1].set_title(
    "Predicted Distribution $p_i(s,a;\\theta)$\n"
    "(online network)",
    fontsize=12
)

axes[1].set_xlabel("Atoms $z_i$")
axes[1].set_ylabel("Probability")
axes[1].set_ylim(0, 0.5)

# Show values
for x, y in zip(z, p):
    axes[1].text(x, y + 0.02,
                 f"{y:.2f}",
                 ha='center')

# =========================================================
# Plot 3 — Cross-entropy contribution
# =========================================================

axes[2].bar(z, cross_entropy_terms)

axes[2].set_title(
    "Cross-Entropy Contribution\n"
    "$-m_i \\ln(p_i)$",
    fontsize=12
)

axes[2].set_xlabel("Atoms $z_i$")
axes[2].set_ylabel("Loss Contribution")

# Show values
for x, y in zip(z, cross_entropy_terms):
    axes[2].text(x, y + 0.01,
                 f"{y:.2f}",
                 ha='center')

# =========================================================
# Global title
# =========================================================

plt.suptitle(
    "C51 Training: Target Distribution vs Online Prediction",
    fontsize=16
)

# ---------------------------------------------------------
# Layout spacing
# ---------------------------------------------------------

plt.tight_layout()

plt.subplots_adjust(
    wspace=0.4,
    top=0.80
)

# ---------------------------------------------------------
# Print exact math
# ---------------------------------------------------------

print("\nCross-Entropy Computation:\n")

terms = []

for zi, mi, pi in zip(z, m, p):

    value = -mi * np.log(pi)

    expression = (
        f"-({mi:.2f}) * ln({pi:.2f})"
        f" = {value:.4f}"
    )

    terms.append(expression)

    print(f"z = {zi}: {expression}")

print("\nTotal Loss:")
print(f"L = {total_loss:.4f}")

# ---------------------------------------------------------
# Show figure
# ---------------------------------------------------------

OUT = "./graphics/c51_loss_cross_entropy_kl.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
