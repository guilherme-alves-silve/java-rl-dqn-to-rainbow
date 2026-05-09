"""
author: Guilherme Alves Silveira
generated with the help: Kimi K.2, DeepSeek and ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ==========================================================
# FIGURE SETUP
# ==========================================================
fig = plt.figure(figsize=(14, 12))
fig.subplots_adjust(hspace=0.32, bottom=0.08)

ax_params = plt.subplot(2, 1, 1)
ax_dist = plt.subplot(2, 1, 2)

# ==========================================================
# ANNEALING PARAMETERS
# ==========================================================
alpha_start = 1.0
alpha_end = 0.0

beta_start = 0.2
beta_end = 1.0

n_steps = 1000
steps = np.arange(n_steps)

alpha_values = alpha_start - (alpha_start - alpha_end) * steps / n_steps
alpha_values = np.maximum(alpha_values, alpha_end)

beta_values = beta_start + (beta_end - beta_start) * steps / n_steps
beta_values = np.minimum(beta_values, beta_end)

# ==========================================================
# TOP PLOT — α AND β EVOLUTION
# ==========================================================
ax_params.plot(
    steps,
    alpha_values,
    color='red',
    linewidth=2,
    alpha=0.3,
    label=r'$\alpha$ (Prioritization Strength)'
)

ax_params.plot(
    steps,
    beta_values,
    color='blue',
    linewidth=2,
    alpha=0.3,
    label=r'$\beta$ (IS Correction Strength)'
)

current_step_line, = ax_params.plot(
    [0, 0],
    [0, 1.2],
    color='purple',
    linewidth=2,
    linestyle='--',
    alpha=0.8
)

current_alpha_marker, = ax_params.plot(
    [],
    [],
    'ro',
    markersize=10,
    zorder=10,
    label='Current α'
)

current_beta_marker, = ax_params.plot(
    [],
    [],
    'bo',
    markersize=10,
    zorder=10,
    label='Current β'
)

ax_params.set_xlim(0, n_steps - 1)
ax_params.set_ylim(-0.05, 1.15)

ax_params.set_xlabel('Training Steps', fontsize=11)
ax_params.set_ylabel('Parameter Value', fontsize=11)

ax_params.set_title(
    r'PER Annealing — $\alpha$ and $\beta$ Evolution',
    fontsize=13,
    fontweight='bold'
)

ax_params.grid(True, alpha=0.3)
ax_params.legend(loc='upper right')

# ==========================================================
# PHASE ANNOTATIONS
# ==========================================================
phase_boundaries = [0, 250, 500, 750, 999]

phase_names = [
    'Phase 1\n0%',
    'Phase 2\n25%',
    'Phase 3\n50%',
    'Phase 4\n75%',
    'Phase 5\n100%'
]

phase_colors = [
    '#e74c3c',
    '#e67e22',
    '#f1c40f',
    '#2ecc71',
    '#3498db'
]

for boundary, name, color in zip(
    phase_boundaries,
    phase_names,
    phase_colors
):
    ax_params.axvline(
        x=boundary,
        color=color,
        linestyle='--',
        alpha=0.5,
        linewidth=1.5
    )

    ax_params.text(
        boundary + 20,
        1.05,
        name,
        fontsize=8,
        color=color,
        fontweight='bold'
    )

# crossover
diff = np.abs(alpha_values - beta_values)
cross_idx = np.argmin(diff)

ax_params.axvline(
    x=cross_idx,
    color='purple',
    linewidth=2,
    alpha=0.6
)

ax_params.text(
    cross_idx + 20,
    0.5,
    r'$\alpha \approx \beta$',
    fontsize=10,
    color='purple',
    fontweight='bold',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.7
    )
)

# ==========================================================
# BOTTOM PLOT — TRUE PER DISTRIBUTION
# ==========================================================
n_categories = 20
x_indices = np.arange(n_categories)

# Artificial priorities (higher on right side)
base_priorities = np.exp(np.linspace(0, 3, n_categories))
base_priorities = base_priorities / base_priorities.sum()

def generate_distribution(alpha):
    """
    True PER sampling distribution.
    Depends ONLY on alpha.
    """

    probs = base_priorities ** alpha
    probs = probs / probs.sum()

    return probs

def compute_is_weights(probs, beta):
    """
    Compute normalized IS weights.

    w_i = (N * P(i))^(-beta)
    """

    N = len(probs)

    weights = (N * probs) ** (-beta)
    weights = weights / weights.max()

    return weights

# initial distribution
initial_dist = generate_distribution(alpha_values[0])

bars = ax_dist.bar(
    x_indices,
    initial_dist,
    width=0.8,
    alpha=0.85
)

ax_dist.set_ylim(0, 0.16)
ax_dist.set_xlim(-0.5, n_categories - 0.5)

ax_dist.set_ylabel(r'Sampling Probability $P(i)$', fontsize=11)

ax_dist.set_xlabel(
    r'Transition Index $i$  $\rightarrow$  Higher TD-Error',
    fontsize=11
)

ax_dist.set_title(
    'PER Sampling Distribution',
    fontsize=12,
    fontweight='bold'
)

ax_dist.grid(axis='y', alpha=0.3, linestyle='--')

# x ticks
tick_positions = np.arange(0, n_categories)

ax_dist.set_xticks(tick_positions)

ax_dist.set_xticklabels(
    [f'$i_{{{i+1}}}$' for i in tick_positions],
    rotation=45,
    ha='right'
)

# High-error region highlight
high_error_region = ax_dist.axvspan(
    n_categories - 5,
    n_categories - 0.5,
    color='red'
)

high_error_text = ax_dist.text(
    n_categories - 3,
    0.14,
    'higher\nTD-error',
    fontsize=8,
    ha='center',
    color='darkred',
    fontweight='bold',
    alpha=1.0
)

# ==========================================================
# TEXT BOXES
# ==========================================================
param_text = ax_dist.text(
    0.02,
    0.95,
    '',
    transform=ax_dist.transAxes,
    fontsize=9,
    fontweight='bold',
    va='top',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.9
    )
)

phase_indicator = ax_dist.text(
    0.98,
    0.95,
    '',
    transform=ax_dist.transAxes,
    fontsize=8,
    ha='right',
    va='top',
    bbox=dict(
        boxstyle='round',
        facecolor='lightgray',
        alpha=0.8
    )
)

# ==========================================================
# PROGRESS BAR
# ==========================================================
progress_rect = Rectangle(
    (0.05, 0.01),
    0,
    0.015,
    transform=fig.transFigure,
    facecolor='steelblue',
    alpha=0.7,
    zorder=100
)

fig.patches.append(progress_rect)

fig.text(
    0.05,
    -0.01,
    'Training Progress →',
    transform=fig.transFigure,
    fontsize=8
)

# ==========================================================
# ANIMATION UPDATE
# ==========================================================
def update(frame):

    stride = max(1, n_steps // 200)
    current_step = min(frame * stride, n_steps - 1)

    current_alpha = alpha_values[current_step]
    current_beta = beta_values[current_step]

    # Fade high-error emphasis
    fade_alpha = current_alpha

    fade_alpha = current_alpha ** 2

    # Fade text
    high_error_text.set_alpha(fade_alpha)
    high_error_region.set_alpha(0.12 * fade_alpha)

    # ----------------------------------------------
    # update top plot
    # ----------------------------------------------
    current_step_line.set_data(
        [current_step, current_step],
        [0, 1.2]
    )

    current_alpha_marker.set_data(
        [current_step],
        [current_alpha]
    )

    current_beta_marker.set_data(
        [current_step],
        [current_beta]
    )

    # ----------------------------------------------
    # TRUE PER DISTRIBUTION
    # depends ONLY on alpha
    # ----------------------------------------------
    dist = generate_distribution(current_alpha)

    # IS weights depend on beta
    is_weights = compute_is_weights(dist, current_beta)

    # ----------------------------------------------
    # phase descriptions
    # ----------------------------------------------
    if current_step < 200:

        phase = 'Phase 1 — Strong Prioritization'
        color = '#e74c3c'

        desc = (
            'α is HIGH\n'
            'β is LOW\n\n'
            'Sampling strongly favors\n'
            'high TD-error transitions.\n'
            'Very little IS correction.'
        )

    elif current_step < 400:

        phase = 'Phase 2 — Priority Dominant'
        color = '#e67e22'

        desc = (
            'High-priority transitions\n'
            'still dominate replay.\n\n'
            'IS correction begins\n'
            'to increase.'
        )

    elif current_step < 600:

        phase = 'Phase 3 — Balanced'
        color = '#f1c40f'

        desc = (
            'Sampling remains biased,\n'
            'but IS correction is now\n'
            'significant.'
        )

    elif current_step < 800:

        phase = 'Phase 4 — Strong IS Correction'
        color = '#2ecc71'

        desc = (
            'Replay still uses PER,\n'
            'but gradients are strongly\n'
            'corrected by IS weights.'
        )

    else:

        phase = 'Phase 5 — Near Uniform Replay'
        color = '#3498db'

        desc = (
            'α approaches zero.\n'
            'Replay distribution becomes\n'
            'nearly uniform.\n\n'
            'β ≈ 1.0'
        )


    # ----------------------------------------------
    # update bars
    # ----------------------------------------------
    for i, (bar, p, w) in enumerate(zip(bars, dist, is_weights)):

        bar.set_height(p)

        # opacity visualizes IS correction
        # high beta -> stronger correction visibility
        opacity = 0.35 + 0.65 * w

        bar.set_alpha(opacity)

        # Original PER color
        original_color = np.array(
            plt.cm.plasma(i / n_categories)
        )

        # Final uniform color
        uniform_color = np.array(
            [0.3, 0.5, 0.9, 1.0]
        )

        # Blend factor
        # α=1 -> original colors
        # α=0 -> uniform color
        blend = current_alpha

        final_color = (
            blend * original_color
            +
            (1 - blend) * uniform_color
        )

        bar.set_color(final_color)

    # ----------------------------------------------
    # update text
    # ----------------------------------------------
    param_text.set_text(
        f"Step: {current_step}\n"
        f"α = {current_alpha:.3f}\n"
        f"β = {current_beta:.3f}\n\n"
        f"{desc}"
    )

    phase_indicator.set_text(phase)

    phase_indicator.set_bbox(
        dict(
            boxstyle='round',
            facecolor=color,
            alpha=0.3
        )
    )

    ax_dist.set_title(
        f'PER Sampling Distribution | '
        f'α={current_alpha:.3f} | '
        f'β={current_beta:.3f}',
        fontsize=11,
        fontweight='bold'
    )

    # progress bar
    progress = current_step / n_steps
    progress_rect.set_width(progress * 0.9)

    return (
        [current_step_line,
         current_alpha_marker,
         current_beta_marker]
        + list(bars)
    )

# ==========================================================
# CREATE ANIMATION
# ==========================================================
print("Creating animation...")

n_frames = 200

anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=50,
    repeat=True,
    blit=False
)

plt.tight_layout()

# ==========================================================
# SAVE GIF
# ==========================================================
print("Saving animation as GIF...")

try:

    anim.save(
        'annealing_animation_corrected.gif',
        writer='pillow',
        fps=20,
        dpi=100
    )

    print("Animation saved as 'annealing_animation_corrected.gif'")

except Exception as e:

    print(f"Error saving GIF: {e}")
    plt.show()

print("Done!")