"""
author: Guilherme Alves Silveira
generated with AI tools
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Colors (light theme)
colors = {
    'ground': '#8B7355',
    'pit': '#4A3728',
    'character': '#2E86AB',
    'jump_arc': '#1e88e5',
    'fall_line': '#e53935',
    'success': '#2ecc71',
    'danger': '#e74c3c'
}

# ========== LEFT: Jumping the pit ==========
ax1 = axes[0]
ax1.set_xlim(-1, 9)
ax1.set_ylim(0, 5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_facecolor('#F5F5F0')

# Ground
ground = Rectangle((-1, 0), 10, 0.3, color=colors['ground'], zorder=1)
ax1.add_patch(ground)

# Pit (hole in the ground)
pit = Rectangle((3, 0), 3, 0.3, color=colors['pit'], zorder=2, hatch='///')
ax1.add_patch(pit)
# Pit depth indication
pit_depth = Rectangle((3, -0.5), 3, 0.5, color='#3a2a1a', alpha=0.7, zorder=0)
ax1.add_patch(pit_depth)

# Character jumping (in mid-air, above the pit)
# Body
jump_body = Circle((5.5, 2.2), 0.35, color=colors['character'], zorder=3)
ax1.add_patch(jump_body)
# Head
jump_head = Circle((5.5, 2.65), 0.25, color='#FFD966', zorder=4)
ax1.add_patch(jump_head)
# Eyes
jump_eye = Circle((5.65, 2.7), 0.07, color='white', zorder=5)
ax1.add_patch(jump_eye)
# Arms outstretched (jumping pose)
ax1.plot([5.2, 4.7], [2.35, 2.1], color=colors['character'], lw=4, zorder=3)
ax1.plot([5.8, 6.3], [2.35, 2.1], color=colors['character'], lw=4, zorder=3)
# Legs
ax1.plot([5.35, 5.0], [1.95, 1.7], color=colors['character'], lw=4, zorder=3)
ax1.plot([5.65, 6.0], [1.95, 1.7], color=colors['character'], lw=4, zorder=3)

# Jump arc (trajectory)
x_jump = np.linspace(2, 7, 50)
y_jump = -0.8 * (x_jump - 4.5)**2 + 2.5
y_jump = np.maximum(y_jump, 0.3)
ax1.plot(x_jump, y_jump, '--', color=colors['jump_arc'], lw=2, alpha=0.6, zorder=1)

# Success indicator
ax1.text(5.5, 3.2, '✓ SUCCESS!', fontsize=12, color=colors['success'],
         ha='center', fontweight='bold', fontfamily='monospace')

# Labels
ax1.text(4.5, -0.8, 'PIT', fontsize=10, color='white', ha='center', fontweight='bold')
ax1.text(5.5, -0.3, 'JUMPING OVER', fontsize=9, color='#555', ha='center', fontstyle='italic')
ax1.set_title('JUMPING THE PIT', fontsize=14, fontweight='bold', color='#2E86AB', pad=15)

# Arrows showing direction
ax1.annotate('', xy=(2.5, 1.5), xytext=(1, 1.5),
             arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))
ax1.text(1.8, 1.7, 'RUN', fontsize=9, color='#999', ha='center')

# ========== RIGHT: Falling into the pit ==========
ax2 = axes[1]
ax2.set_xlim(-1, 9)
ax2.set_ylim(0, 5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_facecolor('#F5F5F0')

# Ground
ground2 = Rectangle((-1, 0), 10, 0.3, color=colors['ground'], zorder=1)
ax2.add_patch(ground2)

# Pit
pit2 = Rectangle((3, 0), 3, 0.3, color=colors['pit'], zorder=2, hatch='///')
ax2.add_patch(pit2)
pit_depth2 = Rectangle((3, -0.5), 3, 0.5, color='#3a2a1a', alpha=0.7, zorder=0)
ax2.add_patch(pit_depth2)

# Character falling (inside the pit, with motion lines)
fall_body = Circle((4.5, 0.2), 0.35, color=colors['character'], zorder=3)
ax2.add_patch(fall_body)
fall_head = Circle((4.5, 0.65), 0.25, color='#FFD966', zorder=4)
ax2.add_patch(fall_head)
# X eyes (showing distress)
ax2.plot([4.4, 4.6], [0.68, 0.72], color='black', lw=1.5, zorder=5)
ax2.plot([4.6, 4.4], [0.68, 0.72], color='black', lw=1.5, zorder=5)
# Arms flailing
ax2.plot([4.1, 3.7], [0.3, 0.0], color=colors['character'], lw=4, zorder=3)
ax2.plot([4.9, 5.3], [0.3, 0.0], color=colors['character'], lw=4, zorder=3)
# Legs
ax2.plot([4.3, 4.0], [-0.05, -0.35], color=colors['character'], lw=4, zorder=3)
ax2.plot([4.7, 5.0], [-0.05, -0.35], color=colors['character'], lw=4, zorder=3)

# Motion lines (falling effect)
for i in range(3):
    ax2.plot([4.5 + i*0.2, 4.5 + i*0.2], [1.5 - i*0.3, 1.0 - i*0.3],
             color='#999', lw=1.5, alpha=0.5)
    ax2.plot([4.5 - i*0.2, 4.5 - i*0.2], [1.5 - i*0.3, 1.0 - i*0.3],
             color='#999', lw=1.5, alpha=0.5)

# Danger indicator
ax2.text(4.5, 3.5, '✗ DEATH!', fontsize=12, color=colors['danger'],
         ha='center', fontweight='bold', fontfamily='monospace')

# Labels
ax2.text(4.5, -0.8, 'PIT', fontsize=10, color='white', ha='center', fontweight='bold')
ax2.text(4.5, -0.3, 'FALLING IN', fontsize=9, color='#555', ha='center', fontstyle='italic')
ax2.set_title('WALKING INTO THE PIT', fontsize=14, fontweight='bold', color='#e53935', pad=15)

# Arrow pointing to pit
ax2.annotate('', xy=(4.5, 1.8), xytext=(4.5, 3.2),
             arrowprops=dict(arrowstyle='->', color='#e53935', lw=2, alpha=0.7))
ax2.text(5.5, 2.5, 'WALK\nFORWARD', fontsize=9, color='#e53935', ha='left', alpha=0.8)

# ========== Add overall title ==========
fig.suptitle('TWO PATHS - TWO OUTCOMES', fontsize=16, fontweight='bold',
             color='#333', y=1.02)

plt.tight_layout()

OUT = "./graphics/pit_scenarios_illustration.jpg"
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor='white')
plt.close()
print(f"Saved → {OUT}")