"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

node_style = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', linewidth=2.5)
random_node_style = dict(facecolor='#A8DADC', edgecolor='#1D3557', linewidth=3)
RED = '#E63946'

# ========== LEFT ==========
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Original Formulation\n(Noisy Nets)', fontsize=16, fontweight='bold', pad=20)

ax1.add_patch(FancyBboxPatch((1, 6.5), 1.5, 1.2, **node_style))
ax1.text(1.75, 7.1, r'$\Phi$', ha='center', va='center', fontsize=20, fontweight='bold')

ax1.add_patch(FancyBboxPatch((1, 4.0), 1.5, 1.2, **node_style))
ax1.text(1.75, 4.6, r'$\mathbf{x}$', ha='center', va='center', fontsize=18, fontweight='bold')

ax1.add_patch(Circle((4.5, 5.5), 0.9, **random_node_style))
ax1.text(4.5, 5.5, r'$\mathbf{z}$', ha='center', va='center', fontsize=18, fontweight='bold', color='#1D3557')

ax1.add_patch(FancyBboxPatch((7.0, 5.0), 1.5, 1.2, **node_style))
ax1.text(7.75, 5.6, r'$\mathbf{t}$', ha='center', va='center', fontsize=18, fontweight='bold')

# feedforward
ax1.annotate('', xy=(3.62, 5.82), xytext=(2.5, 7.1),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
ax1.annotate('', xy=(3.62, 5.18), xytext=(2.5, 4.6),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
ax1.annotate('', xy=(7.0, 5.6), xytext=(5.4, 5.5),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# legend
ly = 1.4
ax1.add_patch(FancyBboxPatch((0.8, ly), 1.3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor='black', linewidth=2))
ax1.text(1.45, ly+0.4, 'Deterministic\nNode', ha='center', va='center', fontsize=9)
ax1.add_patch(Circle((3.8, ly+0.4), 0.38, facecolor='#A8DADC', edgecolor='#1D3557', linewidth=2))
ax1.text(3.8, ly+0.4, 'z', ha='center', va='center', fontsize=11, color='#1D3557')
ax1.text(3.8, ly-0.25, 'Random\nNode', ha='center', va='center', fontsize=9)
ax1.annotate('', xy=(6.8, ly+0.4), xytext=(5.8, ly+0.4),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.text(6.8, ly-0.25, 'Feedforward', ha='center', va='center', fontsize=9)

# ========== RIGHT ==========
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Reparameterized Formulation\n(Noisy Nets with Factorized Noise)',
              fontsize=16, fontweight='bold', pad=20)

# nodes
ax2.add_patch(FancyBboxPatch((0.8, 7.4), 1.5, 1.2, **node_style))
ax2.text(1.55, 8.0, r'$\Phi$', ha='center', va='center', fontsize=20, fontweight='bold')

ax2.add_patch(FancyBboxPatch((0.8, 5.4), 1.5, 1.2, **node_style))
ax2.text(1.55, 6.0, r'$\mathbf{x}$', ha='center', va='center', fontsize=18, fontweight='bold')

ax2.add_patch(Circle((1.55, 3.4), 0.9, **random_node_style))
ax2.text(1.55, 3.4, r'$\boldsymbol{\epsilon}$', ha='center', va='center',
         fontsize=18, fontweight='bold', color='#1D3557')

ax2.add_patch(FancyBboxPatch((4.3, 5.4), 1.5, 1.2, **node_style))
ax2.text(5.05, 6.0, r'$\mathbf{z}$', ha='center', va='center', fontsize=18, fontweight='bold')

ax2.add_patch(FancyBboxPatch((7.4, 5.4), 1.5, 1.2, **node_style))
ax2.text(8.15, 6.0, r'$\mathbf{t}$', ha='center', va='center', fontsize=18, fontweight='bold')

# --- FEEDFORWARD arrows (black) ---
# Φ → z  (upper-left diagonal)
ax2.annotate('', xy=(4.3, 6.3), xytext=(2.3, 8.0),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
# x → z  (horizontal)
ax2.annotate('', xy=(4.3, 5.85), xytext=(2.3, 5.85),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
# ε → z  (lower-left diagonal)
ax2.annotate('', xy=(4.4, 5.55), xytext=(2.42, 3.72),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
# z → t  (horizontal)
ax2.annotate('', xy=(7.4, 6.0), xytext=(5.8, 6.0),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# --- BACKPROP arrows (red) — routed ABOVE feedforward to avoid overlap ---

# ∂t/∂z : from t LEFT side → z RIGHT side, routed above (y=7.0)
ax2.annotate('', xy=(5.8, 6.7), xytext=(7.4, 6.7),
             arrowprops=dict(arrowstyle='->', color=RED, lw=2.5,
                             connectionstyle='arc3,rad=0.0'))
ax2.text(6.6, 7.1, r'$\frac{\partial t}{\partial z}$',
         ha='center', va='bottom', fontsize=13, color=RED, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.95))

# ∂z/∂Φ : from z TOP → Φ TOP, routed along top (y=9.1) well above feedforward
ax2.annotate('', xy=(1.55, 8.9), xytext=(5.05, 8.9),
             arrowprops=dict(arrowstyle='->', color=RED, lw=2.5))
# vertical stubs connecting to the curved route
ax2.plot([5.05, 5.05], [6.6, 8.9], color=RED, lw=2.5)   # z top → route
ax2.plot([1.55, 1.55], [8.62, 8.9], color=RED, lw=2.5)  # route → Φ top
ax2.text(3.3, 9.25, r'$\frac{\partial z}{\partial \Phi}$',
         ha='center', va='bottom', fontsize=13, color=RED, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.95))

# "Backprop" label + small left-pointing arrow (top-right, well clear of nodes)
ax2.text(9.1, 8.8, 'Backprop', ha='center', va='center', fontsize=12,
         color=RED, fontweight='bold')
ax2.annotate('', xy=(8.3, 8.4), xytext=(9.1, 8.6),
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.8,
                             connectionstyle='arc3,rad=-0.25'))

# No gradient to ε — clean X below ε + dashed stub
ax2.plot([1.55, 1.55], [2.42, 2.62], color=RED, lw=1.8, linestyle='--', alpha=0.7)
d = 0.16
cx, cy = 1.55, 2.28
ax2.plot([cx-d, cx+d], [cy-d, cy+d], color=RED, lw=2.5)
ax2.plot([cx-d, cx+d], [cy+d, cy-d], color=RED, lw=2.5)
ax2.text(2.05, 2.1, 'No gradient\nto ε', ha='left', va='center', fontsize=9,
         color=RED, style='italic')

# legend
ly = 1.0
ax2.add_patch(FancyBboxPatch((0.3, ly), 1.3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor='black', linewidth=2))
ax2.text(0.95, ly+0.4, 'Deterministic\nNode', ha='center', va='center', fontsize=9)
ax2.add_patch(Circle((3.1, ly+0.4), 0.38, facecolor='#A8DADC', edgecolor='#1D3557', linewidth=2))
ax2.text(3.1, ly+0.4, 'ε', ha='center', va='center', fontsize=11, color='#1D3557')
ax2.text(3.1, ly-0.2, 'Random\nNode', ha='center', va='center', fontsize=9)
ax2.annotate('', xy=(5.5, ly+0.4), xytext=(4.5, ly+0.4),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax2.text(5.5, ly-0.2, 'Feedforward', ha='center', va='center', fontsize=9)
ax2.annotate('', xy=(8.2, ly+0.4), xytext=(7.2, ly+0.4),
             arrowprops=dict(arrowstyle='->', color=RED, lw=2))
ax2.text(8.2, ly-0.2, 'Differentiation', ha='center', va='center', fontsize=9, color=RED)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('noisy_nets_backprop_explain.jpg', dpi=150, bbox_inches='tight')
print("Done")
