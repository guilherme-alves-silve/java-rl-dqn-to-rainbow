"""

Kullback-Leibler Divergence comparing the distributions for the article.

author: Guilherme Alves Silveira
generated with AI tools
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# ── data ──────────────────────────────────────────────────────────────────────
states = ['A', 'B', 'C', 'D', 'E']
P = np.array([0, 0.5, 0.35, 0.15, 0.0])
Q = np.array([0, 5/11, 2.5/11, 2.5/11, 1/11])

kl_contrib = np.array([
    p * np.log2(p / q) if p > 0 and q > 0 else 0.0
    for p, q in zip(P, Q)
])

overlap = np.minimum(P, Q)
p_only  = np.maximum(0, P - Q)
q_only  = np.maximum(0, Q - P)

# ── style: plain matplotlib default ───────────────────────────────────────────
plt.style.use('default')

prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
COL_P   = prop_cycle[0]   # C0 blue
COL_Q   = prop_cycle[1]   # C1 orange
COL_POS = prop_cycle[2]   # C2 green
COL_NEG = prop_cycle[3]   # C3 red
COL_OVERLAP = prop_cycle[7] if len(prop_cycle) > 7 else '#aaaaaa'

fig = plt.figure(figsize=(11, 14))
fig.suptitle('KL Divergence Between Two Markov Systems', fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.965,
         r'$D_{KL}(P\,\|\,Q)=\sum_{x}P(x)\,\log_2\!\frac{P(x)}{Q(x)}$'
         r'   —   $P$ = System 1,  $Q$ = System 2',
         fontsize=11, ha='center', va='top', style='italic')

gs = gridspec.GridSpec(4, 1, figure=fig,
                       top=0.92, bottom=0.06,
                       hspace=0.6,
                       height_ratios=[1.1, 1.0, 1.0, 1.0])

x = np.arange(len(states))
w = 0.35

# ── 1. stationary distributions ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
b1 = ax1.bar(x - w/2, P, width=w, color=COL_P, label='$P$ — System 1')
b2 = ax1.bar(x + w/2, Q, width=w, color=COL_Q, label='$Q$ — System 2')

for bar, val in zip(b1, P):
    if val > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(b2, Q):
    if val > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(states, fontsize=11)
ax1.set_ylabel('Probability')
ax1.set_ylim(0, 0.65)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title('Stationary Distributions per State', fontsize=12, loc='left')

# ── 2. KL contributions ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
colors_kl = [COL_POS if v >= 0 else COL_NEG for v in kl_contrib]
bars_kl = ax2.bar(x, kl_contrib, width=0.5, color=colors_kl)

for bar, val in zip(bars_kl, kl_contrib):
    if val != 0:
        ypos = val + 0.004 if val > 0 else val - 0.012
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                 f'{val:+.4f}', ha='center',
                 va='bottom' if val > 0 else 'top', fontsize=9)

ax2.axhline(0, color='black', linewidth=0.8)
total = kl_contrib.sum()
ax2.text(0.98, 0.95, f'$D_{{KL}} = {total:.4f}$ bits',
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                   edgecolor='gray', linewidth=0.8))

legend_els = [mpatches.Patch(facecolor=COL_POS, label='Positive (P overestimates Q)'),
              mpatches.Patch(facecolor=COL_NEG, label='Negative (P underestimates Q)')]
ax2.legend(handles=legend_els, loc='upper left', fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels(states, fontsize=11)
ax2.set_ylabel('bits')
ax2.set_ylim(-0.14, 0.30)
ax2.set_title(r'Per-State Contribution to $D_{KL}(P\,\|\,Q)$',
              fontsize=12, loc='left')

# ── 3. overlap ────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.bar(x, overlap, width=0.5, color=COL_OVERLAP, label='Overlap — min(P, Q)')
ax3.bar(x, p_only,  width=0.5, color=COL_P, bottom=overlap, label='P excess')
ax3.bar(x, q_only,  width=0.5, color=COL_Q, bottom=overlap, label='Q excess')

ax3.set_xticks(x)
ax3.set_xticklabels(states, fontsize=11)
ax3.set_ylabel('Probability')
ax3.set_ylim(0, 0.65)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax3.legend(loc='upper right', fontsize=9)
ax3.set_title('Distribution Overlap per State', fontsize=12, loc='left')

# ── 4. entropy summary ────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
metrics    = ['H(P)\nSystem 1', 'H(Q)\nSystem 2', r'$D_{KL}(P\,\|\,Q)$']
values     = [0.4406, 0.6752, 0.1969]
bar_colors = [COL_P, COL_Q, COL_POS]

for i, (label, val, col) in enumerate(zip(metrics, values, bar_colors)):
    ax4.bar(i, val, width=0.45, color=col)
    ax4.text(i, val + 0.012, f'{val:.4f} bits',
             ha='center', va='bottom', fontsize=10)

ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(metrics, fontsize=11)
ax4.set_ylabel('bits')
ax4.set_ylim(0, 0.85)
ax4.set_title('Entropy Rate and KL Divergence Summary', fontsize=12, loc='left')

ax4.text(0.5, -0.28,
         r'$D_{KL}$ is asymmetric: $D_{KL}(P\,\|\,Q)\neq D_{KL}(Q\,\|\,P)$.  '
         r'Result measures the cost of encoding System 1 sequences using System 2 as the reference model.',
         transform=ax4.transAxes, ha='center', va='top',
         fontsize=9, style='italic')

OUT = "./graphics/kl_divergence.jpg"
plt.savefig(OUT, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
