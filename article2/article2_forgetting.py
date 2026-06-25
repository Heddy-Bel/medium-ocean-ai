"""
article2_forgetting.py
----------------------
Article 2 illustration C: The Forgetting Landscape
Left panel:  2D heatmap — token position x conversation turn.
             Each cell: influence of past token on current prediction.
             OU decay curve overlaid as theoretical prediction.
Right panel: Three forgetting rates — long memory, typical LLM, short memory.

Run from article2/ folder:
  python3 article2_forgetting.py
Output: outputs/article2_forgetting.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

RNG = np.random.default_rng(42)

# ── Parameters ────────────────────────────────────────────────────────────────
N_TOKENS = 120
N_TURNS  = 60
LAM      = 0.055

# ── Forgetting landscape ──────────────────────────────────────────────────────
landscape = np.zeros((N_TURNS, N_TOKENS))
for i in range(N_TURNS):
    for j in range(N_TOKENS):
        dist = i - j * (N_TOKENS / N_TURNS)
        if dist >= 0:
            influence = np.exp(-LAM * dist)
            noise = 0.08 * RNG.standard_normal()
            if RNG.random() < 0.04:
                influence = min(1.0, influence + 0.3)
            landscape[i, j] = np.clip(influence + noise, 0, 1)

landscape = gaussian_filter(landscape, sigma=1.2)
landscape = np.clip(landscape, 0, 1)

# OU decay curve
tau_range = np.linspace(0, N_TOKENS, 300)
ou_decay  = np.exp(-LAM * tau_range * (N_TURNS / N_TOKENS))

# ── Colormap ──────────────────────────────────────────────────────────────────
memory_cmap = LinearSegmentedColormap.from_list('memory', [
    (0.00, '#f0f4f8'),
    (0.20, '#c8dff0'),
    (0.40, '#7ab8d8'),
    (0.60, '#2196c4'),
    (0.78, '#1565a0'),
    (0.90, '#2e8b57'),
    (1.00, '#1b4332'),
], N=512)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                          gridspec_kw={'width_ratios': [2.2, 1]})
fig.patch.set_facecolor('white')

# ── Panel 1: Landscape ────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#f0f4f8')

im = ax.imshow(landscape, cmap=memory_cmap, aspect='auto',
               origin='lower', vmin=0, vmax=1,
               extent=[0, N_TOKENS, 0, N_TURNS],
               interpolation='bicubic')

# OU decay overlay
ou_turn = ou_decay * N_TURNS
ax.plot(tau_range, ou_turn, color='#d4820a', lw=2.2,
        linestyle='--', alpha=0.90,
        label='OU decay  e^{−λτ}', zorder=5)

# Half-life line
half_life_token = np.log(2) / LAM * (N_TOKENS / N_TURNS)
ax.axvline(half_life_token, color='#333333', lw=1.0,
           linestyle=':', alpha=0.40, zorder=4)
ax.text(half_life_token + 1, 56,
        f'half-life\n({half_life_token:.0f} tokens)',
        color='#333333', fontsize=8, alpha=0.65, va='top')

# Annotations
ax.text(5, 52, 'still\nremembered',
        color='#1b4332', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white', edgecolor='#2e8b57', alpha=0.85))

ax.text(95, 8, 'forgotten',
        color='#666666', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white', edgecolor='#aaaaaa', alpha=0.85))

ax.annotate('occasional\nlong-range\nattention',
            xy=(85, 45), xytext=(68, 52),
            fontsize=8, color='#333333', alpha=0.70,
            arrowprops=dict(arrowstyle='->',
                           color='#333333', alpha=0.45, lw=0.8))

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('influence on current prediction',
               color='#444444', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='#666666')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#666666', fontsize=8)

ax.set_xlabel('token position in context', fontsize=11, color='#444444')
ax.set_ylabel('conversation turn', fontsize=11, color='#444444')
ax.tick_params(colors='#666666')
for sp in ax.spines.values():
    sp.set_edgecolor('#dddddd')

ax.set_title('The forgetting landscape\n'
             'How much influence does each past token still have?',
             color='#1565a0', fontsize=12, pad=10, fontweight='400')
ax.legend(loc='upper right', fontsize=9,
          facecolor='white', edgecolor='#dddddd',
          labelcolor='#d4820a')

# ── Panel 2: Three scenarios ──────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#f8f9fa')

tau = np.linspace(0, N_TOKENS, 300)
scenarios = [
    (0.025, '#1565a0', 'Long memory\n(λ = 0.025)', '--'),
    (0.055, '#2e8b57', 'Typical LLM\n(λ = 0.055)', '-'),
    (0.18,  '#d4820a', 'Short memory\n(λ = 0.18)',  ':'),
]

for lam_i, col, label, ls in scenarios:
    decay = np.exp(-lam_i * tau)
    ax2.plot(tau, decay, color=col, lw=2.2, linestyle=ls, label=label)
    hl = np.log(2) / lam_i
    ax2.axvline(hl, color=col, lw=0.6, alpha=0.25, linestyle=':')

ax2.axhline(0.5, color='#333333', lw=0.8, alpha=0.20, linestyle='--')
ax2.text(115, 0.52, '½', color='#333333', fontsize=9,
         alpha=0.50, ha='right')

ax2.axhspan(0, 0.2, color='#d62828', alpha=0.06)
ax2.text(60, 0.08, 'effectively\nforgotten',
         color='#d62828', fontsize=8, alpha=0.65, ha='center')

ax2.set_xlim(0, N_TOKENS); ax2.set_ylim(0, 1.05)
ax2.set_xlabel('tokens since context was set',
               fontsize=10, color='#444444')
ax2.set_ylabel('remaining influence', fontsize=10, color='#444444')
ax2.tick_params(colors='#666666')
for sp in ax2.spines.values():
    sp.set_edgecolor('#dddddd')
ax2.grid(True, color='#dddddd', linewidth=0.5, alpha=0.8)

ax2.set_title('Forgetting rate\nthree model types',
              color='#1565a0', fontsize=11, pad=10, fontweight='400')
ax2.legend(fontsize=8.5, loc='upper right',
           facecolor='white', edgecolor='#dddddd')

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.97,
         'Memory evaporates — the forgetting landscape of a language model',
         ha='center', fontsize=14, color='#1a1a2e', fontweight='400')
fig.text(0.50, 0.93,
         'Dark = forgotten · Bright = remembered · '
         'The OU decay curve predicts where the boundary falls',
         ha='center', fontsize=9, color='#555555')

plt.tight_layout(rect=[0, 0, 1, 0.92])

os.makedirs('outputs', exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'outputs', 'article2_forgetting.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {out}")
