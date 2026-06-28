"""
article3_depth.py v2
Fixed: label positioning, no overlaps, cleaner layout
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

RNG = np.random.default_rng(42)

# ── Ocean data ────────────────────────────────────────────────────────────────
depths       = np.linspace(0, 1000, 500)
temperatures = np.zeros(500)

surf_mask = depths < 50
temperatures[surf_mask] = (
    18 + 3*np.sin(np.linspace(0, 4*np.pi, surf_mask.sum()) + 0.5)
    + 0.8*RNG.standard_normal(surf_mask.sum())
)

therm_mask = (depths >= 50) & (depths < 300)
therm_d    = depths[therm_mask]
temperatures[therm_mask] = (
    18 - 13 * ((therm_d - 50) / 250)**0.6
    + 0.4*RNG.standard_normal(therm_mask.sum())
)

deep_mask = depths >= 300
deep_d    = depths[deep_mask]
temperatures[deep_mask] = (
    5 - 3 * ((deep_d - 300) / 700)**0.4
    + 0.15*RNG.standard_normal(deep_mask.sum())
)

temperatures = gaussian_filter(temperatures, sigma=3)

# ── Figure — wider to give room ───────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('white')

# ── Left: ocean profile ───────────────────────────────────────────────────────
ax = fig.add_axes([0.05, 0.08, 0.38, 0.78])
ax.set_facecolor('#f0f7ff')

# Regime backgrounds
ax.axhspan(0,   50,  color='#cce8f4', alpha=0.55, zorder=0)
ax.axhspan(50,  300, color='#90c4e0', alpha=0.40, zorder=0)
ax.axhspan(300, 1000,color='#1565a0', alpha=0.25, zorder=0)

# Temperature curve — shifted right so labels have room on left
ax.plot(temperatures, depths, color='#1565a0', lw=2.5, zorder=4)

# Regime boundaries
ax.axhline(50,  color='#333333', lw=1.0, linestyle='--', alpha=0.35)
ax.axhline(300, color='#333333', lw=1.0, linestyle='--', alpha=0.35)

# Regime label boxes — placed in right portion of plot, away from curve
label_x = 8.0   # far right — temperature axis goes 0-22

box_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor='#aaccdd', alpha=0.88)

# Surface label
ax.text(label_x, 25,
        '① Surface layer\n0 – 50 m\nWind-driven · fast-changing\nshort-range structure',
        color='#0d3b6e', fontsize=9, ha="left", va='center',
        bbox=box_props)

# Thermocline label
ax.text(label_x, 175,
        '② Thermocline\n50 – 300 m\nCurrent-driven · medium range\nentity-level structure',
        color='#0a2e5a', fontsize=9, ha="left", va='center',
        bbox=box_props)

# Deep label
ax.text(label_x, 650,
        '③ Deep ocean\n300 – 1000 m\nAbyssal circulation\nlong-range abstract structure',
        color='#0d3b6e', fontsize=9, ha="left", va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1565a0',
                  edgecolor='#0a2e5a', alpha=0.75))

# Temperature annotations — placed left of curve
ax.annotate('18°C  warm surface',
            xy=(18.2, 8),
            xytext=(19.5, 30),
            fontsize=8.5, color='#d4820a', va='center',
            arrowprops=dict(arrowstyle='->', color='#d4820a',
                           lw=0.9, alpha=0.7))

ax.annotate('5°C  cold deep',
            xy=(4.8, 850),
            xytext=(19.5, 850),
            fontsize=8.5, color='#1565a0', va='center', ha='right',
            arrowprops=dict(arrowstyle='->', color='#1565a0',
                           lw=0.9, alpha=0.7))

ax.set_xlim(0, 22)
ax.set_ylim(1000, 0)
ax.set_xlabel('temperature (°C)', fontsize=11, color='#444444')
ax.set_ylabel('depth (metres)', fontsize=11, color='#444444')
ax.tick_params(colors='#666666')
for sp in ax.spines.values():
    sp.set_edgecolor('#dddddd')
ax.grid(True, color='#dddddd', linewidth=0.5, alpha=0.6, axis='x')
ax.set_title('The stratified ocean\nthree regimes · three physics',
             color='#1565a0', fontsize=13, pad=12, fontweight='400')

# ── Right: transformer stack ──────────────────────────────────────────────────
ax2 = fig.add_axes([0.52, 0.08, 0.44, 0.78])
ax2.set_facecolor('#f8f9fa')
ax2.set_xlim(0, 10)
ax2.set_ylim(-1, 25)
ax2.axis('off')

n_layers  = 24
layer_h   = 22 / n_layers

# Regime colours
def regime_for(layer_num):
    if layer_num >= 20:        return 'surface'
    elif layer_num >= 10:      return 'thermocline'
    else:                      return 'deep'

regime_cfg = {
    'surface':     ('#cce8f4', '#0d3b6e', '0.70'),
    'thermocline': ('#90c4e0', '#0a2e5a', '0.60'),
    'deep':        ('#3a7ab5', '#f0f8ff', '0.85'),
}

# Key annotations per layer
layer_annotations = {
    23: 'punctuation · word boundaries',
    21: 'part-of-speech · local syntax',
    19: 'phrase structure',
    16: 'named entities · coreference',
    13: 'semantic roles · relationships',
    10: 'discourse structure',
    7:  'factual knowledge · world model',
    4:  'abstract reasoning · analogy',
    1:  'multi-step inference',
}

for layer_i in range(n_layers):
    y = layer_i * layer_h + 1.0
    layer_num = layer_i   # 0 = bottom (deep), 23 = top (surface)

    regime = regime_for(layer_num)
    bg_col, txt_col, _ = regime_cfg[regime]

    rect = mpatches.FancyBboxPatch(
        (0.3, y + 0.04), 9.4, layer_h - 0.08,
        boxstyle='round,pad=0.04',
        facecolor=bg_col, edgecolor='white',
        linewidth=0.6, alpha=0.88, zorder=3
    )
    ax2.add_patch(rect)

    # Layer number — left side
    ax2.text(0.6, y + layer_h/2,
             f'L{layer_num}',
             va='center', fontsize=6.5, color=txt_col, alpha=0.75)

    # Annotation — right side for key layers
    if layer_num in layer_annotations:
        ax2.text(5.0, y + layer_h/2,
                 layer_annotations[layer_num],
                 va='center', ha='center',
                 fontsize=7.5, color=txt_col, alpha=0.90)

# Regime bracket labels — outside right edge
# Surface bracket
surf_y_bot = regime_for.__code__  # placeholder

surf_bot = 20 * layer_h + 1.0
surf_top = 24 * layer_h + 1.0
therm_bot = 10 * layer_h + 1.0
deep_bot  = 0  * layer_h + 1.0

ax2.annotate('', xy=(9.85, surf_bot), xytext=(9.85, surf_top),
             arrowprops=dict(arrowstyle='<->', color='#0d3b6e',
                            lw=1.4, alpha=0.65))
ax2.text(10.05, (surf_bot+surf_top)/2,
         '① Surface\n(syntax)',
         va='center', fontsize=9, color='#0d3b6e', fontweight='500')

ax2.annotate('', xy=(9.85, therm_bot), xytext=(9.85, surf_bot),
             arrowprops=dict(arrowstyle='<->', color='#0a2e5a',
                            lw=1.4, alpha=0.65))
ax2.text(10.05, (therm_bot+surf_bot)/2,
         '② Mid layers\n(semantics)',
         va='center', fontsize=9, color='#0a2e5a', fontweight='500')

ax2.annotate('', xy=(9.85, deep_bot), xytext=(9.85, therm_bot),
             arrowprops=dict(arrowstyle='<->', color='#1565a0',
                            lw=1.4, alpha=0.65))
ax2.text(10.05, (deep_bot+therm_bot)/2,
         '③ Deep layers\n(reasoning)',
         va='center', fontsize=9, color='#1565a0', fontweight='500')

# Input / output
ax2.text(5.0, 0.3, 'token embeddings  (input)',
         ha='center', fontsize=9, color='#888888', style='italic')
ax2.text(5.0, 24.3, 'next token prediction  (output)',
         ha='center', fontsize=9, color='#888888', style='italic')

ax2.set_title('The stratified transformer\nthree regimes · same physics',
              color='#1565a0', fontsize=13, pad=12, fontweight='400')

# ── Titles ────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.98,
         'The ocean and the transformer share the same depth structure',
         ha='center', fontsize=15, color='#1a1a2e', fontweight='400')
fig.text(0.50, 0.94,
         'Surface dynamics are fast and local  ·  '
         'deep dynamics are slow and abstract  ·  the physics is the same',
         ha='center', fontsize=10, color='#555555')

fig.add_artist(plt.Line2D([0.50, 0.50], [0.06, 0.90],
               transform=fig.transFigure,
               color='#dddddd', linewidth=1.2))

os.makedirs('/home/claude/outputs', exist_ok=True)
out = '/mnt/user-data/outputs/article3_depth.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {out}")
