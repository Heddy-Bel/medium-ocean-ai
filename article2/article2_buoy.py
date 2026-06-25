"""
article2_buoy.py
----------------
Article 2 illustration A: The Buoy
North Atlantic buoy SST anomaly — one year of OU dynamics.
Left panel: stylised North Atlantic map with buoy location.
Right panel: full year time series with storms, seasons, mean reversion.

Run from article2/ folder:
  python3 article2_buoy.py
Output: outputs/article2_buoy.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

RNG = np.random.default_rng(7)

# ── Simulate OU process ───────────────────────────────────────────────────────
n_days = 365
dt     = 1.0
lam    = 0.08
mu     = 0.0
sigma  = 0.38
x0     = 0.5

t = np.arange(n_days)
X = np.zeros(n_days)
X[0] = x0
for i in range(1, n_days):
    X[i] = X[i-1] - lam*(X[i-1]-mu)*dt + sigma*np.sqrt(dt)*RNG.standard_normal()

# ── Season config ─────────────────────────────────────────────────────────────
seasons = {
    'Winter\n(Jan–Mar)': (0,   89,  '#1565a0'),
    'Spring\n(Apr–Jun)': (90,  180, '#2e8b57'),
    'Summer\n(Jul–Sep)': (181, 272, '#f4a261'),
    'Autumn\n(Oct–Dec)': (273, 364, '#8b4513'),
}

storm_days   = [45,  112,         198,          287,        334]
storm_labels = ['Storm\nHercules','Low\npressure','Heat\nanomaly',
                'Storm\nBrian',   'Winter\ngale']

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('white')

# ── Left: map ─────────────────────────────────────────────────────────────────
ax_map = fig.add_axes([0.02, 0.08, 0.30, 0.82])
ax_map.set_facecolor('white')

ocean_cmap = LinearSegmentedColormap.from_list('ocean', [
    (0.0, '#c8e6f5'), (0.4, '#90cce8'),
    (0.7, '#4a9fc4'), (1.0, '#1565a0'),
], N=256)

x2d = np.linspace(0, 1, 200)
y2d = np.linspace(0, 1, 200)
XX, YY = np.meshgrid(x2d, y2d)
ocean = (0.5 + 0.3*np.sin(XX*6 + YY*4) +
         0.2*np.cos(XX*10 - YY*6 + 0.5) +
         0.1*np.sin(XX*15 + YY*8))
ocean = (ocean - ocean.min()) / (ocean.max() - ocean.min())
ax_map.imshow(ocean, cmap=ocean_cmap, origin='lower',
              extent=[0,1,0,1], alpha=0.70)

# Coastlines
na_coast_y = [0.85,0.80,0.72,0.65,0.58,0.52,0.46,0.40,0.35,0.28,0.20]
na_coast_x = [0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20]
ax_map.fill_betweenx(
    np.linspace(0.20, 0.85, 50), 0,
    np.interp(np.linspace(0.20,0.85,50), na_coast_y[::-1], na_coast_x[::-1]),
    color='#d4e8c8', alpha=0.90)

eu_coast_x = [0.75,0.78,0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.95,1.0]
eu_coast_y = [1.00,0.92,0.85,0.78,0.72,0.65,0.58,0.50,0.40,0.28,0.0]
ax_map.fill_betweenx(
    np.linspace(0.0, 1.0, 50),
    np.interp(np.linspace(0.0,1.0,50), eu_coast_y[::-1], eu_coast_x[::-1]),
    1.0, color='#d4e8c8', alpha=0.90)

greenland_x = [0.20,0.25,0.30,0.32,0.28,0.22,0.18,0.15,0.20]
greenland_y = [0.95,0.98,0.96,0.90,0.85,0.88,0.92,0.90,0.95]
ax_map.fill(greenland_x, greenland_y, color='#d4e8c8', alpha=0.90)

# Current arrows
angle = np.radians(42)
for sx, sy in [(0.25,0.35),(0.35,0.42),(0.45,0.49),
               (0.55,0.56),(0.30,0.48),(0.40,0.55)]:
    ex = sx + 0.08*np.cos(angle)
    ey = sy + 0.08*np.sin(angle)
    ax_map.annotate('', xy=(ex,ey), xytext=(sx,sy),
                   arrowprops=dict(arrowstyle='->',
                                  color='#1565a0', lw=1.2, alpha=0.65))

ax_map.text(0.32, 0.30, 'North Atlantic\nCurrent',
            color='#1565a0', fontsize=7.5, ha='center', style='italic')

# Buoy
ax_map.scatter([0.62],[0.62], s=180, color='#d4820a',
               zorder=10, edgecolors='#333333', linewidths=1.5)
ax_map.text(0.66, 0.65, 'Buoy\n47°N 15°W',
            color='#d4820a', fontsize=8, fontweight='500')

ax_map.text(0.12, 0.50, 'Atlantic\nOcean',
            color='#1565a0', fontsize=9, ha='center',
            style='italic', alpha=0.60)

ax_map.set_xlim(0,1); ax_map.set_ylim(0,1)
ax_map.axis('off')
ax_map.set_title('Buoy location — 47°N 15°W\nwest of Ireland',
                 color='#1565a0', fontsize=9, pad=8, fontweight='400')

# ── Right: time series ────────────────────────────────────────────────────────
ax_ts = fig.add_axes([0.36, 0.12, 0.60, 0.75])
ax_ts.set_facecolor('#f8f9fa')

for label, (t0, t1, col) in seasons.items():
    ax_ts.axvspan(t0, t1, color=col, alpha=0.07)
    ax_ts.text((t0+t1)/2, 2.85, label,
               ha='center', va='top', fontsize=8,
               color=col, alpha=0.75)

ou_std = sigma / np.sqrt(2*lam)
ax_ts.fill_between(t, -2*ou_std, 2*ou_std,
                   color='#2196c4', alpha=0.08)
ax_ts.fill_between(t, -ou_std, ou_std,
                   color='#2196c4', alpha=0.10)

ax_ts.plot(t, X, color='#1565a0', lw=1.8, zorder=4)
ax_ts.axhline(0, color='#333333', lw=1.0, alpha=0.25,
              linestyle='--', zorder=3)
ax_ts.text(5, 0.08, 'μ = long-term mean',
           color='#333333', fontsize=8, alpha=0.50)

for sd, sl in zip(storm_days, storm_labels):
    yval = X[sd]
    ax_ts.scatter([sd],[yval], s=60, color='#d4820a',
                  zorder=6, edgecolors='#333333', linewidths=1)
    offset_y = 0.35 if yval > 0 else -0.45
    ax_ts.annotate(sl, xy=(sd,yval), xytext=(sd, yval+offset_y),
                   ha='center', fontsize=7.5, color='#d4820a',
                   arrowprops=dict(arrowstyle='-', color='#d4820a',
                                  alpha=0.40, lw=0.8))

for idx in [80, 200, 310]:
    yv = X[idx]
    if abs(yv) > 0.3:
        ax_ts.annotate('',
                       xy=(idx+18, yv*0.55), xytext=(idx, yv),
                       arrowprops=dict(arrowstyle='->',
                                      color='#2e8b57', lw=1.2, alpha=0.65))

ax_ts.text(95, X[80]*0.75, 'pulled back\ntoward mean',
           color='#2e8b57', fontsize=8)

ax_ts.set_xlim(0, 364); ax_ts.set_ylim(-3.2, 3.2)
ax_ts.set_xlabel('day of year', fontsize=11, color='#444444')
ax_ts.set_ylabel('SST anomaly (°C)', fontsize=11, color='#444444')
ax_ts.tick_params(colors='#666666')
for sp in ax_ts.spines.values():
    sp.set_edgecolor('#dddddd')
ax_ts.grid(True, color='#dddddd', linewidth=0.5, alpha=0.8)

months     = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
month_days = [0,31,59,90,120,151,181,212,243,273,304,334]
ax_ts.set_xticks(month_days)
ax_ts.set_xticklabels(months, fontsize=9, color='#666666')

ax_ts.set_title('One year of SST anomaly — buoy 47°N 15°W\n'
                'The ocean drifts, but always comes back',
                color='#1565a0', fontsize=12, pad=10, fontweight='400')

ax_ts.text(330, -2.7, 'Ornstein-Uhlenbeck\nprocess',
           color='#1565a0', fontsize=9, ha='right',
           bbox=dict(boxstyle='round,pad=0.4',
                     facecolor='white', edgecolor='#1565a0', alpha=0.8))

fig.text(0.50, 0.97,
         'A buoy in the North Atlantic — one year of memory',
         ha='center', fontsize=15, color='#1a1a2e', fontweight='400')
fig.text(0.50, 0.93,
         'Sea surface temperature drifts continuously — '
         'storms push it away, the ocean always pulls it back · '
         'this is the Ornstein-Uhlenbeck process',
         ha='center', fontsize=9.5, color='#555555')

os.makedirs('outputs', exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'outputs', 'article2_buoy.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {out}")
