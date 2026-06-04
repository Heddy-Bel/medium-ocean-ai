"""
article1_figure.py
------------------
Figure for Article 1: "What the ocean taught me about AI"

Visualises the core idea from Baxevani, Podgórski & Rychlik (2010):

  Y(p,t) = ∫ f(t,s) ξ(ψ_{t,s-t}(p); ds)

A static field has velocities centred at zero — no organised movement.
Composing with a deterministic flow ψ introduces directional structure.
The three dynamics (current, wind, waves) correspond to:

  CURRENT — the flow map ψ: deterministic transport operator
  WIND    — short-range modifier of the innovation covariance r_S
  WAVES   — spectral / periodic component of r_S

Four panels:
  ① Static field (no flow) — isotropic, no organisation
  ② Current (flow map ψ) — long-range anisotropy, directional transport
  ③ Wind + Waves — innovation covariance modifiers
  ④ Full field: static + flow + wind + waves (non-separable)

Imports style_config.py for visual consistency across the series.
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
from scipy.linalg import cholesky
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import style_config as sc

RNG = np.random.default_rng(42)

# ── Grid ──────────────────────────────────────────────────────────────────────
N      = 55
x      = np.linspace(0, 10, N)
y      = np.linspace(0, 10, N)
XX, YY = np.meshgrid(x, y)
coords = np.column_stack([XX.ravel(), YY.ravel()])

# ── Physical parameters ───────────────────────────────────────────────────────
# Current (flow map ψ)
CURRENT_ANGLE = 35.0
CURRENT_ANISO = 3.2
CURRENT_LS    = 2.2
CURRENT_VAR   = 2.8

# Wind (surface innovation modifier)
WIND_ANGLE    = 110.0
WIND_ANISO    = 2.0
WIND_LS       = 0.65
WIND_VAR      = 1.0

# Waves (spectral periodic component)
WAVE_ANGLE    = 60.0
WAVE_LENGTH   = 2.5
WAVE_DECAY    = 4.0
WAVE_VAR      = 0.7

# Static (isotropic baseline)
STATIC_LS     = 1.8
STATIC_VAR    = 3.5

NUGGET        = 0.05

# ── Kernel functions ──────────────────────────────────────────────────────────

def rotation_matrix(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)],
                     [np.sin(a),  np.cos(a)]])

def matern32(d):
    s = np.sqrt(3) * d
    return (1 + s) * np.exp(-s)

def aniso_dist(coords, ls, aniso, angle):
    R   = rotation_matrix(angle)
    D   = np.diag([1/(ls * aniso)**2, 1/ls**2])
    M   = R @ D @ R.T
    diff = coords[:, None, :] - coords[None, :, :]
    d2   = np.einsum('ijk,kl,ijl->ij', diff, M, diff)
    return np.sqrt(np.clip(d2, 0, None))

def iso_dist(coords, ls):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1)) / ls

# ── Panel kernels ─────────────────────────────────────────────────────────────

def K_static(coords):
    """Isotropic baseline — no flow, velocities centred at zero."""
    d = iso_dist(coords, STATIC_LS)
    K = STATIC_VAR * matern32(d)
    return K + NUGGET * np.eye(len(coords)) + 1e-7 * np.eye(len(coords))

def K_current(coords):
    """Flow map ψ — anisotropic diffusion operator, dominant transport."""
    d = aniso_dist(coords, CURRENT_LS, CURRENT_ANISO, CURRENT_ANGLE)
    K = CURRENT_VAR * matern32(d)
    return K + NUGGET * np.eye(len(coords)) + 1e-7 * np.eye(len(coords))

def K_wind_wave(coords):
    """Innovation covariance modifiers — wind (short-range) + waves (periodic)."""
    # Wind component
    d_wind = aniso_dist(coords, WIND_LS, WIND_ANISO, WIND_ANGLE)
    K_wind = WIND_VAR * matern32(d_wind)

    # Wave component — spectral kernel
    angle_rad = np.radians(WAVE_ANGLE)
    wdir      = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    diff      = coords[:, None, :] - coords[None, :, :]
    d_proj    = np.einsum('ijk,k->ij', diff, wdir)
    d_iso     = np.sqrt(np.sum(diff**2, axis=-1))
    K_wave    = WAVE_VAR * np.cos(2*np.pi*d_proj/WAVE_LENGTH) * \
                np.exp(-d_iso**2 / (2*WAVE_DECAY**2))

    K = K_wind + K_wave
    return K + NUGGET * np.eye(len(coords)) + 1e-7 * np.eye(len(coords))

def K_full(coords):
    """Full non-separable field — flow + wind + waves superposed."""
    d_cur  = aniso_dist(coords, CURRENT_LS, CURRENT_ANISO, CURRENT_ANGLE)
    K_cur  = CURRENT_VAR * matern32(d_cur)

    d_wind = aniso_dist(coords, WIND_LS, WIND_ANISO, WIND_ANGLE)
    Kw     = WIND_VAR * matern32(d_wind)

    angle_rad = np.radians(WAVE_ANGLE)
    wdir      = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    diff      = coords[:, None, :] - coords[None, :, :]
    d_proj    = np.einsum('ijk,k->ij', diff, wdir)
    d_iso     = np.sqrt(np.sum(diff**2, axis=-1))
    Kwv       = WAVE_VAR * np.cos(2*np.pi*d_proj/WAVE_LENGTH) * \
                np.exp(-d_iso**2 / (2*WAVE_DECAY**2))

    K = K_cur + Kw + Kwv
    return K + NUGGET * np.eye(len(coords)) + 1e-7 * np.eye(len(coords))

# ── Sample GP ─────────────────────────────────────────────────────────────────

def sample_gp(K, offset=20.0):
    L = cholesky(K, lower=True)
    return (L @ RNG.standard_normal(len(K))).reshape(N, N) + offset

print("Sampling GP fields...")
sst_static   = sample_gp(K_static(coords),   offset=20.0)
sst_current  = sample_gp(K_current(coords),  offset=20.0)
sst_windwave = sample_gp(K_wind_wave(coords), offset=0.0)   # anomaly
sst_full     = sample_gp(K_full(coords),      offset=20.0)

# ── Arrow / overlay geometry ──────────────────────────────────────────────────

step        = 7
xa, ya      = x[::step], y[::step]
Xa, Ya      = np.meshgrid(xa, ya)

def vec_field(angle_deg, scale):
    a  = np.radians(angle_deg)
    return (np.cos(a)*np.ones_like(Xa)*scale,
            np.sin(a)*np.ones_like(Ya)*scale)

Uc, Vc = vec_field(CURRENT_ANGLE, 0.42)
Uw, Vw = vec_field(WIND_ANGLE,    0.28)

def wave_crests(angle_deg, n=4):
    perp = np.radians(angle_deg + 90)
    prop = np.radians(angle_deg)
    lines = []
    for i in range(n):
        t  = (i + 1) / (n + 1) * 10
        cx, cy = t*np.cos(prop), t*np.sin(prop)
        hw = 2.8
        lines.append((
            5 + cx - hw*np.cos(perp), 5 + cy - hw*np.sin(perp),
            5 + cx + hw*np.cos(perp), 5 + cy + hw*np.sin(perp),
        ))
    return lines

crests = wave_crests(WAVE_ANGLE)

# ── Figure ────────────────────────────────────────────────────────────────────

print("Building figure...")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "① Static field  —  no flow<br>"
        "<sup>Isotropic · velocities centred at zero · no organised movement</sup>",

        "② Current  —  flow map ψ introduced<br>"
        "<sup>Deterministic transport operator · direction 35° · anisotropy 3.2×</sup>",

        "③ Wind + Waves  —  innovation covariance r_S<br>"
        "<sup>Wind: short-range surface perturbation · Waves: spectral periodic structure</sup>",

        "④ Full field  —  flow ψ  ⊕  r_S(wind, waves)<br>"
        "<sup>Non-separable covariance · all three dynamics coupled · "
        "Baxevani, Podgórski & Rychlik (2010)</sup>",
    ],
    horizontal_spacing=0.07,
    vertical_spacing=0.17,
)

vmin = min(sst_static.min(), sst_current.min(), sst_full.min())
vmax = max(sst_static.max(), sst_current.max(), sst_full.max())
wmin, wmax = sst_windwave.min(), sst_windwave.max()

# ── Panel 1: Static field ─────────────────────────────────────────────────────
fig.add_trace(go.Heatmap(
    z=sst_static, x=x, y=y,
    colorscale=sc.OCEAN_CMAP, zmin=vmin, zmax=vmax,
    colorbar=sc.colorbar(0.455, 0.78, "°C"),
    name="Static", showscale=True,
), row=1, col=1)

def pill(text, x, y, xref, yref, light=False):
    """Add a pill-style label annotation."""
    if light:
        style = dict(bgcolor=sc.COLORS["highlight_bg"],
                     font=dict(size=sc.FONT_ANNOT, color=sc.COLORS["highlight_fg"]))
    else:
        style = dict(bgcolor=sc.COLORS["annotation_bg"],
                     font=dict(size=sc.FONT_ANNOT, color=sc.COLORS["annotation_fg"]))
    fig.add_annotation(x=x, y=y, xref=xref, yref=yref,
                       text=text, showarrow=False, borderpad=3, **style)

pill("no directional structure", 5.0, 0.6, "x", "y")

# ── Panel 2: Current (flow map) ───────────────────────────────────────────────
fig.add_trace(go.Heatmap(
    z=sst_current, x=x, y=y,
    colorscale=sc.OCEAN_CMAP, zmin=vmin, zmax=vmax,
    colorbar=sc.colorbar(1.01, 0.78, "°C"),
    name="Current", showscale=True,
), row=1, col=2)

# Flow map arrows
for i in range(Xa.shape[0]):
    for j in range(Xa.shape[1]):
        fig.add_annotation(**sc.arrow_annotation(
            Xa[i,j]+Uc[i,j], Ya[i,j]+Vc[i,j],
            Xa[i,j]-Uc[i,j], Ya[i,j]-Vc[i,j],
            "x2", "y2", sc.COLORS["current_arrow"]
        ))

pill("flow map ψ →", 0.7, 0.6, "x2", "y2")

# ── Panel 3: Wind + Waves ─────────────────────────────────────────────────────
fig.add_trace(go.Heatmap(
    z=sst_windwave, x=x, y=y,
    colorscale=sc.WAVE_CMAP, zmin=wmin, zmax=wmax,
    colorbar=sc.colorbar(0.455, 0.22, "anomaly (°C)"),
    name="Wind+Waves", showscale=True,
), row=2, col=1)

# Wind arrows
for i in range(Xa.shape[0]):
    for j in range(Xa.shape[1]):
        fig.add_annotation(**sc.arrow_annotation(
            Xa[i,j]+Uw[i,j], Ya[i,j]+Vw[i,j],
            Xa[i,j]-Uw[i,j], Ya[i,j]-Vw[i,j],
            "x3", "y3", sc.COLORS["wind_arrow"], width=1.0, size=0.8
        ))

# Wave crest lines
for (x0, y0, x1, y1) in crests:
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                  xref="x3", yref="y3",
                  line=dict(color="rgba(255,255,255,0.55)",
                            width=1.4, dash="dot"))

# Wave propagation arrow
pr = np.radians(WAVE_ANGLE)
fig.add_annotation(**sc.arrow_annotation(
    5+1.6*np.cos(pr), 5+1.6*np.sin(pr),
    5-1.6*np.cos(pr), 5-1.6*np.sin(pr),
    "x3", "y3", "rgba(255,255,255,0.9)", width=2.0, size=1.2
))

pill("wind →", 0.6, 9.3, "x3", "y3")
pill("wave crests", 8.5, 0.6, "x3", "y3")

# ── Panel 4: Full field ───────────────────────────────────────────────────────
fig.add_trace(go.Heatmap(
    z=sst_full, x=x, y=y,
    colorscale=sc.OCEAN_CMAP, zmin=vmin, zmax=vmax,
    colorbar=sc.colorbar(1.01, 0.22, "°C"),
    name="Full", showscale=True,
), row=2, col=2)

# Current arrows (lighter on full panel)
for i in range(Xa.shape[0]):
    for j in range(Xa.shape[1]):
        fig.add_annotation(**sc.arrow_annotation(
            Xa[i,j]+Uc[i,j]*0.75, Ya[i,j]+Vc[i,j]*0.75,
            Xa[i,j]-Uc[i,j]*0.75, Ya[i,j]-Vc[i,j]*0.75,
            "x4", "y4", "rgba(255,255,255,0.50)"
        ))

# Wave crests (subtle)
for (x0, y0, x1, y1) in crests:
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                  xref="x4", yref="y4",
                  line=dict(color="rgba(255,255,255,0.28)",
                            width=1.2, dash="dot"))

pill("ψ ⊕ wind ⊕ waves", 0.7, 0.6, "x4", "y4")

# ── Global layout ─────────────────────────────────────────────────────────────
fig.update_layout(
    **sc.LAYOUT,
    title=dict(
        text=(
            "From static field to dynamically evolving ocean surface"
            "<br><sup>Introducing a deterministic flow map ψ creates organised movement · "
            "just as training creates directional structure in a language model</sup>"
        ),
        font=dict(size=sc.FONT_TITLE, family=sc.FONT_FAMILY),
        x=0.5,
    ),
    height=sc.FIGURE_HEIGHT,
    width=sc.FIGURE_WIDTH,
)

for row in [1, 2]:
    fig.update_xaxes(title_text="longitude (°)", showgrid=False,
                     zeroline=False, row=row)
    fig.update_yaxes(title_text="latitude (°)", showgrid=False,
                     zeroline=False, row=row)

# ── Export ────────────────────────────────────────────────────────────────────
out = "/mnt/user-data/outputs/article1_figure.html"
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print(f"Saved → {out}")
