"""
article2_figure.py
------------------
Figure for Article 2: "Why the ocean forgets slowly — and so does GPT"

Visualises Theorem 7 from Baxevani, Podgórski & Rychlik (2010):
The discretized autoregression converges to the continuous OU field.

  Continuous:   dX_t = -λ(X_t - μ)dt + σ dW_t
  Discrete:     X(t) = ρ·X(t-dt) + √(1-ρ²)·ε_t   where ρ = e^{-λ·dt}

Three panels:
  ① Continuous OU process — SST anomaly drifting, always pulled back
  ② Discrete AR(1) — GPT-style: same dynamics, discretized
  ③ Memory decay — covariance e^{-λ|τ|} for three λ values,
     annotated as "short context", "medium context", "long context"

Uses style_config.py for visual consistency.
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import style_config as sc

RNG = np.random.default_rng(17)

# ── OU simulation (continuous, Euler-Maruyama) ────────────────────────────────
def simulate_ou(n_steps, dt, lam, mu, sigma, x0):
    """Euler-Maruyama discretization of dX = -λ(X-μ)dt + σ dW."""
    x    = np.zeros(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        dW   = RNG.standard_normal() * np.sqrt(dt)
        x[i] = x[i-1] - lam*(x[i-1] - mu)*dt + sigma*dW
    return x

# ── AR(1) simulation (discrete, exact ρ = e^{-λ·dt}) ─────────────────────────
def simulate_ar1(n_steps, rho, sigma_innov, x0):
    """Exact AR(1): X_t = ρ·X_{t-1} + √(1-ρ²)·ε_t"""
    x    = np.zeros(n_steps)
    x[0] = x0
    innov_std = sigma_innov * np.sqrt(1 - rho**2)
    for i in range(1, n_steps):
        x[i] = rho * x[i-1] + innov_std * RNG.standard_normal()
    return x

# ── Parameters ────────────────────────────────────────────────────────────────
N_STEPS   = 400
DT_CONT   = 0.05          # continuous time step (days, effectively)
DT_DISC   = 0.5           # AR(1) discrete step — coarser
LAM       = 0.18          # mean reversion rate
MU        = 0.0           # long-term mean (anomaly centred at 0)
SIGMA     = 0.55          # diffusion coefficient
X0        = 1.2           # starting anomaly

t_cont = np.arange(N_STEPS) * DT_CONT
t_disc = np.arange(N_STEPS) * DT_DISC

rho = np.exp(-LAM * DT_DISC)

ou_path   = simulate_ou(N_STEPS, DT_CONT, LAM, MU, SIGMA, X0)
ar1_path  = simulate_ar1(N_STEPS, rho, SIGMA, X0)

# ── Three OU realisations for context (panel 1) ───────────────────────────────
ou_paths = [simulate_ou(N_STEPS, DT_CONT, LAM, MU, SIGMA,
                         RNG.uniform(-1.5, 1.5)) for _ in range(3)]

# ── Memory decay curves (panel 3) ─────────────────────────────────────────────
tau     = np.linspace(0, 25, 300)
lambdas = [0.08, 0.18, 0.55]
labels  = ["Long memory<br>(slow forgetting)", 
           "Medium memory",
           "Short memory<br>(fast forgetting)"]
nlp_labels = ["Very long context", "Typical LLM", "Narrow context window"]
decay_colors = [sc.COLORS["ocean_mid"], sc.COLORS["warm_light"], 
                sc.COLORS["warm_deep"]]

# ── Figure ────────────────────────────────────────────────────────────────────
print("Building Article 2 figure...")

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "① Continuous OU process<br>"
        "<sup>SST anomaly · mean-reverting · dX = −λ(X−μ)dt + σ dW</sup>",

        "② Discrete AR(1)  →  language model<br>"
        "<sup>X_t = ρ·X_{t−1} + √(1−ρ²)·ε_t · same dynamics, discretized · "
        "Theorem 7, Baxevani et al. 2010</sup>",

        "③ Memory decay: covariance e^{−λ|τ|}<br>"
        "<sup>λ controls how fast past context is forgotten · "
        "ocean and language share the same forgetting function</sup>",
    ],
    horizontal_spacing=0.09,
)

# ── Panel 1: Continuous OU ────────────────────────────────────────────────────

# Background realisations
for i, path in enumerate(ou_paths):
    fig.add_trace(go.Scatter(
        x=t_cont, y=path,
        mode="lines",
        line=dict(color="rgba(33,150,196,0.18)", width=1),
        showlegend=False, name=f"OU realisation {i}",
    ), row=1, col=1)

# Main path
fig.add_trace(go.Scatter(
    x=t_cont, y=ou_path,
    mode="lines",
    line=dict(color=sc.COLORS["ocean_mid"], width=2.2),
    name="SST anomaly (OU)", showlegend=True,
), row=1, col=1)

# Mean line
fig.add_hline(y=MU, line_dash="dot",
              line_color="rgba(0,0,0,0.35)", line_width=1.2,
              row=1, col=1)

# Mean reversion annotation
fig.add_annotation(
    x=t_cont[80], y=ou_path[80] + 0.35,
    xref="x", yref="y",
    text="pulled back<br>toward mean",
    font=dict(size=9, color=sc.COLORS["highlight_fg"]),
    showarrow=True, arrowhead=2, arrowcolor=sc.COLORS["border"],
    ax=30, ay=-25,
    bgcolor=sc.COLORS["highlight_bg"], borderpad=3,
)

fig.add_annotation(
    x=t_cont[-1]*0.05, y=MU + 0.12,
    xref="x", yref="y",
    text="μ = long-term mean",
    font=dict(size=8.5, color="rgba(0,0,0,0.5)"),
    showarrow=False,
)

# ── Panel 2: Discrete AR(1) ───────────────────────────────────────────────────

fig.add_trace(go.Scatter(
    x=t_disc, y=ar1_path,
    mode="lines+markers",
    line=dict(color=sc.COLORS["warm_light"], width=2.0),
    marker=dict(size=3.5, color=sc.COLORS["warm_deep"], opacity=0.6),
    name="AR(1) / language model", showlegend=True,
), row=1, col=2)

fig.add_hline(y=MU, line_dash="dot",
              line_color="rgba(0,0,0,0.35)", line_width=1.2,
              row=1, col=2)

# Annotation linking to OU
fig.add_annotation(
    x=t_disc[60], y=ar1_path[60] - 0.4,
    xref="x2", yref="y2",
    text=f"ρ = e<sup>−λ·dt</sup> = {rho:.3f}<br>same forgetting rate",
    font=dict(size=9, color=sc.COLORS["highlight_fg"]),
    showarrow=True, arrowhead=2, arrowcolor=sc.COLORS["border"],
    ax=-35, ay=30,
    bgcolor=sc.COLORS["highlight_bg"], borderpad=3,
)

fig.add_annotation(
    x=t_disc[160], y=1.55,
    xref="x2", yref="y2",
    text="← each token conditioned<br>on all previous tokens",
    font=dict(size=8.5, color="rgba(0,0,0,0.55)"),
    showarrow=False,
    bgcolor="rgba(255,255,255,0.7)", borderpad=2,
)

# ── Panel 3: Memory decay curves ─────────────────────────────────────────────

for i, (lam_i, label, nlp, col) in enumerate(
        zip(lambdas, labels, nlp_labels, decay_colors)):

    decay = np.exp(-lam_i * tau)

    fig.add_trace(go.Scatter(
        x=tau, y=decay,
        mode="lines",
        line=dict(color=col, width=2.2),
        name=f"λ={lam_i} — {nlp}",
        showlegend=True,
    ), row=1, col=3)

    # Label at right end
    fig.add_annotation(
        x=tau[-1] + 0.3, y=decay[-1],
        xref="x3", yref="y3",
        text=f"λ={lam_i}<br>{nlp}",
        font=dict(size=8, color=col),
        showarrow=False, xanchor="left",
    )

# Half-life marker
fig.add_hline(y=0.5, line_dash="dot",
              line_color="rgba(0,0,0,0.25)", line_width=1,
              row=1, col=3)
fig.add_annotation(
    x=1.0, y=0.52,
    xref="x3", yref="y3",
    text="half-life",
    font=dict(size=8.5, color="rgba(0,0,0,0.45)"),
    showarrow=False,
)

# ── Layout ────────────────────────────────────────────────────────────────────

layout2 = {k: v for k, v in sc.LAYOUT.items() if k != "margin"}
fig.update_layout(
    **layout2,
    title=dict(
        text=(
            "The ocean forgets slowly — and so does GPT"
            "<br><sup>The Ornstein-Uhlenbeck process and the autoregressive language model "
            "are the same dynamical system · continuous vs discrete · "
            "Theorem 7, Baxevani, Podgórski &amp; Rychlik (2010)</sup>"
        ),
        font=dict(size=sc.FONT_TITLE, family=sc.FONT_FAMILY),
        x=0.5,
    ),
    height=500,
    width=sc.FIGURE_WIDTH,
    margin=dict(t=130, b=80, l=60, r=120),
    legend=dict(
        orientation="h", y=-0.22, x=0.5, xanchor="center",
        font=dict(size=9),
    ),
)

fig.update_xaxes(title_text="time (days)", showgrid=True,
                 gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                 col=1, row=1)
fig.update_xaxes(title_text="token position", showgrid=True,
                 gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                 col=2, row=1)
fig.update_xaxes(title_text="lag τ", showgrid=True,
                 gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                 col=3, row=1)

fig.update_yaxes(title_text="SST anomaly (°C)", col=1, row=1)
fig.update_yaxes(title_text="semantic state", col=2, row=1)
fig.update_yaxes(title_text="covariance  e^{−λτ}", col=3, row=1,
                 range=[0, 1.05])

# ── Export ────────────────────────────────────────────────────────────────────
out = "/mnt/user-data/outputs/article2_figure.html"
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print(f"Saved → {out}")
