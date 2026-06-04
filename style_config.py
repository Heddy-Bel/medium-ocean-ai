"""
style_config.py
---------------
Shared visual identity for the Medium article series:
"What the ocean taught me about AI"

Import this in every figure script to ensure consistency
across all four articles.
"""

# ── Color palette ─────────────────────────────────────────────────────────────

COLORS = {
    # Field / heatmap
    "ocean_deep":      "#053061",
    "ocean_mid":       "#2196c4",
    "ocean_light":     "#74c5dd",
    "neutral":         "#f5f0e8",
    "warm_light":      "#f4a261",
    "warm_deep":       "#d62828",
    "warm_darkest":    "#7b0c0c",

    # Dynamic overlays
    "current_arrow":   "#ffffff",          # white — current vectors
    "wind_arrow":      "#f4c842",          # warm yellow — wind vectors
    "wave_crest":      "#74c5dd",          # soft cyan — wave crest lines
    "wave_arrow":      "#ffffff",          # white — wave propagation

    # Annotations
    "annotation_bg":   "rgba(0,0,0,0.50)",
    "annotation_fg":   "#ffffff",
    "highlight_bg":    "rgba(255,255,255,0.80)",
    "highlight_fg":    "#1a1a1a",

    # Layout
    "background":      "#f8f8f6",          # off-white paper
    "panel_bg":        "#ffffff",
    "border":          "#2e5f8a",          # slate blue — takeaway box

    # Covariance panels
    "cov_high":        "#d62828",
    "cov_low":         "#053061",
}

# ── Oceanographic SST colorscale (Plotly format) ──────────────────────────────

OCEAN_CMAP = [
    [0.00, COLORS["ocean_deep"]],
    [0.15, "#1565a0"],
    [0.30, COLORS["ocean_mid"]],
    [0.45, COLORS["ocean_light"]],
    [0.55, COLORS["neutral"]],
    [0.65, COLORS["warm_light"]],
    [0.80, COLORS["warm_deep"]],
    [1.00, COLORS["warm_darkest"]],
]

WAVE_CMAP = [
    [0.0,  "#1a3a5c"],
    [0.25, "#2e7eb8"],
    [0.5,  "#f0ede4"],
    [0.75, "#c97a3a"],
    [1.0,  "#7b2d00"],
]

DIVERGING_CMAP = "RdBu_r"

# ── Typography ────────────────────────────────────────────────────────────────

FONT_FAMILY  = "Arial, sans-serif"
FONT_TITLE   = 15
FONT_SUBPLOT = 12
FONT_AXIS    = 11
FONT_ANNOT   = 10

# ── Layout defaults ───────────────────────────────────────────────────────────

LAYOUT = dict(
    paper_bgcolor = COLORS["background"],
    plot_bgcolor  = COLORS["panel_bg"],
    font          = dict(family=FONT_FAMILY, size=FONT_AXIS),
    margin        = dict(t=110, b=55, l=55, r=85),
)

FIGURE_WIDTH  = 1080
FIGURE_HEIGHT = 880

# ── Colorbar defaults ─────────────────────────────────────────────────────────

def colorbar(x, y, label, shared=True):
    """Return a consistent colorbar dict."""
    return dict(
        x=x, y=y,
        len=0.44, thickness=11,
        title=dict(text=label, side="right"),
        tickfont=dict(size=9),
    )

# ── Annotation helpers ────────────────────────────────────────────────────────

def dark_label(text):
    """Dark pill annotation — for dynamic labels on heatmaps."""
    return dict(
        text=text,
        font=dict(size=FONT_ANNOT, color=COLORS["annotation_fg"]),
        showarrow=False,
        bgcolor=COLORS["annotation_bg"],
        borderpad=3,
    )

def light_label(text):
    """Light pill annotation — for callouts on covariance panels."""
    return dict(
        text=text,
        font=dict(size=FONT_ANNOT, color=COLORS["highlight_fg"]),
        showarrow=True,
        arrowhead=2,
        bgcolor=COLORS["highlight_bg"],
        borderpad=3,
    )

# ── Arrow helpers ─────────────────────────────────────────────────────────────

def arrow_annotation(x1, y1, x0, y0, xref, yref, color, width=1.1, size=0.9):
    """Single arrow annotation for vector field overlays."""
    return dict(
        x=x1, y=y1, ax=x0, ay=y0,
        xref=xref, yref=yref,
        axref=xref, ayref=yref,
        showarrow=True,
        arrowhead=2, arrowsize=size,
        arrowwidth=width,
        arrowcolor=color,
    )

# ── Series metadata ───────────────────────────────────────────────────────────

SERIES_SUBTITLE = (
    "What the ocean taught me about AI"
    "<br><sup>A Medium series by Heddy Bellout · "
    "Spatio-temporal stochastic processes → NLP</sup>"
)
