"""
Tema visual centralizado para todas las figuras del proyecto.

Paletas elegidas según la investigación en visualización de datos:
  - CATEGÓRICA: Okabe-Ito (Wong, Nature Methods 2011). Colorblind-safe y
    perceptualmente distinguible; evita depender de un solo tono.
  - ORDINAL (nivel de ingreso): cividis (Nuñez et al. 2018), perceptualmente
    uniforme y optimizada para daltonismo; va de azul oscuro a amarillo (sin verde).
  - DIVERGENTE (correlaciones, cargas): RdBu de ColorBrewer (Brewer), azul ↔ rojo.

Toda figura importa este módulo y llama `apply_theme()` una vez.
"""
import numpy as np
from matplotlib import rcParams, cm

# ── Paleta categórica Okabe-Ito ───────────────────────────────────────────── #
NARANJA   = "#E69F00"  # orange
CELESTE   = "#56B4E9"  # sky blue
VERDE     = "#009E73"  # bluish green
AMARILLO  = "#F0E442"  # yellow
AZUL      = "#0072B2"  # blue
BERMELLON = "#D55E00"  # vermillion
PURPURA   = "#CC79A7"  # reddish purple
GRIS      = "#999999"

CATEGORICAL = [AZUL, BERMELLON, VERDE, NARANJA, CELESTE, PURPURA, AMARILLO, GRIS]

# ── Neutros ───────────────────────────────────────────────────────────────── #
NEGRO      = "#222222"
GRIS_MEDIO = "#8C8C8C"
GRIS_CLARO = "#E8E8E8"
NEUTRO     = "#F7F7F7"

# ── Divergente (correlaciones / cargas +/-) ──────────────────────────────── #
DIVERGING = cm.get_cmap("RdBu")   # 0 = rojo (negativo), 1 = azul (positivo)
DIV_POS = AZUL
DIV_NEG = BERMELLON
POS_FILL = AZUL
NEG_FILL = BERMELLON

# ── Conglomerados (cualitativo, ordenado por PC1) ─────────────────────────── #
CLUSTER2_COLORS = {0: BERMELLON, 1: AZUL}
CLUSTER2_LABELS = {0: "En desarrollo", 1: "Desarrollado"}
CLUSTER3_COLORS = {0: BERMELLON, 1: NARANJA, 2: AZUL}
CLUSTER3_LABELS = {0: "Bajo desarrollo", 1: "Emergente", 2: "Desarrollado"}

# ── Nivel de ingreso (ORDINAL → cividis) ──────────────────────────────────── #
INCOME_ORDER = ["Low income", "Lower middle income",
                "Upper middle income", "High income"]
_civ = cm.get_cmap("cividis")
_civ_pts = [_civ(t) for t in (0.08, 0.40, 0.66, 0.95)]


def _to_hex(rgba):
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgba[:3])


INCOME_COLORS = {lv: _to_hex(c) for lv, c in zip(INCOME_ORDER, _civ_pts)}
INCOME_COLORS["Not classified"] = GRIS
INCOME_LABELS = {
    "Low income": "Bajo", "Lower middle income": "Medio-bajo",
    "Upper middle income": "Medio-alto", "High income": "Alto",
    "Not classified": "Sin clasificar",
}


def apply_theme():
    """Configura matplotlib globalmente con el estilo del proyecto."""
    rcParams.update({
        "font.family":       "Arial",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   NEGRO,
        "axes.labelsize":    11,
        "axes.labelcolor":   NEGRO,
        "axes.edgecolor":    GRIS_MEDIO,
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        GRIS_CLARO,
        "grid.linewidth":    0.9,
        "figure.facecolor":  "white",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.color":       NEGRO,
        "ytick.color":       NEGRO,
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "figure.dpi":        110,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


def income_palette(levels):
    return [INCOME_COLORS.get(l, GRIS) for l in levels]


# ── Alias de compatibilidad (nombres semánticos antiguos → paleta nueva) ──── #
VERDE_AGUA = AZUL
CORAL = BERMELLON
LAVANDA = CELESTE
ROSA = PURPURA
LIMA = VERDE
ARENA = NARANJA
