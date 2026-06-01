"""
Tema visual centralizado para todas las figuras del proyecto.

Paleta "Suave Set2" (ColorBrewer): pastel pero legible, buen equilibrio entre
baja fatiga visual y contraste/accesibilidad. Toda figura del pipeline importa
este módulo y llama `apply_theme()` una vez, de modo que el estilo sea
consistente en `reports/figures/`.
"""
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ── Paleta categórica (ColorBrewer Set2) ──────────────────────────────────── #
VERDE_AGUA = "#66C2A5"
CORAL      = "#FC8D62"
LAVANDA    = "#8DA0CB"
ROSA       = "#E78AC3"
LIMA       = "#A6D854"
AMARILLO   = "#FFD92F"
ARENA      = "#E5C494"
GRIS       = "#B3B3B3"

CATEGORICAL = [VERDE_AGUA, CORAL, LAVANDA, ROSA, LIMA, AMARILLO, ARENA, GRIS]

# ── Neutros ───────────────────────────────────────────────────────────────── #
NEGRO      = "#33373B"
GRIS_MEDIO = "#9AA0A6"
GRIS_CLARO = "#ECEFF1"
NEUTRO     = "#F5F5F5"

# ── Diverging (correlaciones y cargas +/-) ────────────────────────────────── #
# Coral (negativo) ── neutro ── teal (positivo). Versión algo más profunda que
# la categórica para que el texto sobre celdas tenga contraste.
DIV_NEG = "#D9694A"   # coral profundo
DIV_POS = "#3C8E7D"   # teal profundo
POS_FILL = VERDE_AGUA  # relleno de barras positivas
NEG_FILL = CORAL       # relleno de barras negativas

DIVERGING = LinearSegmentedColormap.from_list(
    "set2_div", [DIV_NEG, NEUTRO, DIV_POS]
)

# ── Conglomerados ─────────────────────────────────────────────────────────── #
# Orden por PC1: 0 = menos desarrollado, etc.
CLUSTER2_COLORS = {0: CORAL, 1: VERDE_AGUA}
CLUSTER2_LABELS = {0: "En desarrollo", 1: "Desarrollado"}
CLUSTER3_COLORS = {0: CORAL, 1: LAVANDA, 2: VERDE_AGUA}
CLUSTER3_LABELS = {0: "Bajo desarrollo", 1: "Emergente", 2: "Desarrollado"}

# ── Nivel de ingreso (ORDINAL → rampa secuencial teal) ────────────────────── #
INCOME_ORDER = ["Low income", "Lower middle income",
                "Upper middle income", "High income"]
INCOME_COLORS = {
    "Low income":          "#CDE7DF",
    "Lower middle income": "#94CFC0",
    "Upper middle income": "#56A593",
    "High income":         "#2A7563",
    "Not classified":      GRIS,
}
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
    """Devuelve la lista de colores para una secuencia de niveles de ingreso."""
    return [INCOME_COLORS.get(l, GRIS) for l in levels]
