"""
Tema visual centralizado para todas las figuras del proyecto.

Paleta base "Stormy Teal / Tangerine" (registro apagado y elegante):
    Stormy Teal #006D77 · Pearl Aqua #83C5BE · Alice Blue #EDF6F9 ·
    Almond Silk #FFDDD2 · Tangerine Dream #E29578

Su eje frío↔cálido (teal ↔ terracota) se aprovecha semánticamente: mapea con el
gradiente "desarrollado ↔ en desarrollo" y sirve para divergentes, clusters e
ingreso. Como base trae solo 3 colores fuertes (los dos pálidos son casi blancos
y no distinguen marcas sobre fondo blanco), se EXTIENDE con tonos apagados de la
misma familia para soportar las ~7 regiones, manteniendo el registro:
  - CATEGÓRICA: teal, terracota, oro, ciruela, aguamarina, oliva, rosa, siena.
    Apoyada en Gramazio, Laidlaw & Schloss (Colorgorical, IEEE TVCG 2016): la
    armonía depende de las RELACIONES entre tonos (luminosidad + separación de
    matiz), no de colores individuales — se prioriza cohesión estética con
    suficiente separación de matiz para discriminar.
  - ORDINAL (nivel de ingreso): rampa cálido→frío (rust → terracota → aguamarina
    → teal), visible sobre blanco y alineada con la dirección del desarrollo.
  - DIVERGENTE (correlaciones, cargas +/-): terracota ↔ teal, centro neutro
    cálido, extremos oscurecidos para contraste con texto blanco.
  - Alice Blue #EDF6F9 se reserva como fondo neutro (NEUTRO).

Toda figura importa este módulo y llama `apply_theme()` una vez.
"""
import numpy as np
from matplotlib import rcParams, cm
from matplotlib.colors import LinearSegmentedColormap

# ── Paleta categórica "Stormy Teal / Tangerine" extendida ─────────────────── #
# Los nombres son handles semánticos heredados (AZUL = color "frío/desarrollo",
# BERMELLON = color "cálido/en desarrollo", etc.); los hex son la paleta nueva.
AZUL      = "#006D77"  # stormy teal  (hero frío / desarrollo)
BERMELLON = "#E29578"  # tangerine    (terracota / en desarrollo)
NARANJA   = "#E9C46A"  # muted gold   (oro apagado)
VERDE     = "#80A468"  # muted olive  (verde oliva)
CELESTE   = "#83C5BE"  # pearl aqua   (aguamarina)
PURPURA   = "#6D597A"  # muted plum   (ciruela)
ROSA      = "#B56576"  # dusty rose   (rosa viejo)
MARRON    = "#BC6C4F"  # burnt sienna (siena)
GRIS      = "#B0AAA2"  # warm gray    (gris cálido)
AMARILLO  = NARANJA    # alias semántico (la paleta no tiene amarillo puro)

CATEGORICAL = [AZUL, BERMELLON, NARANJA, PURPURA, CELESTE,
               VERDE, ROSA, MARRON, GRIS]

# ── Neutros ───────────────────────────────────────────────────────────────── #
NEGRO      = "#2A363B"   # casi-negro con matiz pizarra (cohesivo con el teal)
GRIS_MEDIO = "#8C8C8C"
GRIS_CLARO = "#E6E9EA"
NEUTRO     = "#EDF6F9"   # Alice Blue: fondo neutro de la paleta base

# ── Divergente terracota ↔ teal (correlaciones / cargas +/-) ──────────────── #
# Convención: 0 = terracota (negativo), 1 = teal (positivo). Extremos oscuros
# para que el texto blanco superpuesto tenga contraste suficiente.
DIVERGING = LinearSegmentedColormap.from_list("TealTerracota", [
    "#a8432f",   # 0.00  rust oscuro (negativo fuerte)
    "#E29578",   # 0.25  terracota
    "#F1EDE6",   # 0.50  neutro cálido
    "#5AA39B",   # 0.75  teal medio
    "#006D77",   # 1.00  stormy teal (positivo fuerte)
])
DIV_POS = AZUL
DIV_NEG = BERMELLON
POS_FILL = AZUL
NEG_FILL = BERMELLON

# ── Conglomerados (cualitativo, ordenado por PC1) ─────────────────────────── #
CLUSTER2_COLORS = {0: BERMELLON, 1: AZUL}
CLUSTER2_LABELS = {0: "En desarrollo", 1: "Desarrollado"}
CLUSTER3_COLORS = {0: MARRON, 1: NARANJA, 2: AZUL}
CLUSTER3_LABELS = {0: "Bajo desarrollo", 1: "Emergente", 2: "Desarrollado"}

# ── Nivel de ingreso (ORDINAL → rampa cálido→frío de la paleta) ───────────── #
INCOME_ORDER = ["Low income", "Lower middle income",
                "Upper middle income", "High income"]
_income_ramp = ["#C44536", "#E29578", "#83C5BE", "#006D77"]  # rust→terra→aqua→teal


def _to_hex(rgba):
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgba[:3])


INCOME_COLORS = {lv: c for lv, c in zip(INCOME_ORDER, _income_ramp)}
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
LIMA = VERDE
ARENA = NARANJA
