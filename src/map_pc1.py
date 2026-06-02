"""
Mapa coroplético: países coloreados por su score en PC1 (año 2023).

Genera dos salidas:
  - reports/figures/map_pc1_2023.png            (estática, alta resolución)
  - informe_final/figures/map_pc1_2023.html     (interactiva, Plotly)
  - informe_final/figures/map_pc1_2023.png      (copia para el informe)

Usa el shapefile de Natural Earth (110m) incluido en pyogrio.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

try:
    from . import config, theme
except ImportError:
    import config, theme

theme.apply_theme()

# Shapefile bundled con pyogrio (Natural Earth 110m lowres)
_pyogrio = Path(gpd.__file__).parent.parent / "pyogrio" / "tests" / "fixtures" / "naturalearth_lowres"
SHP_PATH = _pyogrio / "naturalearth_lowres.shp"

INFORME_FIGURES = config.ROOT / "informe_final" / "figures"
INFORME_FIGURES.mkdir(parents=True, exist_ok=True)

OUT_PNG  = config.FIGURES / "map_pc1_2023.png"
OUT_HTML = INFORME_FIGURES / "map_pc1_2023.html"


def _diverging_plotly():
    """Convierte el colormap matplotlib DIVERGING a escala Plotly."""
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    scale = []
    for p in positions:
        r, g, b, _ = theme.DIVERGING(p)
        scale.append((p, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"))
    return scale


def _make_static(scores, world):
    """Genera el PNG estático con matplotlib/geopandas."""
    vmin = scores["PC1"].quantile(0.02)
    vmax = scores["PC1"].quantile(0.98)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = theme.DIVERGING
    no_data_color = "#CCCCCC"

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor(theme.NEUTRO)
    fig.patch.set_facecolor(theme.NEUTRO)
    ax.axis("off")

    world[world["PC1"].isna()].plot(
        ax=ax, color=no_data_color, edgecolor="#999999", linewidth=0.3,
    )
    world[world["PC1"].notna()].plot(
        ax=ax, column="PC1", cmap=cmap, norm=norm,
        edgecolor="#999999", linewidth=0.3,
    )

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.025, pad=0.02, aspect=50, shrink=0.6)
    cbar.set_label("Score en PC1", fontsize=10, color=theme.NEGRO)
    cbar.ax.tick_params(labelsize=8, colors=theme.NEGRO)
    cbar.outline.set_edgecolor(theme.GRIS_MEDIO)
    cbar.ax.text(0.0, -1.6, "← Menor desarrollo", transform=cbar.ax.transAxes,
                 ha="left", va="top", fontsize=7.5, color=theme.NEGRO)
    cbar.ax.text(1.0, -1.6, "Mayor desarrollo →", transform=cbar.ax.transAxes,
                 ha="right", va="top", fontsize=7.5, color=theme.NEGRO)

    no_data_patch = mpatches.Patch(facecolor=no_data_color, edgecolor="#999999",
                                   linewidth=0.5, label="Sin datos")
    ax.legend(handles=[no_data_patch], loc="lower left", fontsize=8,
              frameon=True, framealpha=0.8, edgecolor=theme.GRIS_CLARO)

    ax.set_title("PC1 por país — 2023", fontsize=14, fontweight="bold",
                 color=theme.NEGRO, pad=8)
    ax.set_xlabel("Terracota = menor desarrollo relativo  ·  Teal = mayor desarrollo relativo",
                  fontsize=9, color=theme.GRIS_MEDIO, labelpad=4)
    ax.xaxis.set_label_position("top")

    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG estático  → {OUT_PNG}")

    dst = INFORME_FIGURES / "map_pc1_2023.png"
    shutil.copy2(OUT_PNG, dst)
    print(f"  PNG (informe) → {dst}")


def _make_interactive(scores, world):
    """Genera el HTML interactivo con Plotly choropleth."""
    vmin = scores["PC1"].quantile(0.02)
    vmax = scores["PC1"].quantile(0.98)
    colorscale = _diverging_plotly()

    world_data = world[["iso_a3", "name", "PC1"]].copy()
    world_data = world_data.rename(columns={"iso_a3": "iso3", "name": "pais"})

    world_data["hover"] = world_data.apply(
        lambda r: (
            f"<b>{r['pais']}</b><br>PC1: {r['PC1']:.3f}"
            if pd.notna(r["PC1"])
            else f"<b>{r['pais']}</b><br>Sin datos"
        ),
        axis=1,
    )

    fig = go.Figure(go.Choropleth(
        locations=world_data["iso3"],
        z=world_data["PC1"],
        text=world_data["hover"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=colorscale,
        zmin=vmin,
        zmid=0,
        zmax=vmax,
        colorbar=dict(
            title=dict(text="Score PC1", font=dict(size=12, color=theme.NEGRO)),
            tickfont=dict(size=10, color=theme.NEGRO),
            len=0.55,
            thickness=14,
            x=1.01,
        ),
        marker_line_color="#999999",
        marker_line_width=0.4,
        showscale=True,
    ))

    fig.update_layout(
        title=dict(
            text=(
                "PC1 por país — 2023<br>"
                "<sup style='color:#8C8C8C'>"
                "Terracota = menor desarrollo · Teal = mayor desarrollo"
                "</sup>"
            ),
            font=dict(size=15, color=theme.NEGRO, family="Arial"),
            x=0.5,
            xanchor="center",
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#AAAAAA",
            coastlinewidth=0.5,
            showland=True,
            landcolor=theme.NEUTRO,
            showocean=True,
            oceancolor="#D6EAF8",
            showlakes=False,
            projection_type="natural earth",
            bgcolor=theme.NEUTRO,
        ),
        paper_bgcolor=theme.NEUTRO,
        plot_bgcolor=theme.NEUTRO,
        margin=dict(l=0, r=0, t=70, b=0),
        height=520,
    )

    fig.write_html(
        OUT_HTML,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )
    print(f"  HTML interactivo → {OUT_HTML}")


def run():
    scores = pd.read_csv(
        config.DATA_PROC / "pca_scores_2023.csv",
        usecols=["countryiso3code", "PC1", "country"],
    )

    world = gpd.read_file(SHP_PATH)
    world["PC1"] = world["iso_a3"].map(
        dict(zip(scores["countryiso3code"], scores["PC1"]))
    )

    print("Generando mapa PC1 …")
    _make_static(scores, world)
    _make_interactive(scores, world)
    print("Listo.")


if __name__ == "__main__":
    run()
