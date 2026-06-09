"""
Transforma los figure-cards del informe para usar gráficos interactivos (iframe
a los .html de Plotly) con la imagen .png como respaldo de impresión.

Idempotente: si una figura ya fue convertida, la deja igual.
Uso: python src/_embed_interactive.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "informe_final" / "index.html"

# Altura del iframe por figura (coincide con la altura del gráfico Plotly)
HEIGHTS = {
    "eda_histogramas_2023": 960,
    "eda_correlacion_2023": 720,
    "eda_categoricas_2023": 480,
    "eda_boxplots_ingreso_2023": 720,
    "pca_scree_2023": 480,
    "pca_loadings_barras_2023": 620,
    "pca_circulo_correlaciones_2023": 640,
    "pca_biplot_ingreso_2023": 680,
    "clust_metricas_k_2023": 720,
    "clust_dendrograma_2023": 520,
    "clust_silhouette_k2_2023": 560,
    "clust_pca_k2_2023": 640,
    "clust_pca_k3_2023": 640,
    "compare_trayectorias": 680,
    "compare_transiciones": 420,
    "compare_descomposicion_dPC1": 560,
    "compare_loadings_pc1": 620,
    "map_pc1_2023": 540,
}
DEFAULT_H = 560

html = INDEX.read_text(encoding="utf-8")

# ── 1) Reemplazar cada <img src="figures/X.png" alt="ALT" loading="lazy"> ──────
img_re = re.compile(
    r'<img src="figures/(?P<name>[^"]+)\.png" alt="(?P<alt>[^"]*)" loading="lazy">'
)


def repl(m):
    name, alt = m.group("name"), m.group("alt")
    h = HEIGHTS.get(name, DEFAULT_H)
    return (
        f'<div class="chart-interactive">\n'
        f'          <iframe src="figures/{name}.html" height="{h}" '
        f'loading="lazy" title="{alt}"></iframe>\n'
        f'        </div>\n'
        f'        <img class="chart-static" src="figures/{name}.png" alt="{alt}" loading="lazy">'
    )


html, n_img = img_re.subn(repl, html)

# ── 2) Unificar el mapa (ya es iframe) para que tenga respaldo .png ───────────
# Solo si todavía no tiene la imagen de respaldo.
if 'class="chart-static" src="figures/map_pc1_2023.png"' not in html:
    map_iframe_re = re.compile(
        r'<iframe\s+src="figures/map_pc1_2023\.html".*?</iframe>',
        re.DOTALL,
    )
    h = HEIGHTS["map_pc1_2023"]
    map_alt = "Mapa interactivo: score PC1 por país, 2023"
    map_block = (
        f'<div class="chart-interactive">\n'
        f'          <iframe src="figures/map_pc1_2023.html" height="{h}" '
        f'loading="lazy" title="{map_alt}"></iframe>\n'
        f'        </div>\n'
        f'        <img class="chart-static" src="figures/map_pc1_2023.png" '
        f'alt="{map_alt}" loading="lazy">'
    )
    html, n_map = map_iframe_re.subn(map_block, html)
else:
    n_map = 0

INDEX.write_text(html, encoding="utf-8")
print(f"Figuras <img> convertidas: {n_img}")
print(f"Mapa unificado: {n_map}")
