"""
Versiones INTERACTIVAS (Plotly) de todas las figuras del informe.

Genera un HTML por figura en informe_final/figures/<nombre>.html, más una copia
única de la librería (plotly.min.js) en esa misma carpeta para que los gráficos
funcionen sin conexión. Los PNG estáticos (generados por eda/pca/clustering/
compare_years) se conservan como respaldo para impresión/PDF.

Reusa los datos ya calculados en data/processed/ y solo recalcula lo mínimo
(preprocesamiento, dendrograma, silueta, gap) con las mismas semillas.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from scipy.stats import skew
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score

try:
    from . import config, prep, theme
except ImportError:
    import config, prep, theme

T = theme

# ── Salida ──────────────────────────────────────────────────────────────────
INFORME_FIG = config.ROOT / "informe_final" / "figures"
INFORME_FIG.mkdir(parents=True, exist_ok=True)

# ── Etiquetas ────────────────────────────────────────────────────────────────
LABELS = config.INDICATORS
SHORT = {
    "NY.GDP.PCAP.KD": "PBI p/c", "NY.GDP.MKTP.KD.ZG": "Crec. PBI",
    "FP.CPI.TOTL.ZG": "Inflación", "SL.UEM.TOTL.ZS": "Desempleo",
    "NE.GDI.TOTL.ZS": "Cap. fijo", "NE.EXP.GNFS.ZS": "Export.",
    "NE.IMP.GNFS.ZS": "Import.", "NV.AGR.TOTL.ZS": "Agricultura",
    "NV.IND.TOTL.ZS": "Industria", "NV.SRV.TOTL.ZS": "Servicios",
    "SP.URB.TOTL.IN.ZS": "Urbanización", "SP.DYN.LE00.IN": "Esp. vida",
    "IT.NET.USER.ZS": "Internet", "EN.GHG.CO2.PC.CE.AR5": "CO2 p/c",
}

# ── Colores del tema ─────────────────────────────────────────────────────────
NEGRO, GRIS_MEDIO, GRIS_CLARO = T.NEGRO, T.GRIS_MEDIO, T.GRIS_CLARO
AZUL, BERMELLON, NARANJA = T.AZUL, T.BERMELLON, T.NARANJA
GRID = GRIS_CLARO


def _diverging_scale():
    """colorscale de Plotly a partir del colormap divergente del tema."""
    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = []
    for p in pos:
        r, g, b, _ = T.DIVERGING(p)
        out.append([p, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])
    return out


DIVSCALE = _diverging_scale()


# ── Helpers de layout y guardado ─────────────────────────────────────────────
def _layout(title=None, height=460, **kw):
    lay = dict(
        font=dict(family="Arial, sans-serif", size=12, color=NEGRO),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=30, t=72 if title else 28, b=55),
        height=height,
        legend=dict(font=dict(size=11)),
        hoverlabel=dict(font=dict(family="Arial, sans-serif", size=12)),
    )
    if title:
        lay["title"] = dict(text=title, font=dict(size=15, color=NEGRO),
                            x=0.5, xanchor="center")
    lay.update(kw)
    return lay


def _style_axes(fig):
    fig.update_xaxes(gridcolor=GRID, linecolor=GRIS_MEDIO, showline=True,
                     zeroline=False, ticks="outside", tickcolor=GRIS_MEDIO)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRIS_MEDIO, showline=True,
                     zeroline=False, ticks="outside", tickcolor=GRIS_MEDIO)
    return fig


def _save(fig, name):
    path = INFORME_FIG / f"{name}.html"
    fig.write_html(
        path,
        include_plotlyjs="plotly.min.js",   # referencia local (offline)
        full_html=True,
        config=dict(
            displaylogo=False,
            displayModeBar=True,
            modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d",
                                    "zoomIn2d", "zoomOut2d"],
            responsive=True,
        ),
    )
    print(f"  ✓ {name}.html")


# ── Carga de datos compartida ────────────────────────────────────────────────
def _load():
    """Reconstruye/lee todo lo necesario una sola vez."""
    d = {}
    year = config.YEAR_MODERN
    num, cats = prep.load_wide(year)
    d["num"] = num                       # crudas (para histogramas)
    d["cats"] = cats
    pre = prep.Preprocessor().fit(num)
    d["std"] = pre.transform(num)        # transformadas+estandarizadas (corr)
    imp = prep.impute_knn(d["std"])
    d["imp"] = imp                       # X_full (clustering)
    d["cols"] = config.INDICATORS_FINAL

    d["scores"] = pd.read_csv(config.DATA_PROC / f"pca_scores_{year}.csv")
    d["loadings"] = pd.read_csv(config.DATA_PROC / f"pca_loadings_{year}.csv",
                                index_col=0)
    d["eig"] = pd.read_csv(config.DATA_PROC / f"pca_autovalores_{year}.csv")
    with open(config.DATA_PROC / f"pca_info_{year}.json", encoding="utf-8") as f:
        d["pca_info"] = json.load(f)
    with open(config.DATA_PROC / f"clustering_resultados_{year}.json",
              encoding="utf-8") as f:
        d["clu"] = json.load(f)
    d["clusters"] = pd.read_csv(config.DATA_PROC / f"clusters_{year}.csv")
    d["traj"] = pd.read_csv(config.DATA_PROC / "trayectorias.csv", index_col=0)
    d["trans"] = pd.read_csv(config.DATA_PROC / "transiciones_cluster.csv",
                             index_col=0)
    d["dec"] = pd.read_csv(config.DATA_PROC / "descomposicion_dPC1.csv")
    d["load_yr"] = pd.read_csv(config.DATA_PROC / "loadings_pc1_por_anio.csv",
                               index_col=0)
    return d


# ═════════════════════════════════════════════════════════════════════════════
# EDA
# ═════════════════════════════════════════════════════════════════════════════
def fig_histograms(d):
    cols = d["cols"]
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    titles = [f"{LABELS[v]}<br><sub>asimetría = {skew(d['num'][v].dropna()):.2f}</sub>"
              for v in cols]
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles,
                        vertical_spacing=0.09, horizontal_spacing=0.06)
    for i, v in enumerate(cols):
        r, c = i // ncol + 1, i % ncol + 1
        data = d["num"][v].dropna()
        fig.add_trace(
            go.Histogram(x=data, nbinsx=25, marker_color=AZUL,
                         marker_line_color="white", marker_line_width=0.6,
                         hovertemplate=f"{SHORT[v]}<br>rango: %{{x}}<br>n: %{{y}}<extra></extra>",
                         showlegend=False),
            row=r, col=c)
    fig.update_layout(**_layout(f"Distribuciones univariadas — {config.YEAR_MODERN}",
                                height=235 * nrow, bargap=0.05))
    _style_axes(fig)
    for ann in fig.layout.annotations:
        ann.font.size = 10
    _save(fig, "eda_histogramas_2023")


def fig_correlation(d):
    cols = d["cols"]
    corr = d["std"][cols].corr(method="pearson")
    labels = [SHORT[c] for c in cols]
    n = len(cols)
    z = corr.values
    # listas nativas con None en el triángulo superior (evita encoding base64 2D)
    zmask = [[None if j > i else float(z[i, j]) for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=zmask, x=labels, y=labels,
        colorscale=DIVSCALE, zmid=0, zmin=-1, zmax=1,
        xgap=2, ygap=2,
        colorbar=dict(title="r", len=0.7, thickness=14),
        hovertemplate="%{y} – %{x}<br>r = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        f"Correlaciones de Pearson entre las 14 variables transformadas — {config.YEAR_MODERN}",
        height=720))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-40)
    # anotar r en cada celda del triángulo inferior
    anns = []
    for i in range(n):
        for j in range(i + 1):          # triángulo inferior + diagonal
            rval = float(z[i, j])
            anns.append(dict(x=labels[j], y=labels[i], text=f"{rval:.2f}",
                             showarrow=False,
                             font=dict(size=8,
                                       color="white" if abs(rval) > 0.5 else NEGRO)))
    fig.update_layout(annotations=anns)
    _save(fig, "eda_correlacion_2023")


def fig_categoricals(d):
    cats = d["cats"]
    reg = cats["region"].value_counts()
    inc = cats["income"].value_counts().reindex(T.INCOME_ORDER).dropna()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"Países por región — {config.YEAR_MODERN}",
                                        f"Países por nivel de ingreso — {config.YEAR_MODERN}"),
                        horizontal_spacing=0.18)
    fig.add_trace(go.Bar(
        y=reg.index[::-1], x=reg.values[::-1], orientation="h",
        marker_color=AZUL, marker_line_color="white", marker_line_width=0.6,
        hovertemplate="%{y}<br>%{x} países<extra></extra>", showlegend=False),
        row=1, col=1)
    fig.add_trace(go.Bar(
        x=[T.INCOME_LABELS.get(i, i) for i in inc.index], y=inc.values,
        marker_color=AZUL, marker_line_color="white", marker_line_width=0.6,
        hovertemplate="%{x}<br>%{y} países<extra></extra>", showlegend=False),
        row=1, col=2)
    fig.update_layout(**_layout(height=480))
    _style_axes(fig)
    _save(fig, "eda_categoricas_2023")


def fig_boxplots_income(d):
    df = d["num"].join(d["cats"])
    keyvars = ["NY.GDP.PCAP.KD", "SP.DYN.LE00.IN", "IT.NET.USER.ZS", "NV.AGR.TOTL.ZS"]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[LABELS[v] for v in keyvars],
                        vertical_spacing=0.12, horizontal_spacing=0.1)
    for idx, v in enumerate(keyvars):
        r, c = idx // 2 + 1, idx % 2 + 1
        for lv in T.INCOME_ORDER:
            vals = df.loc[df["income"] == lv, v].dropna()
            fig.add_trace(go.Box(
                y=vals, name=T.INCOME_LABELS[lv],
                marker_color=T.AZUL, fillcolor=T.CELESTE,
                line=dict(color=T.AZUL, width=1.2),
                marker=dict(size=3, color=GRIS_MEDIO),
                boxpoints="outliers", showlegend=False,
                hovertemplate=f"{T.INCOME_LABELS[lv]}<br>%{{y:.1f}}<extra></extra>"),
                row=r, col=c)
    fig.update_layout(**_layout(f"Variables clave por nivel de ingreso — {config.YEAR_MODERN}",
                                height=720))
    _style_axes(fig)
    _save(fig, "eda_boxplots_ingreso_2023")


# ═════════════════════════════════════════════════════════════════════════════
# PCA
# ═════════════════════════════════════════════════════════════════════════════
def fig_scree(d):
    eig = d["eig"]
    p = len(eig)
    x = list(range(1, p + 1))
    n_keep = d["pca_info"]["parallel_analysis"]
    evr = np.array(d["pca_info"]["evr"])
    cum = np.cumsum(evr)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                        subplot_titles=("Scree plot y análisis paralelo de Horn",
                                        "Varianza explicada y acumulada"))
    # panel 1
    fig.add_vrect(x0=0.5, x1=n_keep + 0.5, fillcolor=AZUL, opacity=0.08,
                  line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=eig["autovalor_obs"], mode="lines+markers",
                             name="Autovalores observados", line=dict(color=AZUL, width=2),
                             marker=dict(size=7),
                             hovertemplate="PC%{x}<br>autovalor = %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=eig["umbral_horn_p95"], mode="lines+markers",
                             name="Umbral Horn (p95)", line=dict(color=BERMELLON, width=2, dash="dash"),
                             marker=dict(size=6, symbol="square"),
                             hovertemplate="PC%{x}<br>umbral = %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hline(y=1, line=dict(color=GRIS_MEDIO, dash="dot"), row=1, col=1,
                  annotation_text="Kaiser = 1", annotation_position="top right",
                  annotation_font_size=10)
    # panel 2
    bar_colors = [AZUL if i < n_keep else GRIS_CLARO for i in range(p)]
    fig.add_trace(go.Bar(x=x, y=evr * 100, marker_color=bar_colors,
                         marker_line_color="white", marker_line_width=0.5,
                         name="Varianza individual",
                         hovertemplate="PC%{x}<br>%{y:.1f}%<extra></extra>"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=cum * 100, mode="lines+markers",
                             name="Varianza acumulada", line=dict(color=BERMELLON, width=2),
                             marker=dict(size=6),
                             hovertemplate="PC%{x}<br>acum. %{y:.1f}%<extra></extra>"),
                  row=1, col=2)
    fig.add_hline(y=config.VAR_CUM_TARGET * 100, line=dict(color=GRIS_MEDIO, dash="dot"),
                  row=1, col=2, annotation_text=f"{int(config.VAR_CUM_TARGET*100)}%",
                  annotation_position="bottom right", annotation_font_size=10)
    fig.update_xaxes(title_text="Componente", row=1, col=1)
    fig.update_xaxes(title_text="Componente", row=1, col=2)
    fig.update_yaxes(title_text="Autovalor", row=1, col=1)
    fig.update_yaxes(title_text="Varianza (%)", row=1, col=2)
    fig.update_layout(**_layout(f"PCA — retención de componentes ({config.YEAR_MODERN})",
                                height=480, legend=dict(orientation="h", y=-0.18,
                                                        font=dict(size=10))))
    _style_axes(fig)
    _save(fig, "pca_scree_2023")


def fig_loadings_bars(d):
    load = d["loadings"].iloc[:, :3].copy()
    load = load.reindex(load["PC1"].abs().sort_values().index)
    evr = np.array(d["pca_info"]["evr"])
    titles = [f"PC1 ({evr[0]*100:.1f}%)<br><sub>Gradiente de desarrollo</sub>",
              f"PC2 ({evr[1]*100:.1f}%)<br><sub>Estructura productiva</sub>",
              f"PC3 ({evr[2]*100:.1f}%)<br><sub>Apertura comercial</sub>"]
    ylabels = [SHORT.get(i, i) for i in load.index]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.08, shared_yaxes=False)
    for ci, col in enumerate(["PC1", "PC2", "PC3"], start=1):
        vals = load[col].values
        colors = [AZUL if v >= 0 else BERMELLON for v in vals]
        fig.add_trace(go.Bar(
            y=ylabels, x=vals, orientation="h", marker_color=colors,
            marker_line_color="white", marker_line_width=0.6, showlegend=False,
            text=[f"{v:.2f}" for v in vals], textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{y}<br>" + col + " = %{x:.2f}<extra></extra>"),
            row=1, col=ci)
        fig.update_xaxes(range=[-1.15, 1.15], title_text="Carga (r)", row=1, col=ci)
    fig.update_layout(**_layout(f"Cargas de los tres primeros componentes — {config.YEAR_MODERN}",
                                height=620))
    _style_axes(fig)
    for ann in fig.layout.annotations:
        ann.font.size = 11
    _save(fig, "pca_loadings_barras_2023")


def fig_correlation_circle(d):
    load = d["loadings"]
    evr = np.array(d["pca_info"]["evr"])

    def blend(c_from, c_to, t):
        from matplotlib.colors import to_rgb
        a = np.array(to_rgb(c_from)); b = np.array(to_rgb(c_to))
        r, g, bb = (1 - t) * a + t * b
        return f"rgb({int(r*255)},{int(g*255)},{int(bb*255)})"

    fig = go.Figure()
    # círculos guía
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                             line=dict(color=GRIS_MEDIO, width=1), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=0.6 * np.cos(th), y=0.6 * np.sin(th), mode="lines",
                             line=dict(color=GRIS_CLARO, width=1, dash="dash"),
                             hoverinfo="skip", showlegend=False))
    anns = []
    hov_x, hov_y, hov_t, hov_c = [], [], [], []
    for v in load.index:
        lx, ly = float(load.loc[v, "PC1"]), float(load.loc[v, "PC2"])
        norm = float(np.hypot(lx, ly))
        t = abs(lx) / norm if norm > 0 else 0.0
        base = AZUL if lx >= 0 else NARANJA
        color = blend(NEGRO, base, t)
        anns.append(dict(x=lx, y=ly, ax=0, ay=0, xref="x", yref="y",
                         axref="x", ayref="y", showarrow=True, arrowhead=2,
                         arrowsize=1.2, arrowwidth=2, arrowcolor=color))
        anns.append(dict(x=lx * 1.08, y=ly * 1.08, text=SHORT.get(v, v),
                         showarrow=False, font=dict(size=9, color=color),
                         xanchor="left" if lx >= 0 else "right"))
        hov_x.append(lx); hov_y.append(ly); hov_c.append(color)
        hov_t.append(f"<b>{SHORT.get(v, v)}</b><br>PC1 = {lx:.2f}<br>PC2 = {ly:.2f}")
    fig.add_trace(go.Scatter(x=hov_x, y=hov_y, mode="markers",
                             marker=dict(size=10, color=hov_c), text=hov_t,
                             hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.add_hline(y=0, line=dict(color=GRIS_MEDIO, width=0.6, dash="dash"))
    fig.add_vline(x=0, line=dict(color=GRIS_MEDIO, width=0.6, dash="dash"))
    fig.update_layout(**_layout(f"Círculo de correlaciones (PC1-PC2) — {config.YEAR_MODERN}",
                                height=640, annotations=anns))
    fig.update_xaxes(range=[-1.3, 1.3], title_text=f"PC1 — gradiente de desarrollo ({evr[0]*100:.1f}%)",
                     constrain="domain")
    fig.update_yaxes(range=[-1.3, 1.3], title_text=f"PC2 — estructura productiva ({evr[1]*100:.1f}%)",
                     scaleanchor="x", scaleratio=1)
    _style_axes(fig)
    _save(fig, "pca_circulo_correlaciones_2023")


def fig_biplot(d):
    sc = d["scores"]
    load = d["loadings"]
    evr = np.array(d["pca_info"]["evr"])
    s0, s1 = sc["PC1"].values, sc["PC2"].values

    fig = go.Figure()
    for lev in T.INCOME_ORDER:
        m = sc["income"] == lev
        fig.add_trace(go.Scatter(
            x=s0[m], y=s1[m], mode="markers", name=T.INCOME_LABELS[lev],
            marker=dict(size=8, color=T.INCOME_COLORS[lev],
                        line=dict(color="white", width=0.5)),
            text=sc.loc[m, "country"],
            customdata=np.stack([s0[m], s1[m]], axis=1).tolist(),
            hovertemplate="<b>%{text}</b><br>PC1 = %{customdata[0]:.2f}"
                          "<br>PC2 = %{customdata[1]:.2f}<extra></extra>"))
    # flechas de cargas
    sx = 0.9 * np.abs(s0).max() / np.abs(load["PC1"]).max()
    sy = 0.9 * np.abs(s1).max() / np.abs(load["PC2"]).max()
    scale = min(sx, sy)
    anns = []
    for v in load.index:
        tx, ty = float(load.loc[v, "PC1"]) * scale, float(load.loc[v, "PC2"]) * scale
        anns.append(dict(x=tx, y=ty, ax=0, ay=0, xref="x", yref="y",
                         axref="x", ayref="y", showarrow=True, arrowhead=2,
                         arrowsize=1, arrowwidth=1.3, arrowcolor="rgba(42,54,59,0.6)"))
        anns.append(dict(x=tx * 1.07, y=ty * 1.07, text=SHORT.get(v, v),
                         showarrow=False, font=dict(size=8, color=NEGRO),
                         xanchor="left" if tx >= 0 else "right"))
    fig.add_hline(y=0, line=dict(color=GRIS_MEDIO, width=0.6, dash="dash"))
    fig.add_vline(x=0, line=dict(color=GRIS_MEDIO, width=0.6, dash="dash"))
    fig.update_layout(**_layout(f"Biplot del PCA coloreado por nivel de ingreso — {config.YEAR_MODERN}",
                                height=680, annotations=anns,
                                legend=dict(title="Nivel de ingreso", font=dict(size=10))))
    fig.update_xaxes(title_text=f"PC1 — gradiente de desarrollo ({evr[0]*100:.1f}%)")
    fig.update_yaxes(title_text=f"PC2 — estructura productiva ({evr[1]*100:.1f}%)")
    _style_axes(fig)
    _save(fig, "pca_biplot_ingreso_2023")


# ═════════════════════════════════════════════════════════════════════════════
# Clustering
# ═════════════════════════════════════════════════════════════════════════════
def _cluster_style(k):
    if k <= 2:
        return T.CLUSTER2_COLORS, T.CLUSTER2_LABELS
    return T.CLUSTER3_COLORS, T.CLUSTER3_LABELS


def _aligned_labels(d, col):
    clusters = d["clusters"].set_index("countryiso3code")
    return clusters.reindex(d["imp"].index)[col].values


def fig_metrics_k(d, gap_df, k_opt=2):
    mk = pd.DataFrame(d["clu"]["metricas_k"])
    specs = [
        ("inercia", "Codo (inercia)", mk["k"], mk["inercia"]),
        ("silhouette", "Silhouette (↑ mejor)", mk["k"], mk["silhouette"]),
        ("calinski_harabasz", "Calinski-Harabasz (↑ mejor)", mk["k"], mk["calinski_harabasz"]),
        ("davies_bouldin", "Davies-Bouldin (↓ mejor)", mk["k"], mk["davies_bouldin"]),
        ("gap", "Gap statistic (↑ mejor)", gap_df["k"], gap_df["gap"]),
    ]
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.14, horizontal_spacing=0.09,
                        subplot_titles=[s[1] for s in specs] + [""])
    for i, (key, tit, xk, yv) in enumerate(specs):
        r, c = i // 3 + 1, i % 3 + 1
        xk = np.asarray(xk); yv = np.asarray(yv)
        fig.add_trace(go.Scatter(x=xk, y=yv, mode="lines+markers",
                                 line=dict(color=AZUL, width=2), marker=dict(size=7),
                                 showlegend=False,
                                 hovertemplate="k = %{x}<br>%{y:.3f}<extra></extra>"),
                      row=r, col=c)
        hit = np.where(xk == k_opt)[0]
        if len(hit):
            fig.add_trace(go.Scatter(x=[k_opt], y=[yv[hit[0]]], mode="markers",
                                     marker=dict(size=18, color=BERMELLON,
                                                 line=dict(color="white", width=2)),
                                     showlegend=False, hoverinfo="skip"),
                          row=r, col=c)
        fig.update_xaxes(title_text="k", row=r, col=c)
    # celda 6 (vacía): leyenda textual en coordenadas de papel (bottom-right)
    fig.add_annotation(x=0.84, y=0.18, xref="paper", yref="paper",
                       text=f"<b>k = {k_opt}</b> elegido<br>(consenso de criterios)",
                       showarrow=False, align="center", font=dict(size=13, color=NEGRO))
    fig.update_xaxes(visible=False, row=2, col=3)
    fig.update_yaxes(visible=False, row=2, col=3)
    fig.update_layout(**_layout("Selección del número de conglomerados k — criterios internos",
                                height=720))
    _style_axes(fig)
    _save(fig, "clust_metricas_k_2023")


def fig_dendrogram(d, k=2):
    from scipy.cluster.hierarchy import set_link_color_palette
    X = d["imp"].values
    Z = linkage(X, method="ward")
    thr = (Z[-1, 2] + Z[-2, 2]) / 2
    set_link_color_palette([T.CLUSTER2_COLORS[0], T.CLUSTER2_COLORS[1]])
    dd = dendrogram(Z, no_plot=True, color_threshold=thr,
                    above_threshold_color=GRIS_MEDIO)
    set_link_color_palette(None)

    fig = go.Figure()
    for xs, ys, col in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color=col, width=1.1),
                                 hoverinfo="skip", showlegend=False))
    fig.add_hline(y=thr, line=dict(color=NEGRO, dash="dot", width=1.2),
                  annotation_text=f"corte k={k}", annotation_position="top right",
                  annotation_font_size=11)
    fig.update_layout(**_layout(f"Dendrograma (Ward) — variables completas, {config.YEAR_MODERN}",
                                height=520))
    fig.update_xaxes(title_text="Países", showticklabels=False, showgrid=False)
    fig.update_yaxes(title_text="Distancia de fusión")
    _style_axes(fig)
    _save(fig, "clust_dendrograma_2023")


def fig_silhouette(d, k=2):
    X = d["imp"].values
    lab = _aligned_labels(d, "cluster_k2" if k == 2 else "cluster_k3").astype(int)
    countries = d["clusters"].set_index("countryiso3code").reindex(d["imp"].index)["country"].values
    sv = silhouette_samples(X, lab)
    avg = silhouette_score(X, lab)
    colors, names = _cluster_style(k)

    fig = go.Figure()
    y0 = 0
    for c in range(k):
        idx = np.where(lab == c)[0]
        order = np.argsort(sv[idx])
        vals = sv[idx][order]
        cs = countries[idx][order]
        ypos = np.arange(y0, y0 + len(vals))
        fig.add_trace(go.Bar(
            x=vals, y=ypos, orientation="h", marker_color=colors[c],
            marker_line_width=0, name=f"{names[c]} (n={len(vals)})",
            text=cs, hovertemplate="%{text}<br>silhouette = %{x:.3f}<extra></extra>"))
        y0 += len(vals) + 6
    fig.add_vline(x=avg, line=dict(color=NEGRO, dash="dash", width=1.4),
                  annotation_text=f"promedio = {avg:.3f}", annotation_position="top right",
                  annotation_font_size=11)
    fig.add_vline(x=0, line=dict(color=GRIS_MEDIO, width=0.9))
    fig.update_layout(**_layout(f"Diagrama de silhouette (k = {k})", height=560,
                                bargap=0, legend=dict(font=dict(size=10))))
    fig.update_xaxes(title_text="Coeficiente de silhouette")
    fig.update_yaxes(title_text="Países (ordenados por conglomerado)", showticklabels=False)
    _style_axes(fig)
    _save(fig, f"clust_silhouette_k{k}_2023")


def fig_clusters_on_pca(d, k):
    df = d["clusters"]
    evr = np.array(d["pca_info"]["evr"])
    col = "cluster_k2" if k == 2 else "cluster_k3"
    colors, names = _cluster_style(k)
    fig = go.Figure()
    for c in sorted(df[col].unique()):
        m = df[col] == c
        fig.add_trace(go.Scatter(
            x=df.loc[m, "PC1"], y=df.loc[m, "PC2"], mode="markers",
            name=f"{names[c]} (n={int(m.sum())})",
            marker=dict(size=9, color=colors[c], line=dict(color="white", width=0.5)),
            text=df.loc[m, "country"],
            customdata=df.loc[m, ["PC1", "PC2"]].values.tolist(),
            hovertemplate="<b>%{text}</b><br>PC1 = %{customdata[0]:.2f}"
                          "<br>PC2 = %{customdata[1]:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GRIS_MEDIO, width=0.6, dash="dash"))
    fig.add_vline(x=0, line=dict(color=GRIS_MEDIO, width=0.6, dash="dash"))
    titulo = ("Conglomerados (k=2) sobre el plano del PCA" if k == 2
              else "Conglomerados (k=3) sobre el plano del PCA")
    fig.update_layout(**_layout(f"{titulo} — {config.YEAR_MODERN}", height=640,
                                legend=dict(title="Conglomerados", font=dict(size=10))))
    fig.update_xaxes(title_text=f"PC1 — gradiente de desarrollo ({evr[0]*100:.1f}%)")
    fig.update_yaxes(title_text=f"PC2 — estructura productiva ({evr[1]*100:.1f}%)")
    _style_axes(fig)
    _save(fig, f"clust_pca_k{k}_2023")


# ═════════════════════════════════════════════════════════════════════════════
# Comparación temporal
# ═════════════════════════════════════════════════════════════════════════════
RELEVANTES = ["CHN", "IND", "USA", "BRA", "RUS", "IDN", "NGA", "KOR", "VNM",
              "POL", "TUR", "ZAF", "MEX", "DEU"]


def fig_trajectories(d):
    traj = d["traj"]
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             line=dict(color=GRIS_MEDIO, dash="dash", width=1.2),
                             name="Sin cambio relativo", hoverinfo="skip"))
    bucket = {"Ascenso relativo (catch-up)": (AZUL, traj["dperc"] > 3),
              "Descenso relativo": (BERMELLON, traj["dperc"] < -3),
              "Estable": (GRIS_CLARO, traj["dperc"].abs() <= 3)}
    for name, (col, mask) in bucket.items():
        sub = traj[mask]
        fig.add_trace(go.Scatter(
            x=sub["perc_e"], y=sub["perc_m"], mode="markers", name=name,
            marker=dict(size=9, color=col, line=dict(color="white", width=0.5)),
            text=sub["country"],
            customdata=np.stack([sub["perc_e"], sub["perc_m"], sub["dperc"]], axis=1).tolist(),
            hovertemplate="<b>%{text}</b><br>percentil " + str(YE) + " = %{customdata[0]:.1f}"
                          "<br>percentil " + str(YM) + " = %{customdata[1]:.1f}"
                          "<br>Δ = %{customdata[2]:+.1f}<extra></extra>"))
    # etiquetas de países relevantes + grandes movimientos
    destacar = set(RELEVANTES) | set(traj["dperc"].abs().sort_values(ascending=False).head(8).index)
    anns = []
    for iso in destacar:
        if iso in traj.index:
            r = traj.loc[iso]
            anns.append(dict(x=r["perc_e"], y=r["perc_m"], text=r["country"],
                             showarrow=False, font=dict(size=8, color=NEGRO),
                             xshift=6, yshift=8, xanchor="left"))
    fig.update_layout(**_layout(f"Movilidad relativa en el gradiente de desarrollo — {YE} vs {YM}",
                                height=680, annotations=anns,
                                legend=dict(font=dict(size=10))))
    fig.update_xaxes(title_text=f"Percentil de desarrollo en {YE}", range=[-2, 102])
    fig.update_yaxes(title_text=f"Percentil de desarrollo en {YM}", range=[-2, 102])
    _style_axes(fig)
    _save(fig, "compare_trayectorias")


def fig_transition(d):
    trans = d["trans"]
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    M = trans.values.astype(int)
    dest_color = {0: BERMELLON, 1: AZUL}
    dest_name = {0: f"Terminó en desarrollo ({YM})", 1: f"Terminó desarrollado ({YM})"}
    row_label = {0: f"En desarrollo en {YE}<br>(n={M[0].sum()})",
                 1: f"Desarrollado en {YE}<br>(n={M[1].sum()})"}
    fig = go.Figure()
    for dcol in (0, 1):
        fig.add_trace(go.Bar(
            y=[row_label[0], row_label[1]], x=[M[0, dcol], M[1, dcol]],
            orientation="h", name=dest_name[dcol], marker_color=dest_color[dcol],
            marker_line_color="white", marker_line_width=1,
            text=[M[0, dcol] or "", M[1, dcol] or ""], textposition="inside",
            insidetextfont=dict(color="white", size=13),
            hovertemplate="%{y}<br>" + dest_name[dcol] + ": %{x}<extra></extra>"))
    fig.update_layout(**_layout(f"¿A dónde fue cada grupo? Transición de tier relativo {YE} → {YM}",
                                height=420, barmode="stack",
                                legend=dict(font=dict(size=10))))
    fig.update_xaxes(title_text="Cantidad de países")
    _style_axes(fig)
    _save(fig, "compare_transiciones")


def fig_decomposicion(d):
    dec = d["dec"].sort_values("pct_del_total")
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    colors = [BERMELLON if x < 0 else AZUL for x in dec["pct_del_total"]]
    cd = np.stack([dec["peso_PC1(w)"], dec["mean_dz"], dec["contrib_a_dPC1"]], axis=1).tolist()
    fig = go.Figure(go.Bar(
        y=dec["variable"], x=dec["pct_del_total"], orientation="h",
        marker_color=colors, marker_line_color="white", marker_line_width=0.6,
        text=[f"{p:.1f}%" for p in dec["pct_del_total"]], textposition="outside",
        textfont=dict(size=9), customdata=cd,
        hovertemplate="<b>%{y}</b><br>peso en PC1 (w) = %{customdata[0]:.3f}"
                      "<br>cambio medio (Δz̄) = %{customdata[1]:.3f}"
                      "<br>contribución (w·Δz̄) = %{customdata[2]:.4f}"
                      "<br><b>%{x:.1f}%</b> del total<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=NEGRO, width=0.8))
    fig.update_layout(**_layout(
        f"Composición del corrimiento ABSOLUTO de PC1 ({YE}→{YM})", height=560))
    fig.update_xaxes(title_text="Contribución al corrimiento medio de PC1 (%)")
    _style_axes(fig)
    _save(fig, "compare_descomposicion_dPC1")


def fig_loadings_compare(d):
    ly = d["load_yr"]
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    c05, c21 = f"PC1_{YE}", f"PC1_{YM}"
    lim = [min(ly[c05].min(), ly[c21].min()) - 0.12,
           max(ly[c05].max(), ly[c21].max()) + 0.12]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                             line=dict(color=GRIS_MEDIO, dash="dash"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=ly[c05], y=ly[c21], mode="markers+text",
        marker=dict(size=9, color=AZUL, line=dict(color="white", width=0.6)),
        text=[str(v)[:16] for v in ly.index], textposition="top center",
        textfont=dict(size=8), showlegend=False,
        customdata=np.stack([ly[c05], ly[c21]], axis=1).tolist(),
        hovertemplate="<b>%{customdata[0]:.2f}</b> (" + str(YE) + ")"
                      "<br><b>%{customdata[1]:.2f}</b> (" + str(YM) + ")<extra></extra>"))
    fig.update_layout(**_layout("Estabilidad de la estructura: cargas de PC1 por año",
                                height=620))
    fig.update_xaxes(title_text=f"Carga en PC1 ({YE})", range=lim)
    fig.update_yaxes(title_text=f"Carga en PC1 ({YM})", range=lim)
    _style_axes(fig)
    _save(fig, "compare_loadings_pc1")


# ═════════════════════════════════════════════════════════════════════════════
# Orquestación
# ═════════════════════════════════════════════════════════════════════════════
def run():
    print("Generando gráficos interactivos (Plotly)…")
    # librería local para uso offline
    (INFORME_FIG / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    d = _load()

    # EDA
    fig_histograms(d)
    fig_correlation(d)
    fig_categoricals(d)
    fig_boxplots_income(d)
    # PCA
    fig_scree(d)
    fig_loadings_bars(d)
    fig_correlation_circle(d)
    fig_biplot(d)
    # Clustering
    try:
        import clustering
    except ImportError:
        from . import clustering
    print("  (recalculando gap statistic…)")
    gap_df, kopt = clustering.gap_statistic(d["imp"].values)
    fig_metrics_k(d, gap_df, k_opt=2)
    fig_dendrogram(d)
    fig_silhouette(d, k=2)
    fig_clusters_on_pca(d, k=2)
    fig_clusters_on_pca(d, k=3)
    # Temporal
    fig_trajectories(d)
    fig_transition(d)
    fig_decomposicion(d)
    fig_loadings_compare(d)

    print("Listo.")


if __name__ == "__main__":
    run()
