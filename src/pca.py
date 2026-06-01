"""
PCA sobre la matriz transversal del año moderno (config.YEAR_MODERN; transformada,
estandarizada, imputada).

Decisiones clave:
  - Estandarización previa OBLIGATORIA (variables en escalas incomparables).
  - Retención de componentes por CONVERGENCIA de criterios, con énfasis en el
    ANÁLISIS PARALELO de Horn (implementado a mano), no solo Kaiser/80%.
  - Se distingue explícitamente:
        n_clust  = componentes para CLUSTERIZAR (los que indica el análisis
                   paralelo / ~80% de varianza)
        n_vis    = 2-3 componentes SOLO para visualizar
  - Loadings reportados como correlación variable-componente.
  - Las categóricas (región/ingreso) son ILUSTRATIVAS: colorean los scores, no
    intervienen en el ajuste del PCA.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from sklearn.decomposition import PCA

try:
    from . import config, prep, theme
except ImportError:
    import config, prep, theme

T = theme
T.apply_theme()
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


def parallel_analysis(X, nsim=None, pctl=None, random_state=None):
    """Análisis paralelo de Horn sobre la matriz de correlación.

    Compara los autovalores observados contra los de datos aleatorios normales
    de la MISMA dimensión (n x p). Retiene los componentes cuyo autovalor supera
    el percentil `pctl` de la distribución nula.
    """
    nsim = nsim or config.PARALLEL_ANALYSIS_NSIM
    pctl = pctl or config.PARALLEL_ANALYSIS_PCTL
    rs = config.RANDOM_STATE if random_state is None else random_state
    n, p = X.shape
    rng = np.random.default_rng(rs)
    obs = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    sim = np.empty((nsim, p))
    for i in range(nsim):
        R = rng.standard_normal((n, p))
        sim[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(R, rowvar=False)))[::-1]
    thresh = np.percentile(sim, pctl, axis=0)
    n_keep = int(np.sum(obs > thresh))
    return obs, thresh, n_keep


def fit_pca(X):
    pca = PCA(svd_solver="full", random_state=config.RANDOM_STATE)
    scores = pca.fit_transform(X.values)
    return pca, scores


def loadings_as_correlations(pca, columns):
    """Loadings = autovector * sqrt(autovalor) = corr(variable, componente)."""
    load = pca.components_.T * np.sqrt(pca.explained_variance_)
    cols = [f"PC{i+1}" for i in range(load.shape[1])]
    return pd.DataFrame(load, index=columns, columns=cols)


def plot_scree(obs, thresh, evr, n_keep, fname):
    p = len(obs)
    x = np.arange(1, p + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # panel izquierdo: autovalores vs umbral de Horn
    axes[0].axvspan(0.5, n_keep + 0.5, color=T.VERDE_AGUA, alpha=0.10, zorder=0)
    axes[0].plot(x, obs, "o-", label="Autovalores observados", color=T.DIV_POS, zorder=3)
    axes[0].plot(x, thresh, "s--", label="Umbral análisis paralelo (p95)",
                 color=T.CORAL, zorder=3)
    axes[0].axhline(1, color=T.GRIS_MEDIO, ls=":", lw=1, label="Kaiser (autovalor = 1)")
    axes[0].set_xlabel("Componente")
    axes[0].set_ylabel("Autovalor")
    axes[0].set_title("Scree plot y análisis paralelo de Horn")
    axes[0].text(n_keep / 2 + 0.5, obs.max() * 0.92,
                 f"{n_keep} retenidos\n(Horn)", ha="center", fontsize=9,
                 color=T.DIV_POS, fontweight="bold")
    axes[0].legend(fontsize=8)

    cum = np.cumsum(evr)
    bar_colors = [T.VERDE_AGUA if i < n_keep else T.GRIS_CLARO for i in range(p)]
    axes[1].bar(x, evr * 100, color=bar_colors, edgecolor="white",
                label="Varianza individual")
    axes[1].plot(x, cum * 100, "o-", color=T.CORAL, label="Varianza acumulada")
    axes[1].axhline(config.VAR_CUM_TARGET * 100, color=T.GRIS_MEDIO, ls=":",
                    label=f"{int(config.VAR_CUM_TARGET*100)}%")
    axes[1].set_xlabel("Componente")
    axes[1].set_ylabel("Varianza (%)")
    axes[1].set_title("Varianza explicada y acumulada")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def plot_loadings_bars(load, evr, fname, year=config.YEAR_MODERN):
    """Cargas de PC1-PC3 como barras horizontales diverging (azul +, coral -)."""
    data = load.iloc[:, :3].copy()
    data = data.reindex(data["PC1"].abs().sort_values().index)
    titulos = [
        f"PC1 ({evr[0]*100:.1f}%)\nGradiente de desarrollo",
        f"PC2 ({evr[1]*100:.1f}%)\nEstructura productiva",
        f"PC3 ({evr[2]*100:.1f}%)\nApertura comercial",
    ]
    # sharey=False + etiquetas en CADA panel: así el nombre de la variable queda
    # junto a su barra también en PC2 y PC3 (no solo en PC1).
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    ylabels = [SHORT.get(i, i) for i in data.index]
    for ax, col, tit in zip(axes, ["PC1", "PC2", "PC3"], titulos):
        vals = data[col].values
        colors = [T.POS_FILL if v >= 0 else T.NEG_FILL for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", height=0.74)
        ax.axvline(0, color=T.NEGRO, lw=0.8)
        for xv in (-0.5, 0.5):
            ax.axvline(xv, ls=":", color=T.GRIS_MEDIO, lw=0.7)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-0.7, len(data) - 0.3)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(ylabels, fontsize=8.5)
        ax.set_title(tit, fontsize=10, color=T.DIV_POS)
        ax.set_xlabel("Carga (r variable–componente)", fontsize=9)
        for k, v in enumerate(vals):
            ax.text(v + (0.04 if v >= 0 else -0.04), k, f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.5, color=T.NEGRO)
    fig.suptitle(f"Cargas de los tres primeros componentes — {year}", fontsize=13)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def plot_biplot(scores, load, evr, cats, fname, color_by="income", year=config.YEAR_MODERN):
    fig, ax = plt.subplots(figsize=(11.5, 9))
    cseries = cats[color_by].reindex(cats.index)
    if color_by == "income":
        for lev in T.INCOME_ORDER:
            m = (cseries == lev).values
            ax.scatter(scores[m, 0], scores[m, 1], s=40, color=T.INCOME_COLORS[lev],
                       label=T.INCOME_LABELS[lev], alpha=0.95, edgecolor="white", lw=0.4)
    else:
        levs = [l for l in cseries.dropna().unique()]
        for i, lev in enumerate(sorted(levs)):
            m = (cseries == lev).values
            ax.scatter(scores[m, 0], scores[m, 1], s=40,
                       color=T.CATEGORICAL[i % len(T.CATEGORICAL)],
                       label=lev, alpha=0.95, edgecolor="white", lw=0.4)
    # escala de las flechas y LÍMITES que garantizan que TODAS entren
    sx = 0.9 * np.abs(scores[:, 0]).max() / np.abs(load["PC1"]).max()
    sy = 0.9 * np.abs(scores[:, 1]).max() / np.abs(load["PC2"]).max()
    scale = min(sx, sy)
    tips = load[["PC1", "PC2"]].values * scale
    for v, (tx, ty) in zip(load.index, tips):
        ax.annotate("", xy=(tx, ty), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=T.NEGRO, alpha=0.6, lw=1.3))
        ax.text(tx * 1.06, ty * 1.06, SHORT.get(v, v), fontsize=8, color=T.NEGRO,
                ha="left" if tx >= 0 else "right",
                va="bottom" if ty >= 0 else "top")
    lim_x = 1.18 * max(np.abs(scores[:, 0]).max(), np.abs(tips[:, 0]).max())
    lim_y = 1.18 * max(np.abs(scores[:, 1]).max(), np.abs(tips[:, 1]).max())
    ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y)
    ax.axhline(0, color=T.GRIS_MEDIO, lw=0.6, ls="--")
    ax.axvline(0, color=T.GRIS_MEDIO, lw=0.6, ls="--")
    ax.set_xlabel(f"PC1 — gradiente de desarrollo ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 — estructura productiva ({evr[1]*100:.1f}%)")
    titulo = "nivel de ingreso" if color_by == "income" else "región"
    ax.set_title(f"Biplot del PCA coloreado por {titulo} — {year}")
    ax.legend(fontsize=8, title=titulo, loc="best")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def plot_correlation_circle(load, evr, fname, year=config.YEAR_MODERN):
    """Color CONTINUO según la carga en PC1: teal hacia la derecha (PC1 positivo,
    desarrollo) y oro hacia la izquierda (PC1 negativo, agricultura). Cuanto MENOS
    pesa PC1 y más PC2 (flecha más vertical), el color se DESATURA hacia el negro
    (la variable "pierde color" porque no define el gradiente de desarrollo)."""
    def _blend(c_from, c_to, t):           # t=0 -> c_from, t=1 -> c_to
        a = np.array(to_rgb(c_from)); b = np.array(to_rgb(c_to))
        return tuple((1 - t) * a + t * b)

    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=T.GRIS_MEDIO, lw=1.0)
    ax.plot(0.6 * np.cos(th), 0.6 * np.sin(th), color=T.GRIS_CLARO, lw=0.9, ls="--")
    ax.axhline(0, color=T.GRIS_MEDIO, lw=0.6, ls="--")
    ax.axvline(0, color=T.GRIS_MEDIO, lw=0.6, ls="--")
    for v in load.index:
        lx, ly = load.loc[v, "PC1"], load.loc[v, "PC2"]
        norm = float(np.hypot(lx, ly))
        t = abs(lx) / norm if norm > 0 else 0.0     # 1 = puro PC1, 0 = puro PC2
        base = T.DIV_POS if lx >= 0 else T.NARANJA   # derecha teal / izquierda oro
        color = _blend(T.NEGRO, base, t)             # menos PC1 -> pierde color
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.9))
        ax.text(lx * 1.07, ly * 1.07, SHORT.get(v, v), fontsize=8, color=color,
                fontweight="bold", ha="left" if lx >= 0 else "right",
                va="bottom" if ly >= 0 else "top")
    ax.set_xlim(-1.28, 1.28); ax.set_ylim(-1.28, 1.28)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 — gradiente de desarrollo ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 — estructura productiva ({evr[1]*100:.1f}%)")
    ax.set_title(f"Círculo de correlaciones (PC1-PC2) — {year}")
    legend = [
        Line2D([0], [0], color=T.DIV_POS, lw=3, label="Carga + en PC1 (→ desarrollo)"),
        Line2D([0], [0], color=T.NARANJA, lw=3, label="Carga − en PC1 (← agricultura)"),
        Line2D([0], [0], color=T.NEGRO, lw=3, label="Dominada por PC2 (pierde color)"),
        Line2D([0], [0], ls="--", color=T.GRIS_CLARO, lw=1, label="Umbral de calidad (r = 0,6)"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def run(year=config.YEAR_MODERN):
    imp, std, cats, pre = prep.prepare_single(year)
    pca, scores = fit_pca(imp)
    evr = pca.explained_variance_ratio_
    obs, thresh, n_keep = parallel_analysis(imp.values)

    # criterios de retención
    kaiser = int(np.sum(obs > 1))
    cum = np.cumsum(evr)
    n_80 = int(np.searchsorted(cum, config.VAR_CUM_TARGET) + 1)

    load = loadings_as_correlations(pca, imp.columns)
    score_df = pd.DataFrame(scores, index=imp.index,
                            columns=[f"PC{i+1}" for i in range(scores.shape[1])])

    # guardar resultados
    score_df.join(cats).to_csv(config.DATA_PROC / f"pca_scores_{year}.csv", encoding="utf-8")
    load.to_csv(config.DATA_PROC / f"pca_loadings_{year}.csv", encoding="utf-8")
    eig_df = pd.DataFrame({
        "componente": [f"PC{i+1}" for i in range(len(obs))],
        "autovalor_obs": obs,
        "umbral_horn_p95": thresh,
        "var_explicada": evr,
        "var_acumulada": cum,
    })
    eig_df.to_csv(config.DATA_PROC / f"pca_autovalores_{year}.csv", index=False, encoding="utf-8")

    info = {
        "year": year,
        "n_obs": int(imp.shape[0]),
        "n_vars": int(imp.shape[1]),
        "kaiser": kaiser,
        "n_80pct": n_80,
        "parallel_analysis": n_keep,
        "n_clust": n_keep,   # componentes para clusterizar
        "n_vis": 2,          # componentes para visualizar
        "var_acum_n_clust": float(cum[n_keep - 1]),
        "evr": [float(x) for x in evr],
    }
    with open(config.DATA_PROC / f"pca_info_{year}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # figuras
    plot_scree(obs, thresh, evr, n_keep, f"pca_scree_{year}.png")
    plot_loadings_bars(load, evr, f"pca_loadings_barras_{year}.png", year)
    plot_biplot(scores, load, evr, cats, f"pca_biplot_ingreso_{year}.png", "income", year)
    plot_biplot(scores, load, evr, cats, f"pca_biplot_region_{year}.png", "region", year)
    plot_correlation_circle(load, evr, f"pca_circulo_correlaciones_{year}.png", year)

    # impresión
    print(f"== PCA {year}: n={info['n_obs']} países, p={info['n_vars']} variables ==")
    print(f"Varianza explicada (primeros 6): {np.round(evr[:6]*100,1)}")
    print(f"Varianza acumulada (primeros 6): {np.round(cum[:6]*100,1)}")
    print(f"Retención -> Kaiser: {kaiser} | 80% var: {n_80} | Horn (paralelo): {n_keep}")
    print(f"  => n_clust = {n_keep} (captura {cum[n_keep-1]*100:.1f}% de varianza)")
    print("\nLoadings (correlación variable-componente), PC1-PC4:")
    print(load.iloc[:, :4].round(2).to_string())
    return info, load, eig_df


if __name__ == "__main__":
    run(config.YEAR_MODERN)
