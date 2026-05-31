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
import seaborn as sns
from sklearn.decomposition import PCA

try:
    from . import config, prep
except ImportError:
    import config, prep

sns.set_theme(style="whitegrid", context="notebook")
LABELS = config.INDICATORS


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
    axes[0].plot(x, obs, "o-", label="Autovalores observados", color="#4C72B0")
    axes[0].plot(x, thresh, "s--", label="Umbral análisis paralelo (p95)", color="#C44E52")
    axes[0].axhline(1, color="gray", ls=":", lw=1, label="Kaiser (autovalor=1)")
    axes[0].axvline(n_keep + 0.5, color="green", ls="-.", lw=1.2,
                    label=f"Retenidos por Horn = {n_keep}")
    axes[0].set_xlabel("Componente")
    axes[0].set_ylabel("Autovalor")
    axes[0].set_title("Scree plot + análisis paralelo de Horn")
    axes[0].legend(fontsize=8)

    cum = np.cumsum(evr)
    axes[1].bar(x, evr, color="#4C72B0", alpha=0.7, label="Varianza individual")
    axes[1].plot(x, cum, "o-", color="#DD8452", label="Varianza acumulada")
    axes[1].axhline(config.VAR_CUM_TARGET, color="gray", ls=":",
                    label=f"{int(config.VAR_CUM_TARGET*100)}%")
    axes[1].set_xlabel("Componente")
    axes[1].set_ylabel("Proporción de varianza")
    axes[1].set_title("Varianza explicada")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_biplot(scores, load, evr, cats, fname, color_by="income", year=config.YEAR_MODERN):
    order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    palette = dict(zip(order, sns.color_palette("viridis", 4)))
    fig, ax = plt.subplots(figsize=(11, 9))
    cseries = cats[color_by].reindex(cats.index)
    if color_by == "income":
        for lev in order:
            m = cseries == lev
            ax.scatter(scores[m.values, 0], scores[m.values, 1], s=35,
                       color=palette[lev], label=lev, alpha=0.8, edgecolor="white", lw=0.3)
    else:
        for lev, sub in cseries.groupby(cseries):
            m = cseries == lev
            ax.scatter(scores[m.values, 0], scores[m.values, 1], s=35, label=lev, alpha=0.8)
    # vectores de carga (escalados para visibilidad)
    scale = np.abs(scores[:, :2]).max() / np.abs(load.values[:, :2]).max() * 0.9
    for v in load.index:
        ax.arrow(0, 0, load.loc[v, "PC1"] * scale, load.loc[v, "PC2"] * scale,
                 color="#444444", alpha=0.6, head_width=0.12, length_includes_head=True)
        ax.text(load.loc[v, "PC1"] * scale * 1.08, load.loc[v, "PC2"] * scale * 1.08,
                LABELS[v][:18], fontsize=7, color="#222222")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% de varianza)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% de varianza)")
    ax.set_title(f"Biplot PCA (coloreado por {color_by}) — {year}")
    ax.legend(fontsize=8, title=color_by)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_correlation_circle(load, evr, fname, year=config.YEAR_MODERN):
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, color="gray", fill=False, ls="--")
    ax.add_artist(circle)
    for v in load.index:
        ax.arrow(0, 0, load.loc[v, "PC1"], load.loc[v, "PC2"],
                 color="#4C72B0", alpha=0.7, head_width=0.02, length_includes_head=True)
        ax.text(load.loc[v, "PC1"] * 1.06, load.loc[v, "PC2"] * 1.06,
                LABELS[v][:20], fontsize=7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(f"Círculo de correlaciones (PC1-PC2) — {year}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=config.FIG_DPI)
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
