"""
Análisis exploratorio (EDA): univariado + bivariado + categóricas.

Objetivo metodológico: MIRAR cada variable antes de transformar. Reporta
asimetría (skewness) por variable, histogramas, matriz de correlación, y
descripción de las categóricas (región, ingreso). También chequea el cierre
composicional de las participaciones sectoriales (agr+ind+srv).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import skew

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


def load_wide(year):
    df = pd.read_csv(config.DATA_PROC / f"wide_{year}.csv", index_col="countryiso3code")
    num = df[config.INDICATORS_FINAL]
    cats = df[["country", "region", "income"]]
    return num, cats


def skewness_table():
    """Asimetría por variable, en cada año y combinada (sobre datos observados)."""
    num21, _ = load_wide(config.YEAR_MODERN)
    num05, _ = load_wide(config.YEAR_EARLY)
    comb = pd.concat([num05, num21], axis=0)
    rows = []
    for v in config.INDICATORS_FINAL:
        rows.append({
            "variable": v,
            "label": LABELS[v],
            f"skew_{config.YEAR_EARLY}": round(float(skew(num05[v].dropna())), 3),
            f"skew_{config.YEAR_MODERN}": round(float(skew(num21[v].dropna())), 3),
            "skew_comb": round(float(skew(comb[v].dropna())), 3),
            "min_comb": round(float(comb[v].min()), 2),
            "max_comb": round(float(comb[v].max()), 2),
            "tiene_negativos": bool((comb[v] < 0).any()),
            "tiene_ceros": bool((comb[v] == 0).any()),
        })
    return pd.DataFrame(rows)


def plot_histograms(year=config.YEAR_MODERN):
    num, _ = load_wide(year)
    n = len(config.INDICATORS_FINAL)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.2 * nrow))
    for ax, v in zip(axes.ravel(), config.INDICATORS_FINAL):
        data = num[v].dropna()
        ax.hist(data, bins=25, color=T.VERDE_AGUA, edgecolor="white", linewidth=0.6)
        ax.set_title(f"{LABELS[v]}\nasimetría = {skew(data):.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Distribuciones univariadas — {year}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(config.FIGURES / f"eda_histogramas_{year}.png")
    plt.close(fig)


def plot_correlation(year=config.YEAR_MODERN):
    """Matriz de correlación de Pearson sobre las variables TRANSFORMADAS, con
    histogramas en la diagonal. Pearson sobre transformadas es coherente con el
    PCA (que también opera sobre las variables transformadas y estandarizadas)."""
    num, _ = load_wide(year)
    pre = prep.Preprocessor().fit(num)
    tdf = pre.transform(num)            # transformadas+estandarizadas (corr invariante)
    cols = config.INDICATORS_FINAL
    corr = tdf[cols].corr(method="pearson").values
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(16, 14.5))
    plt.subplots_adjust(hspace=0.07, wspace=0.07)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.25); sp.set_color("#d9d9d9")
            if i == j:
                ax.hist(tdf[cols[i]].dropna(), bins=16, color=T.AZUL,
                        alpha=0.85, linewidth=0)
                ax.set_facecolor(T.NEUTRO)
            elif i > j:                       # triángulo INFERIOR
                r = corr[i, j]
                ax.set_facecolor(T.DIVERGING((r + 1) / 2))
                ax.text(0.5, 0.5, f"{r:.2f}", ha="center", va="center",
                        fontsize=7, fontweight="bold", transform=ax.transAxes,
                        color="white" if abs(r) > 0.5 else T.NEGRO)
            else:                             # triángulo superior: vacío
                ax.set_facecolor("white")
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(SHORT[cols[i]], fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=38)
            if i == n - 1:
                ax.set_xlabel(SHORT[cols[j]], fontsize=7, rotation=40,
                              ha="right", va="top")
    leg = [
        Patch(facecolor=T.DIVERGING(0.92), label="Correlación positiva"),
        Patch(facecolor=T.DIVERGING(0.08), label="Correlación negativa"),
        Patch(facecolor=T.NEUTRO, label="Distribución marginal (diagonal)"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Correlaciones de Pearson entre las 14 variables transformadas — {year}\n"
        "Triángulo inferior: coeficiente r  ·  Diagonal: distribución de cada variable",
        fontsize=13, y=1.01)
    fig.savefig(config.FIGURES / f"eda_correlacion_{year}.png")
    plt.close(fig)
    return corr


def plot_categoricals(year=config.YEAR_MODERN):
    num, cats = load_wide(year)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    cats["region"].value_counts().plot.barh(ax=axes[0], color=T.LAVANDA,
                                             edgecolor="white")
    axes[0].set_title(f"Países por región — {year}")
    axes[0].invert_yaxis()
    inc = cats["income"].value_counts().reindex(T.INCOME_ORDER).dropna()
    axes[1].bar(range(len(inc)), inc.values,
                color=T.income_palette(inc.index), edgecolor="white")
    axes[1].set_xticks(range(len(inc)))
    axes[1].set_xticklabels([T.INCOME_LABELS.get(i, i) for i in inc.index],
                            rotation=15, fontsize=9)
    axes[1].set_title(f"Países por nivel de ingreso — {year}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"eda_categoricas_{year}.png")
    plt.close(fig)


def plot_bivariate_by_income(year=config.YEAR_MODERN):
    """Boxplots de variables clave por nivel de ingreso (numérica x categórica)."""
    num, cats = load_wide(year)
    df = num.join(cats)
    keyvars = ["NY.GDP.PCAP.KD", "SP.DYN.LE00.IN", "IT.NET.USER.ZS", "NV.AGR.TOTL.ZS"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, v in zip(axes.ravel(), keyvars):
        groups = [df.loc[df["income"] == lv, v].dropna().values
                  for lv in T.INCOME_ORDER]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.6,
                        medianprops=dict(color=T.NEGRO, lw=1.4),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor=T.GRIS_MEDIO,
                                        markeredgecolor="none", alpha=0.6))
        for patch, lv in zip(bp["boxes"], T.INCOME_ORDER):
            patch.set_facecolor(T.INCOME_COLORS[lv]); patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(T.INCOME_ORDER) + 1))
        ax.set_xticklabels([T.INCOME_LABELS[lv] for lv in T.INCOME_ORDER],
                           rotation=15, fontsize=8)
        ax.set_title(LABELS[v], fontsize=10)
    fig.suptitle(f"Variables clave por nivel de ingreso — {year}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(config.FIGURES / f"eda_boxplots_ingreso_{year}.png")
    plt.close(fig)


def check_sector_closure():
    """¿agr+ind+srv suman ~constante? (cierre composicional / colinealidad)."""
    num, _ = load_wide(config.YEAR_MODERN)
    s = num[["NV.AGR.TOTL.ZS", "NV.IND.TOTL.ZS", "NV.SRV.TOTL.ZS"]].sum(axis=1)
    return s.describe()


def main():
    sk = skewness_table()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("== Asimetría (skewness) por variable ==")
    print(sk.to_string(index=False))
    sk.to_csv(config.DATA_PROC / "skewness.csv", index=False, encoding="utf-8")

    print(f"\n== Cierre composicional sectorial (agr+ind+srv), {config.YEAR_MODERN} ==")
    print(check_sector_closure().round(2).to_string())

    for year in (config.YEAR_MODERN, config.YEAR_EARLY):
        plot_histograms(year)
    plot_correlation(config.YEAR_MODERN)
    plot_categoricals(config.YEAR_MODERN)
    plot_bivariate_by_income(config.YEAR_MODERN)
    print("\nFiguras EDA guardadas en reports/figures/. OK.")


if __name__ == "__main__":
    main()
