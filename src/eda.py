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
import seaborn as sns
from scipy.stats import skew

try:
    from . import config
except ImportError:
    import config

sns.set_theme(style="whitegrid", context="notebook")

LABELS = config.INDICATORS


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
            "skew_2005": round(float(skew(num05[v].dropna())), 3),
            "skew_2021": round(float(skew(num21[v].dropna())), 3),
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
        ax.hist(data, bins=25, color="#4C72B0", edgecolor="white")
        ax.set_title(f"{LABELS[v]}\nskew={skew(data):.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Distribuciones univariadas — {year}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(config.FIGURES / f"eda_histogramas_{year}.png", dpi=config.FIG_DPI)
    plt.close(fig)


def plot_correlation(year=config.YEAR_MODERN):
    num, _ = load_wide(year)
    corr = num.corr(method="pearson")
    short = [LABELS[v][:22] for v in corr.columns]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, xticklabels=short, yticklabels=short,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Matriz de correlación de Pearson — {year}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"eda_correlacion_{year}.png", dpi=config.FIG_DPI)
    plt.close(fig)
    return corr


def plot_categoricals(year=config.YEAR_MODERN):
    num, cats = load_wide(year)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    cats["region"].value_counts().plot.barh(ax=axes[0], color="#55A868")
    axes[0].set_title(f"Países por región — {year}")
    axes[0].invert_yaxis()
    order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    inc = cats["income"].value_counts().reindex(order).dropna()
    inc.plot.bar(ax=axes[1], color="#C44E52")
    axes[1].set_title(f"Países por nivel de ingreso — {year}")
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"eda_categoricas_{year}.png", dpi=config.FIG_DPI)
    plt.close(fig)


def plot_bivariate_by_income(year=config.YEAR_MODERN):
    """Boxplots de variables clave por nivel de ingreso (numérica x categórica)."""
    num, cats = load_wide(year)
    df = num.join(cats)
    order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    keyvars = ["NY.GDP.PCAP.KD", "SP.DYN.LE00.IN", "IT.NET.USER.ZS", "NV.AGR.TOTL.ZS"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, v in zip(axes.ravel(), keyvars):
        sns.boxplot(data=df, x="income", y=v, order=order, ax=ax,
                    palette="viridis", hue="income", legend=False)
        ax.set_title(LABELS[v], fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle(f"Variables clave por nivel de ingreso — {year}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(config.FIGURES / f"eda_boxplots_ingreso_{year}.png", dpi=config.FIG_DPI)
    plt.close(fig)


def check_sector_closure():
    """¿agr+ind+srv suman ~constante? (cierre composicional / colinealidad)."""
    num21, _ = load_wide(config.YEAR_MODERN)
    s = num21[["NV.AGR.TOTL.ZS", "NV.IND.TOTL.ZS", "NV.SRV.TOTL.ZS"]].sum(axis=1)
    return s.describe()


def main():
    sk = skewness_table()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("== Asimetría (skewness) por variable ==")
    print(sk.to_string(index=False))
    sk.to_csv(config.DATA_PROC / "skewness.csv", index=False, encoding="utf-8")

    print("\n== Cierre composicional sectorial (agr+ind+srv), 2021 ==")
    print(check_sector_closure().round(2).to_string())

    for year in (config.YEAR_MODERN, config.YEAR_EARLY):
        plot_histograms(year)
    plot_correlation(config.YEAR_MODERN)
    plot_categoricals(config.YEAR_MODERN)
    plot_bivariate_by_income(config.YEAR_MODERN)
    print("\nFiguras EDA guardadas en reports/figures/. OK.")


if __name__ == "__main__":
    main()
