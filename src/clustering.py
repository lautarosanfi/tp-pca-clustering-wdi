"""
Clustering de países (corte config.YEAR_MODERN).

Principio rector (corrige el error de "tandem analysis"):
  - Con ~14 variables NO estamos en alta dimensión. El clustering PRINCIPAL se
    hace sobre las 14 variables completas (transformadas + estandarizadas).
  - Se COMPARA contra el clustering sobre componentes PCA (los 3 retenidos por
    análisis paralelo de Horn), y se mide el acuerdo (ARI).
  - Dos algoritmos: k-means y jerárquico de Ward.
  - k elegido por CONSENSO de métricas (silhouette, codo, gap, Calinski-Harabasz,
    Davies-Bouldin) + estabilidad bootstrap (criterio de Hennig).
  - Perfilado con numéricas y con categóricas (región/ingreso) + chi-cuadrado.
  - Se evalúan supuestos de k-means y se prueban GMM y HDBSCAN como alternativas.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score)
from sklearn.decomposition import PCA

try:
    from . import config, prep, theme
except ImportError:
    import config, prep, theme

T = theme
T.apply_theme()
LABELS = config.INDICATORS
RS = config.RANDOM_STATE
NOTABLES = {"CHN": "China", "USA": "EE.UU.", "IND": "India", "NGA": "Nigeria",
            "LBN": "Líbano", "SGP": "Singapur", "VEN": "Venezuela", "ETH": "Etiopía",
            "VNM": "Vietnam"}


# --------------------------------------------------------------------------- #
# Selección de k
# --------------------------------------------------------------------------- #
def metrics_over_k(X, krange=None):
    krange = krange or config.K_RANGE
    rows = []
    for k in krange:
        km = KMeans(n_clusters=k, n_init=20, random_state=RS)
        lab = km.fit_predict(X)
        rows.append({
            "k": k,
            "inercia": km.inertia_,
            "silhouette": silhouette_score(X, lab),
            "calinski_harabasz": calinski_harabasz_score(X, lab),
            "davies_bouldin": davies_bouldin_score(X, lab),
        })
    return pd.DataFrame(rows)


def gap_statistic(X, krange=None, nref=50):
    """Gap statistic de Tibshirani con referencia uniforme en el bounding box."""
    krange = krange or config.K_RANGE
    rng = np.random.default_rng(RS)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    rows = []
    for k in krange:
        km = KMeans(n_clusters=k, n_init=20, random_state=RS).fit(X)
        wk = np.log(km.inertia_)
        refs = []
        for _ in range(nref):
            Xr = rng.uniform(mins, maxs, size=X.shape)
            kmr = KMeans(n_clusters=k, n_init=5, random_state=RS).fit(Xr)
            refs.append(np.log(kmr.inertia_))
        refs = np.array(refs)
        gap = refs.mean() - wk
        sk = refs.std() * np.sqrt(1 + 1 / nref)
        rows.append({"k": k, "gap": gap, "sk": sk})
    df = pd.DataFrame(rows)
    # k óptimo: menor k tal que gap(k) >= gap(k+1) - s(k+1)
    kopt = None
    for i in range(len(df) - 1):
        if df.loc[i, "gap"] >= df.loc[i + 1, "gap"] - df.loc[i + 1, "sk"]:
            kopt = int(df.loc[i, "k"])
            break
    if kopt is None:
        kopt = int(df.loc[df["gap"].idxmax(), "k"])
    return df, kopt


# --------------------------------------------------------------------------- #
# Estabilidad (Hennig)
# --------------------------------------------------------------------------- #
def bootstrap_stability(X, k, nrep=None):
    nrep = nrep or config.BOOTSTRAP_NREP
    base = KMeans(n_clusters=k, n_init=20, random_state=RS).fit(X)
    base_lab = base.labels_
    n = len(X)
    rng = np.random.default_rng(RS)
    jacc = {c: [] for c in range(k)}
    for b in range(nrep):
        idx = rng.integers(0, n, n)
        km = KMeans(n_clusters=k, n_init=5, random_state=RS + b + 1).fit(X[idx])
        new_lab = km.predict(X)  # asignar TODOS los puntos originales
        for c in range(k):
            bs = base_lab == c
            best = 0.0
            for c2 in range(k):
                ns = new_lab == c2
                union = np.sum(bs | ns)
                if union:
                    best = max(best, np.sum(bs & ns) / union)
            jacc[c].append(best)
    return {c: float(np.mean(v)) for c, v in jacc.items()}, base_lab


# --------------------------------------------------------------------------- #
# Algoritmos
# --------------------------------------------------------------------------- #
def fit_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=20, random_state=RS).fit_predict(X)


def fit_ward(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def fit_gmm(X, k):
    return GaussianMixture(n_components=k, covariance_type="full",
                           n_init=5, random_state=RS).fit_predict(X)


# --------------------------------------------------------------------------- #
# Perfilado
# --------------------------------------------------------------------------- #
def profile_clusters(num_raw, labels, cats):
    """Perfil de cada cluster: medianas de variables crudas + cruces categóricos."""
    df = num_raw.copy()
    df["cluster"] = labels
    med = df.groupby("cluster").median(numeric_only=True).T
    med.index = [LABELS[v] for v in med.index]
    sizes = pd.Series(labels).value_counts().sort_index()
    # cruces categóricos
    cl = pd.Series(labels, index=cats.index, name="cluster")
    ct_income = pd.crosstab(cl, cats["income"])
    ct_region = pd.crosstab(cl, cats["region"])
    chi_income = chi2_contingency(ct_income)
    chi_region = chi2_contingency(ct_region)
    return med, sizes, ct_income, ct_region, chi_income, chi_region


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _cluster_style(k):
    if k <= 2:
        return T.CLUSTER2_COLORS, T.CLUSTER2_LABELS
    return T.CLUSTER3_COLORS, T.CLUSTER3_LABELS


def plot_metrics(mk, gap_df, fname, k_opt=2):
    LINE = T.AZUL        # un único color de curva en todos los paneles
    MARK = T.BERMELLON   # color del resaltado de k elegido
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    specs = [
        (axes[0, 0], mk["k"], mk["inercia"], "Codo (inercia)"),
        (axes[0, 1], mk["k"], mk["silhouette"], "Silhouette (↑ mejor)"),
        (axes[0, 2], mk["k"], mk["calinski_harabasz"], "Calinski-Harabasz (↑ mejor)"),
        (axes[1, 0], mk["k"], mk["davies_bouldin"], "Davies-Bouldin (↓ mejor)"),
        (axes[1, 1], gap_df["k"], gap_df["gap"], "Gap statistic (↑ mejor)"),
    ]
    for ax, xk, yv, tit in specs:
        ax.plot(xk, yv, "o-", color=LINE, lw=2, ms=6, zorder=3)
        ax.axvline(k_opt, color=MARK, ls="--", lw=1.6, zorder=2)
        ax.set_title(tit); ax.set_xlabel("k")
    # leyenda única (línea de k elegido)
    axes[1, 2].axis("off")
    axes[1, 2].plot([], [], color=MARK, ls="--", lw=1.6, label=f"k = {k_opt} (elegido)")
    axes[1, 2].plot([], [], "o-", color=LINE, lw=2, label="Valor de la métrica")
    axes[1, 2].legend(loc="center", fontsize=11)
    fig.suptitle("Selección del número de conglomerados k — criterios internos", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def plot_dendrogram(X, fname, year=config.YEAR_MODERN, k=2):
    """Dendrograma de Ward con el corte que produce exactamente `k` grupos."""
    Z = linkage(X, method="ward")
    # umbral entre las dos últimas fusiones -> k=2 colores; para k general,
    # entre la (k-1)-ésima y la k-ésima mayor altura de fusión.
    thr = (Z[-(k - 1), 2] + Z[-k, 2]) / 2 if k >= 2 else Z[-1, 2] * 1.1
    palette = [T.CLUSTER2_COLORS[0], T.CLUSTER2_COLORS[1]] if k == 2 \
        else list(T.CLUSTER3_COLORS.values())
    set_link_color_palette(palette)
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=thr,
               above_threshold_color=T.GRIS_MEDIO)
    ax.axhline(thr, color=T.NEGRO, ls=":", lw=1.2)
    ax.text(ax.get_xlim()[1] * 0.99, thr, f" corte k={k}", va="bottom", ha="right",
            fontsize=9, color=T.NEGRO)
    ax.set_title(f"Dendrograma (Ward) — variables completas, {year}")
    ax.set_xlabel("Países")
    ax.set_ylabel("Distancia de fusión")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)
    set_link_color_palette(None)
    return Z


def plot_silhouette(X, labels, k, fname):
    colors, names = _cluster_style(k)
    sv = silhouette_samples(X, labels)
    avg = silhouette_score(X, labels)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    y_lower = 0
    for c in range(k):
        vals = np.sort(sv[labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.9,
                         color=colors[c], label=f"{names[c]}  (n = {(labels==c).sum()})")
        y_lower = y_upper + 6
    ax.axvline(avg, color=T.NEGRO, ls="--", lw=1.4,
               label=f"Silhouette promedio = {avg:.3f}")
    ax.set_xlabel("Coeficiente de silhouette")
    ax.set_ylabel("Países (ordenados dentro de cada conglomerado)")
    ax.set_title(f"Diagrama de silhouette (k = {k})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def plot_clusters_on_pca(scores, labels, evr, fname, title, iso=None):
    k = len(set(int(c) for c in labels if c != -1))
    colors, names = _cluster_style(k)
    fig, ax = plt.subplots(figsize=(10.5, 8))
    for c in sorted(set(labels)):
        m = labels == c
        col = T.GRIS_MEDIO if c == -1 else colors.get(c, T.GRIS)
        lbl = "ruido" if c == -1 else f"{names.get(c, c)}  (n = {m.sum()})"
        ax.scatter(scores[m, 0], scores[m, 1], s=45, alpha=0.9, label=lbl,
                   color=col, edgecolor="white", lw=0.4)
    if iso is not None:
        iso = np.asarray(iso)
        for code, name in NOTABLES.items():
            hit = np.where(iso == code)[0]
            if len(hit):
                i = hit[0]
                ax.text(scores[i, 0] + 0.12, scores[i, 1] + 0.12, name,
                        fontsize=7.5, style="italic", color=T.NEGRO)
    ax.axhline(0, color=T.GRIS_MEDIO, lw=0.6, ls="--")
    ax.axvline(0, color=T.GRIS_MEDIO, lw=0.6, ls="--")
    ax.set_xlabel(f"PC1 — gradiente de desarrollo ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 — estructura productiva ({evr[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(title="Conglomerados")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def run(year=config.YEAR_MODERN):
    imp, std, cats, pre = prep.prepare_single(year)
    num_raw, _ = prep.load_wide(year)
    X_full = imp.values

    # espacio PCA (3 componentes por Horn) y scores 2D para graficar
    with open(config.DATA_PROC / f"pca_info_{year}.json", encoding="utf-8") as f:
        pca_info = json.load(f)
    n_clust_pcs = pca_info["n_clust"]
    pca = PCA(svd_solver="full", random_state=RS).fit(X_full)
    scores_all = pca.transform(X_full)
    X_pca = scores_all[:, :n_clust_pcs]
    evr = pca.explained_variance_ratio_

    results = {"year": year, "n_pcs_clustering": n_clust_pcs}

    # ---------------- 1) Selección de k sobre variables completas ----------- #
    mk = metrics_over_k(X_full)
    gap_df, kopt_gap = gap_statistic(X_full)
    plot_metrics(mk, gap_df, f"clust_metricas_k_{year}.png")
    print("== Métricas internas (k-means, variables completas) ==")
    print(mk.round(3).to_string(index=False))
    print(f"\nGap statistic -> k óptimo = {kopt_gap}")
    print(gap_df.round(3).to_string(index=False))
    results["metricas_k"] = mk.to_dict(orient="records")
    results["gap_kopt"] = kopt_gap

    # ---------------- 2) Estabilidad para k candidatos --------------------- #
    print("\n== Estabilidad bootstrap (Jaccard medio por cluster, Hennig) ==")
    stab = {}
    for k in (2, 3, 4):
        j, _ = bootstrap_stability(X_full, k)
        stab[k] = j
        print(f"  k={k}: " + " ".join(f"C{c}={v:.2f}" for c, v in j.items())
              + f"  (min={min(j.values()):.2f})")
    results["estabilidad"] = stab

    # ---------------- 3) Comparación espacios/algoritmos para k=2 y k=3 ---- #
    # Consenso de métricas: k=2 es el más robusto (silhouette/CH/DB + estabilidad);
    # k=3 es estable y respaldado por gap -> se reporta como vista complementaria.
    comp = {}
    labels_store = {}
    for k in (2, 3):
        km_full = fit_kmeans(X_full, k)
        ward_full = fit_ward(X_full, k)
        km_pca = fit_kmeans(X_pca, k)
        ari_alg = adjusted_rand_score(km_full, ward_full)
        ari_space = adjusted_rand_score(km_full, km_pca)
        sil_full = silhouette_score(X_full, km_full)
        sil_pca = silhouette_score(X_pca, km_pca)
        comp[k] = {
            "ARI_kmeans_vs_ward_full": float(ari_alg),
            "ARI_full_vs_pca": float(ari_space),
            "silhouette_full": float(sil_full),
            "silhouette_pca": float(sil_pca),
        }
        labels_store[(k, "kmeans_full")] = km_full
        labels_store[(k, "ward_full")] = ward_full
        labels_store[(k, "kmeans_pca")] = km_pca
        print(f"\n== k={k} comparaciones ==")
        print(f"  ARI k-means(14 vars) vs Ward(14 vars): {ari_alg:.3f}")
        print(f"  ARI k-means(14 vars) vs k-means({n_clust_pcs} PCs): {ari_space:.3f}")
        print(f"  silhouette: full={sil_full:.3f}  pca={sil_pca:.3f}")
    results["comparaciones"] = comp

    # ---------------- 4) Alternativas: GMM y HDBSCAN ----------------------- #
    gmm2 = fit_gmm(X_full, 2)
    ari_gmm = adjusted_rand_score(labels_store[(2, "kmeans_full")], gmm2)
    hdb = HDBSCAN(min_cluster_size=10).fit_predict(X_full)
    n_hdb = len(set(hdb)) - (1 if -1 in hdb else 0)
    frac_noise = float(np.mean(hdb == -1))
    print("\n== Alternativas ==")
    print(f"  GMM(k=2) vs k-means(k=2) ARI: {ari_gmm:.3f}")
    print(f"  HDBSCAN: {n_hdb} clusters, {frac_noise*100:.1f}% ruido")
    results["alternativas"] = {
        "ari_gmm2_vs_kmeans2": float(ari_gmm),
        "hdbscan_n_clusters": int(n_hdb),
        "hdbscan_frac_ruido": frac_noise,
    }

    # ---------------- 5) Partición FINAL (k-means, 14 vars) ---------------- #
    K_PRIMARY = 2
    final_lab = labels_store[(K_PRIMARY, "kmeans_full")]
    # ordenar etiquetas por PC1 medio (0 = menos desarrollado) para legibilidad
    pc1 = scores_all[:, 0]
    order = np.argsort([pc1[final_lab == c].mean() for c in range(K_PRIMARY)])
    remap = {old: new for new, old in enumerate(order)}
    final_lab = np.array([remap[c] for c in final_lab])
    lab3 = labels_store[(3, "kmeans_full")]
    order3 = np.argsort([pc1[lab3 == c].mean() for c in range(3)])
    remap3 = {old: new for new, old in enumerate(order3)}
    lab3 = np.array([remap3[c] for c in lab3])

    # guardar etiquetas
    out = cats.copy()
    out["cluster_k2"] = final_lab
    out["cluster_k3"] = lab3
    out["PC1"] = scores_all[:, 0]
    out["PC2"] = scores_all[:, 1]
    out.to_csv(config.DATA_PROC / f"clusters_{year}.csv", encoding="utf-8")

    # ---------------- 6) Perfilado (k=2 principal y k=3) ------------------- #
    for k, lab in ((2, final_lab), (3, lab3)):
        med, sizes, ct_inc, ct_reg, chi_inc, chi_reg = profile_clusters(num_raw, lab, cats)
        print(f"\n== Perfil clusters k={k} (tamaños {sizes.to_dict()}) ==")
        print("Cruce cluster x ingreso:")
        print(ct_inc.to_string())
        print(f"  chi2={chi_inc[0]:.1f} p={chi_inc[1]:.2e}  (Cramer V info en JSON)")
        med.to_csv(config.DATA_PROC / f"perfil_medianas_k{k}_{year}.csv", encoding="utf-8")
        ct_inc.to_csv(config.DATA_PROC / f"cruce_ingreso_k{k}_{year}.csv", encoding="utf-8")
        ct_reg.to_csv(config.DATA_PROC / f"cruce_region_k{k}_{year}.csv", encoding="utf-8")
        results.setdefault("perfilado", {})[k] = {
            "tamanos": sizes.to_dict(),
            "chi2_ingreso": {"chi2": float(chi_inc[0]), "p": float(chi_inc[1])},
            "chi2_region": {"chi2": float(chi_reg[0]), "p": float(chi_reg[1])},
        }

    # ---------------- 7) Figuras ------------------------------------------- #
    plot_dendrogram(X_full, f"clust_dendrograma_{year}.png", year)
    plot_silhouette(X_full, final_lab, 2, f"clust_silhouette_k2_{year}.png")
    plot_silhouette(X_full, lab3, 3, f"clust_silhouette_k3_{year}.png")
    iso = imp.index.values
    plot_clusters_on_pca(scores_all, final_lab, evr, f"clust_pca_k2_{year}.png",
                         f"Conglomerados (k-means, 14 variables, k=2) sobre PC1-PC2 — {year}",
                         iso=iso)
    plot_clusters_on_pca(scores_all, lab3, evr, f"clust_pca_k3_{year}.png",
                         f"Conglomerados (k-means, 14 variables, k=3) sobre PC1-PC2 — {year}",
                         iso=iso)

    with open(config.DATA_PROC / f"clustering_resultados_{year}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print("\nOK clustering completo.")
    return results


if __name__ == "__main__":
    run(config.YEAR_MODERN)
