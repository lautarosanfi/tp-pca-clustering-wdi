"""
Comparación rigurosa 2005 vs 2021.

Para que los dos años sean COMPARABLES, deben vivir en el MISMO espacio:
  - Transformaciones decididas sobre la distribución COMBINADA (ambos años).
  - UN scaler y UN PCA ajustados sobre los datos combinados.
  - Imputación KNN POR AÑO (no cruza años).
  - Scores de todos los puntos país-año en ese espacio común.

ERROR que se evita: ajustar un PCA por año y comparar coordenadas (ejes, signos
y escalas distintos -> no comparables).

Análisis:
  - Trayectorias país (2005 -> 2021) en el espacio PCA común.
  - Transiciones de cluster (matriz 2005 x 2021) sobre el panel común.
  - Estabilidad de la ESTRUCTURA: ¿el gradiente de desarrollo (PC1) se mantiene?
    (se comparan loadings de PCA ajustados por separado en cada año, alineando
    signos; comparar loadings ES válido, comparar scores NO).
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

try:
    from . import config, prep
except ImportError:
    import config, prep

sns.set_theme(style="whitegrid", context="notebook")
LABELS = config.INDICATORS
RS = config.RANDOM_STATE


def _align_sign(loadings, anchor="NY.GDP.PCAP.KD"):
    """Fija el signo de PC1 para que el PBI per cápita cargue positivo."""
    if loadings.loc[anchor, "PC1"] < 0:
        loadings["PC1"] = -loadings["PC1"]
    return loadings


def per_year_loadings(imp, columns):
    pca = PCA(svd_solver="full", random_state=RS).fit(imp.values)
    load = pca.components_.T * np.sqrt(pca.explained_variance_)
    df = pd.DataFrame(load[:, :3], index=columns, columns=["PC1", "PC2", "PC3"])
    return _align_sign(df), pca.explained_variance_ratio_


def run():
    data = prep.prepare_combined()
    pre = data["pre"]
    stacked = data["stacked"]              # MultiIndex (iso, anio), 14 cols
    cols = list(stacked.columns)

    # -------- PCA común sobre los datos combinados --------
    pca = PCA(svd_solver="full", random_state=RS).fit(stacked.values)
    scores = pca.transform(stacked.values)
    evr = pca.explained_variance_ratio_
    sc = pd.DataFrame(scores[:, :3], index=stacked.index, columns=["PC1", "PC2", "PC3"])
    load_comb = _align_sign(
        pd.DataFrame(pca.components_.T[:, :3] * np.sqrt(pca.explained_variance_[:3]),
                     index=cols, columns=["PC1", "PC2", "PC3"]))
    # alinear scores si se invirtió PC1
    if (pca.components_.T[:, 0][cols.index("NY.GDP.PCAP.KD")]) < 0:
        sc["PC1"] = -sc["PC1"]
    print(f"PCA común: var explicada PC1-3 = {np.round(evr[:3]*100,1)} (acum {evr[:3].sum()*100:.1f}%)")

    # -------- Clustering k=2 en el espacio común (14 vars combinadas) --------
    km = KMeans(n_clusters=2, n_init=20, random_state=RS).fit(stacked.values)
    lab = km.labels_
    # ordenar: cluster 0 = menos desarrollado (menor PC1 medio)
    pc1 = sc["PC1"].values
    if pc1[lab == 0].mean() > pc1[lab == 1].mean():
        lab = 1 - lab
    sc["cluster"] = lab

    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    # validar contra el modelo principal del año moderno (debe coincidir mucho)
    main = pd.read_csv(config.DATA_PROC / f"clusters_{YM}.csv",
                       index_col="countryiso3code")
    scM = sc.xs(YM, level="anio")
    commonM = main.index.intersection(scM.index)
    ari_main = adjusted_rand_score(main.loc[commonM, "cluster_k2"],
                                   scM.loc[commonM, "cluster"])
    print(f"ARI cluster espacio-común-{YM} vs modelo-principal-{YM}: {ari_main:.3f}")

    # -------- Panel común y trayectorias --------
    common = pd.read_csv(config.DATA_PROC / "panel_comun.csv")["countryiso3code"].tolist()
    rows = []
    for iso in common:
        try:
            e = sc.loc[(iso, YE)]
            m = sc.loc[(iso, YM)]
        except KeyError:
            continue
        rows.append({
            "iso": iso,
            "PC1_e": e["PC1"], "PC1_m": m["PC1"],
            "PC2_e": e["PC2"], "PC2_m": m["PC2"],
            "cluster_e": int(e["cluster"]), "cluster_m": int(m["cluster"]),
            "dPC1": m["PC1"] - e["PC1"], "dPC2": m["PC2"] - e["PC2"],
        })
    traj = pd.DataFrame(rows).set_index("iso")
    meta = pd.read_csv(config.DATA_PROC / f"countries_{YM}.csv",
                       index_col="countryiso3code")
    traj["country"] = meta["country"].reindex(traj.index)
    traj["region"] = meta["region"].reindex(traj.index)
    traj.to_csv(config.DATA_PROC / "trayectorias.csv", encoding="utf-8")

    print(f"\nMovimiento medio en PC1 ({YE}->{YM}): {traj['dPC1'].mean():+.3f} "
          f"(positivo = avance en el gradiente de desarrollo)")
    print("Países que más avanzaron en PC1:")
    print(traj.sort_values("dPC1", ascending=False).head(8)[["country", "dPC1"]].round(2).to_string())
    print("Países que más retrocedieron en PC1:")
    print(traj.sort_values("dPC1").head(5)[["country", "dPC1"]].round(2).to_string())

    # -------- Transiciones de cluster --------
    trans = pd.crosstab(traj["cluster_e"], traj["cluster_m"],
                        rownames=[f"cluster {YE}"], colnames=[f"cluster {YM}"])
    print("\nMatriz de transición de clusters (panel común):")
    print(trans.to_string())
    movers = traj[traj["cluster_e"] != traj["cluster_m"]]
    print(f"\nPaíses que cambiaron de cluster: {len(movers)}")
    print(movers[["country", "cluster_e", "cluster_m", "dPC1"]].round(2).to_string())
    trans.to_csv(config.DATA_PROC / "transiciones_cluster.csv", encoding="utf-8")

    # -------- Estabilidad de la estructura (loadings por año) --------
    load05, evr05 = per_year_loadings(data["imp"][config.YEAR_EARLY], cols)
    load21, evr21 = per_year_loadings(data["imp"][config.YEAR_MODERN], cols)
    congru = float(np.dot(load05["PC1"], load21["PC1"]) /
                   (np.linalg.norm(load05["PC1"]) * np.linalg.norm(load21["PC1"])))
    corr_pc1 = float(np.corrcoef(load05["PC1"], load21["PC1"])[0, 1])
    print(f"\nEstructura PC1 — var explicada: {YE}={evr05[0]*100:.1f}%  {YM}={evr21[0]*100:.1f}%")
    print(f"Congruencia (Tucker) loadings PC1 {YE} vs {YM}: {congru:.3f}")
    print(f"Correlación loadings PC1 {YE} vs {YM}: {corr_pc1:.3f}")
    cmp = pd.DataFrame({f"PC1_{YE}": load05["PC1"], f"PC1_{YM}": load21["PC1"]})
    cmp["delta"] = cmp[f"PC1_{YM}"] - cmp[f"PC1_{YE}"]
    cmp.index = [LABELS[v] for v in cmp.index]
    print("\nCargas en PC1 por año (variable -> peso en el gradiente de desarrollo):")
    print(cmp.round(2).sort_values(f"PC1_{YM}", ascending=False).to_string())
    cmp.to_csv(config.DATA_PROC / "loadings_pc1_por_anio.csv", encoding="utf-8")

    # -------- Descomposición del movimiento de PC1 por variable --------
    # ¿Qué variables explican el avance promedio en PC1? El score de PC1 es una
    # combinación lineal de las variables ESTANDARIZADAS: PC1 = sum_v w_v * z_v,
    # con w_v = autovector (components_[0]) del PCA común. Entonces el avance medio
    # se descompone como  mean(ΔPC1) = sum_v w_v * mean(Δz_v).
    gdp_idx = cols.index("NY.GDP.PCAP.KD")
    v1 = pca.components_[0].copy()
    if v1[gdp_idx] < 0:        # mismo criterio de signo que los scores
        v1 = -v1
    impE = data["imp"][config.YEAR_EARLY].loc[common]
    impM = data["imp"][config.YEAR_MODERN].loc[common]
    dz = (impM.values - impE.values).mean(axis=0)   # mean Δz por variable
    contrib = v1 * dz
    dec = pd.DataFrame({
        "variable": [LABELS[c] for c in cols],
        "peso_PC1(w)": np.round(v1, 3),
        "mean_dz": np.round(dz, 3),
        "contrib_a_dPC1": np.round(contrib, 4),
    }).sort_values("contrib_a_dPC1", ascending=False)
    dec["pct_del_total"] = np.round(100 * dec["contrib_a_dPC1"] / contrib.sum(), 1)
    print(f"\nDescomposición del avance medio en PC1 (total={contrib.sum():+.3f}):")
    print(dec.to_string(index=False))
    dec.to_csv(config.DATA_PROC / "descomposicion_dPC1.csv", index=False, encoding="utf-8")

    # -------- Figuras --------
    _plot_trajectories(traj, evr, "compare_trayectorias.png")
    _plot_transition(trans, "compare_transiciones.png")
    _plot_loadings_compare(load05["PC1"], load21["PC1"], cols, "compare_loadings_pc1.png")
    _plot_decomposicion(dec, "compare_descomposicion_dPC1.png")

    out = {
        "anio_early": config.YEAR_EARLY,
        "anio_modern": config.YEAR_MODERN,
        "evr_comun": [float(x) for x in evr[:3]],
        "ari_main_modern": float(ari_main),
        "mov_medio_pc1": float(traj["dPC1"].mean()),
        "congruencia_pc1": congru,
        "corr_pc1": corr_pc1,
        "evr_pc1_early": float(evr05[0]), "evr_pc1_modern": float(evr21[0]),
        "n_cambian_cluster": int(len(movers)),
        "dPC1_total_descomp": float(contrib.sum()),
    }
    with open(config.DATA_PROC / "compare_resultados.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nOK comparación temporal completa.")
    return out


def _plot_trajectories(traj, evr, fname):
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    # destacar los mayores movimientos + algunos países conocidos
    big = traj.reindex(traj["dPC1"].abs().sort_values(ascending=False).index).head(12)
    known = [c for c in ["CHN", "IND", "VNM", "KOR", "POL", "RWA", "ETH", "BGD"]
             if c in traj.index]
    sel = sorted(set(big.index) | set(known))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(traj["PC1_e"], traj["PC2_e"], s=12, color="#BBBBBB",
               alpha=0.5, label=str(YE))
    ax.scatter(traj["PC1_m"], traj["PC2_m"], s=12, color="#4C72B0",
               alpha=0.5, label=str(YM))
    for iso in sel:
        r = traj.loc[iso]
        ax.annotate("", xy=(r["PC1_m"], r["PC2_m"]),
                    xytext=(r["PC1_e"], r["PC2_e"]),
                    arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.3, alpha=0.8))
        ax.text(r["PC1_m"], r["PC2_m"], iso, fontsize=7)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%) — gradiente de desarrollo")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%) — estructura productiva")
    ax.set_title(f"Trayectorias {YE}→{YM} en el espacio PCA común (países destacados)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=config.FIG_DPI)
    plt.close(fig)


def _plot_transition(trans, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(trans, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Transiciones de cluster {config.YEAR_EARLY} → {config.YEAR_MODERN}\n"
                 "(0=menos desarrollado, 1=más desarrollado)")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=config.FIG_DPI)
    plt.close(fig)


def _plot_decomposicion(dec, fname):
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    d = dec.sort_values("contrib_a_dPC1")
    colors = ["#C44E52" if x < 0 else "#4C72B0" for x in d["contrib_a_dPC1"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(d["variable"], d["contrib_a_dPC1"], color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Contribución al avance medio en PC1 (unidades de score)")
    ax.set_title(f"¿Qué variables explican el avance medio en PC1 ({YE}→{YM})?\n"
                 "Internet y esperanza de vida dominan; el PBI per cápita (real) aporta poco")
    for y, (v, p) in enumerate(zip(d["contrib_a_dPC1"], d["pct_del_total"])):
        ax.text(v + (0.005 if v >= 0 else -0.005), y, f"{p:.0f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=config.FIG_DPI)
    plt.close(fig)


def _plot_loadings_compare(l05, l21, cols, fname):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(l05, l21, color="#4C72B0")
    lim = [min(l05.min(), l21.min()) - 0.1, max(l05.max(), l21.max()) + 0.1]
    ax.plot(lim, lim, "--", color="gray")
    for v in cols:
        ax.text(l05[v], l21[v], LABELS[v][:14], fontsize=7)
    ax.set_xlabel(f"Carga en PC1 ({config.YEAR_EARLY})")
    ax.set_ylabel(f"Carga en PC1 ({config.YEAR_MODERN})")
    ax.set_title("Estabilidad de la estructura: cargas de PC1 por año")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=config.FIG_DPI)
    plt.close(fig)


if __name__ == "__main__":
    run()
