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
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

try:
    from . import config, prep, theme
except ImportError:
    import config, prep, theme

T = theme
T.apply_theme()
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
    _plot_transicion_sankey(trans, "compare_transicion_sankey.png")
    _plot_transicion_waffle(traj, "compare_transicion_waffle.png")

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
    """Top 5 avances y top 5 retrocesos en PC1, sobre la nube de todos los países."""
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    mejores = traj.nlargest(5, "dPC1")
    peores = traj.nsmallest(5, "dPC1")
    fig, ax = plt.subplots(figsize=(12, 9))
    # nube de contexto (todos los países, ambos años, tenue)
    ax.scatter(traj["PC1_e"], traj["PC2_e"], s=10, color=T.GRIS_CLARO, zorder=1)
    ax.scatter(traj["PC1_m"], traj["PC2_m"], s=10, color=T.GRIS_MEDIO, alpha=0.5, zorder=1)

    def _arrow(r, color):
        ax.scatter(r["PC1_e"], r["PC2_e"], s=45, color=T.GRIS_MEDIO,
                   edgecolor="white", lw=0.8, zorder=3)
        ax.annotate("", xy=(r["PC1_m"], r["PC2_m"]), xytext=(r["PC1_e"], r["PC2_e"]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                    mutation_scale=14), zorder=4)
        ax.scatter(r["PC1_m"], r["PC2_m"], s=70, color=color,
                   edgecolor="white", lw=1.0, zorder=5)
        ax.text(r["PC1_m"] + 0.15, r["PC2_m"] + 0.15, r["country"], fontsize=8.5,
                color=color, fontweight="bold", zorder=6)

    for _, r in mejores.iterrows():
        _arrow(r, T.DIV_POS)
    for _, r in peores.iterrows():
        _arrow(r, T.DIV_NEG)

    ax.axhline(0, color=T.GRIS_CLARO, lw=1.0, ls="--")
    ax.axvline(0, color=T.GRIS_CLARO, lw=1.0, ls="--")
    ax.set_xlabel(f"PC1 — gradiente de desarrollo ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 — estructura productiva ({evr[1]*100:.1f}%)")
    ax.set_title(f"Mayores avances y retrocesos en el gradiente de desarrollo — {YE} a {YM}")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=T.GRIS_MEDIO, ms=8,
               label=f"Posición en {YE}"),
        Line2D([0], [0], marker=">", color=T.DIV_POS, lw=2, ms=8, label="Top 5 — mayor avance"),
        Line2D([0], [0], marker=">", color=T.DIV_NEG, lw=2, ms=8, label="Top 5 — mayor retroceso"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def _plot_transition(trans, fname):
    """Heatmap 2x2 con colores semánticos (permaneció / graduó / regresó)."""
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    n = trans.shape[0]
    for i in range(n):
        for j in range(n):
            val = int(trans.iloc[i, j])
            if i == j:
                color = T.VERDE_AGUA if i == 1 else T.CORAL
            elif j > i:
                color = T.LIMA          # graduó (subió de grupo)
            else:
                color = T.DIV_NEG       # regresó
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=color,
                                       edgecolor="white", lw=2))
            ax.text(j + 0.5, n - 1 - i + 0.5, str(val), ha="center", va="center",
                    fontsize=15, fontweight="bold", color="white")
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal")
    ax.set_xticks([x + 0.5 for x in range(n)])
    ax.set_yticks([y + 0.5 for y in range(n)])
    ax.set_xticklabels(["En desarrollo", "Desarrollado"], fontsize=9)
    ax.set_yticklabels(["Desarrollado", "En desarrollo"], fontsize=9)
    ax.set_xlabel(f"Grupo en {YM}"); ax.set_ylabel(f"Grupo en {YE}")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Transiciones de conglomerado {YE} → {YM}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def _plot_decomposicion(dec, fname):
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    d = dec.sort_values("pct_del_total")
    colors = [T.NEG_FILL if x < 0 else T.POS_FILL for x in d["pct_del_total"]]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(d["variable"], d["pct_del_total"], color=colors, edgecolor="white")
    ax.axvline(0, color=T.NEGRO, lw=0.8)
    ax.set_xlabel("Contribución al avance medio en PC1 (%)")
    ax.set_title(f"¿Qué explica el avance de los países entre {YE} y {YM}?\n"
                 "Internet y esperanza de vida dominan; el PBI per cápita (real) aporta poco")
    for y, p in enumerate(d["pct_del_total"]):
        ax.text(p + (0.6 if p >= 0 else -0.6), y, f"{p:.1f}%", va="center",
                ha="left" if p >= 0 else "right", fontsize=8.5, fontweight="bold",
                color=T.DIV_POS if p >= 0 else T.DIV_NEG)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def _plot_loadings_compare(l05, l21, cols, fname):
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    fig, ax = plt.subplots(figsize=(9, 7.5))
    lim = [min(l05.min(), l21.min()) - 0.12, max(l05.max(), l21.max()) + 0.12]
    ax.plot(lim, lim, "--", color=T.GRIS_MEDIO)
    ax.scatter(l05, l21, s=45, color=T.VERDE_AGUA, edgecolor="white", lw=0.6, zorder=3)
    for v in cols:
        ax.text(l05[v] + 0.01, l21[v] + 0.01, LABELS[v][:16], fontsize=7, color=T.NEGRO)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Carga en PC1 ({YE})")
    ax.set_ylabel(f"Carga en PC1 ({YM})")
    ax.set_title("Estabilidad de la estructura: cargas de PC1 por año\n"
                 "(los puntos sobre la diagonal indican un eje estable)")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def _plot_transicion_sankey(trans, fname):
    """Diagrama de flujo (Sankey) leyendo los conteos reales de `trans`
    (filas = grupo de origen, columnas = grupo de destino)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Rectangle
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    M = trans.values.astype(float)             # 2x2: [origen, destino]
    total = M.sum()
    scale = 10.0 / total
    row_tot = M.sum(axis=1); col_tot = M.sum(axis=0)
    X_L0, X_L1, X_R0, X_R1 = 0.0, 1.2, 4.8, 6.0
    XM = (X_L1 + X_R0) / 2
    flow_color = {(0, 0): T.CORAL, (1, 1): T.VERDE_AGUA,
                  (0, 1): T.LIMA, (1, 0): T.DIV_NEG}

    def bezier(ax, yl_lo, yl_hi, yr_lo, yr_hi, color):
        verts = [(X_L1, yl_hi), (XM, yl_hi), (XM, yr_hi), (X_R0, yr_hi),
                 (X_R0, yr_lo), (XM, yr_lo), (XM, yl_lo), (X_L1, yl_lo), (X_L1, yl_hi)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, alpha=0.55,
                               edgecolor="none"))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-0.5, 7.8); ax.set_ylim(-1.0, 11.5); ax.axis("off")
    bar_col = {0: T.CORAL, 1: T.VERDE_AGUA}
    # cursores de apilado (de abajo hacia arriba): orden de grupos 1 (desarrollado) abajo, 0 arriba
    grp_order = [1, 0]
    left_cur = {}; right_cur = {}
    y = 0.0
    for g in grp_order:
        left_cur[g] = y
        ax.add_patch(Rectangle((X_L0, y), X_L1 - X_L0, row_tot[g] * scale,
                               facecolor=bar_col[g], edgecolor="white", lw=1.5))
        ax.text((X_L0 + X_L1) / 2, y + row_tot[g] * scale / 2, f"{int(row_tot[g])}",
                ha="center", va="center", color="white", fontweight="bold")
        y += row_tot[g] * scale
    y = 0.0
    for g in grp_order:
        right_cur[g] = y
        ax.add_patch(Rectangle((X_R0, y), X_R1 - X_R0, col_tot[g] * scale,
                               facecolor=bar_col[g], edgecolor="white", lw=1.5))
        ax.text((X_R0 + X_R1) / 2, y + col_tot[g] * scale / 2, f"{int(col_tot[g])}",
                ha="center", va="center", color="white", fontweight="bold")
        y += col_tot[g] * scale
    # flujos i->j
    for i in grp_order:
        for j in grp_order:
            w = M[i, j] * scale
            if w <= 0:
                continue
            yl = left_cur[i]; yr = right_cur[j]
            bezier(ax, yl, yl + w, yr, yr + w, flow_color[(i, j)])
            left_cur[i] += w; right_cur[j] += w
    ax.text((X_L0 + X_L1) / 2, 10.5, str(YE), ha="center", fontsize=12, fontweight="bold")
    ax.text((X_R0 + X_R1) / 2, 10.5, str(YM), ha="center", fontsize=12, fontweight="bold")
    leg = [
        Line2D([0], [0], color=T.LIMA, lw=8, label="Graduó al grupo desarrollado"),
        Line2D([0], [0], color=T.VERDE_AGUA, lw=8, label="Permaneció desarrollado"),
        Line2D([0], [0], color=T.CORAL, lw=8, label="Permaneció en desarrollo"),
    ]
    if M[1, 0] > 0:
        leg.append(Line2D([0], [0], color=T.DIV_NEG, lw=8, label="Retrocedió de grupo"))
    ax.legend(handles=leg, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.set_title(f"Flujo de países entre grupos — {YE} a {YM}", x=0.42)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def _plot_transicion_waffle(traj, fname):
    """Waffle comparativo: 1 cuadro = 1 país, distribución por grupo en cada año."""
    from matplotlib.patches import FancyBboxPatch, Patch
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    total = len(traj)
    dist = {
        YE: {0: int((traj["cluster_e"] == 0).sum()), 1: int((traj["cluster_e"] == 1).sum())},
        YM: {0: int((traj["cluster_m"] == 0).sum()), 1: int((traj["cluster_m"] == 1).sum())},
    }
    COLS_W = 15
    ROWS_W = int(np.ceil(total / COLS_W))
    SIZE, GAP = 0.82, 0.12
    STEP = SIZE + GAP
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, year in zip(axes, [YE, YM]):
        ax.set_xlim(-0.3, COLS_W * STEP); ax.set_ylim(-0.4, ROWS_W * STEP + 0.6)
        ax.axis("off")
        n0, n1 = dist[year][0], dist[year][1]
        seq = [T.CORAL] * n0 + [T.VERDE_AGUA] * n1 + [T.GRIS_CLARO] * (COLS_W * ROWS_W - total)
        for idx, color in enumerate(seq):
            col = idx % COLS_W; row = idx // COLS_W
            x = col * STEP; yy = (ROWS_W - 1 - row) * STEP
            ax.add_patch(FancyBboxPatch((x, yy), SIZE, SIZE, boxstyle="round,pad=0.06",
                                        facecolor=color,
                                        edgecolor="white" if color != T.GRIS_CLARO else "#e0e0e0",
                                        lw=0.8))
        ax.set_title(str(year), fontsize=15, color=T.DIV_POS, pad=8)
        ax.text(COLS_W * STEP / 2, -0.2, f"En desarrollo: {n0}   |   Desarrollado: {n1}",
                ha="center", va="top", fontsize=9, color=T.NEGRO)
    leg = [Patch(facecolor=T.CORAL, label="En desarrollo"),
           Patch(facecolor=T.VERDE_AGUA, label="Desarrollado")]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Distribución de países por grupo — cada cuadro = 1 país", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


if __name__ == "__main__":
    run()
