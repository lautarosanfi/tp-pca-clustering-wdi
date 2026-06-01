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


def _year_tiers(year):
    """Clasifica los países en 2 tiers (desarrollado=1 / en desarrollo=0) DENTRO
    de ese año (relativo a sus contemporáneos). Devuelve (tier_serie, pc1_serie)."""
    imp, _, _, _ = prep.prepare_single(year)
    p = PCA(svd_solver="full", random_state=RS).fit(imp.values)
    s1 = p.transform(imp.values)[:, 0]
    gi = list(imp.columns).index("NY.GDP.PCAP.KD")
    if p.components_[0][gi] < 0:
        s1 = -s1
    lab = KMeans(n_clusters=2, n_init=20, random_state=RS).fit_predict(imp.values)
    if s1[lab == 0].mean() > s1[lab == 1].mean():
        lab = 1 - lab
    return pd.Series(lab, index=imp.index), pd.Series(s1, index=imp.index)


def run():
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    data = prep.prepare_combined()
    stacked = data["stacked"]              # MultiIndex (iso, anio), 14 cols
    cols = list(stacked.columns)

    # -------- PCA COMÚN: define el eje "desarrollo" y la comparación de ESTRUCTURA -- #
    pca = PCA(svd_solver="full", random_state=RS).fit(stacked.values)
    scores = pca.transform(stacked.values)
    evr = pca.explained_variance_ratio_
    sc = pd.DataFrame(scores[:, :3], index=stacked.index, columns=["PC1", "PC2", "PC3"])
    if pca.components_[0][cols.index("NY.GDP.PCAP.KD")] < 0:
        sc["PC1"] = -sc["PC1"]
    pc1_e_all = sc.xs(YE, level="anio")["PC1"]
    pc1_m_all = sc.xs(YM, level="anio")["PC1"]
    print(f"PCA común: var explicada PC1-3 = {np.round(evr[:3]*100,1)}")
    print(f"PC1 medio en el espacio común: {YE}={pc1_e_all.mean():+.2f}  "
          f"{YM}={pc1_m_all.mean():+.2f}  <- CORRIMIENTO SECULAR (no es desarrollo "
          f"relativo): comparar posiciones absolutas NO es válido.")

    # -------- Posición RELATIVA: tiers y percentiles DENTRO de cada año ----------- #
    tier_e, _ = _year_tiers(YE)
    tier_m, _ = _year_tiers(YM)
    print(f"Desarrollados (relativo a cada año): {YE}={int(tier_e.sum())}/{len(tier_e)}  "
          f"{YM}={int(tier_m.sum())}/{len(tier_m)}")
    # validación: el tier relativo de YM ~ modelo principal de YM
    main = pd.read_csv(config.DATA_PROC / f"clusters_{YM}.csv", index_col="countryiso3code")
    cm = main.index.intersection(tier_m.index)
    ari_main = adjusted_rand_score(main.loc[cm, "cluster_k2"], tier_m.loc[cm])
    print(f"ARI tier-relativo-{YM} vs modelo-principal-{YM}: {ari_main:.3f}")
    perc_e = pc1_e_all.rank(pct=True) * 100   # percentil de desarrollo dentro del año
    perc_m = pc1_m_all.rank(pct=True) * 100

    # -------- Panel común: trayectorias RELATIVAS ---------------------------------- #
    common = pd.read_csv(config.DATA_PROC / "panel_comun.csv")["countryiso3code"].tolist()
    meta = pd.read_csv(config.DATA_PROC / f"countries_{YM}.csv", index_col="countryiso3code")
    rows = []
    for iso in common:
        if not all(iso in s.index for s in (perc_e, perc_m, tier_e, tier_m)):
            continue
        rows.append({
            "iso": iso,
            "perc_e": perc_e[iso], "perc_m": perc_m[iso],
            "dperc": perc_m[iso] - perc_e[iso],
            "tier_e": int(tier_e[iso]), "tier_m": int(tier_m[iso]),
            "pc1_e": pc1_e_all[iso], "pc1_m": pc1_m_all[iso],
        })
    traj = pd.DataFrame(rows).set_index("iso")
    traj["country"] = meta["country"].reindex(traj.index)
    traj["region"] = meta["region"].reindex(traj.index)
    traj.to_csv(config.DATA_PROC / "trayectorias.csv", encoding="utf-8")

    print(f"\nMayor ASCENSO en posición relativa (Δ percentil, {YE}->{YM}):")
    print(traj.sort_values("dperc", ascending=False).head(8)[
        ["country", "perc_e", "perc_m", "dperc"]].round(1).to_string())
    print("Mayor DESCENSO en posición relativa:")
    print(traj.sort_values("dperc").head(6)[
        ["country", "perc_e", "perc_m", "dperc"]].round(1).to_string())

    # -------- Transiciones de tier RELATIVO ---------------------------------------- #
    trans = pd.crosstab(traj["tier_e"], traj["tier_m"],
                        rownames=[f"tier {YE}"], colnames=[f"tier {YM}"])
    print("\nMatriz de transición de tiers relativos (panel común):")
    print(trans.to_string())
    subio = traj[(traj["tier_e"] == 0) & (traj["tier_m"] == 1)]
    bajo = traj[(traj["tier_e"] == 1) & (traj["tier_m"] == 0)]
    print(f"\nAscendieron de tier (catch-up): {len(subio)} -> {sorted(subio['country'].tolist())}")
    print(f"Descendieron de tier: {len(bajo)} -> {sorted(bajo['country'].tolist())}")
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
    _plot_trajectories(traj, "compare_trayectorias.png")
    _plot_transition(trans, "compare_transiciones.png")
    _plot_loadings_compare(load05["PC1"], load21["PC1"], cols, "compare_loadings_pc1.png")
    _plot_decomposicion(dec, "compare_descomposicion_dPC1.png")
    _plot_transicion_sankey(trans, "compare_transicion_sankey.png")
    _plot_transicion_waffle(traj, "compare_transicion_waffle.png")

    out = {
        "anio_early": config.YEAR_EARLY,
        "anio_modern": config.YEAR_MODERN,
        "evr_comun": [float(x) for x in evr[:3]],
        "ari_tier_modern_vs_main": float(ari_main),
        "pc1_medio_comun_early": float(pc1_e_all.mean()),
        "pc1_medio_comun_modern": float(pc1_m_all.mean()),
        "desarrollados_relativo_early": int(tier_e.sum()),
        "desarrollados_relativo_modern": int(tier_m.sum()),
        "ascendieron_tier": int(len(subio)),
        "descendieron_tier": int(len(bajo)),
        "congruencia_pc1": congru,
        "corr_pc1": corr_pc1,
        "evr_pc1_early": float(evr05[0]), "evr_pc1_modern": float(evr21[0]),
        "dPC1_secular_total": float(contrib.sum()),
    }
    with open(config.DATA_PROC / "compare_resultados.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nOK comparación temporal completa.")
    return out


RELEVANTES = ["CHN", "IND", "USA", "BRA", "RUS", "IDN", "NGA", "KOR", "VNM",
              "POL", "TUR", "ZAF", "MEX", "DEU"]


def _plot_trajectories(traj, fname):
    """Movilidad RELATIVA: percentil de desarrollo (dentro de cada año) en YE vs YM.
    Robusto a la tendencia secular. Sobre la diagonal = sin cambio relativo;
    arriba = ascenso relativo (catch-up); abajo = descenso relativo."""
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ax.plot([0, 100], [0, 100], ls="--", color=T.GRIS_MEDIO, lw=1.2, zorder=1)
    # color por dirección del cambio relativo
    for _, r in traj.iterrows():
        d = r["dperc"]
        col = T.AZUL if d > 3 else (T.BERMELLON if d < -3 else T.GRIS_CLARO)
        ax.scatter(r["perc_e"], r["perc_m"], s=42, color=col, alpha=0.9,
                   edgecolor="white", lw=0.4, zorder=3)
    # etiquetar países relevantes + mayores movimientos relativos
    destacar = set(RELEVANTES) | set(traj["dperc"].abs().sort_values(ascending=False).head(8).index)
    for iso in destacar:
        if iso in traj.index:
            r = traj.loc[iso]
            ax.text(r["perc_e"] + 1.2, r["perc_m"] + 1.2, r["country"], fontsize=7.5,
                    color=T.NEGRO, zorder=5)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_xlabel(f"Percentil de desarrollo en {YE} (relativo a sus pares)")
    ax.set_ylabel(f"Percentil de desarrollo en {YM} (relativo a sus pares)")
    ax.set_title(f"Movilidad relativa en el gradiente de desarrollo — {YE} vs {YM}")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=T.AZUL, ms=9,
               label="Ascenso relativo (catch-up)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=T.BERMELLON, ms=9,
               label="Descenso relativo"),
        Line2D([0], [0], ls="--", color=T.GRIS_MEDIO, label="Sin cambio relativo"),
    ]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def _plot_transition(trans, fname):
    """Barras apiladas por ORIGEN: de cada grupo de YE, ¿cuántos permanecieron y
    cuántos cambiaron de tier en YM? El color marca el DESTINO (dónde terminó);
    los segmentos fuera de la diagonal se anotan como ascenso/descenso."""
    from matplotlib.patches import Patch
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    M = trans.values.astype(int)              # filas=origen YE, cols=destino YM
    dest_color = {0: T.CORAL, 1: T.AZUL}      # 0=en desarrollo, 1=desarrollado
    rows_y = {0: 1.0, 1: 0.0}                 # fila 0 (en desarrollo) arriba
    xmax = M.sum(1).max()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for r in (0, 1):
        y = rows_y[r]; left = 0
        for d in (0, 1):
            w = int(M[r, d])
            if w == 0:
                continue
            ax.barh(y, w, left=left, height=0.6, color=dest_color[d],
                    edgecolor="white")
            note = "↑ ascendió" if d > r else ("↓ descendió" if d < r else "")
            if w / xmax < 0.07:        # segmento muy fino: anotar AFUERA (arriba)
                ax.annotate(f"{w}  {note}".strip(), xy=(left + w / 2, y + 0.30),
                            xytext=(left + w / 2, y + 0.55), ha="center",
                            va="bottom", fontsize=9.5, color=dest_color[d],
                            fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=dest_color[d],
                                            lw=0.9))
            else:                      # segmento ancho: etiqueta DENTRO en blanco
                label = f"{w}" + (f"\n{note}" if note else "")
                ax.text(left + w / 2, y, label, ha="center", va="center",
                        color="white", fontweight="bold", fontsize=11)
            left += w
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([f"En desarrollo\nen {YE}  (n={M[0].sum()})",
                        f"Desarrollado\nen {YE}  (n={M[1].sum()})"], fontsize=10)
    ax.set_xlim(0, M.sum(1).max() * 1.03)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Cantidad de países")
    ax.set_title(f"¿A dónde fue cada grupo? Transición de tier relativo {YE} → {YM}")
    leg = [Patch(facecolor=T.CORAL, label=f"Terminó en desarrollo ({YM})"),
           Patch(facecolor=T.AZUL, label=f"Terminó desarrollado ({YM})")]
    ax.legend(handles=leg, loc="lower right", fontsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=T.GRIS_CLARO); ax.set_axisbelow(True)
    ax.tick_params(length=0)
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
    ax.set_xlabel("Contribución al corrimiento medio absoluto de PC1 (%)")
    ax.set_title(f"Composición del corrimiento ABSOLUTO de PC1 ({YE}→{YM})\n"
                 "Dominado por la difusión de Internet (tendencia secular global), no por el ingreso")
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
    """Slopegraph: cómo cambió el TAMAÑO de cada tier entre los dos años (en lugar
    del Sankey, más difícil de leer). Cada grupo es una línea: su pendiente muestra
    si creció o se achicó; el subtítulo resume cuántos cambiaron de tier."""
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    M = trans.values.astype(int)               # filas=origen YE, cols=destino YM
    size_e = M.sum(axis=1)                      # tamaño por tier en YE
    size_m = M.sum(axis=0)                      # tamaño por tier en YM
    tiers = [(1, "Desarrollado", T.AZUL), (0, "En desarrollo", T.CORAL)]
    x0, x1 = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for idx, name, col in tiers:
        ye_v, ym_v = int(size_e[idx]), int(size_m[idx])
        ax.plot([x0, x1], [ye_v, ym_v], "-o", color=col, lw=3.5, ms=12,
                markeredgecolor="white", markeredgewidth=1.6, zorder=3)
        ax.text(x0 - 0.05, ye_v, f"{name}\n{ye_v}", ha="right", va="center",
                fontsize=10.5, color=col, fontweight="bold")
        ax.text(x1 + 0.05, ym_v, f"{ym_v}", ha="left", va="center",
                fontsize=12, color=col, fontweight="bold")
    grad, desc = int(M[0, 1]), int(M[1, 0])
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([x0, x1]); ax.set_xticklabels([str(YE), str(YM)], fontsize=13)
    ax.set_ylabel("Cantidad de países")
    ax.set_title(f"Cómo cambió el tamaño de cada grupo — {YE} vs {YM}\n"
                 f"{grad} ascendieron de tier (catch-up)  ·  "
                 f"{desc} descendió", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=T.GRIS_CLARO); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname)
    plt.close(fig)


def _plot_transicion_waffle(traj, fname):
    """Waffle comparativo: 1 cuadro = 1 país, distribución por grupo en cada año."""
    from matplotlib.patches import FancyBboxPatch, Patch
    YE, YM = config.YEAR_EARLY, config.YEAR_MODERN
    total = len(traj)
    dist = {
        YE: {0: int((traj["tier_e"] == 0).sum()), 1: int((traj["tier_e"] == 1).sum())},
        YM: {0: int((traj["tier_m"] == 0).sum()), 1: int((traj["tier_m"] == 1).sum())},
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
