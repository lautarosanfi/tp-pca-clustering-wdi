"""
Análisis de sensibilidad: ¿qué tan estables son las conclusiones ante decisiones
metodológicas alternativas? Se compara cada variante contra la línea base
(StandardScaler + imputación KNN + 14 variables completas, k-means k=2) usando el
ARI de la partición y la estabilidad de las cargas de PC1.

Variantes evaluadas:
  - Imputación: KNN (base) vs MICE (IterativeImputer) vs casos completos.
  - Escalado: StandardScaler (base) vs RobustScaler.
  - Espacio de clustering: 14 variables (base) vs 2/3/5 componentes PCA.
  - Outliers: con (base) vs sin países extremos.
"""
import json

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

try:
    from . import config, prep
except ImportError:
    import config, prep

RS = config.RANDOM_STATE
YEAR = config.YEAR_MODERN


def _km2(X):
    return KMeans(n_clusters=2, n_init=20, random_state=RS).fit_predict(X)


def _pc1_loading(X, cols):
    pca = PCA(svd_solver="full", random_state=RS).fit(X)
    l = pca.components_.T[:, 0] * np.sqrt(pca.explained_variance_[0])
    s = pd.Series(l, index=cols)
    if s["NY.GDP.PCAP.KD"] < 0:
        s = -s
    return s


def run():
    num, cats = prep.load_wide(YEAR)
    cols = list(num.columns)

    # ---- línea base ----
    pre = prep.Preprocessor(scaler="standard").fit(num)
    std = pre.transform(num)
    base_imp = prep.impute_knn(std)
    base_lab = pd.Series(_km2(base_imp.values), index=base_imp.index)
    base_pc1 = _pc1_loading(base_imp.values, cols)

    results = []

    def compare(name, labels, pc1, idx=None):
        idx = base_lab.index if idx is None else idx
        ari = adjusted_rand_score(base_lab.loc[idx], pd.Series(labels, index=idx))
        corr = float(np.corrcoef(base_pc1.values, pc1.values)[0, 1])
        results.append({"variante": name, "ARI_vs_base": round(ari, 3),
                        "corr_loadings_PC1": round(corr, 3), "n": len(idx)})

    # ---- MICE ----
    mice = IterativeImputer(random_state=RS, max_iter=20, sample_posterior=False)
    imp_mice = pd.DataFrame(mice.fit_transform(std.values), index=std.index, columns=cols)
    compare("Imputación MICE", _km2(imp_mice.values), _pc1_loading(imp_mice.values, cols))

    # ---- casos completos ----
    cc = std.dropna()
    compare("Casos completos (sin imputar)", _km2(cc.values),
            _pc1_loading(cc.values, cols), idx=cc.index)

    # ---- RobustScaler (con TODOS los países) ----
    # Nota: RobustScaler escala por IQR pero NO acota las colas. Si una variable
    # pesada sin transformar (crecimiento del PBI) tiene un outlier extremo, al
    # dividir por un IQR chico ese valor explota y PCA (que maximiza varianza) lo
    # toma como PC1. Se reporta el caso completo y el caso quitando ese outlier.
    pre_r = prep.Preprocessor(scaler="robust").fit(num)
    imp_r = prep.impute_knn(pre_r.transform(num))
    compare("RobustScaler (todos)", _km2(imp_r.values), _pc1_loading(imp_r.values, cols))

    # outlier extremo de crecimiento (p. ej. Macao 2023, rebote turístico ~75%)
    grow_out = num["NY.GDP.MKTP.KD.ZG"].idxmax()
    num_t = num.drop(index=[grow_out])
    pre_rt = prep.Preprocessor(scaler="robust").fit(num_t)
    imp_rt = prep.impute_knn(pre_rt.transform(num_t))
    compare(f"RobustScaler (sin {grow_out})", _km2(imp_rt.values),
            _pc1_loading(imp_rt.values, cols), idx=imp_rt.index)

    # ---- clustering sobre componentes PCA (2, 3, 5) ----
    pca = PCA(svd_solver="full", random_state=RS).fit(base_imp.values)
    sc = pca.transform(base_imp.values)
    for npc in (2, 3, 5):
        compare(f"Clustering sobre {npc} PCs", _km2(sc[:, :npc]), base_pc1)

    # ---- sin outliers (países con |z| extremo) ----
    maxabs = base_imp.abs().max(axis=1)
    thr = maxabs.quantile(0.95)
    outliers = maxabs[maxabs > thr].index.tolist()
    keep = base_imp.drop(index=outliers)
    pre_o = prep.Preprocessor(scaler="standard").fit(num.drop(index=outliers))
    std_o = pre_o.transform(num.drop(index=outliers))
    imp_o = prep.impute_knn(std_o)
    compare(f"Sin outliers ({len(outliers)} países)", _km2(imp_o.values),
            _pc1_loading(imp_o.values, cols), idx=imp_o.index)

    df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    print("== Sensibilidad (vs línea base: Standard + KNN + 14 vars + k=2) ==")
    print(df.to_string(index=False))
    print(f"\nOutliers detectados (|z|>{thr:.2f} en alguna variable): {outliers}")
    df.to_csv(config.DATA_PROC / "sensibilidad.csv", index=False, encoding="utf-8")
    with open(config.DATA_PROC / "sensibilidad.json", "w", encoding="utf-8") as f:
        json.dump({"tabla": results, "outliers": outliers}, f, ensure_ascii=False, indent=2)
    print("\nOK sensibilidad completa.")
    return df


if __name__ == "__main__":
    run()
