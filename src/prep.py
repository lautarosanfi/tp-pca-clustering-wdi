"""
Preprocesamiento: transformar -> estandarizar -> imputar (en ese orden).

Reglas metodológicas implementadas:
  - ORDEN: primero transformar (corregir asimetría), después estandarizar.
  - Transformaciones decididas variable por variable a partir de la asimetría
    observada en EDA (ver skewness.csv y decisiones_metodologicas.md).
  - Estandarización: se AJUSTA UNA sola vez (sobre los datos de ajuste) y se
    aplica idéntica. Para el análisis transversal 2023 el ajuste es sobre 2023;
    para la comparación temporal el ajuste es sobre los DOS años apilados, de
    modo que ambos vivan en el mismo espacio estandarizado.
  - Imputación: KNN DENTRO DE CADA AÑO por separado (no cruza años), sobre datos
    ya estandarizados (distancias comparables). KNNImputer es determinista.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from scipy.stats import skew

try:
    from . import config
except ImportError:
    import config

# Plan de transformación por variable (decidido tras EDA):
#   'log'        -> log natural (positiva estricta, multi-orden)
#   'log1p'      -> log(1+x) (positiva con ceros)
#   'yeojohnson' -> Yeo-Johnson (admite negativos; λ ajustado por datos)
#   'none'       -> sin transformar (asimetría baja / acotada saturante)
TRANSFORM_PLAN = {
    "NY.GDP.PCAP.KD":       "log",
    "NY.GDP.MKTP.KD.ZG":    "none",
    "FP.CPI.TOTL.ZG":       "yeojohnson",
    "SL.UEM.TOTL.ZS":       "yeojohnson",
    "NE.GDI.TOTL.ZS":       "none",
    "NE.EXP.GNFS.ZS":       "yeojohnson",
    "NE.IMP.GNFS.ZS":       "yeojohnson",
    "NV.AGR.TOTL.ZS":       "yeojohnson",
    "NV.IND.TOTL.ZS":       "yeojohnson",
    "NV.SRV.TOTL.ZS":       "none",
    "SP.URB.TOTL.IN.ZS":    "none",
    "SP.DYN.LE00.IN":       "none",
    "IT.NET.USER.ZS":       "none",
    "EN.GHG.CO2.PC.CE.AR5": "log1p",
}


def load_wide(year):
    df = pd.read_csv(config.DATA_PROC / f"wide_{year}.csv", index_col="countryiso3code")
    num = df[config.INDICATORS_FINAL].copy()
    cats = df[["country", "region", "income"]].copy()
    return num, cats


class Preprocessor:
    """Transforma y estandariza con parámetros ajustados UNA vez."""

    def __init__(self, plan=None, scaler="standard"):
        self.plan = plan or TRANSFORM_PLAN
        self.scaler = scaler
        self.yj = {}
        self.center = None
        self.scale = None
        self.cols = None

    def _transform_only(self, df, fit=False):
        out = df.copy().astype(float)
        for v, how in self.plan.items():
            col = df[v]
            if how == "log":
                out[v] = np.log(col)
            elif how == "log1p":
                out[v] = np.log1p(col)
            elif how == "yeojohnson":
                mask = col.notna()
                if fit:
                    pt = PowerTransformer(method="yeo-johnson", standardize=False)
                    pt.fit(col[mask].values.reshape(-1, 1))
                    self.yj[v] = pt
                pt = self.yj[v]
                out.loc[mask, v] = pt.transform(col[mask].values.reshape(-1, 1)).ravel()
            # 'none' -> sin cambios
        return out

    def fit(self, df):
        self.cols = list(df.columns)
        tdf = self._transform_only(df, fit=True)
        if self.scaler == "standard":
            self.center = tdf.mean()
            self.scale = tdf.std(ddof=0).replace(0, 1.0)
        elif self.scaler == "robust":
            self.center = tdf.median()
            q1 = tdf.quantile(0.25)
            q3 = tdf.quantile(0.75)
            self.scale = (q3 - q1).replace(0, 1.0)
        else:
            raise ValueError(self.scaler)
        return self

    def transform(self, df):
        tdf = self._transform_only(df, fit=False)
        return (tdf - self.center) / self.scale

    def skew_report(self, df):
        """Asimetría antes y después de transformar (sobre observados)."""
        tdf = self._transform_only(df, fit=False)
        rows = []
        for v in df.columns:
            rows.append({
                "variable": v,
                "transformacion": self.plan[v],
                "skew_antes": round(float(skew(df[v].dropna())), 3),
                "skew_despues": round(float(skew(tdf[v].dropna())), 3),
            })
        return pd.DataFrame(rows)


def impute_knn(df_std, n_neighbors=None):
    """KNN sobre datos estandarizados. Determinista (sin random_state)."""
    n = n_neighbors or config.KNN_NEIGHBORS
    imp = KNNImputer(n_neighbors=n)
    arr = imp.fit_transform(df_std.values)
    return pd.DataFrame(arr, index=df_std.index, columns=df_std.columns)


def prepare_single(year, scaler="standard", impute=True):
    """Pipeline para un año: fit en ese año. Devuelve (imp, std, cats, pre)."""
    num, cats = load_wide(year)
    pre = Preprocessor(scaler=scaler).fit(num)
    std = pre.transform(num)
    imp = impute_knn(std) if impute else std
    return imp, std, cats, pre


def prepare_combined(scaler="standard"):
    """Pipeline para comparación temporal: fit sobre los DOS años apilados.

    Transformaciones y escala se ajustan sobre la distribución combinada; la
    imputación KNN se hace POR AÑO. Devuelve un dict con matrices por año y la
    matriz apilada (con índice MultiIndex iso-año).
    """
    num05, cats05 = load_wide(config.YEAR_EARLY)
    num21, cats21 = load_wide(config.YEAR_MODERN)
    comb_fit = pd.concat([num05, num21], ignore_index=True)
    pre = Preprocessor(scaler=scaler).fit(comb_fit)

    std05 = pre.transform(num05)
    std21 = pre.transform(num21)
    imp05 = impute_knn(std05)
    imp21 = impute_knn(std21)

    imp05_t = imp05.copy(); imp05_t["anio"] = config.YEAR_EARLY
    imp21_t = imp21.copy(); imp21_t["anio"] = config.YEAR_MODERN
    stacked = pd.concat([imp05_t, imp21_t])
    stacked = stacked.set_index([stacked.index, "anio"])
    stacked.index.names = ["countryiso3code", "anio"]

    return {
        "pre": pre,
        "imp": {config.YEAR_EARLY: imp05, config.YEAR_MODERN: imp21},
        "std": {config.YEAR_EARLY: std05, config.YEAR_MODERN: std21},
        "cats": {config.YEAR_EARLY: cats05, config.YEAR_MODERN: cats21},
        "stacked": stacked,  # X común para PCA conjunto (cols = variables)
    }


if __name__ == "__main__":
    # Reporte de asimetría antes/después (sobre el año moderno)
    num, _ = load_wide(config.YEAR_MODERN)
    pre = Preprocessor().fit(num)
    rep = pre.skew_report(num)
    pd.set_option("display.width", 200)
    print(f"== Asimetría antes/después de transformar ({config.YEAR_MODERN}) ==")
    print(rep.to_string(index=False))
    rep.to_csv(config.DATA_PROC / "skewness_antes_despues.csv", index=False, encoding="utf-8")
    # chequeo: estandarización deja media~0, sd~1 en el año moderno
    std = pre.transform(num)
    print("\nMedia tras estandarizar (debe ~0):", np.round(std.mean().abs().max(), 4))
    print("SD tras estandarizar (debe ~1):", np.round(std.std(ddof=0).mean(), 4))
