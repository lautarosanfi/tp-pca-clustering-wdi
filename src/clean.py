"""
Limpieza: del panel largo crudo a matrices ancho (país x variable) por año.

Produce:
  - wide_2023.csv, wide_2005.csv  -> matrices crudas (con NaN) ya filtradas
  - countries_2023.csv, countries_2005.csv -> metadata (región, ingreso) alineada
  - reporte_faltantes.csv -> faltantes por variable y año
  - panel_comun.csv -> países presentes en AMBOS años (para trayectorias)

La imputación NO ocurre acá: requiere datos transformados+estandarizados y se
hace por año en prep.py. Acá solo filtramos y reportamos faltantes.
"""
import json

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


def _real_countries():
    meta = pd.read_csv(config.DATA_RAW / "country_metadata.csv")
    real = meta[(meta["region"].notna()) & (meta["region"] != "Aggregates")].copy()
    return real


def build_wide(year: int, real_iso: set) -> pd.DataFrame:
    raw = pd.read_csv(config.DATA_RAW / "wdi_long.csv")
    sub = raw[
        (raw["date"] == year)
        & (raw["indicator"].isin(config.INDICATORS_FINAL))
        & (raw["countryiso3code"].isin(real_iso))
    ]
    wide = sub.pivot_table(index="countryiso3code", columns="indicator", values="value")
    # ordenar columnas según config
    wide = wide.reindex(columns=config.INDICATORS_FINAL)
    return wide


def main():
    real = _real_countries()
    real_iso = set(real["countryiso3code"])
    meta_idx = real.set_index("countryiso3code")

    report = {}
    wides = {}
    for year in (config.YEAR_EARLY, config.YEAR_MODERN):
        wide = build_wide(year, real_iso)
        n0 = len(wide)

        # --- filtro por PAÍS: descartar si > MAX_MISSING_COUNTRY de variables NaN
        frac_missing_country = wide.isna().mean(axis=1)
        keep_country = frac_missing_country <= config.MAX_MISSING_COUNTRY
        dropped_countries = wide.index[~keep_country].tolist()
        wide = wide[keep_country]

        # --- reporte de faltantes por variable (tras filtro de país)
        miss_var = wide.isna().mean().round(3)

        wides[year] = wide
        report[year] = {
            "n_paises_inicial": int(n0),
            "n_paises_final": int(len(wide)),
            "paises_descartados": dropped_countries,
            "faltantes_por_variable": miss_var.to_dict(),
        }

        # adjuntar metadata categórica alineada
        cats = meta_idx.loc[wide.index, ["country", "region", "income"]]
        out = wide.join(cats)
        out.to_csv(config.DATA_PROC / f"wide_{year}.csv", encoding="utf-8")
        cats.to_csv(config.DATA_PROC / f"countries_{year}.csv", encoding="utf-8")
        print(f"[{year}] países: {n0} -> {len(wide)} (descartados {len(dropped_countries)}: {dropped_countries})")
        print(f"[{year}] faltantes por variable (máx {miss_var.max():.3f}):")
        for k, v in miss_var.sort_values(ascending=False).items():
            print(f"    {v:5.3f}  {k}  ({config.INDICATORS[k]})")

    # --- panel común (intersección de países en ambos años) para trayectorias
    common = sorted(set(wides[config.YEAR_EARLY].index) & set(wides[config.YEAR_MODERN].index))
    only_early = sorted(set(wides[config.YEAR_EARLY].index) - set(wides[config.YEAR_MODERN].index))
    only_modern = sorted(set(wides[config.YEAR_MODERN].index) - set(wides[config.YEAR_EARLY].index))
    print(f"\nPaíses en AMBOS años (panel común): {len(common)}")
    print(f"Solo en {config.YEAR_EARLY}: {only_early}")
    print(f"Solo en {config.YEAR_MODERN}: {only_modern}")

    report["panel_comun"] = {
        "n_comun": len(common),
        "solo_early": only_early,
        "solo_modern": only_modern,
        "iso_comun": common,
    }
    pd.Series(common, name="countryiso3code").to_csv(
        config.DATA_PROC / "panel_comun.csv", index=False, encoding="utf-8"
    )

    # reporte de faltantes en formato tabla
    rep_rows = []
    for year in (config.YEAR_EARLY, config.YEAR_MODERN):
        for var, frac in report[year]["faltantes_por_variable"].items():
            rep_rows.append({"anio": year, "variable": var, "frac_faltante": frac})
    pd.DataFrame(rep_rows).to_csv(config.DATA_PROC / "reporte_faltantes.csv", index=False, encoding="utf-8")

    with open(config.DATA_PROC / "reporte_limpieza.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=int)
    print("\nOK limpieza completa.")


if __name__ == "__main__":
    main()
