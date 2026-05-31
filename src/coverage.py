"""Análisis de cobertura: % de países reales con dato por variable y año.

Sirve para (1) fijar el año temprano con criterio y (2) decidir qué variables
entran en la selección final.
"""
import pandas as pd

try:
    from . import config
except ImportError:
    import config


def load_real_panel():
    raw = pd.read_csv(config.DATA_RAW / "wdi_long.csv")
    meta = pd.read_csv(config.DATA_RAW / "country_metadata.csv")
    real = meta[(meta["region"].notna()) & (meta["region"] != "Aggregates")]
    real_iso = set(real["countryiso3code"])
    panel = raw[raw["countryiso3code"].isin(real_iso)].copy()
    return panel, real


def coverage_table(years=(2000, 2005, 2010, 2015, 2021, 2022, 2023, 2024)):
    panel, real = load_real_panel()
    n_countries = real["countryiso3code"].nunique()
    rows = []
    for code, label in config.INDICATORS.items():
        sub = panel[panel["indicator"] == code]
        row = {"code": code, "label": label}
        for y in years:
            vals = sub[(sub["date"] == y) & (sub["value"].notna())]
            row[str(y)] = round(vals["countryiso3code"].nunique() / n_countries, 3)
        rows.append(row)
    cov = pd.DataFrame(rows)
    return cov, n_countries


if __name__ == "__main__":
    cov, n = coverage_table()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Países reales: {n}\n")
    print("Cobertura (proporción de países con dato) por variable y año:\n")
    print(cov.to_string(index=False))
    cov.to_csv(config.DATA_PROC / "cobertura_por_anio.csv", index=False, encoding="utf-8")
