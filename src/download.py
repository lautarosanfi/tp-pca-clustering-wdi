"""
Descarga de datos WDI (World Development Indicators) desde la API del Banco
Mundial y guardado de un snapshot crudo en disco.

Idea: una vez descargado el snapshot (data/raw), TODO el análisis trabaja desde
ese snapshot. Así el pipeline es reproducible aunque la API cambie.
"""
from __future__ import annotations

import time
import json
import sys

import pandas as pd
import requests

try:
    from . import config
except ImportError:  # ejecución como script suelto
    import config


def _get_json(url: str, params: dict, retries: int = 4, timeout: int = 60):
    """GET con reintentos y backoff exponencial. Devuelve el JSON parseado."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"  [reintento {attempt+1}/{retries}] {e} -> espero {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Fallo al descargar {url} con params {params}: {last_err}")


def download_indicator(code: str) -> pd.DataFrame:
    """Descarga un indicador para todos los países y el rango de años, paginando."""
    url = f"{config.WB_BASE}/country/all/indicator/{code}"
    params = {
        "format": "json",
        "source": config.WB_SOURCE,
        "per_page": 20000,
        "date": config.DOWNLOAD_DATE_RANGE,
    }
    first = _get_json(url, params)
    if not isinstance(first, list) or len(first) < 2 or first[1] is None:
        print(f"  AVISO: sin datos para {code}: {first[0] if first else 'vacio'}")
        return pd.DataFrame(columns=["countryiso3code", "country", "date", "value", "indicator"])
    meta, records = first[0], first[1]
    pages = meta.get("pages", 1)
    all_records = list(records)
    for page in range(2, pages + 1):
        p = dict(params, page=page)
        chunk = _get_json(url, p)
        if isinstance(chunk, list) and len(chunk) > 1 and chunk[1]:
            all_records.extend(chunk[1])
    rows = [
        {
            "countryiso3code": rec.get("countryiso3code"),
            "country": (rec.get("country") or {}).get("value"),
            "date": rec.get("date"),
            "value": rec.get("value"),
            "indicator": code,
        }
        for rec in all_records
    ]
    return pd.DataFrame(rows)


def download_country_metadata() -> pd.DataFrame:
    """Metadata de países: región y nivel de ingreso. Filtra agregaciones."""
    url = f"{config.WB_BASE}/country"
    params = {"format": "json", "per_page": 400}
    data = _get_json(url, params)
    records = data[1]
    rows = []
    for rec in records:
        region = (rec.get("region") or {})
        income = (rec.get("incomeLevel") or {})
        rows.append(
            {
                "countryiso3code": rec.get("id"),
                "iso2": rec.get("iso2Code"),
                "country": rec.get("name"),
                "region": region.get("value"),
                "region_id": region.get("id"),
                "income": income.get("value"),
                "income_id": income.get("id"),
                "capital": rec.get("capitalCity"),
                "lat": rec.get("latitude"),
                "lon": rec.get("longitude"),
            }
        )
    df = pd.DataFrame(rows)
    return df


def main():
    print("== Descarga de metadata de países ==")
    meta = download_country_metadata()
    meta.to_csv(config.DATA_RAW / "country_metadata.csv", index=False, encoding="utf-8")
    # entidades reales = región distinta de 'Aggregates' y no nula
    real = meta[(meta["region"].notna()) & (meta["region"] != "Aggregates")]
    print(f"  Entidades totales: {len(meta)} | países reales (no agregados): {len(real)}")

    print("== Descarga de indicadores WDI ==")
    frames = []
    for code, label in config.INDICATORS.items():
        print(f"- {code}  ({label})")
        df = download_indicator(code)
        frames.append(df)
        time.sleep(0.3)
    raw = pd.concat(frames, ignore_index=True)
    # Tipos
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw["date"] = pd.to_numeric(raw["date"], errors="coerce").astype("Int64")
    raw.to_csv(config.DATA_RAW / "wdi_long.csv", index=False, encoding="utf-8")
    print(f"  Snapshot crudo guardado: {len(raw):,} filas (largo) en data/raw/")

    # pequeño manifest para trazabilidad
    manifest = {
        "download_date_range": config.DOWNLOAD_DATE_RANGE,
        "n_indicators": len(config.INDICATORS),
        "indicators": config.INDICATORS,
        "n_rows_long": int(len(raw)),
        "n_countries_meta_real": int(len(real)),
    }
    with open(config.DATA_RAW / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("OK descarga completa.")


if __name__ == "__main__":
    sys.exit(main())
