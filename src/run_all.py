"""
Orquestador del pipeline completo (de punta a punta).

Uso:
    python src/run_all.py            # usa el snapshot crudo si existe
    python src/run_all.py --download # fuerza re-descarga desde la API WDI

Pasos: descarga -> cobertura -> limpieza -> EDA -> PCA -> clustering ->
comparación temporal -> sensibilidad.
"""
import sys
import time

try:
    from . import (config, download, coverage, clean, eda, pca, clustering,
                   compare_years, sensitivity, interactive)
except ImportError:
    import config, download, coverage, clean, eda, pca, clustering, compare_years, sensitivity, interactive


def main(argv=None):
    argv = argv or sys.argv[1:]
    force_dl = "--download" in argv
    snapshot = config.DATA_RAW / "wdi_long.csv"

    t0 = time.time()
    print("=" * 70)
    if force_dl or not snapshot.exists():
        print(">> [1/9] Descarga de datos WDI")
        download.main()
    else:
        print(">> [1/9] Snapshot crudo encontrado; se omite descarga "
              "(usar --download para forzar).")

    print("\n>> [2/9] Análisis de cobertura")
    cov, n = coverage.coverage_table()
    cov.to_csv(config.DATA_PROC / "cobertura_por_anio.csv", index=False, encoding="utf-8")

    print("\n>> [3/9] Limpieza")
    clean.main()

    print("\n>> [4/9] EDA (univariado + bivariado + categóricas)")
    eda.main()

    print(f"\n>> [5/9] PCA ({config.YEAR_MODERN})")
    pca.run(config.YEAR_MODERN)

    print(f"\n>> [6/9] Clustering ({config.YEAR_MODERN})")
    clustering.run(config.YEAR_MODERN)

    print(f"\n>> [7/9] Comparación temporal {config.YEAR_EARLY} vs {config.YEAR_MODERN}")
    compare_years.run()

    print("\n>> [8/9] Sensibilidad")
    sensitivity.run()

    print("\n>> [9/9] Gráficos interactivos (Plotly) para el informe")
    interactive.run()

    print("=" * 70)
    print(f"PIPELINE COMPLETO en {time.time()-t0:.0f}s. "
          f"Resultados en data/processed/, reports/figures/ e informe_final/figures/.")


if __name__ == "__main__":
    main()
