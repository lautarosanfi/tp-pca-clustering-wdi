# PCA y Clustering de indicadores de desarrollo (WDI, Banco Mundial)

TP Final — *Análisis Multivariado y Descubrimiento de Patrones* (Lic. en Ciencia de
Datos, Universidad Austral).

Análisis de Componentes Principales (PCA) + Análisis de Clustering sobre indicadores de
desarrollo de ~191 países, en corte transversal **2023** (año macroeconómicamente
estable, no post-COVID), con comparación temporal contra **2005**.

## Estructura

```
proyecto/
├── data/
│   ├── raw/          # snapshot crudo de la API WDI (no se vuelve a descargar)
│   └── processed/    # datasets limpios, scores PCA, clusters, métricas, trayectorias
├── src/
│   ├── config.py        # parámetros, rutas, indicadores, años, random_state
│   ├── download.py      # descarga API WDI + snapshot
│   ├── coverage.py      # cobertura por variable/año (elección del año temprano)
│   ├── clean.py         # filtrado, faltantes, panel común
│   ├── prep.py          # transformar -> estandarizar -> imputar (KNN)
│   ├── eda.py           # univariado + bivariado + categóricas
│   ├── pca.py           # PCA + análisis paralelo de Horn
│   ├── clustering.py    # k-means/Ward, selección de k, estabilidad, perfilado
│   ├── compare_years.py # comparación 2005 vs 2023 (espacio común, trayectorias)
│   ├── sensitivity.py   # robustez de las conclusiones
│   └── run_all.py       # orquesta todo el pipeline
├── notebooks/presentacion.ipynb   # narrativa que CONSUME los resultados
├── reports/
│   ├── figures/                    # figuras generadas
│   ├── informe_borrador.md         # BORRADOR del informe (estructura de la consigna)
│   └── decisiones_metodologicas.md # justificación de cada decisión
├── requirements.txt
└── README.md
```

## Cómo reproducir

```bash
pip install -r requirements.txt

# Pipeline completo (usa el snapshot crudo si existe):
python src/run_all.py

# Forzar re-descarga desde la API del Banco Mundial:
python src/run_all.py --download
```

El pipeline corre en ~30 s (sin descarga). Todas las salidas quedan en
`data/processed/` y `reports/figures/`. `random_state=42` fijo en todo.

## Notebook de presentación

`notebooks/presentacion.ipynb` **no recalcula** todo: lee `data/processed/` y
`reports/figures/` y arma la narrativa con código + figuras + interpretación. Ejecutar
`run_all.py` antes.

## Renderizar el informe a PDF

`reports/informe_borrador.md` es un **borrador** para que los humanos editen y validen.
Para producir el PDF (requiere [Pandoc](https://pandoc.org/) y un motor LaTeX):

```bash
cd reports
pandoc informe_borrador.md -o informe.pdf --pdf-engine=xelatex \
  --toc -V geometry:margin=2.5cm -V lang=es
```

(Alternativa sin LaTeX: abrir el `.md` en VS Code / Typora y exportar a PDF, o usar
`pandoc ... -o informe.html` y "imprimir a PDF".)

## Notas

- Fuente: World Development Indicators (`source=2`), API del Banco Mundial. Snapshot
  fechado en `data/raw/manifest.json`.
- El indicador de CO2 usa la serie `EN.GHG.CO2.PC.CE.AR5` (la clásica `EN.ATM.CO2E.PC`
  fue discontinuada en WDI).
