# Patrones de desarrollo de los países: PCA y clustering sobre WDI

Trabajo final de **Análisis Multivariado y Descubrimiento de Patrones**  
Licenciatura en Ciencia de Datos, Universidad Austral.

Integrantes: **Benjamin Miles Rouillón, Bernardo Di Rienzo y Lautaro Sanfilippo**.  
Fecha de entrega: **9 de junio de 2026**.

## Resumen

Este repositorio contiene un análisis reproducible de indicadores de desarrollo de
países a partir de los *World Development Indicators* (WDI) del Banco Mundial. El
trabajo combina:

- **Análisis de Componentes Principales (PCA)** para identificar dimensiones latentes
  del desarrollo.
- **Análisis de conglomerados** para agrupar países según sus perfiles multivariados.
- **Comparación temporal 2005 vs. 2023** para estudiar movilidad relativa en el
  gradiente de desarrollo.
- **Análisis de sensibilidad** para evaluar la robustez de las decisiones
  metodológicas.

El corte principal es **2023**, elegido por cobertura y estabilidad macroeconómica. El
año de referencia temprana es **2005**, elegido como el primer año desde 2000 con
cobertura suficiente para las variables finales. La base final contiene **14
indicadores numéricos**, aproximadamente **191 países en 2023** y un **panel común de
183 países** para la comparación temporal.

## Entrega final

El informe final está en:

```text
informe_final/index.html
```

La carpeta `informe_final/` es autocontenida: incluye HTML, CSS, JavaScript y las
figuras necesarias para abrir el informe localmente. Las figuras están organizadas con
atributos `data-chart-id` y `data-static-src`, lo que permite reemplazarlas más adelante
por gráficos interactivos sin reescribir la narrativa.

Para abrirlo, se puede hacer doble clic en `informe_final/index.html`. Si el navegador
bloquea recursos locales, levantar un servidor simple desde la raíz del proyecto:

```bash
python -m http.server 8017
```

y abrir:

```text
http://localhost:8017/informe_final/index.html
```

No se entrega notebook: la carpeta `notebooks/` fue eliminada intencionalmente.

## Estructura del repositorio

```text
proyecto/
├── data/
│   ├── raw/          # snapshot crudo de la API WDI y metadata de países
│   └── processed/    # datasets limpios, PCA, clusters, métricas y trayectorias
├── informe_final/
│   ├── index.html    # informe final de entrega
│   ├── assets/       # estilos y JS del informe
│   └── figures/      # figuras usadas por el informe final
├── reports/
│   ├── figures/      # figuras generadas por el pipeline
│   └── decisiones_metodologicas.md
├── src/
│   ├── config.py        # rutas, indicadores, años, umbrales y semilla
│   ├── download.py      # descarga WDI y snapshot crudo
│   ├── coverage.py      # cobertura por variable y año
│   ├── clean.py         # filtrado, faltantes y panel común
│   ├── prep.py          # transformaciones, estandarización e imputación
│   ├── eda.py           # análisis univariado, bivariado y categórico
│   ├── pca.py           # PCA y análisis paralelo de Horn
│   ├── clustering.py    # k-means, Ward, selección de k y perfilado
│   ├── compare_years.py # comparación temporal 2005 vs. 2023
│   ├── sensitivity.py   # sensibilidad metodológica
│   └── run_all.py       # orquestador del pipeline completo
├── requirements.txt
└── README.md
```

## Reproducibilidad

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el pipeline completo usando el snapshot crudo ya guardado:

```bash
python src/run_all.py
```

Forzar una nueva descarga desde la API del Banco Mundial:

```bash
python src/run_all.py --download
```

Todas las salidas reproducibles se escriben en `data/processed/` y
`reports/figures/`. La semilla global es `random_state = 42`.

## Principales decisiones metodológicas

- Se usan WDI del Banco Mundial porque ofrecen indicadores comparables entre países y
  cobertura internacional amplia.
- Se excluyen agregaciones regionales o de ingreso para que la unidad de análisis sea
  siempre el país.
- Se descartan variables con baja cobertura o redundancia fuerte, como Gini, gasto
  público en educación e INB per cápita PPA.
- Se transforman variables muy asimétricas antes de estandarizar: log para PBI per
  cápita, log1p para CO₂ per cápita y Yeo-Johnson para algunas variables con colas y
  posibles valores negativos.
- El PCA se ajusta sobre variables transformadas, estandarizadas e imputadas, y la
  retención de componentes se decide principalmente por análisis paralelo de Horn.
- El clustering principal se realiza sobre las 14 variables completas
  transformadas/estandarizadas; el clustering sobre componentes PCA se usa como
  comparación para evitar asumir sin evidencia que la reducción no afecta los grupos.
- La comparación temporal usa posición relativa dentro de cada año, no coordenadas
  absolutas, porque variables como Internet tienen una tendencia secular global muy
  fuerte.

El detalle completo está en `reports/decisiones_metodologicas.md`.

## Resultados centrales

- El **PC1** se interpreta como un gradiente de desarrollo: mayor ingreso real,
  esperanza de vida, conectividad, urbanización y menor peso de la agricultura.
- El PCA retiene **3 componentes** por análisis paralelo de Horn, con **64,9%** de
  varianza acumulada.
- El clustering principal identifica **2 grupos**: países en vías de desarrollo y
  países desarrollados.
- La partición de 2 grupos es estable y está fuertemente asociada con la clasificación
  oficial de ingreso del Banco Mundial.
- Entre 2005 y 2023, la estructura del gradiente se mantiene estable; la movilidad se
  interpreta de forma relativa para evitar confundir progreso secular global con
  mejora frente a pares.

## Notas sobre la fuente

- Fuente: World Development Indicators, Banco Mundial (`source=2`).
- El snapshot crudo queda guardado en `data/raw/` para mantener reproducibilidad aunque
  la API cambie.
- El indicador de CO₂ usa `EN.GHG.CO2.PC.CE.AR5`; la serie clásica
  `EN.ATM.CO2E.PC` fue discontinuada en WDI.
