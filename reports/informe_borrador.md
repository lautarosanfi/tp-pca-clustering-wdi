---
title: "Patrones de desarrollo de los países: PCA y clustering sobre indicadores del Banco Mundial"
subtitle: "Trabajo Práctico Final — Análisis Multivariado y Descubrimiento de Patrones"
author: "[Integrantes del grupo] — Lic. en Ciencia de Datos, Universidad Austral"
date: "2026"
lang: es
---

> **⚠️ BORRADOR generado automáticamente.** Pensado para que los integrantes lo
> **editen, validen y completen** (carátula, nombres, redacción final). Todas las
> cifras y figuras provienen de correr `src/run_all.py`. Revisar interpretaciones
> antes de la entrega.

---

# Carátula

- **Materia:** Análisis Multivariado y Descubrimiento de Patrones
- **Carrera:** Licenciatura en Ciencia de Datos — Universidad Austral
- **Trabajo:** Práctico Final — PCA + Análisis de Clustering
- **Integrantes:** *[completar]*
- **Fecha:** *[completar]*
- **Dataset:** World Development Indicators (Banco Mundial), corte 2023 con comparación a 2005.

---

# 1. Introducción

## 1.1 Problema y objetivo

¿Es posible **resumir** el desarrollo socioeconómico de los países con unos pocos ejes
interpretables, y **agruparlos** en perfiles de desarrollo a partir de datos objetivos?
El interés práctico es doble: (i) entender qué dimensiones estructuran las diferencias
entre países y (ii) ver si surgen grupos coherentes y cómo se movieron los países en
~18 años.

Aplicamos dos técnicas multivariadas complementarias:

1. **Análisis de Componentes Principales (PCA):** reduce las variables correlacionadas a
   componentes ortogonales que capturan la mayor variabilidad, para **interpretar** las
   dimensiones del desarrollo.
2. **Análisis de Clustering:** agrupa países por similitud para descubrir **perfiles**.

Elegimos PCA (y no Análisis de Correspondencias) porque los datos son mayormente
**numéricos**. Las variables **categóricas** (región y nivel de ingreso del Banco
Mundial) se usan de forma **suplementaria/ilustrativa** (no construyen los componentes).

## 1.2 Datos

- **Fuente:** World Development Indicators (WDI, `source=2`), API pública del Banco
  Mundial. Se guardó un *snapshot* crudo y todo el análisis trabaja desde él
  (reproducibilidad).
- **Unidad de análisis:** país (corte transversal anual). Se filtran las
  **agregaciones** (World, Euro area, grupos de ingreso) quedando **217 países reales**.
- **Años:** **2023** (moderno) y **2005** (temprano de referencia). La elección de
  ambos años se justifica por **cobertura y estabilidad macro** (§1.3).
- **Tamaño:** ≈191 países × 2 años ≪ 10.000 observaciones. ✔
- **14 variables numéricas** (≥5 ✔) + **2 categóricas**.

| Dimensión | Variables numéricas (WDI) |
|---|---|
| Ingreso / nivel | PBI per cápita (**US$ constantes de 2015** → magnitud **real**) |
| Macro | Crecimiento del PBI, inflación, desempleo |
| Inversión / comercio | Formación bruta de capital, exportaciones, importaciones (% PBI) |
| Estructura productiva | Agricultura, industria, servicios (valor agregado, % PBI) |
| Social / desarrollo | Población urbana (%), esperanza de vida, uso de Internet (%) |
| Ambiental | Emisiones de CO₂ per cápita |

Categóricas (suplementarias): **región** y **nivel de ingreso** del Banco Mundial.

## 1.3 Decisiones de datos (resumen)

- **Año moderno = 2023, no 2021/2022.** Se eligió por **estabilidad macroeconómica**:
  2021 está distorsionado por el **rebote post-COVID** (crecimiento anómalo), 2022 por el
  **shock inflacionario y la guerra de Ucrania**, mientras que 2023 refleja un entorno
  normalizado. 2023 mantiene **cobertura adecuada** en las 14 variables (todas ≥ 78%);
  2024 ya cae por debajo del umbral en 3 variables, por lo que se descartó.
- **Año temprano = 2005**: el más temprano (≥2000) en que *todas* las variables superan
  el 75% de cobertura (la limitante es la formación bruta de capital).
- Se **descartaron** Gini (cobertura 0,26–0,40), gasto en educación (incomparable en el
  tiempo) e INB-PPA (redundante con PBI pc y de baja cobertura).
- **Faltantes** tras filtros ≤ 15% por variable → imputación **KNN por año**.
- Países finales: **192 (2005)**, **191 (2023)**, **panel común = 183**.

*(El detalle y la justificación estadística de cada decisión están en
`reports/decisiones_metodologicas.md`.)*

---

# 2. Desarrollo

## 2.1 Análisis univariado y bivariado

### Univariado y transformaciones

Antes de transformar se examinó la **distribución y asimetría** de cada variable
(Figura 1). Varias presentan fuerte asimetría a derecha y varios órdenes de magnitud
(PBI per cápita, CO₂), otras tienen valores negativos (inflación, crecimiento), y otras
están acotadas en [0,100] (participaciones sectoriales, urbano, Internet).

![Figura 1. Distribuciones univariadas (2023).](figures/eda_histogramas_2023.png)

Decisión **variable por variable** (ver tabla en `decisiones_metodologicas.md` §5):
**log** para PBI pc, **log1p** para CO₂, **Yeo-Johnson** (admite negativos) para
inflación, desempleo, exportaciones, importaciones, agricultura e industria; y **sin
transformar** las de baja asimetría o saturantes. El orden es **transformar →
estandarizar**. Tras transformar, la asimetría queda en |skew| < 0,6 en casi todas.

### Bivariado

La matriz de correlación (Figura 2) muestra un **bloque de desarrollo** fuertemente
correlacionado (PBI pc, esperanza de vida, urbanización, Internet, CO₂; agricultura con
signo opuesto) y correlaciones esperables entre exportaciones e importaciones. Esta
estructura de correlación es justamente lo que el PCA aprovechará.

![Figura 2. Matriz de correlación (2023).](figures/eda_correlacion_2023.png)

Las variables clave por **nivel de ingreso** (categórica) ordenan monótonamente los
grupos (Figura 3): a mayor ingreso, mayor PBI pc, esperanza de vida e Internet, y menor
peso de la agricultura. Esto anticipa que el primer eje del PCA capturará un **gradiente
de desarrollo**.

![Figura 3. Variables clave por nivel de ingreso (2023).](figures/eda_boxplots_ingreso_2023.png)

![Figura 4. Composición de la muestra por región y nivel de ingreso.](figures/eda_categoricas_2023.png)

## 2.2 Técnica 1 — Análisis de Componentes Principales (PCA)

### Supuestos y preparación

PCA sobre la **matriz de correlación** (variables **estandarizadas**: las escalas son
incomparables —dólares, %, años, toneladas—). Sin estandarizar, el PBI pc dominaría
artificialmente.

### ¿Cuántos componentes retener?

No usamos un único criterio. El **análisis paralelo de Horn** (estándar de oro:
compara los autovalores reales contra los de datos aleatorios de la misma dimensión)
retiene **3 componentes**; Kaiser sugiere 4 y el 80% de varianza, 6 (Figura 5). Los
criterios **convergen en 3–4**; adoptamos **3** para la interpretación (64,9% de la
varianza).

![Figura 5. Scree plot con análisis paralelo de Horn y varianza acumulada.](figures/pca_scree_2023.png)

> Distinguimos explícitamente: **3 componentes para analizar/clusterizar** (Horn) y
> **2 componentes solo para visualizar** (los gráficos PC1–PC2). No se mezclan.

### Interpretación de los ejes (en términos del problema)

Cargas como **correlación variable–componente**:

- **PC1 (41,3%) — gradiente de desarrollo.** Positivo: PBI pc (0,93), Internet (0,90),
  esperanza de vida (0,87), CO₂ (0,81); negativo: **agricultura (−0,92)**. Es el eje
  "país rico, urbano, conectado y de servicios" vs "país agrario y de menor ingreso".
- **PC2 (13,1%) — estructura productiva.** Positivo: industria, formación de capital;
  negativo: servicios, desempleo. Separa economías **industriales/extractivas** de
  economías de **servicios**, con independencia del nivel de desarrollo.
- **PC3 (10,5%) — apertura comercial / estabilidad macro.** Importaciones y
  exportaciones (+) frente a inflación (−).

El **círculo de correlaciones** (Figura 6) y el **biplot** coloreado por nivel de
ingreso (Figura 7) confirman la lectura: PC1 separa nítidamente los niveles de ingreso
del Banco Mundial (validación con la categórica ilustrativa).

![Figura 6. Círculo de correlaciones (PC1–PC2).](figures/pca_circulo_correlaciones_2023.png)

![Figura 7. Biplot PCA coloreado por nivel de ingreso (2023).](figures/pca_biplot_ingreso_2023.png)

## 2.3 Técnica 2 — Análisis de Clustering

### ¿Clusterizar sobre las variables o sobre los componentes?

Con ~14 variables **no** estamos en alta dimensión. Clusterizar sobre **todas las
variables** estandarizadas es honesto y evita el riesgo del *tandem analysis* (reducir a
pocos componentes antes de clusterizar puede descartar estructura de grupos, ya que PCA
maximiza varianza y no separación). Por eso:

- **Partición principal: k-means sobre las 14 variables completas.**
- **Comparación: k-means sobre los 3 componentes de Horn**, midiendo el acuerdo (ARI).

**Resultado clave:** ARI(14 vars vs 3 PCs) = **1,000** en k=2 — las particiones son
idénticas. Aquí la estructura de grupos sí vive en los componentes de alta varianza, de
modo que el PCA **no** sesga la partición; pero esto se **demostró**, no se asumió. (El
silhouette sube de 0,28 con 14 variables a 0,40 con 3 PCs: el PCA "limpia" ruido y la
separación *se ve* mejor, aunque la partición es la misma.)

### Algoritmos, selección de k y estabilidad

Usamos **k-means** y **Ward** (jerárquico). El número de clusters se decide por
**consenso de métricas** (Figura 8) + **estabilidad bootstrap** (criterio de Hennig):

| Criterio | Indica |
|---|---|
| Silhouette, Calinski-Harabasz, Davies-Bouldin | **k = 2** |
| Gap statistic | **k = 2** |
| Estabilidad (Jaccard) | k=2: 0,96/0,94 (muy estable); k=3: 0,73–0,88; k=4: 0,27 (se disuelve) |
| ARI k-means vs Ward | 0,62 (k=2); 0,75 (k=3) |

![Figura 8. Métricas internas para elegir k.](figures/clust_metricas_k_2023.png)

![Figura 9. Dendrograma (Ward) sobre las 14 variables.](figures/clust_dendrograma_2023.png)

**Decisión:** **k = 2 como partición principal** — respaldada por silhouette, CH, DB,
**gap** y una estabilidad bootstrap altísima (0,96/0,94). El ARI k-means vs Ward de 0,62
(moderado) es honestamente menor que en k=3 (0,75): refleja que, al ser el desarrollo un
**continuo**, la *ubicación exacta* del corte binario depende algo del algoritmo, aunque
la separación general "desarrollado / en desarrollo" es consistente. Reportamos **k = 3
como vista complementaria** (recupera un gradiente bajo/emergente/desarrollado).

![Figura 10. Diagrama de silhouette (k=2).](figures/clust_silhouette_k2_2023.png)

### Supuestos de k-means y alternativas

k-means asume clusters esféricos de tamaño similar. Probamos alternativas: **GMM**
(elípticos) coincide moderadamente con k-means (ARI 0,62), y **HDBSCAN** (densidad)
marca **~59% de los países como ruido** con solo 2 núcleos densos. Esto es evidencia
**honesta** de que el desarrollo es un **continuo** (el gradiente PC1) más que grumos
densos naturales: el clustering provee una **macro-separación**, **no** una taxonomía
fina. Por eso el silhouette es **moderado**.

### Perfil de los clusters (k=2, en términos del problema)

![Figura 11. Clusters (k=2) proyectados sobre PC1–PC2.](figures/clust_pca_k2_2023.png)

| Variable (mediana) | Cluster 0 — *en desarrollo* (70 países) | Cluster 1 — *desarrollado* (121) |
|---|---:|---:|
| PBI per cápita (US$ 2015) | 1.439 | 14.933 |
| Esperanza de vida (años) | 66,9 | 77,7 |
| Población urbana (%) | 41,2 | 74,0 |
| Uso de Internet (%) | 42,2 | 87,1 |
| Agricultura (% PBI) | 19,2 | 3,0 |
| CO₂ per cápita (t) | 0,5 | 4,4 |
| Exportaciones (% PBI) | 22,7 | 44,9 |

El **cluster 0** reúne economías de menor ingreso, más agrarias, menos urbanas y
conectadas; el **cluster 1**, economías de mayor ingreso, urbanas, de servicios y más
integradas al comercio.

### Validación con las categóricas (suplementarias)

La tabla cruzada cluster × nivel de ingreso da **χ² = 135,8 (p ≈ 2·10⁻²⁸)** en k=2 y
**χ² = 228,9 (p ≈ 5·10⁻⁴⁵)** en k=3: los clusters **coinciden fuertemente** con la
clasificación oficial de ingreso del Banco Mundial. Las **excepciones** son
informativas:

- *Ingreso medio-alto pero perfil "en desarrollo"* (cluster 0): Indonesia, Guatemala,
  Fiji y varias islas del Pacífico (Marshall, Tonga, Samoa) — economías más agrarias /
  menos urbanizadas de lo que sugiere su ingreso.
- *Ingreso medio-bajo pero perfil "desarrollado"* (cluster 1): Vietnam, Marruecos,
  Túnez, Jordania, Líbano — más urbanizadas/conectadas/de servicios que su par de
  ingreso.

## 2.4 Comparación temporal 2005 vs 2023

Para comparar correctamente, ambos años viven en un **espacio común**: se apilan los dos
años, se deciden las transformaciones sobre la distribución combinada y se ajustan **un**
scaler y **un** PCA sobre los datos combinados (imputación siempre **por año**). No se
ajusta un PCA por año para comparar coordenadas (sería inválido). El modelo común
reproduce el análisis principal de 2023 (ARI 0,979).

**Trayectorias (Figura 12).** El movimiento medio en PC1 es **+0,96**: **casi todos los
países avanzaron** en el gradiente de desarrollo. Mayores avances: **Georgia, Zambia,
China, Camboya, Mongolia, Lesoto**; retrocesos: **Líbano, Venezuela, Sudán** (crisis
financiera / colapso económico / conflicto).

![Figura 12. Trayectorias 2005→2023 en el espacio PCA común.](figures/compare_trayectorias.png)

**Transiciones de cluster (Figura 13).** **40 países "graduaron"** de menos a más
desarrollado (Albania, Brasil, China, Georgia, Rumania, Turquía, Vietnam, …) y
**ninguno** retrocedió en términos de pertenencia de cluster (aunque Líbano, Venezuela y
Sudán se movieron hacia atrás sobre el gradiente PC1 sin cruzar la frontera).

![Figura 13. Matriz de transición de clusters 2005→2023.](figures/compare_transiciones.png)

### ¿Es válido decir que "tantos países mejoraron"? (PBI real vs nominal y descomposición)

Es una pregunta metodológica importante. Dos aclaraciones:

1. **El PBI per cápita está en términos REALES, no nominales.** Usamos
   `NY.GDP.PCAP.KD` = *GDP per capita, **constant 2015 US$***. Al estar en dólares
   **constantes**, la inflación **no** infla la comparación (ni entre países ni en el
   tiempo). El temor de que "el avance sea un artefacto de precios" no aplica.

2. **Aun así, el avance NO está dominado por el PBI.** Descomponemos el avance medio en
   PC1 (+0,96) en la contribución de cada variable (Figura 14;
   `descomposicion_dPC1.csv`). El score de PC1 es una combinación lineal de las
   variables estandarizadas, así que `mean(ΔPC1) = Σ wᵥ · mean(Δzᵥ)`:

   | Variable | % del avance medio en PC1 |
   |---|---:|
   | Uso de Internet | **47%** |
   | Esperanza de vida | **22%** |
   | PBI per cápita (real) | 9% |
   | Población urbana | 9% |
   | Servicios | 7% |
   | Caída de la agricultura | 7% |
   | resto (comercio, crecimiento, …) | ≈ 0% |

   El avance está **dominado por la explosión de la conectividad (Internet, 47%) y la
   mejora de la salud (esperanza de vida, 22%)** — ganancias de desarrollo **no
   monetarias**, inmunes a cualquier distorsión de precios. El PBI real aporta solo 9%.

![Figura 14. Descomposición del avance medio en PC1 por variable.](figures/compare_descomposicion_dPC1.png)

**Matiz importante (marea creciente vs reposicionamiento relativo).** Como Internet y la
esperanza de vida subieron *casi en todos los países*, el avance **absoluto** en PC1 es
casi universal: es una "marea creciente" de desarrollo global, no necesariamente
convergencia. Lo verdaderamente informativo es el **movimiento relativo**: qué países
cambiaron de cluster (40 graduaron) y cuáles retrocedieron sobre el gradiente (Líbano,
Venezuela, Sudán). Esa lectura *relativa* es robusta y es la que destacamos.

**Estabilidad de la estructura (Figura 15).** La congruencia de Tucker de las cargas de
PC1 entre 2005 y 2023 es **0,962**: el gradiente de desarrollo es el **mismo eje** a ~18
años. La **agricultura (−0,90)** es el marcador más estable; **exportaciones e
importaciones ganaron** peso (mayor integración comercial). A diferencia del corte 2021,
con 2023 el **crecimiento del PBI** ya **no** distorsiona PC1 (carga ≈ 0 en ambos años),
lo que confirma la conveniencia de usar un año macroeconómicamente estable.

![Figura 15. Estabilidad de las cargas de PC1 por año.](figures/compare_loadings_pc1.png)

## 2.5 Análisis de sensibilidad

Las conclusiones son **robustas** a las decisiones metodológicas (ARI de la partición
k=2 vs línea base = Standard + KNN + 14 vars):

| Variante | ARI vs base | Corr. cargas PC1 |
|---|---:|---:|
| Imputación MICE | 0,958 | 1,000 |
| Casos completos (sin imputar) | 1,000 | 0,996 |
| Clustering sobre 2 / 3 / 5 PCs | 0,94 / 1,00 / 1,00 | 1,000 |
| Sin 10 outliers | 0,934 | 0,982 |
| **RobustScaler (todos los países)** | **−0,004** | **0,009** |
| RobustScaler (sin Macao) | 0,743 | 0,925 |

Ni la imputación, ni la cantidad de componentes, ni la mayoría de los outliers
**dirigen** los resultados. El único quiebre es **RobustScaler**, y es instructivo:
RobustScaler escala por el IQR pero **no acota las colas**; el **crecimiento del PBI de
Macao en 2023 (+75%, rebote turístico post-pandemia)** queda enorme al dividirse por un
IQR pequeño y **secuestra PC1** (que pasa a ser un eje de "crecimiento"). Quitando solo a
Macao, RobustScaler se recompone (ARI 0,74; correlación de PC1 0,93). Esto **justifica**
nuestra elección de `StandardScaler` (cuya SD, inflada por los outliers, los atenúa) +
Yeo-Johnson para las variables asimétricas.

---

# 3. Conclusiones

1. El desarrollo de los países se resume bien en **un eje dominante** (PC1, 41% de la
   varianza): un **gradiente de desarrollo** (ingreso, salud, urbanización,
   conectividad vs peso agrario). Un segundo eje captura la **estructura productiva**
   (industria vs servicios) y un tercero la **apertura/estabilidad macro**.
2. El clustering revela una **macro-separación robusta en dos grupos**
   (desarrollado / en desarrollo), muy alineada con la clasificación de ingreso del
   Banco Mundial pero con excepciones informativas. **No** es una taxonomía fina: el
   desarrollo es esencialmente un **continuo** (silhouette moderado; HDBSCAN ve un
   continuo).
3. Entre **2005 y 2023** casi todos los países **avanzaron** en el gradiente; **40
   graduaron** de grupo y ninguno regresó en términos de cluster. El avance está
   impulsado sobre todo por **conectividad y salud** (no por el PBI, que además es
   **real**), por lo que se trata de una mejora de desarrollo genuina y no de un
   artefacto de precios. La **estructura** del desarrollo es muy **estable**
   (congruencia de PC1 = 0,96).
4. Metodológicamente: clusterizar sobre las **variables completas** y **comparar** contra
   los componentes evita el sesgo del *tandem analysis*; aquí ambos coinciden (ARI 1,00),
   pero se demostró en vez de asumirse.

## Limitaciones

- Corte transversal: **sin inferencia causal**.
- El clustering es una **macro-separación**, no una tipología fina (lo decimos
  explícitamente).
- El avance temporal absoluto refleja en parte una **tendencia secular global** ("marea
  creciente" de conectividad y longevidad); por eso priorizamos la lectura **relativa**
  (transiciones de cluster).
- Imputación (≤15%) y outliers: la sensibilidad muestra que no afectan las conclusiones,
  salvo la interacción documentada entre RobustScaler y el outlier de crecimiento de
  Macao.
- Cambios de definición/cobertura de WDI (p. ej. la serie de CO₂): documentados.

---

# Apéndice (opcional)

- **A1.** Cobertura por variable y año (`data/processed/cobertura_por_anio.csv`).
- **A2.** Asimetría antes/después de transformar (`skewness_antes_despues.csv`).
- **A3.** Autovalores y cargas completas del PCA (`pca_autovalores_2023.csv`,
  `pca_loadings_2023.csv`).
- **A4.** Métricas de selección de k, estabilidad y perfilado completo
  (`clustering_resultados_2023.json`, `perfil_medianas_k*_2023.csv`).
- **A5.** Trayectorias, transiciones y descomposición del avance
  (`trayectorias.csv`, `transiciones_cluster.csv`, `descomposicion_dPC1.csv`).
- **A6.** Tabla de sensibilidad (`sensibilidad.csv`).
- **A7.** Registro completo de decisiones (`reports/decisiones_metodologicas.md`).
