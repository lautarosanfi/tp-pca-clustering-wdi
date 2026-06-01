# Registro de decisiones metodológicas

> Documento de respaldo del TP Final de *Análisis Multivariado y Descubrimiento de
> Patrones*. Para **cada** decisión se indica: qué se decidió, su justificación
> estadística, la alternativa considerada y qué mostró la sensibilidad.
> Las cifras provienen de correr el pipeline (`src/run_all.py`); ver
> `reports/run_all_log.txt` y `data/processed/`.

## Regla de oro aplicada

Antes de cada técnica nos preguntamos: ¿qué supuesto exige?, ¿se cumple?, ¿cómo lo
verifico?, ¿qué pasa si no? Las respuestas están distribuidas en las secciones de
abajo y se reflejan en el código.

---

## 1. Fuente, unidad de análisis y tamaño

- **Decisión:** World Development Indicators (WDI, `source=2`) vía API REST del Banco
  Mundial. Unidad = país; corte transversal anual. Dos años: **2023** (moderno) y
  **2005** (temprano de referencia).
- **Justificación:** datos mayormente numéricos y comparables entre países; ≤ 10.000
  observaciones (≈191 países × 2 años); ≥ 5 variables numéricas. Se guarda un
  **snapshot crudo** (`data/raw/wdi_long.csv`) y todo el análisis trabaja desde él, de
  modo que sea reproducible aunque la API cambie.
- **Filtrado de entidades:** se descartan las **agregaciones** (World, Euro area,
  grupos de ingreso, etc.) quedándose solo con entidades cuya `region` del endpoint de
  países no sea `"Aggregates"` y tengan ISO3 válido (217 países reales).

## 2. Elección de los años (moderno 2023; temprano 2005)

### 2.a Año moderno (2023, no 2021 ni 2022)

- **Decisión:** año moderno = **2023**.
- **Justificación (estabilidad macroeconómica):** PC1 (el eje de interés) y el clustering
  son sensibles a años con shocks idiosincráticos. **2021** está distorsionado por el
  **rebote post-COVID** (crecimiento del PBI anómalamente alto al revertir la caída de
  2020); **2022** por el **shock inflacionario global y la guerra de Ucrania**. **2023**
  refleja un entorno **normalizado**. Evidencia concreta: con el corte 2021 la carga del
  *crecimiento del PBI* en PC1 se invertía de signo (de −0,11 en 2005 a +0,39 en 2021),
  un artefacto del rebote; con 2023 esa carga es ≈ 0 en ambos años (estructura estable).
- **Cobertura:** se verificó que 2023 mantiene cobertura adecuada en las 14 variables
  (todas ≥ 0,78). **2024** se descartó: cae por debajo del umbral del 75% en formación de
  capital (0,719) y comercio (0,737), por rezago de publicación.
- **Alternativa considerada:** 2021 (usado en una versión previa). Se reemplazó tras
  comprobar que introducía el artefacto de crecimiento descrito.

### 2.b Año temprano (2005, no 2000)

- **Decisión:** año temprano = **2005**.
- **Justificación:** se descargó 1990–2024 y se midió **cobertura por variable y año**
  (`data/processed/cobertura_por_anio.csv`). Con el umbral adoptado de **≥ 75% de
  países con dato** por variable, 2005 es el **año más temprano (≥2000)** en que *todas*
  las variables seleccionadas superan el umbral. La variable limitante es la formación
  bruta de capital: 0,728 (2000), 0,742 (2003), 0,747 (2004) y **0,756 (2005)**.
- **Alternativa considerada:** 2000 (maximiza la ventana temporal). Se descartó porque
  la formación de capital y otras quedan por debajo del umbral, forzando más
  imputación en el año temprano. 2005 mantiene una ventana amplia (16 años) con datos
  más sólidos. (Nota: el indicador de Internet, históricamente problemático en 2000,
  aquí tiene buena cobertura por backfill del Banco Mundial: 0,90 en 2000.)

## 3. Selección de variables numéricas (14)

- **Decisión:** 14 indicadores (lista en `config.INDICATORS_FINAL`). Se **descartaron 3**
  de la lista candidata:
  | Variable | Motivo |
  |---|---|
  | `SI.POV.GINI` (Gini) | Cobertura 0,26–0,40: encuestas esporádicas, inviable. |
  | `SE.XPD.TOTL.GD.ZS` (gasto educación) | 0,49–0,55 en años tempranos: incomparable en el tiempo. |
  | `NY.GNP.PCAP.PP.KD` (INB PPA) | 0,53 en 2005 **y** redundante con PBI per cápita (r≈0,98). |
- **Justificación:** se prioriza **comparabilidad temporal** (mismo conjunto de
  variables en ambos años) y cobertura; se evita duplicar la dimensión de ingreso.
- **Nota técnica (CO2):** el código clásico `EN.ATM.CO2E.PC` fue **discontinuado** en
  WDI; se usa el reemplazo `EN.GHG.CO2.PC.CE.AR5` (CO2 per cápita, excl. LULUCF), con
  cobertura estable 0,935 en todos los años.
- **Cierre composicional sectorial:** agricultura + industria + servicios (% PBI) NO
  forman un cierre estricto (suma media ~90%, sd ~13 en el año moderno, por "impuestos menos
  subsidios"). No hay colinealidad perfecta, así que se **mantienen las tres**; el
  análisis paralelo de Horn descartaría un eventual componente degenerado.

## 4. Datos faltantes e imputación

- **Decisión:** filtro por país (descartar si > 40% de variables faltantes en el año) y
  reporte de faltantes por variable. Imputación **KNN (k=5) sobre datos
  estandarizados**, **dentro de cada año por separado**.
- **Justificación:** tras el filtro, el máximo de faltantes por variable es ≤ 15%
  (apto para imputar). KNN sobre datos estandarizados usa distancias comparables.
  Imputar por año evita "usar el futuro para el pasado". KNNImputer es determinista.
- **Resultado:** 192 países (2005), 191 (2023), panel común de **183**.
- **Sensibilidad:** la partición es casi idéntica con **MICE** (ARI 0,979; correlación
  de cargas de PC1 = 1,000) y con **casos completos** sin imputar (ARI 0,972; corr
  0,997). ⇒ la imputación **no** dirige los resultados.

## 5. Transformaciones (variable por variable)

- **Decisión:** se miró la **asimetría** de cada variable antes de transformar
  (`data/processed/skewness.csv`) y se decidió individualmente
  (`data/processed/skewness_antes_despues.csv`):
  | Transformación | Variables | Por qué |
  |---|---|---|
  | log natural | PBI per cápita | Positiva estricta, multi-orden (250–110.873), skew 2,2. |
  | log1p | CO2 per cápita | Positiva con valores ~0, skew 5,3. |
  | Yeo-Johnson | inflación, desempleo, export., import., agricultura, industria | Skew 1–13; **admite negativos** (inflación) y λ se ajusta por datos. |
  | sin transformar | crecimiento PBI, formación de capital, servicios, urbano, esperanza de vida, internet | Asimetría baja o acotada/saturante: no sobre-transformar. |
- **Justificación de detalles:**
  - **Base del logaritmo irrelevante** para PCA/clustering (monótona; la
    estandarización posterior absorbe el factor constante). Se usa log natural por
    interpretabilidad.
  - **Crecimiento del PBI** NO se transforma: su asimetría original es leve, tiene
    valores negativos (sale el log directo) y está dominada por outliers puntuales
    (rebotes post-COVID), de modo que Yeo-Johnson no la mejora de forma estable. Se deja
    en bruto y se estandariza.
  - **Orden correcto:** primero transformar (corregir asimetría), **después**
    estandarizar.
  - Para la comparación temporal, las transformaciones se ajustan sobre la
    **distribución combinada** de ambos años y se aplican idénticas a los dos.
- **Resultado:** la asimetría queda en |skew| < 0,6 en casi todas las variables tras
  transformar.

## 6. Estandarización

- **Decisión:** `StandardScaler` (media 0, sd 1) **siempre** antes de PCA y de
  clustering.
- **Justificación:** las variables están en escalas incomparables (dólares, %, años,
  toneladas). Sin estandarizar, el PBI per cápita dominaría artificialmente.
- **Fit único:** para el análisis transversal 2023 el scaler se ajusta en 2023; para la
  **comparación temporal** se ajusta **una sola vez sobre los dos años apilados** y se
  aplica idéntico a ambos, de modo que las posiciones sean comparables (no se
  estandariza cada año por separado).
- **Sensibilidad (hallazgo instructivo):** con `RobustScaler` la partición se **rompe**
  (ARI −0,004; corr de cargas PC1 = 0,009). La causa NO es aleatoria: RobustScaler escala
  por el IQR pero **no acota las colas**. El crecimiento del PBI de **Macao en 2023
  (+75%, rebote turístico post-pandemia)**, al dividirse por un IQR pequeño (≈3,8), se
  vuelve enorme y **secuestra PC1** (que pasa a ser un eje de "crecimiento"), arrastrando
  el clustering. Quitando **solo a Macao**, RobustScaler se recompone (ARI 0,743; corr de
  PC1 0,925). Conclusión: para datos con colas pesadas sin transformar, `StandardScaler`
  (cuya SD, inflada por los outliers, los atenúa) es **preferible** a RobustScaler; este
  resultado **respalda** nuestra elección por defecto. PCA es sensible a outliers.

## 7. PCA: cuántos componentes retener

- **Decisión:** retención por **convergencia de criterios**, con énfasis en el
  **análisis paralelo de Horn** (implementado a mano con NumPy: 1000 matrices normales
  n×p, percentil 95 de autovalores).
- **Resultado (2023):** Kaiser (autovalor>1) → 4; varianza acumulada 80% → 6;
  **Horn → 3** (64,9% acumulada). Los criterios convergen en 3–4. Se adopta **3** para
  análisis e interpretación.
- **Justificación:** Kaiser sobreestima y el 80% es arbitrario; Horn compara contra
  ruido y es el estándar de oro. El scree muestra el codo en 3–4.
- **Distinción explícita (clave):**
  - **Componentes para CLUSTERIZAR:** los 3 de Horn (usados solo como *comparación*; ver
    §8).
  - **Componentes para VISUALIZAR:** 2 (PC1-PC2), únicamente para graficar.
  - Nunca se mezclan ambos usos.
- **Interpretación de ejes (cargas como correlación variable-componente):**
  - **PC1 (41,3%) = gradiente de desarrollo:** + PBI pc (0,93), internet (0,90),
    esperanza de vida (0,87), CO2 (0,81); − agricultura (−0,92).
  - **PC2 (13,1%) = estructura productiva:** + industria, capital; − servicios, desempleo.
  - **PC3 (10,5%) = apertura comercial / macro:** + importaciones, exportaciones;
    − inflación.

## 8. Clustering: ¿variables completas o componentes PCA? (el punto central)

- **Decisión:** el clustering **principal** se hace sobre las **14 variables completas**
  (transformadas + estandarizadas). Se **compara** contra el clustering sobre los 3
  componentes de Horn y se mide el acuerdo con **ARI**.
- **Justificación:** con ~14 variables **no** estamos en alta dimensión; reducir a
  pocos componentes *antes* de clusterizar (tandem analysis) puede descartar estructura
  de grupos. PCA maximiza varianza, no separación. Por eso clusterizamos sobre las
  variables y usamos PCA como herramienta de des-correlación/interpretación.
- **Resultado (k=2):** ARI(14 vars vs 3 PCs) = **1,000**: las particiones son idénticas.
  ⇒ aquí la estructura de grupos vive en los componentes de alta varianza y el tandem
  **no** sesga — pero lo **demostramos**, no lo asumimos. El silhouette es 0,28 en 14
  vars y 0,40 en 3 PCs: el PCA "limpia" ruido de las dimensiones de baja varianza y la
  separación *se ve* mejor, aunque la **partición es la misma**.
- **Sensibilidad:** clustering sobre 2/3/5 PCs ⇒ ARI 0,94 / 1,00 / 1,00 vs base.

## 9. Algoritmos y número de clusters

- **Decisión:** dos algoritmos — **k-means** y **jerárquico de Ward** — y k por
  **consenso de métricas** + **estabilidad bootstrap (Hennig)**.
- **Supuestos de k-means:** asume clusters esféricos, de tamaño/varianza similar y
  distancia euclidiana. Se verifica que son razonables; ver abajo la advertencia de
  "continuo".
- **Métricas (k-means, 14 vars):** silhouette, Calinski-Harabasz, Davies-Bouldin **y el
  gap statistic coinciden en k=2** (silhouette 0,28; CH 77,1; DB 1,38).
- **Estabilidad (Jaccard medio, criterio de Hennig: >0,75 estable, <0,6 disuelto):**
  k=2 → 0,96/0,94 (muy estable); k=3 → 0,73–0,88; k=4 → mín 0,27 (se disuelve).
- **Acuerdo entre algoritmos:** ARI(k-means vs Ward) = **0,62** en k=2 (moderado) y
  **0,75** en k=3. El acuerdo *menor* en k=2 es honesto y esperable: al ser el desarrollo
  un **continuo**, la ubicación exacta del corte binario depende algo del algoritmo,
  aunque la separación "desarrollado/en desarrollo" es consistente.
- **Decisión final:** **k=2 como partición principal** (respaldada por silhouette, CH,
  DB, **gap** y estabilidad 0,96/0,94), y **k=3 como vista complementaria** (recupera un
  gradiente bajo/emergente/desarrollado, χ² aún más fuerte). Ninguna se "vende" como
  taxonomía fina.
- **Alternativas (supuestos):**
  - **GMM (elípticos):** ARI 0,62 vs k-means k=2 — relacionado pero no idéntico.
  - **HDBSCAN (densidad):** marca **~59% de los países como ruido** y solo halla 2
    núcleos densos. ⇒ evidencia honesta de que el desarrollo es un **continuo** (el
    gradiente PC1), no grumos densos naturales. El clustering particiona un continuo;
    por eso el silhouette es **moderado** y se interpreta como **macro-separación**, no
    como tipología fina.

## 10. Variables categóricas (suplementarias)

- **Decisión:** región y nivel de ingreso del Banco Mundial **no** entran al PCA ni al
  k-means (no se hace one-hot: distorsiona distancias). Se usan de forma
  **descriptiva/ilustrativa**:
  - EDA: conteos, boxplots de numéricas por ingreso, etc.
  - PCA: colorear los scores (biplot por ingreso/región) — proyección, no definición de
    ejes.
  - Perfilado de clusters: tablas cruzadas cluster × ingreso y cluster × región con
    **χ²**.
- **Resultado:** χ²(cluster × ingreso) = 135,8 (p≈2e-28) en k=2 y 228,9 (p≈5e-45) en
  k=3: los clusters **coinciden fuertemente** con la clasificación oficial de ingreso,
  con **excepciones interesantes** (medio-alto en el grupo menos desarrollado: Indonesia,
  Guatemala, Fiji, islas del Pacífico; medio-bajo en el más desarrollado: Vietnam,
  Marruecos, Túnez, Jordania).

## 11. Comparación temporal 2005 vs 2023 (posición RELATIVA)

- **Estructura — espacio común (válido).** Se ajusta **un** PCA sobre los dos años
  apilados (transformaciones/escala sobre la distribución combinada; imputación por año).
  Esto define el eje de desarrollo de forma idéntica en ambos cortes y permite comparar
  la **estructura** (cargas). Congruencia de Tucker de PC1 (2005 vs 2023) = **0,962**
  (corr 0,946): el gradiente es el **mismo eje**.
- **Por qué NO se comparan posiciones absolutas (corrección metodológica clave).**
  Varias variables que definen PC1 tienen fuerte **tendencia secular**: la mediana de
  uso de Internet pasó de **10 % (2005) a 81 % (2023)**. En el espacio común esto corre a
  *todos* los países a la derecha (PC1 medio −0,47 → +0,47). La descomposición del
  corrimiento absoluto (`descomposicion_dPC1.csv`) lo confirma: **Internet 47 %,
  esperanza de vida 22 %**, PBI real solo 9 %. Por lo tanto, el "avance absoluto" mide
  sobre todo **difusión tecnológica/sanitaria mundial**, no desarrollo *relativo*. Tomar
  la posición absoluta haría aparecer a 2005 como menos desarrollado de lo que era solo
  porque Internet aún no se había difundido (p. ej., en el espacio común 2005 tiene 82
  "desarrollados", contra 107 cuando se mide relativo — la diferencia es puro artefacto
  secular). *(El PBI, además, es real: `NY.GDP.PCAP.KD` en US$ constantes de 2015 — no
  hay artefacto de precios.)*
- **Decisión: comparar posición RELATIVA dentro de cada año** (robusta a la tendencia
  secular), de dos formas:
  - **Tiers relativos:** k-means k=2 **dentro de cada año** ⇒ "desarrollado" = grupo alto
    entre los contemporáneos. Resultado: **107/192 desarrollados en 2005** (incluye
    EE.UU., Alemania, Japón, Corea, etc.) y **121/191 en 2023**; la proporción sube
    moderadamente (≈56 % → ≈63 %). Validación: el tier relativo de 2023 coincide con el
    modelo principal (ARI **1,000**).
  - **Percentil de PC1 dentro del año:** mide si un país ganó/perdió terreno frente a sus
    pares.
- **Resultados (movilidad relativa, panel común):** **15 países ascendieron** de tier
  (catch-up: China, Vietnam, Georgia, Albania, Azerbaiyán, Botsuana, Marruecos, …) y
  **solo 1 descendió** (Islas Marshall). Mayores ascensos en percentil: Georgia (+19),
  China (+17), Mongolia (+14). Mayores descensos: **Venezuela (−28), Líbano (−26),
  Argentina (−16)** — países en crisis. Los desarrollados de larga data (EE.UU., Alemania,
  Corea) están en el percentil más alto en **ambos** años.
- **Entidades que cambiaron entre años:** panel común de 183 países; las que solo están
  en un año (p. ej. Siria, fuera de 2023 por faltantes) quedan fuera de las trayectorias
  (`reporte_limpieza.json`).

## 12. Reproducibilidad

- `random_state = 42` en todo (k-means, bootstrap, gap, imputadores, Yeo-Johnson).
- Snapshot crudo guardado; `requirements.txt` con versiones; `src/run_all.py` reproduce
  el pipeline de punta a punta.

## 13. Honestidad intelectual / limitaciones

- El clustering es una **macro-separación** (desarrollado vs en desarrollo), **no** una
  taxonomía fina: silhouette moderado y HDBSCAN viendo un continuo.
- Corte transversal: no hay inferencia causal.
- Imputación (≤15%) y outliers (re-export hubs, petro-estados): la sensibilidad muestra
  que **no** dirigen las conclusiones (sin 10 outliers: ARI 0,934). La única
  excepción documentada es RobustScaler × el outlier de crecimiento de Macao (ver §6).
- El cambio de definición del indicador de CO2 obliga a usar la serie AR5; se documenta.
- WDI tiene rezagos y revisiones; por eso se fija un snapshot fechado.
