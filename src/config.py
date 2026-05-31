"""
Configuración central del proyecto: rutas, indicadores, años y parámetros.

Toda decisión "dura" (qué variables, qué años, semillas, umbrales) vive acá para
que el pipeline sea reproducible y auditable desde un solo lugar.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (DATA_RAW, DATA_PROC, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# API World Bank (World Development Indicators, source=2)
# --------------------------------------------------------------------------- #
WB_BASE = "https://api.worldbank.org/v2"
WB_SOURCE = 2

# Descargamos un rango amplio para poder evaluar cobertura y elegir el año
# temprano con criterio (no fijamos 2000 a ciegas).
DOWNLOAD_DATE_RANGE = "1990:2024"

# --------------------------------------------------------------------------- #
# Indicadores numéricos candidatos (WDI)
# Se descargan todos; la selección final depende de la cobertura observada.
# --------------------------------------------------------------------------- #
INDICATORS = {
    "NY.GDP.PCAP.KD":    "PBI per capita (US$ const. 2015)",
    "NY.GNP.PCAP.PP.KD": "INB per capita PPA ($ int. const.)",
    "NY.GDP.MKTP.KD.ZG": "Crecimiento anual del PBI (%)",
    "FP.CPI.TOTL.ZG":    "Inflacion anual al consumidor (%)",
    "SL.UEM.TOTL.ZS":    "Desempleo (% fuerza laboral)",
    "NE.GDI.TOTL.ZS":    "Formacion bruta de capital (% PBI)",
    "NE.EXP.GNFS.ZS":    "Exportaciones (% PBI)",
    "NE.IMP.GNFS.ZS":    "Importaciones (% PBI)",
    "NV.AGR.TOTL.ZS":    "Agricultura, valor agregado (% PBI)",
    "NV.IND.TOTL.ZS":    "Industria, valor agregado (% PBI)",
    "NV.SRV.TOTL.ZS":    "Servicios, valor agregado (% PBI)",
    "SP.URB.TOTL.IN.ZS": "Poblacion urbana (% del total)",
    "SP.DYN.LE00.IN":    "Esperanza de vida al nacer (anios)",
    "SE.XPD.TOTL.GD.ZS": "Gasto publico en educacion (% PBI)",
    "IT.NET.USER.ZS":    "Personas usando Internet (% poblacion)",
    "EN.GHG.CO2.PC.CE.AR5": "Emisiones de CO2 per capita (ton., excl. LULUCF)",
    "SI.POV.GINI":       "Indice de Gini",
}

# --------------------------------------------------------------------------- #
# Años de análisis
# Valores adoptados tras el análisis de cobertura y estabilidad macro
# (ver decisiones_metodologicas.md §2).
# --------------------------------------------------------------------------- #
# 2023 = año moderno. Se prefiere 2023 sobre 2021/2022 por ESTABILIDAD macro:
#   2021 = rebote post-COVID (crecimiento anómalo); 2022 = shock inflacionario y
#   guerra de Ucrania; 2023 = entorno normalizado, con cobertura adecuada (todas
#   las variables >=78%; 2024 ya cae por debajo del umbral en 3 variables).
YEAR_MODERN = 2023
# 2005 = año más temprano (>=2000) donde TODA variable seleccionada supera el
# umbral de cobertura (>=75%). En 2000-2004 la formación bruta de capital queda
# por debajo (0.728-0.747). Ver decisiones_metodologicas.md.
YEAR_EARLY = 2005

# --------------------------------------------------------------------------- #
# Selección FINAL de variables numéricas (tras análisis de cobertura)
# --------------------------------------------------------------------------- #
# Descartadas y por qué:
#   SI.POV.GINI        -> cobertura 0.26-0.40 (encuestas esporádicas): inviable.
#   SE.XPD.TOTL.GD.ZS  -> 0.49-0.55 en años tempranos: incomparable en el tiempo.
#   NY.GNP.PCAP.PP.KD  -> 0.53 en 2005 y redundante con PBI per cápita (r~0.98).
DROPPED = {
    "SI.POV.GINI": "Cobertura muy baja (0.26-0.40); encuestas esporádicas.",
    "SE.XPD.TOTL.GD.ZS": "Cobertura baja e inestable en años tempranos (~0.50).",
    "NY.GNP.PCAP.PP.KD": "Cobertura baja en 2005 y redundante con PBI per cápita.",
}
INDICATORS_FINAL = [c for c in INDICATORS if c not in DROPPED]

# --------------------------------------------------------------------------- #
# Umbrales de limpieza
# --------------------------------------------------------------------------- #
# Máximo de faltantes tolerado por VARIABLE (proporción de países) en un año
# antes de descartar la variable de la selección final.
MAX_MISSING_VAR = 0.25
# Máximo de faltantes tolerado por PAÍS (proporción de variables) antes de
# descartar el país en ese año.
MAX_MISSING_COUNTRY = 0.40

# Vecinos para imputación KNN (sobre datos estandarizados, dentro de cada año).
KNN_NEIGHBORS = 5

# --------------------------------------------------------------------------- #
# PCA / clustering
# --------------------------------------------------------------------------- #
PARALLEL_ANALYSIS_NSIM = 1000   # matrices aleatorias para el análisis de Horn
PARALLEL_ANALYSIS_PCTL = 95     # percentil de corte
VAR_CUM_TARGET = 0.80           # referencia de varianza acumulada
K_RANGE = range(2, 9)           # k candidatos para clustering
BOOTSTRAP_NREP = 200            # repeticiones bootstrap para estabilidad (Hennig)

# Plantilla de salidas
FIG_DPI = 150
