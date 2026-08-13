import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

try:
    import pyodbc
except ImportError:
    pyodbc = None

import db_utils

CONNECTION_STRING_TEMPLATE = (
    "DRIVER={{{driver}}};"
    "SERVER=RAI-FREITAS\\SQLEXPRESS;"
    "DATABASE=MLB_Predictive;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

DRIVERS_PREFERIDOS = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 13 for SQL Server",
    "SQL Server Native Client 11.0",
]

VENTANA_CARRERAS = 5
VENTANA_VICTORIAS = 10
VENTANA_PITCHER = 5
VENTANA_PITCHER_ULTIMAS3 = 3
VENTANA_PITCHER_TEMPORADA = 20
VENTANA_BULLPEN = 5
VENTANA_FATIGA_3 = 3
VENTANA_FATIGA_5 = 5
LIMITE_OVER = 8.5
MODELOS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
TRANSFORMADORES_PATH = os.path.join(MODELOS_DIR, "transformadores_totales.pkl")

PARK_FACTORES = {
    "Colorado Rockies": 1.31,
    "Cincinnati Reds": 1.15,
    "Boston Red Sox": 1.08,
    "Philadelphia Phillies": 1.08,
    "Houston Astros": 1.06,
    "Arizona Diamondbacks": 1.05,
    "Baltimore Orioles": 1.05,
    "New York Yankees": 1.05,
    "Washington Nationals": 1.04,
    "Chicago Cubs": 1.03,
    "Toronto Blue Jays": 1.03,
    "Texas Rangers": 1.02,
    "Milwaukee Brewers": 1.02,
    "Chicago White Sox": 1.01,
    "Atlanta Braves": 1.01,
    "Minnesota Twins": 1.00,
    "Los Angeles Angels": 1.00,
    "Detroit Tigers": 0.99,
    "San Francisco Giants": 0.98,
    "Tampa Bay Rays": 0.98,
    "Los Angeles Dodgers": 0.97,
    "Kansas City Royals": 0.97,
    "Miami Marlins": 0.97,
    "Pittsburgh Pirates": 0.96,
    "New York Mets": 0.95,
    "St. Louis Cardinals": 0.95,
    "San Diego Padres": 0.95,
    "Oakland Athletics": 0.95,
    "Seattle Mariners": 0.91,
    "Cleveland Guardians": 0.99,
}

COLUMNAS_CRITICAS = ["Fecha", "Estadio", "EquipoLocal", "EquipoVisita",
                     "PitcherLocalId", "PitcherVisitaId", "CarrerasLocal",
                     "CarrerasVisita", "TemperaturaC", "Viento_Velocidad",
                     "Factor_Carreras", "Park_Factor", "ERA_Bullpen_Local",
                     "ERA_Bullpen_Visita", "WHIP_Abridor_Local",
                     "WHIP_Abridor_Visita", "Target_Over",
                     "UmpireHomePlate", "ManoAbridorLocal", "ManoAbridorVisita",
                     "OPSvsLHP_Local", "OPSvsRHP_Local",
                     "OPSvsLHP_Visita", "OPSvsRHP_Visita",
                     "Fatiga_Bullpen_3d_Local", "Fatiga_Bullpen_3d_Visita"]

FEATURES = [
    "Estadio",
    "EquipoLocal",
    "EquipoVisita",
    "TemperaturaC",
    "RachaCarrerasAnotadasLocal",
    "RachaCarrerasAnotadasVisita",
    "RachaCarrerasPermitidasLocal",
    "RachaCarrerasPermitidasVisita",
    "WinRateRecienteLocal",
    "WinRateRecienteVisita",
    "DiffCarreras10_Local",
    "DiffCarreras10_Visita",
    "Juegos_Ultimos_3_Dias_Local",
    "Juegos_Ultimos_3_Dias_Visita",
    "Juegos_Ultimos_5_Dias_Local",
    "Juegos_Ultimos_5_Dias_Visita",
    "ERA_Aproximada_Local",
    "ERA_Aproximada_Visita",
    "ERA_Bullpen_Local",
    "ERA_Bullpen_Visita",
    "Viento_Velocidad",
    "Viento_Direccion",
    "Park_Factor",
    "Factor_Carreras",
    "Fatiga_Bullpen_3d_Local",
    "Fatiga_Bullpen_3d_Visita",
]

PARAM_GRID = {
    "n_estimators": [300, 500, 600],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_child_weight": [5, 10, 15],
    "gamma": [0, 0.1, 0.5],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.5, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
    "scale_pos_weight": [0.6, 0.75, 0.9],
}

# Config regularizada validada con reporte_rendimiento.py (holdout 2026):
# depth 3 / lr 0.03 / 600 est / colsample 0.6 / mcw 5: AUC OOT 0.557 y ROI
# +26% en apuestas de margen alto, contra 0.554 y +10.7% de la anterior.
N_ITER_BUSQUEDA = 20
SPLITS_TEMPORALES = 3


def obtener_driver_odbc():
    if pyodbc is None:
        raise RuntimeError("pyodbc no instalado; usa SQLite (MLB_SQLITE=1).")
    disponibles = pyodbc.drivers()
    for preferido in DRIVERS_PREFERIDOS:
        if preferido in disponibles:
            return preferido
    if disponibles:
        return disponibles[0]
    raise RuntimeError("No se encontro un driver ODBC de SQL Server instalado.")


def cargar_datos(solo_con_temperatura=True):
    base = ("SELECT g.*, PF.Factor_Carreras, "
            "op1.OPSvsLHP AS OPSvsLHP_Local, op1.OPSvsRHP AS OPSvsRHP_Local, "
            "op2.OPSvsLHP AS OPSvsLHP_Visita, op2.OPSvsRHP AS OPSvsRHP_Visita, "
            "pm1.Mano AS ManoAbridorLocal, pm2.Mano AS ManoAbridorVisita, "
            "fb1.Fatiga_Bullpen_3d AS Fatiga_Bullpen_3d_Local, "
            "fb2.Fatiga_Bullpen_3d AS Fatiga_Bullpen_3d_Visita "
            "FROM dbo.GameLog g "
            "LEFT JOIN dbo.ParkFactors PF ON g.EquipoLocal = PF.EquipoLocal "
            "LEFT JOIN dbo.TeamOPS_Handedness op1 "
            "ON op1.Equipo = g.EquipoLocal AND op1.Temporada = YEAR(g.Fecha) "
            "LEFT JOIN dbo.TeamOPS_Handedness op2 "
            "ON op2.Equipo = g.EquipoVisita AND op2.Temporada = YEAR(g.Fecha) "
            "LEFT JOIN dbo.PitcherMano pm1 ON pm1.PitcherId = g.PitcherLocalId "
            "LEFT JOIN dbo.PitcherMano pm2 ON pm2.PitcherId = g.PitcherVisitaId "
            "LEFT JOIN dbo.vwFatigaBullpen3d fb1 "
            "ON fb1.Team = g.EquipoLocal AND fb1.Fecha = g.Fecha "
            "LEFT JOIN dbo.vwFatigaBullpen3d fb2 "
            "ON fb2.Team = g.EquipoVisita AND fb2.Fecha = g.Fecha")
    consulta = base + (" WHERE g.TemperaturaC IS NOT NULL "
                       "AND (g.CarrerasLocal <> 0 OR g.CarrerasVisita <> 0)")
    if not solo_con_temperatura:
        consulta = base
    return db_utils.leer_sql(consulta)


def preprocesar(df):
    df = df.copy()

    columnas_entradas = [c for c in df.columns
                         if c.lower() in ("entradas", "entradasjugadas",
                                          "innings", "inningsjugados")]
    if columnas_entradas:
        antes = len(df)
        df = df[df[columnas_entradas[0]] <= 9]
        if antes > len(df):
            print(f"[PREPROCESO] Excluidos {antes - len(df)} partidos "
                  f"de extra innings (>9 entradas) para aprender solo "
                  f"sobre los 9 innings reglamentarios.")

    df["Target_Over"] = (df["CarrerasLocal"] + df["CarrerasVisita"] > LIMITE_OVER).astype(int)

    df["Park_Factor"] = df["EquipoLocal"].map(PARK_FACTORES).fillna(1.00)

    for columna in ["EquipoLocal", "EquipoVisita", "Estadio"]:
        codificador = LabelEncoder()
        df[columna] = codificador.fit_transform(df[columna])

    df["TemperaturaC"] = pd.to_numeric(df["TemperaturaC"], errors="coerce")
    df["TemperaturaC"] = df["TemperaturaC"].fillna(df["TemperaturaC"].median())
    df["Viento_Velocidad"] = pd.to_numeric(df["Viento_Velocidad"], errors="coerce")
    df["Viento_Velocidad"] = df["Viento_Velocidad"].fillna(df["Viento_Velocidad"].median())
    df["Factor_Carreras"] = df["Factor_Carreras"].fillna(df["Factor_Carreras"].median())
    df["ERA_Bullpen_Local"] = df["ERA_Bullpen_Local"].fillna(4.00)
    df["ERA_Bullpen_Visita"] = df["ERA_Bullpen_Visita"].fillna(4.00)
    df["PitcherLocalId"] = df["PitcherLocalId"].fillna(0)
    df["PitcherVisitaId"] = df["PitcherVisitaId"].fillna(0)

    # Fatiga de bullpen de 72 horas: nulos (inicio de temporada, partidos sin
    # boxscore aun ingerido o juegos de hoy sin PitcherGameLog) -> 0 fatiga.
    for columna_fatiga in ("Fatiga_Bullpen_3d_Local", "Fatiga_Bullpen_3d_Visita"):
        if columna_fatiga not in df.columns:
            df[columna_fatiga] = float("nan")
        df[columna_fatiga] = pd.to_numeric(
            df[columna_fatiga], errors="coerce").fillna(0).clip(lower=0)

    df = pd.get_dummies(df, columns=["Viento_Direccion"], dummy_na=True, drop_first=False)

    df = df.sort_values("Fecha").reset_index(drop=True)

    columnas_viento = [c for c in df.columns if c.startswith("Viento_Direccion_")]
    return df[COLUMNAS_CRITICAS + columnas_viento]


def racha(serie, ventana):
    return serie.shift(1).rolling(ventana, min_periods=1).mean()


def feature_engineering_rachas(df):
    df = df.copy()
    df["Partido"] = df.index

    partes = []
    for lado in ["Local", "Visita"]:
        parte = pd.DataFrame(
            {
                "Partido": df.index,
                "Fecha": df["Fecha"],
                "Equipo": df[f"Equipo{lado}"],
                "CarrerasAnotadas": df[f"Carreras{lado}"],
                "CarrerasPermitidas": df["CarrerasVisita" if lado == "Local" else "CarrerasLocal"],
                "Lado": lado,
            }
        )
        parte["Gano"] = (parte["CarrerasAnotadas"] > parte["CarrerasPermitidas"]).astype(int)
        partes.append(parte)

    apariciones = pd.concat(partes, ignore_index=True)
    apariciones = apariciones.sort_values(["Fecha", "Partido"])

    grupo = apariciones.groupby("Equipo", sort=False)
    apariciones["RachaCarrerasAnotadas"] = grupo["CarrerasAnotadas"].transform(
        lambda s: racha(s, VENTANA_CARRERAS))
    apariciones["RachaCarrerasPermitidas"] = grupo["CarrerasPermitidas"].transform(
        lambda s: racha(s, VENTANA_CARRERAS))
    apariciones["WinRateReciente"] = grupo["Gano"].transform(
        lambda s: racha(s, VENTANA_VICTORIAS))
    apariciones["DiffCarreras10"] = (
        grupo["CarrerasAnotadas"].transform(lambda s: racha(s, VENTANA_VICTORIAS))
        - grupo["CarrerasPermitidas"].transform(lambda s: racha(s, VENTANA_VICTORIAS))
    )

    for lado in ["Local", "Visita"]:
        registro = apariciones.loc[
            apariciones["Lado"] == lado,
            ["Partido", "RachaCarrerasAnotadas", "RachaCarrerasPermitidas",
             "WinRateReciente", "DiffCarreras10"],
        ]
        registro = registro.rename(
            columns={
                "RachaCarrerasAnotadas": f"RachaCarrerasAnotadas{lado}",
                "RachaCarrerasPermitidas": f"RachaCarrerasPermitidas{lado}",
                "WinRateReciente": f"WinRateReciente{lado}",
                "DiffCarreras10": f"DiffCarreras10_{lado}",
            }
        )
        df = df.merge(registro, on="Partido", how="left")

    return df


def feature_engineering_pitchers(df, mediana_carreras):
    df = df.copy()
    df["Partido"] = df.index

    partes = []
    for lado in ["Local", "Visita"]:
        parte = pd.DataFrame(
            {
                "Partido": df.index,
                "Fecha": df["Fecha"],
                "Pitcher": df[f"Pitcher{lado}Id"],
                "CarrerasPermitidas": df["CarrerasVisita" if lado == "Local" else "CarrerasLocal"],
                "Lado": lado,
            }
        )
        partes.append(parte)

    apariciones = pd.concat(partes, ignore_index=True)
    apariciones = apariciones[apariciones["Pitcher"] > 0]
    apariciones = apariciones.sort_values(["Fecha", "Partido"])

    grupo = apariciones.groupby("Pitcher", sort=False)
    apariciones["CarrerasPermitidasUltimas5"] = grupo["CarrerasPermitidas"].transform(
        lambda s: racha(s, VENTANA_PITCHER))
    apariciones["CarrerasPermitidasUltimas3"] = grupo["CarrerasPermitidas"].transform(
        lambda s: racha(s, VENTANA_PITCHER_ULTIMAS3))
    apariciones["CarrerasPermitidasTemporada"] = grupo["CarrerasPermitidas"].transform(
        lambda s: racha(s, VENTANA_PITCHER_TEMPORADA))

    for lado in ["Local", "Visita"]:
        registro = apariciones.loc[
            apariciones["Lado"] == lado,
            ["Partido", "CarrerasPermitidasUltimas5",
             "CarrerasPermitidasUltimas3", "CarrerasPermitidasTemporada"],
        ]
        registro = registro.rename(
            columns={
                "CarrerasPermitidasUltimas5": f"ERA_Aproximada_{lado}",
                "CarrerasPermitidasUltimas3": f"ERA_Ultimas3_{lado}",
                "CarrerasPermitidasTemporada": f"ERA_Temporada_{lado}",
            }
        )
        df = df.merge(registro, on="Partido", how="left")
        df[f"ERA_Aproximada_{lado}"] = df[f"ERA_Aproximada_{lado}"].fillna(mediana_carreras)
        df[f"ERA_Ultimas3_{lado}"] = df[f"ERA_Ultimas3_{lado}"].fillna(mediana_carreras)
        df[f"ERA_Temporada_{lado}"] = df[f"ERA_Temporada_{lado}"].fillna(mediana_carreras)

    return df


def feature_engineering_bullpen(df):
    """Calcula el rendimiento reciente del bullpen por equipo.

    ERA_Bullpen_Local/Visita es una fotografia de la temporada capturada por el
    ETL en cada partido; promediar las ultimas VENTANA_BULLPEN apariciones
    (aprox. 7-10 dias de calendario) refleja la solidez reciente de los
    relevistas de cada equipo.
    """
    df = df.copy()
    df["Partido"] = df.index

    partes = []
    for lado in ["Local", "Visita"]:
        parte = pd.DataFrame(
            {
                "Partido": df.index,
                "Fecha": df["Fecha"],
                "Equipo": df[f"Equipo{lado}"],
                "EraBullpen": df[f"ERA_Bullpen_{lado}"],
                "Lado": lado,
            }
        )
        partes.append(parte)

    apariciones = pd.concat(partes, ignore_index=True)
    apariciones = apariciones[apariciones["EraBullpen"].notna()]
    apariciones = apariciones.sort_values(["Fecha", "Partido"])

    grupo = apariciones.groupby("Equipo", sort=False)
    apariciones["EraBullpenReciente"] = grupo["EraBullpen"].transform(
        lambda s: s.shift(1).rolling(VENTANA_BULLPEN, min_periods=1).mean())

    for lado in ["Local", "Visita"]:
        registro = apariciones.loc[
            apariciones["Lado"] == lado,
            ["Partido", "EraBullpenReciente"],
        ]
        registro = registro.rename(
            columns={"EraBullpenReciente": f"ERA_Bullpen_Reciente_{lado}"}
        )
        df = df.merge(registro, on="Partido", how="left")

    mediana_bullpen = df["ERA_Bullpen_Local"].median()
    for lado in ["Local", "Visita"]:
        df[f"ERA_Bullpen_Reciente_{lado}"] = (
            df[f"ERA_Bullpen_Reciente_{lado}"].fillna(mediana_bullpen))

    return df


def _contar_juegos_ventana(apariciones, dias):
    """Cuenta por equipo los juegos en [fecha - (dias-1), fecha] (inclusive)."""
    orden = apariciones.sort_values(["Equipo", "Fecha", "Partido"])
    equipos = orden["Equipo"].to_numpy()
    fechas = orden["Fecha"].to_numpy()
    cambios = np.concatenate((
        [0], np.flatnonzero(equipos[1:] != equipos[:-1]) + 1, [len(equipos)]))
    conteo = np.empty(len(orden), dtype=int)
    for ini, fin in zip(cambios[:-1], cambios[1:]):
        fechas_equipo = fechas[ini:fin]
        limite = fechas_equipo - np.timedelta64(dias - 1, "D")
        indices = np.searchsorted(fechas_equipo, limite, side="left")
        posiciones = np.arange(len(fechas_equipo))
        conteo[ini:fin] = posiciones - indices + 1
    return orden.assign(JuegosVentana=conteo)


def feature_engineering_fatiga(df):
    """Indice de Fatiga: juegos jugados por equipo en los ultimos 3 y 5 dias.

    La ventana incluye el partido actual (sin dias de descanso); un equipo
    con Juegos_Ultimos_3_Dias == 3 juega 3 dias consecutivos y su bullpen
    arrastra desgaste.
    """
    df = df.copy()
    df["Partido"] = df.index

    partes = []
    for lado in ["Local", "Visita"]:
        parte = pd.DataFrame(
            {
                "Partido": df.index,
                "Fecha": df["Fecha"],
                "Equipo": df[f"Equipo{lado}"],
                "Lado": lado,
            }
        )
        partes.append(parte)

    apariciones = pd.concat(partes, ignore_index=True)
    apariciones["Fecha"] = pd.to_datetime(apariciones["Fecha"])

    for dias in (VENTANA_FATIGA_3, VENTANA_FATIGA_5):
        conteos = _contar_juegos_ventana(apariciones, dias)
        for lado in ["Local", "Visita"]:
            registro = conteos.loc[
                conteos["Lado"] == lado, ["Partido", "JuegosVentana"]
            ]
            registro = registro.rename(
                columns={"JuegosVentana": f"Juegos_Ultimos_{dias}_Dias_{lado}"}
            )
            df = df.merge(registro, on="Partido", how="left")

    return df


def features_finales(df):
    columnas_dummy_viento = [c for c in df.columns if c.startswith("Viento_Direccion_")]
    return [f for f in FEATURES if f != "Viento_Direccion"] + columnas_dummy_viento


def ajustar_transformadores(df_entrenamiento):
    """Ajusta transformadores usando SOLO datos de entrenamiento (sin fuga).

    - Amp eys: Target Encoding del porcentaje de OVERS historico por amposer
      de home, con suavizado hacia el promedio global de la liga.
    - Frecuencia LHP para matchup (neutral para manos desconocidas).
    - Medianas de imputacion para los OPS vs LHP/RHP.
    """
    transformadores = {}
    transformadores["umpire_prior"] = float(df_entrenamiento["Target_Over"].mean())

    mapa = {}
    if "UmpireHomePlate" in df_entrenamiento.columns:
        grupo = df_entrenamiento.groupby("UmpireHomePlate")["Target_Over"].agg(
            ["sum", "count"])
        alfa = 15.0
        for amp, fila in grupo.iterrows():
            mapa[amp] = float(
                (fila["sum"] + alfa * transformadores["umpire_prior"])
                / (fila["count"] + alfa))
    transformadores["umpire_mapa"] = mapa

    manos = pd.concat([
        df_entrenamiento.get("ManoAbridorLocal"),
        df_entrenamiento.get("ManoAbridorVisita")], ignore_index=True)
    manos = manos.dropna().astype(str).str.upper()
    transformadores["frecuencia_lhp"] = (
        float((manos == "L").mean()) if len(manos) else 0.5)

    transformadores["imputaciones"] = {}
    for lado in ("Local", "Visita"):
        for mano in ("LHP", "RHP"):
            col = f"OPSvs{mano}_{lado}"
            if col in df_entrenamiento.columns:
                serie = pd.to_numeric(
                    df_entrenamiento[col], errors="coerce")
                if serie.notna().any():
                    transformadores["imputaciones"][col] = float(serie.median())
                    continue
            transformadores["imputaciones"][col] = 0.0
    return transformadores


def construir_caracteristicas_finales(df, transformadores):
    """Construye el DataFrame FINAL con exactamente las columnas del modelo.

    - Viento_Velocidad numerica y Viento_Direccion en one-hot (con nulos).
    - Matchup LHP/RHP: OPS del equipo rival contra la mano del abridor y el
      indicador EsLHP del abridor (mano desconocida -> frecuencia de la liga).
    - Amp eys de Home: Target Encoding (OVER% historico) con promedio global
      de la liga para ampayers desconocidos en inferencia.
    - reindex a las columnas guardadas (los datos faltantes se rellenan).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=transformadores.get("columnas") or [])
    indice = df.index
    trabajo = df.copy()
    if not trabajo.columns.is_unique:
        trabajo = trabajo.loc[:, ~trabajo.columns.duplicated()]

    if "Viento_Direccion" in trabajo.columns:
        columnas_viento_previas = [c for c in trabajo.columns
                                   if c.startswith("Viento_Direccion_")]
        if columnas_viento_previas:
            trabajo = trabajo.drop(columns=columnas_viento_previas)
        dummies = pd.get_dummies(
            trabajo["Viento_Direccion"].astype("object"),
            prefix="Viento_Direccion", dummy_na=True, drop_first=False)
        trabajo = pd.concat([trabajo, dummies], axis=1)

    if "Viento_Velocidad" in trabajo.columns:
        trabajo["Viento_Velocidad"] = pd.to_numeric(
            trabajo["Viento_Velocidad"], errors="coerce")

    frecuencia_lhp = transformadores.get("frecuencia_lhp", 0.5)
    imputaciones = transformadores.get("imputaciones", {})

    def serie_ops(prefix):
        if prefix in trabajo.columns:
            return pd.to_numeric(trabajo[prefix], errors="coerce")
        return pd.Series(float("nan"), index=indice)

    for lado in ("Local", "Visita"):
        ops_lhp = serie_ops(f"OPSvsLHP_{lado}").fillna(
            imputaciones.get(f"OPSvsLHP_{lado}", 0.0))
        ops_rhp = serie_ops(f"OPSvsRHP_{lado}").fillna(
            imputaciones.get(f"OPSvsRHP_{lado}", 0.0))
        if f"ManoAbridor{lado}" in trabajo.columns:
            manos = trabajo[f"ManoAbridor{lado}"].astype(str).str.upper()
            es_lhp = manos.eq("L").astype(float)
            es_rhp = manos.eq("R").astype(float)
            conocido = manos.str.strip().isin(["L", "R"])
            ops_contra = np.where(
                es_lhp == 1.0, ops_lhp,
                np.where(es_rhp == 1.0, ops_rhp, (ops_lhp + ops_rhp) / 2.0))
            es_lhp = np.where(conocido, es_lhp, frecuencia_lhp)
        else:
            ops_contra = (ops_lhp + ops_rhp) / 2.0
            es_lhp = frecuencia_lhp
        trabajo[f"EsLHP_Abridor{lado}"] = es_lhp
        trabajo[f"OPS_Contra_Mano{lado}"] = ops_contra

    prior = transformadores.get("umpire_prior", 0.5)
    mapa_ump = transformadores.get("umpire_mapa", {})
    if "UmpireHomePlate" in trabajo.columns:
        trabajo["Umpire_Over_Tasa"] = trabajo["UmpireHomePlate"].map(mapa_ump)
    else:
        trabajo["Umpire_Over_Tasa"] = float("nan")
    trabajo["Umpire_Over_Tasa"] = trabajo["Umpire_Over_Tasa"].fillna(prior)

    base = trabajo.reindex(columns=[f for f in FEATURES if f != "Viento_Direccion"])
    viento = trabajo[[c for c in trabajo.columns
                      if c.startswith("Viento_Direccion_")]]
    derivadas = pd.DataFrame(index=indice)
    columnas_derivadas = (
        [f"EsLHP_Abridor{lado}" for lado in ("Local", "Visita")]
        + [f"OPS_Contra_Mano{lado}" for lado in ("Local", "Visita")]
        + ["Umpire_Over_Tasa"])
    for c in columnas_derivadas:
        derivadas[c] = trabajo[c] if c in trabajo.columns else float("nan")

    final = pd.concat([base, viento, derivadas], axis=1)
    if "columnas" in transformadores:
        final = final.reindex(columns=transformadores["columnas"],
                              fill_value=0.0)
    else:
        transformadores["columnas"] = final.columns.tolist()
    return final


def division_cronologica(df, proporcion_entrenamiento=0.8):
    indice_corte = int(len(df) * proporcion_entrenamiento)

    X = df[features_finales(df)]
    y = df["Target_Over"]

    X_entrenamiento = X.iloc[:indice_corte]
    X_prueba = X.iloc[indice_corte:]
    y_entrenamiento = y.iloc[:indice_corte]
    y_prueba = y.iloc[indice_corte:]

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, df


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/7] Cargando datos desde SQL Server (MLB_Predictive)...")
    df = cargar_datos()
    print(f"      {len(df)} registros cargados de dbo.GameLog (solo con temperatura).")

    print("[2/7] Preprocesamiento y creacion del target...")
    df = preprocesar(df)
    print(f"      {len(df)} partidos validos (los empates son validos para OVER/UNDER).")
    print(f"      Target Target_Over = (CarrerasLocal + CarrerasVisita > {LIMITE_OVER}) creado.")

    if len(df) < 50:
        print("      [AVISO] Poca data disponible; el modelo tendra poca confiabilidad.")

    print("[3/7] Feature engineering: rachas, fatiga y bullpen...")
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    mediana_carreras = df["CarrerasVisita"].median()
    df = feature_engineering_pitchers(df, mediana_carreras)
    df = feature_engineering_bullpen(df)
    print(f"      Rachas ({VENTANA_CARRERAS} juegos), win rate ({VENTANA_VICTORIAS}), "
          f"fatiga ({VENTANA_FATIGA_3}/{VENTANA_FATIGA_5} dias), ERAs de abridores "
          f"y bullpen reciente generados.")

    print("[4/7] Division cronologica 80/20 (sin mezclar el tiempo)...")
    indice_corte = int(len(df) * 0.8)
    df_entrenamiento = df.iloc[:indice_corte].copy()
    df_prueba = df.iloc[indice_corte:].copy()
    y_entrenamiento = df_entrenamiento["Target_Over"]
    y_prueba = df_prueba["Target_Over"]
    fecha_corte = df_entrenamiento.iloc[-1]["Fecha"]
    print(f"      Entrenamiento: {len(df_entrenamiento)} partidos (hasta {fecha_corte}).")
    print(f"      Prueba: {len(df_prueba)} partidos.")

    print("[5/7] Ajuste de transformadores sobre entrenamiento (sin fuga)...")
    transformadores = ajustar_transformadores(df_entrenamiento)
    X_entrenamiento = construir_caracteristicas_finales(
        df_entrenamiento, transformadores)
    transformadores["columnas"] = X_entrenamiento.columns.tolist()
    X_prueba = construir_caracteristicas_finales(df_prueba, transformadores)
    n_viento = len([c for c in X_entrenamiento.columns
                    if c.startswith("Viento_Direccion_") or c == "Viento_Velocidad"])
    n_amp = 1
    n_matchup = len([c for c in X_entrenamiento.columns
                     if c.startswith("EsLHP_") or c.startswith("OPS_Contra_Mano")])
    print(f"      Features finales: {len(X_entrenamiento.columns)} columnas "
          f"(viento: {n_viento}, matchup LHP/RHP: {n_matchup}, ampayer: {n_amp}).")
    print(f"      Ampayers distintos en entrenamiento: "
          f"{len(transformadores['umpire_mapa'])} "
          f"(prior global de OVER: {transformadores['umpire_prior']:.4f}).")

    print("[6/7] Busqueda de hiperparametros con RandomizedSearchCV + TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=SPLITS_TEMPORALES)
    xgb_base = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_estimators=600, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5)
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER_BUSQUEDA,
        scoring="neg_log_loss",
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_entrenamiento, y_entrenamiento)
    print(f"      Mejores parametros encontrados (RandomizedSearchCV): "
          f"{random_search.best_params_}")
    print(f"      Mejor neg_log_loss en validacion temporal: "
          f"{random_search.best_score_:.4f}")

    print("[7/7] Evaluacion del mejor modelo en el conjunto de prueba...")
    mejor_modelo = random_search.best_estimator_
    predicciones = mejor_modelo.predict(X_prueba)
    probabilidades = mejor_modelo.predict_proba(X_prueba)[:, 1]

    proporcion_over = y_entrenamiento.mean()
    tasa_base = max(proporcion_over, 1 - proporcion_over)
    exactitud = accuracy_score(y_prueba, predicciones)
    auc_roc = roc_auc_score(y_prueba, probabilidades)

    print(f"      Proporcion de OVER en entrenamiento: {proporcion_over:.4f}")
    print(f"      Baseline (siempre la clase mayoritaria): {tasa_base:.4f}")
    print(f"      Accuracy del mejor modelo en test: {exactitud:.4f}")
    print(f"      ROC AUC del mejor modelo en test: {auc_roc:.4f}")

    importancias = mejor_modelo.feature_importances_
    nombres = list(mejor_modelo.feature_names_in_)
    orden = np.argsort(importancias)[::-1][:15]
    print(f"\n=== IMPORTANCIA DE CARACTERISTICAS (Top {len(orden)}) ===")
    for i in orden:
        print(f"   {nombres[i]:<36}{importancias[i]:.4f}")

    suma_viento = sum(importancias[i] for i, n in enumerate(nombres)
                      if n == "Viento_Velocidad" or n.startswith("Viento_Direccion_"))
    suma_amp = sum(importancias[i] for i, n in enumerate(nombres)
                   if n == "Umpire_Over_Tasa")
    suma_matchup = sum(importancias[i] for i, n in enumerate(nombres)
                       if n.startswith("EsLHP_") or n.startswith("OPS_Contra_Mano"))
    suma_fatiga = sum(importancias[i] for i, n in enumerate(nombres)
                      if n.startswith("Fatiga_Bullpen_3d"))
    print("=== EVIDENCIA: variables nuevas con peso real en el modelo ===")
    print(f"   Viento (velocidad + direccion one-hot): {suma_viento:.4f}")
    print(f"   Ampayer de Home (target encoding):      {suma_amp:.4f}")
    print(f"   Matchup LHP/RHP vs OPS:                  {suma_matchup:.4f}")
    print(f"   Fatiga de Bullpen 72h (3d):              {suma_fatiga:.4f}")
    print("   (valores > 0 confirman que llegan al XGBoost)")
    print("==================================================")

    # Calibracion isotonica sobre la prueba cronologica: corrige el
    # sobreconfiado (P=0.70 estimada -> ~0.58 real segun reporte OOT 2026).
    # Se guarda junto al modelo para aplicarla en produccion.
    calibrador = IsotonicRegression(out_of_bounds="clip")
    calibrador.fit(probabilidades, y_prueba)
    joblib.dump(calibrador, os.path.join(MODELOS_DIR, "calibracion_totales.pkl"))
    p_cal = calibrador.predict(probabilidades)
    print(f"      Calibracion isotonica ajustada sobre {len(probabilidades)} "
          f"partidos de prueba.")
    print(f"      AUC post-calibracion (no cambia): {roc_auc_score(y_prueba, p_cal):.4f}")

    joblib.dump(mejor_modelo, os.path.join(MODELOS_DIR, "modelo_mlb_totales.pkl"))
    joblib.dump(X_entrenamiento.columns.tolist(), os.path.join(MODELOS_DIR, "columnas_totales.pkl"))
    joblib.dump(transformadores, TRANSFORMADORES_PATH)
    print("Modelo, columnas y transformadores guardados con exito.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
