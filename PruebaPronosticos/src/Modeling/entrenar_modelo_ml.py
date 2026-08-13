import os
import sys

import joblib
import numpy as np
import pandas as pd
import pyodbc
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from entrenar_modelo import (
    CONNECTION_STRING_TEMPLATE,
    PARK_FACTORES,
    VENTANA_CARRERAS,
    VENTANA_VICTORIAS,
    VENTANA_PITCHER,
    VENTANA_PITCHER_ULTIMAS3,
    VENTANA_PITCHER_TEMPORADA,
    VENTANA_BULLPEN,
    VENTANA_FATIGA_3,
    VENTANA_FATIGA_5,
    _contar_juegos_ventana,
    obtener_driver_odbc,
    racha,
)

MODELOS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
MODELO_PATH = os.path.join(MODELOS_DIR, "modelo_mlb_moneyline.pkl")
COLUMNAS_PATH = os.path.join(MODELOS_DIR, "columnas_moneyline.pkl")
CALIBRACION_PATH = os.path.join(MODELOS_DIR, "calibracion_moneyline.pkl")
TRANSFORMADORES_PATH = os.path.join(MODELOS_DIR, "transformadores_moneyline.pkl")

# Blend de ERA del abridor (misma ponderacion que produccion de totals):
# lo reciente pesa mas, la temporada completa da estabilidad.
PESO_ERA_ULTIMAS3 = 0.35
PESO_ERA_APROXIMADA = 0.25
PESO_ERA_TEMPORADA = 0.40
DESCANSO_ESTANDAR = 5
LIMITE_TEMPERATURA_FRIO_C = 15.0

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
}

N_ITER_BUSQUEDA = 20
SPLITS_TEMPORALES = 3

FEATURES_ML = [
    "Estadio",
    "EquipoLocal",
    "EquipoVisita",
    "TemperaturaC",
    "Aire_Frio",
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
    "ERA_Blend_Local",
    "ERA_Blend_Visita",
    "ERA_Blend_Diff",
    "WHIP_Abridor_Local",
    "WHIP_Abridor_Visita",
    "WHIP_Abridor_Diff",
    "Descanso_Abridor_Local",
    "Descanso_Abridor_Visita",
    "ERA_Bullpen_Local",
    "ERA_Bullpen_Visita",
    "ERA_Bullpen_Reciente_Local",
    "ERA_Bullpen_Reciente_Visita",
    "Fatiga_Bullpen_3d_Local",
    "Fatiga_Bullpen_3d_Visita",
    "Viento_Velocidad",
    "Park_Factor",
    "Factor_Carreras",
]


def cargar_datos():
    connection_string = CONNECTION_STRING_TEMPLATE.format(driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        consulta = ("SELECT g.*, PF.Factor_Carreras, "
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
                    "ON fb2.Team = g.EquipoVisita AND fb2.Fecha = g.Fecha "
                     "WHERE g.CarrerasLocal IS NOT NULL "
                     "AND g.CarrerasVisita IS NOT NULL")
        return pd.read_sql(consulta, conexion)
    finally:
        conexion.close()


def preprocesar(df):
    df = df.copy()
    # Moneyline: el ganador se decide con extra innings incluidos, por lo
    # que NO se excluyen partidos de mas de 9 entradas (a diferencia del
    # modelo de totals, donde la linea aplica solo a los 9 reglamentarios).
    df["Target_GanaLocal"] = (df["CarrerasLocal"] > df["CarrerasVisita"]).astype(int)
    df["Park_Factor"] = df["EquipoLocal"].map(PARK_FACTORES).fillna(1.00)
    df["Factor_Carreras"] = pd.to_numeric(
        df["Factor_Carreras"], errors="coerce").fillna(
        df["Factor_Carreras"].median())

    for columna in ["EquipoLocal", "EquipoVisita", "Estadio"]:
        codificador = LabelEncoder()
        df[columna] = codificador.fit_transform(df[columna])

    df["TemperaturaC"] = pd.to_numeric(df["TemperaturaC"], errors="coerce")
    df["TemperaturaC"] = df["TemperaturaC"].fillna(df["TemperaturaC"].median())
    df["Aire_Frio"] = (df["TemperaturaC"] < LIMITE_TEMPERATURA_FRIO_C).astype(float)
    df["Viento_Velocidad"] = pd.to_numeric(
        df["Viento_Velocidad"], errors="coerce").fillna(
        df["Viento_Velocidad"].median())
    df["WHIP_Abridor_Local"] = pd.to_numeric(
        df["WHIP_Abridor_Local"], errors="coerce").fillna(1.30)
    df["WHIP_Abridor_Visita"] = pd.to_numeric(
        df["WHIP_Abridor_Visita"], errors="coerce").fillna(1.30)
    df["ERA_Bullpen_Local"] = pd.to_numeric(
        df["ERA_Bullpen_Local"], errors="coerce").fillna(4.00)
    df["ERA_Bullpen_Visita"] = pd.to_numeric(
        df["ERA_Bullpen_Visita"], errors="coerce").fillna(4.00)
    df["PitcherLocalId"] = df["PitcherLocalId"].fillna(0)
    df["PitcherVisitaId"] = df["PitcherVisitaId"].fillna(0)

    for columna_fatiga in ("Fatiga_Bullpen_3d_Local", "Fatiga_Bullpen_3d_Visita"):
        if columna_fatiga not in df.columns:
            df[columna_fatiga] = float("nan")
        df[columna_fatiga] = pd.to_numeric(
            df[columna_fatiga], errors="coerce").fillna(0).clip(lower=0)

    df = pd.get_dummies(
        df, columns=["Viento_Direccion"], dummy_na=True, drop_first=False)
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df


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


def feature_engineering_fatiga(df):
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


def feature_engineering_descanso_abridor(df):
    df = df.copy()
    for lado in ("Local", "Visita"):
        col_pitcher = f"Pitcher{lado}Id"
        col_descanso = f"Descanso_Abridor_{lado}"
        df[col_descanso] = DESCANSO_ESTANDAR
        if col_pitcher not in df.columns:
            continue
        sub = df.dropna(subset=[col_pitcher]).copy()
        sub["Fecha"] = pd.to_datetime(sub["Fecha"])
        sub = sub.sort_values("Fecha")
        sub["Fecha_Prev"] = sub.groupby(col_pitcher)["Fecha"].shift(1)
        dias = (sub["Fecha"] - sub["Fecha_Prev"]).dt.days
        dias = dias.fillna(DESCANSO_ESTANDAR).clip(lower=0)
        df.loc[sub.index, col_descanso] = dias
    return df


def ajustar_transformadores(df_entrenamiento):
    """Transformadores ajustados SOLO con datos de entrenamiento (sin fuga)."""
    transformadores = {}
    transformadores["ganalocal_prior"] = float(df_entrenamiento["Target_GanaLocal"].mean())

    mapa = {}
    if "UmpireHomePlate" in df_entrenamiento.columns:
        grupo = df_entrenamiento.groupby("UmpireHomePlate")["Target_GanaLocal"].agg(
            ["sum", "count"])
        alfa = 15.0
        prior = transformadores["ganalocal_prior"]
        for amp, fila in grupo.iterrows():
            mapa[amp] = float(
                (fila["sum"] + alfa * prior) / (fila["count"] + alfa))
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
                serie = pd.to_numeric(df_entrenamiento[col], errors="coerce")
                if serie.notna().any():
                    transformadores["imputaciones"][col] = float(serie.median())
                    continue
            transformadores["imputaciones"][col] = 0.0
    return transformadores


def construir_caracteristicas_finales(df, transformadores):
    """DataFrame FINAL con exactamente las columnas del modelo ML."""
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

    prior = transformadores.get("ganalocal_prior", 0.5)
    mapa_ump = transformadores.get("umpire_mapa", {})
    if "UmpireHomePlate" in trabajo.columns:
        trabajo["Umpire_GanaLocal_Tasa"] = trabajo["UmpireHomePlate"].map(mapa_ump)
    else:
        trabajo["Umpire_GanaLocal_Tasa"] = float("nan")
    trabajo["Umpire_GanaLocal_Tasa"] = trabajo["Umpire_GanaLocal_Tasa"].fillna(prior)

    for lado in ("Local", "Visita"):
        era3 = pd.to_numeric(trabajo[f"ERA_Ultimas3_{lado}"], errors="coerce")
        era5 = pd.to_numeric(trabajo[f"ERA_Aproximada_{lado}"], errors="coerce")
        era20 = pd.to_numeric(trabajo[f"ERA_Temporada_{lado}"], errors="coerce")
        trabajo[f"ERA_Blend_{lado}"] = (
            PESO_ERA_ULTIMAS3 * era3
            + PESO_ERA_APROXIMADA * era5
            + PESO_ERA_TEMPORADA * era20)
        trabajo[f"WHIP_Abridor_{lado}"] = pd.to_numeric(
            trabajo[f"WHIP_Abridor_{lado}"], errors="coerce").fillna(1.30)
    trabajo["ERA_Blend_Diff"] = (
        trabajo["ERA_Blend_Local"] - trabajo["ERA_Blend_Visita"])
    trabajo["WHIP_Abridor_Diff"] = (
        trabajo["WHIP_Abridor_Local"] - trabajo["WHIP_Abridor_Visita"])

    base = trabajo.reindex(columns=FEATURES_ML)
    viento = trabajo[[c for c in trabajo.columns
                      if c.startswith("Viento_Direccion_")]]
    derivadas = pd.DataFrame(index=indice)
    columnas_derivadas = (
        [f"EsLHP_Abridor{lado}" for lado in ("Local", "Visita")]
        + [f"OPS_Contra_Mano{lado}" for lado in ("Local", "Visita")]
        + ["Umpire_GanaLocal_Tasa"])
    for c in columnas_derivadas:
        derivadas[c] = trabajo[c] if c in trabajo.columns else float("nan")

    final = pd.concat([base, viento, derivadas], axis=1)
    if "columnas" in transformadores:
        final = final.reindex(columns=transformadores["columnas"], fill_value=0.0)
    else:
        transformadores["columnas"] = final.columns.tolist()
    return final


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/7] Cargando datos desde SQL Server (MLB_Predictive)...")
    df = cargar_datos()
    print(f"      {len(df)} partidos finalizados desde 2023 cargados de dbo.GameLog.")

    print("[2/7] Preprocesamiento y creacion del target...")
    df = preprocesar(df)
    print(f"      Target Target_GanaLocal = (CarrerasLocal > CarrerasVisita) "
          f"creado; proporcion de victorias locales: {df['Target_GanaLocal'].mean():.4f}.")

    if len(df) < 50:
        print("      [AVISO] Poca data disponible; el modelo tendra poca confiabilidad.")

    print("[3/7] Feature engineering: rachas, pitchers, fatiga y bullpen...")
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    mediana_carreras = df["CarrerasVisita"].median()
    df = feature_engineering_pitchers(df, mediana_carreras)
    df = feature_engineering_bullpen(df)
    df = feature_engineering_descanso_abridor(df)
    print(f"      Rachas ({VENTANA_CARRERAS} juegos), win rate ({VENTANA_VICTORIAS}), "
          f"fatiga ({VENTANA_FATIGA_3}/{VENTANA_FATIGA_5} dias), ERAs de abridores "
          f"(blend {int(PESO_ERA_ULTIMAS3*100)}/{int(PESO_ERA_APROXIMADA*100)}/"
          f"{int(PESO_ERA_TEMPORADA*100)}), bullpen y descanso generados.")

    print("[4/7] Division cronologica 80/20 (sin mezclar el tiempo)...")
    indice_corte = int(len(df) * 0.8)
    df_entrenamiento = df.iloc[:indice_corte].copy()
    df_prueba = df.iloc[indice_corte:].copy()
    y_entrenamiento = df_entrenamiento["Target_GanaLocal"]
    y_prueba = df_prueba["Target_GanaLocal"]
    fecha_corte = df_entrenamiento.iloc[-1]["Fecha"]
    print(f"      Entrenamiento: {len(df_entrenamiento)} partidos (hasta {fecha_corte}).")
    print(f"      Prueba: {len(df_prueba)} partidos.")

    print("[5/7] Ajuste de transformadores sobre entrenamiento (sin fuga)...")
    transformadores = ajustar_transformadores(df_entrenamiento)
    X_entrenamiento = construir_caracteristicas_finales(
        df_entrenamiento, transformadores)
    transformadores["columnas"] = X_entrenamiento.columns.tolist()
    X_prueba = construir_caracteristicas_finales(df_prueba, transformadores)
    print(f"      Features finales: {len(X_entrenamiento.columns)} columnas "
          f"(incluye ERA_Blend_Diff y WHIP_Abridor_Diff).")

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
    print(f"      Mejores parametros: {random_search.best_params_}")
    print(f"      Mejor neg_log_loss en validacion temporal: "
          f"{random_search.best_score_:.4f}")

    print("[7/7] Evaluacion del mejor modelo en el conjunto de prueba...")
    mejor_modelo = random_search.best_estimator_
    predicciones = mejor_modelo.predict(X_prueba)
    probabilidades = mejor_modelo.predict_proba(X_prueba)[:, 1]

    proporcion_local = y_entrenamiento.mean()
    tasa_base = max(proporcion_local, 1 - proporcion_local)
    exactitud = accuracy_score(y_prueba, predicciones)
    auc_roc = roc_auc_score(y_prueba, probabilidades)

    print(f"      Proporcion de victorias locales en entrenamiento: {proporcion_local:.4f}")
    print(f"      Baseline (siempre la clase mayoritaria): {tasa_base:.4f}")
    print(f"      Accuracy del mejor modelo en test: {exactitud:.4f}")
    print(f"      ROC AUC del mejor modelo en test: {auc_roc:.4f}")

    importancias = mejor_modelo.feature_importances_
    nombres = list(mejor_modelo.feature_names_in_)
    orden = np.argsort(importancias)[::-1][:15]
    print(f"\n=== IMPORTANCIA DE CARACTERISTICAS (Top {len(orden)}) ===")
    for i in orden:
        print(f"   {nombres[i]:<36}{importancias[i]:.4f}")

    calibrador = IsotonicRegression(out_of_bounds="clip")
    calibrador.fit(probabilidades, y_prueba)
    joblib.dump(calibrador, CALIBRACION_PATH)
    p_cal = calibrador.predict(probabilidades)
    print(f"      Calibracion isotonica ajustada sobre {len(probabilidades)} "
          f"partidos de prueba.")
    print(f"      AUC post-calibracion (no cambia): "
          f"{roc_auc_score(y_prueba, p_cal):.4f}")

    joblib.dump(mejor_modelo, MODELO_PATH)
    joblib.dump(X_entrenamiento.columns.tolist(), COLUMNAS_PATH)
    joblib.dump(transformadores, TRANSFORMADORES_PATH)
    print("Modelo ML, columnas y transformadores guardados con exito.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
