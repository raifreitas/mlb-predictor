"""Entrena un modelo de regresion (XGBoost) para predecir el TOTAL de carreras.

Objetivo (y): CarrerasLocal + CarrerasVisita de dbo.GameLog.
Predictoras (X): clima (TemperaturaC, Viento_Velocidad, Viento_Direccion),
WHIP de ambos abridores, mano de los abridores (LHP/RHP via dbo.PitcherMano)
y nombre del ampayer principal (one-hot encoding).
Division TEMPORAL (no aleatoria): entrenamiento 2023-2025, prueba 2026.

Uso:
    python entrenar_regresion.py
"""

import os
import sys

import joblib
import pandas as pd
import pyodbc
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score)
from xgboost import XGBRegressor

CONNECTION_STRING_TEMPLATE = (
    "DRIVER={{{driver}}};"
    "SERVER=RAI-FREITAS\\SQLEXPRESS;"
    "DATABASE=MLB_Predictive;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

MODELOS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
MODELO_PATH = os.path.join(MODELOS_DIR, "modelo_regresion_totales.pkl")
COLUMNAS_PATH = os.path.join(MODELOS_DIR, "columnas_regresion_totales.pkl")

DRIVERS_PREFERIDOS = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 13 for SQL Server",
    "SQL Server Native Client 11.0",
]

MANO_MAP = {"L": 0, "R": 1}

VENTANA_PITCHER = 5
VENTANA_PITCHER_ULTIMAS3 = 3


def obtener_driver_odbc():
    disponibles = pyodbc.drivers()
    for preferido in DRIVERS_PREFERIDOS:
        if preferido in disponibles:
            return preferido
    if disponibles:
        return disponibles[0]
    raise RuntimeError("No se encontro un driver ODBC de SQL Server instalado.")


def cargar_datos():
    """Extrae GameLog + PitcherMano a un DataFrame (solo partidos finales)."""
    connection_string = CONNECTION_STRING_TEMPLATE.format(driver=obtener_driver_odbc())
    consulta = """
        SELECT g.Fecha,
               g.CarrerasLocal,
               g.CarrerasVisita,
               g.Estadio,
               g.TemperaturaC,
               g.Viento_Velocidad,
               g.Viento_Direccion,
               g.WHIP_Abridor_Local,
               g.WHIP_Abridor_Visita,
               g.ERA_Bullpen_Local,
               g.ERA_Bullpen_Visita,
               g.PitcherLocalId,
               g.PitcherVisitaId,
               g.UmpireNombre,
               PF.Factor_Carreras,
               pm1.Mano AS ManoAbridorLocal,
               pm2.Mano AS ManoAbridorVisita
        FROM dbo.GameLog g
        LEFT JOIN dbo.ParkFactors PF ON PF.EquipoLocal = g.EquipoLocal
        LEFT JOIN dbo.PitcherMano pm1 ON pm1.PitcherId = g.PitcherLocalId
        LEFT JOIN dbo.PitcherMano pm2 ON pm2.PitcherId = g.PitcherVisitaId
        WHERE g.TemperaturaC IS NOT NULL
          AND (g.CarrerasLocal <> 0 OR g.CarrerasVisita <> 0)
    """
    conexion = pyodbc.connect(connection_string)
    try:
        return pd.read_sql(consulta, conexion)
    finally:
        conexion.close()


def feature_engineering_eras_recientes(df):
    """ERAs aproximadas por abridor (ventanas de 5 y 3 salidas), sin lookahead.

    Carreras permitidas = carreras del rival en juegos donde fue titular;
    se promedian SOLO con salidas anteriores (shift(1)), igual que el
    pipeline de predecir_hoy.py.
    """
    df = df.copy()
    df = df.sort_values("Fecha").reset_index(drop=True)
    df["Partido"] = df.index

    def racha(serie, ventana):
        return serie.shift(1).rolling(ventana, min_periods=1).mean()

    partes = []
    for lado in ("Local", "Visita"):
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
    apariciones["CarrerasPermitidas5"] = grupo["CarrerasPermitidas"].transform(
        lambda s: racha(s, VENTANA_PITCHER))
    apariciones["CarrerasPermitidas3"] = grupo["CarrerasPermitidas"].transform(
        lambda s: racha(s, VENTANA_PITCHER_ULTIMAS3))

    mediana_carreras = df["CarrerasVisita"].median()
    for lado in ("Local", "Visita"):
        registro = apariciones.loc[
            apariciones["Lado"] == lado,
            ["Partido", "CarrerasPermitidas5", "CarrerasPermitidas3"],
        ]
        registro = registro.rename(
            columns={
                "CarrerasPermitidas5": f"ERA_Aproximada_{lado}",
                "CarrerasPermitidas3": f"ERA_Ultimas3_{lado}",
            }
        )
        df = df.merge(registro, on="Partido", how="left")
        df[f"ERA_Aproximada_{lado}"] = df[f"ERA_Aproximada_{lado}"].fillna(mediana_carreras)
        df[f"ERA_Ultimas3_{lado}"] = df[f"ERA_Ultimas3_{lado}"].fillna(mediana_carreras)
    return df


def preparar_datos(df):
    """Construye X (features) e y (total de carreras) a partir del DataFrame."""
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)
    df = feature_engineering_eras_recientes(df)

    y = df["CarrerasLocal"] + df["CarrerasVisita"]
    y = y.astype(float)

    df["UmpireNombre"] = df["UmpireNombre"].fillna("Desconocido")
    df["Viento_Direccion"] = df["Viento_Direccion"].fillna("ND")
    df["Estadio"] = df["Estadio"].fillna("Desconocido")

    for columna in ("WHIP_Abridor_Local", "WHIP_Abridor_Visita",
                    "ERA_Bullpen_Local", "ERA_Bullpen_Visita",
                    "TemperaturaC", "Viento_Velocidad"):
        mediana = df[columna].median()
        df[columna] = df[columna].fillna(mediana)
    df["Factor_Carreras"] = df["Factor_Carreras"].fillna(1.0)

    for columna in ("ManoAbridorLocal", "ManoAbridorVisita"):
        df[columna] = df[columna].map(MANO_MAP).fillna(-1)

    numericas = ["TemperaturaC", "Viento_Velocidad",
                 "WHIP_Abridor_Local", "WHIP_Abridor_Visita",
                 "ERA_Bullpen_Local", "ERA_Bullpen_Visita",
                 "ERA_Aproximada_Local", "ERA_Aproximada_Visita",
                 "ERA_Ultimas3_Local", "ERA_Ultimas3_Visita",
                 "Factor_Carreras",
                 "ManoAbridorLocal", "ManoAbridorVisita"]
    categoricas = pd.get_dummies(
        df[["Viento_Direccion", "UmpireNombre", "Estadio"]],
        columns=["Viento_Direccion", "UmpireNombre", "Estadio"],
        dtype=float)

    X = pd.concat([df[numericas].astype(float), categoricas], axis=1)
    return X, y, df["Fecha"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/4] Extrayendo datos de SQL Server...")
    df = cargar_datos()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    print(f"      {len(df)} partidos con clima y marcador final.")
    print("      Distribucion por temporada:")
    print(df.groupby(df["Fecha"].dt.year).size().to_string())

    print("[2/4] Construyendo features y objetivo...")
    X, y, fechas = preparar_datos(df)

    # Division TEMPORAL: entrenamiento 2023-2025, prueba 2026.
    umbral = pd.Timestamp("2026-01-01")
    es_entrenamiento = fechas < umbral
    es_prueba = fechas >= umbral
    print(f"      Entrenamiento: {int(es_entrenamiento.sum())} partidos "
          f"({fechas[es_entrenamiento].min().date()} -> "
          f"{fechas[es_entrenamiento].max().date()})")
    print(f"      Prueba: {int(es_prueba.sum())} partidos "
          f"({fechas[es_prueba].min().date()} -> "
          f"{fechas[es_prueba].max().date()})")

    X_entrenamiento = X.loc[es_entrenamiento]
    y_entrenamiento = y.loc[es_entrenamiento]
    X_prueba = X.loc[es_prueba]
    y_prueba = y.loc[es_prueba]

    # One-hot alineado: columnas de prueba rellenadas contra las de entrenamiento.
    X_prueba = X_prueba.reindex(columns=X_entrenamiento.columns, fill_value=0.0)

    # Validacion temporal interna: ultimo 15% del entrenamiento (en orden cronologico).
    n_val = max(1, int(len(X_entrenamiento) * 0.15))
    X_val = X_entrenamiento.iloc[-n_val:]
    y_val = y_entrenamiento.iloc[-n_val:]
    X_entrenamiento_fit = X_entrenamiento.iloc[:-n_val]
    y_entrenamiento_fit = y_entrenamiento.iloc[:-n_val]

    print("[3/4] Entrenando XGBoost (early stopping con validacion temporal)...")
    modelo = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(
        X_entrenamiento_fit, y_entrenamiento_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    mejor_iteracion = int(modelo.best_iteration) if modelo.best_iteration else "n/d"
    print(f"      Mejor iteracion (early stopping): {mejor_iteracion}")

    print("[4/4] Evaluando sobre 2026 (prueba)...")
    predicciones = modelo.predict(X_prueba)

    mae = mean_absolute_error(y_prueba, predicciones)
    rmse = root_mean_squared_error(y_prueba, predicciones)
    r2 = r2_score(y_prueba, predicciones)
    print(f"      MAE  (2026): {mae:.3f} carreras")
    print(f"      RMSE (2026): {rmse:.3f} carreras")
    print(f"      R2   (2026): {r2:.3f}")

    joblib.dump(modelo, MODELO_PATH)
    joblib.dump(list(X_entrenamiento.columns), COLUMNAS_PATH)
    print(f"      Modelo guardado en {MODELO_PATH}")
    print(f"      Columnas guardadas en {COLUMNAS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
