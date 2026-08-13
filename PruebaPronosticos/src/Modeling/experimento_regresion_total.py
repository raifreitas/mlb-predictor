# -*- coding: utf-8 -*-
"""Experimento: pronosticar el TOTAL de carreras con TODAS las variables
(mismo pipeline de features del clasificador, 393 columnas) y usar esa
proyeccion vs linea para decidir OVER/UNDER.

Evalua la viabilidad real de la idea "saco un numero de carreras y muevo
la linea segun mi proyeccion":
  - MAE/RMSE/R2 out-of-time (2026): si RMSE >= desviacion de la liga
    (~3.6), la proyeccion NO tiene senal util para lineas.
  - Simulacion de apuesta por margen |mu - linea| a cuota 1.91.
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score)

from entrenar_modelo import (
    ajustar_transformadores,
    cargar_datos,
    construir_caracteristicas_finales,
    feature_engineering_bullpen,
    feature_engineering_fatiga,
    feature_engineering_pitchers,
    feature_engineering_rachas,
    preprocesar,
)
from predecir_hoy import (
    feature_engineering_ampayer,
    feature_engineering_descanso_abridor,
    feature_engineering_matchup,
)

LINEA = 8.5
CUOTA = 1.91


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df_raw = cargar_datos()
    df = preprocesar(df_raw)
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    df = feature_engineering_pitchers(df, df["CarrerasVisita"].median())
    df = feature_engineering_bullpen(df)
    df = feature_engineering_ampayer(df)
    df = feature_engineering_descanso_abridor(df)
    df = feature_engineering_matchup(df)
    df["Total_Carreras"] = df["CarrerasLocal"] + df["CarrerasVisita"]
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    ent = df[df["Fecha"].dt.year < 2026]
    prb = df[df["Fecha"].dt.year >= 2026]
    print(f"Entrenamiento: {len(ent)} | Prueba 2026: {len(prb)}")

    trans = ajustar_transformadores(ent)
    X_ent = construir_caracteristicas_finales(ent, trans)
    X_prb = construir_caracteristicas_finales(prb, trans)
    y_ent = ent["Total_Carreras"].values
    y_prb = prb["Total_Carreras"].values

    modelo = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
        random_state=42)
    modelo.fit(X_ent, y_ent)

    mae = mean_absolute_error(y_prb, modelo.predict(X_prb))
    rmse = root_mean_squared_error(y_prb, modelo.predict(X_prb))
    r2 = r2_score(y_prb, modelo.predict(X_prb))
    sd_liga = y_prb.std()
    print(f"MAE  2026: {mae:.3f}")
    print(f"RMSE 2026: {rmse:.3f}  (desv. estandar liga: {sd_liga:.3f})")
    print(f"R2   2026: {r2:.3f}")
    if rmse >= sd_liga:
        print("VEREDICTO: RMSE >= ruido de la liga -> sin senal para mover lineas.")
    else:
        print("VEREDICTO: RMSE < ruido de la liga -> hay senal potencial.")

    mu = pd.Series(modelo.predict(X_prb), name="mu")
    print(f"NaN en predicciones: {int(mu.isna().sum())} de {len(mu)}")
    res = mu - LINEA
    real = pd.Series(y_prb, name="total")

    print(f"\nSIMULACION (OVER si mu-linea >= margen, UNDER si <= -margen, "
          f"cuota {CUOTA}):")
    print(f"  {'|margen|':<10} {'N':>5} {'OV':>4} {'UN':>4} "
          f"{'Acierto':>8} {'ROI':>8}")
    for margen in (0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50):
        over = (res >= margen).fillna(False)
        under = (res <= -margen).fillna(False)
        n = int(over.sum()) + int(under.sum())
        if n == 0:
            continue
        g_ov = int((real[over] > LINEA).sum())
        g_un = int((real[under] < LINEA).sum())
        aciertos = g_ov + g_un
        roi = (aciertos * (CUOTA - 1.0) - (n - aciertos)) / n
        print(f"  {margen:<10} {n:>5} {g_ov:>4} {g_un:>4} "
              f"{aciertos / n:>8.3f} {roi:>+8.3f}")

    print("\nDistribucion del error de la proyeccion (2026):")
    err = (mu - real).abs()
    for p in (50, 75, 90, 95):
        print(f"  |error| P{p}: {err.quantile(p / 100):.2f} carreras")
    return 0


if __name__ == "__main__":
    sys.exit(main())