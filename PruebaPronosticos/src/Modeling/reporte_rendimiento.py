# -*- coding: utf-8 -*-
"""Reporte de rendimiento HONESTO del modelo (holdout cronologico).

Mide la capacidad real del clasificador sin depender de lineas de
mercado (aun no hay historico de cuotas). Evalua el modelo contra su
propio objetivo de entrenamiento (P(Total > 8.5)) pero en un periodo
futuro NO visto (2026) y con transformadores ajustados solo con datos
de entrenamiento (sin fuga).

Metrcias:
  - AUC: capacidad de discriminacion (0.50 = moneda al aire).
  - Brier / calibracion: si el modelo dice 60% OVER, ~casi 60% acierta.
  - Tabla por margen de confianza: acierto y ROI simulando apuesta
    OVER/UNDER a cuota 1.91, segun el margen |P(Over) - 0.50|.

Uso:
    python src\\Modeling\\reporte_rendimiento.py
"""

import sys
from datetime import date

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss

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

LIMITE_OVER = 8.5
CUOTA_DECIMAL = 1.91
CORTE = date(2026, 1, 1)
RUTA_REPORTE = None  # output_predicciones/reporte_rendimiento.txt


def construir_df_modelo():
    df_raw = cargar_datos()  # como en entrenamiento (solo con temperatura)
    df = preprocesar(df_raw)
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    df = feature_engineering_pitchers(df, df["CarrerasVisita"].median())
    df = feature_engineering_bullpen(df)
    df = feature_engineering_ampayer(df)
    df = feature_engineering_descanso_abridor(df)
    df = feature_engineering_matchup(df)
    df["Total_Carreras"] = df["CarrerasLocal"] + df["CarrerasVisita"]
    df["Target_Over"] = (df["Total_Carreras"] > LIMITE_OVER).astype(int)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/4] Cargando y construyendo features (pipeline de produccion)...")
    df = construir_df_modelo()
    print(f"      {len(df)} partidos con temperatura y target.")

    entrenamiento = df[df["Fecha"].dt.date < CORTE]
    prueba = df[df["Fecha"].dt.date >= CORTE]
    print(f"      Entrenamiento: {len(entrenamiento)} | "
          f"Prueba (2026): {len(prueba)}")

    print("[2/4] Ajustando transformadores SOLO con entrenamiento...")
    trans = ajustar_transformadores(entrenamiento)
    X_ent = construir_caracteristicas_finales(entrenamiento, trans)
    X_prueba = construir_caracteristicas_finales(prueba, trans)
    y_ent = entrenamiento["Target_Over"].values
    y_prueba = prueba["Target_Over"].values
    total_prueba = prueba["Total_Carreras"].values
    print(f"      Features finales: {X_ent.shape[1]} columnas "
          f"(con columnas fijas en entrenamiento).")

    print("[3/4] Entrenando XGBoost (mismas familias de hiperparametros)...")
    import os
    parametros = {
        "n_estimators": int(os.environ.get("XGB_N", "400")),
        "learning_rate": float(os.environ.get("XGB_LR", "0.05")),
        "max_depth": int(os.environ.get("XGB_DEPTH", "5")),
        "subsample": float(os.environ.get("XGB_SUB", "0.8")),
        "colsample_bytree": float(os.environ.get("XGB_COLS", "0.85")),
        "min_child_weight": int(os.environ.get("XGB_MCW", "1")),
    }
    print(f"      Parametros: {parametros}")
    modelo = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_estimators=parametros["n_estimators"],
        learning_rate=parametros["learning_rate"],
        max_depth=parametros["max_depth"],
        subsample=parametros["subsample"],
        colsample_bytree=parametros["colsample_bytree"],
        min_child_weight=parametros["min_child_weight"])
    modelo.fit(X_ent, y_ent)

    p_prueba = modelo.predict_proba(X_prueba)[:, 1]
    p_ent = modelo.predict_proba(X_ent)[:, 1]

    lineas = ["=" * 78]
    def registro(texto):
        print(texto)
        lineas.append(texto)

    registro("")
    registro("REPORTE DE RENDIMIENTO (holdout cronologico 2026, "
             "objetivo Total > 8.5)")
    registro("=" * 78)
    tasa_base = max(y_ent.mean(), 1 - y_ent.mean())
    registro(f"  OVER rate en entrenamiento   : {y_ent.mean():.4f}")
    registro(f"  Baseline (clase mayoritaria) : {tasa_base:.4f}")
    registro(f"  AUC entrenamiento (in-sample): {roc_auc_score(y_ent, p_ent):.4f}")
    registro(f"  AUC PRUEBA 2026 (out-of-time): {roc_auc_score(y_prueba, p_prueba):.4f}")
    registro(f"  Brier PRUEBA                : {brier_score_loss(y_prueba, p_prueba):.4f} "
             "(0=perfecto, 0.25=azar)")
    registro("")

    registro("CALIBRACION (que tan de fiar es la probabilidad):")
    registro(f"  {'Rango P(Over)':<16} {'N':>6} {'P media':>9} "
             f"{'OVER real':>10}")
    bordes = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    tabla = pd.DataFrame(
        {"p": p_prueba, "y": y_prueba, "total": total_prueba})
    for i in range(len(bordes) - 1):
        mascara = tabla["p"].ge(bordes[i]) & tabla["p"].lt(bordes[i + 1])
        sub = tabla[mascara]
        if len(sub) == 0:
            continue
        registro(f"  {bordes[i]:.2f}-{bordes[i+1]:.2f}        "
                 f"{len(sub):>6} {sub['p'].mean():>9.3f} "
                 f"{sub['y'].mean():>10.3f}")
    mascara = tabla["p"].ge(bordes[-1])
    sub = tabla[mascara]
    if len(sub):
        registro(f"  >= {bordes[-1]:.2f}           {len(sub):>6} "
                 f"{sub['p'].mean():>9.3f} {sub['y'].mean():>10.3f}")
    registro("")

    registro("SIMULACION DE APUESTA (OVER si P(Over)>0.5+delta, "
             "UNDER si <0.5-delta, cuota 1.91):")
    registro(f"  {'|margen|':<10} {'N':>5} {'OV':>4} {'UN':>4} "
             f"{'Acierto':>8} {'ROI':>8}")
    for delta in (0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
        over = tabla["p"] >= 0.50 + delta
        under = tabla["p"] <= 0.50 - delta
        n = int(over.sum()) + int(under.sum())
        if n == 0:
            continue
        gana_over = (tabla.loc[over, "total"].values > LIMITE_OVER + 0.0).sum()
        gana_under = (tabla.loc[under, "total"].values < LIMITE_OVER - 0.0).sum()
        aciertos = int(gana_over) + int(gana_under)
        roi = (aciertos * (CUOTA_DECIMAL - 1.0) - (n - aciertos)) / n
        registro(f"  {delta:<10} {n:>5} {int(gana_over):>4} "
                 f"{int(gana_under):>4} {aciertos / n:>8.3f} "
                 f"{roi:>+8.3f}")
    registro("")

    registro("NOTA: el objetivo es Total > 8.5 (linea fija). Sin lineas de "
             "mercado historicas, esto mide el SEÑAL INTERNO del modelo; el "
             "ROI real contra Vegas solo es medible con cuotas diarias "
             "(acumulandose desde 2026-08-04).")
    registro("=" * 78)

    if RUTA_REPORTE:
        import os
        os.makedirs(os.path.dirname(RUTA_REPORTE), exist_ok=True)
        with open(RUTA_REPORTE, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")
        print(f"\nReporte guardado en {RUTA_REPORTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())