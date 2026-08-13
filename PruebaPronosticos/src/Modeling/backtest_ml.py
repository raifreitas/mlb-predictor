# -*- coding: utf-8 -*-
"""Backtest del mercado Moneyline (ganador del partido) con walk-forward SIN fuga.

Mercado AISLADO del flujo Over/Under: usa entrenar_modelo_ml.py
(target: gana el equipo local) y la logica de decision de
predecir_ml.decidir_jugada_ml (margen minimo, regresion al mercado,
tope de edge, Kelly media, valor negativo, datos faltantes).

Regimen unico para todo el periodo (2023-2026):
  - Sin cuotas h2h historicas (The Odds API solo captura desde 2026-08-06):
    cuota proxy 1.91 para ambos lados -> p_mercado implicita 0.5.
  - La calibracion isotonica se ajusta sobre el 20% final del
    entrenamiento (igual que produccion).

Walk-forward:
  - 2023: entrena 60% inicial de 2023, calibra el 20% siguiente,
    evalua el 40% final (no hay datos previos).
  - 2024/2025/2026: entrena con TODOS los partidos anteriores al anio.
"""

import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from entrenar_modelo import obtener_driver_odbc
from entrenar_modelo_ml import (
    ajustar_transformadores,
    cargar_datos,
    construir_caracteristicas_finales,
    feature_engineering_bullpen,
    feature_engineering_descanso_abridor,
    feature_engineering_fatiga,
    feature_engineering_pitchers,
    feature_engineering_rachas,
    preprocesar,
)
import predecir_ml as pml

CUOTA = 1.91
ANIOS = [2023, 2024, 2025, 2026]

_iso_actual = None


def construir_df():
    df_raw = cargar_datos()
    df_raw["Fecha"] = pd.to_datetime(df_raw["Fecha"])
    if "Viento_Direccion" in df_raw.columns:
        df_raw["Viento_Direccion"] = df_raw["Viento_Direccion"].fillna("ND")
    for columna in ("WHIP_Abridor_Local", "WHIP_Abridor_Visita"):
        df_raw[columna] = df_raw[columna].fillna(1.30)

    df = preprocesar(df_raw)
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    df = feature_engineering_pitchers(df, df["CarrerasVisita"].median())
    df = feature_engineering_bullpen(df)
    df = feature_engineering_descanso_abridor(df)

    # Solo partidos finalizados (carreras reales conocidas).
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.dropna(subset=["CarrerasLocal", "CarrerasVisita"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    from sklearn.preprocessing import LabelEncoder
    decodificadores = {}
    for columna in ["EquipoLocal", "EquipoVisita"]:
        codificador = LabelEncoder()
        codificador.fit(df_raw[columna])
        decodificadores[columna] = codificador
    return df, decodificadores


def evaluar_ano(df, decodificadores, anio):
    global _iso_actual
    if anio == 2023:
        f = df[df["Fecha"].dt.year == 2023]
        corte_cal = f["Fecha"].quantile(0.60)
        corte_test = f["Fecha"].quantile(0.80)
        train = df[df["Fecha"] < corte_cal]
        cal = df[(df["Fecha"] >= corte_cal) & (df["Fecha"] < corte_test)]
        test = df[df["Fecha"] >= corte_test]
        tipo = "OOS dentro de 2023 (2H)"
    else:
        train = df[df["Fecha"].dt.year < anio]
        cal = train.tail(max(1, int(len(train) * 0.2)))
        test = df[df["Fecha"].dt.year == anio]
        tipo = "out-of-time"

    transformadores = ajustar_transformadores(train)
    X_ent = construir_caracteristicas_finales(train, transformadores)
    X_cal = construir_caracteristicas_finales(cal, transformadores)
    X_tst = construir_caracteristicas_finales(test, transformadores)

    modelo = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_estimators=600, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5)
    modelo.fit(X_ent, train["Target_GanaLocal"].values)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(modelo.predict_proba(X_cal)[:, 1], cal["Target_GanaLocal"].values)
    _iso_actual = iso
    pml.cargar_calibrador = lambda: _iso_actual

    prob_raw = modelo.predict_proba(X_tst)[:, 1]

    apuestas = []
    sin_jugada = []
    for (_, fila), p in zip(test.iterrows(), prob_raw):
        fila = fila.copy()
        # Sin cuotas historicas: proxy 1.91 -> p_mercado implicita 0.5.
        decision = pml.decidir_jugada_ml(fila, p, CUOTA, CUOTA, decodificadores)
        if decision["sugerencia"] == pml.SUGERENCIA_NO_BET_ML:
            motivo = decision["motivo_anulacion"] or "Zona Neutra"
            if "Edge Excesivo" in motivo:
                key = "EdgeExcesivo"
            elif "Valor Negativo" in motivo:
                key = "ValorNegativo"
            elif "Margen" in motivo:
                key = "Margen"
            else:
                key = "Neutro"
            sin_jugada.append(key)
            continue

        local = float(fila["CarrerasLocal"])
        visita = float(fila["CarrerasVisita"])
        gana_local = local > visita
        pick = (decision["sugerencia"] == pml.SUGERENCIA_HOME)
        gano = gana_local == pick

        apuestas.append({
            "stake": decision["stake"],
            "gano": gano,
            "cuota": CUOTA,
            "pick": "HOME" if pick else "AWAY",
            "desacuerdo": decision["desacuerdo"],
            "tope_edge": decision["tope_edge"],
            "faltantes": decision["datos_faltantes_cap"],
        })
    return {"tipo": tipo, "apuestas": apuestas, "sin_jugada": sin_jugada}


def resumen_anio(anio, r):
    a = r["apuestas"]
    n = len(a)
    if n == 0:
        return (f"  {anio:<6} {r['tipo']:<22} {0:>4} {0:>6} {0:>6} "
                f"{0:>8.1%} {0:>9.2f} {0:>+9.2f} {0:>+8.1%}")
    wins = sum(1 for x in a if x["gano"])
    losses = n - wins
    stake_total = sum(x["stake"] for x in a)
    profit = sum(x["stake"] * (x["cuota"] - 1.0) if x["gano"]
                 else -x["stake"] for x in a)
    roi = profit / stake_total if stake_total else 0.0
    roi_1u = (wins * (CUOTA - 1.0) - losses) / n
    s1 = sum(1 for x in a if x["stake"] >= 1.0)
    s05 = n - s1
    tope = sum(1 for x in a if x["tope_edge"])
    no_bet = len(r["sin_jugada"])
    linea = (f"  {anio:<6} {r['tipo']:<22} {n:>4} {wins:>6} {losses:>6} "
             f"{wins / n:>8.1%} {stake_total:>9.2f} {profit:>+9.2f} "
             f"{roi:>+8.1%}  (1u: {roi_1u:+.1%} | 1.0u:{s1} 0.5u:{s05} "
             f"topedge:{tope} sinbet:{no_bet})")
    return linea


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Construyendo features (pipeline de produccion ML)...")
    df, _decodificadores = construir_df()
    print(f"      {len(df)} partidos finales "
          f"({df['Fecha'].dt.year.min()} - {df['Fecha'].dt.year.max()}).")

    print("[2/3] Evaluando temporadas (walk-forward, logica de produccion)...")
    resultados = {}
    for anio in ANIOS:
        r = evaluar_ano(df, _decodificadores, anio)
        resultados[anio] = r
        print(resumen_anio(anio, r))

    print("[3/3] Reporte final...")
    lineas = ["=" * 100,
              "BACKTEST MONEYLINE - LOGICA DE PRODUCCION COMPLETA",
              "(modelo XGBoost target GanaLocal | margen minimo | regresion",
              "al mercado | tope de edge | valor negativo | Kelly media |",
              "datos faltantes 0.5u)",
              "=" * 100,
              f"  {'Anio':<6} {'Tipo':<22} {'Ap':>4} {'OK':>6} {'KO':>6} "
              f"{'Acierto':>8} {'Apostado':>9} {'Unid':>9} {'ROI':>8} "
              "desglose",
              "-" * 100]
    tot = {"n": 0, "wins": 0, "stake": 0.0, "profit": 0.0}
    for anio in ANIOS:
        l = resumen_anio(anio, resultados[anio])
        lineas.append(l)
        r = resultados[anio]
        for x in r["apuestas"]:
            tot["n"] += 1
            tot["wins"] += 1 if x["gano"] else 0
            tot["stake"] += x["stake"]
            tot["profit"] += x["stake"] * (x["cuota"] - 1.0) if x["gano"] \
                else -x["stake"]
    roi = tot["profit"] / tot["stake"] if tot["stake"] else 0.0
    hr = tot["wins"] / tot["n"] if tot["n"] else 0.0
    lineas.append("-" * 100)
    lineas.append(f"  {'TOTAL':<6} {'':<22} {tot['n']:>4} {tot['wins']:>6} "
                  f"{tot['n'] - tot['wins']:>6} {hr:>8.1%} {tot['stake']:>9.2f} "
                  f"{tot['profit']:>+9.2f} {roi:>+8.1%}")
    lineas.append("=" * 100)
    lineas.append("NOTAS:")
    lineas.append("  - Sin cuotas h2h historicas: 1.91 fija (p_mercado 0.5).")
    lineas.append("  - 2023 no tiene datos previos: se evalua la 2H dentro ")
    lineas.append("    de la temporada (entrena 60% inicial, calibra 20%).")
    lineas.append("  - Punto de equilibrio con cuota 1.91: 52.4% de acierto.")
    lineas.append("  - Las cuotas reales h2h (LineaSnapshotsML) existen solo ")
    lineas.append("    desde 2026-08-06: el CLV real se sigue acumulando.")
    lineas.append("=" * 100)

    texto = "\n".join(lineas)
    print("\n" + texto)
    import os
    os.makedirs("output_predicciones", exist_ok=True)
    with open("output_predicciones/backtest_moneyline.txt", "w",
              encoding="utf-8") as f:
        f.write(texto + "\n")
    print("\nReporte guardado en output_predicciones/backtest_moneyline.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
