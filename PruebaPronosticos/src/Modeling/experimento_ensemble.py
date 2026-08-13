# -*- coding: utf-8 -*-
"""EXPERIMENTO: ENSEMBLE DE 3 SENALES para el mercado Over/Under.

Objetivo: subir la CERTEZA (calidad de la probabilidad final) sin tocar
ningun umbral de riesgo, para que el mismo EDGE_MINIMO deje pasar mas
picks legitimos y siga rechazando los dudosos.

Senales por partido (independientes):
  A) XGBoost calibrado (isotonica) - la senal actual.
  B) Formula de carreras: Phi((Expected_Runs_Ajustada - Linea)/SIGMA_TOTAL)
     probabilidad implicita de la proyeccion propia del motor.
  C) Skellam por equipo (diferencia de carreras validada en ML):
     Phi(((ExpRunsLocal+ExpRunsVisita) - Linea)/SIGMA_TOTAL).

Ensemble: p_final = wA*pA + wB*pB + wC*pC, probado con varios pesos.
La decision usa decidir_jugada con prob_efectiva_previa = p_final
(los filtros defensivos de produccion se mantienen intactos).

Walk-forward identico a backtest_efectividad (sin fuga).
"""

import sys
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

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
import predecir_hoy as ph

CUOTA = 1.91
LINEA_PROXY = 8.5
ANIOS = [2023, 2024, 2025, 2026]

feature_engineering_ampayer = ph.feature_engineering_ampayer
feature_engineering_descanso_abridor = ph.feature_engineering_descanso_abridor
feature_engineering_matchup = ph.feature_engineering_matchup

_iso_actual = None

PESOS_A_PROBAR = [
    ("Baseline (solo XGB)", {"A": 1.0, "B": 0.0, "C": 0.0}),
    ("Equilibrado 1/3", {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}),
    ("XGB 0.5 + formula 0.25 + skellam 0.25", {"A": 0.5, "B": 0.25, "C": 0.25}),
    ("XGB 0.4 + formula 0.3 + skellam 0.3", {"A": 0.4, "B": 0.3, "C": 0.3}),
    ("XGB 0.6 + formula 0.2 + skellam 0.2", {"A": 0.6, "B": 0.2, "C": 0.2}),
]


def construir_df():
    df_raw = cargar_datos(solo_con_temperatura=False)
    df_raw["Viento_Direccion"] = df_raw["Viento_Direccion"].fillna("ND")
    for columna in ("WHIP_Abridor_Local", "WHIP_Abridor_Visita"):
        df_raw[columna] = df_raw[columna].fillna(ph.WHIP_UMBRAL_VOLATILIDAD)
    aux = df_raw[["EsFinal"]].copy()

    df = preprocesar(df_raw)
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    df = feature_engineering_pitchers(df, df["CarrerasVisita"].median())
    df = feature_engineering_bullpen(df)
    df = feature_engineering_ampayer(df)
    df = feature_engineering_descanso_abridor(df)
    df = feature_engineering_matchup(df)
    df = df.join(aux, how="left")
    df["Total_Carreras"] = df["CarrerasLocal"] + df["CarrerasVisita"]
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df[df["EsFinal"] == 1].sort_values("Fecha").reset_index(drop=True)

    # Senal C: proyeccion Skellam por equipo (reutiliza el pipeline ML).
    # Se calcula sobre una COPIA separada: ambas proyecciones crean las
    # columnas Anotadas10*/Permitidas10* y los merges colisionarian.
    from backtest_skellam_ml import aplicar_ajustes_por_lado, expected_runs_por_lado
    df_sk = df.copy()
    df_sk = expected_runs_por_lado(df_sk, df_sk)
    df_sk = aplicar_ajustes_por_lado(df_sk)
    df["ExpRunsTotal"] = (
        df_sk["ExpRunsLocal"] + df_sk["ExpRunsVisita"]).values

    df = ph.calcular_expected_runs(df, df)

    df["Linea_Casino"] = LINEA_PROXY
    df["Cuota"] = CUOTA
    df["Cuota_Over"] = CUOTA
    df["Cuota_Under"] = CUOTA
    df = ph.aplicar_ajustes_y_edge(df)
    decodificadores = ph.construir_decodificadores(df_raw)
    return df, decodificadores


def entrenar_ano(df, anio):
    """Entrena y devuelve (test, proba_xgb_raw) para el anio."""
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
    modelo.fit(X_ent, train["Target_Over"].values)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(modelo.predict_proba(X_cal)[:, 1], cal["Target_Over"].values)
    _iso_actual = iso

    prob_raw = modelo.predict_proba(X_tst)[:, 1]
    return test, prob_raw, tipo


def senales(test, prob_raw):
    """pA (XGB calibrada), pB (formula), pC (skellam) por partido."""
    pA = np.array([_iso_actual.predict([p])[0] for p in prob_raw])
    pB = np.asarray(norm.cdf(
        (test["Expected_Runs_Ajustada"] - test["Linea_Casino"]) / ph.SIGMA_TOTAL))
    pC = np.asarray(norm.cdf(
        (test["ExpRunsTotal"] - test["Linea_Casino"]) / ph.SIGMA_TOTAL))
    return pA, pB, pC


def evaluar(test, proba_final, decodificadores):
    """Aplica la logica de produccion completa con la prob inyectada."""
    enc_local = decodificadores["EquipoLocal"]
    codigo_a_nombre = {i: nombre for i, nombre in enumerate(enc_local.classes_)}
    media_estadio = test.groupby("EquipoLocal")["Total_Carreras"].mean()
    media_estadio = media_estadio.rename(index=codigo_a_nombre)
    partidos_por_dia = test["Fecha"].dt.date.value_counts()

    apuestas = []
    sin_jugada = []
    for (_, fila), p in zip(test.iterrows(), proba_final):
        fila = fila.copy()
        fecha = fila["Fecha"]
        inercia_rota = any(
            int(partidos_por_dia.get(fecha.date() - timedelta(days=k), 0))
            <= ph.MIN_PARTIDOS_DESCANSO
            for k in range(1, ph.DIAS_REVISION_DESCANSO + 1))
        decision = ph.decidir_jugada(
            fila, p, media_estadio, partidos_por_dia, fecha,
            inercia_rota, decodificadores, prob_efectiva_previa=p)
        if decision["sugerencia"] not in (ph.SUGERENCIA_OVER, ph.SUGERENCIA_UNDER):
            motivo = decision["motivo_anulacion"] or "Zona Neutra"
            if "Contradiccion" in motivo:
                key = "Contradiccion"
            elif "Extremo" in motivo:
                key = "Extremo"
            elif "Margen" in motivo:
                key = "Margen"
            elif "Volatilidad" in motivo:
                key = "Volatilidad"
            elif "Fatiga" in motivo:
                key = "Fatiga"
            elif "Viento" in motivo:
                key = "Viento"
            elif "Proyeccion" in motivo:
                key = "Proyeccion"
            else:
                key = "Neutro"
            sin_jugada.append(key)
            continue
        tipo_apuesta = "OVER" if decision["sugerencia"] == ph.SUGERENCIA_OVER \
            else "UNDER"
        linea = float(fila["Linea_Casino"])
        total = float(fila["Total_Carreras"])
        gano = (total > linea and tipo_apuesta == "OVER") \
            or (total < linea and tipo_apuesta == "UNDER")
        apuestas.append({
            "stake": decision["stake"],
            "gano": gano,
            "cuota": CUOTA,
            "tipo": tipo_apuesta,
        })
    return {"apuestas": apuestas, "sin_jugada": sin_jugada}


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    print("[1/3] Construyendo features (pipeline completo + skellam)...")
    df, decodificadores = construir_df()
    print(f"      {len(df)} partidos finales "
          f"({df['Fecha'].dt.year.min()} - {df['Fecha'].dt.year.max()}).")

    print("[2/3] Entrenando walk-forward (una sola vez)...")
    anos = {}
    for anio in ANIOS:
        test, prob_raw, tipo = entrenar_ano(df, anio)
        pA, pB, pC = senales(test, prob_raw)
        anos[anio] = {"test": test, "tipo": tipo, "pA": pA, "pB": pB, "pC": pC}
        print(f"      {anio} ({tipo}): {len(test)} partidos.")

    print("[3/3] Evaluando combinaciones de pesos...")
    print()
    print(f"{'Pesos (A/B/C)':<40} {'Ap':>6} {'OK':>6} {'Acierto':>9} "
          f"{'Apostado':>9} {'Unid':>9} {'ROI':>9}")
    print("-" * 100)
    resultados = {}
    for nombre, pesos in PESOS_A_PROBAR:
        tot = {"n": 0, "wins": 0, "stake": 0.0, "profit": 0.0}
        for anio in ANIOS:
            info = anos[anio]
            p_final = (pesos["A"] * info["pA"]
                       + pesos["B"] * info["pB"]
                       + pesos["C"] * info["pC"])
            r = evaluar(info["test"], p_final, decodificadores)
            for x in r["apuestas"]:
                tot["n"] += 1
                tot["wins"] += 1 if x["gano"] else 0
                tot["stake"] += x["stake"]
                tot["profit"] += x["stake"] * (x["cuota"] - 1.0) if x["gano"] \
                    else -x["stake"]
        roi = tot["profit"] / tot["stake"] if tot["stake"] else 0.0
        hr = tot["wins"] / tot["n"] if tot["n"] else 0.0
        resultados[nombre] = tot
        print(f"{nombre:<40} {tot['n']:>6} {tot['wins']:>6} {hr:>9.1%} "
              f"{tot['stake']:>9.2f} {tot['profit']:>+9.2f} {roi:>+9.1%}")

    print()
    print("Punto de equilibrio (cuota 1.91): 52.4% de acierto.")
    print("Nota: prob_efectiva del ensemble se inyecta con los filtros de")
    print("produccion intactos (margen 1.45, contradiccion, extremos, etc.).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
