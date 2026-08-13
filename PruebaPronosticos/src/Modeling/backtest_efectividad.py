# -*- coding: utf-8 -*-
"""Backtest de efectividad por temporada con la LOGICA DE PRODUCCION COMPLETA.

Incluye los 4 cambios aprobados via predecir_hoy.decidir_jugada real:
#1 Kelly corregido para UNDER + filtro de desacuerdo; #2 ERA de
temporada con regresion a media (shrinkage); #3 OPS splits reales
LHP/RHP, viento a favor y dato faltante -> stake 0.5u; #4 regresion al
mercado (peso minimo) y tope de edge contra la linea del mercado.

Walk-forward SIN fuga:
  - 2015: no hay datos previos -> se divide dentro de la temporada
    (entrena 60% inicial, calibra el 20% siguiente, evalua el 40% final).
  - 2016-2026: entrena con TODOS los partidos anteriores al anio.
  - Calibracion isotonica sobre el 20% final del entrenamiento (igual
    que produccion: entrenar_modelo.py).

Linea: 8.5 fija para todos (las lineas de mercado reales solo existen
desde 2026-08-04, 15 juegos; se omiten para mantener un regimen unico).
Cuota: 1.91 fija. Resultado contra la linea apostada con regla .5
(8.5 ya es media linea: sin pushes posibles con totales enteros).
"""

import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
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
from backtest_skellam_ml import (
    aplicar_ajustes_por_lado,
    expected_runs_por_lado,
)

CUOTA = 1.91
LINEA_PROXY = 8.5
ANIOS = list(range(2015, 2027))

feature_engineering_ampayer = ph.feature_engineering_ampayer
feature_engineering_descanso_abridor = ph.feature_engineering_descanso_abridor
feature_engineering_matchup = ph.feature_engineering_matchup

_iso_actual = None


def linea_media_entera(linea, tipo):
    if linea is None:
        return None
    linea = float(linea)
    if abs(linea - round(linea)) > 1e-9:
        return linea
    return linea - 0.5 if tipo == "OVER" else linea + 0.5


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

    # Senal C del ENSEMBLE: proyeccion Skellam por equipo, sobre una COPIA
    # separada (ambas proyecciones crean Anotadas10*/Permitidas10* y los
    # merges colisionarian).
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


def evaluar_ano(df, decodificadores, anio):
    global _iso_actual
    if anio == ANIOS[0]:
        f = df[df["Fecha"].dt.year == anio]
        corte_cal = f["Fecha"].quantile(0.60)
        corte_test = f["Fecha"].quantile(0.80)
        train = df[df["Fecha"] < corte_cal]
        cal = df[(df["Fecha"] >= corte_cal) & (df["Fecha"] < corte_test)]
        test = df[df["Fecha"] >= corte_test]
        tipo = f"OOS dentro de {anio} (2H)"
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
    ph.cargar_calibrador = lambda: _iso_actual

    prob_raw = modelo.predict_proba(X_tst)[:, 1]

    enc_local = decodificadores["EquipoLocal"]
    codigo_a_nombre = {i: nombre for i, nombre in enumerate(enc_local.classes_)}
    media_estadio = train.groupby("EquipoLocal")["Total_Carreras"].mean()
    media_estadio = media_estadio.rename(index=codigo_a_nombre)
    partidos_por_dia = train["Fecha"].dt.date.value_counts()

    apuestas = []
    sin_jugada = []
    for (_, fila), p in zip(test.iterrows(), prob_raw):
        fila = fila.copy()
        fecha = fila["Fecha"]
        inercia_rota = any(
            int(partidos_por_dia.get(fecha.date() - timedelta(days=k), 0))
            <= ph.MIN_PARTIDOS_DESCANSO
            for k in range(1, ph.DIAS_REVISION_DESCANSO + 1))
        decision = ph.decidir_jugada(
            fila, p, media_estadio, partidos_por_dia, fecha,
            inercia_rota, decodificadores)
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
            elif "Viento" in motivo or "viento" in motivo:
                key = "Viento"
            elif "Proyeccion" in motivo:
                key = "Proyeccion"
            else:
                key = "Neutro"
            sin_jugada.append(key)
            continue
        tipo_apuesta = "OVER" if decision["sugerencia"] == ph.SUGERENCIA_OVER \
            else "UNDER"
        linea_apostada = linea_media_entera(float(fila["Linea_Casino"]),
                                            tipo_apuesta)
        total = float(fila["Total_Carreras"])
        gano = (total > linea_apostada and tipo_apuesta == "OVER") \
            or (total < linea_apostada and tipo_apuesta == "UNDER")
        apuestas.append({
            "stake": decision["stake"],
            "gano": gano,
            "cuota": CUOTA,
            "tipo": tipo_apuesta,
            "desacuerdo": decision["desacuerdo_cap"],
            "inercia": decision["inercia_cap"],
            "faltantes": decision["datos_faltantes_cap"],
            "tope_edge": decision["tope_edge"],
            "linea": linea_apostada,
            "total": total,
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
    desac = sum(1 for x in a if x["desacuerdo"])
    inercia = sum(1 for x in a if x["inercia"])
    falt = sum(1 for x in a if x["faltantes"])
    tope = sum(1 for x in a if x["tope_edge"])
    no_bet = len(r["sin_jugada"])
    linea = (f"  {anio:<6} {r['tipo']:<22} {n:>4} {wins:>6} {losses:>6} "
             f"{wins / n:>8.1%} {stake_total:>9.2f} {profit:>+9.2f} "
             f"{roi:>+8.1%}  (1u: {roi_1u:+.1%} | 1.0u:{s1} 0.5u:{s05} "
             f"desac:{desac} iner:{inercia} falt:{falt} topedge:{tope} "
             f"sinbet:{no_bet})")
    return linea


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Construyendo features (pipeline de produccion completo)...")
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
              "BACKTEST DE EFECTIVIDAD - LOGICA DE PRODUCCION COMPLETA (4 CAMBIOS)",
              "(#1 Kelly corregido UNDER + desacuerdo_cap | #2 ERA temporada ",
              "con regresion a media | #3 OPS splits reales + viento-out + ",
              "dato faltante 0.5u | #4 regresion al mercado + tope de edge)",
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
    lineas.append("  - Sin lineas de mercado historicas: 8.5 fija + cuota 1.91.")
    lineas.append("  - 2023 no tiene datos previos: se evalua la 2H dentro ")
    lineas.append("    de la temporada (entrena 60% inicial, calibra 20%).")
    lineas.append("  - Punto de equilibrio con cuota 1.91: 52.4% de acierto.")
    lineas.append("  - Las lineas reales (The Odds API) existen solo desde ")
    lineas.append("    2026-08-04: el CLV real se sigue acumulando.")
    lineas.append("=" * 100)

    texto = "\n".join(lineas)
    print("\n" + texto)
    import os
    os.makedirs("output_predicciones", exist_ok=True)
    with open("output_predicciones/backtest_efectividad.txt", "w",
              encoding="utf-8") as f:
        f.write(texto + "\n")
    print("\nReporte guardado en output_predicciones/backtest_efectividad.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
