# -*- coding: utf-8 -*-
"""Backtest MONEYLINE POR DIFERENCIA DE CARRERAS (enfoque Skellam).

En lugar del clasificador XGBoost sobre un target binario ruidoso
(gana local ~52% baseline), se usa la proyeccion por EQUIPO del motor
de carreras ya validado en Totals (calcular_expected_runs): cada lado
tiene su Expected_Runs, y la diferencia D = exp_local - exp_visita se
convierte en P(gana local) con una sigmoide de UN parametro beta
calibrado por walk-forward (sin sobreajuste).

Regimen del backtest:
  - Cuota proxy 1.91 para ambos lados (sin h2h historico).
  - Filtros de produccion: margen minimo (MARGEN_MIN_PROB_ML),
    tope de edge, valor negativo (break-even), Kelly media.
  - Variante con regresion al mercado fuerte (PESO_MERCADO_MAX 0.25,
    p_mercado implicita 0.5 con cuota 1.91) para ver como sobrevive.

Walk-forward:
  - 2023: entrena 60% inicial, calibra 20% siguiente, evalua 40% final.
  - 2024/2025/2026: entrena con todos los partidos anteriores al anio.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from entrenar_modelo import cargar_datos
from entrenar_modelo_ml import (
    feature_engineering_bullpen,
    feature_engineering_descanso_abridor,
    feature_engineering_fatiga,
    feature_engineering_pitchers,
    feature_engineering_rachas,
    preprocesar,
)
import predecir_hoy as ph

CUOTA = 1.91
ANIOS = [2023, 2024, 2025, 2026]

# Constantes de produccion ML (mismas que predecir_ml.py, ajustadas).
MARGEN_MIN_PROB_ML = 0.07
EDGE_MAXIMO_ML = 0.25
PESO_MERCADO_MAX_ML = 0.25
DESACUERDO_MERCADO_REF_ML = 0.20
LIMITE_STAKE_ALTO_ML = 0.08
SUGERENCIA_HOME = "APOSTAR LOCAL"
SUGERENCIA_AWAY = "APOSTAR VISITA"
SUGERENCIA_NO_BET_ML = "NO APOSTAR"

PESO_ERA_ULTIMAS3 = 0.35
PESO_ERA_APROXIMADA = 0.25
PESO_ERA_TEMPORADA = 0.40
PESO_ABRIDOR = 0.6
PESO_BULLPEN = 0.4


def expected_runs_por_lado(partidos, df_historico):
    """Replica calcular_expected_runs de predecir_hoy pero por EQUIPO.

    Devuelve exp_local y exp_visita (carreras esperadas de cada lado) con
    la misma ponderacion 70/30, park factor y ajustes de frio repartidos
    mitad a cada lado.
    """
    df = df_historico.copy()
    df["Partido"] = df.index

    partes = []
    for lado in ["Local", "Visita"]:
        parte = pd.DataFrame(
            {
                "Partido": df.index,
                "Fecha": df["Fecha"],
                "Equipo": df[f"Equipo{lado}"],
                "Anotadas": df[f"Carreras{lado}"],
                "Permitidas": df["CarrerasVisita" if lado == "Local" else "CarrerasLocal"],
                "Lado": lado,
            }
        )
        partes.append(parte)

    apariciones = pd.concat(partes, ignore_index=True)
    apariciones = apariciones.sort_values(["Fecha", "Partido"])
    grupo = apariciones.groupby("Equipo", sort=False)
    apariciones["Anotadas10"] = grupo["Anotadas"].transform(
        lambda s: s.shift(1).rolling(ph.VENTANA_ESPERADOS, min_periods=1).mean())
    apariciones["Permitidas10"] = grupo["Permitidas"].transform(
        lambda s: s.shift(1).rolling(ph.VENTANA_ESPERADOS, min_periods=1).mean())

    for lado in ["Local", "Visita"]:
        registro = apariciones.loc[
            apariciones["Lado"] == lado,
            ["Partido", "Anotadas10", "Permitidas10"],
        ]
        registro = registro.rename(
            columns={
                "Anotadas10": f"Anotadas10{lado}",
                "Permitidas10": f"Permitidas10{lado}",
            }
        )
        partidos = partidos.merge(registro, on="Partido", how="left")

    mediana_carreras = df_historico["CarrerasVisita"].median()
    anotadas_local = partidos["Anotadas10Local"].fillna(mediana_carreras)
    permitidas_visita = partidos["Permitidas10Visita"].fillna(mediana_carreras)
    anotadas_visita = partidos["Anotadas10Visita"].fillna(mediana_carreras)
    permitidas_local = partidos["Permitidas10Local"].fillna(mediana_carreras)

    era_local = (PESO_ERA_ULTIMAS3 * partidos["ERA_Ultimas3_Local"]
                 + PESO_ERA_APROXIMADA * partidos["ERA_Aproximada_Local"]
                 + PESO_ERA_TEMPORADA * partidos["ERA_Temporada_Local"])
    era_visita = (PESO_ERA_ULTIMAS3 * partidos["ERA_Ultimas3_Visita"]
                  + PESO_ERA_APROXIMADA * partidos["ERA_Aproximada_Visita"]
                  + PESO_ERA_TEMPORADA * partidos["ERA_Temporada_Visita"])

    park = partidos["Factor_Carreras"].fillna(1.0)
    era_local = era_local.fillna(mediana_carreras)
    era_visita = era_visita.fillna(mediana_carreras)

    mediana_bullpen = partidos["ERA_Bullpen_Local"].median() \
        if "ERA_Bullpen_Local" in partidos.columns else 4.00
    era_bull_local = partidos["ERA_Bullpen_Reciente_Local"].fillna(mediana_bullpen)
    era_bull_visita = partidos["ERA_Bullpen_Reciente_Visita"].fillna(mediana_bullpen)

    base_off_local = (anotadas_local + permitidas_visita) / 2.0
    base_off_visita = (anotadas_visita + permitidas_local) / 2.0

    pitching_allow_local = (PESO_ABRIDOR * era_local
                            + PESO_BULLPEN * era_bull_local)
    pitching_allow_visita = (PESO_ABRIDOR * era_visita
                             + PESO_BULLPEN * era_bull_visita)

    exp_runs_local = 0.30 * base_off_local + 0.70 * pitching_allow_visita
    exp_runs_visita = 0.30 * base_off_visita + 0.70 * pitching_allow_local

    aire_frio = partidos["TemperaturaC"] < ph.LIMITE_TEMPERATURA_FRIO_C
    penalizacion_frio = ph.PENALIZACION_FRIO_CARRERAS / 2.0

    partidos["ExpRunsLocal"] = (exp_runs_local * park
                                - penalizacion_frio * aire_frio)
    partidos["ExpRunsVisita"] = (exp_runs_visita * park
                                 - penalizacion_frio * aire_frio)
    return partidos


def aplicar_ajustes_por_lado(partidos):
    """Ajustes dinamicos (ampayer, descanso, matchup, viento) repartidos
    mitad a cada lado, como se hace con el total en produccion."""
    for columna_ajuste in ("Ajuste_Ampayer", "Ajuste_Descanso", "Ajuste_Matchup"):
        if columna_ajuste in partidos.columns:
            partidos[columna_ajuste] = partidos[columna_ajuste].fillna(0.0)
    direcciones_viento = partidos.get("Viento_Direccion")
    if direcciones_viento is None:
        direcciones_viento = pd.Series("ND", index=partidos.index)
    partidos["Ajuste_Viento"] = [
        ph._ajuste_viento_carreras(v, d) for v, d in zip(
            partidos.get("Viento_Velocidad", pd.Series(0.0, index=partidos.index)),
            direcciones_viento)]
    partidos["Ajuste_Dinamico"] = (
        partidos.get("Ajuste_Ampayer", 0.0)
        + partidos.get("Ajuste_Descanso", 0.0)
        + partidos.get("Ajuste_Matchup", 0.0)
        + partidos["Ajuste_Viento"]
    ).clip(-ph.LIMITE_AJUSTE_DINAMICO, ph.LIMITE_AJUSTE_DINAMICO)
    partidos["ExpRunsLocal"] = partidos["ExpRunsLocal"] + partidos["Ajuste_Dinamico"] / 2.0
    partidos["ExpRunsVisita"] = partidos["ExpRunsVisita"] + partidos["Ajuste_Dinamico"] / 2.0
    return partidos


def prob_gana_local(exp_local, exp_visita, beta):
    """P(gana local) = sigmoid(beta * (exp_local - exp_visita))."""
    d = exp_local - exp_visita
    z = np.clip(beta * d, -12.0, 12.0)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-4, 1 - 1e-4)


def calibrar_beta(df_entrenamiento):
    """Beta unico que minimiza log-loss sobre el entrenamiento."""
    d = (df_entrenamiento["ExpRunsLocal"] - df_entrenamiento["ExpRunsVisita"])
    y = df_entrenamiento["Target_GanaLocal"].values
    mejor = (None, 1e9)
    for beta in np.arange(0.05, 2.01, 0.05):
        p = prob_gana_local(d, pd.Series(0.0, index=d.index), beta)
        ll = -np.mean(y * np.log(np.clip(p, 1e-9, 1))
                      + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        if ll < mejor[1]:
            mejor = (beta, ll)
    return mejor[0] or 0.5


def construir_df():
    df_raw = cargar_datos()
    df_raw["Viento_Direccion"] = df_raw["Viento_Direccion"].fillna("ND")
    for columna in ("WHIP_Abridor_Local", "WHIP_Abridor_Visita"):
        df_raw[columna] = df_raw[columna].fillna(ph.WHIP_UMBRAL_VOLATILIDAD)

    df = preprocesar(df_raw)
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    df = feature_engineering_pitchers(df, df["CarrerasVisita"].median())
    df = feature_engineering_bullpen(df)
    df = ph.feature_engineering_ampayer(df)
    df = ph.feature_engineering_descanso_abridor(df)
    df = ph.feature_engineering_matchup(df)

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.dropna(subset=["CarrerasLocal", "CarrerasVisita"])
    df = df.sort_values("Fecha").reset_index(drop=True)
    df = expected_runs_por_lado(df, df)
    df = aplicar_ajustes_por_lado(df)
    return df


def decidir(p, cuota_home, cuota_away):
    """Decision con filtros de produccion (regresion al mercado opcional).

    Devuelve (sugerencia, stake, motivo).
    """
    ch = float(cuota_home)
    ca = float(cuota_away)
    p_mercado = (1.0 / ch) / (1.0 / ch + 1.0 / ca)
    p_mercado = min(max(p_mercado, 1e-4), 1 - 1e-4)

    desacuerdo = abs(p - p_mercado)
    peso_mercado = min(PESO_MERCADO_MAX_ML,
                       desacuerdo * PESO_MERCADO_MAX_ML / DESACUERDO_MERCADO_REF_ML)
    p_final = min(max((1.0 - peso_mercado) * p + peso_mercado * p_mercado, 0.0), 1.0)

    if p_final >= 0.5 + MARGEN_MIN_PROB_ML:
        sugerencia = SUGERENCIA_HOME
    elif p_final <= 0.5 - MARGEN_MIN_PROB_ML:
        sugerencia = SUGERENCIA_AWAY
    else:
        sugerencia = SUGERENCIA_NO_BET_ML

    if sugerencia == SUGERENCIA_NO_BET_ML:
        return sugerencia, None, "Zona Neutra"

    if desacuerdo < 0.055:
        return SUGERENCIA_NO_BET_ML, None, "Margen Insuficiente"
    if desacuerdo > EDGE_MAXIMO_ML:
        return SUGERENCIA_NO_BET_ML, None, "Edge Excesivo"

    cuota_pick = ch if sugerencia == SUGERENCIA_HOME else ca
    cuota_breakeven = float(cuota_pick) if float(cuota_pick) > 1.0 else CUOTA
    if p_final <= 1.0 / cuota_breakeven:
        return SUGERENCIA_NO_BET_ML, None, "Valor Negativo"

    b = cuota_breakeven - 1.0
    f = max((p_final * b - (1.0 - p_final)) / b, 0.0) * 0.5
    stake = 1.0 if f >= LIMITE_STAKE_ALTO_ML else 0.5
    return sugerencia, stake, None


def evaluar_ano(df, anio):
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

    beta = calibrar_beta(train)

    # Calibracion isotonica sobre la mitad de calibracion.
    p_cal = prob_gana_local(cal["ExpRunsLocal"], cal["ExpRunsVisita"], beta)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal.values, cal["Target_GanaLocal"].values)

    apuestas = []
    sin_jugada = []
    for (_, fila) in test.iterrows():
        p_raw = prob_gana_local(fila["ExpRunsLocal"], fila["ExpRunsVisita"], beta)
        p = float(iso.predict([p_raw])[0])
        sugerencia, stake, motivo = decidir(p, CUOTA, CUOTA)
        if sugerencia == SUGERENCIA_NO_BET_ML:
            sin_jugada.append(motivo)
            continue
        gana_local = fila["CarrerasLocal"] > fila["CarrerasVisita"]
        pick = sugerencia == SUGERENCIA_HOME
        apuestas.append({
            "stake": stake,
            "gano": gana_local == pick,
            "cuota": CUOTA,
            "p": p,
            "desacuerdo": abs(p - 0.5),
        })
    return {"tipo": tipo, "beta": beta, "apuestas": apuestas,
            "sin_jugada": sin_jugada}


def resumen_anio(anio, r):
    a = r["apuestas"]
    n = len(a)
    if n == 0:
        return (f"  {anio:<6} {r['tipo']:<22} {0:>4} {0:>6} {0:>6} "
                f"{0:>8.1%} {0:>9.2f} {0:>+9.2f} {0:>+8.1%}  beta={r['beta']:.2f}")
    wins = sum(1 for x in a if x["gano"])
    losses = n - wins
    stake_total = sum(x["stake"] for x in a)
    profit = sum(x["stake"] * (x["cuota"] - 1.0) if x["gano"]
                 else -x["stake"] for x in a)
    roi = profit / stake_total if stake_total else 0.0
    roi_1u = (wins * (CUOTA - 1.0) - losses) / n
    return (f"  {anio:<6} {r['tipo']:<22} {n:>4} {wins:>6} {losses:>6} "
            f"{wins / n:>8.1%} {stake_total:>9.2f} {profit:>+9.2f} "
            f"{roi:>+8.1%}  (1u: {roi_1u:+.1%} | beta={r['beta']:.2f})")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Construyendo features (pipeline de carreras por equipo)...")
    df = construir_df()
    print(f"      {len(df)} partidos finales "
          f"({df['Fecha'].dt.year.min()} - {df['Fecha'].dt.year.max()}).")

    print("[2/3] Evaluando temporadas (walk-forward)...")
    resultados = {}
    for anio in ANIOS:
        r = evaluar_ano(df, anio)
        resultados[anio] = r
        print(resumen_anio(anio, r))

    print("[3/3] Reporte final...")
    lineas = ["=" * 100,
              "BACKTEST MONEYLINE POR DIFERENCIA DE CARRERAS (Skellam/normal)",
              "(proyeccion por equipo del motor de carreras | sigmoide 1",
              "parametro | margen 0.07 | tope edge 0.25 | valor negativo |",
              "regresion al mercado peso 0.25 | Kelly media)",
              "=" * 100,
              f"  {'Anio':<6} {'Tipo':<22} {'Ap':>4} {'OK':>6} {'KO':>6} "
              f"{'Acierto':>8} {'Apostado':>9} {'Unid':>9} {'ROI':>8} "
              "desglose",
              "-" * 100]
    tot = {"n": 0, "wins": 0, "stake": 0.0, "profit": 0.0}
    for anio in ANIOS:
        l = resumen_anio(anio, resultados[anio])
        lineas.append(l)
        for x in resultados[anio]["apuestas"]:
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
    lineas.append("  - Cuota 1.91 fija (p_mercado 0.5): el edge real vs el")
    lineas.append("    mercado h2h se medira con CLV acumulado en vivo.")
    lineas.append("  - 2023 sin datos previos: entrena 60% inicial, calibra 20%.")
    lineas.append("=" * 100)

    texto = "\n".join(lineas)
    print("\n" + texto)
    import os
    os.makedirs("output_predicciones", exist_ok=True)
    with open("output_predicciones/backtest_skellam_ml.txt", "w",
              encoding="utf-8") as f:
        f.write(texto + "\n")
    print("\nReporte guardado en output_predicciones/backtest_skellam_ml.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
