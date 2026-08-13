# -*- coding: utf-8 -*-
"""Prueba historica completa: 2023, 2024, 2025 (y 2026 parcial).

Walk-forward honesto: para cada temporada se entrena SOLO con anos
anteriores (sin fuga) + calibracion isotonica sobre el 20% final de
entrenamiento (igual que produccion). Cada partido recibe P(Over)
calibrada y el sistema pronostica OVER/UNDER contra la linea base 8.5:

  - OVER  si P >= 0.5 + MARGEN (0.055)
  - UNDER si P <= 0.5 - MARGEN
  - NEUTRO si no pasa el margen (no se pronostica)

Resultado real: total > 8.5 (acierto OVER), < 8.5 (acierto UNDER),
= 8.5 (empate/push, se devuelve el dinero).

NOTA: sin historico de lineas de mercado (solo existen desde 2026-08-04),
el acierto es contra 8.5 fija; contra la linea real de Vegas se medira con
las cuotas que se estan acumulando.
"""

import sys
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from reporte_rendimiento import construir_df_modelo
from entrenar_modelo import ajustar_transformadores, construir_caracteristicas_finales

MARGEN = 0.055
LIMITE = 8.5
CUOTA = 1.91
ANIOS = [2023, 2024, 2025, 2026]


def evaluar_ano(df, anio, usar_solo_previos):
    """Entrena con anos previos (o el mismo si no hay) y evalua el anio."""
    train = df[df["Fecha"].dt.year < anio] if usar_solo_previos \
        else df[df["Fecha"].dt.year == anio]
    test = df[df["Fecha"].dt.year == anio]
    n_fin = max(1, int(len(train) * 0.2))
    cal_tail = train.tail(n_fin)

    trans = ajustar_transformadores(train)
    X_ent = construir_caracteristicas_finales(train, trans)
    X_cal = construir_caracteristicas_finales(cal_tail, trans)
    X_tst = construir_caracteristicas_finales(test, trans)

    modelo = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_estimators=600, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5)
    modelo.fit(X_ent, train["Target_Over"].values)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(modelo.predict_proba(X_cal)[:, 1], cal_tail["Target_Over"].values)
    p = iso.predict(modelo.predict_proba(X_tst)[:, 1]).clip(1e-4, 1 - 1e-4)
    total = test["Total_Carreras"].values

    pronostico = pd.Series("NEUTRO", index=test.index)
    pronostico.iloc[p >= 0.5 + MARGEN] = "OVER"
    pronostico.iloc[p <= 0.5 - MARGEN] = "UNDER"

    res = pd.DataFrame({
        "pronostico": pronostico.values,
        "total": total,
    })
    res["resultado"] = None
    res.loc[res["total"] > LIMITE, "resultado"] = "OVER"
    res.loc[res["total"] < LIMITE, "resultado"] = "UNDER"
    res.loc[res["total"] == LIMITE, "resultado"] = "PUSH"

    aciertos = int(((res["pronostico"] == "OVER") & (res["resultado"] == "OVER")).sum()
                   + ((res["pronostico"] == "UNDER") & (res["resultado"] == "UNDER")).sum())
    fallos = int(((res["pronostico"] == "OVER") & (res["resultado"] == "UNDER")).sum()
                 + ((res["pronostico"] == "UNDER") & (res["resultado"] == "OVER")).sum())
    pushes = int(((res["pronostico"].isin(["OVER", "UNDER"])) & (res["resultado"] == "PUSH")).sum())
    neutros = int((res["pronostico"] == "NEUTRO").sum())
    pronosticados = aciertos + fallos + pushes
    efectivos = aciertos + fallos

    return {
        "anio": anio,
        "juegos": len(test),
        "pronosticados": pronosticados,
        "aciertos": aciertos,
        "fallos": fallos,
        "pushes": pushes,
        "neutros": neutros,
        "pct": aciertos / efectivos if efectivos else 0.0,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/2] Construyendo features (pipeline de produccion)...")
    df = construir_df_modelo()

    print("[2/2] Evaluando temporadas (walk-forward)...")
    filas = []
    for anio in ANIOS:
        solo_previos = anio > 2023
        r = evaluar_ano(df, anio, solo_previos)
        filas.append(r)
        nota = "" if solo_previos else "  (in-sample: sin datos previos)"
        print(f"      {anio}: {r['pronosticados']} pronosticados "
              f"| {r['aciertos']} acertados | {r['fallos']} fallados "
              f"| {r['pushes']} push | {r['neutros']} neutros"
              f" -> {r['pct']:.1%}{nota}")

    print("")
    print("=" * 88)
    print("PRUEBA HISTORICA: pronosticos vs 8.5 (walk-forward, sin fuga)")
    print("=" * 88)
    print(f"  {'Anio':<8} {'Juegos':>7} {'Pronost':>8} {'Aciertos':>9} "
          f"{'Fallos':>7} {'Push':>5} {'Neutro':>7} {'Acierto%':>9} {'Tipo':>16}")
    totales = {"juegos": 0, "pronosticados": 0, "aciertos": 0,
               "fallos": 0, "pushes": 0, "neutros": 0}
    for r in filas:
        tipo = "out-of-time" if r["anio"] > 2023 else "in-sample"
        print(f"  {r['anio']:<8} {r['juegos']:>7} {r['pronosticados']:>8} "
              f"{r['aciertos']:>9} {r['fallos']:>7} {r['pushes']:>5} "
              f"{r['neutros']:>7} {r['pct']:>8.1%} {tipo:>16}")
        for k in totales:
            totales[k] += r[k]
    for k in ("pronosticados", "aciertos", "fallos", "pushes", "neutros"):
        if k in ("aciertos", "fallos", "pushes", "neutros") and \
                k not in ("juegos",):
            pass
    t = totales
    pct_total = t["aciertos"] / (t["aciertos"] + t["fallos"]) \
        if (t["aciertos"] + t["fallos"]) else 0.0
    print(f"  {'TOTAL':<8} {t['juegos']:>7} {t['pronosticados']:>8} "
          f"{t['aciertos']:>9} {t['fallos']:>7} {t['pushes']:>5} "
          f"{t['neutros']:>7} {pct_total:>8.1%} {'--':>16}")
    print("=" * 88)
    print(f"  Acierto de TODOS los juegos (siempre se eligiera lado): "
          f"{pct_total:.1%}  |  Solo out-of-time (2024-2026): ", end="")
    oo = [r for r in filas if r["anio"] > 2023]
    a = sum(r["aciertos"] for r in oo)
    f = sum(r["fallos"] for r in oo)
    print(f"{a / (a + f) if (a + f) else 0:.1%}")
    print("  Baseline por azar: 50% (linea 8.5 fija) | Cuota 1.91: "
          "se gana con >52.4%")
    print("  NOTA: 2023 es in-sample (no hay datos previos); 2024-2026 "
          "son la prueba honesta.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())