# -*- coding: utf-8 -*-
"""Valida la mejora de calibracion + Kelly contra la politica anterior.

Esquema honesto, sin fuga:
  - Modelo entrenado con datos <= 2024.
  - Isotonica ajustada con predicciones de 2025 (periodo de calibracion).
  - Evaluacion: 2026 (out-of-time, no visto).

Politicas comparadas a cuota 1.91:
  - VIEJA (proxy): P(Over) cruda del XGBoost con umbrales 0.62/0.38
    (el blend real incluye la proyeccion heuristica, no replicable aqui;
    esta proxy TIENE la misma ventaja de seleccion, sin el ruido).
  - NUEVA: P calibrada con margen 0.055 + staking media Kelly.

Nota: sin lineas de mercado historicas, L = 8.5 (el ajuste a la linea real
solo afecta a produccion, donde Linea_Casino viene de The Odds API).
"""

import sys
from sklearn.isotonic import IsotonicRegression
import pandas as pd

from reporte_rendimiento import construir_df_modelo
import xgboost as xgb

CUOTA = 1.91
MARGEN_MIN = 0.055
LIMITE_OVER_VIEJO = 0.62
LIMITE_UNDER_VIEJO = 0.38


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/4] Construyendo features (pipeline de produccion)...")
    df = construir_df_modelo()
    ent = df[df["Fecha"].dt.year <= 2024]
    cal = df[df["Fecha"].dt.year == 2025]
    prb = df[df["Fecha"].dt.year >= 2026]
    print(f"      Entrenamiento <=2024: {len(ent)} | "
          f"Calibracion 2025: {len(cal)} | Prueba 2026: {len(prb)}")

    from entrenar_modelo import ajustar_transformadores, construir_caracteristicas_finales
    trans = ajustar_transformadores(ent)
    X_ent = construir_caracteristicas_finales(ent, trans)
    X_cal = construir_caracteristicas_finales(cal, trans)
    X_prb = construir_caracteristicas_finales(prb, trans)
    y_ent = ent["Target_Over"].values
    y_cal = cal["Target_Over"].values
    y_prb = prb["Target_Over"].values
    total_prb = prb["Total_Carreras"].values

    print("[2/4] Entrenando XGBoost (<=2024)...")
    modelo = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_estimators=600, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5)
    modelo.fit(X_ent, y_ent)
    p_cal_raw = modelo.predict_proba(X_cal)[:, 1]
    p_prb_raw = modelo.predict_proba(X_prb)[:, 1]

    print("[3/4] Ajustando isotonica con 2025 y aplicando a 2026...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal)
    p_prb_cal = pd.Series(iso.predict(p_prb_raw).clip(1e-4, 1 - 1e-4))
    y_s = pd.Series(y_prb)
    t_s = pd.Series(total_prb)

    def sim(p, umbral_over, umbral_under, apuesta_1u=True):
        over = p >= umbral_over
        under = p <= umbral_under
        n = int(over.sum()) + int(under.sum())
        if n == 0:
            return (0, 0, 0.0, 0.0)
        g_ov = int((t_s[over] > 8.5).sum())
        g_un = int((t_s[under] < 8.5).sum())
        aciertos = g_ov + g_un
        roi = (aciertos * (CUOTA - 1.0) - (n - aciertos)) / n
        return (n, aciertos, aciertos / n, roi)

    def sim_kelly(p):
        """Media Kelly igual que en produccion: para UNDER usa 1 - P(OVER)."""
        over = p >= 0.5 + MARGEN_MIN
        under = p <= 0.5 - MARGEN_MIN
        n = int(over.sum()) + int(under.sum())
        if n == 0:
            return (0, 0, 0.0, 0.0)
        g_ov = (t_s[over] > 8.5)
        g_un = (t_s[under] < 8.5)
        b = CUOTA - 1.0
        pv = pd.concat([p[over], p[under].map(lambda x: 1.0 - x)])
        gv = pd.concat([g_ov, g_un]).astype(int).values
        f = ((pv * b - (1.0 - pv)) / b).clip(lower=0.0).values * 0.5
        unidades = float(f.sum())
        ganancia = float((f * gv).sum() * b)
        perdida = float((f * (1 - gv)).sum())
        return (n, int(gv.sum()), float(gv.mean()),
                (ganancia - perdida) / unidades if unidades > 0 else 0.0)

    print("[4/4] Comparacion sobre 2026 (out-of-time)...")
    print("")
    print(f"  {'Politica':<42} {'N':>5} {'Acierto':>8} {'ROI':>8}")
    n, ac, hr, roi = sim(p_prb_raw, LIMITE_OVER_VIEJO, LIMITE_UNDER_VIEJO)
    print(f"  VIEJA (umbrales crudos 0.62/0.38)     {n:>5} "
          f"{hr:>8.3f} {roi:>+8.3f}")
    n2, ac2, hr2, roi2 = sim_kelly(p_prb_cal)
    print(f"  NUEVA (calibrada margen 0.055 + Kelly){n2:>5} "
          f"{hr2:>8.3f} {roi2:>+8.3f}")
    print("")
    if roi2 > roi and n2 >= 5:
        print("VEREDICTO: la politica nueva supera a la vieja en "
              "out-of-time (ROI unitario y Kelly).")
    else:
        print("VEREDICTO: sin ventaja clara de la nueva politica en 2026; "
              "revisar margen/umbrales.")
    print("")

    print("CALIBRACION de la probabilidad nueva (2026):")
    print(f"  {'Rango P':<12} {'N':>5} {'P media':>9} {'OVER real':>10}")
    for lo, hi in ((0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
                   (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)):
        m = p_prb_cal.ge(lo) & p_prb_cal.lt(hi)
        if m.sum() == 0:
            continue
        print(f"  {lo:.2f}-{hi:.2f}      {int(m.sum()):>5} "
              f"{p_prb_cal[m].mean():>9.3f} {y_s[m].mean():>10.3f}")
    print("")
    print("NOTA: L = 8.5 (sin historial de lineas de mercado). En "
          "produccion la probabilidad se traslada a la linea real del dia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())