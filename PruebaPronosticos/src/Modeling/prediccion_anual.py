# -*- coding: utf-8 -*-
"""Prediccion Monte Carlo de UNA temporada completa (1 anio de apuestas).

Base honesta: las 491 apuestas reales producidas por la politica
calibrada+Kelly sobre 2026 out-of-time (entrena <=2024, calibra 2025,
prueba 2026). Se bootstrapa esa biblioteca de apuestas con sus
RESULTADOS REALES para simular temporadas completas (~550 apuestas):

  - Banquillo inicial: 100 unidades.
  - Cada apuesta arriesga su fraccion Kelly (media Kelly, cuota real 1.91).
  - Salidas: percentiles del banquillo, P(acabar positivo), P(>=+20u),
    unidades netas esperadas y mes a mes.

Advertencia: asume que el edge medido (ROI ~+30% sobre capital) persiste.
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from reporte_rendimiento import construir_df_modelo
from entrenar_modelo import ajustar_transformadores, construir_caracteristicas_finales

CUOTA = 1.91
BANQUILLO_INICIAL = 100.0
MARGEN_MIN = 0.055
BET_POR_DIA = 3.0
DIAS_TEMPORADA = 183
RNG = np.random.default_rng(42)


def construir_biblioteca():
    df = construir_df_modelo()
    ent = df[df["Fecha"].dt.year <= 2024]
    cal = df[df["Fecha"].dt.year == 2025]
    prb = df[df["Fecha"].dt.year >= 2026]

    trans = ajustar_transformadores(ent)
    X_ent = construir_caracteristicas_finales(ent, trans)
    X_cal = construir_caracteristicas_finales(cal, trans)
    X_prb = construir_caracteristicas_finales(prb, trans)

    modelo = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_estimators=600, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5)
    modelo.fit(X_ent, ent["Target_Over"].values)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(modelo.predict_proba(X_cal)[:, 1], cal["Target_Over"].values)
    p = pd.Series(iso.predict(modelo.predict_proba(X_prb)[:, 1])
                  .clip(1e-4, 1 - 1e-4))
    total = prb["Total_Carreras"].values

    over = p >= 0.5 + MARGEN_MIN
    under = p <= 0.5 - MARGEN_MIN
    mask_ov = over.values
    mask_un = under.values
    sel = mask_ov | mask_un
    tot = prb["Total_Carreras"].values[sel]
    lado_over = mask_ov[sel]
    pv = p.values[sel]
    ganador = ((tot > 8.5) & lado_over) | ((tot < 8.5) & ~lado_over)

    b = CUOTA - 1.0
    f = ((pv * b - (1.0 - pv)) / b).clip(min=0.0) * 0.5
    return pd.DataFrame({
        "f": f,
        "ganador": ganador.astype(int),
        "p": pv,
        "lado_over": lado_over,
    })


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Construyendo biblioteca de apuestas (OOT 2026)...")
    lib = construir_biblioteca()
    n_bets = int(round(BET_POR_DIA * DIAS_TEMPORADA))
    print(f"      {len(lib)} apuestas reales 2026 | temporada simulada: "
          f"{n_bets} apuestas ({BET_POR_DIA}/dia)")

    print("[2/3] Simulando 10000 temporadas (bootstrap con resultados reales)...")
    n_sim = 10000
    prof_por_bet = lib["f"].values * (CUOTA - 1.0) * lib["ganador"].values \
        - lib["f"].values * (1 - lib["ganador"].values)
    saldos = np.empty((n_sim, n_bets + 1))
    for i in range(n_sim):
        idx = RNG.integers(0, len(lib), size=n_bets)
        saldos[i, 0] = BANQUILLO_INICIAL
        saldos[i, 1:] = BANQUILLO_INICIAL + np.cumsum(prof_por_bet[idx])
    finales = saldos[:, -1]

    print("[3/3] Reporte de prediccion anual...")
    lineas = []
    def r(texto):
        print(texto)
        lineas.append(texto)

    r("")
    r("=" * 74)
    r("PREDICCION DE 1 ANIO (temporada completa, banquillo 100u)")
    r("=" * 74)
    r(f"  Apuestas por temporada        : {n_bets}")
    r(f"  Cuota media asumida           : {CUOTA:.2f} (1.91)")
    r(f"  Hit rate de la biblioteca     : {lib['ganador'].mean():.3f}")
    r(f"  ROI por unidad de la biblioteca: "
      f"{(lib['ganador'].mean() * (CUOTA - 1.0) - (1 - lib['ganador'].mean())):+.3f}")
    r("")
    r("  RESULTADO FINAL (100u de banquillo):")
    r(f"    Mediana del banquillo        : {np.median(finales):>7.1f} u "
      f"({np.median(finales) - 100:+.1f})")
    r(f"    P25 / P75                    : {np.percentile(finales, 25):>7.1f} / "
      f"{np.percentile(finales, 75):>7.1f} u")
    r(f"    P05 / P95                    : {np.percentile(finales, 5):>7.1f} / "
      f"{np.percentile(finales, 95):>7.1f} u")
    r(f"    P(acabar positivo)           : {(finales > 100).mean():.1%}")
    r(f"    P(ganar >= 20u)              : {(finales >= 120).mean():.1%}")
    r(f"    P(ganar >= 50u)              : {(finales >= 150).mean():.1%}")
    r(f"    P(perder mas de 20u)         : {(finales < 80).mean():.1%}")
    r("")
    r("  EVOLUCION MES A MES (mediana del banquillo, 30 dias/mes):")
    r(f"    {'Mes':<6} {'Mediana':>8} {'P05':>8} {'P95':>8}")
    pasos_mes = [min(int(round(BET_POR_DIA * 30 * m)), n_bets)
                 for m in range(1, 7)]
    for m in range(1, 7):
        paso = pasos_mes[m - 1]
        r(f"    {m:<6} {np.median(saldos[:, paso]):>8.1f} "
          f"{np.percentile(saldos[:, paso], 5):>8.1f} "
          f"{np.percentile(saldos[:, paso], 95):>8.1f}")
    r("")
    r("  NOTA: asume que el edge OOT 2026 persiste y cuota plana 1.91;")
    r("  el CLV real contra la linea de cierre se mide en vivo desde")
    r("  2026-08-04 con las cuotas acumuladas.")
    r("=" * 74)

    import os
    os.makedirs("output_predicciones", exist_ok=True)
    with open("output_predicciones/prediccion_anual.txt", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print("\nReporte guardado en output_predicciones/prediccion_anual.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())