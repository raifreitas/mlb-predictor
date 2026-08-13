# -*- coding: utf-8 -*-
"""PRUEBA TRAMPA: entrenar UNA vez con TODOS los anos (2023-2026) y
pronosticar los mismos partidos (100% in-sample).

El proposito es demostrativo: mostrar lo que NUNCA se debe usar como
evidencia. El modelo ya "vio" los resultados de cada partido durante el
entrenamiento, asi que el acierto sale inflado en TODOS los anos.
"""

import sys
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from reporte_rendimiento import construir_df_modelo
from entrenar_modelo import ajustar_transformadores, construir_caracteristicas_finales

MARGEN = 0.055
LIMITE = 8.5


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Construyendo features (pipeline de produccion)...")
    df = construir_df_modelo()

    print("[2/3] Entrenando UNA vez con TODOS los partidos 2023-2026 "
          "(in-sample total)...")
    trans = ajustar_transformadores(df)
    X = construir_caracteristicas_finales(df, trans)
    modelo = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_estimators=600, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5)
    modelo.fit(X, df["Target_Over"].values)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(modelo.predict_proba(X)[:, 1], df["Target_Over"].values)
    p = iso.predict(modelo.predict_proba(X)[:, 1]).clip(1e-4, 1 - 1e-4)

    pronostico = pd.Series("NEUTRO", index=df.index)
    pronostico.iloc[p >= 0.5 + MARGEN] = "OVER"
    pronostico.iloc[p <= 0.5 - MARGEN] = "UNDER"
    total = df["Total_Carreras"].values

    res = pd.DataFrame({"pronostico": pronostico.values,
                        "total": total, "anio": df["Fecha"].dt.year.values})
    res["resultado"] = "PUSH"
    res.loc[res["total"] > LIMITE, "resultado"] = "OVER"
    res.loc[res["total"] < LIMITE, "resultado"] = "UNDER"

    print("[3/3] Reporte (TRAMPA / in-sample total)...")
    print("")
    print("=" * 78)
    print("PRUEBA TRAMPA: entrenado con TODO, pronosticado TODO")
    print("=" * 78)
    print(f"  {'Anio':<8} {'Juegos':>7} {'Pronost':>8} {'Aciertos':>9} "
          f"{'Fallos':>7} {'Neutro':>7} {'Acierto%':>9}")
    t_p, t_a, t_f = 0, 0, 0
    for anio, sub in res.groupby("anio"):
        pron = sub[sub["pronostico"].isin(["OVER", "UNDER"])]
        a = int(((pron["pronostico"] == "OVER") & (pron["resultado"] == "OVER")).sum()
                + ((pron["pronostico"] == "UNDER") & (pron["resultado"] == "UNDER")).sum())
        f = int(((pron["pronostico"] == "OVER") & (pron["resultado"] == "UNDER")).sum()
                + ((pron["pronostico"] == "UNDER") & (pron["resultado"] == "OVER")).sum())
        t_p += len(pron)
        t_a += a
        t_f += f
        print(f"  {anio:<8} {len(sub):>7} {len(pron):>8} {a:>9} {f:>7} "
              f"{int((sub['pronostico'] == 'NEUTRO').sum()):>7} "
              f"{a / (a + f) if (a + f) else 0:.1%}")
    print(f"  {'TOTAL':<8} {len(res):>7} {t_p:>8} {t_a:>9} {t_f:>7} "
          f"{int((res['pronostico'] == 'NEUTRO').sum()):>7} "
          f"{t_a / (t_a + t_f) if (t_a + t_f) else 0:.1%}")
    print("=" * 78)
    print("  Este numero NO tiene valor predictivo: el modelo conocia")
    print("  los resultados de TODOS estos partidos durante el")
    print("  entrenamiento. La prueba honesta sigue siendo 52.8%")
    print("  (walk-forward 2024-2026, sin ver el futuro).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())