# -*- coding: utf-8 -*-
"""Calibra los parametros de produccion del enfoque MONEYLINE POR
DIFERENCIA DE CARRERAS (Skellam/normal) sobre todo el historico.

Replica el regimen del backtest walk-forward pero en modo produccion:
  - Beta de la sigmoide: minimiza log-loss sobre el 80% mas antiguo.
  - Calibracion isotonica: sobre el 20% final (igual que entrenar_modelo).
  - Guarda models/beta_skellam_ml.pkl y models/calibracion_skellam_ml.pkl
    que usa predecir_ml.py en produccion.

Uso:
    python calibrar_skellam_produccion.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from backtest_skellam_ml import (
    aplicar_ajustes_por_lado,
    calibrar_beta,
    construir_df,
    prob_gana_local,
)

MODELOS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
BETA_PATH = os.path.join(MODELOS_DIR, "beta_skellam_ml.pkl")
CALIBRACION_PATH = os.path.join(MODELOS_DIR, "calibracion_skellam_ml.pkl")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Construyendo features (pipeline de carreras por equipo)...")
    df = construir_df()
    print(f"      {len(df)} partidos finales "
          f"({df['Fecha'].dt.year.min()} - {df['Fecha'].dt.year.max()}).")

    print("[2/3] Division cronologica 80/20 y calibracion...")
    indice_corte = int(len(df) * 0.8)
    train = df.iloc[:indice_corte]
    cal = df.iloc[indice_corte:]

    beta = calibrar_beta(train)
    print(f"      Beta optimo (log-loss sobre entrenamiento): {beta:.2f}")

    p_cal = prob_gana_local(cal["ExpRunsLocal"], cal["ExpRunsVisita"], beta)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal.values, cal["Target_GanaLocal"].values)

    p_cal_aj = iso.predict(p_cal.values)
    ll_antes = -np.mean(
        cal["Target_GanaLocal"].values * np.log(np.clip(p_cal, 1e-9, 1))
        + (1 - cal["Target_GanaLocal"].values) * np.log(np.clip(1 - p_cal, 1e-9, 1)))
    ll_despues = -np.mean(
        cal["Target_GanaLocal"].values * np.log(np.clip(p_cal_aj, 1e-9, 1))
        + (1 - cal["Target_GanaLocal"].values) * np.log(np.clip(1 - p_cal_aj, 1e-9, 1)))
    print(f"      Log-loss en calibracion: antes {ll_antes:.4f} -> "
          f"despues {ll_despues:.4f}")
    acierto_bruto = np.mean((p_cal > 0.5) == cal["Target_GanaLocal"].values)
    print(f"      Acierto bruto pre-calibracion (P>0.5): {acierto_bruto:.1%}")

    print("[3/3] Guardando parametros de produccion...")
    os.makedirs(MODELOS_DIR, exist_ok=True)
    joblib.dump(float(beta), BETA_PATH)
    joblib.dump(iso, CALIBRACION_PATH)
    print(f"      Beta guardado: {BETA_PATH}")
    print(f"      Calibracion guardada: {CALIBRACION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
