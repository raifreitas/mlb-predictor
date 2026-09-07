"""Entrenamiento XGBoost (logloss) para Batter Props con Walk-Forward.

Predice dos targets binarios por aparicion al plato:
  - target_strikeout : P(el bateador se poncha >= 1 vez en el juego)
  - target_walk      : P(el bateador recibe >= 1 base por bolas)

Validacion WALK-FORWARD (cronologica, sin split estatico 80/20):
  Para cada plegue se entrena con TODO el pasado (ventana expansiva) y se
  evalua exclusivamente el HORIZONTE_DIAS siguiente; la ventana avanza
  hasta la fecha actual. El early stopping usa el tramo final del propio
  entrenamiento (nunca datos de prueba).

Hiperparametros orientados a minimizar LogLoss y a controlar el sobreajuste
(min_child_weight y regularizacion L1/L2 altos, learning_rate bajo).

Uso:
  python entrenar_batter_props.py --db ..\\..\\..\\data\\mlb.db --desde 2023-04-01
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from features_batter_props import FEATURES, TARGETS, construir_features, leer_datos

MODELOS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
RUTA_MODELO_SO = os.path.join(MODELOS_DIR, "batter_props_strikeout.pkl")
RUTA_MODELO_BB = os.path.join(MODELOS_DIR, "batter_props_walk.pkl")
RUTA_COLUMNAS = os.path.join(MODELOS_DIR, "columnas_batter_props.pkl")
RUTA_TRANSFORMADORES = os.path.join(MODELOS_DIR, "transformadores_batter_props.pkl")

# Control del sobreajuste: LogLoss es la metrica objetivo (binary:logistic).
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "gamma": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "n_estimators": 1500,
    "random_state": 42,
    "n_jobs": -1,
}
EARLY_STOPPING = 75
FRACCION_EVAL_INTERNO = 0.10

WARMUP_JUEGOS = 30000
MIN_JUEGOS_PRUEBA = 3000
MIN_PLEGUES = 6


def _imputar_medianas(X_entrenamiento, X_prueba):
    """Rellena NaN con la mediana calculada SOLO sobre entrenamiento."""
    medianas = X_entrenamiento.median()
    X_entrenamiento = X_entrenamiento.fillna(medianas)
    X_prueba = X_prueba.fillna(medianas)
    return X_entrenamiento, X_prueba, medianas.to_dict()


def _division_interna_train_eval(df, mascara_entrenamiento):
    """Ultimo 10% del entrenamiento (por fecha) como set interno de early stop."""
    fechas_tr = sorted(df.loc[mascara_entrenamiento, "Fecha"].unique())
    if not fechas_tr:
        return mascara_entrenamiento, np.zeros(len(df), dtype=bool)
    indice = max(0, int(len(fechas_tr) * (1 - FRACCION_EVAL_INTERNO)) - 1)
    corte = fechas_tr[indice]
    fit_mask = mascara_entrenamiento & (df["Fecha"] < corte)
    eval_mask = mascara_entrenamiento & (df["Fecha"] >= corte)
    return fit_mask, eval_mask


def _entrenar_modelo(df, target, mascara_entrenamiento):
    fit_mask, eval_mask = _division_interna_train_eval(df, mascara_entrenamiento)
    X_fit, y_fit = df.loc[fit_mask, FEATURES], df.loc[fit_mask, target]
    X_eval, y_eval = df.loc[eval_mask, FEATURES], df.loc[eval_mask, target]
    X_fit, X_eval, _ = _imputar_medianas(X_fit, X_eval)
    modelo = xgb.XGBClassifier(**PARAMS, early_stopping_rounds=EARLY_STOPPING)
    modelo.fit(X_fit, y_fit,
               eval_set=[(X_eval, y_eval)],
               verbose=False)
    return modelo


def _walk_forward(df, target, horizonte_dias, min_plegues):
    fechas = sorted(df["Fecha"].dt.floor("D").unique())
    plegues = 0
    filas_test = []   # (fold, fecha, y, p)
    resumen = []
    indice = 0
    while indice < len(fechas):
        corte = fechas[indice]
        fin = corte + pd.Timedelta(days=horizonte_dias)
        mask_tr = df["Fecha"] < corte
        mask_te = (df["Fecha"] >= corte) & (df["Fecha"] < fin)
        n_tr = int(mask_tr.sum())
        n_te = int(mask_te.sum())
        if n_tr < WARMUP_JUEGOS or n_te < MIN_JUEGOS_PRUEBA:
            indice += 1
            continue

        modelo = _entrenar_modelo(df, target, mask_tr)
        X_te = df.loc[mask_te, FEATURES]
        y_te = df.loc[mask_te, target]
        _, X_te, _ = _imputar_medianas(
            df.loc[mask_tr, FEATURES].copy(), X_te.copy())
        p = modelo.predict_proba(X_te)[:, 1]

        tasa_tr = float(df.loc[mask_tr, target].mean())
        baseline = log_loss(y_te, np.full(len(y_te), tasa_tr),
                            labels=[0, 1])
        loss = log_loss(y_te, p, labels=[0, 1])
        auc = (roc_auc_score(y_te, p)
               if len(np.unique(y_te)) > 1 else float("nan"))
        resumen.append({
            "plegue": plegues + 1,
            "corte": str(pd.Timestamp(corte).date()),
            "juegos_train": n_tr,
            "juegos_test": n_te,
            "tasa_base_train": tasa_tr,
            "logloss": loss,
            "logloss_baseline": baseline,
            "auc": auc,
            "iteraciones": getattr(modelo, "best_iteration", None),
        })
        filas_test.append(pd.DataFrame({
            "fold": plegues + 1,
            "Fecha": df.loc[mask_te, "Fecha"].values,
            "y": y_te.values,
            "p": p,
        }))
        plegues += 1
        indice += horizonte_dias

    if not filas_test:
        return pd.DataFrame(), pd.DataFrame(resumen), 0
    return pd.concat(filas_test, ignore_index=True), pd.DataFrame(resumen), plegues


def _entrenar_final(df, target):
    """Modelo final con TODO el historico (eval interno = ultimo 10%)."""
    mask_total = pd.Series(True, index=df.index)
    return _entrenar_modelo(df, target, mask_total)


def _imprimir_resumen(resumen, filas_test, target):
    if filas_test.empty:
        print(f"  [{target}] Sin plegues suficientes (min requerido: "
              f"{MIN_PLEGUES}). Revisa WARMUP_JUEGOS/MIN_JUEGOS_PRUEBA.")
        return
    n = len(filas_test)
    loss = log_loss(filas_test["y"], filas_test["p"], labels=[0, 1])
    baseline = log_loss(filas_test["y"],
                        np.full(n, filas_test["y"].mean()), labels=[0, 1])
    auc = roc_auc_score(filas_test["y"], filas_test["p"])
    acc = accuracy_score(filas_test["y"], (filas_test["p"] >= 0.5).astype(int))
    tasa = filas_test["y"].mean()
    print(f"\n=== {target} | Walk-Forward ({len(resumen)} plegues) ===")
    print(f"  Filas evaluadas: {n} | Tasa base: {tasa:.4f}")
    print(f"  LogLoss modelo : {loss:.4f} | LogLoss baseline (tasa): {baseline:.4f}")
    print(f"  Mejora logloss : {baseline - loss:+.4f}")
    print(f"  ROC AUC        : {auc:.4f}")
    print(f"  Accuracy@0.5   : {acc:.4f}")
    por_plegue = resumen.copy()
    print("  Por plegue:")
    for _, r in resumen.iterrows():
        print(f"    #{int(r['plegue']):>2} corte={r['corte']} "
              f"train={int(r['juegos_train']):>6} test={int(r['juegos_test']):>5} "
              f"ll={r['logloss']:.4f} (base {r['logloss_baseline']:.4f}) "
              f"auc={r['auc']:.4f}")


def _aplicar_config(args):
    global WARMUP_JUEGOS, MIN_JUEGOS_PRUEBA, MIN_PLEGUES
    WARMUP_JUEGOS = args.warmup_juegos
    MIN_JUEGOS_PRUEBA = args.min_juegos_prueba
    MIN_PLEGUES = args.min_plegues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=r"..\..\..\data\mlb.db")
    parser.add_argument("--desde", default="2023-04-01",
                        help="fecha minima de muestras (Y-m-d)")
    parser.add_argument("--horizonte-dias", type=int, default=7)
    parser.add_argument("--min-plegues", type=int, default=MIN_PLEGUES)
    parser.add_argument("--warmup-juegos", type=int, default=WARMUP_JUEGOS)
    parser.add_argument("--min-juegos-prueba", type=int, default=MIN_JUEGOS_PRUEBA)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    _aplicar_config(args)

    print("[1/4] Cargando crudo de la BD...")
    bateadores, abridores, manos = leer_datos(args.db)
    print(f"      {len(bateadores)} apariciones | {len(abridores)} "
          f"salidas de abridores | {len(manos)} manos.")

    print("[2/4] Feature engineering (sin fuga: shift+ventana 30d por entidad)...")
    df = construir_features(bateadores, abridores, manos)
    df = df[df["Fecha"] >= pd.Timestamp(args.desde)].copy()
    print(f"      {len(df)} filas utiles desde {args.desde} "
          f"({len(FEATURES)} features, {len(TARGETS)} targets).")
    for t in TARGETS:
        print(f"      Tasa base {t}: {df[t].mean():.4f}")

    resultados_globales = {}
    print("[3/4] Walk-Forward: ventana expansiva, evaluacion 'dia a dia'...")
    for target in TARGETS:
        filas_test, resumen, n = _walk_forward(
            df, target, args.horizonte_dias, args.min_plegues)
        _imprimir_resumen(resumen, filas_test, target)
        resultados_globales[target] = {
            "filas_test": filas_test, "resumen": resumen, "n_plegues": n}

    print("[4/4] Entrenando modelos finales con TODO el historico...")
    os.makedirs(MODELOS_DIR, exist_ok=True)
    medianas_finales = {}
    modelos = {}
    for target, ruta in ((TARGETS[0], RUTA_MODELO_SO),
                         (TARGETS[1], RUTA_MODELO_BB)):
        modelo = _entrenar_final(df, target)
        X_all = df[FEATURES].copy()
        medianas = X_all.median()
        X_all = X_all.fillna(medianas)
        # Verificacion de cierre (sin test dedicado: predecir train para sanity).
        p = modelo.predict_proba(X_all)[:, 1]
        print(f"      {target}: train logloss {log_loss(df[target], p):.4f} "
              f"(early stop iters {getattr(modelo, 'best_iteration', 'max')})")
        joblib.dump(modelo, ruta)
        modelos[target] = modelo
        medianas_finales[target] = medianas.to_dict()

    transformadores = {
        "features": FEATURES,
        "targets": TARGETS,
        "medianas": medianas_finales,
        "params": PARAMS,
        "horizonte_dias": args.horizonte_dias,
        "walk_forward": {
            t: {"n_plegues": v["n_plegues"], "resumen": v["resumen"].to_dict("records")}
            for t, v in resultados_globales.items()},
    }
    joblib.dump(FEATURES, RUTA_COLUMNAS)
    joblib.dump(transformadores, RUTA_TRANSFORMADORES)
    print(f"\nGuardados: {RUTA_MODELO_SO}, {RUTA_MODELO_BB}, "
          f"{RUTA_COLUMNAS}, {RUTA_TRANSFORMADORES}")


if __name__ == "__main__":
    raise SystemExit(main())