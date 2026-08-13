"""Motor de decision para el mercado Moneyline (ganador del partido).

Mercado AISLADO del flujo Over/Under: usa el modelo XGBoost entrenado
por entrenar_modelo_ml.py (target: gana el equipo local) y replica la
filosofia de riesgo del mercado de totals:

- Probabilidad calibrada (isotonica) de victoria LOCAL.
- Probabilidad implicita del mercado h2h (sin vig) desde las cuotas.
- Regresion al mercado (cambio #4): la prob final mezcla modelo y
  mercado; el peso del mercado crece con el desacuerdo.
- Tope de edge: desacuerdo extremo modelo vs mercado se anula.
- Stake por media Kelly con la cuota real del pick; 0.5u estandar.
- Tope a 0.5u si faltan datos de matchup LHP/RHP (>= 2 OPS splits).

Uso:
    python predecir_ml.py [--fecha YYYY-MM-DD] [--inicio X] [--fin Y]
"""

import io
import os
import re
import sys
from datetime import date, datetime

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from entrenar_modelo import obtener_driver_odbc
from entrenar_modelo_ml import (
    CALIBRACION_PATH,
    COLUMNAS_PATH,
    MODELO_PATH,
    TRANSFORMADORES_PATH,
    construir_caracteristicas_finales,
    feature_engineering_bullpen,
    feature_engineering_descanso_abridor,
    feature_engineering_fatiga,
    feature_engineering_pitchers,
    feature_engineering_rachas,
    preprocesar,
)
from entrenar_modelo import CONNECTION_STRING_TEMPLATE

MODELOS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
CARPETA_REPORTE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output_predicciones"))
FECHA_INICIO_DEFECTO = date(2026, 7, 1)
FECHA_FIN_DEFECTO = date(2026, 7, 31)

# Constantes del mercado Moneyline (sufijo ML: aisladas de Totals).
# Ajustadas con el backtest por diferencia de carreras (Skellam):
# margen mas exigente y regresion al mercado mas fuerte, porque el
# mercado h2h es mucho mas eficiente que el de totals.
CUOTA_PROXY_ML = 1.91
MARGEN_MIN_PROB_ML = 0.07
EDGE_MINIMO_ML = 0.055
EDGE_MAXIMO_ML = 0.25
PESO_MERCADO_MAX_ML = 0.25
DESACUERDO_MERCADO_REF_ML = 0.20
LIMITE_STAKE_ALTO_ML = 0.08

# Parametros del enfoque por diferencia de carreras (Skellam/normal).
# Se calibran con calibrar_skellam_produccion.py.
BETA_SKELLAM_PATH = os.path.join(MODELOS_DIR, "beta_skellam_ml.pkl")
CALIBRACION_SKELLAM_PATH = os.path.join(MODELOS_DIR, "calibracion_skellam_ml.pkl")

SUGERENCIA_HOME = "APOSTAR LOCAL"
SUGERENCIA_AWAY = "APOSTAR VISITA"
SUGERENCIA_NO_BET_ML = "NO APOSTAR"

_calibrador_cache = None


_beta_skellam_cache = None
_calibrador_skellam_cache = None


def cargar_beta_skellam():
    """Beta de la sigmoide del enfoque por diferencia de carreras."""
    global _beta_skellam_cache
    if _beta_skellam_cache is None:
        if os.path.exists(BETA_SKELLAM_PATH):
            try:
                _beta_skellam_cache = float(joblib.load(BETA_SKELLAM_PATH))
            except Exception:
                _beta_skellam_cache = 0.10
        else:
            _beta_skellam_cache = 0.10
    return _beta_skellam_cache


def cargar_calibrador_skellam():
    """Isotonica del enfoque por diferencia de carreras."""
    global _calibrador_skellam_cache
    if _calibrador_skellam_cache is None:
        if os.path.exists(CALIBRACION_SKELLAM_PATH):
            try:
                _calibrador_skellam_cache = joblib.load(CALIBRACION_SKELLAM_PATH)
            except Exception:
                _calibrador_skellam_cache = False
        else:
            _calibrador_skellam_cache = False
    return _calibrador_skellam_cache or None


def probabilidad_skellam_ml(exp_local, exp_visita):
    """P(gana local) por DIFERENCIA DE CARRERAS (enfoque Skellam).

    sigmoid(beta * (exp_local - exp_visita)) con beta calibrado y
    recalibracion isotonica sobre el 20% final del historico.
    """
    from backtest_skellam_ml import prob_gana_local

    beta = cargar_beta_skellam()
    p = prob_gana_local(exp_local, exp_visita, beta)
    iso = cargar_calibrador_skellam()
    if isinstance(p, pd.Series):
        if iso is not None:
            p = pd.Series(iso.predict(p.values), index=p.index)
        return p.clip(1e-4, 1 - 1e-4)
    p = float(p)
    if iso is not None:
        p = float(iso.predict([p])[0])
    return min(max(p, 1e-4), 1 - 1e-4)


def cargar_calibrador():
    global _calibrador_cache
    if _calibrador_cache is None:
        if os.path.exists(CALIBRACION_PATH):
            try:
                _calibrador_cache = joblib.load(CALIBRACION_PATH)
            except Exception:
                _calibrador_cache = False
        else:
            _calibrador_cache = False
    return _calibrador_cache or None


def probabilidad_calibrada_ml(prob_local):
    """P(gana local) calibrada con la isotonica del modelo ML."""
    p = float(prob_local)
    calibrador = cargar_calibrador()
    if calibrador is not None:
        p = float(calibrador.predict([p])[0])
    return min(max(p, 1e-4), 1 - 1e-4)


def probabilidad_mercado_implicita_h2h(cuota_home, cuota_away):
    """P(gana local) implicita del mercado (sin vig) desde cuotas h2h."""
    try:
        ch = float(cuota_home)
        ca = float(cuota_away)
    except (TypeError, ValueError):
        return None
    if pd.isna(ch) or pd.isna(ca) or ch <= 1.0 or ca <= 1.0:
        return None
    p = (1.0 / ch) / (1.0 / ch + 1.0 / ca)
    return min(max(p, 1e-4), 1 - 1e-4)


def recomendar_stake_kelly_ml(p, cuota):
    """Stake por media Kelly con la probabilidad calibrada y la cuota real.

    f* = (p * b - (1 - p)) / b, con b = cuota - 1; media Kelly.
      f* >= LIMITE_STAKE_ALTO_ML  -> 1.0u (Alto Valor)
      <  LIMITE_STAKE_ALTO_ML    -> 0.5u (Estandar)
    """
    if cuota is None or pd.isna(cuota) or float(cuota) <= 1.0:
        cuota = CUOTA_PROXY_ML
    b = float(cuota) - 1.0
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    f = max((p * b - (1.0 - p)) / b, 0.0) * 0.5
    if f >= LIMITE_STAKE_ALTO_ML:
        return 1.0
    return 0.5


def decidir_jugada_ml(fila, prob_local, cuota_home, cuota_away,
                      decodificadores, ya_calibrada=False):
    """Logica completa de decision Moneyline sobre UNA fila enriquecida.

    Devuelve dict con local, visita, prob_local (cruda), prob_cal
    (calibrada), p_mercado, peso_mercado, prob_decision, sugerencia,
    motivo_anulacion, stake, tope_edge y datos_faltantes_cap.

    ya_calibrada=True: prob_local ya incluye la recalibracion isotonica
    del motor (enfoque por diferencia de carreras).
    """
    local = decodificadores["EquipoLocal"].inverse_transform(
        [fila["EquipoLocal"]])[0]
    visita = decodificadores["EquipoVisita"].inverse_transform(
        [fila["EquipoVisita"]])[0]

    prob_cal = float(prob_local) if ya_calibrada \
        else probabilidad_calibrada_ml(prob_local)

    p_mercado = probabilidad_mercado_implicita_h2h(cuota_home, cuota_away)
    if p_mercado is None:
        p_mercado = probabilidad_mercado_implicita_h2h(
            CUOTA_PROXY_ML, CUOTA_PROXY_ML)

    desacuerdo = abs(prob_cal - p_mercado)
    peso_mercado = min(
        PESO_MERCADO_MAX_ML,
        desacuerdo * PESO_MERCADO_MAX_ML / DESACUERDO_MERCADO_REF_ML)
    prob_decision = min(max(
        (1.0 - peso_mercado) * prob_cal + peso_mercado * p_mercado,
        0.0), 1.0)

    if prob_decision >= 0.5 + MARGEN_MIN_PROB_ML:
        sugerencia = SUGERENCIA_HOME
    elif prob_decision <= 0.5 - MARGEN_MIN_PROB_ML:
        sugerencia = SUGERENCIA_AWAY
    else:
        sugerencia = SUGERENCIA_NO_BET_ML

    motivo_anulacion = None
    tope_edge = False

    if sugerencia != SUGERENCIA_NO_BET_ML and desacuerdo < EDGE_MINIMO_ML:
        sugerencia = SUGERENCIA_NO_BET_ML
        motivo_anulacion = (f"Anulado por Margen Insuficiente "
                            f"(edge vs mercado {desacuerdo:.3f} < "
                            f"{EDGE_MINIMO_ML:.3f})")

    if sugerencia != SUGERENCIA_NO_BET_ML and desacuerdo > EDGE_MAXIMO_ML:
        sugerencia = SUGERENCIA_NO_BET_ML
        tope_edge = True
        motivo_anulacion = (f"Anulado por Edge Excesivo vs Mercado "
                            f"(desacuerdo {desacuerdo:.3f} > "
                            f"{EDGE_MAXIMO_ML:.3f})")

    cuota_pick = cuota_home if sugerencia == SUGERENCIA_HOME else cuota_away
    stake = recomendar_stake_kelly_ml(prob_decision, cuota_pick) \
        if sugerencia != SUGERENCIA_NO_BET_ML else None

    # Filtro de valor (break-even): la cuota corta exige una probabilidad
    # alta; si la prob final no supera 1/cuota, el EV es negativo aunque
    # la direccion coincida con el mercado. Sin cuota valida se usa el
    # fallback (1.91 -> break-even 0.524).
    if sugerencia != SUGERENCIA_NO_BET_ML:
        cuota_breakeven = float(cuota_pick if (cuota_pick is not None
                                               and not pd.isna(cuota_pick)
                                               and float(cuota_pick) > 1.0)
                                else CUOTA_PROXY_ML)
        if prob_decision <= 1.0 / cuota_breakeven:
            sugerencia = SUGERENCIA_NO_BET_ML
            stake = None
            motivo_anulacion = (f"Anulado por Valor Negativo (prob final "
                                f"{prob_decision:.3f} <= break-even "
                                f"{1.0 / cuota_breakeven:.3f} de cuota "
                                f"{cuota_breakeven:.2f})")

    datos_faltantes_cap = False
    if stake is not None and stake > 0.5:
        faltantes = sum(
            1 for columna in ("OPSvsLHP_Local", "OPSvsRHP_Local",
                              "OPSvsLHP_Visita", "OPSvsRHP_Visita")
            if pd.isna(fila.get(columna)))
        if faltantes >= 2:
            stake = 0.5
            datos_faltantes_cap = True

    return {
        "local": local,
        "visita": visita,
        "prob_local": float(prob_local),
        "prob_cal": prob_cal,        "p_mercado": p_mercado,
        "peso_mercado": peso_mercado,
        "prob_decision": prob_decision,
        "desacuerdo": desacuerdo,
        "sugerencia": sugerencia,
        "motivo_anulacion": motivo_anulacion,
        "stake": stake,
        "tope_edge": tope_edge,
        "datos_faltantes_cap": datos_faltantes_cap,
    }


def cargar_datos_ml():
    """GameLog completo con joins (sin filtrar carreras: permite partidos
    de HOY aun no finalizados concatenados por recomendar_ml.py)."""
    import pyodbc
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        consulta = ("SELECT g.*, PF.Factor_Carreras, "
                    "op1.OPSvsLHP AS OPSvsLHP_Local, op1.OPSvsRHP AS OPSvsRHP_Local, "
                    "op2.OPSvsLHP AS OPSvsLHP_Visita, op2.OPSvsRHP AS OPSvsRHP_Visita, "
                    "pm1.Mano AS ManoAbridorLocal, pm2.Mano AS ManoAbridorVisita, "
                    "fb1.Fatiga_Bullpen_3d AS Fatiga_Bullpen_3d_Local, "
                    "fb2.Fatiga_Bullpen_3d AS Fatiga_Bullpen_3d_Visita "
                    "FROM dbo.GameLog g "
                    "LEFT JOIN dbo.ParkFactors PF ON g.EquipoLocal = PF.EquipoLocal "
                    "LEFT JOIN dbo.TeamOPS_Handedness op1 "
                    "ON op1.Equipo = g.EquipoLocal AND op1.Temporada = YEAR(g.Fecha) "
                    "LEFT JOIN dbo.TeamOPS_Handedness op2 "
                    "ON op2.Equipo = g.EquipoVisita AND op2.Temporada = YEAR(g.Fecha) "
                    "LEFT JOIN dbo.PitcherMano pm1 ON pm1.PitcherId = g.PitcherLocalId "
                    "LEFT JOIN dbo.PitcherMano pm2 ON pm2.PitcherId = g.PitcherVisitaId "
                    "LEFT JOIN dbo.vwFatigaBullpen3d fb1 "
                    "ON fb1.Team = g.EquipoLocal AND fb1.Fecha = g.Fecha "
                    "LEFT JOIN dbo.vwFatigaBullpen3d fb2 "
                     "ON fb2.Team = g.EquipoVisita AND fb2.Fecha = g.Fecha")
        return pd.read_sql(consulta, conexion)
    finally:
        conexion.close()


class SalidaConsolidada:
    def __init__(self, consola):
        self._consola = consola
        self._buffer = io.StringIO()

    def write(self, texto):
        self._consola.write(texto)
        self._buffer.write(texto)
        return len(texto)

    def flush(self):
        self._consola.flush()

    def isatty(self):
        return False

    def contenido(self):
        return self._buffer.getvalue()


def fecha_desde_args(flag):
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            try:
                return date.fromisoformat(sys.argv[i + 1])
            except ValueError:
                print(f"Argumento invalido '{sys.argv[i + 1]}' para {flag}; "
                      f"se usara el valor por defecto.")
    return None


def siguiente_indice_reporte():
    if not os.path.isdir(CARPETA_REPORTE):
        return 1
    mayor = 0
    for nombre in os.listdir(CARPETA_REPORTE):
        coincidencia = re.match(r"reporte_ml_(\d+)_", nombre)
        if coincidencia:
            mayor = max(mayor, int(coincidencia.group(1)))
    return mayor + 1


def _ejecutar_proceso(fecha_inicio, fecha_fin, lineas_por_fecha):
    print("[1/5] Cargando historico completo y aplicando feature engineering...")
    df_raw = cargar_datos_ml()
    df_raw["Fecha"] = pd.to_datetime(df_raw["Fecha"])
    if "Viento_Direccion" in df_raw.columns:
        df_raw["Viento_Direccion"] = df_raw["Viento_Direccion"].fillna("ND")
    for columna_whip in ("WHIP_Abridor_Local", "WHIP_Abridor_Visita"):
        if columna_whip in df_raw.columns:
            df_raw[columna_whip] = df_raw[columna_whip].fillna(1.30)
    decodificadores = {}
    for columna in ["EquipoLocal", "EquipoVisita"]:
        codificador = LabelEncoder()
        codificador.fit(df_raw[columna])
        decodificadores[columna] = codificador

    df = preprocesar(df_raw)
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    df = feature_engineering_pitchers(df, df["CarrerasVisita"].median())
    df = feature_engineering_bullpen(df)
    df = feature_engineering_descanso_abridor(df)
    print(f"      {len(df)} partidos procesados para el mercado Moneyline.")

    print("[2/5] Cargando modelo ML y transformadores...")
    modelo = joblib.load(MODELO_PATH)
    transformadores = joblib.load(TRANSFORMADORES_PATH)

    print("[3/5] Evaluando partidos por fecha...")
    total_partidos = 0
    total_jugadas = 0
    for fecha_actual in pd.date_range(
            start=fecha_inicio, end=fecha_fin).date:
        partidos = df[df["Fecha"].dt.date == fecha_actual]
        print(f"\n=== PREDICCIONES MONEYLINE MLB PARA EL {fecha_actual.isoformat()} ===")
        if partidos.empty:
            print("   (sin partidos registrados)")
            continue

        lineas_fecha = lineas_por_fecha.get(fecha_actual, {})
        if not lineas_fecha:
            print("   (sin cuotas h2h de mercado para esta fecha)")
            continue

        nombres_local = decodificadores["EquipoLocal"].inverse_transform(
            partidos["EquipoLocal"])
        nombres_visita = decodificadores["EquipoVisita"].inverse_transform(
            partidos["EquipoVisita"])
        partidos["CuotaHome"] = [lineas_fecha.get(par, {}).get("cuota_home")
                                 for par in zip(nombres_local, nombres_visita)]
        partidos["CuotaAway"] = [lineas_fecha.get(par, {}).get("cuota_away")
                                 for par in zip(nombres_local, nombres_visita)]
        partidos = partidos.dropna(subset=["CuotaHome", "CuotaAway"])
        if partidos.empty:
            print("   (sin partidos con cuotas h2h disponibles)")
            continue
        total_partidos += len(partidos)

        X_hoy = construir_caracteristicas_finales(partidos, transformadores)
        proba = modelo.predict_proba(X_hoy)[:, 1]

        ancho = 40
        print(" " + "-" * 128)
        print(f" {'Partido'.ljust(ancho)} | {'P(Local)':<9} | "
              f"{'P Mercado':<10} | {'P Final':<8} | {'Cuotas':<13} | "
              f"{'Recomendacion':<13} | Stake")
        print(" " + "-" * 128)
        for (_, fila), prob_local, nombre_local, nombre_visita in zip(
                partidos.iterrows(), proba, nombres_local, nombres_visita):
            cuota_home = float(fila["CuotaHome"])
            cuota_away = float(fila["CuotaAway"])
            decision = decidir_jugada_ml(
                fila, prob_local, cuota_home, cuota_away, decodificadores)
            sugerencia = decision["sugerencia"]
            stake = decision["stake"]
            motivo = decision["motivo_anulacion"]
            if stake is not None:
                total_jugadas += 1
            detalle = f"  [{motivo}]" if motivo else ""
            print(f" {f'{nombre_local} vs {nombre_visita}'.ljust(ancho)} | "
                  f"{decision['prob_cal'] * 100:<8.0f}% | "
                  f"{decision['p_mercado'] * 100:<9.0f}% | "
                  f"{decision['prob_decision'] * 100:<7.0f}% | "
                  f"{cuota_home:.2f}/{cuota_away:.2f} | "
                  f"{sugerencia:<13} | "
                  f"{stake if stake is not None else '-'}{detalle}")
        print(" " + "-" * 128)

    print(f"\n=== RESUMEN MONEYLINE ===")
    print(f"Partidos evaluados: {total_partidos}")
    print(f"Jugadas sugeridas: {total_jugadas}")
    print("=========================")
    return 0


def ejecutar_pipeline():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fecha_inicio = fecha_desde_args("--inicio") or FECHA_INICIO_DEFECTO
    fecha_fin = fecha_desde_args("--fin") or fecha_inicio

    os.makedirs(CARPETA_REPORTE, exist_ok=True)
    indice = siguiente_indice_reporte()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = (f"{CARPETA_REPORTE}/reporte_ml_{indice}_"
                      f"{fecha_inicio.isoformat()}_a_{fecha_fin.isoformat()}"
                      f"_{timestamp}.txt")

    consola = sys.stdout
    salida = SalidaConsolidada(consola)
    sys.stdout = salida
    try:
        codigo = _ejecutar_proceso(fecha_inicio, fecha_fin, {})
    finally:
        sys.stdout = consola
        with open(archivo_salida, "w", encoding="utf-8") as archivo:
            archivo.write(salida.contenido())
    print(f"[EXPORT] Reporte ML guardado en: {archivo_salida}")
    return codigo


def main():
    return ejecutar_pipeline()


if __name__ == "__main__":
    sys.exit(main())
