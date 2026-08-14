from datetime import datetime, date, timedelta
import io
import os
import re
import sys

import joblib
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

from entrenar_modelo import (
    FEATURES,
    TRANSFORMADORES_PATH,
    VENTANA_FATIGA_3,
    cargar_datos,
    construir_caracteristicas_finales,
    preprocesar,
    feature_engineering_rachas,
    feature_engineering_pitchers,
    feature_engineering_bullpen,
    feature_engineering_fatiga,
)

MODELOS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
MODELO_PATH = os.path.join(MODELOS_DIR, "modelo_mlb_totales.pkl")
COLUMNAS_PATH = os.path.join(MODELOS_DIR, "columnas_totales.pkl")
CALIBRACION_PATH = os.path.join(MODELOS_DIR, "calibracion_totales.pkl")
LINEA_BASE = 8.5
FECHA_INICIO_DEFECTO = date(2026, 7, 1)
FECHA_FIN_DEFECTO = date(2026, 7, 31)
CARPETA_REPORTE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output_predicciones"))
VENTANA_ESPERADOS = 10
LIMITE_OVER_PROB = 0.62
LIMITE_UNDER_PROB = 0.38
VIENTO_CRITICO_KMH = 15.0
VIENTO_AJUSTE_CARRERAS = 0.5
# Regresion al mercado (#4): cuando el modelo diverge mucho de la linea
# del mercado, parte del desacuerdo es ruido (sobreajuste, datos
# incompletos). La probabilidad final se mezcla con la implicita de las
# cuotas del mercado; el peso del mercado crece con el desacuerdo hasta
# PESO_MERCADO_MAX, saturado a partir de DESACUERDO_MERCADO_REF carreras.
# Peso MANTENIDO minimo a proposito: en backtest la regresion recorta
# picks de alto edge que eran rentables (la linea 8.5 fija no es mercado
# real). La calibracion con CLV real acumulado (analisis_clv.py) subira
# este peso cuando haya evidencia (~3 semanas).
PESO_MERCADO_MAX = 0.10
DESACUERDO_MERCADO_REF = 3.00
# Tope de edge vs mercado: una proyeccion que supera a la linea del
# mercado por mas de EDGE_MAXIMO carreras es una senal sin respaldo
# (la linea ya incorpora toda la informacion publica): NO se apuesta.
EDGE_MAXIMO = 2.75
# Ruido residual out-of-time (2026) de la proyeccion de carreras: se usa
# para trasladar P(Total > 8.5) a P(Total > linea real del mercado).
SIGMA_TOTAL = 4.8
# Margen calibrado minimo para apostar: P(Over) >= 0.5 +- MARGEN_MIN_PROB.
MARGEN_MIN_PROB = 0.055
WHIP_REFERENCIA_DESCANSO = 1.30
EXPONENTE_WHIP_DESCANSO = 2.0
LIMITE_TEMPERATURA_FRIO_C = 15.0
PENALIZACION_FRIO_CARRERAS = 0.75
LIMITE_AJUSTE_DESCANSO = 0.25
LIMITE_FATIGA_BULLPEN = 0.25
LIMITE_AJUSTE_DINAMICO = 0.60
MARGEN_PROYECCION_EXTREMA = 2.0
PESO_ERA_ULTIMAS3 = 0.35
PESO_ERA_APROXIMADA = 0.25
PESO_ERA_TEMPORADA = 0.40
PESO_ABRIDOR = 0.6
PESO_BULLPEN = 0.4
LINEA_MIN_OVER = 7.5
LINEA_MAX_UNDER = 9.0
LINEA_FRAGIL = 6.5
LINEA_EXIGENTE = 9.5
EDGE_MINIMO = 1.45
EDGE_ALTA_CONFIANZA = 2.0
PENALIZACION_FATIGA = 0.5
PENALIZACION_FATIGA_CRITICA = 1.0
PENALIZACION_FATIGA_MITIGADA = 0.25
FATIGA_CRITICA_DIAS = 5
WHIP_BULLPEN_CRITICO = 1.35
WHIP_BULLPEN_SOLIDO = 1.15
WHIP_UMBRAL_VOLATILIDAD = 1.25
CUOTA_ODDS_FALLBACK = 1.91
LINEA_MAXIMA_OVER = 12.5
DIAS_REVISION_DESCANSO = 2
MIN_PARTIDOS_DESCANSO = 2
AJUSTE_AMPAYER = 0.35
AMPAYER_OVER_PROMEDIO = 9.8
AMPAYER_UNDER_PROMEDIO = 7.8
AMPAYER_OVER_TASA = 0.55
AMPAYER_UNDER_TASA = 0.55
MIN_MUESTRA_AMPAYER = 3
DESCANSO_ESTANDAR = 5
DESCANSO_CORTO_MAX = 4
DESCANSO_EXTENDIDO_MIN = 7
AJUSTE_DESCANSO_CORTO = 0.40
AJUSTE_DESCANSO_EXTENDIDO = 0.20
AJUSTE_MATCHUP = 0.45
OPS_MATCHUP_ALTO = 0.780
OPS_MATCHUP_BAJO = 0.660
SUGERENCIA_OVER = "\U0001F525 OVER (Altas)"
SUGERENCIA_UNDER = "\U0001F9CA UNDER (Bajas)"
SUGERENCIA_NO_BET = "\u26A0\ufe0f NO BET (Zona Neutra)"
SUGERENCIA_NO_BET_VIENTO = "\u26A0\ufe0f NO BET (Viento desfavorable)"


class SalidaConsolidada:
    """Escribe a la consola en tiempo real y acumula en memoria para
    generar un unico archivo de texto consolidado al finalizar."""

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


def viento_desfavorable_para_over(viento_kmh, direccion_texto):
    """True si el viento es fuerte y sopla hacia adentro (apaga los HR).

    Ambas condiciones deben cumplirse: velocidad > VIENTO_CRITICO_KMH y
    direccion dentro del sector considerado "hacia adentro".
    """
    if viento_kmh is None or pd.isna(viento_kmh) or viento_kmh <= VIENTO_CRITICO_KMH:
        return False
    if direccion_texto is None or pd.isna(direccion_texto):
        return False
    try:
        grados = float(str(direccion_texto))
    except ValueError:
        return False
    # Grados meteorologicos (desde donde sopla el viento): 0/360=N, 90=E, 180=S, 270=W.
    # Los estadios de MLB orientan el outfield hacia el NE; el viento fuerte que
    # viene del sector N/NE (0-90) sopla hacia el plato y es desfavorable para OVER.
    return 0.0 <= grados <= 90.0


PUNTOS_CARDINALES = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO",
]


def grados_a_cardinal(grados_str):
    """Convierte grados meteorologicos (0-360) al punto cardinal de 16 vientos.

    Valores no numericos, fuera de rango o "ND" devuelven "Desconocida".
    """
    try:
        grados = float(str(grados_str))
    except (TypeError, ValueError):
        return "Desconocida"
    if not (0.0 <= grados <= 360.0):
        return "Desconocida"
    indice = int((grados + 11.25) // 22.5) % 16
    return PUNTOS_CARDINALES[indice]


def fecha_desde_args(flag):
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            try:
                return date.fromisoformat(sys.argv[i + 1])
            except ValueError:
                print(f"Argumento invalido '{sys.argv[i + 1]}' para {flag}; "
                      f"se usara el valor por defecto.")
    return None


def solicitar_rango_fechas():
    modo_automatico = any(arg in sys.argv for arg in ("--inicio", "--fin"))
    if modo_automatico:
        fecha_inicio = fecha_desde_args("--inicio")
        fecha_fin = fecha_desde_args("--fin")
        if fecha_inicio is None:
            fecha_inicio = FECHA_INICIO_DEFECTO
        if fecha_fin is None:
            fecha_fin = fecha_inicio
        return fecha_inicio, fecha_fin

    texto_inicio = input(
        f"Ingresa la fecha inicio (YYYY-MM-DD, Enter para {FECHA_INICIO_DEFECTO.isoformat()}): "
    ).strip()
    if texto_inicio:
        try:
            fecha_inicio = date.fromisoformat(texto_inicio)
        except ValueError:
            print(f"Fecha invalida '{texto_inicio}'. Se usara la fecha de inicio por defecto.")
            fecha_inicio = FECHA_INICIO_DEFECTO
    else:
        fecha_inicio = FECHA_INICIO_DEFECTO

    texto_fin = input(
        f"Ingresa la fecha fin (YYYY-MM-DD, Enter para {FECHA_FIN_DEFECTO.isoformat()}): "
    ).strip()
    if texto_fin:
        try:
            fecha_fin = date.fromisoformat(texto_fin)
        except ValueError:
            print(f"Fecha invalida '{texto_fin}'. Se usara la fecha fin por defecto.")
            fecha_fin = FECHA_FIN_DEFECTO
    else:
        fecha_fin = FECHA_FIN_DEFECTO

    if fecha_fin < fecha_inicio:
        print("La fecha fin es anterior a la fecha inicio; se procesara solo la fecha inicio.")
        fecha_fin = fecha_inicio

    return fecha_inicio, fecha_fin


def construir_decodificadores(df_raw):
    decodificadores = {}
    for columna in ["EquipoLocal", "EquipoVisita"]:
        codificador = LabelEncoder()
        codificador.fit(df_raw[columna])
        decodificadores[columna] = codificador
    return decodificadores


def calcular_expected_runs(partidos, df_historico):
    """Calcula la linea esperada de carreras por enfrentamiento.

    Cada mitad de la entrada se evalua por separado:
    - base_off_X: ofensiva independiente (anotadas + permitidas del rival / 2).
    - pitching_allow_X: carreras que permite cada staff por partido, a partir
      de ERA absoluta del abridor (PESO_ABRIDOR) y del bullpen (PESO_BULLPEN).
    - Fusion 70/30: el pitcheo ENEMIGO dicta cuanto anota un equipo; la
      ofensiva local enfrenta al pitcheo visita y viceversa.
    Ponderacion final: Factor_Carreras del estadio y aire frio.
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
        lambda s: s.shift(1).rolling(VENTANA_ESPERADOS, min_periods=1).mean())
    apariciones["Permitidas10"] = grupo["Permitidas"].transform(
        lambda s: s.shift(1).rolling(VENTANA_ESPERADOS, min_periods=1).mean())

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

    if "ERA_Ultimas3_Local" in partidos.columns \
            and "ERA_Ultimas3_Visita" in partidos.columns:
        if "ERA_Temporada_Local" in partidos.columns \
                and "ERA_Temporada_Visita" in partidos.columns:
            era_local = (PESO_ERA_ULTIMAS3 * partidos["ERA_Ultimas3_Local"]
                         + PESO_ERA_APROXIMADA * partidos["ERA_Aproximada_Local"]
                         + PESO_ERA_TEMPORADA * partidos["ERA_Temporada_Local"])
            era_visita = (PESO_ERA_ULTIMAS3 * partidos["ERA_Ultimas3_Visita"]
                          + PESO_ERA_APROXIMADA * partidos["ERA_Aproximada_Visita"]
                          + PESO_ERA_TEMPORADA * partidos["ERA_Temporada_Visita"])
        else:
            era_local = (PESO_ERA_ULTIMAS3 * partidos["ERA_Ultimas3_Local"]
                         + (1 - PESO_ERA_ULTIMAS3) * partidos["ERA_Aproximada_Local"])
            era_visita = (PESO_ERA_ULTIMAS3 * partidos["ERA_Ultimas3_Visita"]
                          + (1 - PESO_ERA_ULTIMAS3) * partidos["ERA_Aproximada_Visita"])
    else:
        era_local = partidos["ERA_Aproximada_Local"]
        era_visita = partidos["ERA_Aproximada_Visita"]

    park = partidos["Factor_Carreras"].fillna(1.0)
    era_local = era_local.fillna(mediana_carreras)
    era_visita = era_visita.fillna(mediana_carreras)

    mediana_bullpen = partidos["ERA_Bullpen_Local"].median() \
        if "ERA_Bullpen_Local" in partidos.columns else 4.00
    if "ERA_Bullpen_Reciente_Local" in partidos.columns \
            and "ERA_Bullpen_Reciente_Visita" in partidos.columns:
        era_bull_local = partidos["ERA_Bullpen_Reciente_Local"].fillna(mediana_bullpen)
        era_bull_visita = partidos["ERA_Bullpen_Reciente_Visita"].fillna(mediana_bullpen)
    else:
        era_bull_local = partidos["ERA_Bullpen_Local"].fillna(mediana_bullpen)
        era_bull_visita = partidos["ERA_Bullpen_Visita"].fillna(mediana_bullpen)

    # Base ofensiva independiente de cada mitad (anotadas + permitidas / 2).
    base_off_local = (anotadas_local + permitidas_visita) / 2.0
    base_off_visita = (anotadas_visita + permitidas_local) / 2.0

    # Carreras que permite estadisticamente cada staff de pitcheo por partido.
    pitching_allow_local = (PESO_ABRIDOR * era_local
                            + PESO_BULLPEN * era_bull_local)
    pitching_allow_visita = (PESO_ABRIDOR * era_visita
                             + PESO_BULLPEN * era_bull_visita)

    # Fusion 70/30: el pitcheo enemigo dicta cuanto anota cada equipo.
    exp_runs_local = 0.30 * base_off_local + 0.70 * pitching_allow_visita
    exp_runs_visita = 0.30 * base_off_visita + 0.70 * pitching_allow_local

    aire_frio = partidos["TemperaturaC"] < LIMITE_TEMPERATURA_FRIO_C
    expected = (exp_runs_local + exp_runs_visita) * park
    expected = expected - PENALIZACION_FRIO_CARRERAS * aire_frio

    partidos["Expected_Runs"] = expected.round(2)
    partidos["Aire_Frio"] = aire_frio
    return partidos


def recomendar_stake_kelly(p, cuota, edge, sugerencia):
    """Stake por media Kelly con la probabilidad calibrada y la cuota real.

    f* = (p * b - (1 - p)) / b, con b = cuota - 1; se usa media Kelly.
      f* >= 0.08  -> 1.0u (Jugada de Alto Valor: solo con respaldo fuerte
                    del clasificador calibrado)
      < 0.08      -> 0.5u (Jugada Estandar; el edge heuristico jamas sube
                    el stake por si solo, sin ventaja del clasificador)
    """
    if sugerencia not in (SUGERENCIA_OVER, SUGERENCIA_UNDER):
        return None
    if cuota is None or pd.isna(cuota) or cuota <= 1.0:
        cuota = CUOTA_ODDS_FALLBACK
    if sugerencia == SUGERENCIA_UNDER:
        p = 1.0 - float(p)
    b = float(cuota) - 1.0
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    f = max((p * b - (1.0 - p)) / b, 0.0) * 0.5
    if f >= 0.08:
        return 1.0
    return 0.5


def feature_engineering_ampayer(df):
    """Ajuste por tendencia historica del ampayers de home plate.

    Un ampayers con sesgo a las altas (promedio >= AMPAYER_OVER_PROMEDIO o
    tasa de Over > AMPAYER_OVER_TASA) suma carreras; con sesgo a las bajas
    (promedio <= AMPAYER_UNDER_PROMEDIO o tasa de Under > AMPAYER_UNDER_TASA)
    las resta. Sin historial suficiente o dato ausente: ajuste 0.0.
    """
    df = df.copy()
    df["Ajuste_Ampayer"] = 0.0
    if "UmpireNombre" not in df.columns:
        return df

    historial = df.dropna(subset=["UmpireNombre"]).copy()
    if historial.empty:
        return df

    historial["Total_Carreras"] = historial["CarrerasLocal"] + historial["CarrerasVisita"]
    resumen = historial.groupby("UmpireNombre")["Total_Carreras"].agg(["mean", "count"])
    sobre = historial[historial["Total_Carreras"] > LINEA_BASE]
    resumen["Tasa_Over"] = (
        sobre.groupby("UmpireNombre")["Total_Carreras"].count() / resumen["count"]
    ).fillna(0.0)

    ajustes = {}
    for umpire, fila in resumen.iterrows():
        if fila["count"] < MIN_MUESTRA_AMPAYER:
            ajustes[umpire] = 0.0
        elif fila["mean"] >= AMPAYER_OVER_PROMEDIO or fila["Tasa_Over"] > AMPAYER_OVER_TASA:
            ajustes[umpire] = AJUSTE_AMPAYER
        elif fila["mean"] <= AMPAYER_UNDER_PROMEDIO \
                or (1.0 - fila["Tasa_Over"]) > AMPAYER_UNDER_TASA:
            ajustes[umpire] = -AJUSTE_AMPAYER
        else:
            ajustes[umpire] = 0.0

    df["Ajuste_Ampayer"] = df["UmpireNombre"].map(ajustes).fillna(0.0)
    return df


def feature_engineering_descanso_abridor(df):
    """Dias de descanso desde la ultima apertura de cada lanzador abridor.

    Debut de temporada o sin registro previo: DESCANSO_ESTANDAR (5 dias).
    Descanso corto (<= DESCANSO_CORTO_MAX): penalizacion por desgaste fisico
    a favor de la ofensiva rival; descanso extendido (>= DESCANSO_EXTENDIDO_MIN,
    fuera de ritmo): factor de volatilidad que reduce la confianza de Under.

    El impacto NO es una suma fija: se escala exponencialmente segun el WHIP
    base del abridor (multiplicador porcentual). Un abridor con WHIP alto
    (ya castigado por la ofensiva) sufre un castigo mayor por el descanso
    alterado; uno con WHIP bajo lo mitiga.
    """
    df = df.copy()
    for lado in ("Local", "Visita"):
        col_pitcher = f"Pitcher{lado}Id"
        col_descanso = f"Descanso_Abridor_{lado}"
        df[col_descanso] = DESCANSO_ESTANDAR
        if col_pitcher not in df.columns:
            continue

        sub = df.dropna(subset=[col_pitcher]).copy()
        sub["Fecha"] = pd.to_datetime(sub["Fecha"])
        sub = sub.sort_values("Fecha")
        sub["Fecha_Prev"] = sub.groupby(col_pitcher)["Fecha"].shift(1)
        dias = (sub["Fecha"] - sub["Fecha_Prev"]).dt.days
        dias = dias.fillna(DESCANSO_ESTANDAR).clip(lower=0)
        df.loc[sub.index, col_descanso] = dias

    for lado in ("Local", "Visita"):
        col_descanso = f"Descanso_Abridor_{lado}"
        col_whip = f"WHIP_Abridor_{lado}"
        if col_whip in df.columns:
            whip = df[col_whip].fillna(WHIP_REFERENCIA_DESCANSO)
        else:
            whip = pd.Series(WHIP_REFERENCIA_DESCANSO, index=df.index)
        multiplicador = whip / WHIP_REFERENCIA_DESCANSO
        multiplicador = multiplicador.clip(0.5, 2.0) ** EXPONENTE_WHIP_DESCANSO
        corto = df[col_descanso] <= DESCANSO_CORTO_MAX
        extendido = df[col_descanso] >= DESCANSO_EXTENDIDO_MIN
        df[f"Ajuste_Descanso_{lado}"] = (
            (AJUSTE_DESCANSO_CORTO * corto.astype(int)
             + AJUSTE_DESCANSO_EXTENDIDO * extendido.astype(int)) * multiplicador)

    df["Ajuste_Descanso"] = (
        df["Ajuste_Descanso_Local"] + df["Ajuste_Descanso_Visita"]
    ).clip(-LIMITE_AJUSTE_DESCANSO, LIMITE_AJUSTE_DESCANSO).round(2)
    return df


def feature_engineering_matchup(df):
    """Matchup de la alineacion vs la mano del lanzador abridor (LHP/RHP).

    La alineacion local batea contra la mano del abridor visitante y
    viceversa. Si el OPS del equipo vs esa mano es alto (> OPS_MATCHUP_ALTO)
    suma +AJUSTE_MATCHUP; si se desploma (< OPS_MATCHUP_BAJO) resta. Mano u
    OPS no disponibles: ajuste 0.0 (neutral).
    """
    df = df.copy()
    df["Ajuste_Matchup"] = 0.0
    if not {"ManoAbridorLocal", "ManoAbridorVisita",
            "OPSvsLHP_Local", "OPSvsRHP_Local",
            "OPSvsLHP_Visita", "OPSvsRHP_Visita"}.issubset(df.columns):
        return df

    ajuste_total = pd.Series(0.0, index=df.index)
    for lado in ("Local", "Visita"):
        mano = df["ManoAbridorVisita" if lado == "Local" else "ManoAbridorLocal"]
        col_lhp = f"OPSvsLHP_{lado}"
        col_rhp = f"OPSvsRHP_{lado}"
        ops_lhp_num = pd.to_numeric(df[col_lhp], errors="coerce").astype(float)
        ops_rhp_num = pd.to_numeric(df[col_rhp], errors="coerce").astype(float)
        ops_efectivo = ops_lhp_num.where(
            mano == "L", ops_rhp_num.where(mano == "R", float("nan")))
        alto = (ops_efectivo > OPS_MATCHUP_ALTO).fillna(False)
        bajo = (ops_efectivo < OPS_MATCHUP_BAJO).fillna(False)
        ajuste_total = (ajuste_total + AJUSTE_MATCHUP * alto.astype(float)
                        - AJUSTE_MATCHUP * bajo.astype(float))

    df["Ajuste_Matchup"] = ajuste_total.round(2)
    return df


_calibrador_cache = None


def cargar_calibrador():
    """Carga la isotonica ajustada sobre la prueba cronologica (una sola vez)."""
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


def probabilidad_calibrada_ajustada(prob_over, linea_casino):
    """P(OVER) calibrada (isotonica) y trasladada a la linea real del mercado.

    1) La isotonica corrige la sobreconfianza del XGBoost (en prueba 2026,
       el modelo decia 0.70 y acertaba ~0.58).
    2) Traslado normal de la linea base 8.5 a la linea real L usando el
       ruido residual de la proyeccion (SIGMA_TOTAL):
       P(Total > L) = Phi(Phi^-1(Pcal) - (L - 8.5) / SIGMA_TOTAL).
       Sin linea de casino disponible se conserva Pcal sin trasladar.
    """
    calibrador = cargar_calibrador()
    p = float(prob_over)
    if calibrador is not None:
        p = float(calibrador.predict([p])[0])
    p = min(max(p, 1e-4), 1 - 1e-4)
    if linea_casino is not None and not pd.isna(linea_casino) and linea_casino > 0.0:
        z0 = norm.ppf(p)
        z_ajustado = z0 - (float(linea_casino) - LINEA_BASE) / SIGMA_TOTAL
        p = float(norm.cdf(z_ajustado))
    return min(max(p, 1e-4), 1 - 1e-4)


def probabilidad_ensemble(prob_over, fila):
    """P(OVER) del ENSEMBLE DE SENALES (experimento validado en backtest).

    Combina 3 senales independientes:
      A) XGBoost calibrado (isotonica sin traslado: la linea ya entra
         implicita en B y C).
      B) Formula de carreras: Phi((Expected_Runs_Ajustada - Linea)/SIGMA).
      C) Skellam por equipo: Phi((ExpRunsTotal - Linea)/SIGMA).

    Pesos validados por walk-forward (margen 1.45 intacto): 0.5/0.25/0.25
    dio 62.2% de acierto y +20.2% ROI vs 60.5%/+18.0% del XGB solo, con
    un 25% mas de apuestas (749 vs 597).
    """
    pA = float(prob_over)
    calibrador = cargar_calibrador()
    if calibrador is not None:
        pA = float(calibrador.predict([pA])[0])
    pA = min(max(pA, 1e-4), 1 - 1e-4)

    linea = fila.get("Linea_Casino", None)
    if linea is None or pd.isna(linea) or float(linea) <= 0.0:
        return pA

    linea = float(linea)
    exp_formula = fila.get("Expected_Runs_Ajustada", None)
    pB = norm.cdf((float(exp_formula) - linea) / SIGMA_TOTAL) \
        if exp_formula is not None and not pd.isna(exp_formula) else pA

    exp_skellam = fila.get("ExpRunsTotal", None)
    pC = norm.cdf((float(exp_skellam) - linea) / SIGMA_TOTAL) \
        if exp_skellam is not None and not pd.isna(exp_skellam) else pA

    p = 0.5 * pA + 0.25 * pB + 0.25 * pC
    return min(max(p, 1e-4), 1 - 1e-4)


def probabilidad_mercado_implicita(cuota_over, cuota_under):
    """Probabilidad implicita del mercado (sin vig) a partir de las cuotas.

    p = (1/co) / (1/co + 1/cu). Devuelve None si alguna cuota no es
    valida (ausente, <= 1.0 o NaN).
    """
    try:
        co = float(cuota_over)
        cu = float(cuota_under)
    except (TypeError, ValueError):
        return None
    if pd.isna(co) or pd.isna(cu) or co <= 1.0 or cu <= 1.0:
        return None
    p = (1.0 / co) / (1.0 / co + 1.0 / cu)
    return min(max(p, 1e-4), 1 - 1e-4)


def _ajuste_viento_carreras(viento_kmh, direccion_texto):
    """Ajuste de carreras por viento fuerte en los sectores que importan.

    Viento fuerte (>= VIENTO_CRITICO_KMH) que viene del sector N/NE
    (0-90 grados) sopla hacia el plato y apaga los HR (-0.5 carreras);
    el que viene del sector S/SO (180-270) sopla hacia el jardin central
    y los impulsa (+0.5 carreras). Sin direccion valida o velocidad
    insuficiente: 0.0.
    """
    if viento_kmh is None or pd.isna(viento_kmh) or viento_kmh <= VIENTO_CRITICO_KMH:
        return 0.0
    try:
        grados = float(str(direccion_texto))
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= grados <= 90.0:
        return -VIENTO_AJUSTE_CARRERAS
    if 180.0 <= grados <= 270.0:
        return VIENTO_AJUSTE_CARRERAS
    return 0.0


def aplicar_ajustes_y_edge(partidos):
    """Ajustes dinamicos (ampayer, descanso, matchup, viento) + Edge vs linea.

    Calcula Ajuste_Dinamico (clip a LIMITE_AJUSTE_DINAMICO), la linea
    esperada ajustada y el Edge/Diferencia contra la Linea_Casino del
    partido (requiere Expected_Runs y Linea_Casino ya presentes).
    """
    for columna_ajuste in ("Ajuste_Ampayer", "Ajuste_Descanso", "Ajuste_Matchup"):
        if columna_ajuste in partidos.columns:
            partidos[columna_ajuste] = partidos[columna_ajuste].fillna(0.0)
    direcciones_viento = partidos.get("Viento_Direccion")
    if direcciones_viento is None:
        direcciones_viento = pd.Series("ND", index=partidos.index)
    partidos["Ajuste_Viento"] = [
        _ajuste_viento_carreras(v, d) for v, d in zip(
            partidos.get("Viento_Velocidad", pd.Series(0.0, index=partidos.index)),
            direcciones_viento)]
    partidos["Ajuste_Dinamico"] = (
        partidos.get("Ajuste_Ampayer", 0.0)
        + partidos.get("Ajuste_Descanso", 0.0)
        + partidos.get("Ajuste_Matchup", 0.0)
        + partidos["Ajuste_Viento"]
    ).clip(-LIMITE_AJUSTE_DINAMICO, LIMITE_AJUSTE_DINAMICO).round(2)
    partidos["Expected_Runs_Ajustada"] = (
        partidos["Expected_Runs"] + partidos["Ajuste_Dinamico"]).round(2)
    partidos["Edge"] = (
        partidos["Expected_Runs_Ajustada"] - partidos["Linea_Casino"]).abs().round(2)
    partidos["Diferencia"] = (
        partidos["Expected_Runs_Ajustada"] - partidos["Linea_Casino"]).round(2)
    return partidos


def decidir_jugada(fila, prob_over, media_carreras_estadio, partidos_por_dia,
                   fecha_actual, inercia_rota, decodificadores,
                   prob_efectiva_previa=None):
    """Logica completa de decision sobre UNA fila ya enriquecida.

    Combina la probabilidad del modelo (prob_over) con la logistica de la
    linea esperada vs linea real, y aplica TODOS los filtros defensivos:
    contradiccion, extremos, proyeccion extrema, margen insuficiente
    (EDGE_MINIMO), fatiga de bullpen y volatilidad del abridor
    (WHIP_UMBRAL_VOLATILIDAD). Devuelve un dict con la jugada decidida.

    prob_efectiva_previa: si se provee (experimento de ensemble de
    senales), se usa como prob_efectiva directamente, sin recalcular la
    calibracion interna. Si NO se provee, se usa el ensemble de senales
    (XGB + formula + Skellam, validado en walk-forward: 62.2% acierto y
    +20.2% ROI con 25% mas de apuestas que el XGB solo); si la fila no
    trae ExpRunsTotal se cae a la calibracion simple.
    """
    local = decodificadores["EquipoLocal"].inverse_transform(
        [fila["EquipoLocal"]])[0]
    visita = decodificadores["EquipoVisita"].inverse_transform(
        [fila["EquipoVisita"]])[0]
    linea_referencia_estadio = float(media_carreras_estadio.get(local, LINEA_BASE))

    linea_esperada = fila["Expected_Runs_Ajustada"]
    if prob_efectiva_previa is not None:
        prob_efectiva = min(max(float(prob_efectiva_previa), 0.0), 1.0)
    else:
        exp_skellam = fila.get("ExpRunsTotal", None)
        if exp_skellam is not None and not pd.isna(exp_skellam):
            prob_efectiva = probabilidad_ensemble(prob_over, fila)
        else:
            prob_efectiva = probabilidad_calibrada_ajustada(
                prob_over, fila.get("Linea_Casino", None))
        prob_efectiva = min(max(prob_efectiva, 0.0), 1.0)

    # Regresion al mercado (#4): la prob del modelo se mezcla con la
    # implicita de las cuotas; el peso del mercado crece con el
    # desacuerdo entre la proyeccion del sistema y la linea real.
    p_mercado = probabilidad_mercado_implicita(
        fila.get("Cuota_Over", None), fila.get("Cuota_Under", None))
    if p_mercado is None:
        p_mercado = probabilidad_mercado_implicita(
            CUOTA_ODDS_FALLBACK, CUOTA_ODDS_FALLBACK)
    desacuerdo_mercado = abs(fila["Diferencia"])
    peso_mercado = min(
        PESO_MERCADO_MAX,
        desacuerdo_mercado * PESO_MERCADO_MAX / DESACUERDO_MERCADO_REF)
    prob_decision = min(max(
        (1.0 - peso_mercado) * prob_efectiva + peso_mercado * p_mercado,
        0.0), 1.0)

    if prob_decision >= 0.5 + MARGEN_MIN_PROB:
        if viento_desfavorable_para_over(
                fila["Viento_Velocidad"], fila.get("Viento_Direccion")):
            sugerencia = SUGERENCIA_NO_BET_VIENTO
        else:
            sugerencia = SUGERENCIA_OVER
    elif prob_decision <= 0.5 - MARGEN_MIN_PROB:
        sugerencia = SUGERENCIA_UNDER
    elif fila["Diferencia"] >= EDGE_MINIMO:
        # El sistema proyecta mas carreras que el casino (linea esperada
        # - linea casino >= EDGE_MINIMO): la tendencia apunta al OVER.
        if viento_desfavorable_para_over(
                fila["Viento_Velocidad"], fila.get("Viento_Direccion")):
            sugerencia = SUGERENCIA_NO_BET_VIENTO
        else:
            sugerencia = SUGERENCIA_OVER
    else:
        sugerencia = SUGERENCIA_NO_BET

    motivo_anulacion = None

    if sugerencia == SUGERENCIA_NO_BET:
        motivo_anulacion = (
            f"Sin senal direccional (desacuerdo "
            f"{fila['Diferencia']:+.2f} carreras, prob_over "
            f"{prob_over:.1%})")
    elif sugerencia == SUGERENCIA_NO_BET_VIENTO:
        motivo_anulacion = (
            f"Viento desfavorable para OVER "
            f"(vel {fila['Viento_Velocidad']} km/h, "
            f"dir {fila.get('Viento_Direccion', 'n/a')})")

    if sugerencia == SUGERENCIA_OVER and linea_esperada < LINEA_MIN_OVER:
        sugerencia = SUGERENCIA_NO_BET
        motivo_anulacion = (f"Anulado por Filtro de Contradiccion "
                            f"(linea {linea_esperada:.2f} < {LINEA_MIN_OVER})")
    elif sugerencia == SUGERENCIA_UNDER and linea_esperada > LINEA_MAX_UNDER:
        sugerencia = SUGERENCIA_NO_BET
        motivo_anulacion = (f"Anulado por Filtro de Contradiccion "
                            f"(linea {linea_esperada:.2f} > {LINEA_MAX_UNDER})")

    if linea_esperada <= LINEA_FRAGIL and sugerencia == SUGERENCIA_UNDER:
        sugerencia = SUGERENCIA_NO_BET
        motivo_anulacion = (f"Anulado por Filtro Extremo "
                            f"(linea {linea_esperada:.2f} <= {LINEA_FRAGIL})")
    elif linea_esperada >= LINEA_EXIGENTE and sugerencia == SUGERENCIA_OVER \
            and fila["Diferencia"] < EDGE_MINIMO:
        # El filtro extremo alto solo descarta OVERs sin respaldo direccional:
        # un OVER con edge positivo real (linea esperada - casino >= EDGE_MINIMO)
        # se considera una oportunidad viable y no se anula.
        sugerencia = SUGERENCIA_NO_BET
        motivo_anulacion = (f"Anulado por Filtro Extremo "
                            f"(linea {linea_esperada:.2f} >= {LINEA_EXIGENTE})")

    limite_proyeccion = max(
        LINEA_MAXIMA_OVER,
        linea_referencia_estadio + MARGEN_PROYECCION_EXTREMA)
    if sugerencia == SUGERENCIA_OVER \
            and linea_esperada >= limite_proyeccion:
        # Proyecciones extremas (efecto Coors Field / estadios de altura):
        # el modelo sobreestima la varianza real de la ofensiva. El
        # umbral sube solo cuando el estadio tiene una media nativa
        # altisima (Park Factor), donde una linea alta es legitima.
        sugerencia = SUGERENCIA_NO_BET
        motivo_anulacion = (f"Anulado por Proyeccion Extrema "
                            f"(linea {linea_esperada:.2f} >= "
                            f"{limite_proyeccion:.2f})")

    if sugerencia in (SUGERENCIA_OVER, SUGERENCIA_UNDER) \
            and fila["Edge"] < EDGE_MINIMO:
        sugerencia = SUGERENCIA_NO_BET
        motivo_anulacion = (f"Anulado por Margen Insuficiente "
                            f"(edge {fila['Edge']:.2f} < {EDGE_MINIMO})")

    tope_edge = False
    if sugerencia in (SUGERENCIA_OVER, SUGERENCIA_UNDER) \
            and abs(fila["Diferencia"]) > EDGE_MAXIMO:
        # Tope de edge (#4): proyecciones que superan a la linea del
        # mercado por mas de EDGE_MAXIMO carreras se anulan: el modelo
        # esta apostando contra el mercado sin evidencia suficiente.
        sugerencia = SUGERENCIA_NO_BET
        tope_edge = True
        motivo_anulacion = (f"Anulado por Edge Excesivo vs Mercado "
                            f"(edge {abs(fila['Diferencia']):.2f} > "
                            f"{EDGE_MAXIMO})")

    penalizacion_fatiga = None
    if sugerencia == SUGERENCIA_UNDER:
        juegos_seguidos = max(
            fila.get("Juegos_Ultimos_3_Dias_Local", 0),
            fila.get("Juegos_Ultimos_3_Dias_Visita", 0))
        if juegos_seguidos >= VENTANA_FATIGA_3:
            whip_local = fila.get("WHIP_Abridor_Local", WHIP_UMBRAL_VOLATILIDAD)
            whip_visita = fila.get("WHIP_Abridor_Visita", WHIP_UMBRAL_VOLATILIDAD)
            whip_max = max(whip_local, whip_visita)
            whip_min = min(whip_local, whip_visita)
            if juegos_seguidos >= FATIGA_CRITICA_DIAS \
                    and whip_max > WHIP_BULLPEN_CRITICO:
                penalizacion = PENALIZACION_FATIGA_CRITICA
            elif juegos_seguidos < FATIGA_CRITICA_DIAS \
                    and whip_min < WHIP_BULLPEN_SOLIDO:
                penalizacion = PENALIZACION_FATIGA_MITIGADA
            else:
                penalizacion = PENALIZACION_FATIGA
            penalizacion = min(penalizacion, LIMITE_FATIGA_BULLPEN)
            fila["Edge"] = round(fila["Edge"] - penalizacion, 2)
            penalizacion_fatiga = (juegos_seguidos, penalizacion)
            if fila["Edge"] < EDGE_MINIMO:
                sugerencia = SUGERENCIA_NO_BET
                motivo_anulacion = (f"Anulado por Fatiga de Bullpen "
                                    f"({juegos_seguidos} juegos seguidos)")

    if sugerencia == SUGERENCIA_UNDER:
        whip_local = fila.get("WHIP_Abridor_Local", WHIP_UMBRAL_VOLATILIDAD)
        whip_visita = fila.get("WHIP_Abridor_Visita", WHIP_UMBRAL_VOLATILIDAD)
        if whip_local > WHIP_UMBRAL_VOLATILIDAD \
                or whip_visita > WHIP_UMBRAL_VOLATILIDAD:
            sugerencia = SUGERENCIA_NO_BET
            motivo_anulacion = (f"Anulado por Volatilidad del Abridor "
                                f"(WHIP Local: {whip_local:.2f} | "
                                f"Visita: {whip_visita:.2f})")

    stake = recomendar_stake_kelly(
        prob_decision, fila.get("Cuota", None), fila["Edge"], sugerencia)
    inercia_cap = (inercia_rota and stake is not None and stake > 0.5
                   and fila["Edge"] < EDGE_ALTA_CONFIANZA)
    if inercia_cap:
        stake = 0.5

    desacuerdo_cap = False
    if stake is not None and stake > 0.5 and sugerencia in (
            SUGERENCIA_OVER, SUGERENCIA_UNDER):
        # Filtro de desacuerdo: si la direccion del clasificador calibrado
        # (P(OVER) respecto a 0.5) y la de la linea esperada heuristica
        # (Expected vs Linea_Casino) apuntan a lados opuestos, la senal
        # real es debil: el stake nunca supera 0.5 unidades.
        direccion_modelo = prob_decision > 0.5
        direccion_formula = fila["Diferencia"] > 0
        if direccion_modelo != direccion_formula:
            stake = 0.5
            desacuerdo_cap = True

    datos_faltantes_cap = False
    if stake is not None and stake > 0.5:
        # Datos faltantes (matchup LHP/RHP): sin OPS splits de ambos
        # equipos el ajuste de matchup es neutro por desconocimiento,
        # no por ausencia de efecto: se limita a 0.5 unidades.
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
        "linea_esperada": linea_esperada,
        "prob_over": prob_over,
        "prob_efectiva": prob_efectiva,
        "sugerencia": sugerencia,
        "motivo_anulacion": motivo_anulacion,
        "stake": stake,
        "inercia_cap": inercia_cap,
        "desacuerdo_cap": desacuerdo_cap,
        "datos_faltantes_cap": datos_faltantes_cap,
        "penalizacion_fatiga": penalizacion_fatiga,
        "linea_referencia_estadio": linea_referencia_estadio,
        "prob_decision": prob_decision,
        "peso_mercado": peso_mercado,
        "tope_edge": tope_edge,
    }


def _ejecutar_proceso(fecha_inicio, fecha_fin):
    print("[1/4] Cargando historico completo y aplicando feature engineering...")
    df_raw = cargar_datos(solo_con_temperatura=False)
    if "Viento_Direccion" in df_raw.columns:
        df_raw["Viento_Direccion"] = df_raw["Viento_Direccion"].fillna("ND")
    for columna_whip in ("WHIP_Abridor_Local", "WHIP_Abridor_Visita"):
        if columna_whip in df_raw.columns:
            df_raw[columna_whip] = df_raw[columna_whip].fillna(WHIP_UMBRAL_VOLATILIDAD)
    decodificadores = construir_decodificadores(df_raw)

    passthrough = df_raw[["Fecha", "EquipoLocal", "EquipoVisita"]].copy()
    passthrough["Fecha"] = pd.to_datetime(passthrough["Fecha"])
    if "Viento_Direccion" in df_raw.columns:
        passthrough["Viento_Direccion"] = df_raw["Viento_Direccion"]
    if "Linea_Casino_Real" in df_raw.columns:
        # Linea REAL de cierre (ETL The Odds API -> dbo.GameLog).
        passthrough["Linea_Casino"] = df_raw["Linea_Casino_Real"]
        if "Cuota_Over_Real" in df_raw.columns:
            passthrough["Cuota_Over_Real"] = df_raw["Cuota_Over_Real"]
        if "Cuota_Under_Real" in df_raw.columns:
            passthrough["Cuota_Under_Real"] = df_raw["Cuota_Under_Real"]
    elif "Linea_Casino" in df_raw.columns:
        passthrough["Linea_Casino"] = df_raw["Linea_Casino"]
    enc_local = LabelEncoder().fit(df_raw["EquipoLocal"])
    enc_visita = LabelEncoder().fit(df_raw["EquipoVisita"])
    passthrough["EquipoLocal"] = enc_local.transform(passthrough["EquipoLocal"])
    passthrough["EquipoVisita"] = enc_visita.transform(passthrough["EquipoVisita"])

    df = preprocesar(df_raw)
    df = feature_engineering_rachas(df)
    df = feature_engineering_fatiga(df)
    df = feature_engineering_pitchers(df, df["CarrerasVisita"].median())
    df = feature_engineering_bullpen(df)
    df = feature_engineering_ampayer(df)
    df = feature_engineering_descanso_abridor(df)
    df = feature_engineering_matchup(df)
    df["Total_Carreras"] = df["CarrerasLocal"] + df["CarrerasVisita"]
    media_carreras_estadio_por_codigo = df.groupby("EquipoLocal")["Total_Carreras"].mean()
    codigo_a_nombre = {i: nombre for i, nombre in enumerate(enc_local.classes_)}
    media_carreras_estadio = media_carreras_estadio_por_codigo.rename(
        index=codigo_a_nombre)
    print(f"      {len(df)} partidos procesados para calcular rachas, ERAs, bullpen reciente, "
          f"ampayers, descanso de abridores y matchups LHP/RHP.")

    lista_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin).date
    print(f"      {len(lista_fechas)} dia(s) a procesar: "
          f"{fecha_inicio.isoformat()} -> {fecha_fin.isoformat()}.")

    # Convertir la columna Fecha a datetime antes de filtrar
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    partidos_por_dia = df["Fecha"].dt.date.value_counts()

    print("[2/4] Cargando modelo y columnas guardadas...")
    modelo = joblib.load(MODELO_PATH)
    columnas = joblib.load(COLUMNAS_PATH)
    transformadores = joblib.load(TRANSFORMADORES_PATH)

    total_partidos = 0
    total_jugadas = 0
    wins = 0
    losses = 0
    pushes = 0
    pendientes = 0
    sin_linea_real_total = 0
    unidades = 0.0
    total_apostado = 0.0
    jugadas_sin_cuota = 0

    print("[3/4] Prediciendo por fecha...")
    for fecha_actual in lista_fechas:
        partidos = df[df["Fecha"].dt.date == fecha_actual]
        print(f"\n=== PREDICCIONES MLB PARA EL {fecha_actual.isoformat()} ===")

        if partidos.empty:
            print("   (sin partidos registrados)")
            continue

        # Ruptura de inercia: si en los dias previos hubo muy pocos partidos en
        # todo el calendario (pausa del All-Star, descanso prolongado), el ritmo
        # de los equipos se corta y la confiabilidad de la proyeccion baja.
        inercia_rota = any(
            int(partidos_por_dia.get(
                fecha_actual - timedelta(days=k), 0)) <= MIN_PARTIDOS_DESCANSO
            for k in range(1, DIAS_REVISION_DESCANSO + 1))
        if inercia_rota:
            print("   Post-descanso prolongado (inercia rota): "
                  "stakes limitados a 0.5 Unidades")

        partidos = partidos.merge(
            passthrough, on=["Fecha", "EquipoLocal", "EquipoVisita"], how="left")

        partidos = calcular_expected_runs(partidos, df)

        if "Linea_Casino" not in partidos.columns:
            # Sin datos de linea real del mercado: el partido NO es evaluable.
            # No se usa ningun fallback al promedio del estadio.
            partidos["Linea_Casino"] = float("nan")
        if "Cuota_Over_Real" not in partidos.columns:
            partidos["Cuota_Over_Real"] = float("nan")
        if "Cuota_Under_Real" not in partidos.columns:
            partidos["Cuota_Under_Real"] = float("nan")

        sin_linea_real_total += int(partidos["Linea_Casino"].isna().sum())
        partidos_sin_linea = partidos[partidos["Linea_Casino"].isna()].copy()
        partidos_con_linea = partidos[partidos["Linea_Casino"].notna()].copy()
        partidos = partidos_con_linea

        if not partidos_sin_linea.empty:
            print(f"      {len(partidos_sin_linea)} partido(s) sin linea real "
                  "(no evaluables):")
            for _, fila_sin in partidos_sin_linea.iterrows():
                local_sin = decodificadores["EquipoLocal"].inverse_transform(
                    [fila_sin["EquipoLocal"]])[0]
                visita_sin = decodificadores["EquipoVisita"].inverse_transform(
                    [fila_sin["EquipoVisita"]])[0]
                print(f"         {local_sin} vs {visita_sin}")

        if partidos.empty:
            print("      (sin partidos con linea real de cierre; "
                  "dia no evaluable)")
            continue

        # Ajustes dinamicos: ampayer, descanso del abridor y matchup LHP/RHP.
        # Se aplican a la linea esperada ANTES de calcular edge, diferencia y
        # de pasar por los filtros defensivos (Margen, Extremo, Fatiga, Clima).
        partidos = aplicar_ajustes_y_edge(partidos)

        print(f"      {len(partidos)} partidos evaluados contra la "
              "Linea_Casino_Real (cierre The Odds API).")

        X_hoy = construir_caracteristicas_finales(partidos, transformadores)
        predicciones_proba = modelo.predict_proba(X_hoy)[:, 1]

        total_partidos += len(partidos)

        for (_, fila), prob_over in zip(partidos.iterrows(), predicciones_proba):
            decision = decidir_jugada(
                fila, prob_over, media_carreras_estadio, partidos_por_dia,
                fecha_actual, inercia_rota, decodificadores)
            local = decision["local"]
            visita = decision["visita"]
            linea_esperada = decision["linea_esperada"]
            linea_referencia_estadio = decision["linea_referencia_estadio"]
            sugerencia = decision["sugerencia"]
            motivo_anulacion = decision["motivo_anulacion"]
            stake = decision["stake"]

            if decision["penalizacion_fatiga"] is not None:
                juegos_seguidos, penalizacion_fatiga = decision["penalizacion_fatiga"]
                print(f"   Penalizacion por Fatiga de Bullpen: "
                      f"-{penalizacion_fatiga:.2f} al edge "
                      f"({juegos_seguidos} juegos seguidos)")

            print(f"\u26BE {local} vs {visita}")
            print(f"   Probabilidad de OVER (modelo): {decision['prob_over'] * 100:.0f}%")
            print(f"   Probabilidad ajustada: {decision['prob_efectiva'] * 100:.0f}%")
            print(f"   Linea Esperada (sistema): {fila['Expected_Runs']:.2f}")
            piezas_ajuste = []
            if fila["Ajuste_Ampayer"] != 0.0:
                nombre_umpire = fila.get("UmpireNombre", "")
                if pd.isna(nombre_umpire):
                    nombre_umpire = "desconocido"
                piezas_ajuste.append(
                    f"ampayer {nombre_umpire}: {fila['Ajuste_Ampayer']:+.2f}")
            if fila["Ajuste_Descanso"] != 0.0:
                piezas_ajuste.append(
                    f"descanso abridor (local {fila['Descanso_Abridor_Local']}d, "
                    f"visita {fila['Descanso_Abridor_Visita']}d): "
                    f"{fila['Ajuste_Descanso']:+.2f}")
            if fila["Ajuste_Matchup"] != 0.0:
                piezas_ajuste.append(f"matchup LHP/RHP: {fila['Ajuste_Matchup']:+.2f}")
            if piezas_ajuste:
                print(f"   Ajustes dinamicos: {', '.join(piezas_ajuste)}")
            print(f"   Linea Esperada (final): {linea_esperada:.2f}")
            print(f"   Linea de referencia (estadio): {linea_referencia_estadio:.2f}")
            if pd.notna(fila["TemperaturaC"]):
                texto_temp = f"Temperatura: {fila['TemperaturaC']:.0f}Â°C"
                if fila["Aire_Frio"]:
                    texto_temp += (f" (aire frio: -{PENALIZACION_FRIO_CARRERAS:.2f} "
                                   "carreras)")
                print(f"   {texto_temp}")
            if pd.notna(fila["Viento_Velocidad"]):
                direccion = fila["Viento_Direccion"]
                try:
                    grados_viento = float(str(direccion))
                    texto_viento = (f"Viento: {fila['Viento_Velocidad']:.0f} km/h "
                                    f"(direccion {grados_viento:.0f}\u00B0 - "
                                    f"{grados_a_cardinal(direccion)})")
                except (TypeError, ValueError):
                    texto_viento = (f"Viento: {fila['Viento_Velocidad']:.0f} km/h "
                                    f"(direccion {direccion} - "
                                    f"{grados_a_cardinal(direccion)})")
                print(f"   {texto_viento}")
            if motivo_anulacion:
                print(f"   {motivo_anulacion}")
            if decision["inercia_cap"]:
                print("   Post-descanso prolongado (inercia rota): "
                      "stake limitado a 0.5 Unidades")
            print(f"   Linea Casino: {fila['Linea_Casino']:.2f} | "
                  f"Edge vs mercado: {fila['Edge']:.2f}")
            if stake is not None:
                total_jugadas += 1
                texto_stake = f"Stake: {stake:.1f} Unidades"
                if stake >= 1.0:
                    texto_stake += " (Jugada de Alto Valor)"
                else:
                    texto_stake += " (Jugada Estandar / Defensiva)"
                print(f"   {texto_stake}")
            print(f"   Sugerencia: {sugerencia}")
            if stake is not None:
                # Autoevaluacion contra las carreras reales del partido
                # (CarrerasLocal + CarrerasVisita de dbo.GameLog).
                real_local = fila.get("CarrerasLocal")
                real_visita = fila.get("CarrerasVisita")
                carreras_reales = None
                if not (pd.isna(real_local) or pd.isna(real_visita)
                        or (real_local == 0 and real_visita == 0)):
                    carreras_reales = float(real_local) + float(real_visita)
                if carreras_reales is None:
                    pendientes += 1
                    print("   Resultado Real: â€” Carreras | â³ PENDIENTE")
                else:
                    linea_usada = float(fila["Linea_Casino"])
                    if sugerencia == SUGERENCIA_OVER:
                        es_acierto = carreras_reales > linea_usada
                        es_fallo = carreras_reales < linea_usada
                    else:
                        es_acierto = carreras_reales < linea_usada
                        es_fallo = carreras_reales > linea_usada

                    cuota_ganadora = (
                        fila.get("Cuota_Over_Real")
                        if es_acierto else fila.get("Cuota_Under_Real"))
                    if pd.isna(cuota_ganadora):
                        cuota_ganadora = CUOTA_ODDS_FALLBACK
                        jugadas_sin_cuota += 1
                    total_apostado += stake
                    if es_acierto:
                        unidades += stake * (float(cuota_ganadora) - 1.0)
                    elif es_fallo:
                        unidades -= stake
                    if es_acierto:
                        wins += 1
                        print(f"   Resultado Real: {carreras_reales:.0f} "
                              "Carreras | âœ… ACIERTO")
                    elif es_fallo:
                        losses += 1
                        print(f"   Resultado Real: {carreras_reales:.0f} "
                              "Carreras | âŒ FALLO")
                    else:
                        pushes += 1
                        print(f"   Resultado Real: {carreras_reales:.0f} "
                              "Carreras | âž– EMPATE")
            print()

    total_jugadas_resueltas = wins + losses + pushes
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
    roi_porcentaje = ((unidades / total_apostado * 100)
                      if total_apostado > 0 else 0.0)
    print("\n=== RESULTADOS CONTRA LA LINEA REAL DE CIERRE (The Odds API) ===")
    print(f"Unidades ganadas/perdidas: {unidades:+.2f} u | "
          f"Apostado: {total_apostado:.1f} u")
    print(f"ROI real: {roi_porcentaje:+.2f}%")
    if jugadas_sin_cuota:
        print(f"Aviso: {jugadas_sin_cuota} jugada(s) sin cuota real; "
              f"se asumio cuota {CUOTA_ODDS_FALLBACK}.")
    if sin_linea_real_total:
        print(f"Partidos sin linea real de cierre (no evaluables): "
              f"{sin_linea_real_total}")
    print(f"\n=== RESUMEN FINAL DEL BACKTEST ===")
    print(f"Partidos evaluados: {total_partidos}")
    print(f"Jugadas sugeridas: {total_jugadas}")
    print(f"âœ… Aciertos: {wins}")
    print(f"âŒ Fallos: {losses}")
    print(f"âž– Empates (Pushes): {pushes}")
    print(f"â³ Pendientes: {pendientes}")
    print(f"ðŸ“Š Efectividad (Win Rate): {win_rate:.2f}%")
    print("==================================")
    return 0


def siguiente_indice_reporte():
    """Indice secuencial de reportes: mayor reporte_N_* existente + 1.

    Sin reportes previos comienza en 1 (reporte_1_...).
    """
    if not os.path.isdir(CARPETA_REPORTE):
        return 1
    mayor = 0
    for nombre in os.listdir(CARPETA_REPORTE):
        coincidencia = re.match(r"reporte_(\d+)_", nombre)
        if coincidencia:
            mayor = max(mayor, int(coincidencia.group(1)))
    return mayor + 1


def ejecutar_pipeline():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fecha_inicio, fecha_fin = solicitar_rango_fechas()
    os.makedirs(CARPETA_REPORTE, exist_ok=True)

    indice = siguiente_indice_reporte()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = (f"{CARPETA_REPORTE}/reporte_{indice}_"
                      f"{fecha_inicio.isoformat()}_a_{fecha_fin.isoformat()}"
                      f"_{timestamp}.txt")

    consola = sys.stdout
    salida = SalidaConsolidada(consola)
    sys.stdout = salida
    try:
        codigo = _ejecutar_proceso(fecha_inicio, fecha_fin)
    finally:
        sys.stdout = consola
        with open(archivo_salida, "w", encoding="utf-8") as archivo:
            archivo.write(salida.contenido())
    print(f"[EXPORT] Reporte consolidado guardado en: {archivo_salida}")
    return codigo


def main():
    return ejecutar_pipeline()


if __name__ == "__main__":
    sys.exit(main())


