"""Recomendador de apuestas Over/Under en vivo con el pipeline de predecir_hoy.py.

Unifica ambos mundos del proyecto:

- MOTOR (predecir_hoy.py): feature engineering avanzado (rachas, fatiga, ERAs
  de abridores, bullpen reciente), Expected_Runs 70/30, ajustes dinamicos
  (ampayer, descanso, matchup), probabilidad logistica vs linea real y TODOS
  los filtros defensivos (EDGE_MINIMO, WHIP_UMBRAL_VOLATILIDAD, contradiccion,
  extremos, proyeccion extrema, fatiga de bullpen, volatilidad del abridor)
  con las mismas constantes de riesgo del backtest (59.09%).
- PRODUCCION: linea Over/Under REAL de The Odds API (moda entre casas) y
  abridores probables de la MLB StatsAPI para el dia en curso.

La API Key de The Odds API se lee de la variable de entorno
THE_ODDS_API_KEY, de un archivo .env o (fallback) del appsettings.json.

Uso:
    python recomendar_apuestas.py [--fecha YYYY-MM-DD] [--ventana-min MIN]
"""

import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone, timedelta
from statistics import median as _mediana

import joblib
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

import db_utils

from entrenar_modelo import (
    FEATURES,
    TRANSFORMADORES_PATH,
    cargar_datos,
    construir_caracteristicas_finales,
    feature_engineering_bullpen,
    feature_engineering_fatiga,
    feature_engineering_pitchers,
    feature_engineering_rachas,
    preprocesar,
)
from backtest_skellam_ml import (
    aplicar_ajustes_por_lado,
    expected_runs_por_lado,
)
from predecir_hoy import (
    DIAS_REVISION_DESCANSO,
    EDGE_MINIMO,
    LINEA_BASE,
    MIN_PARTIDOS_DESCANSO,
    SUGERENCIA_NO_BET,
    SUGERENCIA_NO_BET_VIENTO,
    SUGERENCIA_OVER,
    SUGERENCIA_UNDER,
    WHIP_UMBRAL_VOLATILIDAD,
    aplicar_ajustes_y_edge,
    calcular_expected_runs,
    construir_decodificadores,
    decidir_jugada,
    feature_engineering_ampayer,
    feature_engineering_descanso_abridor,
    feature_engineering_matchup,
)

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.normpath(os.path.join(CARPETA_SCRIPT, "..", "..", "models"))
MODELO_CLASIFICADOR_PATH = os.path.join(MODELOS_DIR, "modelo_mlb_totales.pkl")
COLUMNAS_CLASIFICADOR_PATH = os.path.join(MODELOS_DIR, "columnas_totales.pkl")
TRANSFORMADORES_CLASIFICADOR_PATH = os.path.join(MODELOS_DIR, "transformadores_totales.pkl")

API_KEY_VAR = "THE_ODDS_API_KEY"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
REGION = "us"
MERCADO = "totals"
FORMATO_CUOTAS = "decimal"
TIMEOUT_SEGUNDOS = 30


def obtener_api_key():
    """API Key desde variable de entorno, .env o appsettings.json."""
    clave = os.environ.get(API_KEY_VAR)
    if clave:
        return clave.strip()
    try:
        from dotenv import load_dotenv
        for ruta in (os.path.join(CARPETA_SCRIPT, ".env"),
                     os.path.join(CARPETA_SCRIPT, "..", "..", ".env")):
            load_dotenv(ruta)
    except ImportError:
        pass
    clave = os.environ.get(API_KEY_VAR)
    if clave:
        return clave.strip()
    ruta_config = os.path.normpath(os.path.join(
        CARPETA_SCRIPT, "..", "..", "config", "appsettings.json"))
    if os.path.exists(ruta_config):
        with open(ruta_config, "r", encoding="utf-8") as archivo:
            return (json.load(archivo).get("Apis", {})
                    .get("TheOddsApiKey") or "").strip()
    return None


def obtener_lineas(api_key):
    """Linea principal (moda entre casas) de Totals para cada partido.

    Devuelve dict {(home, away): {"linea": float, "casas": int,
    "cuota_over": float, "cuota_under": float}} con la cuota decimal
    mediana (solo casas que ofrecen esa linea).
    """
    respuesta = requests.get(
        ODDS_BASE_URL,
        params={"apiKey": api_key, "regions": REGION,
                "markets": MERCADO, "oddsFormat": FORMATO_CUOTAS},
        timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()

    lineas = {}
    for evento in respuesta.json():
        puntos = []
        casas = 0
        por_casa = []  # (punto, precio_over, precio_under)
        for casa in evento.get("bookmakers", []):
            for mercado in casa.get("markets", []):
                if mercado.get("key") != MERCADO:
                    continue
                casas += 1
                precios = {}
                for resultado in mercado.get("outcomes", []):
                    nombre = resultado.get("name")
                    if nombre in ("Over", "Under") \
                            and resultado.get("point") is not None \
                            and resultado.get("price") is not None:
                        precios[nombre] = (
                            float(resultado["point"]),
                            float(resultado["price"]))
                if "Over" in precios:
                    puntos.append(precios["Over"][0])
                if precios:
                    por_casa.append((
                        precios.get("Over", (None, None))[0],
                        precios.get("Over", (None, None))[1],
                        precios.get("Under", (None, None))[1]))
        if not puntos:
            continue
        linea = Counter(puntos).most_common(1)[0][0]
        over = [p for (pt, p, _u) in por_casa if pt == linea and p is not None]
        under = [p for (pt, _o, p) in por_casa if pt == linea and p is not None]
        lineas[(evento["home_team"], evento["away_team"])] = {
            "linea": linea,
            "casas": casas,
            "cuota_over": _mediana(over) if over else None,
            "cuota_under": _mediana(under) if under else None}
    return lineas


def obtener_lineas_snapshot(fecha):
    """Lineas del mercado para una fecha PASADA desde LineaSnapshots.

    The Odds API solo devuelve partidos vigentes; para predecir dias ya
    jugados se usan los snapshots que el ETL captura a diario. Para cada
    partido se toma la linea MAS RECIENTE de cada casa (maximo
    CapturadoUtc por Casa) para no mezclar lineas antiguas con las
    vigentes. Replica la logica de obtener_lineas(): linea moda entre
    casas y cuota mediana de las casas que ofrecen esa linea. Devuelve
    el mismo formato
    {(home, away): {"linea", "casas", "cuota_over", "cuota_under"}}.
    """
    filas = db_utils.leer_sql("""
        SELECT s.EquipoLocal, s.EquipoVisita, s.Linea,
               s.CuotaOver, s.CuotaUnder
        FROM LineaSnapshots s
        INNER JOIN (
            SELECT EquipoLocal, EquipoVisita, Casa,
                   MAX(CapturadoUtc) AS Ultima
            FROM LineaSnapshots
            WHERE Fecha = ?
            GROUP BY EquipoLocal, EquipoVisita, Casa
        ) t ON s.Fecha = ? AND s.EquipoLocal = t.EquipoLocal
           AND s.EquipoVisita = t.EquipoVisita
           AND s.Casa = t.Casa AND s.CapturadoUtc = t.Ultima
        WHERE s.CuotaOver IS NOT NULL
        ORDER BY s.EquipoLocal, s.EquipoVisita, s.Linea""",
        params=[fecha, fecha])
    if filas.empty:
        return {}

    lineas = {}
    for (local, visita), grupo in filas.groupby(
            ["EquipoLocal", "EquipoVisita"], sort=False):
        puntos = list(grupo["Linea"].astype(float))
        linea = Counter(puntos).most_common(1)[0][0]
        en_moda = grupo[grupo["Linea"].astype(float) == linea]
        over = [float(c) for c in en_moda["CuotaOver"].dropna()]
        under = [float(c) for c in en_moda["CuotaUnder"].dropna()]
        lineas[(local, visita)] = {
            "linea": linea,
            "casas": len(grupo),
            "cuota_over": _mediana(over) if over else None,
            "cuota_under": _mediana(under) if under else None}
    return lineas


def obtener_calendario(fecha, solo_no_iniciados=False, ventana_min=0,
                       recuperar_min=0):
    """Partidos MLB del dia con abridores probables (StatsAPI).

    Si solo_no_iniciados=True solo devuelve partidos en estado
    "Preview" (no han empezado) cuya hora de inicio aun no llego:
    evita predecir juegos ya finalizados o en vivo, cuyas
    lineas/cuotas ya no son apostables. En modo retroactivo (fechas
    pasadas) se conservan todos los partidos para poder contrastar
    lineas del snapshot.

    Si ventana_min > 0 solo devuelve partidos cuyo inicio este a
    MENOS de ventana_min minutos en el futuro (0 < inicio - ahora
    <= ventana_min): es el modo "runner por partido" que pronostica
    cada juego cerca de su primer pitch (~30-45 min antes) con la
    linea de mercado mas fresca posible.

    recuperar_min > 0 admite ademas partidos que ya iniciaron hace
    hasta recuperar_min minutos (modo recuperacion por huecos del
    cron): se evaluan igual con la linea del snapshot pre-juego en
    vez de perderse sin evaluar.
    """
    respuesta = requests.get(
        f"{MLB_BASE_URL}/schedule",
        params={"sportId": 1, "date": fecha.isoformat(),
                "hydrate": "probablePitcher,venue,team"},
        timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()

    ahora_utc = datetime.now(timezone.utc)
    partidos = []
    for dia in respuesta.json().get("dates", []):
        for juego in dia.get("games", []):
            estado = (juego.get("status") or {})
            estado_detallado = estado.get("detailedState", "")
            if estado_detallado in ("Postponed", "Cancelled"):
                continue
            if solo_no_iniciados and estado.get("abstractGameState") != "Preview":
                continue
            local = juego["teams"]["home"]["team"]
            visita = juego["teams"]["away"]["team"]
            probable_local = juego["teams"]["home"].get("probablePitcher") or {}
            probable_visita = juego["teams"]["away"].get("probablePitcher") or {}
            hora_inicio = None
            if juego.get("gameDate"):
                try:
                    hora_inicio = datetime.fromisoformat(
                        juego["gameDate"].replace("Z", "+00:00"))
                except ValueError:
                    hora_inicio = None
            # Regla de validez: solo se pronostica lo que aun no empezo.
            if solo_no_iniciados and hora_inicio is not None \
                    and hora_inicio <= ahora_utc:
                continue
            # Modo ventana: solo partidos que inician dentro de
            # ventana_min minutos (a partir de ahora), admitiendo los
            # que ya iniciaron hace hasta recuperar_min minutos.
            if ventana_min > 0:
                if hora_inicio is None:
                    continue
                minutos_restantes = (
                    hora_inicio - ahora_utc).total_seconds() / 60.0
                if minutos_restantes <= -recuperar_min \
                        or minutos_restantes > ventana_min:
                    continue
            partidos.append({
                "local": local.get("fullName") or local.get("name"),
                "visita": visita.get("fullName") or visita.get("name"),
                "estadio": (juego.get("venue") or {}).get("name"),
                "pitcher_local": probable_local.get("id"),
                "pitcher_visita": probable_visita.get("id"),
                "hora_inicio_utc": hora_inicio,
            })
    return partidos


def pares_ya_ejecutados(fecha):
    """Pares (local, visita) que el planificador ya marcó como ejecutados
    en data/horarios.json para la fecha.

    El runner no debe re-evaluarlos en ticks posteriores: cada re-evaluación
    en modo ventana reemplaza el pick PENDIENTE con un CreadoUtc nuevo, y si
    ese tick cae después del primer pitch el verifier lo marca NO_VALIDA
    aunque la decisión original sí fue tomada en ventana prejuego.
    """
    ruta = db_utils.RAIZ / "data" / "horarios.json"
    if not ruta.exists():
        return set()
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    ejecutados = set()
    for p in datos.get(fecha.isoformat(), {}).values():
        if p.get("estado") == "ejecutado":
            ejecutados.add((p.get("local"), p.get("visita")))
    return ejecutados


def datos_auxiliares(fecha):
    """Ultimos valores por pitcher/equipo para features de partidos de hoy."""
    whip = db_utils.leer_sql("""
        SELECT PitcherLocalId AS PitcherId, WHIP_Abridor_Local AS WHIP, Fecha
        FROM GameLog WHERE WHIP_Abridor_Local IS NOT NULL
        UNION ALL
        SELECT PitcherVisitaId, WHIP_Abridor_Visita, Fecha
        FROM GameLog WHERE WHIP_Abridor_Visita IS NOT NULL""")
    whip = whip.sort_values("Fecha")
    whips = {fila.PitcherId: fila.WHIP for fila in whip.itertuples()}

    bullpen = db_utils.leer_sql("""
        SELECT EquipoLocal AS Equipo, ERA_Bullpen_Local AS ERA, Fecha
        FROM GameLog WHERE ERA_Bullpen_Local IS NOT NULL
        UNION ALL
        SELECT EquipoVisita, ERA_Bullpen_Visita, Fecha
        FROM GameLog WHERE ERA_Bullpen_Visita IS NOT NULL""")
    bullpen = bullpen.sort_values("Fecha")
    eras = {fila.Equipo: fila.ERA for fila in bullpen.itertuples()}

    tabla_manos = db_utils.leer_sql(
        "SELECT PitcherId, Mano FROM PitcherMano")
    manos = dict(zip(tabla_manos["PitcherId"], tabla_manos["Mano"]))
    tabla_parques = db_utils.leer_sql(
        "SELECT EquipoLocal, Factor_Carreras FROM ParkFactors")
    parques = dict(zip(tabla_parques["EquipoLocal"],
                       tabla_parques["Factor_Carreras"]))

    # Fatiga de bullpen de 72 horas: suma de pitcheos de relevistas
    # (IsStarter = 0) en los 3 dias calendario anteriores a la fecha.
    tabla_fatiga = db_utils.leer_sql("""
        SELECT sub.Team, SUM(sub.ReliefPitches) AS Fatiga
        FROM (
            SELECT pgl.Team, pgl.Fecha,
                   SUM(pgl.PitchesThrown) AS ReliefPitches
            FROM PitcherGameLog pgl
            WHERE pgl.IsStarter = 0
              AND pgl.Fecha >= DATEADD(DAY, -3, ?)
              AND pgl.Fecha < ?
            GROUP BY pgl.Team, pgl.Fecha
        ) sub
        GROUP BY sub.Team""",
        params=[fecha, fecha])
    fatigas = dict(zip(tabla_fatiga["Team"], tabla_fatiga["Fatiga"]))
    return whips, eras, manos, parques, fatigas


def construir_partidos_hoy(partidos_mlb, whips, eras, manos, parques, fatigas, fecha):
    """Filas sinteticas de hoy con el mismo esquema de cargar_datos()."""
    columnas = ["Id", "Fecha", "Estadio", "EquipoLocal", "EquipoVisita",
                "PitcherLocalId", "PitcherVisitaId", "CarrerasLocal",
                "CarrerasVisita", "TemperaturaC", "Viento_Velocidad",
                "Viento_Direccion", "ERA_Bullpen_Local", "ERA_Bullpen_Visita",
                "WHIP_Abridor_Local", "WHIP_Abridor_Visita", "UmpireNombre",
                "UmpireHomePlate", "Factor_Carreras",
                "OPSvsLHP_Local", "OPSvsRHP_Local",
                "OPSvsLHP_Visita", "OPSvsRHP_Visita",
                "ManoAbridorLocal", "ManoAbridorVisita",
                "Fatiga_Bullpen_3d_Local", "Fatiga_Bullpen_3d_Visita",
                "local", "visita", "estadio",
                "pitcher_local", "pitcher_visita"]
    filas = []
    for partido in partidos_mlb:
        if not partido["pitcher_local"] or not partido["pitcher_visita"]:
            continue
        filas.append({
            "Id": None,
            "Fecha": fecha,
            "Estadio": partido["estadio"],
            "EquipoLocal": partido["local"],
            "EquipoVisita": partido["visita"],
            "PitcherLocalId": partido["pitcher_local"],
            "PitcherVisitaId": partido["pitcher_visita"],
            "CarrerasLocal": None,
            "CarrerasVisita": None,
            "TemperaturaC": None,
            "Viento_Velocidad": None,
            "Viento_Direccion": "ND",
            "ERA_Bullpen_Local": eras.get(partido["local"]),
            "ERA_Bullpen_Visita": eras.get(partido["visita"]),
            "WHIP_Abridor_Local": whips.get(partido["pitcher_local"]),
            "WHIP_Abridor_Visita": whips.get(partido["pitcher_visita"]),
            "UmpireNombre": None,
            "UmpireHomePlate": None,
            "Factor_Carreras": parques.get(partido["local"]),
            "OPSvsLHP_Local": None,
            "OPSvsRHP_Local": None,
            "OPSvsLHP_Visita": None,
            "OPSvsRHP_Visita": None,
            "ManoAbridorLocal": manos.get(partido["pitcher_local"]),
            "ManoAbridorVisita": manos.get(partido["pitcher_visita"]),
            "Fatiga_Bullpen_3d_Local": fatigas.get(partido["local"], 0),
            "Fatiga_Bullpen_3d_Visita": fatigas.get(partido["visita"], 0),
            "local": partido["local"],
            "visita": partido["visita"],
            "estadio": partido["estadio"],
            "pitcher_local": partido["pitcher_local"],
            "pitcher_visita": partido["pitcher_visita"],
        })
    return pd.DataFrame(filas, columns=columnas)


def guardar_predicciones(jugadas, fecha, reemplazar_pares=None):
    """Registra las jugadas en Predicciones (upsert, Estado=PENDIENTE).

    Comportamiento por modo:
    - Modo normal (reemplazar_pares=None): se eliminan TODOS los picks
      PENDIENTE del dia antes de guardar (aunque no haya jugadas), porque
      la linea del mercado pudo cambiar entre runs y la recomendacion
      anterior quedo obsoleta. Las ya resueltas se conservan.
    - Modo ventana (reemplazar_pares=lista de pares (local, visita)):
      solo se re-evaluan los partidos que siguen dentro de la ventana
      pre-juego: se retira su PENDIENTE anterior y se guarda la decision
      con la linea mas fresca. Los picks de partidos ya fuera de la
      ventana (iniciados) NO se tocan.
    """
    con = db_utils.conexion()
    try:
        if reemplazar_pares is not None:
            for local, visita in reemplazar_pares:
                con.execute(
                    "DELETE FROM Predicciones "
                    "WHERE Fecha = ? AND Estado = 'PENDIENTE' "
                    "  AND EquipoLocal = ? AND EquipoVisita = ?",
                    [fecha, local, visita])
        else:
            con.execute(
                "DELETE FROM Predicciones WHERE Fecha = ? AND Estado = 'PENDIENTE'",
                [fecha])
        if not jugadas:
            con.commit()
            return 0
        for (local, visita, stake, tipo_apuesta, linea, edge, cuota) in jugadas:
            existe = con.execute(
                "SELECT 1 FROM Predicciones WHERE Fecha = ? AND EquipoLocal = ? "
                "AND EquipoVisita = ? AND TipoApuesta = ?",
                [fecha, local, visita, tipo_apuesta]).fetchone()
            if existe:
                con.execute("""
                    UPDATE Predicciones
                    SET Linea = ?, Unidades = ?, Edge = ?, Cuota = ?,
                        Estado = CASE WHEN EXISTS (
                            SELECT 1 FROM GameLog g
                            WHERE g.Fecha = Predicciones.Fecha
                              AND g.EquipoLocal = Predicciones.EquipoLocal
                              AND g.EquipoVisita = Predicciones.EquipoVisita
                              AND g.EsFinal = 1)
                            THEN Predicciones.Estado
                            ELSE 'PENDIENTE' END
                    WHERE Fecha = ? AND EquipoLocal = ?
                      AND EquipoVisita = ? AND TipoApuesta = ?""",
                    [linea, stake, edge, cuota,
                     fecha, local, visita, tipo_apuesta])
            else:
                con.execute("""
                    INSERT INTO Predicciones
                        (Fecha, EquipoLocal, EquipoVisita, TipoApuesta,
                         Linea, Unidades, Edge, Cuota, Estado, CreadoUtc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDIENTE', ?)""",
                    [fecha, local, visita, tipo_apuesta,
                     linea, stake, edge, cuota,
                     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")])
        con.commit()
        return len(jugadas)
    finally:
        con.close()


def fecha_desde_args():
    for i, arg in enumerate(sys.argv):
        if arg == "--fecha" and i + 1 < len(sys.argv):
            try:
                return date.fromisoformat(sys.argv[i + 1])
            except ValueError:
                print(f"Argumento invalido '{sys.argv[i + 1]}' para --fecha; "
                      "se usara la fecha actual.")
    return date.today()


def ventana_desde_args():
    """Minutos de ventana pre-juego (0 = pronostico de todo el dia)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--ventana-min" and i + 1 < len(sys.argv):
            try:
                return max(0, int(sys.argv[i + 1]))
            except ValueError:
                print(f"Argumento invalido '{sys.argv[i + 1]}' para "
                      "--ventana-min; se usara 0 (todo el dia).")
    return 0


def recuperar_desde_args():
    """Minutos despues del inicio en que un partido aun se evalua
    (recuperacion por huecos del cron; 0 = solo pre-juego)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--recuperar-min" and i + 1 < len(sys.argv):
            try:
                return max(0, int(sys.argv[i + 1]))
            except ValueError:
                print(f"Argumento invalido '{sys.argv[i + 1]}' para "
                      "--recuperar-min; se usara 0.")
    return 0


def linea_en_media_entera(linea, tipo_apuesta):
    """Convierte una linea ENTERA (8.0) a media linea (.5) a favor del pick.

    Regla: OVER -> linea - 0.5 | UNDER -> linea + 0.5. Asi TODA apuesta
    recomendada termina en .5 (sin posibilidad de push) y la zona del
    empate (P(total = 8) ~ 7.4% medido en 10k juegos) queda del lado de
    la apuesta: un 4-4 pasa a ser GANADA en OVER 7.5 en vez de devolucion.
    """
    if linea is None:
        return linea
    linea = float(linea)
    if abs(linea - round(linea)) > 1e-9:
        return linea  # ya es media linea
    if tipo_apuesta == "OVER":
        return linea - 0.5
    if tipo_apuesta == "UNDER":
        return linea + 0.5
    return linea


def guardar_evaluaciones(evaluaciones, fecha):
    """Registra el resultado de CADA partido evaluado (APOSTAR o NO APOSTAR)
    en la tabla Evaluaciones, para que la web muestre el panorama completo
    de los picks que no pasaron los filtros. Upsert por (Fecha, local, visita):
    la linea de la evaluacion mas reciente gana."""
    con = db_utils.conexion()
    try:
        for (local, visita, linea, prediccion, prob_over, edge,
             recomendacion, motivo) in evaluaciones:
            con.execute(
                "DELETE FROM Evaluaciones "
                "WHERE Fecha = ? AND EquipoLocal = ? AND EquipoVisita = ?",
                [fecha, local, visita])
            con.execute(
                "INSERT INTO Evaluaciones "
                "(Fecha, EquipoLocal, EquipoVisita, Linea, Prediccion, "
                " ProbOver, Edge, Recomendacion, Motivo, EvaluadoUtc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [fecha, local, visita, linea, prediccion, prob_over, edge,
                 recomendacion, motivo,
                 datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")])
        con.commit()
        return len(evaluaciones)
    finally:
        con.close()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fecha = fecha_desde_args()
    ventana_min = ventana_desde_args()
    recuperar_min = recuperar_desde_args()
    retroactivo = fecha < date.today()
    modo_etiqueta = f"ventana {ventana_min} min pre-juego" if ventana_min > 0 \
        else "todo el dia"
    fuente = "SQLite (mlb.db)" if db_utils.usar_sqlite() else "SQL Server"
    print(f"BD fuente: {fuente} | modo: {modo_etiqueta}")

    # La API Key solo se requiere en modo "todo el dia" (The Odds API).
    # En modo ventana las lineas salen de LineaSnapshots (sin gastar cuota).
    api_key = obtener_api_key()

    # El calendario se consulta SIEMPRE primero: StatsAPI es gratuita.
    # En modo ventana, si no hay partidos que inicien dentro de la
    # ventana la corrida termina sin tocar The Odds API (ahorro de cuota).
    print(f"[1/6] Obteniendo calendario MLB ({fecha.isoformat()}, "
          f"{modo_etiqueta}) de StatsAPI...")
    try:
        partidos_mlb = obtener_calendario(
            fecha,
            solo_no_iniciados=not retroactivo and ventana_min <= 0,
            ventana_min=ventana_min if ventana_min > 0 and not retroactivo else 0,
            recuperar_min=recuperar_min if ventana_min > 0 and not retroactivo else 0)
    except requests.RequestException as ex:
        print(f"[ERROR] MLB StatsAPI: {ex}")
        return 1
    print(f"      {len(partidos_mlb)} partidos programados.")
    if not partidos_mlb:
        print("      (sin partidos para evaluar en esta corrida)")
        return 0

    # Modo ventana: saltar los partidos que el planificador ya evaluó en
    # un tick previo (estado "ejecutado" en horarios.json). Sin este
    # filtro el pick se re-inserta con CreadoUtc fresco (bug NO_VALIDA).
    if ventana_min > 0 and not retroactivo:
        ya_ejecutados = pares_ya_ejecutados(fecha)
        if ya_ejecutados:
            antes = len(partidos_mlb)
            partidos_mlb = [p for p in partidos_mlb
                            if (p["local"], p["visita"]) not in ya_ejecutados]
            omitidos = antes - len(partidos_mlb)
            if omitidos:
                print(f"      {omitidos} partido(s) ya evaluados en un tick "
                      "previo (horarios.json): omitidos (no se re-inserta "
                      "su pick).")
            if not partidos_mlb:
                print("      (todos los partidos de la ventana ya fueron "
                      "evaluados en ticks previos)")
                return 0

    if retroactivo or ventana_min > 0:
        # Fecha pasada o runner de ventana: se usan los snapshots que el
        # ETL captura a lo largo del dia. En modo ventana esto evita gastar
        # la cuota gratuita de The Odds API en cada corrida de 15 minutos
        # (la linea de cada juego se tomo en el ultimo snapshot del ETL).
        print(f"[2/6] Obteniendo lineas Totals del snapshot "
              f"({fecha.isoformat()}, LineaSnapshots)...")
        lineas = obtener_lineas_snapshot(fecha)
        if not lineas:
            print("      (sin snapshot para esa fecha en LineaSnapshots)")
            return 0
        print(f"      {len(lineas)} partidos con linea Over/Under "
              "en el snapshot.")
    else:
        if not api_key:
            print(f"[ERROR] No se encontro la API Key de The Odds API "
                  f"(variable {API_KEY_VAR}, .env o appsettings.json).")
            return 1
        print(f"[2/6] Obteniendo lineas Totals de The Odds API ({REGION})...")
        try:
            lineas = obtener_lineas(api_key)
        except requests.RequestException as ex:
            print(f"[ERROR] The Odds API: {ex}")
            return 1
        print(f"      {len(lineas)} partidos con linea Over/Under disponible.")
    if not lineas:
        print("      (sin lineas disponibles; reintenta mas tarde)")
        return 0

    print("[3/6] Cargando modelo clasificador y columnas (predecir_hoy)...")
    modelo = joblib.load(MODELO_CLASIFICADOR_PATH)
    columnas = joblib.load(COLUMNAS_CLASIFICADOR_PATH)
    transformadores = joblib.load(TRANSFORMADORES_CLASIFICADOR_PATH)

    print("[4/6] Construyendo features con el pipeline de predecir_hoy...")
    df_raw = cargar_datos(solo_con_temperatura=False)
    whips, eras, manos, parques, fatigas = datos_auxiliares(fecha)
    df_hoy = construir_partidos_hoy(
        partidos_mlb, whips, eras, manos, parques, fatigas, fecha)
    sin_abridor = len(partidos_mlb) - len(df_hoy)
    if sin_abridor:
        print(f"      {sin_abridor} partido(s) sin abridor probable: "
              "no evaluables.")
    if df_hoy.empty:
        print("      (sin partidos con abridores probables para predecir)")
        return 0

    df_raw = pd.concat([df_raw, df_hoy], ignore_index=True)
    df_raw["Fecha"] = pd.to_datetime(df_raw["Fecha"])
    df_raw = df_raw.drop_duplicates(
        subset=["Fecha", "EquipoLocal", "EquipoVisita"], keep="last")
    for columna in ("TemperaturaC", "Viento_Velocidad",
                    "CarrerasLocal", "CarrerasVisita",
                    "WHIP_Abridor_Local", "WHIP_Abridor_Visita",
                    "ERA_Bullpen_Local", "ERA_Bullpen_Visita",
                    "Factor_Carreras", "PitcherLocalId", "PitcherVisitaId",
                    "Fatiga_Bullpen_3d_Local", "Fatiga_Bullpen_3d_Visita"):
        df_raw[columna] = pd.to_numeric(df_raw[columna], errors="coerce")
    df_raw["Viento_Direccion"] = df_raw["Viento_Direccion"].fillna("ND")
    for columna_whip in ("WHIP_Abridor_Local", "WHIP_Abridor_Visita"):
        df_raw[columna_whip] = df_raw[columna_whip].fillna(
            WHIP_UMBRAL_VOLATILIDAD)

    decodificadores = construir_decodificadores(df_raw)

    passthrough = df_raw[["Fecha", "EquipoLocal", "EquipoVisita"]].copy()
    passthrough["Viento_Direccion"] = df_raw["Viento_Direccion"]
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
    print(f"      {len(df)} filas procesadas (historico + partidos de hoy).")

    partidos = df[df["Fecha"].dt.date == fecha]
    # Solo partidos SIN resultado: el ETL ya pudo haber cargado los
    # juegos finalizados del dia en GameLog, y sus filas historicas
    # (con CarrerasLocal reales) no deben volver a predecirse ni
    # registrarse como apuestas.
    partidos = partidos[partidos["CarrerasLocal"].isna()]
    # Modo ventana: evaluar SOLO los partidos dentro de la ventana
    # pre-juego (los del calendario filtrado). Los demas juegos del dia
    # (cargados por el ETL en GameLog) se ignoran hasta que les toque
    # su ventana.
    if ventana_min > 0:
        pares_ventana = {(p["local"], p["visita"]) for p in partidos_mlb}
        nombres_equipos = pd.DataFrame({
            "local": decodificadores["EquipoLocal"].inverse_transform(
                partidos["EquipoLocal"]),
            "visita": decodificadores["EquipoVisita"].inverse_transform(
                partidos["EquipoVisita"]),
        }, index=partidos.index)
        en_ventana = nombres_equipos.apply(
            lambda f: (f["local"], f["visita"]) in pares_ventana, axis=1)
        partidos = partidos[en_ventana.values]
        print(f"      {len(partidos)} partido(s) en ventana "
              "pre-juego de la evaluacion.")
    partidos = partidos.merge(
        passthrough, on=["Fecha", "EquipoLocal", "EquipoVisita"], how="left")

    # Senal C del ENSEMBLE: proyeccion Skellam por equipo. Se calcula sobre
    # una COPIA separada porque ambas proyecciones crean las columnas
    # Anotadas10*/Permitidas10* y los merges colisionarian.
    df_sk = partidos.copy()
    df_sk = expected_runs_por_lado(df_sk, df)
    df_sk = aplicar_ajustes_por_lado(df_sk)
    partidos["ExpRunsTotal"] = (
        df_sk["ExpRunsLocal"] + df_sk["ExpRunsVisita"]).values

    partidos = calcular_expected_runs(partidos, df)

    # Linea REAL del mercado (The Odds API) como Linea_Casino de cada partido.
    linea_por_equipo_local_y_cuota = {
        par[0]: detalle for par, detalle in lineas.items()}
    linea_por_equipo_local = {
        par: detalle["linea"]
        for par, detalle in linea_por_equipo_local_y_cuota.items()}
    nombres_local = pd.Series(
        decodificadores["EquipoLocal"].inverse_transform(partidos["EquipoLocal"]),
        index=partidos.index)
    partidos["Linea_Casino"] = (
        nombres_local.map(linea_por_equipo_local).astype(float))
    partidos["Cuota_Over"] = nombres_local.map(
        lambda par: (linea_por_equipo_local_y_cuota.get(par) or {}).get(
            "cuota_over"))
    partidos["Cuota_Under"] = nombres_local.map(
        lambda par: (linea_por_equipo_local_y_cuota.get(par) or {}).get(
            "cuota_under"))
    sin_linea = int(partidos["Linea_Casino"].isna().sum())
    partidos = partidos.dropna(subset=["Linea_Casino"])
    if partidos.empty:
        print("      (ningun partido con linea disponible y abridor probable)")
        return 0
    print(f"      {len(partidos)} partidos con linea en vivo "
          f"(sin linea: {sin_linea}).")

    partidos = aplicar_ajustes_y_edge(partidos)

    partidos_por_dia = df["Fecha"].dt.date.value_counts()
    inercia_rota = any(
        int(partidos_por_dia.get(fecha - timedelta(days=k), 0)) <= MIN_PARTIDOS_DESCANSO
        for k in range(1, DIAS_REVISION_DESCANSO + 1))
    if inercia_rota:
        print("      Post-descanso prolongado (inercia rota): "
              "stakes limitados a 0.5 Unidades")

    X_hoy = construir_caracteristicas_finales(partidos, transformadores)
    predicciones_proba = modelo.predict_proba(X_hoy)[:, 1]

    print("[5/6] Evaluando con la logica del 59% (filtros defensivos)...")
    print()
    ancho_partido = 40
    print(" " + "-" * 132)
    print(f" {'Partido'.ljust(ancho_partido)} | {'Linea Vegas':<11} | "
          f"{'Prediccion':<10} | {'P(Over)':<8} | {'Edge':<7} | "
          f"{'Recomendacion':<13} | Stake")
    print(" " + "-" * 132)

    totales = Counter()
    jugadas = []
    evaluaciones = []
    for (_, fila), prob_over in zip(partidos.iterrows(), predicciones_proba):
        decision = decidir_jugada(
            fila, prob_over, media_carreras_estadio, partidos_por_dia,
            fecha, inercia_rota, decodificadores)
        linea_mercado = float(fila["Linea_Casino"])
        prediccion = float(fila["Expected_Runs_Ajustada"])
        sugerencia = decision["sugerencia"]
        stake = decision["stake"]
        motivo = decision["motivo_anulacion"]

        if sugerencia == SUGERENCIA_OVER:
            tipo_apuesta = "OVER"
        elif sugerencia == SUGERENCIA_UNDER:
            tipo_apuesta = "UNDER"
        else:
            tipo_apuesta = None
        if tipo_apuesta is not None:
            recomendacion = f"APOSTAR {tipo_apuesta}"
        else:
            recomendacion = "NO APOSTAR"
            if not motivo:
                motivo = (f"Sin senal direccional concluyente "
                          f"(desacuerdo {fila['Diferencia']:+.2f} carreras "
                          f"vs umbral {EDGE_MINIMO:.2f})")
        totales[recomendacion] += 1
        cuota = None
        if tipo_apuesta == "OVER":
            cuota = linea_por_equipo_local_y_cuota[decision["local"]]["cuota_over"]
        elif tipo_apuesta == "UNDER":
            cuota = linea_por_equipo_local_y_cuota[decision["local"]]["cuota_under"]

        # REGLA .5: toda apuesta recomendada termina en media linea.
        linea = linea_en_media_entera(linea_mercado, tipo_apuesta)
        edge_apuesta = abs(prediccion - linea) if linea is not None \
            else abs(fila["Edge"])
        if linea is not None and abs(linea - linea_mercado) > 1e-9:
            print(f"   [REGLA .5] {decision['local']} vs "
                  f"{decision['visita']}: linea entera {linea_mercado:.1f} "
                  f"-> recomendada {linea:.1f} ({tipo_apuesta})")
        if stake is not None:
            jugadas.append(
                (decision["local"], decision["visita"], stake,
                 tipo_apuesta, linea, edge_apuesta, cuota))
        evaluaciones.append(
            (decision["local"], decision["visita"],
             float(linea_mercado), prediccion, float(prob_over),
             float(abs(fila["Edge"])), recomendacion, motivo or ""))

        partido = f"{decision['local']} vs {decision['visita']}"
        detalle = ""
        if motivo:
            detalle = f"  [{motivo}]"
        print(f" {partido.ljust(ancho_partido)} | {linea:<11.1f} | "
              f"{prediccion:<10.2f} | {prob_over * 100:<7.0f}% | "
              f"{abs(fila['Edge']):<7.2f} | "
              f"{recomendacion:<13} | {stake if stake is not None else '-'}"
              f"{detalle}")
    print(" " + "-" * 132)
    print()

    print("[6/6] Resumen...")
    print(f"Partidos con linea (fuente): {len(lineas)}")
    print(f"Partidos MLB {modo_etiqueta}: {len(partidos_mlb)} "
          f"(sin abridor: {sin_abridor})")
    print(f"Evaluados con pipeline del 59%: {len(partidos)} "
          f"(sin linea en vivo: {sin_linea})")
    print(f"Recomendaciones -> OVER: {totales['APOSTAR OVER']} | "
          f"UNDER: {totales['APOSTAR UNDER']} | "
          f"NO APOSTAR: {totales['NO APOSTAR']}")
    if jugadas:
        print("Jugadas sugeridas:")
        for local, visita, stake, tipo_apuesta, linea, _edge, cuota in jugadas:
            cuota_txt = f" @ {cuota:.2f}" if cuota else ""
            print(f"   {stake:.1f} Unidades -> {local} vs {visita} "
                  f"| APUESTA: {tipo_apuesta} a la línea de {linea:.1f}"
                  f"{cuota_txt}")
    if retroactivo:
        print(f"[SIN GUARDAR] modo retroactivo: los pronosticos del "
              f"{fecha.isoformat()} no se registran en dbo.Predicciones.")
        return 0
    reemplazar_pares = None
    if ventana_min > 0:
        # Solo los partidos dentro de la ventana se re-evaluan y
        # reemplazan; los de fuera (ya iniciados) se conservan.
        reemplazar_pares = [
            (p["local"], p["visita"]) for p in partidos_mlb]
    registradas = guardar_predicciones(jugadas, fecha, reemplazar_pares)
    print(f"[GUARDADO] {registradas} predicciones registradas "
          f"en dbo.Predicciones.")
    if evaluaciones:
        registradas_eval = guardar_evaluaciones(evaluaciones, fecha)
        print(f"[GUARDADO] {registradas_eval} evaluaciones registradas "
              f"en Evaluaciones (historial NO APOSTAR).")
    print(f"Filtros activos: EDGE_MINIMO={EDGE_MINIMO:.2f} | "
          f"WHIP abridor <= {WHIP_UMBRAL_VOLATILIDAD:.2f} (UNDER) | "
          f"fatiga bullpen | contradiccion | extremos | viento")
    print("==================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
