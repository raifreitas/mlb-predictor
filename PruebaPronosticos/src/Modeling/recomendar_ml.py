"""Recomendador de apuestas Moneyline (ganador del partido) en vivo.

Mercado AISLADO del flujo Over/Under: ninguna tabla del flujo de totals
se modifica. Este modulo:

- Obtiene las cuotas h2h ACTUALES de The Odds API (moda de cuota LOCAL
  entre casas) o, en modo retroactivo (--fecha pasada), las modas de
  dbo.LineaSnapshotsML capturadas por el ETL.
- Obtiene el calendario MLB con abridores probables (StatsAPI).
- Construye features con el pipeline de entrenar_modelo_ml.py y calcula
  la proyeccion de carreras por EQUIPO (enfoque por diferencia de
  carreras, ver backtest_skellam_ml.py): P(gana local) = sigmoide sobre
  la diferencia de Expected Runs, recalibrada con isotonica. La decision
  usa los filtros de riesgo de predecir_ml.py (margen, tope de edge,
  Kelly, datos faltantes).
- Registra las jugadas en dbo.PrediccionesML (Estado=PENDIENTE), primero
  eliminando los PENDIENTE del mismo dia (lineas obsoletas entre runs).

Uso:
    python recomendar_ml.py [--fecha YYYY-MM-DD]
"""

import json
import os
import sys
from collections import Counter
from datetime import date
from statistics import median as _mediana

import joblib
import pandas as pd
import pyodbc
import requests
from sklearn.preprocessing import LabelEncoder

from entrenar_modelo import obtener_driver_odbc, CONNECTION_STRING_TEMPLATE
from entrenar_modelo_ml import (
    feature_engineering_bullpen,
    feature_engineering_fatiga,
    feature_engineering_pitchers,
    feature_engineering_rachas,
    preprocesar,
)
from predecir_ml import (
    SUGERENCIA_AWAY,
    SUGERENCIA_HOME,
    decidir_jugada_ml,
    probabilidad_skellam_ml,
)
import predecir_hoy as ph

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.normpath(os.path.join(CARPETA_SCRIPT, "..", "..", "models"))

API_KEY_VAR = "THE_ODDS_API_KEY"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
REGION = "us"
MERCADO = "h2h"
FORMATO_CUOTAS = "decimal"
TIMEOUT_SEGUNDOS = 30


def obtener_api_key():
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


def _normalizar_equipo(nombre):
    """Mismo diccionario de normalizacion que el ETL (OddsFetcherML)."""
    mapa = {
        "L.A. Dodgers": "Los Angeles Dodgers",
        "LA Dodgers": "Los Angeles Dodgers",
        "Chi Cubs": "Chicago Cubs",
        "Chi White Sox": "Chicago White Sox",
        "NY Mets": "New York Mets",
        "NY Yankees": "New York Yankees",
        "S.F. Giants": "San Francisco Giants",
        "S.D. Padres": "San Diego Padres",
        "TB Rays": "Tampa Bay Rays",
    }
    limpio = (nombre or "").strip()
    return mapa.get(limpio, limpio)


def obtener_lineas_h2h(api_key):
    """Cuotas h2h ACTUALES: moda de la cuota LOCAL entre casas.

    Devuelve {(home, away): {"cuota_home": float, "cuota_away": float,
    "casas": int}} con nombres normalizados al estilo del ETL.
    """
    respuesta = requests.get(
        ODDS_BASE_URL,
        params={"apiKey": api_key, "regions": REGION,
                "markets": MERCADO, "oddsFormat": FORMATO_CUOTAS},
        timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()

    lineas = {}
    for evento in respuesta.json():
        home = _normalizar_equipo(evento.get("home_team"))
        away = _normalizar_equipo(evento.get("away_team"))
        por_casa = []  # (cuota_home, cuota_away)
        for casa in evento.get("bookmakers", []):
            for mercado in casa.get("markets", []):
                if mercado.get("key") != MERCADO:
                    continue
                precios = {}
                for resultado in mercado.get("outcomes", []):
                    nombre = resultado.get("name")
                    if resultado.get("price") is not None:
                        precios[_normalizar_equipo(nombre)] = float(resultado["price"])
                if home in precios and away in precios:
                    por_casa.append((precios[home], precios[away]))
        if not por_casa:
            continue
        ch = Counter(p for p, _a in por_casa).most_common(1)[0][0]
        casas_ch = [a for p, a in por_casa if p == ch]
        lineas[(home, away)] = {
            "cuota_home": ch,
            "cuota_away": _mediana(casas_ch),
            "casas": len(por_casa)}
    return lineas


def obtener_lineas_snapshot_ml(fecha):
    """Cuotas h2h de una fecha PASADA desde dbo.LineaSnapshotsML.

    Moda de cuota LOCAL entre casas y mediana de la cuota visita de las
    casas que ofrecen esa moda. Mismo formato que obtener_lineas_h2h().
    """
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        filas = pd.read_sql("""
            SELECT EquipoLocal, EquipoVisita, CuotaHome, CuotaAway
            FROM dbo.LineaSnapshotsML
            WHERE Fecha = ? AND CuotaHome IS NOT NULL AND CuotaAway IS NOT NULL
            ORDER BY EquipoLocal, EquipoVisita, CapturadoUtc""",
            conexion, params=[fecha])
    finally:
        conexion.close()
    if filas.empty:
        return {}

    lineas = {}
    for (local, visita), grupo in filas.groupby(
            ["EquipoLocal", "EquipoVisita"], sort=False):
        cuotas_home = [float(c) for c in grupo["CuotaHome"].dropna()]
        if not cuotas_home:
            continue
        ch = Counter(cuotas_home).most_common(1)[0][0]
        en_moda = grupo[grupo["CuotaHome"].astype(float) == ch]
        away = [float(c) for c in en_moda["CuotaAway"].dropna()]
        lineas[(local, visita)] = {
            "cuota_home": ch,
            "cuota_away": _mediana(away) if away else None,
            "casas": len(grupo)}
    return lineas


def obtener_calendario(fecha):
    respuesta = requests.get(
        f"{MLB_BASE_URL}/schedule",
        params={"sportId": 1, "date": fecha.isoformat(),
                "hydrate": "probablePitcher,venue,team"},
        timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()

    partidos = []
    for dia in respuesta.json().get("dates", []):
        for juego in dia.get("games", []):
            estado = (juego.get("status") or {}).get("detailedState", "")
            if estado in ("Postponed", "Cancelled"):
                continue
            local = juego["teams"]["home"]["team"]
            visita = juego["teams"]["away"]["team"]
            probable_local = juego["teams"]["home"].get("probablePitcher") or {}
            probable_visita = juego["teams"]["away"].get("probablePitcher") or {}
            partidos.append({
                "local": local.get("fullName") or local.get("name"),
                "visita": visita.get("fullName") or visita.get("name"),
                "estadio": (juego.get("venue") or {}).get("name"),
                "pitcher_local": probable_local.get("id"),
                "pitcher_visita": probable_visita.get("id"),
            })
    return partidos


def datos_auxiliares(fecha):
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        whip = pd.read_sql("""
            SELECT PitcherLocalId AS PitcherId, WHIP_Abridor_Local AS WHIP, Fecha
            FROM dbo.GameLog WHERE WHIP_Abridor_Local IS NOT NULL
            UNION ALL
            SELECT PitcherVisitaId, WHIP_Abridor_Visita, Fecha
            FROM dbo.GameLog WHERE WHIP_Abridor_Visita IS NOT NULL""",
            conexion)
        whip = whip.sort_values("Fecha")
        whips = {fila.PitcherId: fila.WHIP for fila in whip.itertuples()}

        bullpen = pd.read_sql("""
            SELECT EquipoLocal AS Equipo, ERA_Bullpen_Local AS ERA, Fecha
            FROM dbo.GameLog WHERE ERA_Bullpen_Local IS NOT NULL
            UNION ALL
            SELECT EquipoVisita, ERA_Bullpen_Visita, Fecha
            FROM dbo.GameLog WHERE ERA_Bullpen_Visita IS NOT NULL""",
            conexion)
        bullpen = bullpen.sort_values("Fecha")
        eras = {fila.Equipo: fila.ERA for fila in bullpen.itertuples()}

        tabla_manos = pd.read_sql(
            "SELECT PitcherId, Mano FROM dbo.PitcherMano", conexion)
        manos = dict(zip(tabla_manos["PitcherId"], tabla_manos["Mano"]))
        tabla_parques = pd.read_sql(
            "SELECT EquipoLocal, Factor_Carreras FROM dbo.ParkFactors", conexion)
        parques = dict(zip(tabla_parques["EquipoLocal"],
                           tabla_parques["Factor_Carreras"]))

        tabla_fatiga = pd.read_sql("""
            SELECT sub.Team, SUM(sub.ReliefPitches) AS Fatiga
            FROM (
                SELECT pgl.Team, pgl.Fecha,
                       SUM(pgl.PitchesThrown) AS ReliefPitches
                FROM dbo.PitcherGameLog pgl
                WHERE pgl.IsStarter = 0
                  AND pgl.Fecha >= DATEADD(DAY, -3, ?)
                  AND pgl.Fecha < ?
                GROUP BY pgl.Team, pgl.Fecha
            ) sub
            GROUP BY sub.Team""",
            conexion, params=[fecha, fecha])
        fatigas = dict(zip(tabla_fatiga["Team"], tabla_fatiga["Fatiga"]))
        return whips, eras, manos, parques, fatigas
    finally:
        conexion.close()


def construir_partidos_hoy(partidos_mlb, whips, eras, manos, parques, fatigas, fecha):
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


def guardar_predicciones_ml(jugadas, fecha):
    """Registra las jugadas en dbo.PrediccionesML (upsert, PENDIENTE).

    Primero elimina los PENDIENTE del mismo dia (lineas obsoletas entre
    runs); las ya resueltas se conservan. TipoApuesta: 'HOME'/'AWAY'.
    """
    if not jugadas:
        return 0
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        conexion.execute(
            "DELETE FROM dbo.PrediccionesML WHERE Fecha = ? AND Estado = 'PENDIENTE'",
            fecha)
        for (local, visita, stake, tipo_apuesta, cuota, edge, prob_modelo) in jugadas:
            conexion.execute("""
                IF EXISTS (SELECT 1 FROM dbo.PrediccionesML
                           WHERE Fecha = ? AND EquipoLocal = ?
                             AND EquipoVisita = ? AND TipoApuesta = ?)
                BEGIN
                    UPDATE dbo.PrediccionesML
                    SET Unidades = ?, Linea = ?, Cuota = ?, Edge = ?,
                        ProbModelo = ?,
                        Estado = CASE WHEN EXISTS (
                            SELECT 1 FROM dbo.GameLog g
                            WHERE g.Fecha = dbo.PrediccionesML.Fecha
                              AND g.EquipoLocal = dbo.PrediccionesML.EquipoLocal
                              AND g.EquipoVisita = dbo.PrediccionesML.EquipoVisita
                              AND g.EsFinal = 1)
                            THEN dbo.PrediccionesML.Estado
                            ELSE 'PENDIENTE' END,
                        FechaVerificacion = NULL
                    WHERE Fecha = ? AND EquipoLocal = ?
                      AND EquipoVisita = ? AND TipoApuesta = ?
                END
                ELSE
                BEGIN
                    INSERT INTO dbo.PrediccionesML
                        (Fecha, EquipoLocal, EquipoVisita, TipoApuesta,
                         Unidades, Linea, Cuota, Edge, ProbModelo, Estado)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDIENTE')
                END""",
                fecha, local, visita, tipo_apuesta,
                stake, cuota, cuota, edge, prob_modelo,
                fecha, local, visita, tipo_apuesta,
                fecha, local, visita, tipo_apuesta,
                stake, cuota, cuota, edge, prob_modelo)
        conexion.commit()
        return len(jugadas)
    finally:
        conexion.close()


def fecha_desde_args():
    for i, arg in enumerate(sys.argv):
        if arg == "--fecha" and i + 1 < len(sys.argv):
            try:
                return date.fromisoformat(sys.argv[i + 1])
            except ValueError:
                print(f"Argumento invalido '{sys.argv[i + 1]}' para --fecha; "
                      "se usara la fecha actual.")
    return date.today()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fecha = fecha_desde_args()
    retroactivo = fecha < date.today()

    api_key = obtener_api_key()
    if not api_key:
        print(f"[ERROR] No se encontro la API Key de The Odds API "
              f"(variable {API_KEY_VAR}, .env o appsettings.json).")
        return 1

    if retroactivo:
        print(f"[1/6] Obteniendo cuotas h2h del snapshot "
              f"({fecha.isoformat()}, LineaSnapshotsML)...")
        lineas = obtener_lineas_snapshot_ml(fecha)
        if not lineas:
            print("      (sin snapshot ML para esa fecha)")
            return 0
        print(f"      {len(lineas)} partidos con cuotas h2h en el snapshot.")
    else:
        print(f"[1/6] Obteniendo cuotas Moneyline de The Odds API ({REGION})...")
        try:
            lineas = obtener_lineas_h2h(api_key)
        except requests.RequestException as ex:
            print(f"[ERROR] The Odds API: {ex}")
            return 1
        print(f"      {len(lineas)} partidos con cuotas h2h disponibles.")
    if not lineas:
        print("      (sin cuotas disponibles; reintenta mas tarde)")
        return 0

    print(f"[2/6] Obteniendo calendario MLB ({fecha.isoformat()}) de StatsAPI...")
    try:
        partidos_mlb = obtener_calendario(fecha)
    except requests.RequestException as ex:
        print(f"[ERROR] MLB StatsAPI: {ex}")
        return 1
    print(f"      {len(partidos_mlb)} partidos programados.")
    if not partidos_mlb:
        print("      (sin partidos programados para esta fecha)")
        return 0

    print("[3/6] Cargando parametros del motor por diferencia de carreras...")
    from backtest_skellam_ml import aplicar_ajustes_por_lado, expected_runs_por_lado

    print("[4/6] Construyendo features con el pipeline de entrenar_modelo_ml...")
    from predecir_ml import cargar_datos_ml
    df_raw = cargar_datos_ml()
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
    df = ph.feature_engineering_ampayer(df)
    df = ph.feature_engineering_descanso_abridor(df)
    df = ph.feature_engineering_matchup(df)
    df = expected_runs_por_lado(df, df)
    df = aplicar_ajustes_por_lado(df)
    print(f"      {len(df)} filas procesadas (historico + partidos de hoy).")

    partidos = df[df["Fecha"].dt.date == fecha].copy()

    linea_por_local = {par[0]: detalle for par, detalle in lineas.items()}
    nombres_local = pd.Series(
        decodificadores["EquipoLocal"].inverse_transform(partidos["EquipoLocal"]),
        index=partidos.index)
    partidos["CuotaHome"] = nombres_local.map(
        lambda par: (linea_por_local.get(par) or {}).get("cuota_home"))
    partidos["CuotaAway"] = nombres_local.map(
        lambda par: (linea_por_local.get(par) or {}).get("cuota_away"))
    sin_linea = int(partidos["CuotaHome"].isna().sum())
    partidos = partidos.dropna(subset=["CuotaHome", "CuotaAway"])
    if partidos.empty:
        print("      (ningun partido con cuotas h2h y abridores probables)")
        return 0
    print(f"      {len(partidos)} partidos con cuotas en vivo "
          f"(sin cuotas: {sin_linea}).")

    nombres_local = pd.Series(
        decodificadores["EquipoLocal"].inverse_transform(partidos["EquipoLocal"]),
        index=partidos.index)
    nombres_visita = pd.Series(
        decodificadores["EquipoVisita"].inverse_transform(partidos["EquipoVisita"]),
        index=partidos.index)

    proba = probabilidad_skellam_ml(
        partidos["ExpRunsLocal"], partidos["ExpRunsVisita"]).values

    print("[5/6] Evaluando con los filtros Moneyline (margen, tope de edge, Kelly)...")
    print()
    ancho = 40
    print(" " + "-" * 128)
    print(f" {'Partido'.ljust(ancho)} | {'P(Local)':<9} | "
          f"{'P Mercado':<10} | {'P Final':<8} | {'Cuotas':<13} | "
          f"{'Recomendacion':<13} | Stake")
    print(" " + "-" * 128)

    totales = Counter()
    jugadas = []
    for (_, fila), prob_local in zip(partidos.iterrows(), proba):
        nombre_local = nombres_local.loc[fila.name]
        nombre_visita = nombres_visita.loc[fila.name]
        cuota_home = float(fila["CuotaHome"])
        cuota_away = float(fila["CuotaAway"])
        decision = decidir_jugada_ml(
            fila, prob_local, cuota_home, cuota_away, decodificadores,
            ya_calibrada=True)
        sugerencia = decision["sugerencia"]
        stake = decision["stake"]
        motivo = decision["motivo_anulacion"]

        if sugerencia == SUGERENCIA_HOME:
            tipo_apuesta = "HOME"
        elif sugerencia == SUGERENCIA_AWAY:
            tipo_apuesta = "AWAY"
        else:
            tipo_apuesta = None
        if tipo_apuesta is not None:
            recomendacion = f"APOSTAR {tipo_apuesta}"
        else:
            recomendacion = "NO APOSTAR"
        totales[recomendacion] += 1

        cuota_pick = cuota_home if tipo_apuesta == "HOME" else cuota_away
        if stake is not None and tipo_apuesta is not None:
            jugadas.append(
                (decision["local"], decision["visita"], stake,
                 tipo_apuesta, cuota_pick, round(decision["desacuerdo"], 4),
                 round(decision["prob_cal"], 4)))

        partido = f"{decision['local']} vs {decision['visita']}"
        detalle = f"  [{motivo}]" if motivo else ""
        print(f" {partido.ljust(ancho)} | {decision['prob_cal'] * 100:<8.0f}% | "
              f"{decision['p_mercado'] * 100:<9.0f}% | "
              f"{decision['prob_decision'] * 100:<7.0f}% | "
              f"{cuota_home:.2f}/{cuota_away:.2f} | "
              f"{recomendacion:<13} | "
              f"{stake if stake is not None else '-'}{detalle}")
    print(" " + "-" * 128)
    print()

    print("[6/6] Resumen...")
    print(f"Partidos con cuotas h2h: {len(lineas)}")
    print(f"Partidos MLB {fecha.isoformat()}: {len(partidos_mlb)} "
          f"(sin abridor: {sin_abridor})")
    print(f"Evaluados: {len(partidos)} (sin cuotas en vivo: {sin_linea})")
    print(f"Recomendaciones -> LOCAL: {totales['APOSTAR HOME']} | "
          f"VISITA: {totales['APOSTAR AWAY']} | "
          f"NO APOSTAR: {totales['NO APOSTAR']}")
    if jugadas:
        print("Jugadas sugeridas:")
        for local, visita, stake, tipo_apuesta, cuota, edge, _prob in jugadas:
            cuota_txt = f" @ {cuota:.2f}" if cuota else ""
            print(f"   {stake:.1f} Unidades -> {local} vs {visita} "
                  f"| APUESTA: {tipo_apuesta}{cuota_txt}")
    if retroactivo:
        print(f"[SIN GUARDAR] modo retroactivo: los pronosticos del "
              f"{fecha.isoformat()} no se registran en dbo.PrediccionesML.")
        return 0
    registradas = guardar_predicciones_ml(jugadas, fecha)
    print(f"[GUARDADO] {registradas} predicciones ML registradas "
          f"en dbo.PrediccionesML.")
    print(f"Filtros ML: margen minimo | tope edge vs mercado | "
          f"Kelly media | datos faltantes")
    print(f"Motor: diferencia de carreras (beta+isotonica) | "
          f"regresion mercado 0.25 | margen 0.07")
    print("==================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
