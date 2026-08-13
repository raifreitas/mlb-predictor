"""Backfill de PitcherGameLog (pitcheos por jugador) desde la MLB StatsAPI.

La tabla dbo.PitcherGameLog alimenta la vista dbo.vwFatigaBullpen3d (fatiga
de bullpen de 72 horas). Este script rellena la tabla para TODO el historico
de dbo.GameLog:

1. Para cada FECHA distinta descarga el schedule de la StatsAPI y construye
   el mapa (Fecha, Local, Visita, HL, VA) -> gamePk (solo partidos Final).
2. Descarga en paralelo (aiohttp + semaforo) el boxscore de cada partido.
3. Extrae los lanzadores en orden de aparicion (el primero = abridor,
   IsStarter=1) con su numberOfPitches; si no hay relevistas inserta una fila
   semilla (PitcherId=0, IsStarter=0, 0 pitcheos) para la serie diaria.
4. UPSERT idempotente en dbo.PitcherGameLog.

- Cache de schedule por fecha en JSON para no repetir llamadas.
- Los partidos ya cubiertos en PitcherGameLog (ambos equipos de la Fecha) se
  omiten para re-ejecutar en modo incremental.
"""

import asyncio
import json
import os
import sys
import time
from datetime import date, datetime

import aiohttp
import pyodbc

CONNECTION_STRING_TEMPLATE = (
    "DRIVER={{{driver}}};"
    "SERVER=RAI-FREITAS\\SQLEXPRESS;"
    "DATABASE=MLB_Predictive;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

API_BASE_URL = "https://statsapi.mlb.com/api/v1"
CACHE_SCHEDULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache_schedule_fatiga.json")
SEMAFORO = 20
TIMEOUT_SEGUNDOS = 30
REINTENTOS = 3
USER_AGENT = "MlbPredictiveFatigaBackfill/1.0"

DRIVERS_PREFERIDOS = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 13 for SQL Server",
    "SQL Server Native Client 11.0",
]


def obtener_driver_odbc():
    disponibles = pyodbc.drivers()
    for preferido in DRIVERS_PREFERIDOS:
        if preferido in disponibles:
            return preferido
    if disponibles:
        return disponibles[0]
    raise RuntimeError("No se encontro un driver ODBC de SQL Server instalado.")


def _parsear_fecha(valor):
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    return datetime.strptime(str(valor)[:10], "%Y-%m-%d").date()


async def obtener_schedule(sesion, semaforo, fecha):
    """Devuelve lista de tuplas finalizadas: (gamePk, fecha, local, visit, hl, va)."""
    url = (f"{API_BASE_URL}/schedule?sportId=1&startDate={fecha.isoformat()}"
           f"&endDate={fecha.isoformat()}&hydrate=team,probablePitcher,venue")
    for intento in range(REINTENTOS):
        try:
            async with semaforo:
                async with sesion.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEGUNDOS)) as r:
                    if r.status == 429:
                        await asyncio.sleep(2 * (intento + 1))
                        continue
                    r.raise_for_status()
                    datos = await r.json()
            juegos = []
            for fecha_json in datos.get("dates") or []:
                for juego in fecha_json.get("games") or []:
                    estado = (juego.get("status") or {}).get(
                        "abstractGameState") or ""
                    if estado.lower() != "final":
                        continue
                    try:
                        local = juego["teams"]["home"]["team"]["name"]
                        visit = juego["teams"]["away"]["team"]["name"]
                        hl = int(juego.get("teams", {}).get("home", {}).get("score", 0) or 0)
                        va = int(juego.get("teams", {}).get("away", {}).get("score", 0) or 0)
                    except (KeyError, TypeError, ValueError):
                        continue
                    juegos.append((
                        int(juego["gamePk"]), _parsear_fecha(juego["officialDate"]),
                        local, visit, hl, va))
            return juegos
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if intento + 1 == REINTENTOS:
                return None
            await asyncio.sleep(1.5 * (intento + 1))
        except Exception:
            return None
    return None


async def obtener_boxscore(sesion, semaforo, game_id):
    """Devuelve una lista de filas (gameId, team, pid, is_starter, pitches)."""
    url = f"{API_BASE_URL}/game/{game_id}/boxscore"
    for intento in range(REINTENTOS):
        try:
            async with semaforo:
                async with sesion.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEGUNDOS)) as r:
                    if r.status == 429:
                        await asyncio.sleep(2 * (intento + 1))
                        continue
                    r.raise_for_status()
                    datos = await r.json()
            filas = []
            for lado in ("home", "away"):
                lado_json = datos.get("teams", {}).get(lado) or {}
                nombre = ((lado_json.get("team") or {}).get("name")) or "Desconocido"
                pitchers = lado_json.get("pitchers") or []
                if not pitchers:
                    filas.append((int(game_id), nombre, 0, False, 0))
                    continue
                for indice, pid in enumerate(pitchers):
                    n_p = 0
                    jugador = (lado_json.get("players") or {}).get(f"ID{pid}") or {}
                    pitching = (jugador.get("stats") or {}).get("pitching") or {}
                    val = pitching.get("numberOfPitches")
                    if val is not None:
                        try:
                            n_p = int(val)
                        except (TypeError, ValueError):
                            n_p = 0
                    filas.append((int(game_id), nombre, int(pid),
                                  indice == 0, n_p))
            return filas
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if intento + 1 == REINTENTOS:
                return None
            await asyncio.sleep(1.5 * (intento + 1))
        except Exception:
            return None
    return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rango_inicio = sys.argv[1] if len(sys.argv) > 1 else "2023-03-01"
    rango_fin = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()

    connection_string = CONNECTION_STRING_TEMPLATE.format(driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT g.Fecha, g.EquipoLocal, g.EquipoVisita,
               g.CarrerasLocal, g.CarrerasVisita,
                   CASE WHEN EXISTS (
                   SELECT 1 FROM dbo.PitcherGameLog p
                   WHERE p.Fecha = g.Fecha
                     AND p.Team IN (g.EquipoLocal, g.EquipoVisita))
                    THEN 1 ELSE 0 END AS Cubierto
        FROM dbo.GameLog g
        WHERE g.Fecha >= ? AND g.Fecha <= ?""",
                   [rango_inicio, rango_fin])
    juegos = cursor.fetchall()
    print(f"[INFO] {len(juegos)} partidos en el rango {rango_inicio} -> {rango_fin}")

    pendientes = [j for j in juegos if not j.Cubierto]
    print(f"[INFO] {len(pendientes)} partidos sin PitcherGameLog (a procesar).")
    if not pendientes:
        print("[INFO] Nada pendiente. Salida.")
        cursor.close()
        conexion.close()
        return 0

    fechas = sorted({_parsear_fecha(j.Fecha) for j in pendientes})
    print(f"[INFO] {len(fechas)} fechas distintas.")

    cache_schedule = {}
    if os.path.exists(CACHE_SCHEDULE_PATH):
        with open(CACHE_SCHEDULE_PATH, "r", encoding="utf-8") as fh:
            cache_schedule = json.load(fh)
        print(f"[CACHE] Fechas con schedule cacheado: {len(cache_schedule)}.")

    fechas_faltantes = [d for d in fechas
                        if d.isoformat() not in cache_schedule]

    async def descargar():
        semaforo = asyncio.Semaphore(SEMAFORO)
        cabeceras = {"User-Agent": USER_AGENT}
        conector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        async with aiohttp.ClientSession(headers=cabeceras,
                                         connector=conector) as sesion:
            if fechas_faltantes:
                inicio = time.time()
                tareas = [obtener_schedule(sesion, semaforo, d)
                          for d in fechas_faltantes]
                resultados = await asyncio.gather(*tareas)
                for d, juegos in zip(fechas_faltantes, resultados):
                    cache_schedule[d.isoformat()] = [
                        (pk, f.isoformat(), local, visit, hl, va)
                        for (pk, f, local, visit, hl, va) in (juegos or [])]
                print(f"[API] {len(fechas_faltantes)} schedules en "
                      f"{time.time() - inicio:.1f}s.")
                with open(CACHE_SCHEDULE_PATH, "w", encoding="utf-8") as fh:
                    json.dump(cache_schedule, fh)

            juego_a_pk = {}
            for lista in cache_schedule.values():
                for (pk, f, local, visit, hl, va) in lista:
                    juego_a_pk[(_parsear_fecha(f), local, visit, hl, va)] = pk

            resolved = []
            sin_match = 0
            for j in pendientes:
                fecha_j = _parsear_fecha(j.Fecha)
                key = (fecha_j, j.EquipoLocal, j.EquipoVisita,
                       j.CarrerasLocal, j.CarrerasVisita)
                pk = juego_a_pk.get(key)
                if pk is None:
                    sin_match += 1
                else:
                    resolved.append((pk, fecha_j))
            print(f"[INFO] {len(resolved)} partidos con gamePk | "
                  f"{sin_match} sin match en el schedule.")

            pks = list(dict.fromkeys(pk for pk, _ in resolved))
            inicio = time.time()
            tareas = [obtener_boxscore(sesion, semaforo, pk) for pk in pks]
            resultados = await asyncio.gather(*tareas)
            n_ok = sum(1 for r in resultados if r is not None)
            print(f"[API] {len(pks)} boxscores en {time.time() - inicio:.1f}s "
                  f"({len(pks) / max(time.time() - inicio, 0.1):.0f} req/s); "
                  f"{n_ok} con datos.")
        return resultados, pks, resolved

    resultados, pks, resolved = asyncio.run(descargar())
    box_por_pk = {pk: res for pk, res in zip(pks, resultados)
                  if res is not None}

    filas_db = []
    nulos_box = 0
    for pk, fecha_j in resolved:
        filas = box_por_pk.get(pk)
        if not filas:
            nulos_box += 1
            continue
        for (game_id, team, pid, es_abridor, lanzados) in filas:
            filas_db.append((game_id, fecha_j, team, pid,
                             int(es_abridor), int(lanzados)))

    print(f"[INFO] {len(filas_db)} filas de PitcherGameLog a insertar "
          f"({nulos_box} partidos sin datos de boxscore).")

    if filas_db:
        cursor.fast_executemany = True
        cursor.executemany("""
            INSERT INTO dbo.PitcherGameLog
                (GameID, Fecha, Team, PitcherID, IsStarter, PitchesThrown)
            SELECT ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM dbo.PitcherGameLog
                WHERE GameID = ? AND Team = ? AND PitcherID = ?)""",
            [(f[0], f[1], f[2], f[3], f[4], f[5],
              f[0], f[2], f[3]) for f in filas_db])
        conexion.commit()
        print(f"[OK] {len(filas_db)} filas insertadas en dbo.PitcherGameLog.")

    cursor.execute("SELECT COUNT(*) FROM dbo.PitcherGameLog")
    print(f"[RESUMEN] PitcherGameLog total: {cursor.fetchone()[0]}")

    cursor.close()
    conexion.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
