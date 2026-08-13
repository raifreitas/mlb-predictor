"""Backfill asincrono de WHIP de abridores desde la MLB StatsAPI.

La base no almacena H/BB/IP por lanzador, asi que el WHIP se obtiene de la
API (stats=gameLog por pitcher-temporada) y se calcula de forma LOCAL el
WHIP acumulado en SALIDAS PREVIAS (sin lookahead bias): para cada partido
de dbo.GameLog se usa el WHIP acumulado con las salidas con fecha anterior
a la del juego. La primera salida de la temporada usa el WHIP de temporada.

- asyncio + aiohttp con semaforo (llamadas concurrentes, no un bucle sincrono).
- Cache en JSON para no repetir llamadas en re-ejecuciones.
- Actualiza dbo.GameLog.WHIP_Abridor_Local / WHIP_Abridor_Visita.
"""

import asyncio
import bisect
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
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cache_gamelog_whip.json")
SEMAFORO = 30
TIMEOUT_SEGUNDOS = 30
REINTENTOS = 3
USER_AGENT = "MlbPredictiveBackfill/1.0"

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


async def obtener_gamelog(sesion, semaforo, pitcher_id, temporada):
    """Devuelve lista de (fecha, hits, bb, ip) de las salidas del pitcher."""
    url = (f"{API_BASE_URL}/people/{pitcher_id}/stats"
           f"?stats=gameLog&group=pitching&season={temporada}")
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
            splits = (datos.get("stats") or [{}])[0].get("splits") or []
            salidas = []
            for split in splits:
                stat = split.get("stat") or {}
                hits = stat.get("hits")
                bb = stat.get("baseOnBalls")
                ip = stat.get("inningsPitched")
                fecha = split.get("date")
                if fecha is None or hits is None or bb is None or ip is None:
                    continue
                salidas.append((_parsear_fecha(fecha),
                                float(hits), float(bb), float(ip)))
            salidas.sort(key=lambda x: x[0])
            return salidas
        except (aiohttp.ClientError, asyncio.TimeoutError) as ex:
            if intento + 1 == REINTENTOS:
                return None
            await asyncio.sleep(1.5 * (intento + 1))
        except Exception:
            return None
    return None


def calcular_whip_acumulado(salidas):
    """WHIP por fecha (acumulado en salidas PREVIAS) + WHIP de temporada.

    Devuelve (dict fecha -> whip_previo, whip_temporada). Para la primera
    salida del anio (sin historico previo) se usa el WHIP de temporada.
    """
    total_h = total_bb = 0.0
    total_ip = 0.0
    for _, h, bb, ip in salidas:
        total_h += h
        total_bb += bb
        total_ip += ip
    whip_temporada = ((total_h + total_bb) / total_ip) if total_ip > 0 else None

    por_fecha = {}
    acum_h = acum_bb = 0.0
    acum_ip = 0.0
    for fecha, h, bb, ip in salidas:
        if acum_ip > 0:
            whip_previo = (acum_h + acum_bb) / acum_ip
        else:
            whip_previo = whip_temporada
        por_fecha[fecha] = whip_previo
        acum_h += h
        acum_bb += bb
        acum_ip += ip
    return por_fecha, whip_temporada


def resolver_whip(por_fecha, whip_temporada, fechas_ordenadas, fecha_juego):
    """WHIP para el juego: ultima salida previa; si no hay, WHIP de temporada."""
    indice = bisect.bisect_left(fechas_ordenadas, fecha_juego) - 1
    if indice >= 0:
        return por_fecha[fechas_ordenadas[indice]]
    return whip_temporada


async def descargar_todo(pares_pendientes):
    """Descarga gameLogs con cache JSON; devuelve dict (pid, temp) -> salidas."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                pid, temp = k.split("-")
                cache[(int(pid), int(temp))] = [
                    (_parsear_fecha(fecha), float(h), float(bb), float(ip))
                    for fecha, h, bb, ip in v]
        print(f"[CACHE] {len(cache)} pares pitcher-temporada cargados del cache.")

    faltantes = [par for par in pares_pendientes if par not in cache]
    print(f"[INFO] {len(faltantes)} pares por descargar de la API...")

    if faltantes:
        semaforo = asyncio.Semaphore(SEMAFORO)
        cabeceras = {"User-Agent": USER_AGENT}
        conector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        async with aiohttp.ClientSession(headers=cabeceras,
                                         connector=conector) as sesion:
            inicio = time.time()
            tareas = [obtener_gamelog(sesion, semaforo, pid, temp)
                      for pid, temp in faltantes]
            resultados = await asyncio.gather(*tareas)
            for (pid, temp), salidas in zip(faltantes, resultados):
                if salidas:
                    cache[(pid, temp)] = salidas
            print(f"[API] {len(faltantes)} llamadas en "
                  f"{time.time() - inicio:.1f}s "
                  f"({(len(faltantes) / max(time.time() - inicio, 0.1)):.0f} req/s).")
            no_datos = sum(1 for s in resultados if not s)
            print(f"[API] {no_datos} pares sin datos de la API.")

        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({f"{pid}-{temp}":
                       [[salida[0].isoformat(), salida[1], salida[2], salida[3]]
                        for salida in salidas]
                       for (pid, temp), salidas in cache.items()}, f)
        print(f"[CACHE] Cache guardado en {CACHE_PATH}.")

    return cache


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    connection_string = CONNECTION_STRING_TEMPLATE.format(driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN WHIP_Abridor_Local IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN WHIP_Abridor_Visita IS NOT NULL THEN 1 ELSE 0 END)
        FROM dbo.GameLog""")
    antes = cursor.fetchone()
    print(f"[ANTES] GameLog total: {antes[0]} | WHIP_Local: {antes[1]} | "
          f"WHIP_Visita: {antes[2]}")

    cursor.execute("""
        SELECT Id, Fecha, PitcherLocalId, PitcherVisitaId
        FROM dbo.GameLog
        WHERE WHIP_Abridor_Local IS NULL OR WHIP_Abridor_Visita IS NULL""")
    pendientes = cursor.fetchall()
    print(f"[INFO] {len(pendientes)} partidos con al menos un WHIP pendiente.")

    pares = set()
    for fila in pendientes:
        fecha = _parsear_fecha(fila.Fecha)
        for pid in (fila.PitcherLocalId, fila.PitcherVisitaId):
            if pid:
                pares.add((pid, fecha.year))
    print(f"[INFO] {len(pares)} pares pitcher-temporada a resolver "
          f"({len(pendientes) * 2} valores de columna).")

    cache = asyncio.run(descargar_todo(pares))

    # Pre-procesa whip por fecha para cada par.
    preparados = {}
    for par, salidas in cache.items():
        preparados[par] = calcular_whip_acumulado(salidas)

    actualizados_local = 0
    actualizados_visita = 0
    sin_pitcher = 0
    sin_datos = 0

    updates_local = []
    updates_visita = []

    for fila in pendientes:
        gid, fecha, pid_local, pid_visita = fila.Id, _parsear_fecha(fila.Fecha), \
            fila.PitcherLocalId, fila.PitcherVisitaId
        temporada = fecha.year

        for pid, rol, lista_updates in (
                (pid_local, "local", updates_local),
                (pid_visita, "visita", updates_visita)):
            if not pid:
                sin_pitcher += 1
                continue
            par = (pid, temporada)
            if par not in preparados or preparados[par][1] is None:
                sin_datos += 1
                continue
            por_fecha, whip_temporada = preparados[par]
            fechas = sorted(por_fecha)
            whip = resolver_whip(por_fecha, whip_temporada, fechas, fecha)
            if whip is None:
                sin_datos += 1
                continue
            lista_updates.append((round(whip, 2), gid))

    cursor.fast_executemany = True
    if updates_local:
        cursor.executemany(
            "UPDATE dbo.GameLog SET WHIP_Abridor_Local = ? WHERE Id = ?",
            updates_local)
        actualizados_local = len(updates_local)
    if updates_visita:
        cursor.executemany(
            "UPDATE dbo.GameLog SET WHIP_Abridor_Visita = ? WHERE Id = ?",
            updates_visita)
        actualizados_visita = len(updates_visita)
    conexion.commit()

    cursor.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN WHIP_Abridor_Local IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN WHIP_Abridor_Visita IS NOT NULL THEN 1 ELSE 0 END)
        FROM dbo.GameLog""")
    despues = cursor.fetchone()

    print(f"[RESUMEN] WHIP_Local actualizados: {actualizados_local} | "
          f"WHIP_Visita actualizados: {actualizados_visita}")
    print(f"[INFO] Partidos sin pitcher id: {sin_pitcher} | "
          f"valores sin datos API: {sin_datos}")
    print(f"[DESPUES] GameLog total: {despues[0]} | WHIP_Local: {despues[1]} | "
          f"WHIP_Visita: {despues[2]}")

    cursor.execute("""
        SELECT COUNT(*) FROM dbo.GameLog
        WHERE WHIP_Abridor_Local IS NOT NULL OR WHIP_Abridor_Visita IS NOT NULL""")
    util = cursor.fetchone()[0]
    print(f"[DESPUES] Partidos con al menos un WHIP util: {util} de {despues[0]}")

    cursor.close()
    conexion.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
