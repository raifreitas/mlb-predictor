"""Backfill incremental de BatterGameLog (2023 -> hoy) desde la MLB StatsAPI.

Por cada partido FINALIZADO del calendario descarga el boxscore y guarda:
  - BatterGameLog: una fila por bateador con K/BB/AB/PA y orden al bate.
  - PitcherGameLog: K/BB/BattersFaced de cada lanzador (alimenta la tasa
    del abridor rival en el feature engineering).

Resume donde se quedo (omite GameIds ya presentes en BatterGameLog), por lo
que puede ejecutarse en tandas. Uso:

  python backfill_batter_props.py                 # 2023-01-01 .. hoy
  python backfill_batter_props.py --desde 2024-01-01 --hasta 2024-12-31

El modo --verificar-conteo auditara cuantos partidos finalizados faltan por
cada mes SIN descargar (util para dimensionar antes de correr).
"""
import argparse
import calendar
from datetime import date, timedelta

from config import RUTA_DB
from game_repository import GameRepository
from mlb_data_fetcher import MlbDataFetcher

DEFAULT_DESDE = date(2023, 1, 1)
ESPERA_ENTRE_LLAMADAS_S = 0.05


def _juegos_finalizados_por_mes(fetcher, inicio, fin):
    """[(gamePk, fecha, equipo_local, equipo_visita)] de un rango de fechas."""
    juegos = []
    url = (f"{fetcher._base_url}/schedule?sportId=1&startDate={inicio:%Y-%m-%d}"
           f"&endDate={fin:%Y-%m-%d}")
    try:
        datos = fetcher._get_json(url)
    except Exception as ex:
        print(f"[SCHEDULE] Error {inicio}..{fin}: {ex}")
        return juegos
    for dia in datos.get("dates", []):
        for juego in dia.get("games", []):
            estado = juego.get("status", {}).get("abstractGameState", "")
            if estado.lower() != "final":
                continue
            if juego.get("gameType", "") in ("S", "E", "A"):
                continue
            game_pk = juego.get("gamePk")
            if not game_pk:
                continue
            local = (juego.get("teams", {}).get("home", {})
                     .get("team", {}).get("name", "Desconocido"))
            visita = (juego.get("teams", {}).get("away", {})
                      .get("team", {}).get("name", "Desconocido"))
            fecha = juego.get("officialDate", "")
            if not fecha:
                continue
            juegos.append((game_pk, fecha, local, visita))
    return juegos


def _game_ids_existentes(repo):
    con = repo._con()
    rows = con.execute("SELECT DISTINCT GameId FROM BatterGameLog").fetchall()
    return {r[0] for r in rows}


def procesar_juegos(fetcher, repo, juegos, espera):
    """Descarga boxscores pendientes y hace UPSERT. Devuelve conteos."""
    bat = 0
    pit = 0
    for indice, (game_pk, fecha, local, visita) in enumerate(juegos, 1):
        try:
            filas_bateadores = fetcher.obtener_batter_logs_partido(
                game_pk, fecha, local, visita)
        except Exception as ex:
            print(f"[{indice}/{len(juegos)}] FALLO {game_pk}: {ex}")
            continue
        if not filas_bateadores:
            print(f"[{indice}/{len(juegos)}] SKIP {fecha} "
                  f"{local} vs {visita} ({game_pk})")
        bat += repo.guardar_batter_game_logs(filas_bateadores)

        # Con 1 llamada al boxscore se actualizan K/BB/BF de los lanzadores.
        try:
            filas_pitchers = fetcher.obtener_pitchers_partido(game_pk, fecha)
            pit += repo.guardar_pitcher_game_logs(filas_pitchers)
        except Exception as ex:
            print(f"[{indice}/{len(juegos)}] FALLO pitcheo {game_pk}: {ex}")

        if indice % 25 == 0:
            print(f"  ... {indice}/{len(juegos)} juegos "
                  f"(bateadores={bat}, pitcheo={pit})")
        if espera:
            import time
            time.sleep(espera)
    return bat, pit


def _verificar_conteo(fetcher, desde, hasta):
    import sqlite3
    con = sqlite3.connect(str(RUTA_DB))
    cubiertos = {r[0] for r in
                 con.execute("SELECT DISTINCT GameId FROM BatterGameLog")}
    con.close()
    total = 0
    actual = desde
    while actual <= hasta:
        ultimo = actual.replace(day=calendar.monthrange(
            actual.year, actual.month)[1])
        juegos = _juegos_finalizados_por_mes(fetcher, actual, ultimo)
        pendientes = [j for j in juegos if j[0] not in cubiertos]
        if juegos:
            print(f"{actual:%Y-%m}: {len(juegos)} finalizados, "
                  f"{len(pendientes)} pendientes")
        total += len(pendientes)
        actual = (ultimo + timedelta(days=1)).replace(day=1)
    print(f"\nTOTAL pendientes: {total} boxscores por descargar")
    from datetime import datetime
    horas = total * (0.5 + 0.4) / 3600
    print(f"Estimado: ~{horas:.1f} h (boxscore + throughput API ~1.1s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desde", default=DEFAULT_DESDE.isoformat())
    parser.add_argument("--hasta", default=date.today().isoformat())
    parser.add_argument("--espera", type=float,
                        default=ESPERA_ENTRE_LLAMADAS_S,
                        help="segundos de pausa entre boxscores")
    parser.add_argument("--verificar-conteo", action="store_true",
                        help="solo cuenta lo pendiente, no descarga")
    args = parser.parse_args()

    desde = date.fromisoformat(args.desde)
    hasta = date.fromisoformat(args.hasta)
    fetcher = MlbDataFetcher("https://statsapi.mlb.com/api/v1")

    if args.verificar_conteo:
        _verificar_conteo(fetcher, desde, hasta)
        return 0

    repo = GameRepository(RUTA_DB)
    existentes = _game_ids_existentes(repo)
    print(f"BatterGameLog ya cubre {len(existentes)} partidos.")

    total_bat = total_pit = 0
    actual = desde
    while actual <= hasta:
        ultimo = min(actual.replace(day=calendar.monthrange(
            actual.year, actual.month)[1]), hasta)
        juegos = _juegos_finalizados_por_mes(fetcher, actual, ultimo)
        pendientes = [j for j in juegos if j[0] not in existentes]
        print(f"[MES {actual:%Y-%m %d}] {len(juegos)} finalizados, "
              f"{len(pendientes)} por procesar.")
        bat, pit = procesar_juegos(fetcher, repo, pendientes, args.espera)
        total_bat += bat
        total_pit += pit
        for j in pendientes:
            existentes.add(j[0])
        actual = ultimo + timedelta(days=1)

    repo.cerrar()
    print(f"\nRESUMEN backfill: {total_bat} filas de bateadores, "
          f"{total_pit} filas de pitcheo.")


if __name__ == "__main__":
    raise SystemExit(main())