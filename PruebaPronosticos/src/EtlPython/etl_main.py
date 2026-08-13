"""Main del ETL Python (port de Program.cs).
Modos:
  python etl_main.py 2026-08-12 2026-08-13        carga diaria/historica reciente
  python etl_main.py --solo-odds 2026-08-12 2026-08-13   solo snapshot de cuotas

El modo --historial (backfill 2015-2022) no se porta: la BD ya viene completa
desde SQL Server; este ETL mantiene el dia a dia, igual que los .bat diarios.
"""
import argparse
from datetime import date, datetime, time, timedelta

from config import cargar_config, RUTA_DB
from game_repository import GameRepository
from mlb_data_fetcher import MlbDataFetcher
from odds_fetcher import OddsFetcher, OddsFetcherML, HTTPErrorOdds
from weather_service import WeatherService


def main():
    args = _parsear_args()
    config = cargar_config()
    inicio, fin = args.inicio, args.fin
    skip_historial = args.no_historial

    if args.solo_odds:
        _modo_solo_odds(config, inicio, fin)
        return 0

    print("==================================================")
    print(" MOTOR ETL MLB - CARGA DIARIA (Python/SQLite)")
    print("==================================================")

    fetcher = MlbDataFetcher(config["MlbStatsApiBaseUrl"])
    print(f"[EXTRACCION] Descargando partidos del {inicio} al {fin}...")
    partidos = fetcher.obtener_partidos(inicio, fin)
    print(f"[EXTRACCION] {len(partidos)} partidos obtenidos.")

    clima = WeatherService(config["OpenMeteoArchiveBaseUrl"],
                           config["OpenMeteoForecastBaseUrl"])
    for partido in partidos:
        temp, viento_kmh, viento_dir = clima.obtener_clima(partido["Estadio"],
                                                           partido["Fecha"])
        partido["TemperaturaC"] = temp
        partido["VientoVelocidad"] = viento_kmh
        partido["VientoDireccion"] = (
            f"{round(viento_dir)}" if viento_dir is not None else "ND")
    print(f"[TRANSFORMACION] Clima resuelto para {len(partidos)} partidos.")

    manos = []
    for partido in partidos:
        for campo in ("PitcherLocalId", "PitcherVisitaId"):
            pitcher_id = partido[campo]
            if pitcher_id is None:
                continue
            mano = fetcher.obtener_mano_lanzamiento(pitcher_id)
            if mano:
                manos.append({"PitcherId": pitcher_id, "Mano": mano})
    print(f"[TRANSFORMACION] Mano de lanzamiento resuelta para {len(manos)} abridores.")

    repo = GameRepository(RUTA_DB)
    insertados = repo.insertar_game_logs(partidos)
    print(f"[CARGA] {insertados} registros insertados o actualizados en SQLite.")

    pitcheos = []
    for partido in partidos:
        if partido["EsFinal"] and partido["GamePk"] is not None:
            pitcheos.extend(fetcher.obtener_pitchers_partido(
                partido["GamePk"], partido["Fecha"]))
    print(f"[TRANSFORMACION] {len(pitcheos)} registros de pitcheo por jugador extraidos.")
    pitcheos_guardados = repo.guardar_pitcher_game_logs(pitcheos)
    print(f"[CARGA] {pitcheos_guardados} pitcheos por jugador en PitcherGameLog.")

    manos_guardadas = repo.guardar_pitcher_mano(manos)
    print(f"[CARGA] {manos_guardadas} registros de mano en PitcherMano.")

    splits_guardados = repo.guardar_ops_splits(fetcher.splits_obtenidos)
    print(f"[CARGA] {splits_guardados} registros de OPS por mano en TeamOPS_Handedness.")

    if not config["TheOddsApiKey"]:
        print("[ODDS] AVISO: sin TheOddsApiKey; no se descargaron lineas.")
    else:
        _captura_odds_historicos(config, repo, inicio, fin)
        _captura_ml(config, repo)

    repo.cerrar()
    print(f"BD: {RUTA_DB}")
    return 0


def _modo_solo_odds(config, inicio, fin):
    print("==================================================")
    print(" MOTOR ETL MLB - SOLO SNAPSHOT DE CUOTAS")
    print("==================================================")
    if not config["TheOddsApiKey"]:
        print("[ODDS] AVISO: sin TheOddsApiKey en config; no se descargo snapshot.")
        return
    repo = GameRepository(RUTA_DB)
    odds = OddsFetcher(config["TheOddsHistoricaBaseUrl"], config["TheOddsApiKey"])
    try:
        lineas = odds.obtener_lineas_actuales()
        guardadas = repo.guardar_lineas_historicas(lineas)
        print(f"[ODDS] Snapshot Totals capturado: {len(lineas)} cotizaciones "
              f"(guardadas: {guardadas}).")
    except HTTPErrorOdds as ex:
        print(f"[ODDS] AVISO (captura Totals): {ex}")
    try:
        odds_ml = OddsFetcherML(config["TheOddsHistoricaBaseUrl"],
                                config["TheOddsApiKey"])
        lineas_h2h = odds_ml.obtener_lineas_h2h_actuales()
        h2h_guardados = repo.guardar_lineas_h2h(lineas_h2h)
        print(f"[ODDS-ML] Snapshot Moneyline capturado: {len(lineas_h2h)} "
              f"cotizaciones (guardadas: {h2h_guardados}).")
    except HTTPErrorOdds as ex_ml:
        print(f"[ODDS-ML] AVISO (captura h2h): {ex_ml}")
    try:
        resueltos = repo.resolver_lineas_reales(inicio, fin)
        print(f"[CARGA] {resueltos} partidos finalizados actualizados con "
              f"Linea_Casino_Real, Cuota_Over_Real y Cuota_Under_Real.")
    except Exception as ex:
        print(f"[ODDS] AVISO (resolver cierre): {ex}")
    repo.cerrar()


def _captura_odds_historicos(config, repo, inicio, fin):
    hora_snapshot = _parsear_hora(config["HoraSnapshotOddsUtc"])
    odds = OddsFetcher(config["TheOddsHistoricaBaseUrl"], config["TheOddsApiKey"])
    print(f"[ODDS] Descargando snapshots historicos de Totals "
          f"(snapshot diario {hora_snapshot:%H:%M} UTC)...")
    snapshots = 0
    disponible = True
    dia = inicio
    while dia <= fin and disponible:
        try:
            lineas = odds.obtener_lineas_historicas(dia, hora_snapshot)
            eventos = len({l["EventoId"] for l in lineas})
            snapshots += repo.guardar_lineas_historicas(lineas)
            print(f"[ODDS] {dia}: {eventos} partidos, {len(lineas)} cotizaciones.")
        except HTTPErrorOdds as ex:
            if ex.codigo == 401 and disponible:
                disponible = False
                print("[ODDS] AVISO: el plan de The Odds API no incluye el endpoint "
                      "historico (se requiere plan de pago). Modo respaldo: captura ACTUAL.")
            else:
                print(f"[ODDS] AVISO ({dia}): {ex}")
        dia += timedelta(days=1)
    print(f"[ODDS] {snapshots} cotizaciones historicas insertadas o actualizadas.")
    try:
        resueltos = repo.resolver_lineas_reales(inicio, fin)
        print(f"[CARGA] {resueltos} partidos finalizados actualizados con "
              f"Linea_Casino_Real, Cuota_Over_Real y Cuota_Under_Real.")
    except Exception as ex:
        print(f"[ODDS] AVISO (resolver cierre): {ex}")


def _captura_ml(config, repo):
    odds_ml = OddsFetcherML(config["TheOddsHistoricaBaseUrl"], config["TheOddsApiKey"])
    try:
        lineas_h2h = odds_ml.obtener_lineas_h2h_actuales()
        h2h_guardados = repo.guardar_lineas_h2h(lineas_h2h)
        print(f"[ODDS-ML] Snapshot Moneyline capturado: {len(lineas_h2h)} "
              f"cotizaciones h2h (guardadas: {h2h_guardados}).")
    except HTTPErrorOdds as ex_ml:
        print(f"[ODDS-ML] AVISO (captura h2h): {ex_ml}")


def _parsear_hora(texto):
    try:
        return datetime.strptime(texto, "%H:%M:%S").time()
    except ValueError:
        return time(21, 30, 0)


def _parsear_args():
    parser = argparse.ArgumentParser(description="ETL MLB a SQLite")
    parser.add_argument("fechas", nargs="*", help="inicio [fin] en formato YYYY-MM-DD")
    parser.add_argument("--solo-odds", action="store_true",
                        help="solo captura de cuotas y resolucion de lineas")
    parser.add_argument("--no-historial", action="store_true",
                        help="omitir captura de snapshots historicos de odds")
    args = parser.parse_args()

    if len(args.fechas) >= 1:
        inicio = datetime.strptime(args.fechas[0], "%Y-%m-%d").date()
    else:
        inicio = date.today() - timedelta(days=1)
    fin = (datetime.strptime(args.fechas[1], "%Y-%m-%d").date()
           if len(args.fechas) >= 2 else inicio)
    args.inicio, args.fin = inicio, fin
    return args


if __name__ == "__main__":
    raise SystemExit(main())