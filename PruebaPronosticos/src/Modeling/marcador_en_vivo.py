# -*- coding: utf-8 -*-
"""Marcador en vivo MLB + estado de las apuestas del dia.

Consulta la MLB StatsAPI (schedule con linescore) para mostrar el
marcador en tiempo real de los partidos del dia y, para cada pick
registrado en dbo.Predicciones, el estado provisional de la apuesta:

  - FINAL: GANADA/PERDIDA (misma logica que verificar_predicciones;
    la BD se actualiza por la tarea QuickVerify, no aqui).
  - EN VIVO: "GANANDO (asegurado)" / "PERDIENDO (superado)" /
    "faltan X carreras" (OVER) / "margen X carreras" (UNDER).
  - PRE-GAME: sin empezar.

No modifica la base de datos: es solo un visor.

Uso:
    python marcador_en_vivo.py [YYYY-MM-DD]
"""

import datetime
import json
import sys
import urllib.request

import pyodbc

CONNECTION_STRING_TEMPLATE = (
    "DRIVER={{{driver}}};"
    "SERVER=RAI-FREITAS\\SQLEXPRESS;"
    "DATABASE=MLB_Predictive;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

DRIVERS_PREFERIDOS = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 13 for SQL Server",
    "SQL Server Native Client 11.0",
]

SCHEDULE_URL = ("https://statsapi.mlb.com/api/v1/schedule"
                "?sportId=1&date={fecha}&hydrate=linescore,team,venue")

TIEMPO_ESPERA_SEGUNDOS = 30


def obtener_driver_odbc():
    disponibles = pyodbc.drivers()
    for preferido in DRIVERS_PREFERIDOS:
        if preferido in disponibles:
            return preferido
    if disponibles:
        return disponibles[0]
    raise RuntimeError("No se encontro un driver ODBC de SQL Server instalado.")


def obtener_calendario_en_vivo(fecha):
    """Partidos del dia con marcador y estado desde la MLB StatsAPI."""
    url = SCHEDULE_URL.format(fecha=fecha.isoformat())
    with urllib.request.urlopen(url, timeout=TIEMPO_ESPERA_SEGUNDOS) as r:
        datos = json.loads(r.read().decode("utf-8"))
    partidos = []
    for dia in datos.get("dates", []):
        for juego in dia.get("games", []):
            estado = (juego.get("status") or {}).get("detailedState", "")
            if estado in ("Postponed", "Cancelled"):
                continue
            local = (juego["teams"]["home"]["team"].get("fullName")
                     or juego["teams"]["home"]["team"].get("name"))
            visita = (juego["teams"]["away"]["team"].get("fullName")
                      or juego["teams"]["away"]["team"].get("name"))
            carreras_local = (juego["teams"]["home"].get("score")
                              if juego["teams"]["home"].get("score") is not None
                              else None)
            carreras_visita = (juego["teams"]["away"].get("score")
                               if juego["teams"]["away"].get("score") is not None
                               else None)
            ls = juego.get("linescore") or {}
            partidos.append({
                "local": local,
                "visita": visita,
                "estado": estado,
                "runs_local": carreras_local,
                "runs_visita": carreras_visita,
                "inning": ls.get("currentInning"),
                "inning_ordinal": ls.get("currentInningOrdinal"),
                "estado_entrada": ls.get("inningState"),
                "outs": ls.get("outs"),
            })
    return partidos


def texto_estado_juego(p):
    """Texto legible del estado del juego (Pre-Game / Final / 7th Top 2 outs)."""
    if p["estado"] == "Final":
        return "FINAL"
    if p["estado"] in ("Pre-Game", "Scheduled"):
        return "PRE-GAME"
    if p["estado"] == "Delayed":
        return "RETRASADO"
    inning = p["inning_ordinal"] or (f"{p['inning']}" if p["inning"] else "?")
    lado = p["estado_entrada"] or ""
    outs = p["outs"]
    if lado in ("Top", "Bottom"):
        detalle = f"{inning} {lado}"
        if outs is not None:
            detalle += f" {outs} outs"
        return detalle
    if lado == "Middle":
        return f"Media {inning}"
    if lado == "End":
        return f"Fin {inning}"
    return f"{p['estado']} ({inning})"


def estado_apuesta(tipo, linea, total_actual, juego_final):
    """Estado provisional de la apuesta con el marcador actual."""
    if juego_final:
        if total_actual > linea:
            return "GANADA" if tipo == "OVER" else "PERDIDA"
        if total_actual < linea:
            return "GANADA" if tipo == "UNDER" else "PERDIDA"
        return "PUSH"
    if total_actual is None:
        return "sin empezar"
    if tipo == "OVER":
        if total_actual > linea:
            return "GANANDO (asegurado)"
        return f"faltan {linea + 0.5 - total_actual:g} carreras"
    if total_actual > linea:
        return "PERDIENDO (superado)"
    return f"margen {linea - total_actual:g} carreras"


def cargar_picks(fecha):
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        filas = conexion.execute(
            "SELECT EquipoLocal, EquipoVisita, TipoApuesta, Linea, "
            "Unidades, Estado FROM dbo.Predicciones "
            "WHERE Fecha = ? ORDER BY Id", fecha
        ).fetchall()
    finally:
        conexion.close()
    return [{"local": f[0], "visita": f[1], "tipo": f[2],
             "linea": float(f[3]), "unidades": float(f[4]),
             "estado_bd": f[5]} for f in filas]


def mostrar(fecha=None):
    if fecha is None:
        fecha = datetime.date.today()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 100)
    print(f"MARCADOR EN VIVO MLB - {fecha.isoformat()} "
          f"({datetime.datetime.now():%H:%M:%S})")
    print("=" * 100)

    partidos = obtener_calendario_en_vivo(fecha)
    if not partidos:
        print("  (sin partidos para esta fecha en la MLB StatsAPI)")
        return 1
    picks = cargar_picks(fecha)
    picks_por_partido = {
        (p["local"], p["visita"]): p for p in picks}

    ancho = 36
    print(f" {'Partido'.ljust(ancho)} | {'Marcador':<9} | "
          f"{'Estado':<24} | Apuesta")
    print(" " + "-" * 96)
    for p in partidos:
        local = p["local"] or "?"
        visita = p["visita"] or "?"
        marcador = " - "
        if p["runs_visita"] is not None and p["runs_local"] is not None:
            marcador = f"{p['runs_visita']} - {p['runs_local']}"
        estado_txt = texto_estado_juego(p)

        pick = picks_por_partido.get((local, visita))
        nombre = f"{local} vs {visita}"
        if pick:
            juego_final = p["estado"] == "Final"
            total_actual = None
            if p["runs_visita"] is not None and p["runs_local"] is not None:
                total_actual = p["runs_visita"] + p["runs_local"]
            linea = pick["linea"]
            if pick["estado_bd"] != "PENDIENTE":
                apuesta_txt = (f"{pick['tipo']} {linea:g} -> "
                               f"{pick['estado_bd']} (BD)")
            else:
                estado_ap = estado_apuesta(
                    pick["tipo"], linea, total_actual, juego_final)
                apuesta_txt = (f"{pick['tipo']} {linea:g} "
                               f"({pick['unidades']:g}u) -> {estado_ap}")
            print(f" *{nombre.ljust(ancho - 1)} | {marcador:<9} | "
                  f"{estado_txt:<24} | {apuesta_txt}")
        else:
            print(f"  {nombre.ljust(ancho - 2)} | {marcador:<9} | "
                  f"{estado_txt:<24} | -")
    print(" " + "-" * 96)
    pendientes = [p for p in picks if p["estado_bd"] == "PENDIENTE"]
    print(f"  Picks de hoy: {len(picks)} | PENDIENTE: {len(pendientes)} "
          "(los estados FINAL los actualiza la tarea QuickVerify en BD)")
    return 0


if __name__ == "__main__":
    fecha = None
    if len(sys.argv) > 1:
        try:
            fecha = datetime.date.fromisoformat(sys.argv[1])
        except ValueError:
            print("Fecha invalida; se usara la fecha de hoy.")
    sys.exit(mostrar(fecha))
