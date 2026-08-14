"""Planificador dinamico por partido (sustituye al runner de 15 minutos).

Para cada partido MLB calcula hora_corrida = primer_pitch - ventana (30 min)
y registra el estado en data/horarios.json. Cada wake de GitHub Actions
llama a --procesar, que ejecuta el runner SOLO para los partidos cuya
hora_corrida ya llego y aun no se ejecuto (una unica evaluacion por juego
con la linea de mercado mas fresca, sin gastar The Odds API de mas).

Uso:
    python planificador.py --generar [--dia YYYY-MM-DD] [--ventana-min 30]
    python planificador.py --procesar [--ventana-min 30] [--siempre]
    python planificador.py --estado
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests

import db_utils

MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
TIMEOUT_SEGUNDOS = 30
ZONA_ESTE = ZoneInfo("America/New_York")
ESTADO_PENDIENTE = "pendiente"
ESTADO_EJECUTADO = "ejecutado"
ESTADO_OMITIDO = "omitido"  # sin hora de inicio conocida

CARPETA_MODELING = os.path.dirname(os.path.abspath(__file__))
RUTA_HORARIOS = db_utils.RAIZ / "data" / "horarios.json"


def dia_mlb():
    """Dia oficial MLB actual en zona Este (align con GameLog.Fecha)."""
    return datetime.now(ZONA_ESTE).date()


def consultar_calendario(fecha):
    """Partidos del dia (Preview) con hora de inicio y abridores probables."""
    respuesta = requests.get(
        f"{MLB_BASE_URL}/schedule",
        params={"sportId": 1, "date": fecha.isoformat(),
                "hydrate": "probablePitcher,venue,team"},
        timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()

    partidos = []
    for dia in respuesta.json().get("dates", []):
        for juego in dia.get("games", []):
            estado = juego.get("status") or {}
            if estado.get("detailedState") in ("Postponed", "Cancelled"):
                continue
            if estado.get("abstractGameState") != "Preview":
                continue
            local = juego["teams"]["home"]["team"]
            visita = juego["teams"]["away"]["team"]
            hora_inicio = None
            if juego.get("gameDate"):
                try:
                    hora_inicio = datetime.fromisoformat(
                        juego["gameDate"].replace("Z", "+00:00"))
                except ValueError:
                    hora_inicio = None
            partidos.append({
                "local": local.get("fullName") or local.get("name"),
                "visita": visita.get("fullName") or visita.get("name"),
                "pitcher_local": ((juego["teams"]["home"].get("probablePitcher")
                                   or {}).get("id")),
                "pitcher_visita": ((juego["teams"]["away"].get("probablePitcher")
                                    or {}).get("id")),
                "hora_inicio_utc": hora_inicio,
            })
    return partidos


def cargar_horarios(fecha):
    datos = {}
    if RUTA_HORARIOS.exists():
        try:
            datos = json.loads(RUTA_HORARIOS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            datos = {}
    return datos.get(fecha.isoformat(), {})


def guardar_horarios(fecha, horarios):
    datos = {}
    if RUTA_HORARIOS.exists():
        try:
            datos = json.loads(RUTA_HORARIOS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            datos = {}
    datos[fecha.isoformat()] = horarios
    RUTA_HORARIOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_HORARIOS.write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PLAN] horarios.json actualizado ({RUTA_HORARIOS}).")


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def generar(fecha, ventana_min):
    """Crea/refresca los horarios del dia (solo partidos no iniciados)."""
    partidos = consultar_calendario(fecha)
    ahora_utc = datetime.now(timezone.utc)
    entradas = {}
    sin_hora = []
    for p in partidos:
        inicio = p["hora_inicio_utc"]
        if inicio is None:
            entradas[p["local"]] = {**_basica(p), "estado": ESTADO_OMITIDO}
            sin_hora.append(p["local"])
            continue
        if inicio <= ahora_utc:
            continue  # ya empezo; no se programa
        entradas[p["local"]] = {
            **_basica(p),
            "hora_inicio_utc": _iso(inicio),
            "hora_corrida_utc": _iso(inicio - timedelta(minutes=ventana_min)),
            "estado": ESTADO_PENDIENTE,
        }
    guardar_horarios(fecha, entradas)
    print(f"[PLAN] {fecha}: {len(entradas)} partidos programados "
          f"({ventana_min} min antes del pitch), "
          f"sin hora: {len(sin_hora)}.")


def _basica(p):
    return {
        "local": p["local"],
        "visita": p["visita"],
        "pitcher_local": p["pitcher_local"],
        "pitcher_visita": p["pitcher_visita"],
    }


def procesar(ventana_min, siempre=False, horizonte_max_min=240):
    """Corre el runner para los partidos pendientes aun no iniciados.

    Cada partido tiene su propia ventana de evaluacion de
    horizonte_max_min antes del primer pitch: si el tick llega dentro de
    esa ventana, se evalua con la linea del snapshot mas reciente (la
    frescura la da el ETL, no el runner). Si el tick llega ANTES de la
    ventana el partido sigue pendiente (no se emiten picks con 20 h de
    anticipacion). Si el tick llega DESPUES del inicio, se marca
    ejecutado (vencido: ya no es apostable).
    """
    fecha = dia_mlb()
    horarios = cargar_horarios(fecha)
    if not horarios:
        print(f"[PLAN] {fecha}: sin horarios, generando...")
        generar(fecha, ventana_min)
        horarios = cargar_horarios(fecha)
        if not horarios:
            print("[PLAN] Sin partidos programados para hoy.")
            return 0

    ahora_utc = datetime.now(timezone.utc)
    pendientes = []
    vencidos = []
    for p in horarios.values():
        if p.get("estado") != ESTADO_PENDIENTE:
            continue
        inicio = p.get("hora_inicio_utc")
        if inicio is None:
            continue
        try:
            hora_inicio = datetime.fromisoformat(inicio.replace("Z", "+00:00"))
        except ValueError:
            continue
        if hora_inicio <= ahora_utc:
            vencidos.append(p)  # ya empezo: ya no es apostable
        else:
            pendientes.append(p)

    for p in vencidos:
        p["estado"] = ESTADO_EJECUTADO
        p["ejecutado_utc"] = _iso(ahora_utc)
        print(f"[PLAN] {p['local']} vs {p['visita']}: inicio pasado "
              "(no evaluable), marcado ejecutado.")

    if not pendientes:
        if vencidos:
            guardar_horarios(fecha, horarios)
        print(f"[PLAN] {fecha}: ningun partido pendiente sin iniciar "
              f"(ahora {ahora_utc:%H:%M} UTC).")
        return 0

    # VENTANA del runner: cubre los pendientes cuyo inicio esta dentro
    # del horizonte (ahora .. ahora + horizonte). Los que aun faltan mas
    # se dejan pendientes para otro wake.
    horizonte = timedelta(minutes=horizonte_max_min)
    fijados = []
    for p in pendientes:
        try:
            i = datetime.fromisoformat(
                p["hora_inicio_utc"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        if i - ahora_utc <= horizonte:
            fijados.append((p, i))
    if not fijados:
        guardar_horarios(fecha, horarios)
        print(f"[PLAN] {fecha}: ningun partido dentro del horizonte de "
              f"{horizonte_max_min} min (now {ahora_utc:%H:%M} UTC); "
              f"{len(pendientes)} pendientes se reintentaran.")
        return 0

    max_restante = max((i - ahora_utc).total_seconds() for _, i in fijados) \
        / 60.0
    ventana_total = max(1, int(max_restante) + 2)

    print(f"[PLAN] {fecha}: {len(pendientes)} partido(s) pendientes "
          f"(sin iniciar). Corriendo runner con ventana "
          f"{ventana_total} min...")
    cmd = [sys.executable, os.path.join(CARPETA_MODELING,
                                        "recomendar_apuestas.py"),
           "--fecha", fecha.isoformat(), "--ventana-min", str(ventana_total)]
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[PLAN] runner fallo (rc={rc}); los partidos quedan pendientes.")
        return rc

    # Marcar SOLO los que el runner evaluo de verdad (tienen EvaluadoUtc
    # fresco). Los demas se reintentan en el proximo wake.
    con = db_utils.conexion()
    try:
        filas = con.execute(
            "SELECT DISTINCT EquipoLocal, EquipoVisita FROM Evaluaciones "
            "WHERE Fecha = ? ORDER BY EquipoLocal",
            [fecha.isoformat()]).fetchall() \
            if db_utils.usar_sqlite() else con.execute(
            "SELECT DISTINCT EquipoLocal, EquipoVisita FROM Evaluaciones "
            "WHERE Fecha = ? ORDER BY EquipoLocal",
            [fecha.isoformat()]).fetchall()
    finally:
        con.close()
    evaluados = {(f[0], f[1]) for f in filas}
    marcados = 0
    for p in pendientes:
        if (p["local"], p["visita"]) in evaluados:
            p["estado"] = ESTADO_EJECUTADO
            p["ejecutado_utc"] = _iso(ahora_utc)
            marcados += 1
    guardar_horarios(fecha, horarios)
    print(f"[PLAN] {marcados}/{len(pendientes)} partido(s) marcados como "
          f"ejecutados (evaluados); el resto se reintentara.")
    return 0


def estado():
    fecha = dia_mlb()
    horarios = cargar_horarios(fecha)
    if not horarios:
        print(f"[PLAN] {fecha}: sin horarios registrados.")
        return
    print(f"[PLAN] Estado del dia {fecha}:")
    for p in sorted(horarios.values(), key=lambda x: x.get("hora_inicio_utc", "")):
        print(f"  {p['local']} vs {p['visita']} | inicio "
              f"{p.get('hora_inicio_utc', '-')[:19]} | "
              f"{p.get('estado')}")


def _parsear_args():
    parser = argparse.ArgumentParser(description="Planificador MLB por partido")
    parser.add_argument("--generar", action="store_true")
    parser.add_argument("--procesar", action="store_true")
    parser.add_argument("--estado", action="store_true")
    parser.add_argument("--dia", type=date.fromisoformat)
    parser.add_argument("--ventana-min", type=int, default=30)
    parser.add_argument("--horizonte-max-min", type=int, default=240,
                        help="maximo de minutos antes del inicio en que se "
                             "emite el pronostico de un partido")
    parser.add_argument("--siempre", action="store_true",
                        help="ejecuta aunque el partido no este en ventana")
    return parser.parse_args()


def main():
    args = _parsear_args()
    fecha = args.dia or dia_mlb()
    if args.generar:
        generar(fecha, args.ventana_min)
        return 0
    if args.procesar:
        return procesar(args.ventana_min, args.siempre,
                        args.horizonte_max_min)
    if args.estado:
        estado()
        return 0
    estado()
    return 0


if __name__ == "__main__":
    sys.exit(main())