"""Genera web/data.json (estado de predicciones y partidos) para GitHub Pages.

Lee la BD (SQLite en la nube, SQL Server en local) y escribe un JSON
compacto que index.html consume:
  - fecha_actualizacion
  - partidos_hoy (con linea de mercado y estado)
  - predicciones (historial de la temporada con resultados)
  - resumen (ganadas, perdidas, unidades)
"""
import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

import db_utils

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def _raiz_repo():
    carpeta = CARPETA_SCRIPT
    while True:
        if os.path.isdir(os.path.join(carpeta, ".git")):
            return carpeta
        padre = os.path.dirname(carpeta)
        if padre == carpeta:
            raise SystemExit("[WEB] no se encontro la raiz del repo (.git).")
        carpeta = padre


def _ruta_data_json():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    return os.path.join(_raiz_repo(), "web", "data.json")


def _jsonable(val):
    if isinstance(val, Decimal):
        return float(val)
    return val


def _ahora_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generar():
    con = db_utils.conexion()
    try:
        preds = con.execute(
            """SELECT Id, Fecha, EquipoLocal, EquipoVisita, TipoApuesta,
                      Linea, Unidades, Edge, Cuota, Estado, CarrerasTotales,
                      CreadoUtc, FechaVerificacion
               FROM Predicciones ORDER BY Fecha, Id""").fetchall()

        ultima_fecha = con.execute(
            "SELECT MAX(Fecha) FROM Predicciones").fetchone()[0]

        evaluaciones = con.execute(
            """SELECT Fecha, EquipoLocal, EquipoVisita, Linea, Prediccion,
                      ProbOver, Edge, Recomendacion, Motivo, EvaluadoUtc
               FROM Evaluaciones
               WHERE Fecha >= date('now', '-6 days')
               ORDER BY Fecha DESC, EvaluadoUtc DESC, EquipoLocal""").fetchall() \
            if db_utils.usar_sqlite() else con.execute(
            """SELECT Fecha, EquipoLocal, EquipoVisita, Linea, Prediccion,
                      ProbOver, Edge, Recomendacion, Motivo, EvaluadoUtc
               FROM Evaluaciones
               WHERE Fecha >= CAST(DATEADD(DAY, -6, GETDATE()) AS DATE)
               ORDER BY Fecha DESC, EvaluadoUtc DESC, EquipoLocal""").fetchall()

        partidos = con.execute(
            """SELECT Fecha, EquipoLocal, EquipoVisita, CarrerasLocal,
                      CarrerasVisita, Linea_Casino_Real, EsFinal,
                      HoraInicioUtc, TemperaturaC
               FROM GameLog
               WHERE Fecha >= date('now', '-2 days')
               ORDER BY Fecha, EquipoLocal""").fetchall() \
            if db_utils.usar_sqlite() else con.execute(
            """SELECT Fecha, EquipoLocal, EquipoVisita, CarrerasLocal,
                      CarrerasVisita, Linea_Casino_Real, EsFinal,
                      HoraInicioUtc, TemperaturaC
               FROM GameLog
               WHERE Fecha >= DATEADD(DAY, -2, CAST(GETDATE() AS DATE))
               ORDER BY Fecha, EquipoLocal""").fetchall()

        predicciones = []
        for (pid, fecha, local, visita, tipo, linea, unid, edge, cuota,
             estado, total, creado, verif) in preds:
            predicciones.append({
                "id": pid, "fecha": str(fecha), "local": local,
                "visita": visita, "tipo": tipo, "linea": _jsonable(linea),
                "unidades": _jsonable(unid), "edge": _jsonable(edge),
                "cuota": _jsonable(cuota),
                "estado": estado, "total": _jsonable(total),
                "creado": str(creado), "verificado": str(verif)
                if verif else None,
            })

        partidos_json = []
        for (fecha, local, visita, carreras_l, carreras_v, linea, es_final,
             hora_utc, temp) in partidos:
            partidos_json.append({
                "fecha": str(fecha), "local": local, "visita": visita,
                "carreras_local": _jsonable(carreras_l),
                "carreras_visita": _jsonable(carreras_v),
                "linea_real": _jsonable(linea), "es_final": bool(es_final),
                "hora_inicio_utc": str(hora_utc) if hora_utc else None,
                "temperatura_c": _jsonable(temp),
            })

        evaluaciones_json = []
        for (fecha, local, visita, linea, prediccion, prob_over, edge,
             rec, motivo, evaluado) in evaluaciones:
            evaluaciones_json.append({
                "fecha": str(fecha), "local": local, "visita": visita,
                "linea": _jsonable(linea),
                "prediccion": _jsonable(prediccion),
                "prob_over": _jsonable(prob_over),
                "edge": _jsonable(edge),
                "recomendacion": rec, "motivo": motivo or "",
                "evaluado_utc": str(evaluado) if evaluado else None,
            })

        ganadas = sum(1 for p in predicciones if p["estado"] == "GANADA")
        perdidas = sum(1 for p in predicciones if p["estado"] == "PERDIDA")
        pushes = sum(1 for p in predicciones if p["estado"] == "PUSH")
        unidades = sum(p["unidades"] for p in predicciones
                       if p["estado"] == "GANADA") \
            - sum(p["unidades"] for p in predicciones
                  if p["estado"] == "PERDIDA")

        salida = {
            "ultima_fecha_prediccion": str(ultima_fecha)
            if ultima_fecha else None,
            "resumen": {
                "ganadas": ganadas, "perdidas": perdidas, "push": pushes,
                "unidades": round(unidades, 2),
            },
            "partidos_hoy": partidos_json,
            "predicciones": predicciones,
            "evaluaciones": evaluaciones_json,
        }
    finally:
        con.close()

    ruta = _ruta_data_json()
    previo = None
    if os.path.exists(ruta):
        try:
            with open(ruta, encoding="utf-8") as f:
                previo = json.load(f)
        except (json.JSONDecodeError, OSError):
            previo = None
    if previo is not None and _sin_timestamp(previo) == salida:
        print(f"[WEB] {ruta} sin cambios (no se reescribe).")
        return 0

    salida["fecha_actualizacion"] = _ahora_iso()
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)
    print(f"[WEB] {ruta} escrito "
          f"({len(predicciones)} predicciones, "
          f"{len(partidos_json)} partidos, "
          f"{len(evaluaciones_json)} evaluaciones, "
          f"{salida['fecha_actualizacion']}).")
    return 0


def _sin_timestamp(datos):
    copia = dict(datos)
    copia.pop("fecha_actualizacion", None)
    return copia


if __name__ == "__main__":
    sys.exit(generar())