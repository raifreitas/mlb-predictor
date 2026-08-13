import sys

import db_utils


def _a_utc(valor):
    """Convierte datetime (pyodbc) o str ISO/yyyymmdd hh:mm:ss.fff (SQLite)
    a datetime con zona UTC, robusto para ambos motores."""
    import datetime as _dt
    if isinstance(valor, str):
        texto = valor.strip()
        if "." in texto:
            texto = texto.split(".")[0]
        try:
            return _dt.datetime.strptime(texto, "%Y-%m-%d %H:%M:%S"
                                         ).replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            pass
        try:
            return _dt.datetime.fromisoformat(texto.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(valor, _dt.datetime):
        if valor.tzinfo is None:
            return valor.replace(tzinfo=_dt.timezone.utc)
        return valor.astimezone(_dt.timezone.utc)
    return None


def verificar():
    con = db_utils.conexion()
    try:
        pendientes = con.execute(
            "SELECT Id, Fecha, EquipoLocal, EquipoVisita, "
            "TipoApuesta, Linea, CreadoUtc FROM Predicciones "
            "WHERE Estado = 'PENDIENTE' ORDER BY Fecha"
        ).fetchall()
        if not pendientes:
            print("[VERIFICAR] No hay predicciones PENDIENTE.")
            return 0

        contadores = {"GANADA": 0, "PERDIDA": 0, "PUSH": 0,
                      "SIN PARTIDO": 0, "NO VALIDA": 0}
        for (pred_id, fecha, local, visita, tipo, linea, creado_utc) in pendientes:
            fila = con.execute(
                "SELECT CarrerasLocal, CarrerasVisita, HoraInicioUtc "
                "FROM GameLog "
                "WHERE Fecha = ? AND EquipoLocal = ? AND EquipoVisita = ? "
                "AND CarrerasLocal IS NOT NULL AND CarrerasVisita IS NOT NULL "
                "AND EsFinal = 1",
                [fecha, local, visita]
            ).fetchone()
            if fila is None:
                contadores["SIN PARTIDO"] += 1
                print(f"[VERIFICAR] {fecha} {local} vs {visita} "
                      f"({tipo} {linea}): aun sin resultado final.")
                continue

            hora_inicio = fila[2]
            if creado_utc is not None and hora_inicio is not None:
                import datetime as _dt
                hora_inicio = _a_utc(hora_inicio)
                creado = _a_utc(creado_utc)
                if creado >= hora_inicio:
                    contadores["NO VALIDA"] += 1
                    con.execute(
                        "UPDATE Predicciones SET Estado = 'NO_VALIDA', "
                        "FechaVerificacion = datetime('now') WHERE Id = ?",
                        [pred_id])
                    print(f"[VERIFICAR] {fecha} {local} vs {visita} "
                          f"({tipo} {linea}): NO VALIDA (pick generado "
                          f"{creado.strftime('%H:%M')} UTC, partido inicio "
                          f"{hora_inicio.strftime('%H:%M')} UTC).")
                    continue

            total = fila[0] + fila[1]
            if total > linea:
                resultado = "GANADA" if tipo == "OVER" else "PERDIDA"
            elif total < linea:
                resultado = "GANADA" if tipo == "UNDER" else "PERDIDA"
            else:
                resultado = "PUSH"
            contadores[resultado] += 1
            con.execute(
                "UPDATE Predicciones SET Estado = ?, CarrerasTotales = ?, "
                "FechaVerificacion = datetime('now') WHERE Id = ?",
                [resultado, total, pred_id])
            print(f"[VERIFICAR] {fecha} {local} vs {visita} "
                  f"({tipo} {linea}, total {total}): {resultado}")
        con.commit()

        print(f"[VERIFICAR] Resumen: GANADA: {contadores['GANADA']} | "
              f"PERDIDA: {contadores['PERDIDA']} | "
              f"PUSH: {contadores['PUSH']} | "
              f"sin resultado: {contadores['SIN PARTIDO']} | "
              f"NO VALIDA: {contadores['NO VALIDA']}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(verificar())