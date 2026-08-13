import sys
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


def obtener_driver_odbc():
    disponibles = pyodbc.drivers()
    for preferido in DRIVERS_PREFERIDOS:
        if preferido in disponibles:
            return preferido
    if disponibles:
        return disponibles[0]
    raise RuntimeError("No se encontro un driver ODBC de SQL Server instalado.")


def verificar():
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        pendientes = conexion.execute(
            "SELECT Id, Fecha, EquipoLocal, EquipoVisita, "
            "TipoApuesta, Linea, CreadoUtc FROM dbo.Predicciones "
            "WHERE Estado = 'PENDIENTE' ORDER BY Fecha"
        ).fetchall()
        if not pendientes:
            print("[VERIFICAR] No hay predicciones PENDIENTE.")
            return 0

        contadores = {"GANADA": 0, "PERDIDA": 0, "PUSH": 0,
                      "SIN PARTIDO": 0, "NO VALIDA": 0}
        for (pred_id, fecha, local, visita, tipo, linea, creado_utc) in pendientes:
            fila = conexion.execute(
                "SELECT TOP 1 CarrerasLocal, CarrerasVisita, HoraInicioUtc "
                "FROM dbo.GameLog "
                "WHERE Fecha = ? AND EquipoLocal = ? AND EquipoVisita = ? "
                "AND CarrerasLocal IS NOT NULL AND CarrerasVisita IS NOT NULL "
                "AND EsFinal = 1",
                fecha, local, visita
            ).fetchone()
            if fila is None:
                contadores["SIN PARTIDO"] += 1
                print(f"[VERIFICAR] {fecha} {local} vs {visita} "
                      f"({tipo} {linea}): aun sin resultado final.")
                continue

            # Regla de validez: un pronostico solo es apostable si se
            # genero ANTES del primer pitch. Si se creo despues del
            # inicio del partido (o con el partido ya comenzado), se
            # anula: no se cuenta como ganada/perdida.
            hora_inicio = fila[2]
            if creado_utc is not None and hora_inicio is not None:
                import datetime as _dt
                hora_inicio_utc = hora_inicio.replace(
                    tzinfo=_dt.timezone.utc)
                creado = creado_utc
                if creado.tzinfo is None:
                    creado = creado.replace(tzinfo=_dt.timezone.utc)
                if creado >= hora_inicio_utc:
                    contadores["NO VALIDA"] += 1
                    conexion.execute(
                        "UPDATE dbo.Predicciones SET Estado = 'NO_VALIDA', "
                        "FechaVerificacion = SYSUTCDATETIME() WHERE Id = ?",
                        pred_id)
                    print(f"[VERIFICAR] {fecha} {local} vs {visita} "
                          f"({tipo} {linea}): NO VALIDA (pick generado "
                          f"{creado.strftime('%H:%M')} UTC, partido inicio "
                          f"{hora_inicio_utc.strftime('%H:%M')} UTC).")
                    continue

            total = fila[0] + fila[1]
            if total > linea:
                resultado = "GANADA" if tipo == "OVER" else "PERDIDA"
            elif total < linea:
                resultado = "GANADA" if tipo == "UNDER" else "PERDIDA"
            else:
                resultado = "PUSH"
            contadores[resultado] += 1
            conexion.execute(
                "UPDATE dbo.Predicciones SET Estado = ?, CarrerasTotales = ?, "
                "FechaVerificacion = SYSUTCDATETIME() WHERE Id = ?",
                resultado, total, pred_id)
            print(f"[VERIFICAR] {fecha} {local} vs {visita} "
                  f"({tipo} {linea}, total {total}): {resultado}")
        conexion.commit()

        print(f"[VERIFICAR] Resumen: GANADA: {contadores['GANADA']} | "
              f"PERDIDA: {contadores['PERDIDA']} | "
              f"PUSH: {contadores['PUSH']} | "
              f"sin resultado: {contadores['SIN PARTIDO']} | "
              f"NO VALIDA: {contadores['NO VALIDA']}")
        return 0
    finally:
        conexion.close()


if __name__ == "__main__":
    sys.exit(verificar())
