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
            "TipoApuesta, Cuota FROM dbo.PrediccionesML "
            "WHERE Estado = 'PENDIENTE' ORDER BY Fecha"
        ).fetchall()
        if not pendientes:
            print("[VERIFICAR-ML] No hay predicciones ML PENDIENTE.")
            return 0

        contadores = {"GANADA": 0, "PERDIDA": 0, "SIN PARTIDO": 0}
        for (pred_id, fecha, local, visita, tipo, cuota) in pendientes:
            fila = conexion.execute(
                "SELECT TOP 1 CarrerasLocal, CarrerasVisita "
                "FROM dbo.GameLog "
                "WHERE Fecha = ? AND EquipoLocal = ? AND EquipoVisita = ? "
                "AND CarrerasLocal IS NOT NULL AND CarrerasVisita IS NOT NULL "
                "AND EsFinal = 1",
                fecha, local, visita
            ).fetchone()
            if fila is None:
                contadores["SIN PARTIDO"] += 1
                print(f"[VERIFICAR-ML] {fecha} {local} vs {visita} "
                      f"({tipo} @ {cuota}): aun sin resultado final.")
                continue

            carreras_local = fila[0]
            carreras_visita = fila[1]
            gana_local = carreras_local > carreras_visita
            pick_es_local = tipo == "HOME"
            resultado = "GANADA" if gana_local == pick_es_local else "PERDIDA"
            contadores[resultado] += 1
            conexion.execute(
                "UPDATE dbo.PrediccionesML SET Estado = ?, "
                "CarrerasLocal = ?, CarrerasVisita = ?, "
                "FechaVerificacion = SYSUTCDATETIME() WHERE Id = ?",
                resultado, carreras_local, carreras_visita, pred_id)
            print(f"[VERIFICAR-ML] {fecha} {local} vs {visita} "
                  f"({tipo} @ {cuota}, {carreras_local}-{carreras_visita}): "
                  f"{resultado}")
        conexion.commit()

        print(f"[VERIFICAR-ML] Resumen: GANADA: {contadores['GANADA']} | "
              f"PERDIDA: {contadores['PERDIDA']} | "
              f"sin resultado: {contadores['SIN PARTIDO']}")
        return 0
    finally:
        conexion.close()


if __name__ == "__main__":
    sys.exit(verificar())
