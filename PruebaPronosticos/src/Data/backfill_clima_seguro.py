import sys
import time

import pyodbc
import requests

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

ESTADIOS = {
    "Yankee Stadium": (40.8296, -73.9262),
    "Fenway Park": (42.3467, -71.0972),
    "Oriole Park at Camden Yards": (39.2840, -76.6215),
    "Tropicana Field": (27.7683, -82.6534),
    "Rogers Centre": (43.6414, -79.3894),
    "Guaranteed Rate Field": (41.8299, -87.6338),
    "Rate Field": (41.8299, -87.6338),
    "Progressive Field": (41.4962, -81.6852),
    "Comerica Park": (42.3390, -83.0485),
    "Kauffman Stadium": (39.0517, -94.4803),
    "Target Field": (44.9817, -93.2778),
    "Minute Maid Park": (29.7573, -95.3555),
    "Angel Stadium": (33.8003, -117.8827),
    "Oakland Coliseum": (37.7516, -122.2005),
    "T-Mobile Park": (47.5914, -122.3325),
    "Globe Life Field": (32.7373, -97.0844),
    "Truist Park": (33.8907, -84.4677),
    "loanDepot park": (25.7783, -80.2196),
    "Citi Field": (40.7571, -73.8458),
    "Citizens Bank Park": (39.9061, -75.1665),
    "Nationals Park": (38.8730, -77.0074),
    "Wrigley Field": (41.9484, -87.6553),
    "Great American Ball Park": (39.0979, -84.5082),
    "American Family Field": (43.0280, -87.9712),
    "PNC Park": (40.4469, -80.0057),
    "Busch Stadium": (38.6226, -90.1928),
    "Chase Field": (33.4455, -112.0667),
    "Coors Field": (39.7559, -104.9942),
    "Dodger Stadium": (34.0739, -118.2400),
    "UNIQLO Field at Dodger Stadium": (34.0739, -118.2400),
    "Petco Park": (32.7076, -117.1570),
    "Oracle Park": (37.7786, -122.3893),
    "Sutter Health Park": (38.5804, -121.5135),
}

API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
SLEEP_ENTRE_PETICIONES = 0.2


def obtener_driver_odbc():
    disponibles = pyodbc.drivers()
    for preferido in DRIVERS_PREFERIDOS:
        if preferido in disponibles:
            return preferido
    if disponibles:
        return disponibles[0]
    raise RuntimeError("No se encontro un driver ODBC de SQL Server instalado.")


def obtener_temperatura(latitud, longitud, fecha):
    parametros = {
        "latitude": latitud,
        "longitude": longitud,
        "start_date": fecha,
        "end_date": fecha,
        "daily": "temperature_2m_mean",
        "timezone": "auto",
    }
    respuesta = requests.get(API_BASE_URL, params=parametros, timeout=30)
    respuesta.raise_for_status()
    datos = respuesta.json()
    temperaturas = datos["daily"]["temperature_2m_mean"]
    if not temperaturas or temperaturas[0] is None:
        return None
    return temperaturas[0]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    connection_string = CONNECTION_STRING_TEMPLATE.format(driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    cursor = conexion.cursor()

    cursor.execute("SELECT Id, Fecha, Estadio FROM dbo.GameLog WHERE TemperaturaC IS NULL")
    registros = cursor.fetchall()
    total = len(registros)

    print(f"[INFO] {total} partidos pendientes de temperatura en dbo.GameLog.")
    if total == 0:
        cursor.close()
        conexion.close()
        print("[INFO] Nada que actualizar. Finalizando.")
        return 0

    actualizados = 0
    fallidos = 0

    for indice, fila in enumerate(registros, start=1):
        partido_id, fecha, estadio = fila.Id, fila.Fecha, fila.Estadio
        fecha_str = fecha.strftime("%Y-%m-%d")

        try:
            if estadio not in ESTADIOS:
                raise ValueError(f"Estadio no catalogado (sin coordenadas): {estadio}")

            latitud, longitud = ESTADIOS[estadio]
            temperatura = obtener_temperatura(latitud, longitud, fecha_str)

            if temperatura is None:
                raise ValueError(f"La API no devolvio temperature_2m_mean para {fecha_str}")

            cursor.execute(
                "UPDATE dbo.GameLog SET TemperaturaC = ? WHERE Id = ?",
                (temperatura, partido_id),
            )
            conexion.commit()

            actualizados += 1
            print(f"[{indice}/{total}] Actualizado {estadio}: {temperatura:.1f}°C")
        except Exception as ex:
            fallidos += 1
            print(f"[{indice}/{total}] Error en Id={partido_id} ({estadio}): {ex}")
        finally:
            time.sleep(SLEEP_ENTRE_PETICIONES)

    cursor.close()
    conexion.close()

    print(f"[RESUMEN] {actualizados} actualizados, {fallidos} fallidos, "
          f"{total - actualizados - fallidos} pendientes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
