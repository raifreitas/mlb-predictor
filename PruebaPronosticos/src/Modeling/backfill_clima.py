import os
import re
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

ESTADIO_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ETL", "EstadioCatalog.cs"
)

API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
SLEEP_ENTRE_PETICIONES = 0.5

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


def cargar_coordenadas_estadios(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()

    patron = r'\{\s*"([^"]+)"\s*,\s*\(([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*\d+\s*\)\s*\}'
    coords = {}
    for nombre, latitud, longitud in re.findall(patron, contenido):
        coords[nombre] = (float(latitud), float(longitud))
    return coords


def obtener_temperatura_historica(latitud, longitud, fecha):
    parametros = {
        "latitude": latitud,
        "longitude": longitud,
        "start_date": fecha.strftime("%Y-%m-%d"),
        "end_date": fecha.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max",
        "timezone": "auto",
    }
    respuesta = requests.get(API_BASE_URL, params=parametros, timeout=30)
    respuesta.raise_for_status()
    datos = respuesta.json()
    temperaturas = datos["daily"]["temperature_2m_max"]
    if not temperaturas or temperaturas[0] is None:
        return None
    return temperaturas[0]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not os.path.exists(ESTADIO_CATALOG_PATH):
        print(f"[ERROR] No se encontro EstadioCatalog.cs en {ESTADIO_CATALOG_PATH}")
        return 1

    coords = cargar_coordenadas_estadios(ESTADIO_CATALOG_PATH)
    print(f"[INFO] {len(coords)} estadios cargados del catalogo.")

    connection_string = CONNECTION_STRING_TEMPLATE.format(driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    cursor = conexion.cursor()

    cursor.execute("SELECT Id, Fecha, Estadio FROM dbo.GameLog WHERE TemperaturaC IS NULL")
    registros = cursor.fetchall()
    print(f"[INFO] {len(registros)} partidos pendientes de temperatura.")

    actualizados = 0
    fallidos = 0

    for fila in registros:
        partido_id, fecha, estadio = fila.Id, fila.Fecha, fila.Estadio
        try:
            if estadio not in coords:
                raise ValueError(f"Estadio sin coordenadas en el catalogo: {estadio}")

            latitud, longitud = coords[estadio]
            temperatura = obtener_temperatura_historica(latitud, longitud, fecha)

            if temperatura is None:
                raise ValueError(f"La API no devolvio temperatura para {fecha} en {estadio}")

            cursor.execute(
                "UPDATE dbo.GameLog SET TemperaturaC = ? WHERE Id = ?",
                (temperatura, partido_id),
            )
            conexion.commit()
            actualizados += 1
            print(f"[OK] Id={partido_id} {estadio} ({fecha}) -> {temperatura:.1f} C")
        except Exception as ex:
            fallidos += 1
            print(f"[ERROR] Id={partido_id} {estadio}: {ex}")
        finally:
            time.sleep(SLEEP_ENTRE_PETICIONES)

    cursor.close()
    conexion.close()

    print(f"[RESUMEN] {actualizados} actualizados, {fallidos} fallidos, "
          f"{len(registros) - actualizados - fallidos} pendientes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
