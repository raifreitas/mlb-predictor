# -*- coding: utf-8 -*-
"""Verificacion ligera cada 15 min (tarea programada QuickVerify).

Solo hace ETL de ayer-hoy si hay predicciones PENDIENTE y luego verifica
cada pick de forma independiente: apenas un partido pronosticado termina,
su estado pasa a GANADA/PERDIDA aunque los demas sigan en curso.
"""
import datetime
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src", "Modeling"))
from verificar_predicciones import obtener_driver_odbc, CONNECTION_STRING_TEMPLATE
import pyodbc


def anotar(log, mensaje):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {mensaje}\n")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(
        log_dir, "verificacion_rapida_"
                f"{datetime.date.today():%Y%m%d}.txt")

    anotar(log, "===== INICIO VERIFICACION RAPIDA =====")
    try:
        conexion = pyodbc.connect(CONNECTION_STRING_TEMPLATE.format(
            driver=obtener_driver_odbc()))
        pendientes = conexion.execute(
            "SELECT COUNT(*) FROM dbo.Predicciones "
            "WHERE Estado = 'PENDIENTE'").fetchone()[0]
        conexion.close()
    except Exception as e:
        anotar(log, f"[ERROR] No se pudo consultar pendientes: {e}")
        return 0

    if not pendientes:
        anotar(log, "Sin predicciones PENDIENTE: ETL no necesario.")
        return 0

    ayer = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    hoy = datetime.date.today().isoformat()
    cmd = (f"dotnet run --project src\\ETL\\PruebaPronosticos.csproj "
           f"{ayer} {hoy} < NUL")
    anotar(log, f"ETL: {cmd}")
    with open(log, "a", encoding="utf-8") as f:
        r_etl = subprocess.run(cmd, cwd=root, shell=True,
                               stdout=f, stderr=subprocess.STDOUT)
    anotar(log, f"ETL exit: {r_etl.returncode}")

    r_ver = subprocess.run(
        [sys.executable, "src\\Modeling\\verificar_predicciones.py"],
        cwd=root)
    anotar(log, f"VERIFICAR exit: {r_ver.returncode}")
    anotar(log, "===== FIN VERIFICACION RAPIDA =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
