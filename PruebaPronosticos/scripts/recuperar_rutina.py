# -*- coding: utf-8 -*-
"""Recuperacion de rutinas MLB perdidas por apagones.

Como funciona:
  - Un marcador en logs\\estado_diario.txt guarda la ultima fecha
    en que la rutina principal (ETL + verificar + entrenar +
    predecir) proceso el dia ("al dia").
  - Cada vez que se enciende / se ejecuta una tarea, este script
    compara el marcador con HOY y procesa (mediante el ETL por
    fechas) todos los dias pasados que quedaron sin procesar,
    luego re-ejecuta verificar + entrenar + predecir una sola vez.

Modos:
  python scripts\\recuperar_rutina.py               -> procesa pendientes
  python scripts\\recuperar_rutina.py --simular     -> solo informa
  python scripts\\recuperar_rutina.py --marcar-hoy  -> marca HOY como hecho
  python scripts\\recuperar_rutina.py --ya-procesado-hoy
                               -> exit 0 si HOY ya fue procesado, sino 1
"""

import os
import subprocess
import sys
from datetime import date, timedelta

RAIZ = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
MARCADOR = os.path.join(RAIZ, "logs", "estado_diario.txt")

PROYECTO_ETL = os.path.join("src", "ETL", "PruebaPronosticos.csproj")


def hoy():
    return date.today()


def leer_marcador():
    try:
        with open(MARCADOR, encoding="utf-8") as f:
            contenido = f.read().strip()
        return date.fromisoformat(contenido)
    except Exception:
        return hoy() - timedelta(days=1)


def escribir_marcador(fecha):
    os.makedirs(os.path.dirname(MARCADOR), exist_ok=True)
    with open(MARCADOR, "w", encoding="utf-8") as f:
        f.write(fecha.isoformat())


def dias_pendientes(marcador):
    """Dias estrictamente pasados sin procesar (marcador+1 .. ayer)."""
    pendientes = []
    dia = marcador + timedelta(days=1)
    limite = hoy() - timedelta(days=1)
    while dia <= limite:
        pendientes.append(dia)
        dia += timedelta(days=1)
    return pendientes


def ejecutar(argumentos):
    print("[RECUP] Ejecutando: " + " ".join(argumentos))
    try:
        resultado = subprocess.run(argumentos, cwd=RAIZ,
                                   stdin=subprocess.DEVNULL)
        return resultado.returncode
    except FileNotFoundError:
        print(f"[RECUP] ERROR: no se encontro el programa: {argumentos[0]}")
        return 1


def procesar_pendientes(simular=False):
    marcador = leer_marcador()
    pendientes = dias_pendientes(marcador)
    if not pendientes:
        print(f"[RECUP] Sin dias pendientes (marcador: {marcador.isoformat()}).")
        return

    print(f"[RECUP] Marcador: {marcador.isoformat()} | "
          f"Faltan: {', '.join(d.isoformat() for d in pendientes)}")
    if simular:
        print("[RECUP] (modo --simular: nada se ejecuta)")
        return

    for dia in pendientes:
        iso = dia.isoformat()
        ejecutar(["dotnet", "run", "--project", PROYECTO_ETL, iso, iso])

    ejecutar(["python", os.path.join("src", "Modeling",
                                     "verificar_predicciones.py")])
    ejecutar(["python", os.path.join("src", "Modeling",
                                     "entrenar_modelo.py")])
    ejecutar(["python", os.path.join("src", "Modeling",
                                     "recomendar_apuestas.py")])

    nuevo = hoy() - timedelta(days=1)
    escribir_marcador(nuevo)
    print(f"[RECUP] Listo. {len(pendientes)} dia(s) procesado(s); "
          f"marcador actualizado a {nuevo.isoformat()}.")


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else ""

    if modo == "--marcar-hoy":
        escribir_marcador(hoy())
        print(f"[RECUP] Marcador = HOY ({hoy().isoformat()}).")
        return
    if modo == "--ya-procesado-hoy":
        if leer_marcador() >= hoy():
            return 0
        return 1
    if modo == "--simular":
        procesar_pendientes(simular=True)
        return

    procesar_pendientes()
    return 0


if __name__ == "__main__":
    sys.exit(main())