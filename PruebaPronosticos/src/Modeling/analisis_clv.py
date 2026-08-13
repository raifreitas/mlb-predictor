"""Analisis de CLV (Closing Line Value) y resultados reales.

Mide, contra las lineas y cuotas REALES de cierre del mercado (ETL
The Odds API -> dbo.GameLog), la calidad de cada pick registrado en
dbo.Predicciones:

- CLV en puntos: cuanto mejoro (o empeoro) nuestra linea apostada frente
  a la linea de cierre del mercado. CLV > 0 => apostamos mejor que el
  cierre; es la unica metrica que se puede medir antes del resultado.
- ROI real: unidades ganadas con la cuota apostada vs la cuota de cierre.
- Relacion edge del modelo vs acierto: alimenta la calibracion futura
  del peso de la regresion al mercado (PESO_MERCADO_MAX / EDGE_MAXIMO).

Uso:
    python analisis_clv.py [--detalle]
"""

import sys
from datetime import date

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


def analizar():
    detalle = "--detalle" in sys.argv
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        filas = conexion.execute("""
            SELECT p.Fecha, p.EquipoLocal, p.EquipoVisita, p.TipoApuesta,
                   p.Linea, p.Cuota, p.Unidades, p.Edge, p.Estado,
                   g.Linea_Casino_Real, g.Cuota_Over_Real, g.Cuota_Under_Real,
                   g.CarrerasLocal + g.CarrerasVisita AS CarrerasTotales
            FROM dbo.Predicciones p
            LEFT JOIN dbo.GameLog g
              ON g.Fecha = p.Fecha
             AND g.EquipoLocal = p.EquipoLocal
             AND g.EquipoVisita = p.EquipoVisita
            WHERE p.Estado <> 'PENDIENTE'
            ORDER BY p.Fecha, p.Id
        """).fetchall()
    finally:
        conexion.close()

    lineas = []
    if not filas:
        print("No hay predicciones resueltas aun (Estado <> PENDIENTE).")
        return

    for fila in filas:
        (fecha, local, visita, tipo, linea_apostada, cuota_apostada,
         unidades, edge, estado, linea_cierre, cuota_over_cierre,
         cuota_under_cierre, total) = fila

        linea_cierre = float(linea_cierre) if linea_cierre else None
        cuota_over_cierre = float(cuota_over_cierre) if cuota_over_cierre else None
        cuota_under_cierre = float(cuota_under_cierre) if cuota_under_cierre else None

        clv = None
        if linea_cierre is not None and linea_apostada is not None:
            linea_apostada = float(linea_apostada)
            if tipo == "OVER":
                clv = linea_cierre - linea_apostada
            elif tipo == "UNDER":
                clv = linea_apostada - linea_cierre
        if clv is not None and abs(clv) < 0.05:
            clv = 0.0

        cuota_cierre = None
        if tipo == "OVER":
            cuota_cierre = cuota_over_cierre
        elif tipo == "UNDER":
            cuota_cierre = cuota_under_cierre

        unidades = float(unidades) if unidades else 0.0
        cuota_apostada = float(cuota_apostada) if cuota_apostada else None
        ganancia = 0.0
        if estado == "GANADA" and cuota_apostada:
            ganancia = unidades * (cuota_apostada - 1.0)
        elif estado == "PERDIDA":
            ganancia = -unidades

        lineas.append({
            "fecha": fecha.isoformat() if hasattr(fecha, "isoformat") else str(fecha),
            "partido": f"{local} vs {visita}",
            "tipo": tipo,
            "linea_apostada": linea_apostada,
            "linea_cierre": linea_cierre,
            "clv": clv,
            "cuota_apostada": cuota_apostada,
            "cuota_cierre": cuota_cierre,
            "edge": float(edge) if edge else None,
            "unidades": unidades,
            "estado": estado,
            "total": total,
            "ganancia": ganancia,
        })

    con_cierre = [l for l in lineas if l["clv"] is not None]
    clv_medio = sum(l["clv"] for l in con_cierre) / len(con_cierre) \
        if con_cierre else None

    print("=" * 100)
    print("ANALISIS CLV Y RESULTADOS REALES (vs linea de cierre The Odds API)")
    print("=" * 100)
    encabezado = (f"  {'Fecha':<10} {'Partido':<36} {'Tipo':<6} "
                  f"{'Linea':<6} {'Cierre':<6} {'CLV':<7} "
                  f"{'Cuota':<6} {'CuotaC':<6} {'Edge':<6} {'u':<5} "
                  f"{'Estado':<8} {'Unid':>7}")
    print(encabezado)
    print("  " + "-" * 96)
    for l in lineas:
        clv_txt = f"{l['clv']:+.1f}" if l["clv"] is not None else "  -  "
        cuota_txt = f"{l['cuota_apostada']:.2f}" if l["cuota_apostada"] else "  -  "
        cierre_txt = f"{l['cuota_cierre']:.2f}" if l["cuota_cierre"] else "  -  "
        linea_txt = f"{l['linea_apostada']:.1f}" if l["linea_apostada"] else "  -  "
        cierre_linea = f"{l['linea_cierre']:.1f}" if l["linea_cierre"] else "  -  "
        edge_txt = f"{l['edge']:.2f}" if l["edge"] is not None else "  -  "
        print(f"  {l['fecha']:<10} {l['partido']:<36} {l['tipo']:<6} "
              f"{linea_txt:<6} {cierre_linea:<6} {clv_txt:<7} "
              f"{cuota_txt:<6} {cierre_txt:<6} {edge_txt:<6} "
              f"{l['unidades']:<5.2f} {l['estado']:<8} {l['ganancia']:>+7.2f}")

    print("  " + "-" * 96)
    total_u = sum(l["ganancia"] for l in lineas)
    total_apostado = sum(l["unidades"] for l in lineas)
    ganadas = sum(1 for l in lineas if l["estado"] == "GANADA")
    perdidas = sum(1 for l in lineas if l["estado"] == "PERDIDA")
    pushes = sum(1 for l in lineas if l["estado"] == "PUSH")
    print(f"  Resueltas: {len(lineas)} (G {ganadas} | P {perdidas} "
          f"| PUSH {pushes})")
    print(f"  Unidades apostadas: {total_apostado:.2f} | "
          f"Unidades ganadas: {total_u:+.2f}")
    if clv_medio is not None:
        print(f"  CLV medio vs cierre: {clv_medio:+.2f} puntos "
              f"({len(con_cierre)} picks con linea de cierre)")
    if total_apostado > 0:
        print(f"  ROI real: {total_u / total_apostado:+.1%}")

    # Relacion edge vs acierto (alimenta la calibracion del #4).
    with_edge = [l for l in lineas if l["edge"] is not None]
    if len(with_edge) >= 5:
        print()
        print("  Edge del modelo vs acierto (bucket > 2.5):")
        altos = [l for l in with_edge if l["edge"] > 2.5]
        bajos = [l for l in with_edge if l["edge"] <= 2.5]
        for nombre, grupo in (("edge <= 2.5", bajos), ("edge > 2.5", altos)):
            if not grupo:
                continue
            ok = sum(1 for l in grupo if l["estado"] == "GANADA")
            print(f"    {nombre:<12} n={len(grupo):<3} acierto={ok / len(grupo):.0%}")
    else:
        print(f"  (aun sin muestra suficiente de edge: {len(with_edge)} picks "
              "resueltos)")


if __name__ == "__main__":
    analizar()
