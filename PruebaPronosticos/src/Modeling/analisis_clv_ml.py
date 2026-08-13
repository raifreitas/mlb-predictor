"""Analisis de CLV (Closing Line Value) del mercado Moneyline.

Mide, contra las cuotas h2h de cierre del mercado (snapshots del ETL en
dbo.LineaSnapshotsML -> moda de cierre por partido), la calidad de cada
pick registrado en dbo.PrediccionesML:

- CLV en puntos de cuota: cuanto mejoro la cuota apostada frente al
  cierre del mercado (CLV > 0 => apostamos mejor que el cierre).
- ROI real: unidades ganadas con la cuota apostada.
- Relacion desacuerdo modelo vs mercado (edge en probabilidad) y acierto:
  alimenta la calibracion futura de PESO_MERCADO_MAX_ML / EDGE_MAXIMO_ML.

La cuota de cierre por partido se toma de la MODA de CuotaHome entre
casas en LineaSnapshotsML; para picks AWAY se usa la moda de CuotaAway.

Uso:
    python analisis_clv_ml.py [--detalle]
"""

import sys
from collections import Counter
from statistics import median as _mediana

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


def cuota_cierre(conexion, fecha, local, visita, tipo):
    """Moda de la cuota del lado apostado entre casas (snapshot de cierre).

    El snapshot mas reciente capturado por el ETL es la mejor proxy de la
    linea de cierre disponible con el plan gratuito de The Odds API.
    """
    filas = conexion.execute(
        "SELECT CuotaHome, CuotaAway FROM dbo.LineaSnapshotsML "
        "WHERE Fecha = ? AND EquipoLocal = ? AND EquipoVisita = ? "
        "AND CuotaHome IS NOT NULL AND CuotaAway IS NOT NULL "
        "ORDER BY CapturadoUtc DESC",
        fecha, local, visita).fetchall()
    if not filas:
        return None
    valores = [float(f[0]) for f in filas] if tipo == "HOME" \
        else [float(f[1]) for f in filas]
    if not valores:
        return None
    moda = Counter(valores).most_common(1)[0][0]
    return moda


def analizar():
    detalle = "--detalle" in sys.argv
    connection_string = CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc())
    conexion = pyodbc.connect(connection_string)
    try:
        filas = conexion.execute("""
            SELECT p.Fecha, p.EquipoLocal, p.EquipoVisita, p.TipoApuesta,
                   p.Cuota, p.Unidades, p.Edge, p.ProbModelo, p.Estado,
                   p.CarrerasLocal, p.CarrerasVisita
            FROM dbo.PrediccionesML p
            WHERE p.Estado <> 'PENDIENTE'
            ORDER BY p.Fecha, p.Id
        """).fetchall()
    finally:
        pass

    lineas = []
    if not filas:
        print("No hay predicciones ML resueltas aun (Estado <> PENDIENTE).")
        return

    for fila in filas:
        (fecha, local, visita, tipo, cuota_apostada, unidades,
         edge, prob_modelo, estado, carreras_local, carreras_visita) = fila

        cierre = cuota_cierre(conexion, fecha, local, visita, tipo)

        clv = None
        if cierre is not None and cuota_apostada:
            cuota_apostada = float(cuota_apostada)
            # CLV en puntos de cuota: mejor apostar 1.80 que cerrar en 1.75
            # (apostamos a mayor precio). Positivo = a nuestro favor.
            clv = cuota_apostada - cierre
            if abs(clv) < 0.01:
                clv = 0.0

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
            "cuota_apostada": cuota_apostada,
            "cuota_cierre": cierre,
            "clv": clv,
            "edge": float(edge) if edge else None,
            "prob_modelo": float(prob_modelo) if prob_modelo else None,
            "unidades": unidades,
            "estado": estado,
            "marcador": (f"{carreras_local}-{carreras_visita}"
                         if carreras_local is not None
                         and carreras_visita is not None else None),
            "ganancia": ganancia,
        })
    conexion.close()

    con_cierre = [l for l in lineas if l["clv"] is not None]
    clv_medio = sum(l["clv"] for l in con_cierre) / len(con_cierre) \
        if con_cierre else None

    print("=" * 100)
    print("ANALISIS CLV MONEYLINE (vs cuotas h2h de cierre LineaSnapshotsML)")
    print("=" * 100)
    if detalle or True:
        print(f"  {'Fecha':<10} {'Partido':<36} {'Tipo':<5} "
              f"{'Cuota':<6} {'Cierre':<6} {'CLV':<7} "
              f"{'Edge':<6} {'P(modelo)':<9} {'u':<5} "
              f"{'Estado':<8} {'Marcador':<7} {'Unid':>7}")
        print("  " + "-" * 108)
        for l in lineas:
            cuota_txt = f"{l['cuota_apostada']:.2f}" if l["cuota_apostada"] else "  -  "
            cierre_txt = f"{l['cuota_cierre']:.2f}" if l["cuota_cierre"] else "  -  "
            clv_txt = f"{l['clv']:+.2f}" if l["clv"] is not None else "  -  "
            edge_txt = f"{l['edge']:.3f}" if l["edge"] is not None else "  -  "
            prob_txt = f"{l['prob_modelo'] * 100:.0f}%" if l["prob_modelo"] else "  -  "
            marcador = l["marcador"] or "  -  "
            print(f"  {l['fecha']:<10} {l['partido']:<36} {l['tipo']:<5} "
                  f"{cuota_txt:<6} {cierre_txt:<6} {clv_txt:<7} "
                  f"{edge_txt:<6} {prob_txt:<9} "
                  f"{l['unidades']:<5.2f} {l['estado']:<8} {marcador:<7} "
                  f"{l['ganancia']:>+7.2f}")
        print("  " + "-" * 108)

    total_u = sum(l["ganancia"] for l in lineas)
    total_apostado = sum(l["unidades"] for l in lineas)
    ganadas = sum(1 for l in lineas if l["estado"] == "GANADA")
    perdidas = sum(1 for l in lineas if l["estado"] == "PERDIDA")
    print(f"  Resueltas: {len(lineas)} (G {ganadas} | P {perdidas})")
    print(f"  Unidades apostadas: {total_apostado:.2f} | "
          f"Unidades ganadas: {total_u:+.2f}")
    if clv_medio is not None:
        print(f"  CLV medio vs cierre: {clv_medio:+.2f} puntos de cuota "
              f"({len(con_cierre)} picks con cuota de cierre)")
    if total_apostado > 0:
        print(f"  ROI real: {total_u / total_apostado:+.1%}")

    with_edge = [l for l in lineas if l["edge"] is not None]
    if len(with_edge) >= 5:
        print()
        print("  Desacuerdo modelo vs mercado y acierto (bucket > 0.08):")
        altos = [l for l in with_edge if l["edge"] > 0.08]
        bajos = [l for l in with_edge if l["edge"] <= 0.08]
        for nombre, grupo in (("desac <= 0.08", bajos), ("desac > 0.08", altos)):
            if not grupo:
                continue
            ok = sum(1 for l in grupo if l["estado"] == "GANADA")
            print(f"    {nombre:<14} n={len(grupo):<3} "
                  f"acierto={ok / len(grupo):.0%}")
    else:
        print(f"  (aun sin muestra suficiente de edge: {len(with_edge)} "
              "picks resueltos)")


if __name__ == "__main__":
    analizar()
