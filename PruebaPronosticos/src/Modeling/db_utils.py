"""Capa de acceso a BD: SQL Server (pyodbc, local) o SQLite (nube).

Activacion: variable de entorno MLB_SQLITE=1 -> usa data/mlb.db.
Traduce el T-SQL del pipeline (dbo., DATEADD, YEAR, SYSUTCDATETIME)
a SQLite en vuelo y crea la vista vwFatigaBullpen3d si falta.
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
RAIZ = Path(CARPETA_SCRIPT).resolve().parents[2]  # raiz del repo (con data/)
RUTA_SQLITE = Path(os.environ.get(
    "MLB_DB_PATH", str(RAIZ / "data" / "mlb.db"))).resolve()

SQL_CREATE_EVALUACIONES = """
CREATE TABLE IF NOT EXISTS Evaluaciones (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Fecha TEXT NOT NULL,
    EquipoLocal TEXT NOT NULL,
    EquipoVisita TEXT NOT NULL,
    Linea REAL,
    Prediccion REAL,
    ProbOver REAL,
    Edge REAL,
    Recomendacion TEXT,
    Motivo TEXT,
    EvaluadoUtc TEXT
)
"""


def crear_tabla_evaluaciones(con):
    """Crea la tabla Evaluaciones si falta (SQLite)."""
    con.execute(SQL_CREATE_EVALUACIONES)
    con.commit()


SQL_VIEW_FATIGA = """
CREATE VIEW IF NOT EXISTS vwFatigaBullpen3d AS
WITH Diario AS (
    SELECT Team, Fecha, SUM(PitchesThrown) AS TotalPitches
    FROM PitcherGameLog
    WHERE IsStarter = 0
    GROUP BY Team, Fecha
)
SELECT Team, Fecha,
       IFNULL(SUM(TotalPitches) OVER (
           PARTITION BY Team ORDER BY Fecha
           ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING), 0) AS Fatiga_Bullpen_3d
FROM Diario
"""


def usar_sqlite():
    return os.environ.get("MLB_SQLITE", "").strip() == "1"


def _conexion_sqlite():
    con = sqlite3.connect(str(RUTA_SQLITE), timeout=60)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute(SQL_VIEW_FATIGA)
    crear_tabla_evaluaciones(con)
    return con


def conexion():
    """Devuelve conexion sqlite3 (nube) o pyodbc (local)."""
    if usar_sqlite():
        return _conexion_sqlite()
    import pyodbc
    from entrenar_modelo import CONNECTION_STRING_TEMPLATE, obtener_driver_odbc
    return pyodbc.connect(CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc()))


def _traducir_sql(consulta):
    if not usar_sqlite():
        return consulta
    t = consulta.replace("dbo.", "")
    t = t.replace("SYSUTCDATETIME()", "datetime('now')")
    return t


def _traducir_sqlite_extra(sql):
    """Ajustes puntuales T-SQL -> SQLite no cubiertos por reemplazos globales."""
    sql = sql.replace("YEAR(g.Fecha)", "CAST(strftime('%Y', g.Fecha) AS INTEGER)")
    sql = sql.replace("DATEADD(DAY, -3, ?)", "date(?, '-3 day')")
    import re
    sql = re.sub(r"SELECT\s+TOP\s+(\d+)\s+(.*)$",
                 lambda m: f"SELECT {m.group(2).rstrip()} LIMIT {m.group(1)}",
                 sql, flags=re.S)
    sql = sql.replace("ISNULL(", "IFNULL(")
    return sql


def _adaptar_params(params):
    if not usar_sqlite():
        return params
    if params is None:
        return None
    return [str(p) if hasattr(p, "isoformat") else p for p in params]


def leer_sql(consulta, params=None):
    """DataFrame con el resultado de una SELECT (T-SQL o SQLite)."""
    con = conexion()
    try:
        sql = _traducir_sql(consulta)
        if usar_sqlite():
            sql = _traducir_sqlite_extra(sql)
        return pd.read_sql(sql, con, params=_adaptar_params(params))
    finally:
        con.close()


def ejecutar(consulta, params=None):
    """Ejecuta un UPDATE/DELETE/INSERT y hace commit."""
    con = conexion()
    try:
        cur = con.cursor()
        cur.execute(_traducir_sql(consulta), _adaptar_params(params))
        con.commit()
    finally:
        con.close()