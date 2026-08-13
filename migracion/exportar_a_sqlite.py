"""Exporta MLB_Predictive (SQL Server) a SQLite (data/mlb.db) con esquema fiel."""
import os
import sqlite3
import pyodbc

SQL_SERVER = ("DRIVER={ODBC Driver 18 for SQL Server};SERVER=RAI-FREITAS\\SQLEXPRESS;"
              "DATABASE=MLB_Predictive;Trusted_Connection=yes;Encrypt=no;")
DESTINO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "data", "mlb.db"))

TABLAS = [
    ("GameLog", [
        ("Id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("Fecha", "TEXT NOT NULL"),
        ("Estadio", "TEXT NOT NULL"),
        ("EquipoLocal", "TEXT NOT NULL"),
        ("EquipoVisita", "TEXT NOT NULL"),
        ("PitcherLocalId", "INTEGER"),
        ("PitcherVisitaId", "INTEGER"),
        ("CarrerasLocal", "INTEGER NOT NULL"),
        ("CarrerasVisita", "INTEGER NOT NULL"),
        ("TemperaturaC", "REAL"),
        ("Viento_Velocidad", "REAL"),
        ("Viento_Direccion", "TEXT"),
        ("ERA_Bullpen_Local", "REAL"),
        ("ERA_Bullpen_Visita", "REAL"),
        ("WHIP_Abridor_Local", "REAL"),
        ("WHIP_Abridor_Visita", "REAL"),
        ("UmpireNombre", "TEXT"),
        ("UmpireHomePlate", "TEXT"),
        ("Linea_Casino_Real", "REAL"),
        ("Cuota_Over_Real", "REAL"),
        ("Cuota_Under_Real", "REAL"),
        ("EsFinal", "INTEGER NOT NULL"),
        ("HoraInicioUtc", "TEXT"),
    ]),
    ("Predicciones", [
        ("Id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("Fecha", "TEXT NOT NULL"),
        ("EquipoLocal", "TEXT NOT NULL"),
        ("EquipoVisita", "TEXT NOT NULL"),
        ("TipoApuesta", "TEXT NOT NULL"),
        ("Linea", "REAL NOT NULL"),
        ("Unidades", "REAL NOT NULL"),
        ("Edge", "REAL"),
        ("Estado", "TEXT NOT NULL"),
        ("CarrerasTotales", "INTEGER"),
        ("FechaVerificacion", "TEXT"),
        ("CreadoUtc", "TEXT NOT NULL"),
        ("Cuota", "REAL"),
    ]),
    ("LineaSnapshots", [
        ("Id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("EventoId", "TEXT NOT NULL"),
        ("Casa", "TEXT NOT NULL"),
        ("Fecha", "TEXT NOT NULL"),
        ("EquipoLocal", "TEXT NOT NULL"),
        ("EquipoVisita", "TEXT NOT NULL"),
        ("Linea", "REAL"),
        ("CuotaOver", "REAL"),
        ("CuotaUnder", "REAL"),
        ("CapturadoUtc", "TEXT NOT NULL"),
    ]),
    ("LineaSnapshotsML", [
        ("Id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("EventoId", "TEXT"),
        ("Casa", "TEXT"),
        ("Fecha", "TEXT"),
        ("EquipoLocal", "TEXT NOT NULL"),
        ("EquipoVisita", "TEXT NOT NULL"),
        ("CuotaHome", "REAL"),
        ("CuotaAway", "REAL"),
        ("CapturadoUtc", "TEXT NOT NULL"),
    ]),
    ("PrediccionesML", [
        ("Id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("Fecha", "TEXT NOT NULL"),
        ("EquipoLocal", "TEXT NOT NULL"),
        ("EquipoVisita", "TEXT NOT NULL"),
        ("TipoApuesta", "TEXT NOT NULL"),
        ("Linea", "REAL"),
        ("Unidades", "REAL NOT NULL"),
        ("Edge", "REAL"),
        ("ProbModelo", "REAL"),
        ("Estado", "TEXT NOT NULL"),
        ("CarrerasLocal", "INTEGER"),
        ("CarrerasVisita", "INTEGER"),
        ("FechaVerificacion", "TEXT"),
        ("CreadoUtc", "TEXT NOT NULL"),
        ("Cuota", "REAL"),
    ]),
    ("PitcherGameLog", [
        ("GameID", "INTEGER NOT NULL"),
        ("Fecha", "TEXT NOT NULL"),
        ("Team", "TEXT NOT NULL"),
        ("PitcherID", "INTEGER NOT NULL"),
        ("IsStarter", "INTEGER NOT NULL"),
        ("PitchesThrown", "INTEGER NOT NULL"),
        ("PRIMARY KEY (GameID, Team, PitcherID)",),
    ]),
    ("PitcherMano", [
        ("PitcherId", "INTEGER PRIMARY KEY"),
        ("Mano", "TEXT NOT NULL"),
    ]),
    ("TeamOPS_Handedness", [
        ("Equipo", "TEXT NOT NULL"),
        ("Temporada", "INTEGER NOT NULL"),
        ("OPSvsLHP", "REAL"),
        ("OPSvsRHP", "REAL"),
        ("PRIMARY KEY (Equipo, Temporada)",),
    ]),
    ("HistoricalOdds", [
        ("EventoId", "TEXT NOT NULL"),
        ("Casa", "TEXT NOT NULL"),
        ("Fecha", "TEXT NOT NULL"),
        ("EquipoLocal", "TEXT NOT NULL"),
        ("EquipoVisita", "TEXT NOT NULL"),
        ("CommenceTimeUtc", "TEXT"),
        ("Linea", "REAL"),
        ("CuotaOver", "REAL"),
        ("CuotaUnder", "REAL"),
        ("UltimaActualizacion", "TEXT"),
        ("PRIMARY KEY (EventoId, Casa)",),
    ]),
    ("ParkFactors", [
        ("EquipoLocal", "TEXT PRIMARY KEY"),
        ("Factor_Carreras", "REAL"),
    ]),
]

COLUMNAS = {t[0]: [c[0] for c in t[1] if not c[0].startswith("PRIMARY")
                   and c[0] != "Id" and "PRIMARY KEY" not in c[0]]
            for t in TABLAS}


def convertir(v):
    """Normaliza valores pyodbc a tipos SQLite nativos."""
    import datetime as _dt
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, _dt.datetime):
        return v.isoformat(sep=" ")[:23]
    if isinstance(v, _dt.date):
        return v.isoformat()
    if isinstance(v, _dt.time):
        return str(v)
    if isinstance(v, (int, float, str, bytes)):
        return v
    try:
        return float(v)  # decimal.Decimal y otros numericos
    except (TypeError, ValueError):
        try:
            return str(v)
        except Exception:
            return None


def detectar_columna(col):
    return [c[1] for c in dict(TABLAS)  # placeholder, no usar
            .get("")][0] if False else None


def main():
    if os.path.exists(DESTINO):
        os.remove(DESTINO)
    os.makedirs(os.path.dirname(DESTINO), exist_ok=True)
    conn_sql = pyodbc.connect(SQL_SERVER, timeout=20)
    conn_lite = sqlite3.connect(DESTINO)
    try:
        for nombre, columnas in TABLAS:
            defs = ", ".join(
                c[0] if len(c) == 1 else f"{c[0]} {c[1]}" for c in columnas)
            conn_lite.execute(f"CREATE TABLE {nombre} ({defs})")
        for nombre, columnas in TABLAS:
            cols = [c[0] for c in columnas
                    if not c[0].startswith("PRIMARY") and "PRIMARY KEY" not in c[0]]
            lista_cols = ", ".join(cols)
            lst = ", ".join("?" * len(cols))
            cur = conn_sql.cursor()
            cur.execute(f"SELECT {lista_cols} FROM dbo.{nombre}")
            filas = [tuple(convertir(v) for v in fila) for fila in cur.fetchall()]
            conn_lite.executemany(f"INSERT INTO {nombre} ({lista_cols}) "
                                  f"VALUES ({lst})", filas)
            conn_lite.commit()
            print(f"{nombre:<22} {len(filas):>10} filas -> "
                  f"[{cols[0]} {type(filas[0][0]).__name__ if filas else '-'}]")
    finally:
        conn_sql.close()
        conn_lite.close()
    mb = round(os.path.getsize(DESTINO) / 1e6, 1)
    print(f"\nSQLite creado: {DESTINO} ({mb} MB)")


if __name__ == "__main__":
    main()