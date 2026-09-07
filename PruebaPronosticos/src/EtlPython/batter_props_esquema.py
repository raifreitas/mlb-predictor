"""Esquema (idempotente) para el pipeline de Batter Props.

Extiende la BD SQLite sin romper tablas existentes:
  1) Crea la tabla BatterGameLog (una fila por aparicion de cada bateador,
     con su orden al bate, K y BB reales del partido).
  2) Agrega a PitcherGameLog las columnas StrikeOuts / BaseOnBalls /
     BattersFaced, necesarias para calcular las tasas K% y BB% del abridor.

El proyecto pivota a props individuales de bateador: las filas aqui
capturadas son el insumo crudo (por partido) del feature engineering.
"""
import sqlite3

SQL_CREATE_BATTER_GAME_LOG = """
CREATE TABLE IF NOT EXISTS BatterGameLog (
    GameId             INTEGER NOT NULL,
    Fecha              TEXT NOT NULL,
    EquipoLocal        TEXT NOT NULL,
    EquipoVisita       TEXT NOT NULL,
    IsHome             INTEGER NOT NULL,
    Team               TEXT,
    BatterId           INTEGER NOT NULL,
    BattingOrder       INTEGER,
    PlateAppearances   INTEGER,
    AtBats             INTEGER,
    StrikeOuts         INTEGER,
    BaseOnBalls        INTEGER,
    IsStarter          INTEGER NOT NULL DEFAULT 1,
    OpposingPitcherId  INTEGER,
    PRIMARY KEY (GameId, BatterId)
)
"""

# Columnas nuevas sobre PitcherGameLog (K/BB/BF del lanzador en el partido).
ALTERS_PITCHER_GAME_LOG = [
    "ALTER TABLE PitcherGameLog ADD COLUMN StrikeOuts INTEGER",
    "ALTER TABLE PitcherGameLog ADD COLUMN BaseOnBalls INTEGER",
    "ALTER TABLE PitcherGameLog ADD COLUMN BattersFaced INTEGER",
]


def garantizar_esquema(con):
    """Crea/actualiza el esquema de batter props. Idempotente."""
    con.execute(SQL_CREATE_BATTER_GAME_LOG)
    columnas = {fila[1] for fila in
                con.execute("PRAGMA table_info(PitcherGameLog)")}
    for ddl in ALTERS_PITCHER_GAME_LOG:
        columna = ddl.split("ADD COLUMN ")[1].split()[0]
        if columna in columnas:
            continue
        try:
            con.execute(ddl)
            columnas.add(columna)
        except sqlite3.OperationalError:
            # Carrera de esquema en flujos concurrentes; ya existe.
            pass
    con.commit()