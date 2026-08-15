"""Aviso por Telegram de picks nuevos de Totales (Over/Under).

Corre en runner-mlb inmediatamente despues del planificador: detecta picks
PENDIENTE creados en los ultimos MINUTOS_VENTANA y manda un mensaje por el
Bot API de Telegram. Una tabla Notificaciones (en la misma BD, SQLite en la
nube) guarda los IdPick ya avisados para no duplicar mensajes entre ticks.

Variables de entorno: TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import requests

import db_utils

MINUTOS_VENTANA = 12
API = "https://api.telegram.org/bot{token}/sendMessage"

SQL_CREATE_NOTIFICACIONES = """
CREATE TABLE IF NOT EXISTS Notificaciones (
    IdPick INTEGER PRIMARY KEY,
    NotificadoUtc TEXT NOT NULL
)
"""

SELECT_PICKS = """
SELECT Id, Fecha, EquipoLocal, EquipoVisita, TipoApuesta, Linea,
       Unidades, Edge, Cuota, CreadoUtc
FROM Predicciones
WHERE Estado = 'PENDIENTE' AND CreadoUtc >= ?
ORDER BY CreadoUtc
"""

INSERT_NOTIFICADO = (
    "INSERT OR IGNORE INTO Notificaciones (IdPick, NotificadoUtc) "
    "VALUES (?, ?)")
YA_NOTIFICADO = "SELECT 1 FROM Notificaciones WHERE IdPick = ?"


def _texto_pick(f):
    mercado = (f["TipoApuesta"] or "").upper()
    return (
        "NUEVO PICK MLB\n"
        f"Fecha: {f['Fecha']}\n"
        f"Equipos: {f['EquipoLocal']} vs {f['EquipoVisita']}\n"
        f"Mercado: {mercado} {f['Linea']:.1f} @ {f['Cuota']:.2f}\n"
        f"Unidades: {f['Unidades']:.2f} | Edge: {f['Edge']:.1f}%\n"
        f"Creado: {f['CreadoUtc']} UTC")


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not db_utils.usar_sqlite():
        print("[NOTIF] Sin avisos: solo funciona con SQLite (MLB_SQLITE=1).")
        return 0
    if not token or not chat_id:
        print("[NOTIF] Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID; "
              "revisar secrets del workflow. Sin avisos por ahora.")
        return 0

    corte = (datetime.now(timezone.utc) - timedelta(minutes=MINUTOS_VENTANA)
             ).strftime("%Y-%m-%d %H:%M:%S")
    df = db_utils.leer_sql(SELECT_PICKS, [corte])
    if df.empty:
        print("[NOTIF] Sin picks nuevos en la ventana.")
        return 0

    con = db_utils.conexion()
    try:
        con.execute(SQL_CREATE_NOTIFICACIONES)
        pendientes = []
        for _, f in df.iterrows():
            ya = con.execute(YA_NOTIFICADO, [f["Id"]]).fetchone()
            if not ya:
                pendientes.append(f)
        if not pendientes:
            print("[NOTIF] Todo ya notificado.")
            return 0

        ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for f in pendientes:
            r = requests.post(
                API.format(token=token),
                json={"chat_id": chat_id, "text": _texto_pick(f)},
                timeout=30)
            if r.status_code != 200:
                print(f"[NOTIF] Error de Telegram al avisar pick {f['Id']}: "
                      f"{r.status_code} {r.text[:200]}")
                continue
            con.execute(INSERT_NOTIFICADO, [f["Id"], ahora])
        con.commit()
        print(f"[NOTIF] Avisados {len(pendientes)} pick(s) nuevos por Telegram.")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())