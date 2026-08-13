# -*- coding: utf-8 -*-
"""Backfill dbo.TeamOPS_Handedness para 2015-2022.

Mismo metodo que el ETL diario (MlbDataFetcher.ObtenerOpsSplitsAsync):
  GET /teams/{id}/stats?stats=statSplits&group=hitting&season=YYYY
     &sitCodes=vl,vr&sportId=1
  -> split "vs Left"  -> OPSvsLHP
     split "vs Right" -> OPSvsRHP
UPSERT por (Equipo, Temporada).
"""

import sys
import time
import urllib.request
import json
import importlib.util
import pyodbc

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://statsapi.mlb.com/api/v1"
ANIOS = list(range(2015, 2023))

spec = importlib.util.spec_from_file_location(
    "entrenar_modelo", r"entrenar_modelo.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CONN_STR = mod.CONNECTION_STRING_TEMPLATE.format(driver=mod.obtener_driver_odbc())


def obtener_json(url, intentos=5):
    for intento in range(1, intentos + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MLB-Predictor/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"  [fallo {intento}/{intentos}] {url[:80]}... -> {e}")
            time.sleep(10 * intento)
    return None


def obtener_equipos(temporada):
    doc = obtener_json(f"{BASE}/teams?sportId=1&season={temporada}")
    if not doc:
        return []
    return [(t["id"], t["name"]) for t in doc.get("teams", [])]


def obtener_splits(equipo_id, temporada):
    doc = obtener_json(
        f"{BASE}/teams/{equipo_id}/stats?stats=statSplits&group=hitting"
        f"&season={temporada}&sitCodes=vl,vr&sportId=1")
    if not doc:
        return None, None
    ops_lhp = ops_rhp = None
    for stats in doc.get("stats", []):
        for split in stats.get("splits", []):
            nombre = None
            sp = split.get("split")
            if isinstance(sp, str):
                nombre = sp
            elif isinstance(sp, dict):
                nombre = sp.get("description")
            if not nombre:
                continue
            ops = split.get("stat", {}).get("ops")
            if ops is None:
                continue
            try:
                ops = float(ops)
            except (TypeError, ValueError):
                continue
            if "vs Left" in nombre or "vs LHP" in nombre:
                ops_lhp = ops
            elif "vs Right" in nombre or "vs RHP" in nombre:
                ops_rhp = ops
    return ops_lhp, ops_rhp


def main():
    conn = pyodbc.connect(CONN_STR)
    cur = conn.cursor()
    guardados = 0
    fallos = 0
    for temporada in ANIOS:
        equipos = obtener_equipos(temporada)
        print(f"[{temporada}] {len(equipos)} equipos")
        for equipo_id, nombre in equipos:
            ops_lhp, ops_rhp = obtener_splits(equipo_id, temporada)
            cur.execute(
                "IF EXISTS (SELECT 1 FROM dbo.TeamOPS_Handedness "
                "WHERE Equipo = ? AND Temporada = ?) "
                "UPDATE dbo.TeamOPS_Handedness "
                "SET OPSvsLHP = COALESCE(?, OPSvsLHP), OPSvsRHP = COALESCE(?, OPSvsRHP) "
                "WHERE Equipo = ? AND Temporada = ? "
                "ELSE INSERT INTO dbo.TeamOPS_Handedness "
                "(Equipo, Temporada, OPSvsLHP, OPSvsRHP) VALUES (?, ?, ?, ?)",
                nombre, temporada, ops_lhp, ops_rhp,
                nombre, temporada, nombre, temporada, ops_lhp, ops_rhp)
            if ops_lhp is None and ops_rhp is None:
                fallos += 1
            else:
                guardados += 1
            time.sleep(0.35)
        conn.commit()
        print(f"    guardados acumulados: {guardados}, sin datos: {fallos}")
    conn.close()
    print("=" * 60)
    print(f"TOTAL: {guardados} (Equipo, Temporada) con datos | {fallos} sin datos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
