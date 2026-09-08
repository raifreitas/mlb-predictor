"""Predicciones en vivo de Batter Props (K/BB over 0.5) para partidos de hoy.

Flujo:
  1. Calendario MLB de la fecha (solo juegos Pre-Game/Scheduled).
  2. Por juego lee el boxscore PRE-GAME: alineaciones (battingOrder list) y
     abridor rival (pitchers[0] del equipo contrario), igual que el ETL.
  3. Arma filas sinteticas con los lineups de hoy y las fusiona al historico
     pasando por el MISMO feature engineering del entrenamiento. El diseno
     antifuga (ventanas con Fecha < hoy) garantiza que el modelo solo usa
     datos anteriores a hoy; los targets de hoy quedan como placeholder y
     nunca se usan.
  4. Carga modelos finales + medianas del entrenamiento y predice
     P(K>=1) y P(BB>=1) por bateador.
  5. Registra cada prediccion en la tabla PrediccionBatterProps (UPSERT por
     GameId+BatterId) y muestra una tabla por juego.

Modo --validar: cruza predicciones ya guardadas contra el boxscore REAL
descargado por el ETL (BatterGameLog) y mide hit-rate, logloss y AUC out-of-
sample. Esto es lo que valida si el edge del walk-forward se sostiene en vivo.

Uso:
  python generar_predicciones_hoy.py                       # partidos de hoy
  python generar_predicciones_hoy.py --fecha 2026-09-07
  python generar_predicciones_hoy.py --validar
"""
import argparse
import os
import sqlite3
import sys
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ETL_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "EtlPython"))
if ETL_DIR not in sys.path:
    sys.path.insert(0, ETL_DIR)

from features_batter_props import (FEATURES, construir_features,  # noqa: E402
                                   leer_datos)

RAIZ = os.path.normpath(os.path.join(BASE_DIR, "..", "..", ".."))
DB_DEFECTO = os.path.join(RAIZ, "data", "mlb.db")
MODELOS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "models"))
RUTA_MODELO_SO = os.path.join(MODELOS_DIR, "batter_props_strikeout.pkl")
RUTA_MODELO_BB = os.path.join(MODELOS_DIR, "batter_props_walk.pkl")
RUTA_TRANSFORMADORES = os.path.join(MODELOS_DIR, "transformadores_batter_props.pkl")

UMBRAL_SO_PICK = 0.60

BASE_API = "https://statsapi.mlb.com/api/v1"


def _conectarse(ruta_db):
    con = sqlite3.connect(ruta_db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS PrediccionBatterProps (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            Fecha TEXT NOT NULL,
            GameId INTEGER NOT NULL,
            BatterId INTEGER NOT NULL,
            Nombre TEXT,
            Team TEXT,
            IsHome INTEGER,
            BattingOrder INTEGER,
            OppStarterId INTEGER,
            PStrikeOut REAL,
            PWalk REAL,
            FechaPrediccion TEXT NOT NULL,
            UNIQUE (GameId, BatterId))
    """)
    return con


def _juegos_de_la_fecha(fetcher, fecha):
    """[(gamePk, fecha, equipo_visita, equipo_local)] Pre-Game o Scheduled."""
    url = (f"{BASE_API}/schedule?sportId=1&startDate={fecha}"
           f"&endDate={fecha}")
    try:
        datos = fetcher._get_json(url)
    except Exception as ex:
        print(f"[SCHEDULE] Error {fecha}: {ex}")
        return []
    juegos = []
    for dia in datos.get("dates", []):
        for juego in dia.get("games", []):
            estado = juego.get("status", {}).get("abstractGameState", "")
            if estado not in ("Preview", "Scheduled"):
                continue
            if juego.get("gameType", "") in ("S", "E", "A"):
                continue
            game_pk = juego.get("gamePk")
            if not game_pk:
                continue
            fec = juego.get("officialDate", str(fecha))
            visita = (juego.get("teams", {}).get("away", {})
                      .get("team", {}).get("name", "Desconocido"))
            local = (juego.get("teams", {}).get("home", {})
                     .get("team", {}).get("name", "Desconocido"))
            juegos.append((game_pk, fec, visita, local))
    return juegos


def _lineups_partido(fetcher, game_pk, fecha, eq_visita, eq_local):
    """Filas sinteticas de bateadores + abridores desde el boxscore pre-game.

    Requiere los DOS lineups publicados (teams.*.battingOrder con 9 slots).
    El abridor rival de cada lado es pitchers[0] del equipo contrario,
    exactamente igual que en obtener_batter_logs_partido / obtener_pitchers.
    Devuelve (filas_bateadores, filas_abridores, nombres) o None.
    """
    try:
        datos = fetcher._get_json(f"{BASE_API}/game/{game_pk}/boxscore")
    except Exception as ex:
        print(f"  [MLB] Error boxscore {game_pk}: {ex}")
        return None
    equipos = datos.get("teams")
    if not equipos:
        return None
    filas = []
    abridores = []
    nombres = {}
    lineups_publicados = 0
    for lado, es_home in (("home", 1), ("away", 0)):
        lado_json = equipos.get(lado)
        if not lado_json:
            return None
        nombre_equipo = lado_json.get("team", {}).get("name", "Desconocido")
        orden = lado_json.get("battingOrder", [])
        if not orden:
            print(f"  ... {game_pk} {nombre_equipo}: sin lineup publicado.")
            return None
        jugadores = lado_json.get("players", {})
        oponente = equipos.get("away" if es_home else "home", {})
        abridor_rival = (oponente.get("pitchers") or [None])[0]

        for slot, batter_id in enumerate(orden, 1):
            jugador = jugadores.get(f"ID{batter_id}", {})
            nombre = (jugador.get("person", {}).get("fullName")
                      or f"Batter{batter_id}")
            nombres[batter_id] = nombre
            filas.append({
                "GameId": game_pk,
                "Fecha": fecha,
                "EquipoLocal": eq_local,
                "EquipoVisita": eq_visita,
                "IsHome": es_home,
                "Team": nombre_equipo,
                "BatterId": batter_id,
                "BattingOrder": slot,
                "PlateAppearances": 1,
                "AtBats": 1,
                "StrikeOuts": 0,
                "BaseOnBalls": 0,
                "IsStarter": 1,
                "OpposingPitcherId": abridor_rival,
            })

        abridor_local = (lado_json.get("pitchers") or [None])[0]
        if abridor_local:
            abridores.append({
                "GameId": game_pk,
                "Fecha": fecha,
                "PitcherId": abridor_local,
                "IsStarter": 1,
                "StrikeOuts": 0,
                "BaseOnBalls": 0,
                "BattersFaced": 1,
            })
        lineups_publicados += 1
    if lineups_publicados < 2:
        return None
    return filas, abridores, nombres


def _manos_disponibles(con):
    try:
        rows = con.execute(
            "SELECT PitcherId, Mano FROM PitcherMano").fetchall()
        return {pid: mano for pid, mano in rows if mano}
    except sqlite3.OperationalError:
        return {}


def _predecir(df_hoy, transformadores, modelos, fecha):
    probs = {}
    for target, modelo in modelos.items():
        medianas = transformadores["medianas"][target]
        X = df_hoy[FEATURES].copy().fillna(medianas)
        probs[target] = modelo.predict_proba(X)[:, 1]
    return probs


def _guardar(con, pred):
    con.executemany("""
        INSERT INTO PrediccionBatterProps
            (Fecha, GameId, BatterId, Nombre, Team, IsHome, BattingOrder,
             OppStarterId, PStrikeOut, PWalk, FechaPrediccion)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (GameId, BatterId) DO UPDATE SET
            PStrikeOut = excluded.PStrikeOut,
            PWalk = excluded.PWalk,
            FechaPrediccion = excluded.FechaPrediccion
    """, [tuple(p.values()) for p in pred])
    con.commit()


def _validar(con):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS EvaluacionBatterProps (
            Fecha TEXT NOT NULL,
            GameId INTEGER NOT NULL,
            BatterId INTEGER NOT NULL,
            Mercado TEXT NOT NULL,
            Probabilidad REAL NOT NULL,
            Resultado INTEGER NOT NULL,
            Predicho INTEGER NOT NULL,
            FechaPrediccion TEXT,
            PRIMARY KEY (Fecha, GameId, BatterId, Mercado)
        )
    """)
    q = """
        SELECT p.Fecha, p.GameId, p.BatterId, p.Nombre, p.Team, p.IsHome,
               p.BattingOrder, p.PStrikeOut, p.PWalk, p.FechaPrediccion,
               b.StrikeOuts AS real_so, b.BaseOnBalls AS real_bb
          FROM PrediccionBatterProps p
          JOIN BatterGameLog b
            ON b.GameId = p.GameId AND b.BatterId = p.BatterId
    """
    df = pd.read_sql_query(q, con)
    if df.empty:
        print("Sin predicciones que validar (todavia no hay boxscores "
              "reales descargados).")
        return
    for col, real_col, tag, umbral_pick in (
            ("PStrikeOut", "real_so", "K", UMBRAL_SO_PICK),
            ("PWalk", "real_bb", "BB", None)):
        y = (df[real_col].fillna(0) >= 1).astype(int)
        p = df[col].clip(0.0001, 0.9999)
        n = len(y)
        if umbral_pick is None:
            predic_cuant = np.zeros(n, dtype=int)
        else:
            predic_cuant = (p >= umbral_pick).astype(int)
        from sklearn.metrics import log_loss, roc_auc_score
        ll = log_loss(y, p, labels=[0, 1])
        basel = log_loss(y, np.full(n, y.mean()), labels=[0, 1])
        auc = (roc_auc_score(y, p)
               if len(np.unique(y)) > 1 else float("nan"))
        print(f"\n{tag}: n={n} | tasa real={y.mean():.3f} | "
              f"logloss={ll:.4f} (baseline {basel:.4f}) | AUC={auc:.4f}")
        n_pick = int(predic_cuant.sum())
        if umbral_pick is None:
            print(f"  (BB no se apuesta; solo se muestra como info)")
        elif n_pick:
            gan = int(((predic_cuant == 1) & (y == 1)).sum())
            per = n_pick - gan
            print(f"  picks (P>={umbral_pick}): {n_pick} -> "
                  f"{gan} ganadas / {per} perdidas "
                  f"({gan / n_pick * 100:.1f}%)")
        else:
            print(f"  picks (P>={umbral_pick}): 0")
        mercado = tag
        filas = [(f_r, int(g), int(b), mercado, float(pr),
                  int(re), int(pd), fp)
                 for f_r, g, b, pr, re, pd, fp in zip(
                     df["Fecha"], df["GameId"], df["BatterId"], p.values,
                     y.values, predic_cuant.values,
                     df["FechaPrediccion"])]
        con.executemany("""
            INSERT INTO EvaluacionBatterProps
                (Fecha, GameId, BatterId, Mercado, Probabilidad,
                 Resultado, Predicho, FechaPrediccion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (Fecha, GameId, BatterId, Mercado) DO UPDATE SET
                Probabilidad = excluded.Probabilidad,
                Resultado = excluded.Resultado,
                Predicho = excluded.Predicho,
                FechaPrediccion = excluded.FechaPrediccion
        """, filas)
    con.commit()
    print(f"\nResultados persistidos en EvaluacionBatterProps "
          f"({con.total_changes} filas).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fecha", default=date.today().isoformat())
    parser.add_argument("--db", default=DB_DEFECTO)
    parser.add_argument("--solo-juegos", type=int, nargs="*",
                        help="solo estos gamePk")
    parser.add_argument("--solo-faltantes", action="store_true",
                        help="salir rapido si todos los juegos pre-game del "
                             "dia ya tienen prediccion")
    parser.add_argument("--validar", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    con = _conectarse(args.db)

    if args.validar:
        _validar(con)
        con.close()
        return

    from mlb_data_fetcher import MlbDataFetcher

    fetcher = MlbDataFetcher(BASE_API)
    juegos = _juegos_de_la_fecha(fetcher, args.fecha)
    if args.solo_juegos:
        juegos = [j for j in juegos if j[0] in args.solo_juegos]
    if not juegos:
        print(f"No hay juegos Pre-Game/Scheduled el {args.fecha}.")
        con.close()
        return
    print(f"{len(juegos)} juegos pre-game el {args.fecha}.")

    if args.solo_faltantes:
        predichos = {r[0] for r in con.execute(
            "SELECT DISTINCT GameId FROM PrediccionBatterProps "
            "WHERE Fecha = ?", [args.fecha]).fetchall()}
        faltantes = [j for j in juegos if j[0] not in predichos]
        if not faltantes:
            print(f"Todos los juegos pre-game de {args.fecha} ya tienen "
                  "prediccion. Nada nuevo.")
            con.close()
            return
        print(f"{len(faltantes)} juego(s) sin prediccion aun (lineups "
              "pendientes). Reintentando...")

    modelos = {
        "target_strikeout": joblib.load(RUTA_MODELO_SO),
        "target_walk": joblib.load(RUTA_MODELO_BB),
    }
    transformadores = joblib.load(RUTA_TRANSFORMADORES)
    print(f"Modelos cargados (features={len(FEATURES)}).\n")

    # Historico completo (el mismo insumo del entrenamiento).
    bateadores, abridores, manos = leer_datos(args.db)
    mano_map = _manos_disponibles(con)
    manos_df = pd.DataFrame(
        list(mano_map.items()), columns=["PitcherId", "Mano"])

    total_pred = 0
    fecha_ts = pd.Timestamp(args.fecha)
    for game_pk, fec, eq_visita, eq_local in juegos:
        res = _lineups_partido(fetcher, game_pk, fec, eq_visita, eq_local)
        if res is None:
            print(f"  [SKIP] {eq_visita} @ {eq_local} ({game_pk}): "
                  "lineups aun no publicados.")
            continue
        filas_bat, filas_ab, nombres = res

        # Manos de los abridores de hoy no conocidas -> API (cache local).
        for ab in filas_ab:
            pid = ab["PitcherId"]
            if pid not in mano_map:
                m = fetcher.obtener_mano_lanzamiento(pid)
                mano_map[pid] = m
        manos_df = pd.DataFrame(
            list(mano_map.items()), columns=["PitcherId", "Mano"])

        bat_hoy = pd.DataFrame(filas_bat)
        ab_hoy = pd.DataFrame(filas_ab)
        df = construir_features(
            pd.concat([bateadores, bat_hoy], ignore_index=True),
            pd.concat([abridores, ab_hoy], ignore_index=True, sort=False),
            manos_df)
        df_hoy = df[df["Fecha"] >= fecha_ts].copy()

        probs = _predecir(df_hoy, transformadores, modelos, fecha_ts)
        df_hoy = df_hoy.assign(
            p_so=probs["target_strikeout"], p_bb=probs["target_walk"])
        df_hoy = df_hoy.sort_values(["EsLocal", "Orden_Al_Bate"])

        lookup = {f["BatterId"]: f for f in filas_bat}
        print(f"=== {eq_visita} @ {eq_local} (gamePk {game_pk}) ===")
        print(f"    {'Bateador':<28}{'Team':<6}{'slot':>5}{'H?':>3}"
              f"{'P(K>=1)':>9}{'P(BB>=1)':>9}")
        pred = []
        for _, r in df_hoy.iterrows():
            bid = int(r["BatterId"])
            f_orig = lookup.get(bid, {})
            print(f"    {nombres.get(bid, bid):<28}{str(f_orig.get('Team', ''))[:5]:<6}"
                  f"{int(r['Orden_Al_Bate']):>5}{int(r['EsLocal']):>3}"
                  f"{r['p_so']:>9.4f}{r['p_bb']:>9.4f}")
            pred.append({
                "Fecha": args.fecha,
                "GameId": game_pk,
                "BatterId": bid,
                "Nombre": nombres.get(bid, ""),
                "Team": str(f_orig.get("Team", "")),
                "IsHome": int(r["EsLocal"]),
                "BattingOrder": int(r["Orden_Al_Bate"]),
                "OppStarterId": (None if f_orig.get("OpposingPitcherId") is None
                                 else int(f_orig["OpposingPitcherId"])),
                "PStrikeOut": float(r["p_so"]),
                "PWalk": float(r["p_bb"]),
                "FechaPrediccion": datetime.now().isoformat(timespec="seconds"),
            })
        _guardar(con, pred)
        total_pred += len(pred)
        print()

    print(f"Predicciones guardadas: {total_pred} apariciones en "
          f"PrediccionBatterProps.")
    con.close()


if __name__ == "__main__":
    raise SystemExit(main())