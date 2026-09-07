"""Feature engineering para el mercado individual de Bateador (K/BB over 0.5).

Insume las tablas del esquema batter props (SQLite):
  - BatterGameLog   : una fila por aparicion -> PA/AB/SO/BB y orden al bate.
  - PitcherGameLog  : abridores con StrikeOuts/BaseOnBalls/BattersFaced.
  - PitcherMano     : mano ('L'/'R') de cada lanzador.

Reglas ANTIFUGA (Data Leakage):
  * Todas las tasas del bateador y del abridor usan UNICAMENTE juegos
    ANTERIORES al partido actual: shift(1) sobre acumulados por entidad.
  * La ventana de 30 dias usa asof sobre los acumulados: resta al acumulado
    previo el acumulado del ultimo juego <= (Fecha-30d). Con esto el partido
    actual NUNCA entra en la ventana ni en la tasa de temporada.
  * Los splits platoon (vs LHP / vs RHP) se calculan con la misma regla.
  * Ninguna feature usa el resultado real (SO/BB) del partido que se predice
    ni informacion de fechas futuras.

Modo inspeccion:  python features_batter_props.py --db ../data/mlb.db --salida feats.csv
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd

DIAS_VENTANA = 30
BF_FALLBACK = 27.0

FEATURES = [
    # Bateador: tasa de temporada y de los ultimos 30 dias.
    "bat_k_30", "bat_k_temp", "bat_bb_30", "bat_bb_temp",
    # Bateador: volumen de apariciones (fiabilidad de las tasas).
    "bat_pa_30", "bat_pa_temp",
    # Bateador: splits platoon (rendimiento contra la mano del abridor).
    "bat_k_vsL", "bat_k_vsR", "bat_bb_vsL", "bat_bb_vsR",
    "bat_pa_vsL", "bat_pa_vsR",
    # Cruzado bateador-mano de ese partido.
    "bat_k_vs_mano", "bat_bb_vs_mano", "bat_pa_vs_mano",
    # Abridor rival: K% y BB% recientes y de temporada + volumen (BF).
    "pit_k_30", "pit_k_temp", "pit_bb_30", "pit_bb_temp",
    "pit_pa_30", "pit_pa_temp",
    # Contexto del matchup.
    "EsLHP", "Orden_Al_Bate", "EsLocal", "Mes", "DiaTemporada",
]

TARGETS = ["target_strikeout", "target_walk"]


def leer_datos(ruta_db):
    con = sqlite3.connect(ruta_db)
    try:
        bateadores = pd.read_sql_query("SELECT * FROM BatterGameLog", con)
        abridores = pd.read_sql_query(
            "SELECT GameID AS GameId, Fecha, PitcherID AS PitcherId, "
            "       IsStarter, StrikeOuts, BaseOnBalls, BattersFaced"
            "  FROM PitcherGameLog WHERE IsStarter = 1", con)
        manos = pd.read_sql_query(
            "SELECT PitcherId, Mano FROM PitcherMano", con)
    finally:
        con.close()
    return bateadores, abridores, manos


def _tasas_previas(base, col_entidad, so_col, bb_col, pa_col, sufijo):
    """Tasas de temporada y de 30 dias de una entidad, SOLO con juegos previos.

    Acumula SO/BB/PA por entidad y luego descarta el juego actual (shift(1)).
    La ventana de 30 dias se obtiene restando al acumulado previo el acumulado
    del ultimo juego en o antes de (Fecha - 30 dias), via merge_asof.
    """
    tmp = base[["GameId", col_entidad, "Fecha", so_col, bb_col, pa_col]].copy()
    tmp = tmp.sort_values([col_entidad, "Fecha", "GameId"]).copy()
    grupo = tmp.groupby(col_entidad, sort=False)
    tmp["_so_cum"] = grupo[so_col].cumsum()
    tmp["_bb_cum"] = grupo[bb_col].cumsum()
    tmp["_pa_cum"] = grupo[pa_col].cumsum()
    grupo_prior = tmp.groupby(col_entidad, sort=False)
    tmp["_so_prior"] = grupo_prior["_so_cum"].shift(1)
    tmp["_bb_prior"] = grupo_prior["_bb_cum"].shift(1)
    tmp["_pa_prior"] = grupo_prior["_pa_cum"].shift(1)

    tmp["_window"] = tmp["Fecha"] - pd.Timedelta(days=DIAS_VENTANA)
    asof = tmp[["GameId", col_entidad, "Fecha",
                "_so_cum", "_bb_cum", "_pa_cum"]].rename(columns={
                    "Fecha": "fecha_asof",
                    "_so_cum": "_so_asof", "_bb_cum": "_bb_asof",
                    "_pa_cum": "_pa_asof"})
    # Acumulado "hasta el cierre de ese dia": el ultimo juego del dia (doble
    # cartelera) ya incluye ambos juegos.
    asof = asof.groupby([col_entidad, "fecha_asof"], as_index=False)[
        ["_so_asof", "_bb_asof", "_pa_asof"]].last()

    unido = pd.merge_asof(
        tmp.sort_values("_window"),
        asof.sort_values("fecha_asof"),
        left_on="_window", right_on="fecha_asof",
        by=col_entidad, direction="backward",
    )
    unido["_so_30"] = unido["_so_prior"] - unido["_so_asof"].fillna(0.0)
    unido["_bb_30"] = unido["_bb_prior"] - unido["_bb_asof"].fillna(0.0)
    unido["_pa_30"] = unido["_pa_prior"] - unido["_pa_asof"].fillna(0.0)

    def _tasa(num, den):
        return num / den.replace(0.0, np.nan)

    nombre_entidad = "BatterId" if col_entidad == "BatterId" else "PitcherId"
    salida = pd.DataFrame({
        "GameId": unido["GameId"].values,
        nombre_entidad: unido[col_entidad].values,
        f"{sufijo}_k_30": _tasa(unido["_so_30"], unido["_pa_30"]).values,
        f"{sufijo}_bb_30": _tasa(unido["_bb_30"], unido["_pa_30"]).values,
        f"{sufijo}_pa_30": unido["_pa_30"].values,
        f"{sufijo}_k_temp": _tasa(unido["_so_prior"], unido["_pa_prior"]).values,
        f"{sufijo}_bb_temp": _tasa(unido["_bb_prior"], unido["_pa_prior"]).values,
        f"{sufijo}_pa_temp": unido["_pa_prior"].values,
    })
    return salida


def _splits_platoon(df, so_col, bb_col, pa_col):
    """Acumulados de temporada por mano enfrentada (solo juegos previos)."""
    tmp = df[["GameId", "BatterId", "Fecha", "ManoOpp",
              so_col, bb_col, pa_col]].copy()
    tmp["_mano"] = tmp["ManoOpp"].map({"L": 1.0, "R": 0.0}).fillna(2.0)
    resultado = pd.DataFrame({"GameId": df["GameId"],
                              "BatterId": df["BatterId"]})
    for mano_val, tag in ((0.0, "R"), (1.0, "L")):
        parte = tmp[tmp["_mano"] == mano_val].copy()
        if parte.empty:
            for pref in ("k", "bb", "pa"):
                resultado[f"bat_{pref}_vs{tag}"] = np.nan
            continue
        parte = parte.sort_values(["BatterId", "Fecha", "GameId"]).copy()
        grupo = parte.groupby("BatterId", sort=False)
        parte["_so_cum"] = grupo[so_col].cumsum()
        parte["_bb_cum"] = grupo[bb_col].cumsum()
        parte["_pa_cum"] = grupo[pa_col].cumsum()
        grupo_prior = parte.groupby("BatterId", sort=False)
        parte["_so_prior"] = grupo_prior["_so_cum"].shift(1)
        parte["_bb_prior"] = grupo_prior["_bb_cum"].shift(1)
        parte["_pa_prior"] = grupo_prior["_pa_cum"].shift(1)
        agg = parte.groupby(["GameId", "BatterId"], as_index=False).agg(
            so_p=("_so_prior", "last"),
            bb_p=("_bb_prior", "last"),
            pa_p=("_pa_prior", "last"))
        agg[f"bat_k_vs{tag}"] = agg["so_p"] / agg["pa_p"].replace(0.0, np.nan)
        agg[f"bat_bb_vs{tag}"] = agg["bb_p"] / agg["pa_p"].replace(0.0, np.nan)
        agg[f"bat_pa_vs{tag}"] = agg["pa_p"]
        resultado = resultado.merge(
            agg[["GameId", "BatterId", f"bat_k_vs{tag}", f"bat_bb_vs{tag}",
                 f"bat_pa_vs{tag}"]],
            on=["GameId", "BatterId"], how="left")
    return resultado


def construir_features(bateadores, abridores, manos):
    """DataFrame largo (una fila por aparicion) con features y targets."""
    df = bateadores.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Temporada"] = df["Fecha"].dt.year
    df["PA"] = pd.to_numeric(df["PlateAppearances"], errors="coerce")
    df["SO"] = pd.to_numeric(df["StrikeOuts"], errors="coerce").fillna(0.0)
    df["BB"] = pd.to_numeric(df["BaseOnBalls"], errors="coerce").fillna(0.0)
    df = df[df["PA"] > 0].copy()

    df["OppStarterId"] = pd.to_numeric(
        df["OpposingPitcherId"], errors="coerce")

    # Mano del abridor rival (contexto platoon).
    manos_unicas = manos.drop_duplicates("PitcherId")
    mapa_mano = manos_unicas.set_index("PitcherId")["Mano"].astype(str).str.upper()
    df["ManoOpp"] = df["OppStarterId"].map(mapa_mano).fillna("")
    df["EsLHP"] = (df["ManoOpp"] == "L").astype(float)
    df["EsRHP"] = (df["ManoOpp"] == "R").astype(float)
    df["EsLocal"] = pd.to_numeric(df["IsHome"], errors="coerce").astype(float)
    df["Mes"] = df["Fecha"].dt.month.astype(float)
    df["DiaTemporada"] = (
        df["Fecha"] - pd.to_datetime(df["Temporada"].astype(str) + "-03-01")).dt.days
    df["Orden_Al_Bate"] = pd.to_numeric(df["BattingOrder"], errors="coerce")

    # ---- Bateador: tasas de temporada y 30 dias (solo juegos previos) ----
    tasas_bateador = _tasas_previas(
        df, "BatterId", "SO", "BB", "PA", "bat")
    df = df.merge(tasas_bateador, on=["GameId", "BatterId"], how="left")

    # ---- Abridor rival: tasas K%/BB% de temporada y 30 dias ----
    abridores = abridores.copy()
    abridores["Fecha"] = pd.to_datetime(abridores["Fecha"])
    abridores["SO"] = pd.to_numeric(
        abridores["StrikeOuts"], errors="coerce").fillna(0.0)
    abridores["BB"] = pd.to_numeric(
        abridores["BaseOnBalls"], errors="coerce").fillna(0.0)
    abridores["BF"] = pd.to_numeric(
        abridores["BattersFaced"], errors="coerce").fillna(BF_FALLBACK)
    tasas_abridor = _tasas_previas(
        abridores, "PitcherId", "SO", "BB", "BF", "pit")
    df = df.merge(tasas_abridor,
                  left_on=["GameId", "OppStarterId"],
                  right_on=["GameId", "PitcherId"],
                  how="left")
    df = df.drop(columns=["PitcherId"], errors="ignore")

    # ---- Splits de platoon del bateador (acumulado vs cada mano) ----
    platoon = _splits_platoon(df, "SO", "BB", "PA")
    df = df.merge(platoon, on=["GameId", "BatterId"], how="left")

    # ---- Cruzado: rendimiento del bateador VS la mano del abridor rival ----
    df["bat_k_vs_mano"] = np.where(
        df["EsLHP"] == 1, df["bat_k_vsL"],
        np.where(df["EsRHP"] == 1, df["bat_k_vsR"], np.nan))
    df["bat_bb_vs_mano"] = np.where(
        df["EsLHP"] == 1, df["bat_bb_vsL"],
        np.where(df["EsRHP"] == 1, df["bat_bb_vsR"], np.nan))
    df["bat_pa_vs_mano"] = np.where(
        df["EsLHP"] == 1, df["bat_pa_vsL"],
        np.where(df["EsRHP"] == 1, df["bat_pa_vsR"], np.nan))

    # ---- Targets binarios: batio la linea 0.5? ----
    df["target_strikeout"] = (df["SO"] >= 1).astype(int)
    df["target_walk"] = (df["BB"] >= 1).astype(int)

    columnas_mantenidas = (
        ["Fecha", "GameId", "BatterId", "Temporada", "TargetsMetaPlaceholder"]
        + FEATURES + TARGETS)
    columnas_mantenidas.remove("TargetsMetaPlaceholder")
    df = df[[c for c in columnas_mantenidas if c in df.columns]]
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=r"..\..\..\data\mlb.db")
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()

    bateadores, abridores, manos = leer_datos(args.db)
    print(f"BatterGameLog: {len(bateadores)} filas | "
          f"abridores: {len(abridores)} | PitcherMano: {len(manos)}")
    feats = construir_features(bateadores, abridores, manos)
    print(f"Features: {len(feats)} filas x {len(FEATURES)} columnas.")
    print("Tasa base (SO>=1): {:.4f} | Tasa base (BB>=1): {:.4f}".format(
        feats["target_strikeout"].mean(), feats["target_walk"].mean()))
    feats.to_csv(args.salida, index=False)
    print(f"Guardado en {args.salida}")


if __name__ == "__main__":
    raise SystemExit(main())