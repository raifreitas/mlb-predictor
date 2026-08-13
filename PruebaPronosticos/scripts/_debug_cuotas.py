import sys, os
sys.path.insert(0, r"src/Modeling")
import recomendar_apuestas as r
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
import joblib, pandas as pd

fecha = date.today()
api_key = r.obtener_api_key()
lineas = r.obtener_lineas(api_key)
partidos_mlb = r.obtener_calendario(fecha)

df_raw = r.cargar_datos(solo_con_temperatura=False)
whips, eras, manos, parques, fatigas = r.datos_auxiliares(fecha)
df_hoy = r.construir_partidos_hoy(partidos_mlb, whips, eras, manos, parques, fatigas, fecha)
df_raw = pd.concat([df_raw, df_hoy], ignore_index=True)
df_raw["Fecha"] = pd.to_datetime(df_raw["Fecha"])
df_raw = df_raw.drop_duplicates(subset=["Fecha", "EquipoLocal", "EquipoVisita"], keep="last")
for col in ("TemperaturaC","Viento_Velocidad","CarrerasLocal","CarrerasVisita","WHIP_Abridor_Local","WHIP_Abridor_Visita","ERA_Bullpen_Local","ERA_Bullpen_Visita","Factor_Carreras","PitcherLocalId","PitcherVisitaId","Fatiga_Bullpen_3d_Local","Fatiga_Bullpen_3d_Visita"):
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
df_raw["Viento_Direccion"] = df_raw["Viento_Direccion"].fillna("ND")

decodificadores = r.construir_decodificadores(df_raw)
passthrough = df_raw[["Fecha","EquipoLocal","EquipoVisita"]].copy()
passthrough["Viento_Direccion"] = df_raw["Viento_Direccion"]
enc_local = LabelEncoder().fit(df_raw["EquipoLocal"])
enc_visita = LabelEncoder().fit(df_raw["EquipoVisita"])
passthrough["EquipoLocal"] = enc_local.transform(passthrough["EquipoLocal"])
passthrough["EquipoVisita"] = enc_visita.transform(passthrough["EquipoVisita"])

df = r.preprocesar(df_raw)
df = r.feature_engineering_rachas(df)
df = r.feature_engineering_fatiga(df)
df = r.feature_engineering_pitchers(df, df["CarrerasVisita"].median())
df = r.feature_engineering_bullpen(df)
df = r.feature_engineering_ampayer(df)
df = r.feature_engineering_descanso_abridor(df)
df = r.feature_engineering_matchup(df)
df["Total_Carreras"] = df["CarrerasLocal"] + df["CarrerasVisita"]

partidos = df[df["Fecha"].dt.date == fecha]
partidos = partidos.merge(passthrough, on=["Fecha","EquipoLocal","EquipoVisita"], how="left")
partidos = r.calcular_expected_runs(partidos, df)
print("partidos tras expected_runs:", len(partidos))
print("columnas:", list(partidos.columns)[:12])

linea_por_equipo_local_y_cuota = {par[0]: detalle for par, detalle in lineas.items()}
nombres_local = pd.Series(
    decodificadores["EquipoLocal"].inverse_transform(partidos["EquipoLocal"]),
    index=partidos.index)
print("Decodificados hoy:", sorted(nombres_local.unique())[:5], "...")
faltan = sorted(set(nombres_local.unique()) - set(linea_por_equipo_local_y_cuota.keys()))
print("Decodificados SIN clave en Odds:", faltan)
print("Claves Odds (home):", sorted(linea_por_equipo_local_y_cuota.keys())[:5], "...")