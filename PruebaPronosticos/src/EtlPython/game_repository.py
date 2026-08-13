"""Repositorio SQLite (port de GameRepository.cs).
Las fechas se guardan en el mismo formato del exportador:
  - date -> 'YYYY-MM-DD'
  - datetime -> 'YYYY-MM-DD HH:MM:SS.fff'
"""
import sqlite3
from datetime import datetime, timezone

_FORMAT_DT = lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def _ahora_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class GameRepository:
    def __init__(self, ruta_db):
        self._ruta = str(ruta_db)
        self._conexion = None

    def _con(self):
        if self._conexion is None:
            self._conexion = sqlite3.connect(self._ruta, timeout=60)
        return self._conexion

    def cerrar(self):
        if self._conexion is not None:
            self._conexion.close()
            self._conexion = None

    def insertar_game_logs(self, partidos):
        if not partidos:
            return 0
        con = self._con()
        insertados = 0
        try:
            for p in partidos:
                fila = (
                    str(p["Fecha"]), p["Estadio"], p["EquipoLocal"], p["EquipoVisita"],
                    p["PitcherLocalId"], p["PitcherVisitaId"],
                    p["CarrerasLocal"], p["CarrerasVisita"],
                    1 if p["EsFinal"] else 0,
                    _FORMAT_DT(p["HoraInicioUtc"]) if p["HoraInicioUtc"] else None,
                    p.get("TemperaturaC"), p.get("VientoVelocidad"),
                    p.get("VientoDireccion"),
                    p.get("EraBullpenLocal"), p.get("EraBullpenVisita"),
                    p.get("WhipAbridorLocal"), p.get("WhipAbridorVisita"),
                    p.get("UmpireNombre"), p.get("UmpireHomePlate"),
                )
                existe = con.execute(
                    "SELECT 1 FROM GameLog WHERE Fecha = ? AND EquipoLocal = ? "
                    "AND EquipoVisita = ?",
                    (str(p["Fecha"]), p["EquipoLocal"], p["EquipoVisita"])
                ).fetchone()
                if existe:
                    con.execute(
                        """UPDATE GameLog SET
                            CarrerasLocal = ?, CarrerasVisita = ?, EsFinal = ?,
                            HoraInicioUtc = COALESCE(?, HoraInicioUtc),
                            TemperaturaC = COALESCE(?, TemperaturaC),
                            Viento_Velocidad = COALESCE(?, Viento_Velocidad),
                            Viento_Direccion = COALESCE(?, Viento_Direccion),
                            ERA_Bullpen_Local = COALESCE(?, ERA_Bullpen_Local),
                            ERA_Bullpen_Visita = COALESCE(?, ERA_Bullpen_Visita),
                            WHIP_Abridor_Local = COALESCE(?, WHIP_Abridor_Local),
                            WHIP_Abridor_Visita = COALESCE(?, WHIP_Abridor_Visita),
                            UmpireNombre = COALESCE(?, UmpireNombre),
                            UmpireHomePlate = COALESCE(?, UmpireHomePlate)
                            WHERE Fecha = ? AND EquipoLocal = ? AND EquipoVisita = ?""",
                        (fila[6], fila[7], fila[8], fila[9], fila[10], fila[11],
                         fila[12], fila[13], fila[14], fila[15], fila[16], fila[17],
                         fila[18], str(p["Fecha"]), p["EquipoLocal"], p["EquipoVisita"]))
                else:
                    con.execute(
                        """INSERT INTO GameLog (Fecha, Estadio, EquipoLocal, EquipoVisita,
                            PitcherLocalId, PitcherVisitaId, CarrerasLocal, CarrerasVisita,
                            EsFinal, HoraInicioUtc, TemperaturaC, Viento_Velocidad,
                            Viento_Direccion, ERA_Bullpen_Local, ERA_Bullpen_Visita,
                            WHIP_Abridor_Local, WHIP_Abridor_Visita, UmpireNombre,
                            UmpireHomePlate)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        fila)
                insertados += 1
            con.commit()
        except Exception:
            con.rollback()
            raise
        return insertados

    def guardar_pitcher_mano(self, datos):
        lista = {d["PitcherId"]: d["Mano"] for d in datos}
        if not lista:
            return 0
        con = self._con()
        guardados = 0
        try:
            for pitcher_id, mano in lista.items():
                con.execute(
                    """INSERT INTO PitcherMano (PitcherId, Mano) VALUES (?, ?)
                       ON CONFLICT (PitcherId) DO UPDATE SET Mano = excluded.Mano""",
                    (pitcher_id, mano))
                guardados += 1
            con.commit()
        except Exception:
            con.rollback()
            raise
        return guardados

    def guardar_pitcher_game_logs(self, datos):
        if not datos:
            return 0
        con = self._con()
        guardados = 0
        try:
            for f in datos:
                con.execute(
                    """INSERT INTO PitcherGameLog (GameID, Fecha, Team, PitcherId,
                        IsStarter, PitchesThrown) VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT (GameID, Team, PitcherId)
                       DO UPDATE SET IsStarter = excluded.IsStarter,
                                     PitchesThrown = excluded.PitchesThrown""",
                    (f["GameId"], str(f["Fecha"]), f["Team"], f["PitcherId"],
                     f["IsStarter"], f["PitchesThrown"]))
                guardados += 1
            con.commit()
        except Exception:
            con.rollback()
            raise
        return guardados

    def guardar_ops_splits(self, splits):
        if not splits:
            return 0
        con = self._con()
        guardados = 0
        try:
            for s in splits:
                con.execute(
                    """INSERT INTO TeamOPS_Handedness (Equipo, Temporada, OPSvsLHP, OPSvsRHP)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT (Equipo, Temporada)
                       DO UPDATE SET OPSvsLHP = COALESCE(excluded.OPSvsLHP, OPSvsLHP),
                                     OPSvsRHP = COALESCE(excluded.OPSvsRHP, OPSvsRHP)""",
                    (s.Equipo, s.Temporada, s.OpsVsLhp, s.OpsVsRhp))
                guardados += 1
            con.commit()
        except Exception:
            con.rollback()
            raise
        return guardados

    def guardar_lineas_h2h(self, lineas):
        if not lineas:
            return 0
        con = self._con()
        guardados = 0
        try:
            for l in lineas:
                con.execute(
                    """INSERT INTO LineaSnapshotsML (EventoId, Casa, Fecha, EquipoLocal,
                        EquipoVisita, CuotaHome, CuotaAway, CapturadoUtc)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (l["EventoId"], l["Casa"], str(l["Fecha"]), l["EquipoLocal"],
                     l["EquipoVisita"], l["CuotaHome"], l["CuotaAway"],
                     _FORMAT_DT(_ahora_utc())))
                guardados += 1
            con.commit()
        except Exception:
            con.rollback()
            raise
        return guardados

    def guardar_lineas_historicas(self, lineas):
        if not lineas:
            return 0
        con = self._con()
        guardados = 0
        try:
            for l in lineas:
                con.execute(
                    """INSERT INTO HistoricalOdds (EventoId, Casa, Fecha, EquipoLocal,
                        EquipoVisita, CommenceTimeUtc, Linea, CuotaOver, CuotaUnder,
                        UltimaActualizacion) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (EventoId, Casa)
                       DO UPDATE SET Fecha = excluded.Fecha,
                                     EquipoLocal = excluded.EquipoLocal,
                                     EquipoVisita = excluded.EquipoVisita,
                                     CommenceTimeUtc = COALESCE(excluded.CommenceTimeUtc, CommenceTimeUtc),
                                     Linea = COALESCE(excluded.Linea, Linea),
                                     CuotaOver = COALESCE(excluded.CuotaOver, CuotaOver),
                                     CuotaUnder = COALESCE(excluded.CuotaUnder, CuotaUnder),
                                     UltimaActualizacion = COALESCE(excluded.UltimaActualizacion, UltimaActualizacion)""",
                    (l["EventoId"], l["Casa"], str(l["Fecha"]), l["EquipoLocal"],
                     l["EquipoVisita"],
                     _FORMAT_DT(l["CommenceTimeUtc"]) if l["CommenceTimeUtc"] else None,
                     l["Linea"], l["CuotaOver"], l["CuotaUnder"],
                     _FORMAT_DT(l["UltimaActualizacion"]) if l["UltimaActualizacion"] else None))
                con.execute(
                    """INSERT INTO LineaSnapshots (EventoId, Casa, Fecha, EquipoLocal,
                        EquipoVisita, Linea, CuotaOver, CuotaUnder, CapturadoUtc)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (l["EventoId"], l["Casa"], str(l["Fecha"]), l["EquipoLocal"],
                     l["EquipoVisita"], l["Linea"], l["CuotaOver"], l["CuotaUnder"],
                     _FORMAT_DT(_ahora_utc())))
                guardados += 1
            con.commit()
        except Exception:
            con.rollback()
            raise
        return guardados

    def resolver_lineas_reales(self, inicio, fin):
        """Linea de cierre por partido finalizado: ultimo snapshot por casa;
        linea = moda entre casas; cuotas = mediana de las casas con esa linea."""
        con = self._con()
        partidos = con.execute(
            """SELECT Fecha, EquipoLocal, EquipoVisita FROM GameLog
               WHERE Fecha BETWEEN ? AND ?
                 AND (CarrerasLocal > 0 OR CarrerasVisita > 0)""",
            (str(inicio), str(fin))).fetchall()
        if not partidos:
            return 0

        cotizaciones = con.execute(
            """SELECT EventoId, Casa, Fecha, EquipoLocal, EquipoVisita, Linea,
                      CuotaOver, CuotaUnder, UltimaActualizacion
               FROM HistoricalOdds WHERE Fecha BETWEEN ? AND ?""",
            (str(inicio), str(fin))).fetchall()

        cierres = {}
        por_grupo = {}
        for c in cotizaciones:
            clave = (c[2], c[3], c[4])
            por_grupo.setdefault(clave, []).append(c)

        for clave, grupo in por_grupo.items():
            ultimas_por_casa = []
            por_casa = {}
            for c in grupo:
                casa = c[1]
                if casa not in por_casa or (c[8] or "") > (por_casa[casa][8] or ""):
                    por_casa[casa] = c
            for c in por_casa.values():
                if (c[5] is not None and 6.0 <= c[5] <= 12.0
                        and c[6] is not None and 1.05 <= c[6] <= 5.0
                        and c[7] is not None and 1.05 <= c[7] <= 5.0):
                    ultimas_por_casa.append(c)
            if not ultimas_por_casa:
                continue

            por_linea = {}
            for c in ultimas_por_casa:
                por_linea.setdefault(c[5], []).append(c)
            linea_moda = sorted(por_linea.items(),
                                key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
            casas_con_linea = por_linea[linea_moda]
            cierres[clave] = (
                linea_moda,
                _mediana([c[6] for c in casas_con_linea]),
                _mediana([c[7] for c in casas_con_linea]),
            )

        actualizados = 0
        try:
            for p in partidos:
                if p not in cierres:
                    continue
                linea, over, under = cierres[p]
                cur = con.execute(
                    """UPDATE GameLog SET Linea_Casino_Real = ?, Cuota_Over_Real = ?,
                       Cuota_Under_Real = ?
                       WHERE Fecha = ? AND EquipoLocal = ? AND EquipoVisita = ?""",
                    (linea, over, under, p[0], p[1], p[2]))
                actualizados += cur.rowcount
            con.commit()
        except Exception:
            con.rollback()
            raise
        return actualizados


def _mediana(valores):
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    n = len(ordenados)
    if n % 2 == 1:
        return ordenados[n // 2]
    return (ordenados[n // 2 - 1] + ordenados[n // 2]) / 2.0