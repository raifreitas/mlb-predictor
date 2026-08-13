"""Cliente de StatsAPI de MLB (port de MlbDataFetcher.cs)."""
import json
import urllib.parse
import urllib.request

ERA_POR_DEFECTO = 4.00


class TeamOpsSplits:
    def __init__(self, equipo, temporada, ops_vs_lhp, ops_vs_rhp):
        self.Equipo = equipo
        self.Temporada = temporada
        self.OpsVsLhp = ops_vs_lhp
        self.OpsVsRhp = ops_vs_rhp


class MlbDataFetcher:
    def __init__(self, base_url, timeout=100):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache_era = {}
        self._cache_whip = {}
        self._cache_mano = {}
        self._splits_por_equipo = {}
        self.splits_obtenidos = []

    def _get_json(self, url):
        with urllib.request.urlopen(url, timeout=self._timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def obtener_partidos(self, inicio, fin):
        partidos = []
        url = (
            f"{self._base_url}/schedule?sportId=1"
            f"&startDate={inicio:%Y-%m-%d}&endDate={fin:%Y-%m-%d}"
            "&hydrate=team,probablePitcher,venue,officials"
        )
        try:
            datos = self._get_json(url)
        except Exception as ex:
            print(f"[MLB] Error descargando schedule: {ex}")
            return partidos

        for fecha in datos.get("dates", []):
            for juego in fecha.get("games", []):
                try:
                    home = juego["teams"]["home"]
                    away = juego["teams"]["away"]
                    nombre_local = home["team"].get("name", "Desconocido")
                    nombre_visita = away["team"].get("name", "Desconocido")
                    estado = juego.get("status", {}).get("abstractGameState", "")
                    detalle = juego.get("status", {}).get("detailedState", "")
                    es_final = (
                        estado.lower() == "final"
                        and detalle.lower() not in ("postponed", "cancelled")
                    )
                    es_live = estado.lower() == "live"
                    es_programado = estado.lower() in ("preview", "scheduled")
                    if not (es_final or es_live or es_programado):
                        continue

                    log = {
                        "Fecha": self._extraer_fecha(juego),
                        "Estadio": self._extraer_estadio(juego),
                        "EquipoLocal": nombre_local,
                        "EquipoVisita": nombre_visita,
                        "PitcherLocalId": self._extraer_pitcher_id(home),
                        "PitcherVisitaId": self._extraer_pitcher_id(away),
                        "CarrerasLocal": (
                            int(home["score"]) if es_final and "score" in home else 0),
                        "CarrerasVisita": (
                            int(away["score"]) if es_final and "score" in away else 0),
                        "UmpireNombre": self._extraer_umpire_home_plate(juego),
                        "UmpireHomePlate": self._extraer_umpire_home_plate(juego),
                        "GamePk": juego.get("gamePk"),
                        "EsFinal": es_final,
                        "HoraInicioUtc": self._extraer_hora_inicio_utc(juego),
                    }

                    id_local = self._extraer_team_id(home)
                    id_visita = self._extraer_team_id(away)
                    log["EraBullpenLocal"] = self._obtener_era_bullpen(
                        id_local, log["Fecha"].year)
                    log["EraBullpenVisita"] = self._obtener_era_bullpen(
                        id_visita, log["Fecha"].year)
                    log["WhipAbridorLocal"] = self._obtener_whip_abridor(
                        log["PitcherLocalId"], log["Fecha"].year)
                    log["WhipAbridorVisita"] = self._obtener_whip_abridor(
                        log["PitcherVisitaId"], log["Fecha"].year)

                    if (log["PitcherLocalId"] is None or log["PitcherVisitaId"] is None) \
                            and (es_final or es_live):
                        game_pk = juego.get("gamePk")
                        if game_pk:
                            log["PitcherLocalId"] = log["PitcherLocalId"] or \
                                self._obtener_abridor(game_pk, "home")
                            log["PitcherVisitaId"] = log["PitcherVisitaId"] or \
                                self._obtener_abridor(game_pk, "away")

                    temporada = log["Fecha"].year
                    for lado_id, lado_nombre in ((id_local, nombre_local),
                                                 (id_visita, nombre_visita)):
                        if lado_id is None:
                            continue
                        clave = f"{lado_id}-{temporada}"
                        if clave not in self._splits_por_equipo:
                            self._splits_por_equipo[clave] = \
                                self.obtener_ops_splits(lado_id, lado_nombre, temporada)

                    partidos.append(log)
                    print(f"[MLB] {log['Fecha']:%Y-%m-%d} [{estado}] "
                          f"{log['EquipoLocal']} {log['CarrerasLocal']}-"
                          f"{log['CarrerasVisita']} {log['EquipoVisita']} "
                          f"en {log['Estadio']}")
                except Exception as ex:
                    print(f"[MLB] Partido omitido por error de parseo: {ex}")

        self.splits_obtenidos = list(self._splits_por_equipo.values())
        return partidos

    @staticmethod
    def _extraer_fecha(juego):
        from datetime import datetime
        try:
            return datetime.strptime(juego.get("officialDate", ""),
                                     "%Y-%m-%d").date()
        except ValueError:
            from datetime import date
            return date.min

    @staticmethod
    def _extraer_hora_inicio_utc(juego):
        from datetime import datetime
        texto = juego.get("gameDate")
        if not texto:
            return None
        try:
            return datetime.fromisoformat(texto.replace("Z", "+00:00")) \
                .astimezone(__import__("datetime").timezone.utc) \
                .replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _extraer_estadio(juego):
        try:
            return juego["venue"].get("name", "Desconocido")
        except (KeyError, TypeError):
            return "Desconocido"

    @staticmethod
    def _extraer_pitcher_id(lado):
        try:
            return int(lado["probablePitcher"]["id"])
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _extraer_team_id(lado):
        try:
            return int(lado["team"]["id"])
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _extraer_umpire_home_plate(juego):
        for oficial in juego.get("officials", []):
            tipo = oficial.get("officialType", "")
            if tipo.lower() not in ("home plate", "hp"):
                continue
            nombre = oficial.get("official", {}).get("fullName")
            if nombre:
                return nombre
        return "Desconocido"

    def obtener_mano_lanzamiento(self, pitcher_id):
        if pitcher_id is None:
            return None
        if pitcher_id in self._cache_mano:
            return self._cache_mano[pitcher_id]
        try:
            url = f"{self._base_url}/people/{pitcher_id}"
            datos = self._get_json(url)
            personas = datos.get("people", [])
            mano = None
            if personas and personas[0].get("pitchHand", {}).get("code"):
                mano = personas[0]["pitchHand"]["code"]
            self._cache_mano[pitcher_id] = mano
            return mano
        except Exception as ex:
            print(f"[MLB] Error obteniendo mano del abridor {pitcher_id}: {ex}")
            self._cache_mano[pitcher_id] = None
            return None

    def obtener_ops_splits(self, team_id, nombre_equipo, temporada):
        if team_id is None:
            return TeamOpsSplits(nombre_equipo, temporada, None, None)
        try:
            url = (f"{self._base_url}/teams/{team_id}/stats"
                   f"?stats=statSplits&group=hitting&season={temporada}"
                   "&sitCodes=vl,vr&sportId=1")
            datos = self._get_json(url)
            ops_vs_lhp = ops_vs_rhp = None
            stats = datos.get("stats", [])
            if stats and stats[0].get("splits"):
                for split in stats[0]["splits"]:
                    sp = split.get("split")
                    if isinstance(sp, dict):
                        nombre = sp.get("description")
                    else:
                        nombre = sp
                    if not nombre:
                        continue
                    stat = split.get("stat", {})
                    ops = stat.get("ops")
                    if ops is not None:
                        try:
                            ops = float(ops)
                        except (TypeError, ValueError):
                            ops = None
                    nombre_l = nombre.lower()
                    if "vs left" in nombre_l or "vs lhp" in nombre_l:
                        ops_vs_lhp = ops
                    elif "vs right" in nombre_l or "vs rhp" in nombre_l:
                        ops_vs_rhp = ops
            return TeamOpsSplits(nombre_equipo, temporada, ops_vs_lhp, ops_vs_rhp)
        except Exception as ex:
            print(f"[MLB] Error obteniendo splits OPS del equipo {team_id} "
                  f"({temporada}): {ex}")
            return TeamOpsSplits(nombre_equipo, temporada, None, None)

    def _obtener_era_bullpen(self, team_id, temporada):
        if team_id is None:
            print(f"[MLB] Equipo sin id disponible; ERA bullpen por defecto {ERA_POR_DEFECTO:.2f}.")
            return ERA_POR_DEFECTO
        clave = f"{team_id}-{temporada}"
        if clave in self._cache_era:
            print(f"[MLB] ERA bullpen (cache) equipo {team_id} {temporada}: "
                  f"{self._cache_era[clave]:.2f}")
            return self._cache_era[clave]
        try:
            url = (f"{self._base_url}/teams/{team_id}/stats"
                   f"?stats=season&group=pitching&season={temporada}")
            datos = self._get_json(url)
            stats = datos.get("stats", [])
            era = float(stats[0]["splits"][0]["stat"]["era"])
            print(f"[MLB] ERA bullpen (general) equipo {team_id} {temporada}: {era:.2f}")
            self._cache_era[clave] = era
            return era
        except Exception as ex:
            print(f"[MLB] Error obteniendo ERA bullpen del equipo {team_id} "
                  f"({temporada}): {ex}. Usando {ERA_POR_DEFECTO:.2f}.")
            self._cache_era[clave] = ERA_POR_DEFECTO
            return ERA_POR_DEFECTO

    def _obtener_whip_abridor(self, pitcher_id, temporada):
        if pitcher_id is None:
            print("[MLB] Abridor sin id; WHIP nulo.")
            return None
        clave = f"{pitcher_id}-{temporada}"
        if clave in self._cache_whip:
            print(f"[MLB] WHIP abridor (cache) {pitcher_id} {temporada}: "
                  f"{self._cache_whip[clave]}")
            return self._cache_whip[clave]
        try:
            url = (f"{self._base_url}/people/{pitcher_id}/stats"
                   f"?stats=season&group=pitching&season={temporada}")
            datos = self._get_json(url)
            stats = datos.get("stats", [])
            whip = float(stats[0]["splits"][0]["stat"]["whip"])
            print(f"[MLB] WHIP abridor {pitcher_id} {temporada}: {whip:.2f}")
            self._cache_whip[clave] = whip
            return whip
        except Exception:
            print(f"[MLB] Sin WHIP disponible para el abridor {pitcher_id} ({temporada}).")
            self._cache_whip[clave] = None
            return None

    def obtener_abridor(self, game_pk, lado):
        try:
            url = f"{self._base_url}/game/{game_pk}/boxscore"
            datos = self._get_json(url)
            lanzadores = datos.get("teams", {}).get(lado, {}).get("pitchers", [])
            return lanzadores[0] if lanzadores else None
        except Exception as ex:
            print(f"[MLB] No se pudo obtener el abridor de {lado} del partido "
                  f"{game_pk}: {ex}")
            return None

    def obtener_pitchers_partido(self, game_pk, fecha):
        filas = []
        try:
            url = f"{self._base_url}/game/{game_pk}/boxscore"
            datos = self._get_json(url)
            equipos = datos.get("teams")
            if not equipos:
                print(f"[MLB] Boxscore {game_pk} sin seccion 'teams'.")
                return filas

            for lado in ("home", "away"):
                lado_json = equipos.get(lado)
                if not lado_json:
                    continue
                nombre_equipo = lado_json.get("team", {}).get("name", "Desconocido")
                lanzadores = lado_json.get("pitchers", [])
                if not lanzadores:
                    filas.append({
                        "GameId": game_pk,
                        "Fecha": fecha,
                        "Team": nombre_equipo,
                        "PitcherId": 0,
                        "IsStarter": 0,
                        "PitchesThrown": 0,
                    })
                    continue
                for indice, pitcher_id in enumerate(lanzadores):
                    pitcheos = 0
                    jugador = lado_json.get("players", {}).get(f"ID{pitcher_id}", {})
                    pitching = jugador.get("stats", {}).get("pitching", {})
                    try:
                        pitcheos = int(pitching.get("numberOfPitches", 0))
                    except (TypeError, ValueError):
                        pitcheos = 0
                    filas.append({
                        "GameId": game_pk,
                        "Fecha": fecha,
                        "Team": nombre_equipo,
                        "PitcherId": pitcher_id,
                        "IsStarter": 1 if indice == 0 else 0,
                        "PitchesThrown": pitcheos,
                    })
        except Exception as ex:
            print(f"[MLB] Error leyendo boxscore del partido {game_pk}: {ex}")
        return filas