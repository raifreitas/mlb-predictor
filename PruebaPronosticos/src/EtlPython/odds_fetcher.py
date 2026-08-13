"""Cliente de The Odds API (port de OddsFetcher.cs y OddsFetcherML.cs)."""
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ZONA_ESTE = ZoneInfo("America/New_York")

NOMBRES_NORMALIZADOS = {
    "L.A. Dodgers": "Los Angeles Dodgers",
    "LA Dodgers": "Los Angeles Dodgers",
    "Chi Cubs": "Chicago Cubs",
    "Chi White Sox": "Chicago White Sox",
    "CWS": "Chicago White Sox",
    "NY Mets": "New York Mets",
    "NY Yankees": "New York Yankees",
    "Yankees": "New York Yankees",
    "S.F. Giants": "San Francisco Giants",
    "SF": "San Francisco Giants",
    "S.D. Padres": "San Diego Padres",
    "SD": "San Diego Padres",
    "TB Rays": "Tampa Bay Rays",
    "WSH": "Washington Nationals",
    "LAA": "Los Angeles Angels",
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "OAK": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
}


class OddsFetcher:
    def __init__(self, base_url, api_key, timeout=100):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _leer_respuesta(self, url, tipo):
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            cuerpo = ""
            try:
                cuerpo = e.read().decode("utf-8")
                detalle = json.loads(cuerpo).get("message", "")
            except Exception:
                detalle = ""
            raise HTTPErrorOdds(
                f"The Odds API ({tipo}): HTTP {e.code} "
                f"{(detalle or e.code)}", e.code) from None

    def obtener_lineas_historicas(self, fecha, hora_snapshot_utc):
        fecha_snapshot = datetime.combine(fecha, hora_snapshot_utc, tzinfo=timezone.utc)
        url = (f"{self._base_url}/baseball_mlb/odds"
               f"?apiKey={urllib.parse.quote(self._api_key)}"
               "&regions=us&markets=totals&oddsFormat=decimal&dateFormat=iso"
               f"&date={fecha_snapshot:%Y-%m-%dT%H:%M:%SZ}")
        datos = self._leer_respuesta(url, "historico")
        eventos = datos.get("data", [])
        if not isinstance(eventos, list):
            print("[ODDS] La respuesta historica no contiene 'data'.")
            return []
        return _parsear_eventos_totals(eventos, fecha)

    def obtener_lineas_actuales(self):
        base = self._base_url.replace("/historical", "")
        url = (f"{base}/baseball_mlb/odds"
               f"?apiKey={urllib.parse.quote(self._api_key)}"
               "&regions=us&markets=totals&oddsFormat=decimal")
        eventos = self._leer_respuesta(url, "actual")
        if not isinstance(eventos, list):
            print("[ODDS] La respuesta actual no es un arreglo.")
            return []
        return _parsear_eventos_totals(eventos, None)


class OddsFetcherML:
    def __init__(self, base_url, api_key, timeout=100):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def obtener_lineas_h2h_actuales(self):
        base = self._base_url.replace("/historical", "")
        url = (f"{base}/baseball_mlb/odds"
               f"?apiKey={urllib.parse.quote(self._api_key)}"
               "&regions=us&markets=h2h&oddsFormat=decimal")
        try:
            eventos = _leer_respuesta_ml(url, self._timeout)
        except HTTPErrorOdds as ex:
            raise ex
        if not isinstance(eventos, list):
            print("[ODDS-ML] La respuesta actual no es un arreglo.")
            return []
        return _parsear_eventos_h2h(eventos)


class HTTPErrorOdds(Exception):
    def __init__(self, message, codigo=None):
        super().__init__(message)
        self.codigo = codigo


def _leer_respuesta_ml(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detalle = ""
        try:
            detalle = json.loads(e.read().decode("utf-8")).get("message", "")
        except Exception:
            pass
        raise HTTPErrorOdds(
            f"The Odds API (h2h): HTTP {e.code} {(detalle or e.code)}", e.code) from None


def _normalizar_nombre(nombre):
    if not nombre:
        return ""
    limpio = nombre.strip()
    return NOMBRES_NORMALIZADOS.get(limpio, limpio)


def _fecha_local_este(commence_utc, fecha_por_defecto=None):
    if commence_utc is not None:
        return commence_utc.astimezone(ZONA_ESTE).date()
    return fecha_por_defecto if fecha_por_defecto is not None else datetime.now().date()


def _leer_texto(elemento, nombre):
    valor = elemento.get(nombre)
    return valor if isinstance(valor, str) else None


def _leer_fecha(elemento, nombre):
    texto = _leer_texto(elemento, nombre)
    if texto:
        try:
            return datetime.fromisoformat(texto.replace("Z", "+00:00")) \
                .astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            pass
    valor = elemento.get(nombre)
    if isinstance(valor, (int, float)):
        return datetime.fromtimestamp(valor, tz=timezone.utc).replace(tzinfo=None)
    return None


def _leer_decimal(elemento, nombre):
    valor = elemento.get(nombre)
    if isinstance(valor, (int, float)):
        return float(valor)
    if isinstance(valor, str):
        try:
            return float(valor)
        except ValueError:
            return None
    return None


def _parsear_eventos_totals(eventos, fecha_por_defecto):
    lineas = []
    for evento in eventos:
        evento_id = _leer_texto(evento, "id")
        if not evento_id:
            continue

        commence_utc = _leer_fecha(evento, "commence_time")
        home = _normalizar_nombre(_leer_texto(evento, "home_team"))
        away = _normalizar_nombre(_leer_texto(evento, "away_team"))
        if not home or not away:
            continue

        fecha_local = _fecha_local_este(commence_utc, fecha_por_defecto)

        for casa in evento.get("bookmakers", []):
            casa_key = _leer_texto(casa, "key")
            if not casa_key:
                continue
            ultima_actualizacion = _leer_fecha(casa, "last_update")
            for mercado in casa.get("markets", []):
                if _leer_texto(mercado, "key").lower() != "totals":
                    continue
                ultima_actualizacion = ultima_actualizacion or _leer_fecha(mercado, "last_update")
                linea = cuota_over = cuota_under = None
                for resultado in mercado.get("outcomes", []):
                    nombre = _leer_texto(resultado, "name")
                    precio = _leer_decimal(resultado, "price")
                    if nombre and nombre.lower() == "over":
                        cuota_over = precio
                        punto = _leer_decimal(resultado, "point")
                        if punto is not None:
                            linea = punto
                    elif nombre and nombre.lower() == "under":
                        cuota_under = precio
                if linea is None or cuota_over is None or cuota_under is None:
                    continue
                lineas.append({
                    "EventoId": evento_id,
                    "Casa": casa_key,
                    "Fecha": fecha_local,
                    "EquipoLocal": home,
                    "EquipoVisita": away,
                    "CommenceTimeUtc": commence_utc,
                    "Linea": linea,
                    "CuotaOver": cuota_over,
                    "CuotaUnder": cuota_under,
                    "UltimaActualizacion": ultima_actualizacion,
                })
    return lineas


def _parsear_eventos_h2h(eventos):
    lineas = []
    for evento in eventos:
        evento_id = _leer_texto(evento, "id")
        if not evento_id:
            continue

        commence_utc = _leer_fecha(evento, "commence_time")
        home = _normalizar_nombre(_leer_texto(evento, "home_team"))
        away = _normalizar_nombre(_leer_texto(evento, "away_team"))
        if not home or not away:
            continue

        fecha_local = _fecha_local_este(commence_utc)

        for casa in evento.get("bookmakers", []):
            casa_key = _leer_texto(casa, "key")
            if not casa_key:
                continue
            ultima_actualizacion = _leer_fecha(casa, "last_update")
            for mercado in casa.get("markets", []):
                if _leer_texto(mercado, "key").lower() != "h2h":
                    continue
                ultima_actualizacion = ultima_actualizacion or _leer_fecha(mercado, "last_update")
                cuota_home = cuota_away = None
                for resultado in mercado.get("outcomes", []):
                    nombre = _leer_texto(resultado, "name")
                    precio = _leer_decimal(resultado, "price")
                    if nombre is None or precio is None:
                        continue
                    if nombre.lower() == home.lower():
                        cuota_home = precio
                    elif nombre.lower() == away.lower():
                        cuota_away = precio
                if cuota_home is None or cuota_away is None:
                    continue
                lineas.append({
                    "EventoId": evento_id,
                    "Casa": casa_key,
                    "Fecha": fecha_local,
                    "EquipoLocal": home,
                    "EquipoVisita": away,
                    "CommenceTimeUtc": commence_utc,
                    "CuotaHome": cuota_home,
                    "CuotaAway": cuota_away,
                    "UltimaActualizacion": ultima_actualizacion,
                })
    return lineas