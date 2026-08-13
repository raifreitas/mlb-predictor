"""Cliente de Open-Meteo (port de WeatherService.cs)."""
import json
import urllib.parse
import urllib.request
from datetime import date

TEMPERATURA_POR_DEFECTO = 20.0

from estadio_catalog import obtener_coordenadas


class WeatherService:
    def __init__(self, base_url_archivo, base_url_pronostico, timeout=100):
        self._base_url_archivo = base_url_archivo.rstrip("/")
        self._base_url_pronostico = base_url_pronostico.rstrip("/")
        self._timeout = timeout

    def obtener_clima(self, nombre_estadio, fecha, lanzar_en_error=False):
        try:
            lat, lon, _ = obtener_coordenadas(nombre_estadio)
            if lat is None:
                print(f"[CLIMA] Estadio sin coordenadas registradas: {nombre_estadio}. "
                      f"Usando {TEMPERATURA_POR_DEFECTO}°C.")
                return TEMPERATURA_POR_DEFECTO, None, None

            fecha_str = fecha.strftime("%Y-%m-%d")
            es_hoy_o_futura = fecha >= date.today()
            variables = "temperature_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant"
            if es_hoy_o_futura:
                url = (f"{self._base_url_pronostico}?latitude={lat}&longitude={lon}"
                       f"&daily={variables}&timezone=auto"
                       f"&start_date={fecha_str}&end_date={fecha_str}")
            else:
                url = (f"{self._base_url_archivo}?latitude={lat}&longitude={lon}"
                       f"&daily={variables}&timezone=auto"
                       f"&start_date={fecha_str}&end_date={fecha_str}")

            with urllib.request.urlopen(url, timeout=self._timeout) as r:
                datos = json.loads(r.read().decode("utf-8"))

            diario = datos.get("daily", {})
            temperaturas = diario.get("temperature_2m_mean", [])
            if not temperaturas or temperaturas[0] is None:
                print(f"[CLIMA] Respuesta sin datos para {nombre_estadio} ({fecha_str}). "
                      f"Usando {TEMPERATURA_POR_DEFECTO}°C.")
                return TEMPERATURA_POR_DEFECTO, None, None

            temperatura = float(temperaturas[0])
            viento_velocidad = _extraer_valor_diario(diario, "wind_speed_10m_max")
            viento_direccion = _extraer_valor_diario(diario, "wind_direction_10m_dominant")
            print(f"[CLIMA] {nombre_estadio} ({fecha_str}): {temperatura:.1f}°C, "
                  f"viento {viento_velocidad if viento_velocidad is None else round(viento_velocidad, 1)} km/h "
                  f"({viento_direccion if viento_direccion is None else round(viento_direccion)}°) "
                  f"({'pronostico' if es_hoy_o_futura else 'historico'})")
            return temperatura, viento_velocidad, viento_direccion
        except Exception as ex:
            if lanzar_en_error:
                raise
            print(f"[CLIMA] Error obteniendo clima de {nombre_estadio} "
                  f"({fecha.strftime('%Y-%m-%d')}): {ex}. "
                  f"Usando {TEMPERATURA_POR_DEFECTO}°C.")
            return TEMPERATURA_POR_DEFECTO, None, None


def _extraer_valor_diario(diario, nombre_variable):
    valores = diario.get(nombre_variable, [])
    if valores and valores[0] is not None:
        return float(valores[0])
    return None