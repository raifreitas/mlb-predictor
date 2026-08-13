"""Configuracion del ETL Python: lee appsettings.json y/o variables de entorno."""
import json
import os
from pathlib import Path


def _raiz_proyecto():
    p = Path(__file__).resolve()
    for candidato in (p.parent, *p.parents):
        if (candidato / "data" / "mlb.db").is_file() or (
                candidato / "PruebaPronosticos.sln").exists():
            return candidato
    return p.parents[3]


RAIZ = _raiz_proyecto()
RUTA_DB = Path(os.environ.get("MLB_DB_PATH", RAIZ / "data" / "mlb.db"))


def cargar_config():
    config = {
        "MlbStatsApiBaseUrl": "https://statsapi.mlb.com/api/v1",
        "TheOddsHistoricaBaseUrl": "https://api.the-odds-api.com/v4/historical/sports",
        "TheOddsApiKey": os.environ.get("THE_ODDS_API_KEY", ""),
        "HoraSnapshotOddsUtc": "21:30:00",
        "OpenMeteoArchiveBaseUrl": "https://archive-api.open-meteo.com/v1/archive",
        "OpenMeteoForecastBaseUrl": "https://api.open-meteo.com/v1/forecast",
    }
    ruta = RAIZ / "config" / "appsettings.json"
    if not ruta.exists():
        ruta = RAIZ / "PruebaPronosticos" / "config" / "appsettings.json"
    if ruta.exists():
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            apis = datos.get("Apis", {})
            for clave in list(config):
                if clave in apis and apis[clave]:
                    config[clave] = apis[clave]
        except json.JSONDecodeError:
            pass
    if os.environ.get("THE_ODDS_API_KEY"):
        config["TheOddsApiKey"] = os.environ["THE_ODDS_API_KEY"]
    return config