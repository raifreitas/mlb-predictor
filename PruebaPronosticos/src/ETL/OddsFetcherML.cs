using System.Globalization;
using System.Text.Json;

namespace PruebaPronosticos.ETL;

/// <summary>
/// Captura de cuotas Moneyline (h2h) de The Odds API (endpoint gratuito).
/// Mercado AISLADO del flujo de Totals: ninguna clase ni tabla del flujo
/// Over/Under se modifica. Los snapshots h2h se acumulan dia a dia en
/// dbo.LineaSnapshotsML para poder medir CLV del mercado moneyline.
/// </summary>
public class OddsFetcherML
{
    private static readonly TimeZoneInfo ZonaEste =
        TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");

    private static readonly Dictionary<string, string> NombresNormalizados =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["L.A. Dodgers"] = "Los Angeles Dodgers",
            ["LA Dodgers"] = "Los Angeles Dodgers",
            ["Chi Cubs"] = "Chicago Cubs",
            ["Chi White Sox"] = "Chicago White Sox",
            ["CWS"] = "Chicago White Sox",
            ["NY Mets"] = "New York Mets",
            ["NY Yankees"] = "New York Yankees",
            ["Yankees"] = "New York Yankees",
            ["S.F. Giants"] = "San Francisco Giants",
            ["SF"] = "San Francisco Giants",
            ["S.D. Padres"] = "San Diego Padres",
            ["SD"] = "San Diego Padres",
            ["TB Rays"] = "Tampa Bay Rays",
            ["WSH"] = "Washington Nationals",
            ["LAA"] = "Los Angeles Angels",
            ["ARI"] = "Arizona Diamondbacks",
            ["ATL"] = "Atlanta Braves",
            ["BAL"] = "Baltimore Orioles",
            ["BOS"] = "Boston Red Sox",
            ["CIN"] = "Cincinnati Reds",
            ["CLE"] = "Cleveland Guardians",
            ["COL"] = "Colorado Rockies",
            ["DET"] = "Detroit Tigers",
            ["HOU"] = "Houston Astros",
            ["KC"] = "Kansas City Royals",
            ["MIL"] = "Milwaukee Brewers",
            ["MIN"] = "Minnesota Twins",
            ["NYM"] = "New York Mets",
            ["OAK"] = "Athletics",
            ["PHI"] = "Philadelphia Phillies",
            ["PIT"] = "Pittsburgh Pirates",
            ["SEA"] = "Seattle Mariners",
            ["STL"] = "St. Louis Cardinals",
            ["TEX"] = "Texas Rangers",
            ["TOR"] = "Toronto Blue Jays",
        };

    private readonly HttpClient _http;
    private readonly string _baseUrl;
    private readonly string _apiKey;

    public OddsFetcherML(HttpClient httpClient, string baseUrl, string apiKey)
    {
        _http = httpClient;
        _baseUrl = baseUrl.TrimEnd('/');
        _apiKey = apiKey;
    }

    /// <summary>
    /// Descarga las cuotas Moneyline (h2h) ACTUALES (endpoint gratuito) como
    /// snapshot. Devuelve una cotizacion por (evento, casa de apuestas).
    /// </summary>
    public async Task<List<LineaH2H>> ObtenerLineasH2HActualesAsync()
    {
        string url = $"{_baseUrl.Replace("/historical", "", StringComparison.OrdinalIgnoreCase)}" +
                     "/baseball_mlb/odds" +
                     $"?apiKey={Uri.EscapeDataString(_apiKey)}" +
                     "&regions=us&markets=h2h&oddsFormat=decimal";

        string response = await LeerRespuestaAsync(url);
        using var doc = JsonDocument.Parse(response);

        if (doc.RootElement.ValueKind != JsonValueKind.Array)
        {
            Console.WriteLine("[ODDS-ML] La respuesta actual no es un arreglo.");
            return new List<LineaH2H>();
        }

        var lineas = new List<LineaH2H>();
        foreach (var evento in doc.RootElement.EnumerateArray())
        {
            string eventoId = LeerTexto(evento, "id");
            if (string.IsNullOrWhiteSpace(eventoId))
                continue;

            DateTime? commenceUtc = null;
            if (evento.TryGetProperty("commence_time", out var commenceJson)
                && DateTimeOffset.TryParse(commenceJson.GetString(),
                    CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal,
                    out var commence))
            {
                commenceUtc = commence.UtcDateTime;
            }

            string home = NormalizarNombre(LeerTexto(evento, "home_team"));
            string away = NormalizarNombre(LeerTexto(evento, "away_team"));
            if (string.IsNullOrWhiteSpace(home) || string.IsNullOrWhiteSpace(away))
                continue;

            // Fecha LOCAL del partido (zona Este) para emparejar con dbo.GameLog.
            DateTime fechaLocal = commenceUtc is not null
                ? TimeZoneInfo.ConvertTimeFromUtc(commenceUtc.Value, ZonaEste).Date
                : DateTime.Today;

            if (!evento.TryGetProperty("bookmakers", out var casas))
                continue;

            foreach (var casa in casas.EnumerateArray())
            {
                string casaKey = LeerTexto(casa, "key") ?? string.Empty;
                if (string.IsNullOrWhiteSpace(casaKey))
                    continue;

                DateTime? ultimaActualizacion = LeerFecha(casa, "last_update");

                if (!casa.TryGetProperty("markets", out var mercados))
                    continue;

                foreach (var mercado in mercados.EnumerateArray())
                {
                    string keyMercado = LeerTexto(mercado, "key");
                    if (!keyMercado.Equals("h2h", StringComparison.OrdinalIgnoreCase))
                        continue;

                    ultimaActualizacion ??= LeerFecha(mercado, "last_update");

                    if (!mercado.TryGetProperty("outcomes", out var resultados))
                        continue;

                    decimal? cuotaHome = null, cuotaAway = null;
                    foreach (var resultado in resultados.EnumerateArray())
                    {
                        string nombre = LeerTexto(resultado, "name");
                        decimal? precio = LeerDecimal(resultado, "price");
                        if (nombre is null || precio is null)
                            continue;

                        if (string.Equals(nombre, home, StringComparison.OrdinalIgnoreCase))
                            cuotaHome = precio;
                        else if (string.Equals(nombre, away, StringComparison.OrdinalIgnoreCase))
                            cuotaAway = precio;
                    }

                    if (cuotaHome is null || cuotaAway is null)
                        continue;

                    lineas.Add(new LineaH2H(
                        eventoId, casaKey, fechaLocal, home, away, commenceUtc,
                        cuotaHome.Value, cuotaAway.Value, ultimaActualizacion));
                }
            }
        }

        return lineas;
    }

    private async Task<string> LeerRespuestaAsync(string url)
    {
        using var respuesta = await _http.GetAsync(url);
        string cuerpo = await respuesta.Content.ReadAsStringAsync();
        if (!respuesta.IsSuccessStatusCode)
        {
            string detalle = string.Empty;
            try
            {
                using var doc = JsonDocument.Parse(cuerpo);
                if (doc.RootElement.TryGetProperty("message", out var mensaje))
                    detalle = mensaje.GetString() ?? string.Empty;
            }
            catch (JsonException)
            {
                // cuerpo no es JSON; se reporta el estado HTTP.
            }
            throw new HttpRequestException(
                $"The Odds API (h2h): HTTP {(int)respuesta.StatusCode} "
                + $"{(string.IsNullOrWhiteSpace(detalle) ? respuesta.StatusCode : detalle)}");
        }
        return cuerpo;
    }

    private static string NormalizarNombre(string? nombre)
    {
        if (string.IsNullOrWhiteSpace(nombre))
            return string.Empty;
        string limpio = nombre.Trim();
        return NombresNormalizados.TryGetValue(limpio, out var normalizado)
            ? normalizado
            : limpio;
    }

    private static string? LeerTexto(JsonElement elemento, string nombre)
    {
        if (elemento.TryGetProperty(nombre, out var valor)
            && valor.ValueKind == JsonValueKind.String)
            return valor.GetString();
        return null;
    }

    private static DateTime? LeerFecha(JsonElement elemento, string nombre)
    {
        string? texto = LeerTexto(elemento, nombre);
        if (texto is not null
            && DateTimeOffset.TryParse(texto, CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal, out var fecha))
            return fecha.UtcDateTime;
        if (elemento.TryGetProperty(nombre, out var valor)
            && valor.ValueKind == JsonValueKind.Number)
            return DateTimeOffset.FromUnixTimeSeconds(valor.GetInt64()).UtcDateTime;
        return null;
    }

    private static decimal? LeerDecimal(JsonElement elemento, string nombre)
    {
        if (elemento.TryGetProperty(nombre, out var valor))
        {
            if (valor.ValueKind == JsonValueKind.Number)
                return valor.GetDecimal();
            if (valor.ValueKind == JsonValueKind.String
                && decimal.TryParse(valor.GetString(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out var resultado))
                return resultado;
        }
        return null;
    }

    public record LineaH2H(
        string EventoId,
        string Casa,
        DateTime Fecha,
        string EquipoLocal,
        string EquipoVisita,
        DateTime? CommenceTimeUtc,
        decimal CuotaHome,
        decimal CuotaAway,
        DateTime? UltimaActualizacion);
}
