using System.Text.Json;

namespace PruebaPronosticos.ETL;

public record ClimaInfo(double TemperaturaC, double? VientoVelocidadKmh, double? VientoDireccionGrados);

public class WeatherService
{
    private const double TemperaturaPorDefecto = 20.0;

    private readonly HttpClient _http;
    private readonly string _baseUrlArchivo;
    private readonly string _baseUrlPronostico;

    public WeatherService(HttpClient httpClient, string baseUrlArchivo, string baseUrlPronostico)
    {
        _http = httpClient;
        _baseUrlArchivo = baseUrlArchivo.TrimEnd('/');
        _baseUrlPronostico = baseUrlPronostico.TrimEnd('/');
    }

    public async Task<ClimaInfo> ObtenerClimaAsync(
        string nombreEstadio, DateTime fecha, bool lanzarEnError = false)
    {
        try
        {
            if (!EstadioCatalog.Estadios.TryGetValue(nombreEstadio, out var coords))
            {
                Console.WriteLine($"[CLIMA] Estadio sin coordenadas registradas: {nombreEstadio}. Usando {TemperaturaPorDefecto}°C.");
                return new ClimaInfo(TemperaturaPorDefecto, null, null);
            }

            string fechaStr = fecha.ToString("yyyy-MM-dd");
            bool esHoyOFutura = fecha.Date >= DateTime.Today;
            string variables = "temperature_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant";

            string url = esHoyOFutura
                ? $"{_baseUrlPronostico}?latitude={coords.Lat}&longitude={coords.Lon}&daily={variables}&timezone=auto&start_date={fechaStr}&end_date={fechaStr}"
                : $"{_baseUrlArchivo}?latitude={coords.Lat}&longitude={coords.Lon}&daily={variables}&timezone=auto&start_date={fechaStr}&end_date={fechaStr}";

            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);

            if (!doc.RootElement.TryGetProperty("daily", out var diario)
                || !diario.TryGetProperty("temperature_2m_mean", out var temperaturas)
                || temperaturas.GetArrayLength() == 0
                || temperaturas[0].ValueKind != JsonValueKind.Number)
            {
                Console.WriteLine($"[CLIMA] Respuesta sin datos para {nombreEstadio} ({fechaStr}). Usando {TemperaturaPorDefecto}°C.");
                return new ClimaInfo(TemperaturaPorDefecto, null, null);
            }

            double temperatura = temperaturas[0].GetDouble();
            double? vientoVelocidad = ExtraerValorDiario(diario, "wind_speed_10m_max");
            double? vientoDireccion = ExtraerValorDiario(diario, "wind_direction_10m_dominant");

            Console.WriteLine($"[CLIMA] {nombreEstadio} ({fechaStr}): {temperatura:F1}°C, viento {vientoVelocidad?.ToString("F1") ?? "-"} km/h ({vientoDireccion?.ToString("F0") ?? "ND"}°) ({(esHoyOFutura ? "pronostico" : "historico")})");
            return new ClimaInfo(temperatura, vientoVelocidad, vientoDireccion);
        }
        catch (Exception ex)
        {
            if (lanzarEnError)
                throw;
            Console.WriteLine($"[CLIMA] Error obteniendo clima de {nombreEstadio} ({fecha:yyyy-MM-dd}): {ex.Message}. Usando {TemperaturaPorDefecto}°C.");
            return new ClimaInfo(TemperaturaPorDefecto, null, null);
        }
    }

    private static double? ExtraerValorDiario(JsonElement diario, string nombreVariable)
    {
        if (diario.TryGetProperty(nombreVariable, out var valores)
            && valores.GetArrayLength() > 0
            && valores[0].ValueKind == JsonValueKind.Number)
        {
            return valores[0].GetDouble();
        }
        return null;
    }
}
