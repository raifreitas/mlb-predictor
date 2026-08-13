using System.Globalization;
using System.Text.Json;

namespace PruebaPronosticos.ETL;

public class MlbDataFetcher
{
    private const double EraPorDefecto = 4.00;

    private readonly HttpClient _http;
    private readonly string _baseUrl;
    private readonly Dictionary<string, double?> _cacheEra = new();
    private readonly Dictionary<string, double?> _cacheWhip = new();
    private readonly Dictionary<int, string?> _cacheMano = new();
    private readonly Dictionary<string, TeamOpsSplits> _splitsPorEquipo = new();

    public MlbDataFetcher(HttpClient httpClient, string baseUrl)
    {
        _http = httpClient;
        _baseUrl = baseUrl.TrimEnd('/');
    }

    public IReadOnlyCollection<TeamOpsSplits> SplitsObtenidos => _splitsPorEquipo.Values;

    public async Task<List<GameLog>> ObtenerPartidosAsync(DateTime inicio, DateTime fin)
    {
        var partidos = new List<GameLog>();

        string url = $"{_baseUrl}/schedule?sportId=1&startDate={inicio:yyyy-MM-dd}&endDate={fin:yyyy-MM-dd}&hydrate=team,probablePitcher,venue,officials";
        string response = await _http.GetStringAsync(url);
        using var doc = JsonDocument.Parse(response);

        if (!doc.RootElement.TryGetProperty("dates", out var fechas))
        {
            Console.WriteLine("[MLB] La respuesta del schedule no contiene fechas.");
            return partidos;
        }

        foreach (var fecha in fechas.EnumerateArray())
        {
            if (!fecha.TryGetProperty("games", out var juegos))
                continue;

            foreach (var juego in juegos.EnumerateArray())
            {
                try
                {
                    var home = juego.GetProperty("teams").GetProperty("home");
                    var away = juego.GetProperty("teams").GetProperty("away");

                    string nombreLocal = home.GetProperty("team").GetProperty("name").GetString() ?? "Desconocido";
                    string nombreVisita = away.GetProperty("team").GetProperty("name").GetString() ?? "Desconocido";
                    var status = juego.GetProperty("status");
                    string estado = status.TryGetProperty("abstractGameState", out var abs)
                                   ? (abs.GetString() ?? string.Empty) : string.Empty;
                    // Un juego pospuesto por lluvia aparece con abstractGameState
                    // "Final" pero detailedState "Postponed" y sin score: no debe
                    // tratarse como finalizado (se insertaria 0-0 EsFinal=1).
                    // Tras el ultimo out la API puede reportar detailedState
                    // "Game Over" (abstractGameState "Final"): tambien es final.
                    string detalle = status.TryGetProperty("detailedState", out var det)
                                     ? (det.GetString() ?? string.Empty) : string.Empty;
                    bool esFinal = estado.Equals("Final", StringComparison.OrdinalIgnoreCase)
                                   && !detalle.Equals("Postponed", StringComparison.OrdinalIgnoreCase)
                                   && !detalle.Equals("Cancelled", StringComparison.OrdinalIgnoreCase);
                    bool esLive = estado.Equals("Live", StringComparison.OrdinalIgnoreCase);
                    bool esProgramado = estado.Equals("Preview", StringComparison.OrdinalIgnoreCase)
                                        || estado.Equals("Scheduled", StringComparison.OrdinalIgnoreCase);
                    if (!esFinal && !esLive && !esProgramado)
                        continue;

                    var log = new GameLog
                    {
                        Fecha = ExtraerFecha(juego),
                        Estadio = ExtraerEstadio(juego),
                        EquipoLocal = nombreLocal,
                        EquipoVisita = nombreVisita,
                        PitcherLocalId = ExtraerPitcherId(home),
                        PitcherVisitaId = ExtraerPitcherId(away),
                        // Solo se persiste el marcador de partidos FINALIZADOS.
                        // Si un partido esta en curso (Live) se guarda 0-0 para
                        // que la verificacion de predicciones nunca use un
                        // marcador parcial como si fuera el resultado final.
                        CarrerasLocal = esFinal && home.TryGetProperty("score", out var hs) ? hs.GetInt32() : 0,
                        CarrerasVisita = esFinal && away.TryGetProperty("score", out var asLocal) ? asLocal.GetInt32() : 0,
                        UmpireNombre = ExtraerUmpireHomePlate(juego),
                        UmpireHomePlate = ExtraerUmpireHomePlate(juego),
                        GamePk = juego.TryGetProperty("gamePk", out var gamePkVar)
                                 && gamePkVar.ValueKind == JsonValueKind.Number
                                    ? gamePkVar.GetInt64()
                                    : (long?)null,
                        EsFinal = esFinal,
                        HoraInicioUtc = ExtraerHoraInicioUtc(juego)
                    };

                    log.EraBullpenLocal = await ObtenerEraBullpenAsync(ExtraerTeamId(home), log.Fecha.Year);
                    log.EraBullpenVisita = await ObtenerEraBullpenAsync(ExtraerTeamId(away), log.Fecha.Year);
                    log.WhipAbridorLocal = await ObtenerWhipAbridorAsync(log.PitcherLocalId, log.Fecha.Year);
                    log.WhipAbridorVisita = await ObtenerWhipAbridorAsync(log.PitcherVisitaId, log.Fecha.Year);

                    if ((log.PitcherLocalId is null || log.PitcherVisitaId is null) && (esFinal || esLive))
                    {
                        long gamePk = juego.GetProperty("gamePk").GetInt64();
                        log.PitcherLocalId ??= await ObtenerAbridorAsync(gamePk, "home");
                        log.PitcherVisitaId ??= await ObtenerAbridorAsync(gamePk, "away");
                    }

                    int temporada = log.Fecha.Year;
                    foreach (var lado in new[]
                             {
                                 (Id: ExtraerTeamId(home), Nombre: nombreLocal),
                                 (Id: ExtraerTeamId(away), Nombre: nombreVisita)
                             })
                    {
                        if (lado.Id is null)
                            continue;
                        string claveSplits = $"{lado.Id}-{temporada}";
                        if (!_splitsPorEquipo.ContainsKey(claveSplits))
                            _splitsPorEquipo[claveSplits] =
                                await ObtenerOpsSplitsAsync(lado.Id, lado.Nombre, temporada);
                    }

                    partidos.Add(log);
                    Console.WriteLine($"[MLB] {log.Fecha:yyyy-MM-dd} [{estado}] {log.EquipoLocal} {log.CarrerasLocal}-{log.CarrerasVisita} {log.EquipoVisita} en {log.Estadio}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MLB] Partido omitido por error de parseo: {ex.Message}");
                }
            }
        }

        return partidos;
    }

    private static DateTime ExtraerFecha(JsonElement juego)
    {
        string fechaStr = juego.GetProperty("officialDate").GetString() ?? string.Empty;
        if (DateTime.TryParseExact(fechaStr, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var fecha))
            return fecha.Date;
        return DateTime.MinValue;
    }

    private static DateTime? ExtraerHoraInicioUtc(JsonElement juego)
    {
        if (!juego.TryGetProperty("gameDate", out var gameDateVar))
            return null;
        string fechaStr = gameDateVar.GetString() ?? string.Empty;
        if (DateTimeOffset.TryParse(fechaStr, CultureInfo.InvariantCulture,
                                    DateTimeStyles.None, out var inicio))
            return inicio.ToUniversalTime().UtcDateTime;
        return null;
    }

    private static string ExtraerEstadio(JsonElement juego)
    {
        if (juego.TryGetProperty("venue", out var venue) && venue.TryGetProperty("name", out var nombre))
            return nombre.GetString() ?? "Desconocido";
        return "Desconocido";
    }

    private static int? ExtraerPitcherId(JsonElement lado)
    {
        if (lado.TryGetProperty("probablePitcher", out var pitcher) && pitcher.TryGetProperty("id", out var id))
            return id.GetInt32();
        return null;
    }

    private static int? ExtraerTeamId(JsonElement lado)
    {
        if (lado.TryGetProperty("team", out var equipo) && equipo.TryGetProperty("id", out var id))
            return id.GetInt32();
        return null;
    }

    private static string ExtraerUmpireHomePlate(JsonElement juego)
    {
        if (juego.TryGetProperty("officials", out var oficiales)
            && oficiales.ValueKind == JsonValueKind.Array)
        {
            foreach (var oficial in oficiales.EnumerateArray())
            {
                if (oficial.TryGetProperty("officialType", out var tipo)
                    && (tipo.GetString()?.Equals("Home Plate", StringComparison.OrdinalIgnoreCase) == true
                        || tipo.GetString()?.Equals("HP", StringComparison.OrdinalIgnoreCase) == true)
                    && oficial.TryGetProperty("official", out var detalle)
                    && detalle.TryGetProperty("fullName", out var nombre))
                {
                    string? valor = nombre.GetString();
                    if (!string.IsNullOrWhiteSpace(valor))
                        return valor;
                }
            }
        }
        return "Desconocido";
    }

    public async Task<string?> ObtenerManoLanzamientoAsync(int? pitcherId)
    {
        if (pitcherId is null)
            return null;

        if (_cacheMano.TryGetValue(pitcherId.Value, out var manoCacheada))
            return manoCacheada;

        try
        {
            string url = $"{_baseUrl}/people/{pitcherId}";
            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);

            string? mano = null;
            if (TryGetPropiedad(doc.RootElement, "people", out var personas)
                && personas.ValueKind == JsonValueKind.Array
                && personas.GetArrayLength() > 0
                && personas[0].TryGetProperty("pitchHand", out var manoJson)
                && manoJson.TryGetProperty("code", out var codigo))
            {
                mano = codigo.GetString();
            }

            _cacheMano[pitcherId.Value] = mano;
            return mano;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MLB] Error obteniendo mano del abridor {pitcherId}: {ex.Message}");
            _cacheMano[pitcherId.Value] = null;
            return null;
        }
    }

    public async Task<TeamOpsSplits> ObtenerOpsSplitsAsync(int? teamId, string nombreEquipo, int temporada)
    {
        if (teamId is null)
            return new TeamOpsSplits(nombreEquipo, temporada, null, null);

        try
        {
            string url = $"{_baseUrl}/teams/{teamId}/stats?stats=statSplits&group=hitting&season={temporada}&sitCodes=vl,vr&sportId=1";
            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);

            double? opsVsLhp = null;
            double? opsVsRhp = null;
            if (TryGetPropiedad(doc.RootElement, "stats", out var stats)
                && stats.ValueKind == JsonValueKind.Array
                && stats.GetArrayLength() > 0
                && TryGetPropiedad(stats[0], "splits", out var splits)
                && splits.ValueKind == JsonValueKind.Array)
            {
                foreach (var split in splits.EnumerateArray())
                {
                    string? nombre = null;
                    if (split.TryGetProperty("split", out var sp))
                    {
                        // La API devuelve un objeto {code, description} en
                        // lugar de un string: se usa "description".
                        if (sp.ValueKind == JsonValueKind.String)
                            nombre = sp.GetString();
                        else if (sp.ValueKind == JsonValueKind.Object
                                 && sp.TryGetProperty("description", out var desc)
                                 && desc.ValueKind == JsonValueKind.String)
                            nombre = desc.GetString();
                    }
                    if (nombre is null)
                        continue;

                    double? ops = null;
                    if (TryGetPropiedad(split, "stat", out var stat)
                        && TryGetPropiedad(stat, "ops", out var opsJson))
                    {
                        if (opsJson.ValueKind == JsonValueKind.Number)
                            ops = opsJson.GetDouble();
                        else if (opsJson.ValueKind == JsonValueKind.String
                                 && double.TryParse(opsJson.GetString(), NumberStyles.Float,
                                     CultureInfo.InvariantCulture, out var opsParse))
                            ops = opsParse;
                    }

                    if (nombre.Contains("vs Left", StringComparison.OrdinalIgnoreCase)
                        || nombre.Contains("vs LHP", StringComparison.OrdinalIgnoreCase))
                        opsVsLhp = ops;
                    else if (nombre.Contains("vs Right", StringComparison.OrdinalIgnoreCase)
                             || nombre.Contains("vs RHP", StringComparison.OrdinalIgnoreCase))
                        opsVsRhp = ops;
                }
            }

            return new TeamOpsSplits(nombreEquipo, temporada, opsVsLhp, opsVsRhp);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MLB] Error obteniendo splits OPS del equipo {teamId} ({temporada}): {ex.Message}");
            return new TeamOpsSplits(nombreEquipo, temporada, null, null);
        }
    }

    public record TeamOpsSplits(string Equipo, int Temporada, double? OpsVsLhp, double? OpsVsRhp);

    private async Task<double?> ObtenerEraBullpenAsync(int? teamId, int temporada)
    {
        if (teamId is null)
        {
            Console.WriteLine($"[MLB] Equipo sin id disponible; ERA bullpen por defecto {EraPorDefecto:F2}.");
            return EraPorDefecto;
        }

        string clave = $"{teamId}-{temporada}";
        if (_cacheEra.TryGetValue(clave, out var eraCacheada))
        {
            Console.WriteLine($"[MLB] ERA bullpen (cache) equipo {teamId} {temporada}: {eraCacheada?.ToString("F2") ?? "-"}");
            return eraCacheada;
        }

        try
        {
            string url = $"{_baseUrl}/teams/{teamId}/stats?stats=season&group=pitching&season={temporada}";
            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);

            if (!TryGetPropiedad(doc.RootElement, "stats", out var stats)
                || stats.ValueKind != JsonValueKind.Array
                || stats.GetArrayLength() == 0
                || !TryGetPropiedad(stats[0], "splits", out var splits)
                || splits.ValueKind != JsonValueKind.Array
                || splits.GetArrayLength() == 0
                || !TryGetPropiedad(splits[0], "stat", out var stat)
                || !TryGetPropiedad(stat, "era", out var eraJson))
            {
                throw new InvalidOperationException(
                    $"La respuesta no contiene la ruta stats -> splits -> stat -> era. "
                    + "Puede que el equipo no tenga estadisticas de la temporada.");
            }

            double era;
            if (eraJson.ValueKind == JsonValueKind.String)
            {
                era = double.Parse(eraJson.GetString()!, CultureInfo.InvariantCulture);
            }
            else if (eraJson.ValueKind == JsonValueKind.Number)
            {
                era = eraJson.GetDouble();
            }
            else
            {
                throw new InvalidOperationException($"El campo 'era' tiene tipo inesperado: {eraJson.ValueKind}.");
            }

            Console.WriteLine($"[MLB] ERA bullpen (general) equipo {teamId} {temporada}: {era:F2}");
            _cacheEra[clave] = era;
            return era;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MLB] Error obteniendo ERA bullpen del equipo {teamId} ({temporada}): {ex.Message}. Usando {EraPorDefecto:F2}.");
            _cacheEra[clave] = EraPorDefecto;
            return EraPorDefecto;
        }
    }

    private async Task<double?> ObtenerWhipAbridorAsync(int? pitcherId, int temporada)
    {
        if (pitcherId is null)
        {
            Console.WriteLine("[MLB] Abridor sin id; WHIP nulo.");
            return null;
        }

        string clave = $"{pitcherId}-{temporada}";
        if (_cacheWhip.TryGetValue(clave, out var whipCacheado))
        {
            Console.WriteLine($"[MLB] WHIP abridor (cache) {pitcherId} {temporada}: {whipCacheado?.ToString("F2") ?? "-"}");
            return whipCacheado;
        }

        try
        {
            string url = $"{_baseUrl}/people/{pitcherId}/stats?stats=season&group=pitching&season={temporada}";
            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);

            if (!TryGetPropiedad(doc.RootElement, "stats", out var stats)
                || stats.ValueKind != JsonValueKind.Array
                || stats.GetArrayLength() == 0
                || !TryGetPropiedad(stats[0], "splits", out var splits)
                || splits.ValueKind != JsonValueKind.Array
                || splits.GetArrayLength() == 0
                || !TryGetPropiedad(splits[0], "stat", out var stat)
                || !TryGetPropiedad(stat, "whip", out var whipJson))
            {
                Console.WriteLine($"[MLB] Sin WHIP disponible para el abridor {pitcherId} ({temporada}).");
                _cacheWhip[clave] = null;
                return null;
            }

            double whip;
            if (whipJson.ValueKind == JsonValueKind.String)
            {
                whip = double.Parse(whipJson.GetString()!, CultureInfo.InvariantCulture);
            }
            else if (whipJson.ValueKind == JsonValueKind.Number)
            {
                whip = whipJson.GetDouble();
            }
            else
            {
                throw new InvalidOperationException($"El campo 'whip' tiene tipo inesperado: {whipJson.ValueKind}.");
            }

            Console.WriteLine($"[MLB] WHIP abridor {pitcherId} {temporada}: {whip:F2}");
            _cacheWhip[clave] = whip;
            return whip;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MLB] Error obteniendo WHIP del abridor {pitcherId} ({temporada}): {ex.Message}.");
            _cacheWhip[clave] = null;
            return null;
        }
    }

    private static bool TryGetPropiedad(JsonElement elemento, string nombre, out JsonElement valor)
    {
        if (elemento.ValueKind == JsonValueKind.Object)
        {
            foreach (var propiedad in elemento.EnumerateObject())
            {
                if (string.Equals(propiedad.Name, nombre, StringComparison.OrdinalIgnoreCase))
                {
                    valor = propiedad.Value;
                    return true;
                }
            }
        }
        valor = default;
        return false;
    }

    private async Task<int?> ObtenerAbridorAsync(long gamePk, string lado)
    {
        try
        {
            string url = $"{_baseUrl}/game/{gamePk}/boxscore";
            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);

            if (doc.RootElement.TryGetProperty("teams", out var equipos)
                && equipos.TryGetProperty(lado, out var ladoJson)
                && ladoJson.TryGetProperty("pitchers", out var lanzadores)
                && lanzadores.GetArrayLength() > 0)
            {
                return lanzadores[0].GetInt32();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MLB] No se pudo obtener el abridor de {lado} del partido {gamePk}: {ex.Message}");
        }
        return null;
    }

    /// <summary>
    /// Lee el boxscore del partido y extrae, por cada equipo, el listado de
    /// lanzadores en orden de aparicion (el primero es el abridor, IsStarter=1)
    /// con el numero de pitcheos (numberOfPitches) de cada uno.
    /// Si un equipo no tiene lanzadores registrados (partido sin relevistas),
    /// se devuelve una fila semilla (PitcherId=0, IsStarter=0, 0 pitcheos)
    /// para garantizar la serie diaria de la vista de fatiga.
    /// </summary>
    public async Task<List<PitcherGameLogRow>> ObtenerPitchersPartidoAsync(
        long gamePk, DateTime fecha)
    {
        var filas = new List<PitcherGameLogRow>();
        try
        {
            string url = $"{_baseUrl}/game/{gamePk}/boxscore";
            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);

            if (!doc.RootElement.TryGetProperty("teams", out var equipos))
            {
                Console.WriteLine($"[MLB] Boxscore {gamePk} sin seccion 'teams'.");
                return filas;
            }

            foreach (var lado in new[] { "home", "away" })
            {
                if (!equipos.TryGetProperty(lado, out var ladoJson))
                    continue;

                string nombreEquipo = "Desconocido";
                if (ladoJson.TryGetProperty("team", out var equipoJson)
                    && equipoJson.TryGetProperty("name", out var nombreJson))
                {
                    nombreEquipo = nombreJson.GetString() ?? "Desconocido";
                }

                var lanzadores = ladoJson.TryGetProperty("pitchers", out var pitchersJson)
                    ? pitchersJson.EnumerateArray().Select(p => p.GetInt32()).ToList()
                    : new List<int>();

                if (lanzadores.Count == 0)
                {
                    // Fila semilla: sin relevistas, el dia cuenta con 0 pitcheos.
                    filas.Add(new PitcherGameLogRow
                    {
                        GameId = gamePk,
                        Fecha = fecha.Date,
                        Team = nombreEquipo,
                        PitcherId = 0,
                        IsStarter = false,
                        PitchesThrown = 0
                    });
                    continue;
                }

                for (int indice = 0; indice < lanzadores.Count; indice++)
                {
                    int pitcherId = lanzadores[indice];
                    int pitcheos = 0;
                    if (ladoJson.TryGetProperty("players", out var jugadores)
                        && jugadores.TryGetProperty($"ID{pitcherId}", out var jugador)
                        && jugador.TryGetProperty("stats", out var stats)
                        && stats.TryGetProperty("pitching", out var pitching)
                        && pitching.TryGetProperty("numberOfPitches", out var nPitcheos))
                    {
                        if (nPitcheos.ValueKind == JsonValueKind.Number)
                            pitcheos = nPitcheos.GetInt32();
                        else if (nPitcheos.ValueKind == JsonValueKind.String
                                 && int.TryParse(nPitcheos.GetString(), out var nParse))
                            pitcheos = nParse;
                    }

                    filas.Add(new PitcherGameLogRow
                    {
                        GameId = gamePk,
                        Fecha = fecha.Date,
                        Team = nombreEquipo,
                        PitcherId = pitcherId,
                        IsStarter = indice == 0,
                        PitchesThrown = pitcheos
                    });
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MLB] Error leyendo boxscore del partido {gamePk}: {ex.Message}");
        }
        return filas;
    }
}
