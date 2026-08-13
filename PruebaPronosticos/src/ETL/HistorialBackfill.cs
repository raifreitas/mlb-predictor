using System.Globalization;
using System.Text.Json;

namespace PruebaPronosticos.ETL;

/// <summary>
/// Carga historica 2015-2022 SIN look-ahead: procesa los juegos en orden
/// cronologico y acumula WHIP/ERA por lanzador/equipo desde los boxscores,
/// de modo que cada fila refleja las estadisticas DISPONIBLES AL DIA DEL
/// JUEGO (incluyendo el propio juego, igual que stats=season al dia de la
/// carga del ETL diario 2023-2026).
/// No usa el endpoint stats=season con temporada pasada (devolveria las
/// estadisticas FINALES de la temporada = leakage).
/// Reanudable: checkpoint JSON del estado acumulado por chunk de 30 dias.
/// </summary>
public class HistorialBackfill
{
    private const int DIAS_POR_CHUNK = 30;
    private const int CONCURRENCIA_BOXSCORE = 4;
    private const int CONCURRENCIA_CLIMA = 2;
    private const int MAX_REINTENTOS = 4;

    private readonly HttpClient _http;
    private readonly string _baseUrl;
    private readonly GameRepository _repositorio;
    private readonly WeatherService _clima;
    private readonly string _rutaCheckpoint;

    private readonly Dictionary<int, (double Ip, int H, int Bb, int Er)> _pitchers = new();
    private readonly Dictionary<string, (double Ip, int Er)> _equipos = new();
    private readonly Dictionary<int, string> _manos = new();

    public HistorialBackfill(HttpClient httpClient, string baseUrl,
        GameRepository repositorio, WeatherService clima, string rutaCheckpoint)
    {
        _http = httpClient;
        _baseUrl = baseUrl.TrimEnd('/');
        _repositorio = repositorio;
        _clima = clima;
        _rutaCheckpoint = rutaCheckpoint;
    }

    private class Checkpoint
    {
        public DateTime UltimaFecha { get; set; }
        public int Temporada { get; set; }
        public Dictionary<int, (double Ip, int H, int Bb, int Er)> Pitchers { get; set; } = new();
        public Dictionary<string, (double Ip, int Er)> Equipos { get; set; } = new();
    }

    public async Task<int> EjecutarAsync(DateTime inicio, DateTime fin)
    {
        var estado = CargarCheckpoint();
        DateTime actual = estado.UltimaFecha == default
            ? inicio.Date
            : estado.UltimaFecha.AddDays(1);
        if (estado.UltimaFecha != default)
            Console.WriteLine($"[HISTORIAL] Checkpoint encontrado: continua desde {actual:yyyy-MM-dd}.");

        int procesados = 0;
        int fallidos = 0;
        var semaforoBox = new SemaphoreSlim(CONCURRENCIA_BOXSCORE);
        var semaforoClima = new SemaphoreSlim(CONCURRENCIA_CLIMA);
        var temporadasConRoster = new HashSet<int>();
        int temporadaActual = actual.Year;
        if (estado.Temporada != temporadaActual)
        {
            // El estado acumulado corresponde a otra temporada: se reinicia
            // (stats=season del ETL diario solo cubre la temporada en curso).
            _pitchers.Clear();
            _equipos.Clear();
        }

        while (actual <= fin.Date)
        {
            DateTime finChunk = actual.AddDays(DIAS_POR_CHUNK - 1) > fin
                ? fin.Date : actual.AddDays(DIAS_POR_CHUNK - 1);
            Console.WriteLine($"[HISTORIAL] Chunk {actual:yyyy-MM-dd} .. {finChunk:yyyy-MM-dd}");

            var juegos = await ObtenerJuegosAsync(actual, finChunk);
            if (juegos.Count == 0)
            {
                Console.WriteLine("      (sin juegos finalizados en el chunk)");
            }
            else
            {
                foreach (var temporada in juegos.Select(j => j.Fecha.Year).Distinct())
                {
                    if (temporadasConRoster.Add(temporada))
                    {
                        int manos = await CargarManosRosterAsync(temporada);
                        Console.WriteLine($"      Roster {temporada}: {manos} manos de lanzador cargadas.");
                    }
                }

                // Boxscores + clima con concurrencia limitada y reintentos
                // (429/5xx); despues se aplican en orden de hora de inicio
                // para mantener la acumulacion correcta.
                var tareas = juegos.Select(async juego =>
                {
                    var boxscore = await ObtenerConReintentoAsync(
                        $"{_baseUrl}/game/{juego.GamePk ?? 0}/boxscore",
                        "boxscore",
                        j => ParsearBoxscore(j, juego.GamePk ?? 0),
                        semaforoBox);
                    var clima = await ObtenerClimaConReintentoAsync(juego, semaforoClima);
                    return (juego: juego, boxscore: boxscore, clima: clima);
                }).ToList();

                var resultados = await Task.WhenAll(tareas);
                var gameLogs = new List<GameLog>();
                var pitcherGameLogs = new List<PitcherGameLogRow>();
                foreach (var resultado in resultados.Where(r => r.boxscore is not null)
                             .OrderBy(r => r.juego.HoraInicioUtc ?? DateTime.MaxValue))
                {
                    var juego = resultado.juego;
                    if (juego.Fecha.Year != temporadaActual)
                    {
                        // Nueva temporada: stats de temporada se reinician.
                        _pitchers.Clear();
                        _equipos.Clear();
                        temporadaActual = juego.Fecha.Year;
                    }
                    var boxscore = resultado.boxscore;
                    var clima = resultado.clima;
                    if (boxscore is null || boxscore.Count == 0)
                    {
                        fallidos++;
                        continue;
                    }
                    try
                    {
                        gameLogs.Add(AplicarJuego(juego, boxscore, clima, pitcherGameLogs));
                        procesados++;
                    }
                    catch (Exception ex)
                    {
                        fallidos++;
                        Console.WriteLine($"[HISTORIAL] Error aplicando {juego.Fecha:yyyy-MM-dd} "
                                          + $"{juego.EquipoLocal}: {ex.Message}");
                    }
                }

                if (gameLogs.Count > 0)
                    await _repositorio.InsertarGameLogsAsync(gameLogs);
                if (pitcherGameLogs.Count > 0)
                    await _repositorio.GuardarPitcherGameLogsAsync(pitcherGameLogs);
            }

            estado.UltimaFecha = finChunk;
            estado.Temporada = temporadaActual;
            estado.Pitchers = new Dictionary<int, (double, int, int, int)>(_pitchers);
            estado.Equipos = new Dictionary<string, (double, int)>(_equipos);
            GuardarCheckpoint(estado);
            actual = finChunk.AddDays(1);
        }

        var manosFaltantes = _manos.Select(kv => (kv.Key, kv.Value)).ToList();
        await _repositorio.GuardarPitcherManoAsync(manosFaltantes);
        Console.WriteLine($"[HISTORIAL] Terminado: {procesados} juegos procesados, "
                          + $"{fallidos} fallidos, {_manos.Count} manos de lanzador en BD.");
        return procesados;
    }

    // ============ acumulacion ============

    private GameLog AplicarJuego(GameLog juego, List<PitcherBoxscore> boxscore,
        ClimaInfo clima, List<PitcherGameLogRow> pitcherGameLogs)
    {
        juego.TemperaturaC = clima.TemperaturaC;
        juego.VientoVelocidad = clima.VientoVelocidadKmh;
        juego.VientoDireccion = clima.VientoDireccionGrados?.ToString("F0") ?? "ND";

        // stats=season del ETL diario = solo temporada regular (gameType=R).
        // La fatiga (PitcherGameLog) se registra para TODOS los juegos (el
        // ETL diario tambien lo hace), pero WHIP/ERA solo acumulan en R: en
        // spring/postseason las filas quedan con WHIP nulo, como si el ETL
        // hubiera consultado stats=season ese dia (R-only).
        bool esTemporadaRegular = string.Equals(juego.GameType, "R",
            StringComparison.OrdinalIgnoreCase);

        foreach (var lado in new[] { "home", "away" })
        {
            var filas = boxscore.Where(b => b.Lado == lado).ToList();
            string nombreEquipo = filas.Count > 0 ? filas[0].NombreEquipo : "Desconocido";
            if (esTemporadaRegular)
            {
                foreach (var f in filas)
                {
                    var (ip, h, bb, er) = _pitchers.GetValueOrDefault(f.PitcherId);
                    _pitchers[f.PitcherId] = (ip + f.Ip, h + f.H, bb + f.Bb, er + f.Er);
                    var (ipE, erE) = _equipos.GetValueOrDefault(nombreEquipo);
                    _equipos[nombreEquipo] = (ipE + f.Ip, erE + f.Er);
                }
            }

            // Fatiga: filas PitcherGameLog (igual que el ETL diario, con fila
            // semilla si el lado no registro lanzadores).
            if (filas.Count == 0)
            {
                pitcherGameLogs.Add(new PitcherGameLogRow
                {
                    GameId = juego.GamePk!.Value, Fecha = juego.Fecha,
                    Team = nombreEquipo, PitcherId = 0,
                    IsStarter = false, PitchesThrown = 0
                });
                continue;
            }
            for (int i = 0; i < filas.Count; i++)
            {
                pitcherGameLogs.Add(new PitcherGameLogRow
                {
                    GameId = juego.GamePk!.Value, Fecha = juego.Fecha,
                    Team = nombreEquipo, PitcherId = filas[i].PitcherId,
                    IsStarter = i == 0, PitchesThrown = filas[i].PitchesThrown
                });
            }
        }

        if (!esTemporadaRegular)
            return juego;

        // WHIP/ERA del partido: se calculan CON el estado ya actualizado
        // (incluye el propio juego), igual que el ETL diario 2023-2026.
        var local = boxscore.FirstOrDefault(b => b.Lado == "home" && b.EsAbridor);
        var visita = boxscore.FirstOrDefault(b => b.Lado == "away" && b.EsAbridor);
        if (local is not null)
        {
            juego.PitcherLocalId = local.PitcherId;
            juego.WhipAbridorLocal = CalcularWhip(_pitchers.GetValueOrDefault(local.PitcherId));
        }
        if (visita is not null)
        {
            juego.PitcherVisitaId = visita.PitcherId;
            juego.WhipAbridorVisita = CalcularWhip(_pitchers.GetValueOrDefault(visita.PitcherId));
        }
        juego.EraBullpenLocal = CalcularEra(_equipos.GetValueOrDefault(juego.EquipoLocal));
        juego.EraBullpenVisita = CalcularEra(_equipos.GetValueOrDefault(juego.EquipoVisita));
        return juego;
    }

    private static double? CalcularWhip((double Ip, int H, int Bb, int Er) ac)
        => ac.Ip > 0 ? (ac.H + ac.Bb) / ac.Ip : (double?)null;

    private static double? CalcularEra((double Ip, int Er) ac)
        => ac.Ip > 0 ? 9.0 * ac.Er / ac.Ip : (double?)null;

    // ============ StatsAPI ============

    private async Task<List<GameLog>> ObtenerJuegosAsync(DateTime inicio, DateTime fin)
    {
        string url = $"{_baseUrl}/schedule?sportId=1&startDate={inicio:yyyy-MM-dd}"
                     + $"&endDate={fin:yyyy-MM-dd}&hydrate=venue,officials";
        string response = await _http.GetStringAsync(url);
        using var doc = JsonDocument.Parse(response);

        var juegos = new List<GameLog>();
        if (!doc.RootElement.TryGetProperty("dates", out var fechas))
            return juegos;

        foreach (var fecha in fechas.EnumerateArray())
        {
            if (!fecha.TryGetProperty("games", out var juegosJson))
                continue;
            foreach (var juego in juegosJson.EnumerateArray())
            {
                try
                {
                    // Un juego pospuesto por lluvia aparece en su fecha original
                    // con abstractGameState="Final" pero detailedState="Postponed"
                    // y sin score (officialDate = fecha reprogramada). Exigimos
                    // detailedState="Final" para no insertar filas fantasma 0-0.
                    var status = juego.GetProperty("status");
                    string estado = status.TryGetProperty("abstractGameState", out var e)
                        ? (e.GetString() ?? string.Empty) : string.Empty;
                    string detalle = status.TryGetProperty("detailedState", out var d)
                        ? (d.GetString() ?? string.Empty) : string.Empty;
                    if (!estado.Equals("Final", StringComparison.OrdinalIgnoreCase)
                        || !detalle.Equals("Final", StringComparison.OrdinalIgnoreCase))
                        continue;

                    var home = juego.GetProperty("teams").GetProperty("home");
                    var away = juego.GetProperty("teams").GetProperty("away");
                    string nombreLocal = home.GetProperty("team").GetProperty("name").GetString() ?? "Desconocido";
                    string nombreVisita = away.GetProperty("team").GetProperty("name").GetString() ?? "Desconocido";
                    long gamePk = juego.TryGetProperty("gamePk", out var gpk)
                                  && gpk.ValueKind == JsonValueKind.Number ? gpk.GetInt64() : 0;
                    if (gamePk == 0)
                        continue;

                    juegos.Add(new GameLog
                    {
                        Fecha = ExtraerFecha(juego),
                        Estadio = ExtraerEstadio(juego),
                        EquipoLocal = nombreLocal,
                        EquipoVisita = nombreVisita,
                        CarrerasLocal = home.TryGetProperty("score", out var hs) ? hs.GetInt32() : 0,
                        CarrerasVisita = away.TryGetProperty("score", out var asVar) ? asVar.GetInt32() : 0,
                        UmpireNombre = ExtraerUmpireHomePlate(juego),
                        UmpireHomePlate = ExtraerUmpireHomePlate(juego),
                        GamePk = gamePk,
                        EsFinal = true,
                        HoraInicioUtc = ExtraerHoraInicioUtc(juego),
                        GameType = juego.TryGetProperty("gameType", out var gt)
                                   ? gt.GetString() : null
                    });
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[HISTORIAL] Juego omitido por error de parseo: {ex.Message}");
                }
            }
        }
        return juegos;
    }

    private async Task<List<PitcherBoxscore>?> ObtenerConReintentoAsync(
        string url, string etiqueta, Func<JsonDocument, List<PitcherBoxscore>?> parsear,
        SemaphoreSlim semaforo)
    {
        for (int intento = 1; intento <= MAX_REINTENTOS; intento++)
        {
            await semaforo.WaitAsync();
            try
            {
                string response = await _http.GetStringAsync(url);
                using var doc = JsonDocument.Parse(response);
                return parsear(doc);
            }
            catch (HttpRequestException ex)
            {
                if (intento < MAX_REINTENTOS)
                {
                    // Cualquier fallo transitorio (DNS, 502/5xx, 429, timeout)
                    // se reintenta con backoff: la red nocturna es inestable y
                    // un solo fallo no debe descartar el juego.
                    int espera = (int)Math.Pow(2, intento) * 5;
                    Console.WriteLine($"[HISTORIAL] {etiqueta} fallo de red "
                                      + $"(intento {intento}/{MAX_REINTENTOS}): {ex.Message}");
                    Console.WriteLine($"            espera {espera}s.");
                    await Task.Delay(espera * 1000);
                    continue;
                }
                Console.WriteLine($"[HISTORIAL] Error {etiqueta} tras {MAX_REINTENTOS} "
                                  + $"intentos: {ex.Message}");
                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[HISTORIAL] Error parseando {etiqueta}: {ex.Message}");
                return null;
            }
            finally
            {
                semaforo.Release();
            }
        }
        return null;
    }

    private async Task<ClimaInfo> ObtenerClimaConReintentoAsync(
        GameLog juego, SemaphoreSlim semaforo)
    {
        for (int intento = 1; intento <= MAX_REINTENTOS; intento++)
        {
            await semaforo.WaitAsync();
            try
            {
                return await _clima.ObtenerClimaAsync(juego.Estadio, juego.Fecha,
                                                      lanzarEnError: true);
            }
            catch (Exception ex)
            {
                if (intento < MAX_REINTENTOS)
                {
                    // Cualquier fallo transitorio (DNS, 502/5xx, 429, timeout)
                    // se reintenta con backoff (el clima NO debe perderse).
                    int espera = (int)Math.Pow(2, intento) * 5;
                    Console.WriteLine($"[HISTORIAL] Clima fallo transitorio "
                                      + $"(intento {intento}/{MAX_REINTENTOS}) "
                                      + $"{juego.Estadio}: {ex.Message}");
                    Console.WriteLine($"            espera {espera}s.");
                    await Task.Delay(espera * 1000);
                    continue;
                }
                Console.WriteLine($"[HISTORIAL] Error clima {juego.Estadio} tras "
                                  + $"{MAX_REINTENTOS} intentos: {ex.Message}");
                return new ClimaInfo(20.0, null, null);
            }
            finally
            {
                semaforo.Release();
            }
        }
        return new ClimaInfo(20.0, null, null);
    }

    private static List<PitcherBoxscore>? ParsearBoxscore(
        JsonDocument doc, long gamePk)
    {
        var filas = new List<PitcherBoxscore>();
        if (!doc.RootElement.TryGetProperty("teams", out var equipos))
            return null;

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

            for (int indice = 0; indice < lanzadores.Count; indice++)
            {
                int pitcherId = lanzadores[indice];
                double ip = 0;
                int h = 0, bb = 0, er = 0, pitcheos = 0;
                if (ladoJson.TryGetProperty("players", out var jugadores)
                    && jugadores.TryGetProperty($"ID{pitcherId}", out var jugador)
                    && jugador.TryGetProperty("stats", out var stats)
                    && stats.TryGetProperty("pitching", out var pitching))
                {
                    ip = ExtraerInnings(pitching, "inningsPitched");
                    h = ExtraerEntero(pitching, "hits");
                    bb = ExtraerEntero(pitching, "baseOnBalls");
                    er = ExtraerEntero(pitching, "earnedRuns");
                    pitcheos = ExtraerEntero(pitching, "numberOfPitches");
                }
                filas.Add(new PitcherBoxscore
                {
                    Lado = lado,
                    NombreEquipo = nombreEquipo,
                    PitcherId = pitcherId,
                    EsAbridor = indice == 0,
                    Ip = ip,
                    H = h,
                    Bb = bb,
                    Er = er,
                    PitchesThrown = pitcheos
                });
            }
        }
        return filas;
    }

    /// <summary>Innings en formato MLB "5.1" (5 y 1/3) o "6.0".</summary>
    private static double ExtraerInnings(JsonElement pitching, string propiedad)
    {
        if (!pitching.TryGetProperty(propiedad, out var valor) || valor.ValueKind != JsonValueKind.String)
            return 0;
        string texto = valor.GetString() ?? "0";
        int pos = texto.IndexOf('.');
        if (pos < 0)
            return double.TryParse(texto, NumberStyles.Float, CultureInfo.InvariantCulture, out var entero)
                ? entero : 0;
        if (!double.TryParse(texto.Substring(0, pos), NumberStyles.Float, CultureInfo.InvariantCulture,
                out var completo))
            return 0;
        string parte = texto.Substring(pos + 1);
        int tercios = int.TryParse(parte, out var t) ? t : 0;
        return tercios > 0 ? completo + (double)tercios / 3.0 : completo;
    }

    private static int ExtraerEntero(JsonElement pitching, string propiedad)
    {
        if (!pitching.TryGetProperty(propiedad, out var valor))
            return 0;
        if (valor.ValueKind == JsonValueKind.Number)
            return valor.GetInt32();
        if (valor.ValueKind == JsonValueKind.String
            && int.TryParse(valor.GetString(), out var n))
            return n;
        return 0;
    }

    private async Task<int> CargarManosRosterAsync(int temporada)
    {
        int contador = 0;
        foreach (var equipoId in await ObtenerEquiposTemporadaAsync(temporada))
        {
            try
            {
                string url = $"{_baseUrl}/teams/{equipoId}/roster?season={temporada}&hydrate=person";
                string response = await _http.GetStringAsync(url);
                using var doc = JsonDocument.Parse(response);
                if (!doc.RootElement.TryGetProperty("roster", out var roster))
                    continue;
                foreach (var miembro in roster.EnumerateArray())
                {
                    if (!miembro.TryGetProperty("person", out var persona)
                        || !persona.TryGetProperty("pitchHand", out var manoJson)
                        || !manoJson.TryGetProperty("code", out var codigo))
                        continue;
                    if (!persona.TryGetProperty("id", out var idJson))
                        continue;
                    string? mano = codigo.GetString();
                    if (string.IsNullOrWhiteSpace(mano))
                        continue;
                    int pitcherId = idJson.GetInt32();
                    if (!_manos.ContainsKey(pitcherId))
                    {
                        _manos[pitcherId] = mano;
                        contador++;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[HISTORIAL] Roster equipo {equipoId} ({temporada}): {ex.Message}");
            }
        }
        return contador;
    }

    private async Task<List<int>> ObtenerEquiposTemporadaAsync(int temporada)
    {
        var ids = new List<int>();
        try
        {
            string url = $"{_baseUrl}/teams?sportId=1&season={temporada}";
            string response = await _http.GetStringAsync(url);
            using var doc = JsonDocument.Parse(response);
            if (doc.RootElement.TryGetProperty("teams", out var equipos))
            {
                foreach (var equipo in equipos.EnumerateArray())
                {
                    if (equipo.TryGetProperty("id", out var id))
                        ids.Add(id.GetInt32());
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HISTORIAL] Equipos {temporada}: {ex.Message}");
        }
        return ids;
    }

    private static DateTime ExtraerFecha(JsonElement juego)
    {
        string fechaStr = juego.GetProperty("officialDate").GetString() ?? string.Empty;
        if (DateTime.TryParseExact(fechaStr, "yyyy-MM-dd", CultureInfo.InvariantCulture,
                DateTimeStyles.None, out var fecha))
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

    // ============ checkpoint ============

    private Checkpoint CargarCheckpoint()
    {
        if (!File.Exists(_rutaCheckpoint))
            return new Checkpoint();
        try
        {
            string json = File.ReadAllText(_rutaCheckpoint);
            var estado = JsonSerializer.Deserialize<Checkpoint>(json)
                         ?? new Checkpoint();
            _pitchers.Clear();
            _equipos.Clear();
            foreach (var kv in estado.Pitchers)
                _pitchers[kv.Key] = kv.Value;
            foreach (var kv in estado.Equipos)
                _equipos[kv.Key] = kv.Value;
            return estado;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HISTORIAL] Checkpoint ilegible ({ex.Message}); "
                              + "se reinicia el proceso.");
            return new Checkpoint();
        }
    }

    private void GuardarCheckpoint(Checkpoint estado)
    {
        try
        {
            string json = JsonSerializer.Serialize(estado);
            File.WriteAllText(_rutaCheckpoint, json);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HISTORIAL] No se pudo guardar checkpoint: {ex.Message}");
        }
    }

    private record PitcherBoxscore
    {
        public string Lado { get; init; } = "";
        public string NombreEquipo { get; init; } = "";
        public int PitcherId { get; init; }
        public bool EsAbridor { get; init; }
        public double Ip { get; init; }
        public int H { get; init; }
        public int Bb { get; init; }
        public int Er { get; init; }
        public int PitchesThrown { get; init; }
    }
}
