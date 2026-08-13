using System.Globalization;
using System.Text.Json;
using PruebaPronosticos.ETL;

try
{
    var config = CargarConfiguracion();
    (DateTime inicio, DateTime fin) = ObtenerRangoDeFechas(args);

    if (args.Contains("--historial"))
    {
        Console.WriteLine("==================================================");
        Console.WriteLine(" MOTOR ETL MLB - CARGA HISTORICA 2015-2022");
        Console.WriteLine(" (acumulacion leak-free: WHIP/ERA al dia del juego)");
        Console.WriteLine("==================================================");
        using var httpHist = new HttpClient { Timeout = TimeSpan.FromSeconds(100) };
        var repositorioHist = new GameRepository(config.ConnectionStrings.MlbHistorica);
        var climaHist = new WeatherService(
            httpHist, config.Apis.OpenMeteoArchiveBaseUrl,
            config.Apis.OpenMeteoForecastBaseUrl);
        string dirCheckpoint = Path.GetDirectoryName(ResolverRutaConfig()) ?? ".";
        string rutaCheckpoint = Path.Combine(dirCheckpoint, "historial_checkpoint.json");
        var historial = new HistorialBackfill(
            httpHist, config.Apis.MlbStatsApiBaseUrl,
            repositorioHist, climaHist, rutaCheckpoint);
        int juegos = await historial.EjecutarAsync(inicio, fin);
        Console.WriteLine($"[HISTORIAL] {juegos} juegos procesados en total.");
        Console.ReadLine();
        return 0;
    }

    if (args.Contains("--solo-odds"))
    {
        Console.WriteLine("==================================================");
        Console.WriteLine(" MOTOR ETL MLB - SOLO SNAPSHOT DE CUOTAS");
        Console.WriteLine("==================================================");
        using var httpOdds = new HttpClient { Timeout = TimeSpan.FromSeconds(100) };
        var repositorioOdds = new GameRepository(config.ConnectionStrings.MlbHistorica);

        if (string.IsNullOrWhiteSpace(config.Apis.TheOddsApiKey))
        {
            Console.WriteLine("[ODDS] AVISO: sin TheOddsApiKey en appsettings.json; "
                              + "no se descargo snapshot.");
            return 0;
        }

        var odds = new OddsFetcher(httpOdds, config.Apis.TheOddsHistoricaBaseUrl,
                                   config.Apis.TheOddsApiKey);
        try
        {
            var lineas = await odds.ObtenerLineasActualesAsync();
            int guardadas = await repositorioOdds.GuardarLineasHistoricasAsync(lineas);
            Console.WriteLine($"[ODDS] Snapshot Totals capturado: "
                              + $"{lineas.Count} cotizaciones (guardadas: {guardadas}).");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ODDS] AVISO (captura Totals): {ex.Message}");
        }

        try
        {
            var oddsMl = new OddsFetcherML(httpOdds, config.Apis.TheOddsHistoricaBaseUrl,
                                           config.Apis.TheOddsApiKey);
            var lineasH2H = await oddsMl.ObtenerLineasH2HActualesAsync();
            int h2hGuardados = await repositorioOdds.GuardarLineasH2HAsync(lineasH2H);
            Console.WriteLine($"[ODDS-ML] Snapshot Moneyline capturado: "
                              + $"{lineasH2H.Count} cotizaciones (guardadas: {h2hGuardados}).");
        }
        catch (Exception exMl)
        {
            Console.WriteLine($"[ODDS-ML] AVISO (captura h2h): {exMl.Message}");
        }

        try
        {
            int resueltos = await repositorioOdds.ResolverLineasRealesAsync(inicio, fin);
            Console.WriteLine($"[CARGA] {resueltos} partidos finalizados actualizados "
                              + "con Linea_Casino_Real, Cuota_Over_Real y "
                              + "Cuota_Under_Real.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ODDS] AVISO (resolver cierre): {ex.Message}");
        }

        Console.WriteLine("Snapshot de cuotas finalizado.");
        return 0;
    }

    Console.WriteLine("==================================================");
    Console.WriteLine(" MOTOR ETL MLB - CARGA HISTÓRICA");
    Console.WriteLine("==================================================");

    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(100) };

    var fetcher = new MlbDataFetcher(http, config.Apis.MlbStatsApiBaseUrl);
    Console.WriteLine($"[EXTRACCIÓN] Descargando partidos del {inicio:yyyy-MM-dd} al {fin:yyyy-MM-dd}...");
    var partidos = await fetcher.ObtenerPartidosAsync(inicio, fin);
    Console.WriteLine($"[EXTRACCIÓN] {partidos.Count} partidos finalizados obtenidos.");

    var clima = new WeatherService(http, config.Apis.OpenMeteoArchiveBaseUrl, config.Apis.OpenMeteoForecastBaseUrl);
    foreach (var partido in partidos)
    {
        var info = await clima.ObtenerClimaAsync(partido.Estadio, partido.Fecha);
        partido.TemperaturaC = info.TemperaturaC;
        partido.VientoVelocidad = info.VientoVelocidadKmh;
        partido.VientoDireccion = info.VientoDireccionGrados?.ToString("F0") ?? "ND";
    }
    Console.WriteLine($"[TRANSFORMACIÓN] Clima resuelto para {partidos.Count} partidos (temperatura, viento y dirección; fallback 20°C).");
    Console.WriteLine("[TRANSFORMACIÓN] WHIP de abridores resuelto por partido (filtro contra abridores volátiles).");

    var manos = new List<(int PitcherId, string Mano)>();
    foreach (var partido in partidos)
    {
        string? manoLocal = partido.PitcherLocalId is null
            ? null
            : await fetcher.ObtenerManoLanzamientoAsync(partido.PitcherLocalId);
        string? manoVisita = partido.PitcherVisitaId is null
            ? null
            : await fetcher.ObtenerManoLanzamientoAsync(partido.PitcherVisitaId);
        if (partido.PitcherLocalId is not null && !string.IsNullOrEmpty(manoLocal))
            manos.Add((partido.PitcherLocalId.Value, manoLocal));
        if (partido.PitcherVisitaId is not null && !string.IsNullOrEmpty(manoVisita))
            manos.Add((partido.PitcherVisitaId.Value, manoVisita));
    }
    Console.WriteLine($"[TRANSFORMACIÓN] Mano de lanzamiento resuelta para {manos.Count} abridores (LHP/RHP).");

    var repositorio = new GameRepository(config.ConnectionStrings.MlbHistorica);
    int insertados = await repositorio.InsertarGameLogsAsync(partidos);
    Console.WriteLine($"[CARGA] {insertados} registros insertados o actualizados en SQL Server.");

    // Ingesta de pitcheos por jugador (boxscore StatsAPI) -> dbo.PitcherGameLog.
    // Solo partidos FINALIZADOS (el boxscore ya es definitivo).
    var pitcheos = new List<PitcherGameLogRow>();
    foreach (var partido in partidos.Where(p => p.EsFinal && p.GamePk.HasValue))
    {
        var filas = await fetcher.ObtenerPitchersPartidoAsync(partido.GamePk!.Value, partido.Fecha);
        pitcheos.AddRange(filas);
    }
    Console.WriteLine($"[TRANSFORMACIÓN] {pitcheos.Count} registros de pitcheo por jugador "
                      + "extraidos de los boxscores (abridores y relevistas).");
    int pitcheosGuardados = await repositorio.GuardarPitcherGameLogsAsync(pitcheos);
    Console.WriteLine($"[CARGA] {pitcheosGuardados} pitcheos por jugador insertados o "
            + "actualizados en dbo.PitcherGameLog.");

    int manosGuardadas = await repositorio.GuardarPitcherManoAsync(manos);
    Console.WriteLine($"[CARGA] {manosGuardadas} registros de mano de lanzador insertados o actualizados en SQL Server.");

    var splits = fetcher.SplitsObtenidos.ToList();
    int splitsGuardados = await repositorio.GuardarOpsSplitsAsync(splits);
    Console.WriteLine($"[CARGA] {splitsGuardados} registros de OPS por mano (vs LHP/RHP) insertados o actualizados en SQL Server.");

    if (string.IsNullOrWhiteSpace(config.Apis.TheOddsApiKey))
    {
        Console.WriteLine("[ODDS] AVISO: sin TheOddsApiKey en appsettings.json; "
                          + "no se descargaron lineas reales de cierre.");
    }
    else
    {
        var horaSnapshotUtc = TimeSpan.TryParse(
            config.Apis.HoraSnapshotOddsUtc, CultureInfo.InvariantCulture, out var hora)
            ? hora
            : TimeSpan.FromHours(21.5);
        var odds = new OddsFetcher(http, config.Apis.TheOddsHistoricaBaseUrl, config.Apis.TheOddsApiKey);

        Console.WriteLine($"[ODDS] Descargando snapshots historicos de Totals "
                          + $"(snapshot diario {horaSnapshotUtc:hh\\:mm} UTC)...");
        int snapshotsGuardados = 0;
        bool historicoDisponible = true;
        for (var dia = inicio; dia <= fin; dia = dia.AddDays(1))
        {
            if (!historicoDisponible)
                break;

            try
            {
                var lineas = await odds.ObtenerLineasHistoricasAsync(dia, horaSnapshotUtc);
                int eventos = lineas.Select(l => l.EventoId).Distinct().Count();
                snapshotsGuardados += await repositorio.GuardarLineasHistoricasAsync(lineas);
                Console.WriteLine($"[ODDS] {dia:yyyy-MM-dd}: {eventos} partidos, "
                                  + $"{lineas.Count} cotizaciones (evento+casa).");
            }
            catch (HttpRequestException ex)
            {
                if (ex.Message.Contains("401", StringComparison.OrdinalIgnoreCase)
                    && historicoDisponible)
                {
                    historicoDisponible = false;
                    Console.WriteLine("[ODDS] AVISO: el plan de The Odds API no incluye "
                                      + "el endpoint historico (se requiere plan de pago "
                                      + "con History Lite/Pro).");
                    Console.WriteLine("[ODDS] Modo respaldo: se capturaran las lineas "
                                      + "ACTUALES como snapshot para acumular historial "
                                      + "dia a dia (gratuito).");
                    try
                    {
                        var lineas = await odds.ObtenerLineasActualesAsync();
                        snapshotsGuardados += await repositorio.GuardarLineasHistoricasAsync(lineas);
                        Console.WriteLine($"[ODDS] Snapshot actual capturado: "
                                          + $"{lineas.Count} cotizaciones.");
                    }
                    catch (Exception ex2)
                    {
                        Console.WriteLine($"[ODDS] AVISO (captura actual): {ex2.Message}");
                    }
                    continue;
                }
                Console.WriteLine($"[ODDS] AVISO ({dia:yyyy-MM-dd}): {ex.Message}");
            }
        }
        Console.WriteLine($"[ODDS] {snapshotsGuardados} cotizaciones historicas "
                          + "insertadas o actualizadas en SQL Server.");

        int resueltos = await repositorio.ResolverLineasRealesAsync(inicio, fin);
        Console.WriteLine($"[CARGA] {resueltos} partidos finalizados actualizados con "
                          + "Linea_Casino_Real, Cuota_Over_Real y Cuota_Under_Real.");
    }

    // ============ MERCADO MONEYLINE (aislado de Totals) ============
    // Captura las cuotas h2h actuales como snapshot diario en
    // dbo.LineaSnapshotsML. No modifica ninguna tabla ni archivo del
    // flujo Over/Under: la linea de cierre ML se acumula dia a dia.
    if (!string.IsNullOrWhiteSpace(config.Apis.TheOddsApiKey))
    {
        var oddsMl = new OddsFetcherML(
            http, config.Apis.TheOddsHistoricaBaseUrl, config.Apis.TheOddsApiKey);
        try
        {
            var lineasH2H = await oddsMl.ObtenerLineasH2HActualesAsync();
            int h2hGuardados = await repositorio.GuardarLineasH2HAsync(lineasH2H);
            Console.WriteLine($"[ODDS-ML] Snapshot Moneyline capturado: "
                              + $"{lineasH2H.Count} cotizaciones h2h "
                              + $"(guardadas: {h2hGuardados}).");
        }
        catch (Exception exMl)
        {
            Console.WriteLine($"[ODDS-ML] AVISO (captura h2h): {exMl.Message}");
        }
    }

    Console.WriteLine("==================================================");
    Console.WriteLine("Proceso ETL finalizado. Presiona Enter para salir...");
    Console.ReadLine();
    return 0;
}
catch (Exception ex)
{
    Console.WriteLine($"[ERROR GENERAL] {ex.Message}");
    if (ex.InnerException is not null)
        Console.WriteLine($"  Detalle: {ex.InnerException.Message}");
    Console.ReadLine();
    return 1;
}

static (DateTime Inicio, DateTime Fin) ObtenerRangoDeFechas(string[] args)
{
    DateTime inicio = DateTime.Today.AddDays(-1);
    DateTime fin = inicio;

    var fechas = args.Where(a => !a.StartsWith("--")).ToList();
    if (fechas.Count >= 2
        && DateTime.TryParseExact(fechas[0], "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var d1)
        && DateTime.TryParseExact(fechas[1], "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var d2))
    {
        return (d1, d2);
    }

    Console.WriteLine("[AVISO] Uso esperado: PruebaPronosticos aaaa-MM-dd aaaa-MM-dd. Se cargará el día de ayer.");
    return (inicio, fin);
}

static AppConfig CargarConfiguracion()
{
    string ruta = ResolverRutaConfig();
    string json = File.ReadAllText(ruta);
    var opciones = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
    return JsonSerializer.Deserialize<AppConfig>(json, opciones) ?? throw new InvalidOperationException("El archivo de configuración está vacío.");
}

static string ResolverRutaConfig()
{
    string ruta = Path.Combine(AppContext.BaseDirectory, "config", "appsettings.json");
    if (File.Exists(ruta))
        return ruta;

    var directorio = new DirectoryInfo(AppContext.BaseDirectory);
    for (int i = 0; i < 6 && directorio is not null; i++)
    {
        ruta = Path.Combine(directorio.FullName, "config", "appsettings.json");
        if (File.Exists(ruta))
            return ruta;
        directorio = directorio.Parent;
    }

    ruta = Path.Combine(Directory.GetCurrentDirectory(), "config", "appsettings.json");
    if (!File.Exists(ruta))
        throw new FileNotFoundException("No se encontró el archivo de configuración en config/appsettings.json", ruta);
    return ruta;
}

class AppConfig
{
    public SeccionConnectionStrings ConnectionStrings { get; set; } = new();
    public SeccionApis Apis { get; set; } = new();
}

class SeccionConnectionStrings
{
    public string MlbHistorica { get; set; } = string.Empty;
}

class SeccionApis
{
    public string MlbStatsApiBaseUrl { get; set; } = string.Empty;
    public string TheOddsApiUrl { get; set; } = string.Empty;
    public string TheOddsApiKey { get; set; } = string.Empty;
    public string TheOddsHistoricaBaseUrl { get; set; } = string.Empty;
    public string HoraSnapshotOddsUtc { get; set; } = "21:30:00";
    public string OpenMeteoArchiveBaseUrl { get; set; } = string.Empty;
    public string OpenMeteoForecastBaseUrl { get; set; } = string.Empty;
    public string OpenWeatherApiBaseUrl { get; set; } = string.Empty;
    public string OpenWeatherApiKey { get; set; } = string.Empty;
}
