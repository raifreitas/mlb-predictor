using Microsoft.Data.SqlClient;

namespace PruebaPronosticos.ETL;

public class GameRepository
{
    private readonly string _connectionString;

    public GameRepository(string connectionString)
    {
        _connectionString = connectionString;
    }

    public async Task<int> InsertarGameLogsAsync(IEnumerable<GameLog> partidos, CancellationToken cancellationToken = default)
    {
        var lista = partidos.ToList();
        if (lista.Count == 0)
            return 0;

        const string sql = @"
            IF EXISTS (
                SELECT 1 FROM dbo.GameLog
                WHERE Fecha = @Fecha AND EquipoLocal = @EquipoLocal AND EquipoVisita = @EquipoVisita)
            BEGIN
                UPDATE dbo.GameLog
                SET CarrerasLocal = @CarrerasLocal,
                    CarrerasVisita = @CarrerasVisita,
                    EsFinal = @EsFinal,
                    HoraInicioUtc = COALESCE(@HoraInicioUtc, HoraInicioUtc),
                    TemperaturaC = COALESCE(@TemperaturaC, TemperaturaC),
                    Viento_Velocidad = COALESCE(@VientoVelocidad, Viento_Velocidad),
                    Viento_Direccion = COALESCE(@VientoDireccion, Viento_Direccion),
                    ERA_Bullpen_Local = COALESCE(@EraBullpenLocal, ERA_Bullpen_Local),
                    ERA_Bullpen_Visita = COALESCE(@EraBullpenVisita, ERA_Bullpen_Visita),
                    WHIP_Abridor_Local = COALESCE(@WhipAbridorLocal, WHIP_Abridor_Local),
                    WHIP_Abridor_Visita = COALESCE(@WhipAbridorVisita, WHIP_Abridor_Visita),
                    UmpireNombre = COALESCE(@UmpireNombre, UmpireNombre),
                    UmpireHomePlate = COALESCE(@UmpireHomePlate, UmpireHomePlate)
                WHERE Fecha = @Fecha AND EquipoLocal = @EquipoLocal AND EquipoVisita = @EquipoVisita;
            END
            ELSE
            BEGIN
                INSERT INTO dbo.GameLog (Fecha, Estadio, EquipoLocal, EquipoVisita, PitcherLocalId, PitcherVisitaId, CarrerasLocal, CarrerasVisita, EsFinal, HoraInicioUtc, TemperaturaC, Viento_Velocidad, Viento_Direccion, ERA_Bullpen_Local, ERA_Bullpen_Visita, WHIP_Abridor_Local, WHIP_Abridor_Visita, UmpireNombre, UmpireHomePlate)
                VALUES (@Fecha, @Estadio, @EquipoLocal, @EquipoVisita, @PitcherLocalId, @PitcherVisitaId, @CarrerasLocal, @CarrerasVisita, @EsFinal, @HoraInicioUtc, @TemperaturaC, @VientoVelocidad, @VientoDireccion, @EraBullpenLocal, @EraBullpenVisita, @WhipAbridorLocal, @WhipAbridorVisita, @UmpireNombre, @UmpireHomePlate);
            END";

        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        int insertados = 0;
        using var transaccion = conexion.BeginTransaction();

        try
        {
            foreach (var partido in lista)
            {
                await using var comando = new SqlCommand(sql, conexion, transaccion);
                comando.Parameters.AddWithValue("@Fecha", partido.Fecha.Date);
                comando.Parameters.AddWithValue("@Estadio", partido.Estadio);
                comando.Parameters.AddWithValue("@EquipoLocal", partido.EquipoLocal);
                comando.Parameters.AddWithValue("@EquipoVisita", partido.EquipoVisita);
                comando.Parameters.AddWithValue("@PitcherLocalId", (object?)partido.PitcherLocalId ?? DBNull.Value);
                comando.Parameters.AddWithValue("@PitcherVisitaId", (object?)partido.PitcherVisitaId ?? DBNull.Value);
                comando.Parameters.AddWithValue("@CarrerasLocal", partido.CarrerasLocal);
                comando.Parameters.AddWithValue("@CarrerasVisita", partido.CarrerasVisita);
                comando.Parameters.AddWithValue("@EsFinal", partido.EsFinal ? 1 : 0);
                comando.Parameters.AddWithValue("@HoraInicioUtc", (object?)partido.HoraInicioUtc ?? DBNull.Value);
                comando.Parameters.AddWithValue("@TemperaturaC", (object?)partido.TemperaturaC ?? DBNull.Value);
                comando.Parameters.AddWithValue("@VientoVelocidad", (object?)partido.VientoVelocidad ?? DBNull.Value);
                comando.Parameters.AddWithValue("@VientoDireccion", (object?)partido.VientoDireccion ?? DBNull.Value);
                comando.Parameters.AddWithValue("@EraBullpenLocal", (object?)partido.EraBullpenLocal ?? DBNull.Value);
                comando.Parameters.AddWithValue("@EraBullpenVisita", (object?)partido.EraBullpenVisita ?? DBNull.Value);
                comando.Parameters.AddWithValue("@WhipAbridorLocal", (object?)partido.WhipAbridorLocal ?? DBNull.Value);
                comando.Parameters.AddWithValue("@WhipAbridorVisita", (object?)partido.WhipAbridorVisita ?? DBNull.Value);
                comando.Parameters.AddWithValue("@UmpireNombre", (object?)partido.UmpireNombre ?? DBNull.Value);
                comando.Parameters.AddWithValue("@UmpireHomePlate", (object?)partido.UmpireHomePlate ?? DBNull.Value);

                int filasAfectadas = await comando.ExecuteNonQueryAsync(cancellationToken);

                if (filasAfectadas > 0)
                {
                    insertados++;
                }
            }

            transaccion.Commit();
        }
        catch
        {
            transaccion.Rollback();
            throw;
        }

        return insertados;
    }

    public async Task<int> ContarGameLogsAsync(CancellationToken cancellationToken = default)
    {
        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        await using var comando = new SqlCommand("SELECT COUNT(*) FROM dbo.GameLog", conexion);
        return (int)(await comando.ExecuteScalarAsync(cancellationToken) ?? 0);
    }

    public async Task<int> GuardarPitcherManoAsync(IEnumerable<(int PitcherId, string Mano)> datos,
        CancellationToken cancellationToken = default)
    {
        var lista = datos.DistinctBy(d => d.PitcherId).ToList();
        if (lista.Count == 0)
            return 0;

        const string sql = @"
            IF EXISTS (SELECT 1 FROM dbo.PitcherMano WHERE PitcherId = @PitcherId)
            BEGIN
                UPDATE dbo.PitcherMano SET Mano = @Mano WHERE PitcherId = @PitcherId;
            END
            ELSE
            BEGIN
                INSERT INTO dbo.PitcherMano (PitcherId, Mano) VALUES (@PitcherId, @Mano);
            END";

        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        int guardados = 0;
        using var transaccion = conexion.BeginTransaction();
        try
        {
            foreach (var dato in lista)
            {
                await using var comando = new SqlCommand(sql, conexion, transaccion);
                comando.Parameters.AddWithValue("@PitcherId", dato.PitcherId);
                comando.Parameters.AddWithValue("@Mano", dato.Mano);
                guardados += await comando.ExecuteNonQueryAsync(cancellationToken);
            }
            transaccion.Commit();
        }
        catch
        {
            transaccion.Rollback();
            throw;
        }
        return guardados;
    }

    /// <summary>
    /// UPSERT de los pitcheos diarios por lanzador (dbo.PitcherGameLog).
    /// Idempotente: re-ejecutar el ETL actualiza los pitcheos del mismo
    /// juego sin duplicar filas. La fila semilla (PitcherId = 0) ancla la
    /// serie diaria de fechas sin relevistas en cada equipo.
    /// </summary>
    public async Task<int> GuardarPitcherGameLogsAsync(
        IEnumerable<PitcherGameLogRow> datos, CancellationToken cancellationToken = default)
    {
        var lista = datos.ToList();
        if (lista.Count == 0)
            return 0;

        const string sql = @"
            IF EXISTS (SELECT 1 FROM dbo.PitcherGameLog
                       WHERE GameID = @GameId AND Team = @Team AND PitcherId = @PitcherId)
            BEGIN
                UPDATE dbo.PitcherGameLog
                SET IsStarter = @IsStarter,
                    PitchesThrown = @PitchesThrown
                WHERE GameID = @GameId AND Team = @Team AND PitcherId = @PitcherId;
            END
            ELSE
            BEGIN
                INSERT INTO dbo.PitcherGameLog (GameID, Fecha, Team, PitcherId, IsStarter, PitchesThrown)
                VALUES (@GameId, @Fecha, @Team, @PitcherId, @IsStarter, @PitchesThrown);
            END";

        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        int guardados = 0;
        using var transaccion = conexion.BeginTransaction();
        try
        {
            foreach (var fila in lista)
            {
                await using var comando = new SqlCommand(sql, conexion, transaccion);
                comando.Parameters.AddWithValue("@GameId", fila.GameId);
                comando.Parameters.AddWithValue("@Fecha", fila.Fecha.Date);
                comando.Parameters.AddWithValue("@Team", fila.Team);
                comando.Parameters.AddWithValue("@PitcherId", fila.PitcherId);
                comando.Parameters.AddWithValue("@IsStarter", fila.IsStarter);
                comando.Parameters.AddWithValue("@PitchesThrown", fila.PitchesThrown);
                guardados += await comando.ExecuteNonQueryAsync(cancellationToken);
            }
            transaccion.Commit();
        }
        catch
        {
            transaccion.Rollback();
            throw;
        }
        return guardados;
    }

    public async Task<int> GuardarOpsSplitsAsync(IEnumerable<MlbDataFetcher.TeamOpsSplits> splits,
        CancellationToken cancellationToken = default)
    {
        var lista = splits.ToList();
        if (lista.Count == 0)
            return 0;

        const string sql = @"
            IF EXISTS (SELECT 1 FROM dbo.TeamOPS_Handedness WHERE Equipo = @Equipo AND Temporada = @Temporada)
            BEGIN
                UPDATE dbo.TeamOPS_Handedness
                SET OPSvsLHP = COALESCE(@OpsVsLhp, OPSvsLHP),
                    OPSvsRHP = COALESCE(@OpsVsRhp, OPSvsRHP)
                WHERE Equipo = @Equipo AND Temporada = @Temporada;
            END
            ELSE
            BEGIN
                INSERT INTO dbo.TeamOPS_Handedness (Equipo, Temporada, OPSvsLHP, OPSvsRHP)
                VALUES (@Equipo, @Temporada, @OpsVsLhp, @OpsVsRhp);
            END";

        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        int guardados = 0;
        using var transaccion = conexion.BeginTransaction();
        try
        {
            foreach (var split in lista)
            {
                await using var comando = new SqlCommand(sql, conexion, transaccion);
                comando.Parameters.AddWithValue("@Equipo", split.Equipo);
                comando.Parameters.AddWithValue("@Temporada", split.Temporada);
                comando.Parameters.AddWithValue("@OpsVsLhp", (object?)split.OpsVsLhp ?? DBNull.Value);
                comando.Parameters.AddWithValue("@OpsVsRhp", (object?)split.OpsVsRhp ?? DBNull.Value);
                guardados += await comando.ExecuteNonQueryAsync(cancellationToken);
            }
            transaccion.Commit();
        }
        catch
        {
            transaccion.Rollback();
            throw;
        }
        return guardados;
    }

    /// <summary>
    /// Guarda snapshots Moneyline (h2h) en dbo.LineaSnapshotsML (append-only).
    /// Mercado aislado de Totals: solo escribe en la tabla ML.
    /// </summary>
    public async Task<int> GuardarLineasH2HAsync(
        IEnumerable<OddsFetcherML.LineaH2H> lineas,
        CancellationToken cancellationToken = default)
    {
        var lista = lineas.ToList();
        if (lista.Count == 0)
            return 0;

        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        int guardados = 0;
        using var transaccion = conexion.BeginTransaction();
        try
        {
            foreach (var linea in lista)
            {
                await using var comando = new SqlCommand(@"
                    INSERT INTO dbo.LineaSnapshotsML
                        (EventoId, Casa, Fecha, EquipoLocal, EquipoVisita,
                         CuotaHome, CuotaAway, CapturadoUtc)
                    VALUES
                        (@EventoId, @Casa, @Fecha, @EquipoLocal, @EquipoVisita,
                         @CuotaHome, @CuotaAway, SYSUTCDATETIME());",
                    conexion, transaccion);
                comando.Parameters.AddWithValue("@EventoId", linea.EventoId);
                comando.Parameters.AddWithValue("@Casa", linea.Casa);
                comando.Parameters.AddWithValue("@Fecha", linea.Fecha.Date);
                comando.Parameters.AddWithValue("@EquipoLocal", linea.EquipoLocal);
                comando.Parameters.AddWithValue("@EquipoVisita", linea.EquipoVisita);
                comando.Parameters.AddWithValue("@CuotaHome", linea.CuotaHome);
                comando.Parameters.AddWithValue("@CuotaAway", linea.CuotaAway);
                guardados += await comando.ExecuteNonQueryAsync(cancellationToken);
            }
            transaccion.Commit();
        }
        catch
        {
            transaccion.Rollback();
            throw;
        }
        return guardados;
    }

    public async Task<int> GuardarLineasHistoricasAsync(
        IEnumerable<OddsFetcher.LineaOdds> lineas, CancellationToken cancellationToken = default)
    {
        var lista = lineas.ToList();
        if (lista.Count == 0)
            return 0;

        const string sql = @"
            IF EXISTS (SELECT 1 FROM dbo.HistoricalOdds WHERE EventoId = @EventoId AND Casa = @Casa)
            BEGIN
                UPDATE dbo.HistoricalOdds
                SET Fecha = @Fecha,
                    EquipoLocal = @EquipoLocal,
                    EquipoVisita = @EquipoVisita,
                    CommenceTimeUtc = COALESCE(@CommenceTimeUtc, CommenceTimeUtc),
                    Linea = COALESCE(@Linea, Linea),
                    CuotaOver = COALESCE(@CuotaOver, CuotaOver),
                    CuotaUnder = COALESCE(@CuotaUnder, CuotaUnder),
                    UltimaActualizacion = COALESCE(@UltimaActualizacion, UltimaActualizacion)
                WHERE EventoId = @EventoId AND Casa = @Casa;
            END
            ELSE
            BEGIN
                INSERT INTO dbo.HistoricalOdds (EventoId, Casa, Fecha, EquipoLocal, EquipoVisita, CommenceTimeUtc, Linea, CuotaOver, CuotaUnder, UltimaActualizacion)
                VALUES (@EventoId, @Casa, @Fecha, @EquipoLocal, @EquipoVisita, @CommenceTimeUtc, @Linea, @CuotaOver, @CuotaUnder, @UltimaActualizacion);
            END";

        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        int guardados = 0;
        using var transaccion = conexion.BeginTransaction();
        try
        {
            foreach (var linea in lista)
            {
                await using var comando = new SqlCommand(sql, conexion, transaccion);
                comando.Parameters.AddWithValue("@EventoId", linea.EventoId);
                comando.Parameters.AddWithValue("@Casa", linea.Casa);
                comando.Parameters.AddWithValue("@Fecha", linea.Fecha.Date);
                comando.Parameters.AddWithValue("@EquipoLocal", linea.EquipoLocal);
                comando.Parameters.AddWithValue("@EquipoVisita", linea.EquipoVisita);
                comando.Parameters.AddWithValue("@CommenceTimeUtc", (object?)linea.CommenceTimeUtc ?? DBNull.Value);
                comando.Parameters.AddWithValue("@Linea", linea.Linea);
                comando.Parameters.AddWithValue("@CuotaOver", linea.CuotaOver);
                comando.Parameters.AddWithValue("@CuotaUnder", linea.CuotaUnder);
                comando.Parameters.AddWithValue("@UltimaActualizacion", (object?)linea.UltimaActualizacion ?? DBNull.Value);
                guardados += await comando.ExecuteNonQueryAsync(cancellationToken);

                // Historial append-only para medir el movimiento de linea:
                // cada captura (manana y pre-juego) deja un snapshot nuevo.
                await using var comandoSnapshot = new SqlCommand(@"
                    INSERT INTO dbo.LineaSnapshots
                        (EventoId, Casa, Fecha, EquipoLocal, EquipoVisita,
                         Linea, CuotaOver, CuotaUnder, CapturadoUtc)
                    VALUES
                        (@EventoId, @Casa, @Fecha, @EquipoLocal, @EquipoVisita,
                         @Linea, @CuotaOver, @CuotaUnder, SYSUTCDATETIME());",
                    conexion, transaccion);
                comandoSnapshot.Parameters.AddWithValue("@EventoId", linea.EventoId);
                comandoSnapshot.Parameters.AddWithValue("@Casa", linea.Casa);
                comandoSnapshot.Parameters.AddWithValue("@Fecha", linea.Fecha.Date);
                comandoSnapshot.Parameters.AddWithValue("@EquipoLocal", linea.EquipoLocal);
                comandoSnapshot.Parameters.AddWithValue("@EquipoVisita", linea.EquipoVisita);
                comandoSnapshot.Parameters.AddWithValue("@Linea", (object?)linea.Linea ?? DBNull.Value);
                comandoSnapshot.Parameters.AddWithValue("@CuotaOver", (object?)linea.CuotaOver ?? DBNull.Value);
                comandoSnapshot.Parameters.AddWithValue("@CuotaUnder", (object?)linea.CuotaUnder ?? DBNull.Value);
                await comandoSnapshot.ExecuteNonQueryAsync(cancellationToken);
            }
            transaccion.Commit();
        }
        catch
        {
            transaccion.Rollback();
            throw;
        }
        return guardados;
    }

    /// <summary>
    /// Calcula la linea de cierre por partido finalizado: ultimo snapshot por
    /// casa, linea = moda entre casas, cuotas = mediana de las casas con esa
    /// linea. Actualiza Linea_Casino_Real / Cuota_Over_Real / Cuota_Under_Real
    /// en dbo.GameLog.
    /// </summary>
    public async Task<int> ResolverLineasRealesAsync(DateTime inicio, DateTime fin,
        CancellationToken cancellationToken = default)
    {
        await using var conexion = new SqlConnection(_connectionString);
        await conexion.OpenAsync(cancellationToken);

        var partidos = new List<(DateTime Fecha, string Local, string Visita)>();
        await using (var comandoPartidos = new SqlCommand(@"
            SELECT Fecha, EquipoLocal, EquipoVisita
            FROM dbo.GameLog
            WHERE Fecha BETWEEN @Inicio AND @Fin
              AND (CarrerasLocal > 0 OR CarrerasVisita > 0)", conexion))
        {
            comandoPartidos.Parameters.AddWithValue("@Inicio", inicio.Date);
            comandoPartidos.Parameters.AddWithValue("@Fin", fin.Date);
            await using var lector = await comandoPartidos.ExecuteReaderAsync(cancellationToken);
            while (await lector.ReadAsync(cancellationToken))
                partidos.Add((lector.GetDateTime(0), lector.GetString(1), lector.GetString(2)));
        }

        if (partidos.Count == 0)
            return 0;

        var cotizaciones = new List<(string EventoId, string Casa, DateTime Fecha, string Local, string Visita, decimal Linea, decimal CuotaOver, decimal CuotaUnder, DateTime? UltimaActualizacion)>();
        await using (var comandoOdds = new SqlCommand(@"
            SELECT EventoId, Casa, Fecha, EquipoLocal, EquipoVisita, Linea, CuotaOver, CuotaUnder, UltimaActualizacion
            FROM dbo.HistoricalOdds
            WHERE Fecha BETWEEN @Inicio AND @Fin", conexion))
        {
            comandoOdds.Parameters.AddWithValue("@Inicio", inicio.Date);
            comandoOdds.Parameters.AddWithValue("@Fin", fin.Date);
            await using var lector = await comandoOdds.ExecuteReaderAsync(cancellationToken);
            while (await lector.ReadAsync(cancellationToken))
                cotizaciones.Add((
                    lector.GetString(0), lector.GetString(1), lector.GetDateTime(2).Date,
                    lector.GetString(3), lector.GetString(4), lector.GetDecimal(5),
                    lector.GetDecimal(6), lector.GetDecimal(7),
                    lector.IsDBNull(8) ? null : lector.GetDateTime(8)));
        }

        // Linea de cierre por partido: ultimo snapshot por casa; linea = moda
        // entre casas; cuotas = mediana entre las casas con esa linea. Se
        // descartan cotizaciones no plausibles (alternate/live en modo raro).
        var cierres = new Dictionary<(DateTime, string, string),
            (decimal Linea, decimal CuotaOver, decimal CuotaUnder)>();
        foreach (var grupo in cotizaciones.GroupBy(c => (c.Fecha, c.Local, c.Visita)))
        {
            var ultimasPorCasa = grupo
                .GroupBy(c => c.Casa)
                .Select(g => g.OrderByDescending(c => c.UltimaActualizacion ?? DateTime.MinValue).First())
                .Where(c => c.Linea >= 6.0m && c.Linea <= 12.0m
                            && c.CuotaOver >= 1.05m && c.CuotaOver <= 5.0m
                            && c.CuotaUnder >= 1.05m && c.CuotaUnder <= 5.0m)
                .ToList();
            if (ultimasPorCasa.Count == 0)
                continue;

            var lineaModa = ultimasPorCasa
                .GroupBy(c => c.Linea)
                .OrderByDescending(g => g.Count())
                .ThenBy(g => g.Key)
                .First().Key;
            var casasConLinea = ultimasPorCasa.Where(c => c.Linea == lineaModa).ToList();
            decimal cuotaOver = Mediana(casasConLinea.Select(c => c.CuotaOver).ToList());
            decimal cuotaUnder = Mediana(casasConLinea.Select(c => c.CuotaUnder).ToList());
            cierres[grupo.Key] = (lineaModa, cuotaOver, cuotaUnder);
        }

        int actualizados = 0;
        using var transaccion = conexion.BeginTransaction();
        try
        {
            const string sql = @"
                UPDATE dbo.GameLog
                SET Linea_Casino_Real = @Linea,
                    Cuota_Over_Real = @CuotaOver,
                    Cuota_Under_Real = @CuotaUnder
                WHERE Fecha = @Fecha AND EquipoLocal = @EquipoLocal AND EquipoVisita = @EquipoVisita";

            foreach (var partido in partidos)
            {
                if (!cierres.TryGetValue((partido.Fecha, partido.Local, partido.Visita), out var cierre))
                    continue;

                await using var comando = new SqlCommand(sql, conexion, transaccion);
                comando.Parameters.AddWithValue("@Linea", cierre.Linea);
                comando.Parameters.AddWithValue("@CuotaOver", cierre.CuotaOver);
                comando.Parameters.AddWithValue("@CuotaUnder", cierre.CuotaUnder);
                comando.Parameters.AddWithValue("@Fecha", partido.Fecha);
                comando.Parameters.AddWithValue("@EquipoLocal", partido.Local);
                comando.Parameters.AddWithValue("@EquipoVisita", partido.Visita);
                actualizados += await comando.ExecuteNonQueryAsync(cancellationToken);
            }
            transaccion.Commit();
        }
        catch
        {
            transaccion.Rollback();
            throw;
        }
        return actualizados;
    }

    private static decimal Mediana(List<decimal> valores)
    {
        if (valores.Count == 0)
            return 0m;
        valores.Sort();
        int n = valores.Count;
        return n % 2 == 1
            ? valores[n / 2]
            : (valores[n / 2 - 1] + valores[n / 2]) / 2m;
    }
}
