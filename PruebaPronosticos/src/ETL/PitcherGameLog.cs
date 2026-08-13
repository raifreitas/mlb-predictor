namespace PruebaPronosticos.ETL;

/// <summary>
/// Fila de pitcheo por jugador y partido extraida del boxscore de la
/// StatsAPI MLB (endpoint /game/{gamePk}/boxscore). Destinada a la tabla
/// dbo.PitcherGameLog.
/// </summary>
public class PitcherGameLogRow
{
    public long GameId { get; set; }
    public DateTime Fecha { get; set; }
    public string Team { get; set; } = string.Empty;
    public int PitcherId { get; set; }
    public bool IsStarter { get; set; }
    public int PitchesThrown { get; set; }
}