namespace PruebaPronosticos.ETL;

public class GameLog
{
    public DateTime Fecha { get; set; }
    public string Estadio { get; set; } = string.Empty;
    public string EquipoLocal { get; set; } = string.Empty;
    public string EquipoVisita { get; set; } = string.Empty;
    public int? PitcherLocalId { get; set; }
    public int? PitcherVisitaId { get; set; }
    public int CarrerasLocal { get; set; }
    public int CarrerasVisita { get; set; }
    public double? TemperaturaC { get; set; }
    public double? VientoVelocidad { get; set; }
    public string? VientoDireccion { get; set; }
    public double? EraBullpenLocal { get; set; }
    public double? EraBullpenVisita { get; set; }
    public double? WhipAbridorLocal { get; set; }
    public double? WhipAbridorVisita { get; set; }
    public string? UmpireNombre { get; set; }
    public string? UmpireHomePlate { get; set; }
    public long? GamePk { get; set; }
    public bool EsFinal { get; set; }
    public DateTime? HoraInicioUtc { get; set; }
    public string? GameType { get; set; }
}
