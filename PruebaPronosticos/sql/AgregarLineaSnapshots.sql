-- ============================================================
-- Tabla dbo.LineaSnapshots: historial APPEND-ONLY de cuotas.
-- Cada ejecucion del ETL (manana y pre-juego 17:00) inserta un
-- nuevo snapshot; asi se puede medir el movimiento de linea
-- (HistoricalOdds solo conserva el ultimo por evento/casa).
-- ============================================================

IF OBJECT_ID('dbo.LineaSnapshots', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.LineaSnapshots
    (
        Id           INT IDENTITY(1,1) NOT NULL,
        EventoId     NVARCHAR(64)   NOT NULL,
        Casa         NVARCHAR(50)   NOT NULL,
        Fecha        DATE           NOT NULL,
        EquipoLocal  NVARCHAR(80)   NOT NULL,
        EquipoVisita NVARCHAR(80)   NOT NULL,
        Linea        DECIMAL(4,1)   NULL,
        CuotaOver    DECIMAL(6,3)   NULL,
        CuotaUnder   DECIMAL(6,3)   NULL,
        CapturadoUtc DATETIME2      NOT NULL CONSTRAINT DF_LineaSnapshots_CapturadoUtc
            DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_LineaSnapshots PRIMARY KEY (Id)
    );
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_LineaSnapshots_FechaEquipos'
                 AND object_id = OBJECT_ID('dbo.LineaSnapshots'))
BEGIN
    CREATE INDEX IX_LineaSnapshots_FechaEquipos
        ON dbo.LineaSnapshots (Fecha, EquipoLocal, EquipoVisita);
END
GO
