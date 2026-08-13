-- ============================================================
-- FASE 1: Lineas reales de cierre (The Odds API)
-- 1) Columnas nuevas en dbo.GameLog para la linea de cierre real.
-- 2) Tabla dbo.HistoricalOdds con snapshots por evento y casa.
-- ============================================================

IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
               WHERE TABLE_NAME = 'GameLog' AND COLUMN_NAME = 'Linea_Casino_Real')
BEGIN
    ALTER TABLE dbo.GameLog ADD Linea_Casino_Real DECIMAL(4,1) NULL;
END
GO

IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
               WHERE TABLE_NAME = 'GameLog' AND COLUMN_NAME = 'Cuota_Over_Real')
BEGIN
    ALTER TABLE dbo.GameLog ADD Cuota_Over_Real DECIMAL(6,3) NULL;
END
GO

IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
               WHERE TABLE_NAME = 'GameLog' AND COLUMN_NAME = 'Cuota_Under_Real')
BEGIN
    ALTER TABLE dbo.GameLog ADD Cuota_Under_Real DECIMAL(6,3) NULL;
END
GO

IF OBJECT_ID('dbo.HistoricalOdds', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.HistoricalOdds
    (
        EventoId          NVARCHAR(64)   NOT NULL,
        Casa              NVARCHAR(50)   NOT NULL,
        Fecha             DATE           NOT NULL,
        EquipoLocal       NVARCHAR(80)   NOT NULL,
        EquipoVisita      NVARCHAR(80)   NOT NULL,
        CommenceTimeUtc   DATETIME2      NULL,
        Linea             DECIMAL(4,1)   NULL,
        CuotaOver         DECIMAL(6,3)   NULL,
        CuotaUnder        DECIMAL(6,3)   NULL,
        UltimaActualizacion DATETIME2    NULL,
        CONSTRAINT PK_HistoricalOdds PRIMARY KEY (EventoId, Casa)
    );
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_HistoricalOdds_FechaEquipos'
                 AND object_id = OBJECT_ID('dbo.HistoricalOdds'))
BEGIN
    CREATE INDEX IX_HistoricalOdds_FechaEquipos
        ON dbo.HistoricalOdds (Fecha, EquipoLocal, EquipoVisita);
END
GO
