IF OBJECT_ID('dbo.GameLog', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.GameLog
    (
        Id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_GameLog PRIMARY KEY,
        Fecha DATE NOT NULL,
        Estadio NVARCHAR(120) NOT NULL,
        EquipoLocal NVARCHAR(80) NOT NULL,
        EquipoVisita NVARCHAR(80) NOT NULL,
        PitcherLocalId INT NULL,
        PitcherVisitaId INT NULL,
        CarrerasLocal INT NOT NULL,
        CarrerasVisita INT NOT NULL,
        TemperaturaC DECIMAL(5,2) NULL
    );
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_GameLog_Fecha_Equipos' AND object_id = OBJECT_ID('dbo.GameLog'))
BEGIN
    CREATE UNIQUE INDEX UQ_GameLog_Fecha_Equipos ON dbo.GameLog (Fecha, EquipoLocal, EquipoVisita);
END
GO
