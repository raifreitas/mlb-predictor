-- ============================================================
-- Tabla dbo.Predicciones: cierra el ciclo de apuestas.
-- Cada jugada sugerida por recomendar_apuestas.py se registra
-- como PENDIENTE; verificar_predicciones.py la marca GANADA /
-- PERDIDA / PUSH cuando el partido queda finalizado en GameLog.
-- ============================================================

IF OBJECT_ID('dbo.Predicciones', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Predicciones
    (
        Id                INT IDENTITY(1,1) NOT NULL,
        Fecha             DATE           NOT NULL,
        EquipoLocal       NVARCHAR(80)   NOT NULL,
        EquipoVisita      NVARCHAR(80)   NOT NULL,
        TipoApuesta       NVARCHAR(10)   NOT NULL,
        Linea             DECIMAL(4,1)   NOT NULL,
        Unidades          DECIMAL(4,2)   NOT NULL,
        Edge              DECIMAL(6,2)   NULL,
        Estado            NVARCHAR(12)   NOT NULL CONSTRAINT DF_Predicciones_Estado
            DEFAULT ('PENDIENTE'),
        CarrerasTotales   INT            NULL,
        FechaVerificacion DATETIME2      NULL,
        CreadoUtc         DATETIME2      NOT NULL CONSTRAINT DF_Predicciones_CreadoUtc
            DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_Predicciones PRIMARY KEY (Id)
    );
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'UX_Predicciones_Jugada'
                 AND object_id = OBJECT_ID('dbo.Predicciones'))
BEGIN
    CREATE UNIQUE INDEX UX_Predicciones_Jugada
        ON dbo.Predicciones (Fecha, EquipoLocal, EquipoVisita, TipoApuesta);
END
GO
