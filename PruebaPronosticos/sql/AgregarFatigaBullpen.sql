-- ============================================================
-- FASE 1: Fatiga de Bullpen de 72 horas
-- 1) Tabla dbo.PitcherGameLog con los pitcheos diarios por
--    lanzador (leida del boxscore de la StatsAPI MLB).
-- 2) Vista dbo.vwFatigaBullpen3d: pitcheos acumulados SOLO de
--    relevistas (IsStarter = 0) en los 3 dias previos del
--    historial por equipo, usando una funcion de ventana
--    SUM(...) OVER (PARTITION BY Team ORDER BY Fecha
--    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING).
-- ============================================================

-- ------------------------------------------------------------
-- 1) Tabla de pitcheos por jugador por partido
--    - GameID: gamePk de la StatsAPI MLB (el partido).
--    - Team: nombre del equipo TAL COMO aparece en dbo.GameLog
--      (EquipoLocal / EquipoVisita) para que el JOIN en Python
--      sea directo.
--    - PitcherID: id del lanzador (0 => fila semilla para
--      partidos sin relevistas, garantiza la serie diaria).
-- ============================================================
IF OBJECT_ID('dbo.PitcherGameLog', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.PitcherGameLog
    (
        GameID        BIGINT        NOT NULL,
        Fecha         DATE          NOT NULL,
        Team          NVARCHAR(80)  NOT NULL,
        PitcherID     INT           NOT NULL,
        IsStarter     BIT           NOT NULL DEFAULT 0,
        PitchesThrown INT           NOT NULL DEFAULT 0,
        CONSTRAINT PK_PitcherGameLog PRIMARY KEY (GameID, Team, PitcherID)
    );
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_PitcherGameLog_TeamFecha'
                 AND object_id = OBJECT_ID(N'dbo.PitcherGameLog'))
BEGIN
    CREATE INDEX IX_PitcherGameLog_TeamFecha
        ON dbo.PitcherGameLog (Team, Fecha);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_PitcherGameLog_Fecha'
                 AND object_id = OBJECT_ID(N'dbo.PitcherGameLog'))
BEGIN
    CREATE INDEX IX_PitcherGameLog_Fecha
        ON dbo.PitcherGameLog (Fecha);
END
GO

-- ------------------------------------------------------------
-- 2) Vista de fatiga: pitches acumulados de relevistas en los
--    3 juegos previos por EQUIPO y FECHA.
--    NOTA: Diario agrega por (Team, Fecha) y la ventana ROWS
--    BETWEEN 3 PRECEDING AND 1 PRECEDING suma exclusivamente los
--    tres juegos ANTERIORES (sin incluir el propio juego), que es
--    justo la carga de trabajo de las 72 horas previas.
-- ------------------------------------------------------------

CREATE OR ALTER VIEW dbo.vwFatigaBullpen3d
AS
WITH Diario AS
(
    SELECT Team,
           Fecha,
           SUM(PitchesThrown) AS TotalPitches
    FROM dbo.PitcherGameLog
    WHERE IsStarter = 0
    GROUP BY Team, Fecha
)
SELECT Team,
       Fecha,
       ISNULL(SUM(TotalPitches) OVER (
           PARTITION BY Team ORDER BY Fecha
           ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING), 0) AS Fatiga_Bullpen_3d
FROM Diario;
GO