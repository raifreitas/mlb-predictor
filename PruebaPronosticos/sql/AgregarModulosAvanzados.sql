IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'GameLog' AND COLUMN_NAME = 'UmpireNombre')
BEGIN
    ALTER TABLE dbo.GameLog ADD UmpireNombre nvarchar(100) NULL;
END
GO

IF OBJECT_ID('dbo.PitcherMano', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.PitcherMano (
        PitcherId int NOT NULL PRIMARY KEY,
        Mano varchar(1) NOT NULL
    );
END
GO

IF OBJECT_ID('dbo.TeamOPS_Handedness', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.TeamOPS_Handedness (
        Equipo nvarchar(50) NOT NULL,
        Temporada int NOT NULL,
        OPSvsLHP decimal(4,3) NULL,
        OPSvsRHP decimal(4,3) NULL,
        CONSTRAINT PK_TeamOPS_Handedness PRIMARY KEY (Equipo, Temporada)
    );
END
GO
