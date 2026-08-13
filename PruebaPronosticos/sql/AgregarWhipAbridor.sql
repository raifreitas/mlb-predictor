-- Agrega las columnas de WHIP de abridores a dbo.GameLog (filtro contra abridores volatiles).
-- Idempotente: se puede ejecutar varias veces sin error.

IF COL_LENGTH('dbo.GameLog', 'WHIP_Abridor_Local') IS NULL
BEGIN
    ALTER TABLE dbo.GameLog ADD WHIP_Abridor_Local DECIMAL(4,2) NULL;
END
GO

IF COL_LENGTH('dbo.GameLog', 'WHIP_Abridor_Visita') IS NULL
BEGIN
    ALTER TABLE dbo.GameLog ADD WHIP_Abridor_Visita DECIMAL(4,2) NULL;
END
GO
