-- Agrega la cuota decimal (precio) al momento del pick a dbo.Predicciones
-- para calcular el P/L REAL (ya no estimado con -110).
IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID(N'dbo.Predicciones')
      AND name = N'Cuota'
)
BEGIN
    ALTER TABLE dbo.Predicciones ADD Cuota DECIMAL(6,2) NULL;
END
