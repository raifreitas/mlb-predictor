param(
    [string]$Fecha = (Get-Date -Format "yyyy-MM-dd")
)
$ErrorActionPreference = "Continue"
$raiz = "C:\Users\raifj\source\repos\PruebaPronosticos\PruebaPronosticos"
$log = Join-Path $raiz "logs\recomendar_1730_$(Get-Date -Format 'yyyyMMdd').log"

"[$(Get-Date -Format 'HH:mm:ss')] INICIO corrida $Fecha (lineas pre-first-pitch)" | Out-File $log -Encoding utf8

Set-Location (Join-Path $raiz "src\Modeling")

# 1) ETL diario por si el marcador de algun juego ya se actualizo (no bloquea).
#    Se entuba un Enter al stdin: el ETL termina con Console.ReadLine() y si no
#    recibe entrada se queda colgado en tareas programadas.
"[$(Get-Date -Format 'HH:mm:ss')] ETL $Fecha" | Out-File $log -Append -Encoding utf8
"" | & dotnet run --project (Join-Path $raiz "src\ETL\PruebaPronosticos.csproj") -- $Fecha $Fecha 2>&1 | Out-File $log -Append -Encoding utf8

# 2) Recomendador con las lineas finales pre-juego.
"[$(Get-Date -Format 'HH:mm:ss')] RECOMENDAR_APUESTAS" | Out-File $log -Append -Encoding utf8
python recomendar_apuestas.py --fecha $Fecha 2>&1 | Out-File $log -Append -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] FIN corrida $Fecha" | Out-File $log -Append -Encoding utf8