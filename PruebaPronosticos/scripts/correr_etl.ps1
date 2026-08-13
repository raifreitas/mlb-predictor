param(
    [string]$Fecha = (Get-Date -Format "yyyy-MM-dd")
)
$ErrorActionPreference = "Continue"
$raiz = "C:\Users\raifj\source\repos\PruebaPronosticos\PruebaPronosticos"
$log = Join-Path $raiz "logs\etl_diario_$(Get-Date -Format 'yyyyMMdd').log"

"[$(Get-Date -Format 'HH:mm:ss')] INICIO ETL $Fecha" | Out-File $log -Encoding utf8

# Se entuba un Enter al stdin: el ETL termina con Console.ReadLine() y si no
# recibe entrada se queda colgado en tareas programadas.
"" | & dotnet run --project (Join-Path $raiz "src\ETL\PruebaPronosticos.csproj") -- $Fecha $Fecha 2>&1 | Out-File $log -Append -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] FIN ETL $Fecha" | Out-File $log -Append -Encoding utf8