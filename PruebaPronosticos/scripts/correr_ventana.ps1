param(
    [int]$VentanaMin = 45
)
$ErrorActionPreference = "Continue"
$raiz = "C:\Users\raifj\source\repos\PruebaPronosticos\PruebaPronosticos"
$log = Join-Path $raiz "logs\recomendar_ventana_$(Get-Date -Format 'yyyyMMdd_HHmm').log"

"[$(Get-Date -Format 'HH:mm:ss')] INICIO runner ventana (${VentanaMin} min pre-juego)" | Out-File $log -Encoding utf8

Set-Location (Join-Path $raiz "src\Modeling")
python recomendar_apuestas.py --fecha (Get-Date -Format "yyyy-MM-dd") --ventana-min $VentanaMin 2>&1 | Out-File $log -Append -Encoding utf8

"[$(Get-Date -Format 'HH:mm:ss')] FIN runner ventana" | Out-File $log -Append -Encoding utf8