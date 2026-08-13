@echo off
setlocal

REM ============================================================
REM  RUTINA SNAPSHOT PRE-JUEGO - MLB (modo ligero de cuotas)
REM  Captura las lineas/cuotas ACTUALES (Totals + Moneyline)
REM  cerca del primer pitch (~30 min antes), cuando el mercado
REM  ya fijo la linea de cierre, sin re-descargar partidos,
REM  clima ni boxscores. Rapido y economico en peticiones API.
REM  Salida anexada a logs\snapshot_YYYYMMDD.txt
REM ============================================================

REM Cambia a la raiz del proyecto (scripts\ -> .. es la raiz)
cd /d "%~dp0.."

REM Crea la carpeta de logs si no existe
if not exist logs mkdir logs

REM Nombre del log con la fecha del dia (locale dd/mm/aaaa)
set LOGFILE=logs\snapshot_%date:~-4,4%%date:~-7,2%%date:~-10,2%.txt
set HOY=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%

echo [%date% %time%] ===== INICIO SNAPSHOT PRE-JUEGO (%HOY%) ===== >> "%LOGFILE%"

REM ------------------------------------------------------------
REM ETL modo ligero: SOLO cuotas actuales + resolver cierre
REM ------------------------------------------------------------
dotnet run --project src\ETL\PruebaPronosticos.csproj %HOY% %HOY% --solo-odds < NUL >> "%LOGFILE%" 2>&1
echo [ETL] Codigo de salida: %errorlevel% >> "%LOGFILE%"

echo [%date% %time%] ===== FIN SNAPSHOT PRE-JUEGO ===== >> "%LOGFILE%"
exit /b %errorlevel%
