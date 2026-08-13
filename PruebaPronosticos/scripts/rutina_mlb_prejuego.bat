@echo off
setlocal

REM ============================================================
REM  RUTINA PRE-JUEGO (17:00) - MLB
REM  1) ETL con la fecha de HOY: captura un snapshot de cuotas
REM     actual (movimiento de linea -> dbo.LineaSnapshots) y
REM     refresca abridores/clima sin tocar resultados.
REM  2) Python -> reevalua los juegos de hoy con lineas y
REM     abridores ACTUALIZADOS (ideal para juegos nocturnos).
REM  Salida anexada a logs\prejuego_YYYYMMDD.txt
REM ============================================================

REM Cambia a la raiz del proyecto (scripts\ -> .. es la raiz)
cd /d "%~dp0.."

REM Crea la carpeta de logs si no existe
if not exist logs mkdir logs

REM Nombre del log con la fecha del dia (locale dd/mm/aaaa)
set LOGFILE=logs\prejuego_%date:~-4,4%%date:~-7,2%%date:~-10,2%.txt
set HOY=%date:~-4,4%-%date:~-7,2%-%date:~-10,2%

echo [%date% %time%] ===== INICIO RUTINA PRE-JUEGO (%HOY%) ===== >> "%LOGFILE%"

REM ------------------------------------------------------------
REM PASO 0: Interfaz web local (localhost:8000). Si el puerto ya
REM esta en uso el servidor ya corre y no se toca nada.
REM ------------------------------------------------------------
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "MLB Pronosticos - Servidor" /min cmd /c "python scripts\servidor_predicciones.py"
)

REM ------------------------------------------------------------
REM PASO 0b: recuperacion por apagones (si la de 09:00 no corrio
REM porque la laptop estaba apagada, aqui se recupera HOY mismo).
REM ------------------------------------------------------------
python scripts\recuperar_rutina.py >> "%LOGFILE%" 2>&1

REM ------------------------------------------------------------
REM PASO 1: ETL para HOY (snapshot de cuotas + upsert inofensivo).
REM ------------------------------------------------------------
echo [ETL] Capturando snapshot de cuotas para %HOY%... >> "%LOGFILE%"
dotnet run --project src\ETL\PruebaPronosticos.csproj %HOY% %HOY% < NUL >> "%LOGFILE%" 2>&1
echo [ETL] Codigo de salida: %errorlevel% >> "%LOGFILE%"

REM ------------------------------------------------------------
REM PASO 2: Reevaluacion de los juegos de hoy (Python).
REM ------------------------------------------------------------
echo [PY] Reevaluando juegos de hoy (Python)... >> "%LOGFILE%"
python src\Modeling\recomendar_apuestas.py >> "%LOGFILE%" 2>&1
echo [PY] Codigo de salida: %errorlevel% >> "%LOGFILE%"

echo [%date% %time%] ===== FIN RUTINA PRE-JUEGO ===== >> "%LOGFILE%"

REM Envia el errorlevel mas reciente (Python) como codigo del bat
exit /b %errorlevel%
