@echo off
setlocal

REM ============================================================
REM  RUTINA DIARIA MLB
REM  PASO 0 : asegura la interfaz web local (localhost:8000)
REM  PASO 0b: si HOY ya se proceso la rutina, sale sin repetir
REM  PASO 0c: recupera dias perdidos por apagones (ETL por fecha)
REM  PASO 1 : ETL en C#  -> carga partidos, clima, boxscores, odds
REM  PASO 2 : Python     -> verifica predicciones pendientes
REM  PASO 3 : Python     -> re-entrena el modelo
REM  PASO 4 : Python     -> predicciones y recomendaciones del dia
REM  Toda la salida (stdout + stderr) se anexa a logs\predicciones_YYYYMMDD.txt
REM ============================================================

REM Cambia a la raiz del proyecto (scripts\ -> .. es la raiz)
cd /d "%~dp0.."

REM Crea la carpeta de logs si no existe
if not exist logs mkdir logs

REM Nombre del log con la fecha del dia: predicciones_20260803.txt
set LOGFILE=logs\predicciones_%date:~-4,4%%date:~-7,2%%date:~-10,2%.txt

echo [%date% %time%] ===== INICIO RUTINA MLB ===== >> "%LOGFILE%"

REM ------------------------------------------------------------
REM PASO 0: Interfaz web local (localhost:8000). Si el puerto ya
REM esta en uso el servidor ya corre y no se toca nada.
REM ------------------------------------------------------------
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "MLB Pronosticos - Servidor" /min cmd /c "python scripts\servidor_predicciones.py"
)

REM ------------------------------------------------------------
REM PASO 0b: si HOY ya se ejecuto la rutina principal (marcador),
REM solo se asegura la web y se sale. Asi, si la laptop se enciende
REM varias veces el mismo dia, no se repite el proceso pesado.
REM ------------------------------------------------------------
python scripts\recuperar_rutina.py --ya-procesado-hoy >nul 2>&1
if not errorlevel 1 goto :fin

REM ------------------------------------------------------------
REM PASO 0c: recuperacion por apagones. Procesa los dias pasados
REM que quedaron sin rutina (laptop apagada) antes de trabajar
REM con HOY. No hace nada si no hay pendientes.
REM ------------------------------------------------------------
echo [RECUP] Revisando dias pendientes por apagones... >> "%LOGFILE%"
python scripts\recuperar_rutina.py >> "%LOGFILE%" 2>&1

REM ------------------------------------------------------------
REM PASO 1: ETL (C#). El proyecto vive en src\ETL\PruebaPronosticos.csproj.
REM "< NUL" cierra stdin para que el Console.ReadLine() final
REM del programa no pause la rutina en forma automatica.
REM ------------------------------------------------------------
echo [ETL] Ejecutando la carga de datos (dotnet run)... >> "%LOGFILE%"
dotnet run --project src\ETL\PruebaPronosticos.csproj < NUL >> "%LOGFILE%" 2>&1
echo [ETL] Codigo de salida: %errorlevel% >> "%LOGFILE%"

REM ------------------------------------------------------------
REM PASO 2: Verificacion de predicciones pendientes (ayer y antes)
REM contra los resultados finales recien cargados por el ETL.
REM ------------------------------------------------------------
echo [VERIF] Ejecutando verificacion de predicciones (Python)... >> "%LOGFILE%"
python src\Modeling\verificar_predicciones.py >> "%LOGFILE%" 2>&1
echo [VERIF] Codigo de salida: %errorlevel% >> "%LOGFILE%"

REM ------------------------------------------------------------
REM PASO 3: Re-entrenado del modelo con el ultimo dia finalizado.
REM ------------------------------------------------------------
echo [TRAIN] Re-entrenando modelo clasificador (Python)... >> "%LOGFILE%"
python src\Modeling\entrenar_modelo.py >> "%LOGFILE%" 2>&1
echo [TRAIN] Codigo de salida: %errorlevel% >> "%LOGFILE%"

REM ------------------------------------------------------------
REM PASO 4: Predicciones (Python). Salida anexada al mismo log.
REM ------------------------------------------------------------
echo [PY] Ejecutando generador_de_predicciones (Python)... >> "%LOGFILE%"
python src\Modeling\recomendar_apuestas.py >> "%LOGFILE%" 2>&1
echo [PY] Codigo de salida: %errorlevel% >> "%LOGFILE%"

REM Marcador: HOY ya procesado (evita repeticiones el mismo dia)
python scripts\recuperar_rutina.py --marcar-hoy >> "%LOGFILE%" 2>&1

echo [%date% %time%] ===== FIN RUTINA MLB ===== >> "%LOGFILE%"

:fin
REM Envia el errorlevel mas reciente como codigo del bat
exit /b %errorlevel%