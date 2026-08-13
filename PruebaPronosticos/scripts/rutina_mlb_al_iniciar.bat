@echo off
REM ============================================================
REM  SE EJECUTA AL INICIAR SESION EN WINDOWS (carpeta Inicio).
REM  Llama a la rutina diaria MLB. El marcador interno
REM  (scripts\recuperar_rutina.py) evita repetir el proceso si
REM  HOY ya fue procesado (09:00, 17:00 o 23:30) y recupera los
REM  dias perdidos cuando la laptop estuvo apagada.
REM  NO requiere permisos de administrador.
REM ============================================================
set PROY=C:\Users\raifj\source\repos\PruebaPronosticos\PruebaPronosticos
cd /d "%PROY%"

if not exist "%PROY%\logs" mkdir "%PROY%\logs"
set LOGFILE=%PROY%\logs\inicio_sesion_%date:~-4,4%%date:~-7,2%%date:~-10,2%.txt

echo [%date% %time%] ===== INICIO SESION: RUTINA MLB ===== >> "%LOGFILE%"
call "%PROY%\scripts\rutina_mlb.bat" >> "%LOGFILE%" 2>&1
echo [%date% %time%] ===== FIN SESION: RUTINA MLB ===== >> "%LOGFILE%"
