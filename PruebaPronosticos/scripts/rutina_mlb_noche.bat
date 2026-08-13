@echo off
cd /d "%~dp0.."
echo ================================================
echo Rutina nocturna: verificacion de partidos de hoy
echo %date% %time%
echo ================================================
REM Recuperacion por apagones (si la laptop estuvo apagada,
REM se procesan aqui los dias faltantes antes de verificar)
python scripts\recuperar_rutina.py 2>&1
if errorlevel 1 (
    echo [ERROR] Fallo la recuperacion nocturna
)
python src\Modeling\verificar_predicciones.py
if errorlevel 1 (
    echo [ERROR] Fallo la verificacion nocturna
) else (
    echo [OK] Verificacion nocturna completada
)
