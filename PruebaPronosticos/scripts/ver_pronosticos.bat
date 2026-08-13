@echo off
cd /d "%~dp0.."
start "MLB Pronosticos - Servidor" /min cmd /c "python scripts\servidor_predicciones.py"
timeout /t 2 /nobreak >nul
start "" http://localhost:8000
