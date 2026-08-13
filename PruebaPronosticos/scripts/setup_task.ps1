# ============================================================
#  setup_task.ps1 - Registra las tareas programadas de MLB:
#   - MLB_Predictive_Daily  : todos los dias 09:00
#     (ETL de ayer + verificacion + re-entrenado + pronosticos)
#   - MLB_Predictive_Evening: todos los dias 17:00
#     (snapshot de cuotas de hoy + reevaluacion pre-juego)
#   - MLB_Predictive_Night  : todos los dias 23:30
#     (verifica los partidos ya finalizados para marcar
#      GANADA/PERDIDA/PUSH el mismo dia)
#   - MLB_Predictive_PreGame: todos los dias 18:30
#     (snapshot ligero de cuotas ~30 min antes del primer
#      pitch: linea de cierre real + moneyline, sin redescargar
#      partidos/clima/boxscores)
#   - MLB_Predictive_PreGameLate: todos los dias 20:30
#     (segundo snapshot ligero: cubre los juegos de la costa
#      oeste cuyo primer pitch es ~21:00 ET)
#
#  NOTA: la interfaz web (localhost:8000) NO se registra aqui;
#  la levanta automaticamente la rutina diaria (PASO 0) y la
#  pre-juego. Para abrirla a mano: scripts\ver_pronosticos.bat
#
#  USO (PowerShell como Administrador):
#    Set-ExecutionPolicy -Scope Process Bypass   (una vez)
#    powershell -ExecutionPolicy Bypass -File .\setup_task.ps1
#
#  Caracteristicas:
#   - Disparadores diarios (09:00, 17:00, 18:30, 20:30 y 23:30).
#   - -WakeToRun: la laptop despierta de Suspension
#     (el equipo DEBE tener permitido el "despertador" de la
#     placa: BIOS/UEFI -> Wake on RTC/Alarm; Windows -> ok).
#   - -StartWhenAvailable: si estaba apagada a la hora, se
#     ejecuta al encender.
#   - -DontStopIfGoingOnBatteries: sigue aunque no tenga AC.
# ============================================================

$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\raifj\source\repos\PruebaPronosticos\PruebaPronosticos"

$tareas = @(
    @{
        Nombre    = "MLB_Predictive_Daily"
        Bat       = Join-Path $projectRoot "scripts\rutina_mlb.bat"
        Hora      = "09:00"
        Web       = $false
    },
    @{
        Nombre    = "MLB_Predictive_Evening"
        Bat       = Join-Path $projectRoot "scripts\rutina_mlb_prejuego.bat"
        Hora      = "17:00"
        Web       = $false
    },
    @{
        Nombre    = "MLB_Predictive_PreGame"
        Bat       = Join-Path $projectRoot "scripts\rutina_snapshot_prejuego.bat"
        Hora      = "18:30"
        Web       = $false
    },
    @{
        Nombre    = "MLB_Predictive_PreGameLate"
        Bat       = Join-Path $projectRoot "scripts\rutina_snapshot_prejuego.bat"
        Hora      = "20:30"
        Web       = $false
    },
    @{
        Nombre    = "MLB_Predictive_Night"
        Bat       = Join-Path $projectRoot "scripts\rutina_mlb_noche.bat"
        Hora      = "23:30"
        Web       = $false
    }
)

# Configuracion comun: despertar de suspension + arrancar si
# estaba apagada + limite de tiempo (2h).
# NOTA PS 5.1: estos parametros deben ir como switch DESNUDO
# (-WakeToRun) y NO con valor (-WakeToRun $true da error).
$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Se ejecuta con la sesion interactiva del usuario actual (asi
# ve "python" y el perfil de la laptop; si prefieres correr en
# background sin abrir consola, cambia LogonType a S4U).
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

foreach ($t in $tareas) {
    if (-not (Test-Path -LiteralPath $t.Bat)) {
        throw "No se encontro $($t.Bat)"
    }

    $action  = New-ScheduledTaskAction -Execute $t.Bat
    $trigger = New-ScheduledTaskTrigger -Daily -At $t.Hora

    Register-ScheduledTask `
        -TaskName $t.Nombre `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null

    $tarea = Get-ScheduledTask -TaskName $t.Nombre
    Write-Host "Tarea registrada: $($tarea.TaskName)"
    Write-Host "  Estado          : $($tarea.State)"
    Write-Host "  Accion          : $($tarea.Actions.Execute) $($tarea.Actions.Arguments)"
    Write-Host "  WakeToRun       : $($tarea.Settings.WakeToRun)"
    Write-Host "  StartWhenAvail  : $($tarea.Settings.StartWhenAvailable)"
    Write-Host "  Proxima ejec.   : $((Get-ScheduledTaskInfo -TaskName $t.Nombre).NextRunTime)"
    Write-Host ""
}

# Prueba manual opcional (descomenta para validar la rutina):
# Start-ScheduledTask -TaskName MLB_Predictive_Daily
