# Install auto-start + prevent sleep so IPA keeps running when PC is locked.
param(
    [switch]$SkipPowerSettings
)

$ErrorActionPreference = "Stop"
$WebAppRoot = Split-Path -Parent $PSScriptRoot
$StartScript = Join-Path $PSScriptRoot "start_daemon.ps1"
$TaskName = "IPA-Cell-Analysis"

Write-Host "Installing always-on IPA server..."

# --- Prevent sleep (needs admin for some systems) ---
if (-not $SkipPowerSettings) {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
    if ($isAdmin) {
        powercfg /change standby-timeout-ac 0 | Out-Null
        powercfg /change hibernate-timeout-ac 0 | Out-Null
        powercfg /change standby-timeout-dc 0 | Out-Null
        powercfg /change hibernate-timeout-dc 0 | Out-Null
        Write-Host "Power: sleep/hibernate disabled while on power (and battery if supported)."
    } else {
        Write-Host "Tip: re-run as Administrator to disable system sleep automatically."
        Write-Host "     Or set Windows Power Options -> Sleep = Never manually."
    }
}

# --- Build + start now ---
& $StartScript -RebuildFrontend

# --- Scheduled task: start at logon + restart on failure ---
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`""

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$logonTrigger.Delay = "PT1M"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $logonTrigger `
    -Settings $settings `
    -Description "IPA Cell Analysis Platform - auto-start backend on port 8010" `
    -Force | Out-Null

Write-Host ""
Write-Host "Done. Task '$TaskName' will start the app 1 minute after you sign in."
Write-Host ""
Write-Host "App URL:  http://10.40.20.30:8010"
Write-Host "Stop:     powershell -File $PSScriptRoot\stop_daemon.ps1"
Write-Host "Restart:  powershell -File $StartScript"
Write-Host ""
Write-Host "Keep this PC signed in (locking is OK). Avoid Sleep/Hibernate in Windows Settings."
