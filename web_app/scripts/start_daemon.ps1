# Start IPA app in background (production: backend serves UI on port 8010).
param(
    [switch]$RebuildFrontend
)

$ErrorActionPreference = "Stop"
$WebAppRoot = Split-Path -Parent $PSScriptRoot
$BackendRoot = Join-Path $WebAppRoot "backend"
$FrontendRoot = Join-Path $WebAppRoot "frontend"
$LogDir = Join-Path $WebAppRoot "logs"
$DistDir = Join-Path $FrontendRoot "dist"
$VenvPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Stop-PortListener {
    param([int]$Port)
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

Stop-PortListener -Port 8010
Stop-PortListener -Port 5173
Start-Sleep -Seconds 2

if (-not (Test-Path $VenvPython)) {
    py -m venv (Join-Path $BackendRoot ".venv")
    & $VenvPython -m pip install -r (Join-Path $BackendRoot "requirements.txt") -q
    & $VenvPython -m pip install tifffile imagecodecs scikit-image scipy -q
}

$needsBuild = $RebuildFrontend -or -not (Test-Path (Join-Path $DistDir "index.html"))
if ($needsBuild) {
    Write-Host "Building frontend..."
    Push-Location $FrontendRoot
    npm install --silent
    npm run build
    Pop-Location
}

if (-not (Test-Path (Join-Path $DistDir "index.html"))) {
    throw "Frontend build missing. Run: cd web_app/frontend && npm run build"
}

$logFile = Join-Path $LogDir "backend.log"
$errFile = Join-Path $LogDir "backend.err.log"
$pidFile = Join-Path $LogDir "backend.pid"

$proc = Start-Process -FilePath $VenvPython `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010") `
    -WorkingDirectory $BackendRoot `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile

$proc.Id | Out-File -FilePath $pidFile -Encoding ascii

$health = $null
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 3
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/health" -TimeoutSec 5
        break
    } catch {
        if (-not $proc.HasExited) { continue }
        throw "Backend process exited. See $errFile"
    }
}

if ($health) {
    Write-Host "IPA app running (PID $($proc.Id))"
    Write-Host "  Local: http://localhost:8010"
    Write-Host "  LAN:   http://10.40.20.30:8010"
    Write-Host "  Logs:  $LogDir"
    Write-Host "  Status: $($health.status)"
} else {
    Write-Host "Backend started but health check failed. See $errFile"
    throw "Health check failed"
}
