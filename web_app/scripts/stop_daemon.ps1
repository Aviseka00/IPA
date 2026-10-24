# Stop IPA background server.
$WebAppRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $WebAppRoot "logs"
$PidFile = Join-Path $LogDir "backend.pid"

if (Test-Path $PidFile) {
    $backendPid = [int](Get-Content $PidFile -Raw)
    Stop-Process -Id $backendPid -Force -ErrorAction SilentlyContinue
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

foreach ($port in 8010, 5173) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

Write-Host "IPA app stopped."
