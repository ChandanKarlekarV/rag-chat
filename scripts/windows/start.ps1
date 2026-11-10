# Start rag-chat in the background (no window), wait until it is ready, then open the browser.
# Runs on its own: it does not depend on Claude Code or any editor. Stop it with stop.ps1.
param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
$dataDir = Join-Path $root "data"
$pidFile = Join-Path $dataDir "server.pid"
$logFile = Join-Path $dataDir "server.log"
$url = "http://127.0.0.1:8000"

function Test-Ready {
    try { return (Invoke-WebRequest "$url/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { return $false }
}

if (-not (Test-Path $python)) {
    Write-Host "Setup needed first: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
New-Item -ItemType Directory -Force $dataDir | Out-Null

if (Test-Ready) {
    Write-Host "rag-chat is already running." -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
}

Write-Host "Starting rag-chat (loading the models takes about 20-40 seconds)..."
$proc = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii

for ($i = 0; $i -lt 120; $i++) {
    if (Test-Ready) {
        $health = Invoke-RestMethod "$url/health"
        Write-Host "rag-chat is ready at $url (models on: $($health.device))." -ForegroundColor Green
        if (-not $NoBrowser) { Start-Process $url }
        exit 0
    }
    if ($proc.HasExited) {
        Write-Host "rag-chat stopped while starting. See $logFile.err" -ForegroundColor Red
        Remove-Item $pidFile -ErrorAction SilentlyContinue
        exit 1
    }
    Start-Sleep -Seconds 1
}
Write-Host "rag-chat is taking longer than expected. Check $logFile.err" -ForegroundColor Yellow
exit 1
