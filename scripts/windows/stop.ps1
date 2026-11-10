# Stop rag-chat completely (the server and its child process) so no CPU, RAM or GPU memory stays in use.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$pidFile = Join-Path $root "data\server.pid"
$stopped = $false

# 1) The process started by start.ps1 (the venv launcher spawns a child, so stop the whole tree).
if (Test-Path $pidFile) {
    $serverPid = (Get-Content $pidFile | Select-Object -First 1).Trim()
    if ($serverPid) {
        & taskkill.exe /PID $serverPid /T /F 2>$null | Out-Null
        $stopped = $true
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

# 2) Anything else still serving rag-chat from this folder (e.g. started by hand).
$venv = (Join-Path $root ".venv").ToLower()
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {
    $_.CommandLine -and $_.CommandLine -like "*uvicorn*app.main:app*" -and
    ($_.ExecutablePath -and ($_.ExecutablePath.ToLower().StartsWith($venv) -or $_.CommandLine.ToLower().Contains($venv)))
} | ForEach-Object {
    & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null
    $stopped = $true
}

Start-Sleep -Seconds 2
$left = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like "*uvicorn*app.main:app*" }
if ($left) {
    Write-Host "Some rag-chat processes are still running: $($left.ProcessId -join ', ')" -ForegroundColor Red
    exit 1
}
if ($stopped) { Write-Host "rag-chat is stopped. No models are loaded any more (CPU, RAM and GPU memory released)." -ForegroundColor Green }
else { Write-Host "rag-chat was not running." -ForegroundColor Green }
