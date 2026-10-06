# Runs the daily forward-test update and appends its output to logs\forward_update.log.
#
# Called by the scheduled task that scripts\register_forward_task.ps1 creates,
# but safe to run by hand too:
#   powershell -ExecutionPolicy Bypass -File scripts\run_forward_update.ps1
#
# Uses the project's own .venv Python directly rather than `uv run`, because
# Task Scheduler starts with a minimal environment where uv may not be on PATH.

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $projectRoot "logs"
$log = Join-Path $logDir "forward_update.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $projectRoot
$env:PYTHONIOENCODING = "utf-8"

"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" | Out-File -FilePath $log -Append -Encoding utf8

if (-not (Test-Path $python)) {
    "ERROR: $python not found. Run 'uv sync --all-extras' in the project first." | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}

& $python "scripts\forward_test.py" "update" *>&1 | Out-File -FilePath $log -Append -Encoding utf8
$code = $LASTEXITCODE
"exit code: $code" | Out-File -FilePath $log -Append -Encoding utf8
exit $code
