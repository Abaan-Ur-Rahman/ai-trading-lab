# Registers a daily Windows scheduled task that runs the forward-test update.
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_forward_task.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_forward_task.ps1 -Time 21:00
#
# The task runs as you, only while you're logged in (no password stored).
# If the computer was off or asleep at the scheduled time, it runs as soon as
# possible afterwards, and the update itself catches up on every missed bar.
# Re-running this script replaces the task with the new settings.
#
# To remove it:  Unregister-ScheduledTask -TaskName "AI Trading Lab - forward test update"

param([string]$Time = "08:30")

$taskName = "AI Trading Lab - forward test update"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_forward_update.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`"" `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Fetches new gold, EUR/USD and USD/JPY bars and logs production-model predictions (scripts\forward_test.py update)." `
    -Force | Out-Null

Write-Host "Registered '$taskName' to run daily at $Time."
Write-Host "Output goes to logs\forward_update.log. Run it once now to check:"
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
