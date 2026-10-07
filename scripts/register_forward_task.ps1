# Registers a daily Windows scheduled task that runs the forward-test update.
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_forward_task.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_forward_task.ps1 -Time 21:00
#
# The task runs as you, only while you're logged in (no password stored).
# It starts at -Time and then repeats every 2 hours around the clock, so it
# runs soon after the computer is next on even if it was off or asleep at the
# scheduled time (a single daily run was missed when the PC was off at 08:30
# and Windows' catch-up didn't fire). Extra runs are harmless: with no new bars
# they change nothing, and each run uses 3 Twelve Data requests (about 36 a
# day, well under the free plan's 800). The update catches up on every missed bar.
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
# Repeat every 2 hours for 22 hours each day (the next day's start covers the rest).
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At $Time `
    -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Hours 22)).Repetition
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Fetches new gold, EUR/USD and USD/JPY bars and logs production-model predictions (scripts\forward_test.py update)." `
    -Force | Out-Null

Write-Host "Registered '$taskName' to run from $Time daily, then every 2 hours."
Write-Host "Output goes to logs\forward_update.log. Run it once now to check:"
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"