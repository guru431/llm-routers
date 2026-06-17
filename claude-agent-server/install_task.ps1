# Install Claude Agent Server as a Windows Scheduled Task (auto-start on boot).
#
# ⚠️ If a central task-registry/syncer manages this scheduled task on your machine,
#    do NOT register it by hand here: re-registering drifts from the registry and the
#    next sync reverts it. Change the schedule/delay/restart in your registry and
#    re-sync instead. Use this script only for a STANDALONE deploy with no such
#    central task management.
#
# Usage:
#   .\install_task.ps1                                  # interactive (asks for password)
#   .\install_task.ps1 -ServerPath '<full UNC path>'    # override server.py location
#   .\install_task.ps1 -BindHost 127.0.0.1              # loopback-only (default 0.0.0.0 for LAN)
#   .\install_task.ps1 -Uninstall                       # remove the task
#
# SECURITY: default bind is 0.0.0.0 (LAN-exposed). The server REQUIRES
# CLAUDE_AGENT_TOKEN in its environment and refuses to start without it.
# Set it before installing the task, e.g. Machine-scope:
#   [Environment]::SetEnvironmentVariable('CLAUDE_AGENT_TOKEN','<token>','Machine')
# Machine-scope is recommended so the scheduled task picks it up at boot,
# independent of any user logon. After setting, restart the task:
#   schtasks /end /tn \claude_agent_server
#   schtasks /run /tn \claude_agent_server

param(
    [switch]$Uninstall,
    [string]$ServerPath,
    [string]$BindHost = '0.0.0.0',
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$TaskPath = '\'
$TaskName = 'claude_agent_server'
$FullName = "$TaskPath$TaskName"

if ($Uninstall) {
    schtasks /delete /tn $FullName /f
    Write-Host "Removed task $FullName" -ForegroundColor Green
    return
}

# Discover pythonw.exe (preferred — no console window)
$pythonw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
if (-not $pythonw) {
    $pythonw = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    if (-not $pythonw) {
        Write-Error "Neither pythonw.exe nor python.exe found in PATH. Install Python 3.10+."
        exit 1
    }
    Write-Warning "pythonw.exe not found, falling back to python.exe (will show a console window)"
}

# Resolve server.py path.
# NOTE: Task Scheduler runs at boot, before any mapped network drive (e.g. S:\) is
# mounted. If server.py lives on a mapped drive, pass an explicit UNC path via
# -ServerPath (e.g. \\server\share\...\server.py). By default we use this script's
# own folder, which works when the project sits on a local drive.
if (-not $ServerPath) {
    $ServerPath = Join-Path $PSScriptRoot 'server.py'
}

if (-not (Test-Path $ServerPath)) {
    Write-Error "server.py not found at $ServerPath"
    exit 1
}

# Refuse a mapped network drive (e.g. S:\): a boot task runs in session 0 before
# logon, when such mappings don't exist — the task would silently fail to find the
# script (observed: python exits 2, server never starts after reboot). Require a
# UNC (\\server\share\...) or local path instead.
$root = [System.IO.Path]::GetPathRoot($ServerPath)
if ($root -match '^[A-Za-z]:\\$') {
    $drive = Get-PSDrive -Name $root.Substring(0,1) -ErrorAction SilentlyContinue
    if ($drive -and $drive.DisplayRoot -like '\\*') {
        Write-Error ("ServerPath is on mapped network drive $root ($($drive.DisplayRoot)). " +
            "A boot task runs before logon when this mapping is absent and would fail to start. " +
            "Pass -ServerPath with a UNC path, e.g. '$($drive.DisplayRoot)\...\server.py', or a local path.")
        exit 1
    }
}

# Build the action argument. Quote the script path so spaces don't break it.
$Arguments = "`"$ServerPath`" --host $BindHost --port $Port"

Write-Host "Creating scheduled task: $FullName"
Write-Host "  Exe:  $pythonw"
Write-Host "  Args: $Arguments"
Write-Host ""

# Boot trigger with a delay + restart-on-failure. Fixes the cold-boot race where
# the BootTrigger fires before the network share holding server.py / .env is
# mounted (observed: task fired 9s after boot, UNC not ready, python exited 2, so
# the server never came up after a reboot). The delay lets the network mount;
# RestartCount retries if it is still not ready. MultipleInstances=IgnoreNew plus
# the server's own single-instance guard prevent duplicate listeners on the port.
$action  = New-ScheduledTaskAction -Execute $pythonw -Argument $Arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = 'PT1M'
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -DisallowHardTerminate -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
$settings.ExecutionTimeLimit = 'PT0S'   # no time limit (long-running server)
$settings.Hidden = $true

$user = "$env:USERDOMAIN\$env:USERNAME"
Write-Host "Enter password for $user — LogonType=Password is required so the task has"
Write-Host "network credentials to reach the UNC share at boot (and runs before logon):"
# Register-ScheduledTask -Password requires a plain string; the SecureString →
# plain conversion is unavoidable (cmdlet API limitation). The Get-Credential
# prompt keeps the password out of PS history; $cred is nulled right after use.
$cred = Get-Credential -Credential $user
Register-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -User $user -Password $cred.GetNetworkCredential().Password `
    -Force | Out-Null
$cred = $null
[System.GC]::Collect()

Write-Host ""
Write-Host "Done. Task will start automatically on next boot." -ForegroundColor Green
Write-Host "Start it now with: schtasks /run /tn $FullName"
