# Install Claude Agent Server as a Windows Scheduled Task (auto-start on boot).
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

# Build command. Quote each path independently so spaces in either pythonw or
# server path don't break tokenization (schtasks /tr passes the string to cmd.exe).
$Arguments = "`"$ServerPath`" --host $BindHost --port $Port"
$TaskRun = "`"$pythonw`" $Arguments"

Write-Host "Creating scheduled task: $FullName"
Write-Host "  Exe:  $pythonw"
Write-Host "  Args: $Arguments"
Write-Host "  /tr:  $TaskRun"
Write-Host ""

schtasks /create /tn $FullName /rl highest /tr $TaskRun /sc onstart /f
Set-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Settings $(
    New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit 0 -DisallowHardTerminate
)

$user = "$env:USERDOMAIN\$env:USERNAME"
Write-Host "Enter password for $user to allow the task to run when you are not logged in:"
$cred = Get-Credential -Credential $user
# Set-ScheduledTask -Password requires a plain string; the SecureString → plain
# conversion is unavoidable (cmdlet API limitation). Mitigations:
#   1. Get-Credential prompt — password never enters PS history.
#   2. Plain string is inline, not stored in a named $password variable.
#   3. $cred is nulled immediately after use to release the SecureString sooner.
Set-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName `
    -User $user -Password $cred.GetNetworkCredential().Password
$cred = $null
[System.GC]::Collect()

Write-Host ""
Write-Host "Done. Task will start automatically on next boot." -ForegroundColor Green
Write-Host "Start it now with: schtasks /run /tn $FullName"
