# Nexus 24/7 Scheduled Task Setup for Windows Task Scheduler
# This script sets up Nexus to run 24/7 using Windows Task Scheduler with auto-restart

# Run this script as Administrator:
# PowerShell -ExecutionPolicy Bypass -File setup_scheduled_task.ps1

param(
    [ValidateSet('install', 'remove', 'status')]
    [string]$Action = 'install'
)

$TaskName = "Nexus24x7TradingPlatform"
$TaskPath = "\Nexus\"
$FullTaskName = "$TaskPath$TaskName"
$ProjectDir = (Get-Location).Path
$PythonExe = (Get-Command python).Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command python3).Source
}
if (-not $PythonExe) {
    Write-Host "ERROR: Python executable not found in PATH" -ForegroundColor Red
    exit 1
}
$ScriptPath = Join-Path $ProjectDir "nexus_24_7.py"
$LogDir = Join-Path $ProjectDir "logs"

Write-Host "================================"
Write-Host "Nexus 24/7 Scheduled Task Setup"
Write-Host "================================"
Write-Host "Action: $Action"
Write-Host "Task Name: $FullTaskName"
Write-Host "Project Directory: $ProjectDir"
Write-Host ""

function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Host "ERROR: This script requires administrator privileges!" -ForegroundColor Red
    Write-Host "Please run as Administrator (right-click PowerShell and select 'Run as administrator')"
    exit 1
}

function Install-Task {
    Write-Host "Installing scheduled task..." -ForegroundColor Cyan

    # Create log directory if it doesn't exist
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        Write-Host "[OK] Created logs directory: $LogDir"
    }

    # Create a trigger that runs at system startup
    $trigger = New-ScheduledTaskTrigger -AtStartup

    # Create action that runs Python script
    $action = New-ScheduledTaskAction `
        -Execute $PythonExe `
        -Argument "`"$ScriptPath`"" `
        -WorkingDirectory $ProjectDir

    # Create settings for task
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -RestartCount 10 `
        -RestartInterval (New-TimeSpan -Minutes 5)

    # Create principal (run as SYSTEM with highest privileges)
    $principal = New-ScheduledTaskPrincipal `
        -UserId "NT AUTHORITY\SYSTEM" `
        -LogonType ServiceAccount `
        -RunLevel Highest

    # Check if task already exists
    $existingTask = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue

    if ($existingTask) {
        Write-Host "Task already exists. Removing old task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
        Start-Sleep -Seconds 2
    }

    # Create the scheduled task folder if it doesn't exist
    try {
        Get-ScheduledTask -TaskPath $TaskPath -ErrorAction Stop | Out-Null
    }
    catch {
        Write-Host "Creating task folder: $TaskPath"
        # Folder doesn't exist, we'll create the task in root
        $TaskPath = "\"
        $FullTaskName = "$TaskName"
    }

    # Register the task
    try {
        $task = Register-ScheduledTask `
            -TaskName $TaskName `
            -TaskPath $TaskPath `
            -Trigger $trigger `
            -Action $action `
            -Settings $settings `
            -Principal $principal `
            -Force

        Write-Host "[OK] Task installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Task Details:"
        Write-Host "  Name: $TaskName"
        Write-Host "  Path: $TaskPath"
        Write-Host "  Trigger: At System Startup"
        Write-Host "  Action: $PythonExe `"$ScriptPath`""
        Write-Host "  Principal: SYSTEM (Highest privileges)"
        Write-Host "  Auto-Restart: Yes (10 attempts, 5-minute intervals)"
        Write-Host ""
        Write-Host "To start the task immediately:"
        Write-Host "  Start-ScheduledTask -TaskName `"$TaskName`" -TaskPath `"$TaskPath`""
        Write-Host ""

        return $true
    }
    catch {
        Write-Host "x Failed to install task: $_" -ForegroundColor Red
        return $false
    }
}

function Remove-Task {
    Write-Host "Removing scheduled task..." -ForegroundColor Cyan

    $task = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue

    if (-not $task) {
        Write-Host "Task not found."
        return $false
    }

    try {
        Stop-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
        Write-Host "[OK] Task removed successfully!" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[FAIL] Failed to remove task: $_" -ForegroundColor Red
        return $false
    }
}

function Get-TaskStatus {
    Write-Host "Checking task status..." -ForegroundColor Cyan

    $task = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue

    if (-not $task) {
        Write-Host "[FAIL] Task not found"
        return $false
    }

    Write-Host "[OK] Task found:" -ForegroundColor Green
    Write-Host "  Name: $($task.TaskName)"
    Write-Host "  Path: $($task.TaskPath)"
    Write-Host "  State: $($task.State)"
    Write-Host "  Enabled: $($task.Settings.Enabled)"
    Write-Host ""

    # Get task info
    $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue

    if ($taskInfo) {
        Write-Host "  Last Run Time: $($taskInfo.LastRunTime)"
        Write-Host "  Last Task Result: $($taskInfo.LastTaskResult)"
        Write-Host "  Next Run Time: $($taskInfo.NextRunTime)"
        Write-Host "  Number of Missing Runs: $($taskInfo.NumberOfMissedRuns)"
    }

    return $true
}

# Execute the requested action
switch ($Action) {
    'install' {
        $success = Install-Task
    }
    'remove' {
        $success = Remove-Task
    }
    'status' {
        $success = Get-TaskStatus
    }
    default {
        Write-Host "Unknown action: $Action"
        $success = $false
    }
}

exit $(if ($success) { 0 } else { 1 })
