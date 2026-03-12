param(
  [string]$TaskName = "SupermixQwenTrainingAutoResume"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$autoResumePath = (Resolve-Path (Join-Path $PSScriptRoot "auto_resume_supermix_training.ps1")).Path
$taskArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$autoResumePath`""
$runCommand = "powershell.exe $taskArgs"
$registered = $false

try {
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $taskArgs
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

  Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force `
    -ErrorAction Stop | Out-Null

  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
  Write-Host "[login] registered scheduled task: name=$TaskName state=$($task.State)"
  $registered = $true
}
catch {
  Write-Host "[login] scheduled task unavailable, falling back to HKCU Run: $($_.Exception.Message)"
}

if (-not $registered) {
  $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
  New-Item -Path $runKey -Force | Out-Null
  Set-ItemProperty -Path $runKey -Name $TaskName -Value $runCommand -Type String
  $value = (Get-ItemProperty -Path $runKey -Name $TaskName).$TaskName
  Write-Host "[login] registered HKCU Run entry: name=$TaskName command=$value"
}
