param(
  [string]$BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
  [string]$OutputDir = "artifacts\qwen_supermix_enhanced_v28_clean_eval_robust_ipo",
  [string]$Device = "auto",
  [string]$ResumeWarmStartDir = "artifacts\qwen_supermix_enhanced_v26_full",
  [string]$TrainOutLog = "train_qwen_supermix_v28_clean_eval_robust_ipo.out.log",
  [string]$TrainErrLog = "train_qwen_supermix_v28_clean_eval_robust_ipo.err.log",
  [switch]$NoMonitor
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Get-PythonCommand {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }
  return "python"
}

$outputPattern = [Regex]::Escape($OutputDir)
$outputLeafPattern = [Regex]::Escape((Split-Path $OutputDir -Leaf))
$trainingProcs = @(
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -and
      $_.Name -match "python|pythonw|powershell|cmd" -and
      ($_.CommandLine -match "qwen_supermix_pipeline.py" -or $_.CommandLine -match "run_train_qwen_supermix_v26_full.ps1") -and
      ($_.CommandLine -match $outputPattern -or $_.CommandLine -match $outputLeafPattern)
    }
)
if ($trainingProcs.Count -gt 0) {
  $running = ($trainingProcs | Select-Object -ExpandProperty ProcessId) -join ", "
  Write-Host "[auto-resume] training already running for $OutputDir (pid=$running)"
}
else {
  $trainOutPath = Join-Path $repoRoot $TrainOutLog
  $trainErrPath = Join-Path $repoRoot $TrainErrLog
  $launcherPath = Join-Path $repoRoot "source\run_train_qwen_supermix_v26_full.ps1"
  $trainCommand = (
    'powershell.exe -NoProfile -ExecutionPolicy Bypass ' +
    '-File "{0}" -BaseModel "{1}" -OutputDir "{2}" -Device "{3}" -ResumeWarmStartDir "{4}" ' +
    '1>>"{5}" 2>>"{6}"'
  ) -f $launcherPath, $BaseModel, $OutputDir, $Device, $ResumeWarmStartDir, $trainOutPath, $trainErrPath

  Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $trainCommand) -WorkingDirectory $repoRoot -WindowStyle Minimized | Out-Null
  Write-Host "[auto-resume] launched training for $OutputDir"
}

if (-not $NoMonitor) {
  $monitorNeedle = [Regex]::Escape("training_monitor_gui.py")
  $monitorProcs = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object {
        $_.CommandLine -and
        $_.Name -match "python|pythonw" -and
        $_.CommandLine -match $monitorNeedle
      }
  )
  if ($monitorProcs.Count -gt 0) {
    $running = ($monitorProcs | Select-Object -ExpandProperty ProcessId) -join ", "
    Write-Host "[auto-resume] monitor already running (pid=$running)"
  }
  else {
    Start-Process -FilePath (Get-PythonCommand) -ArgumentList @("source\training_monitor_gui.py", "--root", ".") -WorkingDirectory $repoRoot -WindowStyle Minimized | Out-Null
    Write-Host "[auto-resume] launched training monitor"
  }
}
