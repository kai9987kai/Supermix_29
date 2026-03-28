param(
  [string]$InstallerScript = "installer\SupermixStudioDesktop.iss",
  [string]$Version = "",
  [string]$SetupBaseName = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$CommonIsccPaths = @(
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
  "C:\Program Files\Inno Setup 6\ISCC.exe",
  "C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
  (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
)
$IsccPath = $CommonIsccPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $IsccPath) {
  throw "Inno Setup compiler not found. Install Inno Setup 6, then rerun this script."
}

$IsccArgs = @()
if ($Version) {
  $IsccArgs += "/DMyAppVersion=$Version"
}
if ($SetupBaseName) {
  $IsccArgs += "/DMySetupBaseName=$SetupBaseName"
}
$IsccArgs += $InstallerScript

& $IsccPath @IsccArgs
if ($LASTEXITCODE -ne 0) {
  throw "Inno Setup build failed."
}
