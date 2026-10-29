param([string]$ExistingFolder)
$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
if (-not $ExistingFolder) {
    Add-Type -AssemblyName System.Windows.Forms
    $Picker = New-Object System.Windows.Forms.FolderBrowserDialog
    $Picker.Description = "Select your EXISTING FPL Intelligence folder containing Start.cmd. Your squad and keys will be kept."
    if ($Picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
    $ExistingFolder = $Picker.SelectedPath
}
$ExistingFolder = (Resolve-Path $ExistingFolder).Path
if ($ExistingFolder -eq $PackageRoot) { throw "Extract the update into a separate folder, then select your older installation." }
if (-not (Test-Path (Join-Path $ExistingFolder "Start.cmd"))) { throw "Select the existing application folder containing Start.cmd." }
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCommand) { throw "uv was not found. Open your existing application's terminal and run this update again." }
# Stop only servers whose own interpreter is inside the selected installation.
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and ($_.CommandLine.IndexOf((Join-Path $ExistingFolder ".venv"), [StringComparison]::OrdinalIgnoreCase) -ge 0) -and $_.CommandLine -match 'fpl(?:\.exe)?["\s]+serve'
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
& $UvCommand.Source run --project $PackageRoot python (Join-Path $PSScriptRoot "install_update.py") --target $ExistingFolder
if ($LASTEXITCODE -ne 0) { throw "Update could not be applied. Your account and keys were not replaced." }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ExistingFolder "scripts\start.ps1") -RebuildModel
