param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCommand) {
    Write-Host "Installing uv from its official distribution..."
    $Installer = Join-Path $env:TEMP "fpl-uv-install.ps1"
    Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile $Installer
    & powershell -NoProfile -ExecutionPolicy Bypass -File $Installer
    if ($LASTEXITCODE -ne 0) { throw "uv installation failed." }
    $UvPath = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path $UvPath)) { throw "Restart the terminal after installing uv, then run Start.cmd again." }
} else { $UvPath = $UvCommand.Source }

& $UvPath sync --frozen
if ($LASTEXITCODE -ne 0) { throw "Python dependencies could not be installed." }

if (-not (Test-Path ".env")) {
    $Settings = New-Object System.Collections.Generic.List[string]
    foreach ($Provider in @("GEMINI", "GROQ")) {
        $SecureValue = Read-Host "$Provider API key (Enter skips hosted research)" -AsSecureString
        $PlainValue = (New-Object System.Net.NetworkCredential("", $SecureValue)).Password
        if ($PlainValue) {
            if ($PlainValue -match "[\r\n]") { throw "API keys must contain a single line." }
            $Settings.Add("${Provider}_API_KEY=$PlainValue")
        }
        $PlainValue = $null
    }
    [IO.File]::WriteAllLines((Join-Path $ProjectRoot ".env"), $Settings, (New-Object Text.UTF8Encoding($false)))
    $Acl = Get-Acl ".env"
    $Acl.SetAccessRuleProtection($true, $false)
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Acl.SetAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($Identity,"FullControl","Allow")))
    Set-Acl ".env" $Acl
    $FreeAccounts = Read-Host "Are the entered keys for free, unbilled accounts? (Y/N)"
    if ($FreeAccounts -match "^[Yy]") {
        & $UvPath run fpl configure-providers --unbilled
        if ($LASTEXITCODE -ne 0) { throw "Free provider verification failed; credentials were not printed." }
    }
}

if (-not (Test-Path "apps/web/dist/index.html")) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "This source checkout needs Node 24 to build the interface. The prepared ZIP includes the built interface."
    }
    & npm.cmd --prefix apps/web ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    & npm.cmd --prefix apps/web run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
}

& $UvPath run python -c "from fpl_intelligence.storage import Store; s=Store(); raise SystemExit(0 if all(s.list(kind, 1) for kind in ('market', 'state', 'forecast', 'selected_plan')) else 1)"
$NeedsInitialisation = $LASTEXITCODE -ne 0
if ($NeedsInitialisation) {
    & $UvPath run fpl ingest
    if ($LASTEXITCODE -ne 0) { throw "Official data ingestion failed." }
    & $UvPath run python -c "from fpl_intelligence.storage import Store; raise SystemExit(0 if Store().list('state', 1) else 1)"
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path "data/account-confirmed.json") {
            & $UvPath run fpl reconcile-team --path data/account-confirmed.json --confirmed
        } else { & $UvPath run fpl reconcile-team }
        if ($LASTEXITCODE -ne 0) { throw "Account import needs current valid selling quotes." }
    }
    & $UvPath run fpl import-history --season 2025-26
    if ($LASTEXITCODE -ne 0) { throw "Historical import failed." }
    & $UvPath run fpl build-features
    if ($LASTEXITCODE -ne 0) { throw "Feature generation failed." }
    & $UvPath run fpl train
    if ($LASTEXITCODE -ne 0) { throw "Forecast evaluation failed." }
    & $UvPath run fpl train-components
    if ($LASTEXITCODE -ne 0) { throw "Component calibration failed." }
    & $UvPath run fpl forecast --end 38 --worlds 2000
    if ($LASTEXITCODE -ne 0) { throw "Forecast generation failed." }
    & $UvPath run fpl optimize --end 19 --seconds 60
    if ($LASTEXITCODE -ne 0) { throw "The initial planning pipeline failed; inspect the message above." }
}

$Port = 8000
while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { $Port++ }
$Url = "http://127.0.0.1:$Port"
Write-Host "FPL Intelligence: $Url"
Write-Host "Keep this window open while using the private application. Ctrl+C stops it."
if (-not $NoBrowser) {
    $BrowserJob = Start-Job -ArgumentList $Url -ScriptBlock {
        param($PageUrl)
        for ($Attempt=0; $Attempt -lt 60; $Attempt++) {
            try {
                $Health = Invoke-RestMethod "$PageUrl/api/health" -TimeoutSec 1
                if ($Health.status -eq "ok") { Start-Process $PageUrl; return }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
    }
}
& $UvPath run fpl serve --port $Port
