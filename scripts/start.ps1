param([switch]$NoBrowser, [switch]$ConfigureProviders, [switch]$RebuildModel)
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

if ((Test-Path "data/state.sqlite.gz") -and -not (Test-Path "data/state.sqlite")) {
    & $UvPath run python -c "import gzip,shutil; from pathlib import Path; p=Path('data/state.sqlite'); t=p.with_suffix('.restoring'); a=gzip.open('data/state.sqlite.gz','rb'); b=t.open('wb'); shutil.copyfileobj(a,b); a.close(); b.close(); t.replace(p)"
    if ($LASTEXITCODE -ne 0) { throw "The packaged database could not be restored." }
}

if ($ConfigureProviders -or -not (Test-Path ".env")) {
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
        if ($LASTEXITCODE -ne 0) { Write-Warning "Hosted provider verification could not finish. Numerical forecasts remain available; run Configure keys.cmd to retry." }
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

& $UvPath run python -c "from fpl_intelligence.storage import Store; s=Store(); raise SystemExit(0 if all(s.list(kind, 1) for kind in ('market', 'state', 'forecast', 'selected_plan')) and s.get('forecast').get('model_version') == 'learned-points-v1' else 1)"
$NeedsInitialisation = ($LASTEXITCODE -ne 0) -or $RebuildModel
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
    & $UvPath run fpl train
    if ($LASTEXITCODE -ne 0) { throw "Chronological decision training failed." }
    & $UvPath run fpl forecast --end 38 --worlds 4000
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
