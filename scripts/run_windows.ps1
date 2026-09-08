$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "REHAB AI V0.99 - Full Clinical Workflow Release Candidate" -ForegroundColor Cyan
Write-Host "Windows-first startup: Python 3.11+ + embedded Vue 3 clinical workstation" -ForegroundColor DarkCyan

function Test-PythonExecutable {
    param(
        [string]$Executable,
        [string[]]$PrefixArgs = @()
    )

    try {
        $versionText = & $Executable @PrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $versionText) { return $null }
        $parts = $versionText.Trim().Split('.')
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        if ($major -eq 3 -and $minor -ge 11) {
            return $versionText.Trim()
        }
    } catch {}
    return $null
}

function New-RehabVirtualEnv {
    $candidates = @()

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $candidates += [pscustomobject]@{ Exe = $pythonCmd.Source; Args = @(); Label = "python" }
    }

    $python3Cmd = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python3Cmd) {
        $candidates += [pscustomobject]@{ Exe = $python3Cmd.Source; Args = @(); Label = "python3" }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        # -3 selects the highest registered Python 3 runtime and avoids hard-coding 3.11.
        $candidates += [pscustomobject]@{ Exe = $pyCmd.Source; Args = @("-3"); Label = "py -3" }
        foreach ($minor in 14,13,12,11) {
            $candidates += [pscustomobject]@{ Exe = $pyCmd.Source; Args = @("-3.$minor"); Label = "py -3.$minor" }
        }
    }

    foreach ($candidate in $candidates) {
        $version = Test-PythonExecutable -Executable $candidate.Exe -PrefixArgs $candidate.Args
        if ($version) {
            Write-Host "Using Python $version via $($candidate.Label)" -ForegroundColor Green
            & $candidate.Exe @($candidate.Args) -m venv .venv
            if ($LASTEXITCODE -eq 0 -and (Test-Path ".venv\Scripts\python.exe")) {
                return
            }
            if (Test-Path ".venv") {
                Remove-Item -Recurse -Force ".venv"
            }
        }
    }

    throw @"
No working Python 3.11+ runtime was found.

REHAB AI supports Python 3.11 or newer on Windows.
Run these commands to inspect what Windows can see:
  python --version
  py -0p

If Python is installed but only the Microsoft Store alias is visible, disable the python.exe App Execution Alias or install Python from python.org, then rerun this script.
"@
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    New-RehabVirtualEnv
}

$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment creation failed: $python does not exist."
}

$venvVersion = & $python -c "import sys; print(sys.version.split()[0])"
Write-Host "Virtual environment Python: $venvVersion" -ForegroundColor Green

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

& $python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "REHAB AI dependency installation failed." }

$packageVersion = & $python -c "import importlib.metadata as m; print(m.version('rehab-ai'))"
$sourceVersion = & $python -c "import sys; sys.path.insert(0, 'src'); import rehab_ai; print(rehab_ai.__version__)"
if ($packageVersion.Trim() -ne $sourceVersion.Trim()) {
    Write-Host "Detected stale editable package metadata ($packageVersion vs source $sourceVersion). Rebuilding clean virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
    New-RehabVirtualEnv
    $python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
    & $python -m pip install --upgrade pip
    & $python -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Clean reinstall failed." }
    $packageVersion = & $python -c "import importlib.metadata as m; print(m.version('rehab-ai'))"
    if ($packageVersion.Trim() -ne $sourceVersion.Trim()) { throw "Package version remains inconsistent after clean reinstall." }
}
Write-Host "REHAB AI package version: $packageVersion" -ForegroundColor Green

& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Automated tests failed. Application will not start." }

& $python scripts\phase_apace_diagnostics.py
if ($LASTEXITCODE -ne 0) { throw "APACE diagnostics failed. Application will not start." }

& $python scripts\phase_v08_diagnostics.py
if ($LASTEXITCODE -ne 0) { throw "Computational-depth diagnostics failed. Application will not start." }

& $python scripts\phase_v09_diagnostics.py
if ($LASTEXITCODE -ne 0) { throw "Integrated adaptive-system diagnostics failed. Application will not start." }

& $python scripts\phase_v095_diagnostics.py
if ($LASTEXITCODE -ne 0) { throw "Full-platform diagnostics failed. Application will not start." }

& $python scripts\phase_v099_diagnostics.py
if ($LASTEXITCODE -ne 0) { throw "Full clinical-workflow diagnostics failed. Application will not start." }

$env:PYTHONPATH = (Join-Path (Get-Location) "src")
Start-Process "http://127.0.0.1:8010"
& $python -m uvicorn rehab_ai.web.app:app --host 127.0.0.1 --port 8010
