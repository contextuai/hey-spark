# Build the Hey Spark Windows installer (spec §10 Phase 4, "Installer").
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
#
# 1. syncs the env with the desktop + openWakeWord extras and the build group
# 2. freezes the app with PyInstaller          -> build\dist\HeySpark\HeySpark.exe
# 3. packs it with Inno Setup 6 (if installed) -> build\installer\HeySpark-Setup-<version>.exe
#
# Inno Setup: winget install JRSoftware.InnoSetup  (or https://jrsoftware.org/isdl.php)
# Set SPARK_CONSOLE=1 before running to get a console build for debugging.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> uv sync (desktop, oww, kokoro, build)"
uv sync --extra desktop --extra oww --extra kokoro --group build --locked
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

$version = uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
if ($LASTEXITCODE -ne 0) { throw "could not read the version from pyproject.toml" }
$version = $version.Trim()
Write-Host "==> version $version"

Write-Host "==> PyInstaller"
uv run --group build pyinstaller packaging\hey_spark.spec --noconfirm --distpath build\dist --workpath build\work
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$candidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    $cmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($cmd) { $iscc = $cmd.Source }
}
if (-not $iscc) {
    Write-Warning "Inno Setup 6 not found; skipping the installer. The app is in build\dist\HeySpark."
    Write-Warning "Install it with: winget install JRSoftware.InnoSetup"
    exit 0
}

Write-Host "==> Inno Setup ($iscc)"
& $iscc "/DAppVersion=$version" packaging\windows\hey_spark.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Get-Item "build\installer\HeySpark-Setup-$version.exe" | Select-Object Name, @{n = "MB"; e = { [int]($_.Length / 1MB) } }
