# ==============================================================================
# package_windows.ps1
# Packages Windows audio plugin (VST3) and Standalone application into a 
# self-contained, distributable .zip release.
# ==============================================================================

[CmdletBinding()]
param (
    [string]$BuildDir,
    [string]$OutputDir,
    [string]$Version,
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path "$ScriptDir\..").Path

if (-not $BuildDir)  { $BuildDir  = "$RepoRoot\build" }
if (-not $OutputDir) { $OutputDir = "$RepoRoot\dist" }

# Determine Version
if (-not $Version) {
    try {
        $gitVersion = git -C $RepoRoot describe --tags --always 2>$null
        if ($gitVersion) {
            $Version = $gitVersion.TrimStart('v')
        }
    } catch {
        # Git failed, fallback
    }
    if (-not $Version) {
        $Version = "1.6.0"
    }
}

$JuceBuildDir = "$BuildDir\frontend\juce"

# Auto-detect Config (Release vs Debug) for local testing, but strictly enforce on CI
$Vst3Src = "$JuceBuildDir\nn_bending_plugin_artefacts\$Config\VST3\nn~ Bending.vst3"
if (-not (Test-Path $Vst3Src)) {
    if (-not $env:CI -and $Config -eq "Release" -and (Test-Path "$JuceBuildDir\nn_bending_plugin_artefacts\Debug\VST3\nn~ Bending.vst3")) {
        $Config = "Debug"
        Write-Warning "Release build not found; falling back to Debug configuration for local packaging."
        $Vst3Src = "$JuceBuildDir\nn_bending_plugin_artefacts\$Config\VST3\nn~ Bending.vst3"
    }
}

$StandaloneSrc  = "$JuceBuildDir\nn_bending_standalone_artefacts\$Config\Standalone\nn~ Bending.exe"
$TorchLibDir    = "$RepoRoot\libtorch\lib"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Packaging nn~ Bending Windows Release" -ForegroundColor Cyan
Write-Host " Version:       $Version"
Write-Host " Configuration: $Config"
Write-Host " Build Dir:     $BuildDir"
Write-Host " Output Dir:    $OutputDir"
Write-Host "=================================================================="

# Verify build artifacts exist
if (-not (Test-Path $Vst3Src)) {
    throw "Error: Missing VST3 build artifact: $Vst3Src`nRun 'cmake --build build --config Release --target plugin' first."
}

$StageDir = "$BuildDir\installer_stage_win\nn_bending_Windows_x64_$Version"

if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
New-Item -ItemType Directory -Path "$StageDir\VST3" -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# 1. Package VST3
Write-Host "==> Staging VST3 plugin..." -ForegroundColor Yellow
Copy-Item -Recurse -Force $Vst3Src "$StageDir\VST3\nn~ Bending.vst3"

# Ensure all LibTorch DLLs are inside the VST3 bundle (Contents\x86_64-win)
$Vst3DllDir = "$StageDir\VST3\nn~ Bending.vst3\Contents\x86_64-win"
if (Test-Path $TorchLibDir) {
    Get-ChildItem -Path $TorchLibDir -Filter "*.dll" | ForEach-Object {
        Copy-Item -Force $_.FullName $Vst3DllDir
    }
}

# 2. Package Standalone App (Optional)
$hasStandalone = Test-Path $StandaloneSrc
if ($hasStandalone) {
    New-Item -ItemType Directory -Path "$StageDir\Standalone" -Force | Out-Null
    Write-Host "==> Staging Standalone application..." -ForegroundColor Yellow
    Copy-Item -Force $StandaloneSrc "$StageDir\Standalone\nn~ Bending.exe"
    
    # Copy LibTorch DLLs next to Standalone executable
    if (Test-Path $TorchLibDir) {
        Get-ChildItem -Path $TorchLibDir -Filter "*.dll" | ForEach-Object {
            Copy-Item -Force $_.FullName "$StageDir\Standalone"
        }
    }
} else {
    Write-Host "==> Standalone executable not found (packaging VST3 plugin only)." -ForegroundColor Cyan
}

# 3. Create Installation Instructions
$InstallDoc = @"
nn~ Bending - Windows Installation
==================================
Version: $Version

1. VST3 Plugin:
   Copy the "nn~ Bending.vst3" folder into your VST3 plug-ins directory:
     C:\Program Files\Common Files\VST3\
   (or into your DAW's custom VST3 search folder).

   All required LibTorch and PyTorch C++ runtime libraries are pre-bundled
   inside the VST3 folder (Contents\x86_64-win).
"@

if ($hasStandalone) {
    $InstallDoc += @"


2. Standalone Application:
   Run "nn~ Bending.exe" directly from the Standalone folder.
   Keep the accompanying .dll files in the same folder as the executable.
"@
}

$InstallDoc += "`n`nEnjoy!`n"
Set-Content -Path "$StageDir\README.txt" -Value $InstallDoc -Encoding UTF8

# 4. Create ZIP Archive
$ZipFileName = "nn_bending_Windows_x64_$Version.zip"
$ZipPath     = "$OutputDir\$ZipFileName"

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}

Write-Host "==> Creating ZIP archive: $ZipPath..." -ForegroundColor Yellow
Compress-Archive -Path "$StageDir\*" -DestinationPath $ZipPath -Force

# 5. Build Inno Setup Installer (.exe) if ISCC is available
$isccCandidates = @(
    "iscc.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
)
$isccPath = $null
foreach ($cand in $isccCandidates) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
        $isccPath = $cand
        break
    } elseif (Test-Path $cand -ErrorAction SilentlyContinue) {
        $isccPath = $cand
        break
    }
}

if (-not $isccPath) {
    try {
        $regItem = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup*" -ErrorAction SilentlyContinue
        if ($regItem.InstallLocation -and (Test-Path "$($regItem.InstallLocation)\ISCC.exe")) {
            $isccPath = "$($regItem.InstallLocation)\ISCC.exe"
        }
    } catch {}
}

$IssScript = "$ScriptDir\installer_windows.iss"
if ($isccPath -and (Test-Path $IssScript)) {
    $hasStandaloneInt = if ($hasStandalone) { 1 } else { 0 }
    Write-Host "==> Compiling Windows installer (.exe) with Inno Setup..." -ForegroundColor Yellow
    & $isccPath "/DMyAppVersion=$Version" "/DHasStandalone=$hasStandaloneInt" $IssScript
    $InstallerExe = "$OutputDir\nn_bending_Windows_x64_${Version}_setup.exe"
    if (Test-Path $InstallerExe) {
        Write-Host "==> Built Installer: $InstallerExe" -ForegroundColor Green
    }
} else {
    Write-Warning "Inno Setup compiler (ISCC.exe) not found. Skipping .exe installer generation."
    Write-Host "    (To enable .exe installer generation, install Inno Setup or run: choco install innosetup)" -ForegroundColor Gray
}

Write-Host "==================================================================" -ForegroundColor Green
Write-Host " Packaging Complete!" -ForegroundColor Green
Write-Host " Output Directory: $OutputDir" -ForegroundColor Green
Write-Host "=================================================================="
