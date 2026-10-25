# install.ps1 - install HB9CV Studio on Windows 10/11 for the current user
# Copyright (C) 2026 Emiliano Gonzalez - LU3VEA
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

<#
.SYNOPSIS
    Installs HB9CV Studio for the current user (no administrator rights needed).

.DESCRIPTION
    1. Finds Python 3.9 or newer with Tk (offers to install Python 3.12 with winget when it is missing).
    2. Copies the program to InstallDir (default %LOCALAPPDATA%\Programs\HB9CV-Studio).
    3. Creates a Python virtual environment there with customtkinter, matplotlib and numpy.
    4. Provides nec2c.exe in InstallDir\bin: a copy of -Nec2c, one already in the PATH, or one compiled
       from source with MSYS2 (offers to install MSYS2 with winget).
    5. Creates Start menu and desktop shortcuts and the hb9cv.cmd / hb9cv-studio.cmd launchers.
    Running it again updates an existing installation. uninstall.ps1 removes it.

.PARAMETER InstallDir
    Installation folder.
.PARAMETER Nec2c
    Path of an existing nec2c.exe (or of the folder that contains it) to copy into the installation.
.PARAMETER SkipNec2c
    Do not look for, copy or compile nec2c.
.PARAMETER NoShortcuts
    Do not create Start menu and desktop shortcuts.
.PARAMETER AddToPath
    Add the installation folder to the user PATH, so hb9cv and hb9cv-studio work in any terminal.
.PARAMETER Yes
    Do not ask for confirmation.

.EXAMPLE
    .\install.bat
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -AddToPath -Yes
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\HB9CV-Studio'),
    [string]$Nec2c = '',
    [switch]$SkipNec2c,
    [switch]$NoShortcuts,
    [switch]$AddToPath,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$Src = $PSScriptRoot

function Info([string]$m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn([string]$m) { Write-Host "WARNING: $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Confirm-Step([string]$q) {
    if ($Yes) { return $true }
    $a = Read-Host "$q [Y/n]"
    return ($a -eq '' -or $a -match '^[yYsS]')
}
function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
}

# Returns the full path of the first Python 3.9+ interpreter that has Tk, or $null.
function Find-Python {
    # Windows PowerShell 5.1 turns redirected stderr of a native program into an error; probing must not stop.
    $ErrorActionPreference = 'Continue'
    $candidates = @(
        @('py', '-3'),
        @('python'),
        @('python3')
    )
    foreach ($c in $candidates) {
        $exe = $c[0]
        $pre = @()
        if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $out = & $exe @pre -c "import sys, tkinter; print(sys.executable) if sys.version_info >= (3, 9) else None" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $out) {
            $p = ($out | Select-Object -Last 1).Trim()
            if ($p -and (Test-Path $p)) { return $p }
        }
    }
    foreach ($v in @('314', '313', '312', '311', '310', '39')) {
        $p = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"
        if (Test-Path $p) {
            & $p -c "import tkinter" 2>$null
            if ($LASTEXITCODE -eq 0) { return $p }
        }
    }
    return $null
}

function Find-Bash {
    foreach ($root in @('C:\msys64', (Join-Path $env:LOCALAPPDATA 'Programs\msys64'))) {
        $b = Join-Path $root 'usr\bin\bash.exe'
        if (Test-Path $b) { return $b }
    }
    return $null
}

# Compiles nec2c.exe with MSYS2 (MSYS environment) into $Dest (InstallDir\bin); returns $true on success.
function Build-Nec2c([string]$Dest) {
    $bash = Find-Bash
    if (-not $bash) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Warn 'MSYS2 is not installed and winget is not available.'
            Warn 'Install MSYS2 from https://www.msys2.org/ and run this installer again.'
            return $false
        }
        if (-not (Confirm-Step 'nec2c is not installed. Install MSYS2 (about 500 MB) to compile it?')) { return $false }
        Info 'Installing MSYS2 with winget'
        winget install -e --id MSYS2.MSYS2 --accept-package-agreements --accept-source-agreements
        $bash = Find-Bash
        if (-not $bash) { Warn 'MSYS2 was not found after the installation (expected in C:\msys64).'; return $false }
    }
    $env:MSYSTEM = 'MSYS'
    $env:CHERE_INVOKING = '1'
    Info 'Updating MSYS2 and installing gcc (the first run initialises MSYS2 and can take several minutes)'
    # A core update ends the first pacman process; the second run completes the update.
    & $bash -lc 'pacman -Syu --noconfirm' | Out-Host
    & $bash -lc 'pacman -Syu --noconfirm' | Out-Host
    & $bash -lc 'pacman -S --needed --noconfirm gcc tar curl git' | Out-Host
    if ($LASTEXITCODE -ne 0) { Warn 'pacman could not install gcc in MSYS2.'; return $false }
    Info 'Compiling nec2c'
    # The login shell starts in the current folder (CHERE_INVOKING), so relative paths avoid any quoting
    # of folder names with spaces between PowerShell and bash.
    Push-Location $InstallDir
    try {
        & $bash -lc 'bash tools/build_nec2c.sh bin' | Out-Host
        $ok = ($LASTEXITCODE -eq 0)
    } finally {
        Pop-Location
    }
    $exe = Join-Path $Dest 'nec2c.exe'
    if (-not ($ok -and (Test-Path $exe))) { return $false }
    & $exe -v | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function New-Shortcut([string]$Path, [string]$Target, [string]$Arguments, [string]$Icon, [string]$Description) {
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut($Path)
    $lnk.TargetPath = $Target
    $lnk.Arguments = $Arguments
    $lnk.WorkingDirectory = $InstallDir
    if ($Icon) { $lnk.IconLocation = $Icon }
    $lnk.Description = $Description
    $lnk.Save()
}

# ------------------------------------------------------------------ checks
if (-not (Test-Path (Join-Path $Src 'hb9cv.py')) -or -not (Test-Path (Join-Path $Src 'hb9cv_gui.py'))) {
    Fail 'Run install.ps1 from the HB9CV Studio folder (hb9cv.py not found).'
}
Info "Installing HB9CV Studio in $InstallDir"

# ------------------------------------------------------------------ Python
$py = Find-Python
if (-not $py) {
    Warn 'Python 3.9 or newer with Tk was not found.'
    if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Confirm-Step 'Install Python 3.12 now with winget?')) {
        winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
        Update-SessionPath
        $py = Find-Python
    }
    if (-not $py) {
        Fail ('Install Python 3 from https://www.python.org/downloads/ (tick "Add python.exe to PATH" and keep ' +
              '"tcl/tk and IDLE" selected), then run this installer again.')
    }
}
$pyver = & $py -c "import sys; print(sys.version.split()[0])"
Info "Python: $py ($pyver)"

# ------------------------------------------------------------------ program files
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$same = ((Resolve-Path $Src).Path.TrimEnd('\') -ieq (Resolve-Path $InstallDir).Path.TrimEnd('\'))
if (-not $same) {
    foreach ($f in @('hb9cv.py', 'hb9cv_gui.py', 'requirements.txt', 'LICENSE', 'README.md',
                     'install.ps1', 'install.bat', 'uninstall.ps1', 'uninstall.bat')) {
        $p = Join-Path $Src $f
        if (Test-Path $p) { Copy-Item $p -Destination $InstallDir -Force }
    }
    foreach ($d in @('assets', 'docs', 'tools')) {
        $p = Join-Path $Src $d
        if (Test-Path $p) { Copy-Item $p -Destination $InstallDir -Recurse -Force }
    }
}

# ------------------------------------------------------------------ virtual environment
Info 'Creating the Python environment and installing customtkinter, matplotlib and numpy'
$venvPy = Join-Path $InstallDir 'venv\Scripts\python.exe'
$venvPyw = Join-Path $InstallDir 'venv\Scripts\pythonw.exe'
if (-not (Test-Path $venvPy)) {
    & $py -m venv (Join-Path $InstallDir 'venv')
    if ($LASTEXITCODE -ne 0) { Fail 'Could not create the Python virtual environment.' }
}
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet --upgrade -r (Join-Path $InstallDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Fail 'pip could not install the Python packages (check the internet connection).' }

# ------------------------------------------------------------------ nec2c
$binDir = Join-Path $InstallDir 'bin'
$bundled = Join-Path $binDir 'nec2c.exe'
if (-not $SkipNec2c) {
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    if ($Nec2c) {
        $n = $Nec2c
        if (Test-Path $n -PathType Container) { $n = Join-Path $n 'nec2c.exe' }
        if (-not (Test-Path $n)) { Fail "nec2c.exe not found: $Nec2c" }
        $folder = Split-Path -Parent (Resolve-Path $n).Path
        Copy-Item $n -Destination $bundled -Force
        # MSYS2 / Cygwin builds need their runtime DLLs next to the executable.
        Get-ChildItem $folder -Filter '*.dll' |
            Where-Object { $_.Name -like 'msys-*.dll' -or $_.Name -like 'cyg*.dll' } |
            Copy-Item -Destination $binDir -Force
        Info "nec2c: copied $n"
    } elseif (Test-Path $bundled) {
        Info "nec2c: $bundled"
    } elseif (Get-Command nec2c.exe -ErrorAction SilentlyContinue) {
        Info ("nec2c: " + (Get-Command nec2c.exe).Source)
    } else {
        if (Build-Nec2c $binDir) {
            Info "nec2c: compiled into $bundled"
        } else {
            Warn 'nec2c is not installed: the program starts, but simulations will not run.'
            Warn 'Run this installer again, or use -Nec2c C:\path\to\nec2c.exe, or set its path in the nec2c field of the window.'
        }
    }
}

# ------------------------------------------------------------------ launchers
Set-Content -Path (Join-Path $InstallDir 'hb9cv.cmd') -Encoding ASCII -Value @(
    '@echo off',
    'rem hb9cv.py (command line), installed by install.ps1',
    '"%~dp0venv\Scripts\python.exe" "%~dp0hb9cv.py" %*'
)
Set-Content -Path (Join-Path $InstallDir 'hb9cv-studio.cmd') -Encoding ASCII -Value @(
    '@echo off',
    'rem HB9CV Studio (graphical interface), installed by install.ps1',
    'start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0hb9cv_gui.py" %*'
)

if ($AddToPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    $parts = @($userPath.Split(';') | Where-Object { $_ -ne '' })
    if ($parts -notcontains $InstallDir) {
        [Environment]::SetEnvironmentVariable('Path', ((@($parts) + $InstallDir) -join ';'), 'User')
        Info 'Added the installation folder to the user PATH (open a new terminal to use hb9cv and hb9cv-studio).'
    }
}

if (-not $NoShortcuts) {
    $icon = Join-Path $InstallDir 'assets\hb9cv-studio.ico'
    $guiArgs = '"' + (Join-Path $InstallDir 'hb9cv_gui.py') + '"'
    $menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'HB9CV Studio'
    New-Item -ItemType Directory -Force -Path $menu | Out-Null
    New-Shortcut (Join-Path $menu 'HB9CV Studio.lnk') $venvPyw $guiArgs $icon 'Design, simulate, optimize and match HB9CV antennas'
    New-Shortcut (Join-Path ([Environment]::GetFolderPath('Desktop')) 'HB9CV Studio.lnk') $venvPyw $guiArgs $icon 'HB9CV Studio'
    $manuals = @{
        'User manual (English).lnk'       = 'docs\manual\HB9CV_Studio_User_Manual_EN.pdf'
        'Manual de usuario (Espanol).lnk' = 'docs\manual\HB9CV_Studio_Manual_de_Usuario_ES.pdf'
        'Manuale utente (Italiano).lnk'   = 'docs\manual\HB9CV_Studio_Manuale_Utente_IT.pdf'
    }
    foreach ($k in $manuals.Keys) {
        $pdf = Join-Path $InstallDir $manuals[$k]
        if (Test-Path $pdf) { New-Shortcut (Join-Path $menu $k) $pdf '' '' 'HB9CV Studio manual' }
    }
    $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $unArgs = '-NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $InstallDir 'uninstall.ps1') + '"'
    New-Shortcut (Join-Path $menu 'Uninstall HB9CV Studio.lnk') $ps $unArgs '' 'Remove HB9CV Studio'
}

# ------------------------------------------------------------------ check
& $venvPy (Join-Path $InstallDir 'hb9cv.py') --help | Out-Null
if ($LASTEXITCODE -ne 0) { Fail 'hb9cv.py does not start.' }
& $venvPy -c "import customtkinter, matplotlib, numpy, tkinter"
if ($LASTEXITCODE -ne 0) { Fail 'The Python packages are not usable.' }

Write-Host ''
Info 'HB9CV Studio is installed.'
Write-Host '    Start it from the Start menu or the desktop shortcut "HB9CV Studio".'
Write-Host ('    Command line: "' + (Join-Path $InstallDir 'hb9cv.cmd') + '" --help')
Write-Host ('    Manuals:      ' + (Join-Path $InstallDir 'docs\manual'))
