# uninstall.ps1 - remove HB9CV Studio installed by install.ps1 (Windows)
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
    Removes HB9CV Studio: the installation folder, the shortcuts and the user PATH entry.
.DESCRIPTION
    Python, MSYS2 and your work folders are left untouched. -Purge also removes the GUI settings file
    %USERPROFILE%\.hb9cv_gui.json.
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\HB9CV-Studio'),
    [switch]$Purge,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'

function Set-UserPathEntry([string]$Dir, [bool]$Present) {
    # Adds or removes $Dir in the user PATH. The raw registry value is read and written back as REG_EXPAND_SZ, so
    # entries such as %USERPROFILE%\bin keep their variables; nothing is written when the value does not change.
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    try {
        $raw = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $parts = @($raw.Split(';') | Where-Object { $_ -ne '' -and $_ -ne $Dir })
        if ($Present) { $parts = @($parts) + $Dir }
        $new = ($parts -join ';')
        if ($new -eq $raw) { return $false }
        $key.SetValue('Path', $new, [Microsoft.Win32.RegistryValueKind]::ExpandString)
    } finally {
        $key.Close()
    }
    # Setting and clearing a user variable through .NET broadcasts WM_SETTINGCHANGE, so new terminals see the PATH.
    [Environment]::SetEnvironmentVariable('HB9CV_PATH_REFRESH', '1', 'User')
    [Environment]::SetEnvironmentVariable('HB9CV_PATH_REFRESH', $null, 'User')
    return $true
}

if (-not (Test-Path (Join-Path $InstallDir 'hb9cv_gui.py'))) {
    Write-Host "HB9CV Studio is not installed in $InstallDir"
    exit 0
}
if (-not $Yes) {
    $a = Read-Host "Remove HB9CV Studio from $InstallDir? [y/N]"
    if ($a -notmatch '^[yYsS]') { Write-Host 'Cancelled.'; exit 1 }
}

$menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'HB9CV Studio'
if (Test-Path $menu) { Remove-Item $menu -Recurse -Force }
$desk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'HB9CV Studio.lnk'
if (Test-Path $desk) { Remove-Item $desk -Force }

if (Set-UserPathEntry $InstallDir $false) {
    Write-Host 'Removed the installation folder from the user PATH.'
}

# This script may live inside the folder it removes: leave the folder first.
Set-Location $env:TEMP
Remove-Item $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path $InstallDir) {
    Write-Host "Some files are in use and were not removed; close HB9CV Studio and delete $InstallDir" -ForegroundColor Yellow
}
if ($Purge) {
    $cfg = Join-Path $env:USERPROFILE '.hb9cv_gui.json'
    if (Test-Path $cfg) { Remove-Item $cfg -Force }
}
Write-Host 'HB9CV Studio has been removed. Python and MSYS2 were kept.'
