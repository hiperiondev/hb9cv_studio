@echo off
rem uninstall.bat - remove HB9CV Studio from Windows (runs uninstall.ps1)
rem Copyright (C) 2026 Emiliano Gonzalez - LU3VEA
rem SPDX-License-Identifier: GPL-3.0-or-later
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %*
echo.
pause
