@echo off
rem install.bat - install HB9CV Studio on Windows (runs install.ps1)
rem Copyright (C) 2026 Emiliano Gonzalez - LU3VEA
rem SPDX-License-Identifier: GPL-3.0-or-later
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
