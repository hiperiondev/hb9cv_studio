#!/usr/bin/env bash
# uninstall.sh - remove HB9CV Studio installed by install.sh (Linux)
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
#
# Removes the program folder, the hb9cv-studio and hb9cv commands and the menu entry. The system packages
# (Python, Tk, nec2c) and your work folders are left untouched; --purge also removes the GUI settings file
# ~/.hb9cv_gui.json.
set -euo pipefail

APP_ID="hb9cv-studio"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
PREFIX="$DATA/$APP_ID"
PURGE=no
ASSUME_YES=no

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$(realpath -m "$2")"; shift 2 ;;
        --prefix=*) PREFIX="$(realpath -m "${1#*=}")"; shift ;;
        --purge) PURGE=yes; shift ;;
        -y|--yes) ASSUME_YES=yes; shift ;;
        -h|--help)
            echo "Usage: ./uninstall.sh [--prefix DIR] [--purge] [-y]"
            echo "  --purge  also remove the GUI settings (~/.hb9cv_gui.json)"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -f "$PREFIX/hb9cv_gui.py" ] || [ -e "$HOME/.local/bin/hb9cv-studio" ] || { echo "HB9CV Studio is not installed in $PREFIX"; exit 0; }
if [ "$ASSUME_YES" != yes ]; then
    read -r -p "Remove HB9CV Studio from $PREFIX? [y/N] " a || true
    case "$a" in [yYsS]*) ;; *) echo "Cancelled."; exit 1 ;; esac
fi

rm -f "$HOME/.local/bin/hb9cv-studio" "$HOME/.local/bin/hb9cv"
rm -f "$DATA/applications/$APP_ID.desktop" "$DATA/icons/hicolor/256x256/apps/$APP_ID.png"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$DATA/applications" || true
case "$PREFIX" in
    ""|"/"|"$HOME") echo "Refusing to remove $PREFIX" >&2; exit 1 ;;
esac
rm -rf "$PREFIX"
[ "$PURGE" = yes ] && rm -f "$HOME/.hb9cv_gui.json"
echo "HB9CV Studio has been removed. System packages (python3, Tk, nec2c) were kept."
