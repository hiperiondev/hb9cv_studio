#!/usr/bin/env bash
# install.sh - install HB9CV Studio on Linux (Fedora, Debian, Ubuntu, Linux Mint and derivatives)
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
# Installs for the current user (no root needed except for the system packages, installed through sudo):
#   1. system packages: Python 3 with venv and Tk, and nec2c (dnf on Fedora, apt on Debian/Ubuntu/Mint);
#      when the distribution has no nec2c package, nec2c is compiled from source into PREFIX/bin
#   2. the program files in PREFIX (default ~/.local/share/hb9cv-studio)
#   3. a Python virtual environment PREFIX/venv with customtkinter, matplotlib and numpy
#   4. the commands hb9cv-studio (GUI) and hb9cv (command line) in ~/.local/bin
#   5. a menu entry "HB9CV Studio" with its icon
# Running it again updates an existing installation. uninstall.sh removes it.
set -euo pipefail

APP_ID="hb9cv-studio"
PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/$APP_ID"
BIN_DIR="$HOME/.local/bin"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_PACKAGES=yes
BUILD_NEC2C=auto
DESKTOP=yes
ASSUME_YES=no

usage() {
    cat <<EOF
Usage: ./install.sh [options]

Options:
  --prefix DIR       install the program in DIR (default: $PREFIX)
  --no-system-deps   do not install system packages (Python, Tk, nec2c) with dnf/apt
  --build-nec2c      always compile nec2c from source into PREFIX/bin
  --no-build-nec2c   never compile nec2c (use the system package or an existing nec2c)
  --no-desktop       do not create the menu entry
  -y, --yes          do not ask for confirmation
  -h, --help         show this help

Run it as your normal user (not with sudo); it calls sudo itself for the system packages.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$(realpath -m "$2")"; shift 2 ;;
        --prefix=*) PREFIX="$(realpath -m "${1#*=}")"; shift ;;
        --no-system-deps) SYSTEM_PACKAGES=no; shift ;;
        --build-nec2c) BUILD_NEC2C=yes; shift ;;
        --no-build-nec2c) BUILD_NEC2C=no; shift ;;
        --no-desktop) DESKTOP=no; shift ;;
        -y|--yes) ASSUME_YES=yes; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

info() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
confirm() {
    [ "$ASSUME_YES" = yes ] && return 0
    local a
    read -r -p "$1 [Y/n] " a || true
    case "$a" in ""|[yYsS]*) return 0 ;; *) return 1 ;; esac
}

[ -f "$SRC/hb9cv.py" ] && [ -f "$SRC/hb9cv_gui.py" ] || die "run install.sh from the HB9CV Studio folder (hb9cv.py not found)"
if [ "$(id -u)" -eq 0 ]; then
    warn "you are running install.sh as root: it will install for the root user only."
    warn "Run it as your normal user instead: ./install.sh (it uses sudo when needed)."
    confirm "Continue as root?" || exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 && SUDO="sudo"
fi

# ---------------------------------------------------------------- distribution
DISTRO=unknown
DISTRO_NAME="Linux"
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO_NAME="${PRETTY_NAME:-${NAME:-Linux}}"
    case " ${ID:-} ${ID_LIKE:-} " in
        *" fedora "*|*" rhel "*|*" centos "*) DISTRO=fedora ;;
        *" debian "*|*" ubuntu "*|*" linuxmint "*) DISTRO=debian ;;
    esac
fi
info "System: $DISTRO_NAME"

nec2c_on_system() { command -v nec2c >/dev/null 2>&1; }

if [ "$SYSTEM_PACKAGES" = yes ]; then
    case "$DISTRO" in
        fedora)
            PKGS="python3 python3-pip python3-tkinter"
            info "Installing system packages with dnf: $PKGS nec2c"
            confirm "Install them now (sudo may ask for your password)?" || die "cancelled"
            $SUDO dnf install -y $PKGS
            $SUDO dnf install -y nec2c || warn "the nec2c package could not be installed; it will be compiled from source"
            ;;
        debian)
            PKGS="python3 python3-venv python3-pip python3-tk"
            info "Installing system packages with apt: $PKGS nec2c"
            confirm "Install them now (sudo may ask for your password)?" || die "cancelled"
            $SUDO apt-get update
            $SUDO apt-get install -y $PKGS
            $SUDO apt-get install -y nec2c || warn "the nec2c package could not be installed; it will be compiled from source"
            ;;
        *)
            warn "unknown distribution: install Python 3 (3.9 or newer) with venv and Tk, and nec2c, with your package"
            warn "manager, then run ./install.sh --no-system-deps"
            ;;
    esac
fi

# ---------------------------------------------------------------- Python
# The first Python 3.9+ with Tk and venv: python3, then the versioned interpreters (a system where python3
# points to a newer build without Tk usually still has the distribution's own interpreter with Tk).
PY=""
for cand in python3 python3.14 python3.13 python3.12 python3.11 python3.10 python3.9; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c "import sys, tkinter, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        PY="$(command -v "$cand")"
        break
    fi
done
if [ -z "$PY" ]; then
    command -v python3 >/dev/null 2>&1 || die "python3 not found"
    python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" || die "Python 3.9 or newer is needed"
    python3 -c "import tkinter" 2>/dev/null \
        || die "Python has no Tk support (Fedora: sudo dnf install python3-tkinter; Debian/Ubuntu/Mint: sudo apt install python3-tk)"
    die "Python venv is missing (Debian/Ubuntu/Mint: sudo apt install python3-venv)"
fi
info "Python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

# ---------------------------------------------------------------- program files
info "Installing HB9CV Studio in $PREFIX"
mkdir -p "$PREFIX"
for f in hb9cv.py hb9cv_gui.py requirements.txt LICENSE README.md install.sh uninstall.sh; do
    [ -f "$SRC/$f" ] && cp -f "$SRC/$f" "$PREFIX/"
done
for d in assets docs tools; do
    if [ -d "$SRC/$d" ] && [ "$(realpath "$SRC/$d")" != "$(realpath -m "$PREFIX/$d")" ]; then
        mkdir -p "$PREFIX/$d"
        cp -rf "$SRC/$d/." "$PREFIX/$d/"
    fi
done
chmod +x "$PREFIX/install.sh" "$PREFIX/uninstall.sh" "$PREFIX/tools/build_nec2c.sh" 2>/dev/null || true

# ---------------------------------------------------------------- nec2c
if [ "$BUILD_NEC2C" = yes ] || { [ "$BUILD_NEC2C" = auto ] && ! nec2c_on_system && [ ! -x "$PREFIX/bin/nec2c" ]; }; then
    info "Compiling nec2c from source into $PREFIX/bin"
    if ! command -v gcc >/dev/null 2>&1 && [ "$SYSTEM_PACKAGES" = yes ]; then
        case "$DISTRO" in
            fedora) $SUDO dnf install -y gcc curl tar ;;
            debian) $SUDO apt-get install -y gcc curl tar ;;
        esac
    fi
    bash "$PREFIX/tools/build_nec2c.sh" "$PREFIX/bin" \
        || warn "nec2c could not be compiled: simulations will not run until nec2c is installed"
fi
if nec2c_on_system; then
    info "nec2c: $(command -v nec2c)"
elif [ -x "$PREFIX/bin/nec2c" ]; then
    info "nec2c: $PREFIX/bin/nec2c"
elif [ "$BUILD_NEC2C" = no ]; then
    warn "nec2c not found: install it, or run ./install.sh --build-nec2c"
fi

# ---------------------------------------------------------------- virtual environment
info "Creating the Python environment and installing customtkinter, matplotlib and numpy"
[ -x "$PREFIX/venv/bin/python" ] || "$PY" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade -r "$PREFIX/requirements.txt"

# ---------------------------------------------------------------- commands
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/hb9cv-studio" <<EOF
#!/bin/sh
# HB9CV Studio (graphical interface), installed by install.sh
exec "$PREFIX/venv/bin/python" "$PREFIX/hb9cv_gui.py" "\$@"
EOF
cat > "$BIN_DIR/hb9cv" <<EOF
#!/bin/sh
# hb9cv.py (command line), installed by install.sh
exec "$PREFIX/venv/bin/python" "$PREFIX/hb9cv.py" "\$@"
EOF
chmod +x "$BIN_DIR/hb9cv-studio" "$BIN_DIR/hb9cv"

# ---------------------------------------------------------------- menu entry
if [ "$DESKTOP" = yes ]; then
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
    mkdir -p "$APPS" "$ICONS"
    cp -f "$PREFIX/assets/hb9cv-studio.png" "$ICONS/$APP_ID.png"
    cat > "$APPS/$APP_ID.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=HB9CV Studio
GenericName=HB9CV antenna designer
Comment=Design, simulate, optimize and match HB9CV antennas with nec2c
Comment[es]=Diseña, simula, optimiza y adapta antenas HB9CV con nec2c
Comment[it]=Progetta, simula, ottimizza e adatta antenne HB9CV con nec2c
Exec=$BIN_DIR/hb9cv-studio
Icon=$APP_ID
Terminal=false
Categories=Science;Engineering;Network;HamRadio;
Keywords=antenna;NEC;nec2c;HB9CV;ham;radio;
StartupWMClass=Tk
EOF
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$APPS" || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 \
        && gtk-update-icon-cache -q -t "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" || true
fi

# ---------------------------------------------------------------- check
"$PREFIX/venv/bin/python" "$PREFIX/hb9cv.py" --help >/dev/null || die "hb9cv.py does not start"
"$PREFIX/venv/bin/python" -c "import customtkinter, matplotlib, numpy, tkinter" || die "the Python packages are not usable"

echo
info "HB9CV Studio is installed."
echo "    Start it from the applications menu (HB9CV Studio) or with:  hb9cv-studio"
echo "    Command line:                                                 hb9cv --help"
echo "    Manuals:                                                      $PREFIX/docs/manual/"
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not in your PATH: log out and in again, or run  $BIN_DIR/hb9cv-studio" ;;
esac
