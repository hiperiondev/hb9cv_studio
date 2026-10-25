#!/usr/bin/env bash
# build_nec2c.sh - download and compile nec2c from source (part of HB9CV Studio)
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
# Usage: build_nec2c.sh [--mingw] [DEST_DIR]
#
# Downloads the nec2c source (https://github.com/KJ7LNW/nec2c, GPL-3.0, tag NEC2C_VERSION, default v1.3.3),
# compiles it with gcc and copies the executable to DEST_DIR (default: the bin folder next to this tools
# folder, where hb9cv.py and hb9cv_gui.py look for it). Works on Linux and in the MSYS environment of MSYS2
# on Windows; there it also copies the MSYS runtime DLLs the executable needs, so nec2c.exe runs from a
# normal Windows program. Needs gcc, tar and curl, wget or git.
#
# --mingw: cross-compiles a self-contained Windows nec2c.exe (no runtime DLLs) on Linux with
# x86_64-w64-mingw32-gcc (Debian/Ubuntu/Mint: sudo apt install gcc-mingw-w64-x86-64; Fedora: sudo dnf install
# mingw64-gcc). The POSIX calls nec2c uses that MinGW lacks (times, sysconf, sigaction) come from the shim
# tools/mingw-compat/sys/times.h. This is how the nec2c.exe in the project folder is built.
set -euo pipefail

MINGW=no
if [ "${1:-}" = "--mingw" ]; then
    MINGW=yes
    shift
fi

VERSION="${NEC2C_VERSION:-v1.3.3}"
URL="https://github.com/KJ7LNW/nec2c/archive/refs/tags/${VERSION}.tar.gz"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/../bin}"
if command -v cygpath >/dev/null 2>&1; then
    DEST="$(cygpath -u "$DEST")"
fi
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"

die() { echo "build_nec2c: $*" >&2; exit 1; }

if [ "$MINGW" = yes ]; then
    CC=x86_64-w64-mingw32-gcc
    command -v "$CC" >/dev/null 2>&1 || die "$CC not found (Debian/Ubuntu/Mint: sudo apt install gcc-mingw-w64-x86-64; Fedora: sudo dnf install mingw64-gcc)"
    EXE=nec2c.exe
    CFLAGS_EXTRA="-static -Wl,--no-insert-timestamp -I$HERE/mingw-compat"
else
    CC=gcc
    command -v gcc >/dev/null 2>&1 || die "gcc not found (Debian/Ubuntu/Mint: sudo apt install gcc; Fedora: sudo dnf install gcc; MSYS2: pacman -S gcc)"
    EXE=nec2c
    CFLAGS_EXTRA=""
    case "$(uname -s)" in
        MSYS*|MINGW*|CYGWIN*) EXE=nec2c.exe ;;
    esac
fi
command -v tar >/dev/null 2>&1 || die "tar not found"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "build_nec2c: downloading nec2c $VERSION"
mkdir -p "$TMP/src"
got=no
if command -v curl >/dev/null 2>&1 && curl -fsL "$URL" -o "$TMP/nec2c.tar.gz"; then
    got=yes
elif command -v wget >/dev/null 2>&1 && wget -q "$URL" -O "$TMP/nec2c.tar.gz"; then
    got=yes
fi
if [ "$got" = yes ]; then
    tar -xzf "$TMP/nec2c.tar.gz" -C "$TMP/src" --strip-components=1
elif command -v git >/dev/null 2>&1; then
    rm -rf "$TMP/src"
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$VERSION" https://github.com/KJ7LNW/nec2c.git "$TMP/src" \
        || die "download failed (tried $URL and git clone)"
else
    die "download failed: $URL (install curl, wget or git)"
fi

# Same source list as the upstream Makefile.am; compiling directly avoids the autotools dependency.
SOURCES="calculations.c geometry.c input.c matrix.c network.c shared.c fields.c ground.c main.c misc.c radiation.c somnec.c"
cd "$TMP/src"
for f in $SOURCES; do
    [ -f "$f" ] || die "unexpected source layout: $f is missing"
done
echo "build_nec2c: compiling"
# shellcheck disable=SC2086
$CC -O2 $CFLAGS_EXTRA -DPACKAGE_STRING="\"nec2c ${VERSION#v}\"" -o "$EXE" $SOURCES -lm

mkdir -p "$DEST"
cp -f "$EXE" "$DEST/"
if [ "$MINGW" = yes ]; then
    echo "build_nec2c: cross-compiled $DEST/$EXE (Windows executable, not run here)"
    exit 0
fi
if [ "$EXE" = nec2c.exe ] && command -v ldd >/dev/null 2>&1; then
    # MSYS executables need the MSYS runtime (msys-2.0.dll and friends) next to them.
    ldd "$DEST/$EXE" | awk '$3 ~ /^\/usr\/bin\// {print $3}' | while read -r dll; do
        cp -f "$dll" "$DEST/"
    done
fi
"$DEST/$EXE" -v >/dev/null 2>&1 || die "the compiled nec2c does not run"
echo "build_nec2c: installed $("$DEST/$EXE" -v 2>&1 | head -n 1) in $DEST/$EXE"
