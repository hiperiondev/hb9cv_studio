/* sys/times.h - POSIX shims for building nec2c with MinGW-w64 (part of HB9CV Studio)
 * Copyright (C) 2026 Emiliano Gonzalez - LU3VEA
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * nec2c.h includes <signal.h>, <unistd.h> and then <sys/times.h>. MinGW-w64 has no sys/times.h, no times(),
 * no sysconf() and no sigaction(); this header provides the small subset nec2c uses: process time through
 * clock() and signal handlers registered with signal(). It is found first through "-I tools/mingw-compat".
 */
#ifndef HB9CV_MINGW_SYS_TIMES_H
#define HB9CV_MINGW_SYS_TIMES_H

#include <signal.h>
#include <time.h>

typedef long clock_t_tms;

struct tms {
    clock_t_tms tms_utime;
    clock_t_tms tms_stime;
    clock_t_tms tms_cutime;
    clock_t_tms tms_cstime;
};

static inline clock_t_tms times(struct tms *buf)
{
    clock_t c = clock();
    if (buf) {
        buf->tms_utime = (clock_t_tms)c;
        buf->tms_stime = 0;
        buf->tms_cutime = 0;
        buf->tms_cstime = 0;
    }
    return (clock_t_tms)c;
}

#ifndef _SC_CLK_TCK
#define _SC_CLK_TCK 2
#endif

static inline long hb9cv_sysconf(int name)
{
    return name == _SC_CLK_TCK ? (long)CLOCKS_PER_SEC : -1L;
}
#define sysconf(name) hb9cv_sysconf(name)

typedef int sigset_t_compat;

struct sigaction {
    void (*sa_handler)(int);
    sigset_t_compat sa_mask;
    int sa_flags;
};

static inline int sigemptyset(sigset_t_compat *set)
{
    *set = 0;
    return 0;
}

static inline int sigaction(int sig, const struct sigaction *act, struct sigaction *old)
{
    void (*prev)(int) = act ? signal(sig, act->sa_handler) : SIG_DFL;
    if (old) {
        old->sa_handler = prev;
        old->sa_mask = 0;
        old->sa_flags = 0;
    }
    return prev == SIG_ERR ? -1 : 0;
}

#endif
