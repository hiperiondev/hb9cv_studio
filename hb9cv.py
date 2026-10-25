#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HB9CV toolkit for NEC2C: model generator, optimizer, L-network matching and sweep.

One program, four modes selected with --mode:

  generate  Write an exploratory NEC2C model of an HB9CV (optionally run nec2c).
  optimize  Grid search plus local refinement and a Nelder-Mead polish over the model
            parameters, running nec2c for every candidate. Requires nec2c (--nec2c PATH, the
            HB9CV_NEC2C environment variable or the PATH).
  match     Compute an ideal L-network that matches the NEC input impedance (or a
            manually given R + jX) to a Z0 source at a single frequency.
  sweep     Frequency sweep of an L-network designed at one frequency, using the
            Zin(f) of a NEC frequency sweep (or a constant load impedance).

Typical workflow:
  1. hb9cv.py --mode optimize --network tl          (find a good geometry)
  2. hb9cv.py --mode generate --network tl --fsweep 140,150,0.5 --run
     (add --fsweep-pattern for gain / F/B vs frequency and a CSV summary besides Zin)
  3. hb9cv.py --mode match --nec-out hb9cv_145MHz_tl_isolated.out --freq 145
  4. hb9cv.py --mode sweep --nec-out hb9cv_145MHz_tl_isolated.out
  nec2c prints frequencies with five significant digits; --freq / --design-freq are matched within that
  resolution, and the exact frequencies are taken from the <name>.freqs.json file that generate --run writes
  next to the .out file.

Ground, polarization and pattern (generate / optimize):
- Default --ground free keeps the free-space model and the single azimuth ring at theta=90.
- --ground perfect|average|good with --height-m/--height-lambda and --orientation horizontal|vertical
  puts the antenna over ground (GE 1 + GN). The horizon ring is meaningless there (horizontal
  polarization has no radiation at the horizon over ground), so the pattern is taken in two steps: an
  elevation cut (RP theta 0..90 at phi=0) gives the take-off angle, then the azimuth ring is taken at that
  angle. Reported: gain in dBi (and dBd in free space only: over ground the gain includes the ground
  reflection), take-off angle, azimuth half-power beamwidth, F/B at the take-off angle.
- Gain is the NEC power gain referred to the power accepted at the input: it includes conductor loss and
  excludes the source mismatch; the realized gain adds 10*log10(1 - |Gamma|^2) against --target-impedance.
- F/B: point (phi=180 on the ring), sector (worst point of --rear-sector on the ring) and cone (worst direction
  within the sector half-angle of the back direction at ANY elevation, from a 5 degree rear half-sphere grid).
  optimize --fb-metric chooses the one the score uses (default sector).
- optimize --match-score (default with tl/lc) scores every candidate through the ideal L-network designed at
  --freq against --target-impedance with components of Q --match-q-l/--match-q-c: matched VSWR over the band, gain
  after the network loss and mismatch; the front-back objective keeps the pure F/B and pays --loss-weight dB of
  score per dB of network loss (an efficiency penalty). The Zin window then defaults to
  R 2..500 ohm and |X| <= 500 ohm. --no-match-score scores the raw Zin (direct connection: R 8..80, |X| <= 30).
  When no candidate meets the Zin window at every band frequency, the closest one is written and flagged
  (--strict-zin stops with an error instead).
  --refine-diameter steps through the purchasable --diameter-values (default 6,8,10,12,14,16 mm; 'any' =
  continuous within --diameter-range).
- optimize --band FMIN,FMAX,N scores every candidate at N frequencies and keeps the worst case;
  --tolerance-mc N ranks the best candidates by the 10th percentile score under random build tolerances
  (element lengths and spacings, line length, T-match taps and bar offsets per element, line VF and Z0, series
  capacitor; the same perturbations for every candidate). The series capacitor and the matching network stay as
  designed for the nominal candidate (--tol-retune re-tunes them per sample). Candidates are simulated in parallel
  (--jobs) and every geometry only once (a candidate the search revisits reuses its result). The scan CSV starts
  with '#' comment lines that record the arguments and the nec2c version.

Reflectors and directors (generate / optimize):
- --reflectors N (0..2) adds passive reflectors behind the rear element and --directors N (0..5) passive
  directors in front of the front element (tags 21-22 and 31-35, one continuous tube each, same plane, no source).
  For every one of them the program PRE-CALCULATES a minimum / nominal / maximum length (factors of the
  resonant dipole length of the tube: reflectors 1.00..1.15, directors 0.80..1.00) and a minimum / nominal /
  maximum spacing from the previous element (minimum = the larger of 0.05 lambda and tube radii + --min-gap-mm,
  maximum 0.40 lambda), prints them as a table, and lengthens the boom automatically to hold every element
  plus --boom-overhang-mm at each end.
- optimize searches those lengths and spacings inside the pre-calculated limits: --parasitic-length-offsets /
  --parasitic-spacing-scales are grid dimensions applied to all parasitic elements, and the local refinement
  (--refine, 8 rounds by default when reflectors/directors are present) moves each length and spacing
  separately, never leaving the limits. The CSV has the real lengths, spacings and the boom length needed;
  --build-sheet adds the cut list, positions along the boom and the bill of materials of these elements.
- The HB9CV phasing (rear/front) is unchanged; the parasitic elements only add coupling to the two driven ones.

Model notes (generate / optimize):
- --network ideal uses two independent EX sources. --ideal-drive current (default) prescribes the amplitude/phase
  of the element currents: each source segment gets a lossless 1e6 ohm series reactance (LD 4), so the currents
  follow the source voltages whatever the mutual coupling; --ideal-drive voltage prescribes the voltages.
- --network tl: a single EX and an ideal transmission line (TL card) between the two
  feed points. The TL card is a BALANCED two-wire model: it carries no shield or
  common-mode currents. (--network coaxial is a deprecated alias of tl.)
- Feed points (--feed-style): tmatch (default with tl/lc) is the classic HB9CV feed: each driven element is a
  continuous tube with a T-match (taps at --tap-mm from the center, a bar at --tap-offset-mm toward the other
  element, --tap-wire-mm wire, tags 3 and 4); EX and the phase network connect at the bar centers. center connects
  them at the element centers (split, insulated elements) and is used by default with stepped sections or
  --model-feed-gap. With a center feed and an open-wire line (Z0 ~300 ohm) the element currents cannot reach the
  cardioid relation with a short straight line; the T-match transforms the element impedances so that they can.
- --series-c-pf PF|auto: the series capacitor between the feed line and the fed port (the HB9CV trimmer). It is
  written as a comment card (HB9CV_SERIES_C_PF=... HB9CV_SERIES_C_Q=...) and added, with its ESR = |X|/Q
  (--series-c-q, default 500), to every Zin read from the .out file (generate, optimize, match, sweep, GUI); auto
  cancels X_in at --freq (per candidate in optimize).
- Phase line (--synthesize-line, default on): two probe runs (1 V on one feed port, no network) give the port and
  element-current responses of the driven elements; the line length is chosen so that the far/fed element-current
  ratio is closest to the cardioid target (equal amplitudes, rear leading by 180 - kd), and the free-Z0 exact
  solution is reported. With reflectors/directors in free space the two-element target does not hold: the line
  that maximizes the objective on the horizon ring, computed by superposing the two probe patterns, is chosen
  instead. generate uses it as the default line; optimize centers the line scan of every spacing, Z0 and tap on
  it, and synthesizes it again for the pre-tuned front length. Without it the cardioid theory below is used. A TL line is never shorter than the straight distance
  between its ends plus --line-allowance-mm (default 20 mm open-wire, 40 mm coax) and, by default, never longer than
  1.5 times that distance (--max-line-deg); line lengths never wrap around 360 degrees.
  --line-type openwire (default): balanced air line, crossed by construction (keep
  --crossed), VF 0.95; Z0 is NOT typed in: Z0 = 120*acosh(D/d) ohm from
  --line-wire-diameter d and --line-spacing D (air, exact for two round wires).
  --line-type coax: a coax cannot be crossed, so the line is always straight (delay
  225 deg for the classic front-fed HB9CV), VF 0.66, Z0 from --line-z0; a coax phasing
  section needs a balun/choke at its element end, which the NEC TL does not model.
  A line physically shorter than the distance between the feed points (plus --line-allowance-mm) cannot be
  built: generate warns and optimize skips such lines.
- --network lc: a single EX and an ideal lumped pi LC network (NT card with Y
  parameters) between the centers; L and C come from the target delay or are fixed
  with --lc-l-nh and --lc-c-pf. No losses, parasitics or component radiation. NEC holds NT
  admittances fixed, so every frequency group of the deck carries its own NT card.
- Balun / common mode: the NEC EX is an ideal balanced source, but the real antenna is
  balanced and fed from unbalanced coax; a 1:1 current balun / common-mode choke
  (>= 500-1000 ohm common-mode impedance at the design frequency, to be validated on the
  bench) is MANDATORY at the feed point. --feedline-length-m adds a wire for the coax
  outer conductor (tied to the lower half of the fed port, whose one-segment wire carries
  the source; it runs rearward below the elements, beside a modeled boom at
  --boom-clearance-mm from it; the fed element keeps its tag) and --cm-sensitivity
  simulates the same model with and
  without it and reports the change in F/B and Zin. The wire is an open-ended shield:
  an upper-bound sensitivity test, not a model of a real coax run or of a choke.
- --balun-ratio 4:1 (optimize/match/sweep): the balanced-side reference becomes
  50/4 = 12.5 ohm (default --target-impedance 12.5, default --zin-x-max 10). It is an
  option to evaluate, not a proven fix.
- --build-sheet FILE writes the build sheet: FILE (markdown) and the same data as JSON next to it (.json):
  elements (NEC and cut lengths), spacing/boom, phasing line, feed point and balun, predicted performance,
  tuning procedure and bill of materials. Performance needs --run in generate (or optimize).
- --objective and --target-impedance are informative only in generate mode (they are
  written to the comments and to the output); the criteria are applied by optimize.
- Conductor losses: --conductivity SIGMA (S/m, default 2.5e7 for 6061 aluminium; 0 = perfect conductors) emits
  `LD 5 0 0 0 SIGMA` (wire conductivity, all wires) after the geometry and before FR. With --run (generate) and at
  the end of optimize the model is simulated again with perfect conductors and the change in gain, F/B and R_in is
  reported; the efficiency comes from the NEC power budget of the lossy run (radiated / input power), with the gain
  difference 10^(dG/10) as a cross-check. The TL and NT networks stay lossless (ideal phase line / LC); only the
  antenna conductors lose.
- Element diameters and taper: --element-diameter is the default for both elements; --rear-diameter /
  --front-diameter set each one; --rear-sections / --front-sections (LENGTH_MM:DIAMETER_MM,...,DIAMETER_MM, one half,
  center outward, the last section runs to the tip) model telescoped tubes as consecutive GW cards of the same tag
  (a central wire with an odd segment count holds the source). Every model is checked for the thin-wire rules
  (segment >= 2 radii, <= lambda/10, neighbouring radius <= 2:1 and segment length <= 5:1); problems are printed
  as warnings. The deck uses NEC-2's thin-wire kernel (--extended-kernel adds the EK card), and every run must have
  a physical power budget (positive radiated power, efficiency <= 100.5 %); a run that does not is discarded as a
  numerically unstable model.
- Feed gap: --feed-gap-mm g (older name --gap-mm) goes into the cut list: half = (L_nec - g)/2 - e, the gap being
  inside the tip-to-tip NEC length. --model-feed-gap also models it: halves from +-g/2 to the tips and a 1-segment wire
  of length g that carries EX/TL; the bridge must be at least 4 tube radii and lambda/200 long (not combined with
  the common-mode feed-line wire).
- The boom is not modeled unless --boom-isolated is given (or --boom-material names a metal): a metallic
  conductor with no electrical connection, --boom-clearance-mm (default 10) below the element surfaces. With
  --feed-style tmatch the element centers are at a voltage null and --mount contact bonds them (and the
  reflector/director centers) to the modeled boom through a short wire each; with a center feed the elements are
  always insulated. Boom and mast are segmented by their own radius (segments >= 4 radii, <= lambda/10). Every
  model is checked for overlapping conductors (an error) and for conductors closer than half the larger radius
  (a warning). A square boom of side a is represented with the equivalent radius 0.59*a; a round one with radius
  d/2.
- Mast (--mast-length-m, tag 12): horizontal elements: below the boom center, moved sideways by --mast-offset-mm
  (on the axis it lies in the symmetry plane of the balanced elements and is not excited); vertical elements: a
  side mount parallel to the elements behind the rear end of the boom.
- Checks (need nec2c): --check-convergence (segments n vs 2n+1), --robustness (metallic boom, a mast placed where it
  breaks the symmetry and an unchoked coax shield, each alone; rows the symmetric model cannot excite are marked),
  generate --height-sweep MIN,MAX,N (over ground). The element-current report gives
  |I_rear/I_front| and the phase difference at the feed ports. --json-out FILE writes the main results as JSON.

Phase theory (elements at x=-d/2 rear and x=+d/2 front, beam toward +x):
  for a cardioid with a null at phi=180 and equal-amplitude currents, the rear
  current leads the front one by (180 - kd) degrees (kd = 360*d/lambda; 135 degrees
  for d=0.125 lambda). With the classic HB9CV the front element is fed and a CROSSED
  line of kd electrical degrees (45 degrees) carries the signal to the rear one:
  delay 180 + 45 = 225 = lead of 135 degrees. In NEC a crossed TL line is indicated
  with a negative Z0. Mutual coupling shifts these values, which is why the line is
  synthesized from the element responses (--synthesize-line) and the optimizer sweeps
  around that line.

L-network (match / sweep modes):
  Topology A (R < Z0): series branch on the antenna side, shunt branch on the Z0 side.
  Topology B (R >= Z0): shunt branch on the antenna side, series branch on the Z0 side.
  Each branch is an inductor (jwL) or a capacitor (1/(jwC)) depending on the sign of
  the required reactance at the design frequency; a negative value is never used.
  Each topology has two solutions (positive and negative root); --lnet-solution picks one
  (plus = positive root by default; minus; lowpass / highpass by branch kinds; best by the
  worst VSWR over the sweep) and the other is printed. The VSWR <= 2 bandwidth is the
  contiguous band around the best point. The self-resonance of every component is
  checked against the band (--esl-nh, --coil-cp-pf).
  Neither mode changes or simulates the NEC model; the calculated components are a
  first estimate at a single frequency and need later validation (antenna analyzer/VNA).
"""
import argparse
import cmath
import contextlib
import csv
import datetime
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ProcessPoolExecutor
from argparse import Namespace
from pathlib import Path

MODES = ("generate", "optimize", "match", "sweep")
# Optimizer defaults that depend on --match-score: scoring through a matching network admits any Zin the network
# can transform (its loss is part of the score); a direct connection needs a Zin close to the target.
MATCHED_ZIN_R_RANGE = "2,500"
DIRECT_ZIN_R_RANGE = "8,80"
MATCHED_ZIN_X_MAX = 500.0
DEFAULT_DIAMETER_VALUES = "6,8,10,12,14,16"
# Speed of light used for lambda = C_LIGHT / f. NEC computes its own wavelength internally with c of about
# 299.8 (MHz*m); the difference (~0.0025 %) is negligible for the geometry. It is also why TL lengths are written
# from the physical length / VF in meters (see build_deck) instead of being re-derived from electrical degrees:
# the meter value does not depend on which c each program uses, so no correction is needed here.
C_LIGHT = 299_792_458.0


# ======================================================================================
# Argument types
# ======================================================================================

def positive_float(value):
    try:
        x = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(x) or x <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return x


def phase_float(value):
    try:
        x = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(x):
        raise argparse.ArgumentTypeError("must be finite")
    if not -360 <= x <= 360:
        raise argparse.ArgumentTypeError("must be between -360 and 360 degrees")
    return x


def nonneg_float(value):
    try:
        x = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(x) or x < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def sections_list(value):
    """'300:12,200:10,8' -> [(300.0, 12.0), (200.0, 10.0), (None, 8.0)].

    Stepped element, ONE half, from the center (boom axis) outward: LENGTH_MM:DIAMETER_MM for every section but the
    last, which runs to the tip and takes only its DIAMETER_MM."""
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("must be LENGTH_MM:DIAMETER_MM,...,DIAMETER_MM")
    out = []
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        try:
            nums = [float(b) for b in part.split(":")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"section '{part}': numbers expected") from exc
        if not all(math.isfinite(x) and x > 0 for x in nums):
            raise argparse.ArgumentTypeError(f"section '{part}': lengths and diameters must be greater than zero")
        if last and len(nums) != 1:
            raise argparse.ArgumentTypeError("the last section runs to the tip: give only its DIAMETER_MM")
        if not last and len(nums) != 2:
            raise argparse.ArgumentTypeError(f"section '{part}': every section except the last needs "
                                             "LENGTH_MM:DIAMETER_MM")
        out.append((None, nums[0]) if last else (nums[0], nums[1]))
    return out


def series_c_arg(value):
    """--series-c-pf: a capacitance in pF (> 0) or 'auto'."""
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        x = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a capacitance in pF or 'auto'") from exc
    if not math.isfinite(x) or x <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero (pF) or 'auto'")
    return x


def finite_float(value):
    try:
        x = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(x):
        raise argparse.ArgumentTypeError("must be finite")
    return x


# ======================================================================================
# NEC2C output parsing (ANTENNA INPUT PARAMETERS and RADIATION PATTERNS tables)
# ======================================================================================

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
FREQ_RE = re.compile(rf"FREQUENCY\s*:\s*({FLOAT})\s*MHZ", re.IGNORECASE)
ROW_RE = re.compile(rf"^\s*(\d+)\s+(\d+)((?:\s+{FLOAT}){{9}})\s*$")
# One row of a RADIATION PATTERNS table: theta, phi, vert/major dB, hor/minor dB, total dB, axial ratio.
PATTERN_ROW = re.compile(rf"^\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})")
# The same row with the polarization sense and the complex far field: ... TILT SENSE |E_theta| phase |E_phi| phase.
PATTERN_FIELD_ROW = re.compile(rf"^\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+{FLOAT}\s+"
                               rf"{FLOAT}\s+[A-Za-z]*\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")

# Messages that invalidate a run even if nec2c managed to print a pattern.
FATAL_RE = re.compile(
    r"ERROR|INCORRECT|ILLEGAL|FAULTY|NO SEGMENT HAS|DOES NOT EXIST|INTERSECT|OVERLAP"
    r"|STRUCTURE.*WARNING|WARNING.*STRUCTURE", re.IGNORECASE)
# Warnings that do not invalidate the run but are counted and reported.
WARN_RE = re.compile(r"WARNING|CAUTION", re.IGNORECASE)


def nec_blocks(text):
    """Split a nec2c output text into frequency blocks: [(printed frequency or None, lines)].

    The first entry holds the lines before the first FREQUENCY header (frequency None). A block that repeats the
    previous one (same printed frequency and either no ANTENNA INPUT PARAMETERS table or the same table, as printed
    when several RP cards follow one single-frequency FR card) is folded into it: only its RADIATION PATTERNS
    tables are kept, so every solved frequency is one block whatever the number of pattern requests."""
    parts, cur_f, cur = [], None, []
    for line in text.splitlines():
        m = FREQ_RE.search(line)
        if m:
            parts.append((cur_f, cur))
            cur_f, cur = float(m.group(1)), [line]
            continue
        cur.append(line)
    parts.append((cur_f, cur))
    out = [parts[0]]
    for f, lines in parts[1:]:
        prev_f, prev = out[-1]
        if len(out) > 1 and prev_f == f:
            rows_new = _input_rows(lines)
            if not rows_new or rows_new == _input_rows(prev):
                k = next((i for i, l in enumerate(lines) if "RADIATION PATTERNS" in l), None)
                if k is not None:
                    prev.extend(lines[k:])
                continue
        out.append((f, list(lines)))
    return out


def _input_rows(lines):
    """Raw rows of the ANTENNA INPUT PARAMETERS table of one block (used to recognize a repeated block)."""
    rows, on = [], False
    for line in lines:
        if "ANTENNA INPUT PARAMETERS" in line:
            on = True
            continue
        if on and ("CURRENTS AND LOCATION" in line or "RADIATION PATTERNS" in line or "POWER BUDGET" in line):
            break
        if on and ROW_RE.match(line):
            rows.append(line.strip())
    return rows


def parse_input_parameters(path):
    """Read the ANTENNA INPUT PARAMETERS table of a nec2c .out file.

    Returns one dict per frequency and per EX source: freq_mhz, fidx, tag, seg, voltage,
    current, z, y, power_w. fidx is the index of the frequency block the row belongs to (run
    order, see nec_blocks); rows are grouped by it, never by the printed frequency, which nec2c rounds to five
    significant digits. When a <name>.freqs.json file written by generate sits next to the .out
    file and lists one frequency per block, its exact values replace the printed ones. The table columns are:
    TAG, SEG, VOLTAGE (real, imag), CURRENT (real, imag), IMPEDANCE (real, imag), ADMITTANCE (real, imag), POWER.
    Raises ValueError if the file does not contain the table.
    """
    text = Path(path).read_text(errors="replace")
    if "ANTENNA INPUT PARAMETERS" not in text:
        raise ValueError("ANTENNA INPUT PARAMETERS not found in the .out file")
    entries = []
    for fidx, (freq, lines) in enumerate(nec_blocks(text)[1:]):
        for row in _input_rows(lines):
            m = ROW_RE.match(row)
            v = [float(x) for x in m.group(3).split()]
            entries.append({
                "freq_mhz": freq,
                "fidx": fidx,
                "tag": int(m.group(1)),
                "seg": int(m.group(2)),
                "voltage": complex(v[0], v[1]),
                "current": complex(v[2], v[3]),
                "z": complex(v[4], v[5]),
                "y": complex(v[6], v[7]),
                "power_w": v[8],
            })
    if not entries:
        raise ValueError("the ANTENNA INPUT PARAMETERS table contains no readable rows")
    if any(e["freq_mhz"] is None for e in entries):
        raise ValueError("could not determine the frequency of the impedance table")
    exact = read_freq_sidecar(path)
    if exact is not None and len(exact) == max(e["fidx"] for e in entries) + 1:
        for e in entries:
            if abs(exact[e["fidx"]] - e["freq_mhz"]) <= freq_print_tol(e["freq_mhz"]):
                e["freq_mhz"] = exact[e["fidx"]]
    # Series feed capacitor declared in the model comments: it sits between the feed line and the port, so it
    # adds its impedance (reactance plus ESR = |X|/Q) to the impedance of a single-source model. z_port keeps the
    # NEC value.
    head = text.split("STRUCTURE SPECIFICATION", 1)[0]
    mc = SERIES_C_RE.search(head)
    c_pf = float(mc.group(1)) if mc else 0.0
    mq = SERIES_C_Q_RE.search(head)
    c_q = float(mq.group(1)) if mq else 0.0
    for e in entries:
        e["z_port"] = e["z"]
        if c_pf > 0:
            e["z"] = e["z"] + series_c_impedance(c_pf, e["freq_mhz"], c_q)
            e["y"] = 1.0 / e["z"] if e["z"] != 0 else e["y"]
    return entries



def freq_sidecar_path(out_path):
    """<name>.freqs.json next to <name>.out: the exact frequencies of the run, in run order."""
    return Path(out_path).with_suffix(".freqs.json")


def write_freq_sidecar(out_path, freqs):
    try:
        freq_sidecar_path(out_path).write_text(json.dumps({"freqs_mhz": [float(f) for f in freqs]}),
                                               encoding="utf-8")
    except OSError as exc:
        print(f"WARNING: could not write {freq_sidecar_path(out_path)}: {exc}", file=sys.stderr)


def read_freq_sidecar(out_path):
    """Exact frequency list of a run, or None when there is no readable sidecar file."""
    try:
        data = json.loads(freq_sidecar_path(out_path).read_text(encoding="utf-8"))
        freqs = [float(f) for f in data["freqs_mhz"]]
        return freqs if all(math.isfinite(f) and f > 0 for f in freqs) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def freq_print_tol(f_mhz):
    """Matching tolerance (MHz) for a frequency printed by nec2c with five significant digits:
    0.6 units of the fifth digit (145 MHz -> 0.006 MHz, 28.5 MHz -> 0.0006 MHz)."""
    return 0.6 * 10.0 ** (math.floor(math.log10(abs(f_mhz))) - 4) if f_mhz else 1e-6


def single_source_zin(path):
    """Return [(freq_mhz, Zin)] sorted by frequency, one row per FREQUENCY block.

    Rejects the file if any frequency block has more than one EX source: in that case there
    is no single input port and a single Z has no meaning.
    """
    entries = parse_input_parameters(path)
    per_block = {}
    for e in entries:
        per_block.setdefault(e["fidx"], []).append(e)
    for rows in per_block.values():
        if len(rows) != 1:
            raise ValueError(
                f"the model has {len(rows)} EX sources at {rows[0]['freq_mhz']:g} MHz; there is no single "
                "50 ohm port. Use a model with --network tl (a single EX).")
    return sorted(((rows[0]["freq_mhz"], rows[0]["z"]) for rows in per_block.values()), key=lambda r: r[0])


def pick_frequency(zin_table, freq_mhz, tol_mhz=None):
    """Return the (f, Z) row whose frequency matches freq_mhz within the printed resolution of nec2c
    (tol_mhz overrides it)."""
    tol = freq_print_tol(freq_mhz) if tol_mhz is None else tol_mhz
    f, z = min(zin_table, key=lambda item: abs(item[0] - freq_mhz))
    if abs(f - freq_mhz) > tol:
        raise ValueError(f"{freq_mhz:g} MHz is not in the NEC sweep (closest: {f:g} MHz)")
    return f, z


def scan_nec_messages(text):
    """Return (fatal, warnings): .out lines that invalidate the run and accepted warnings.

    The COMMENTS block (echo of the CM cards) is ignored so that a comment of the
    model does not trigger false positives.
    """
    marker = text.find("STRUCTURE SPECIFICATION")
    body = text[marker:] if marker >= 0 else text
    fatal, warnings = [], []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if FATAL_RE.search(stripped):
            fatal.append(stripped)
        elif WARN_RE.search(stripped):
            warnings.append(stripped)
    return fatal, warnings


MAX_EFFICIENCY_PCT = 100.5   # above this the NEC power budget of a passive model is not physical


def validate_input_table(path):
    """Require a valid ANTENNA INPUT PARAMETERS table and a physical POWER BUDGET. Per frequency: with one source
    Re(Zin) > 0; with several, the total input power > 0 (the R of an individual source can be negative because of
    coupling); the radiated power is positive and the efficiency at most MAX_EFFICIENCY_PCT."""
    groups = {}
    for e in parse_input_parameters(path):
        groups.setdefault(e["fidx"], (e["freq_mhz"], []))[1].append(e)
    for f, rows in groups.values():
        if len(rows) == 1:
            if not rows[0]["z"].real > 0:
                raise ValueError(f"Re(Zin)={rows[0]['z'].real:.4g} ohm is not positive at {f:g} MHz")
        elif not sum(r["power_w"] for r in rows) > 0:
            raise ValueError(f"the total input power is not positive at {f:g} MHz")
    # A passive structure cannot radiate more than it accepts: a negative radiated power or an efficiency above
    # 100 % means the current solution itself is not physical (a numerically unstable model).
    text = Path(path).read_text(errors="replace")
    k = 0
    while True:
        b = parse_power_budget(text, k)
        if b is None:
            break
        f = groups.get(k, (None,))[0]
        where = f" at {f:g} MHz" if f else f" in frequency block {k + 1}"
        if not b["radiated_w"] > 0 or b.get("efficiency_pct", 0.0) > MAX_EFFICIENCY_PCT:
            raise ValueError(f"the NEC solution is not physical{where} (radiated power {b['radiated_w']:.3e} W, "
                             f"efficiency {b.get('efficiency_pct', float('nan')):.1f} %): numerically unstable "
                             "model (thick wires on short segments, closely spaced conductors)")
        k += 1


def zin_in_order(path, n_expected):
    """Zin of every frequency in the order nec2c ran them (one EX source per frequency).

    Matching by order, not by value: the .out file prints frequencies with only five
    significant digits.
    """
    entries = parse_input_parameters(path)
    if len(entries) != n_expected:
        raise ValueError(f"expected {n_expected} single-source impedance rows, found {len(entries)}")
    return [e["z"] for e in entries]


DBI_TO_DBD = 2.15   # a half-wave dipole has 2.15 dBi in free space
FB_METRIC_KEYS = {"point": "fb", "sector": "robust_fb", "cone": "cone_fb"}


def parse_pattern_sections(text):
    """Split the RADIATION PATTERNS tables of a nec2c .out file, one section per frequency block (nec_blocks).

    Returns [{"freq_mhz", "points"}] in run order, where points maps (theta, phi) -> (vert_or_major_dB,
    hor_or_minor_dB, total_dB, axial_ratio). Only rows after a RADIATION PATTERNS header are read, so
    the integer-led rows of the impedance table are never mistaken for pattern rows. Several pattern tables of
    one block merge into its section; sections without pattern rows are dropped.
    """
    sections = []
    for freq, lines in nec_blocks(text):
        sec, in_pat = {"freq_mhz": freq, "points": {}}, False
        for line in lines:
            if "RADIATION PATTERNS" in line:
                in_pat = True
                continue
            if not in_pat:
                continue
            m = PATTERN_ROW.match(line)
            if m:
                theta, phi, vert, hor, total, axial = (float(x) for x in m.groups())
                sec["points"][(round(theta, 2), round(phi % 360.0, 5))] = (vert, hor, total, axial)
        if sec["points"]:
            sections.append(sec)
    return sections



def ring_from_section(section, theta):
    """(phi -> total gain dB for the valid points of the ring at theta, number of rows of that ring)."""
    pts, phis = {}, set()
    for (th, ph), vals in section["points"].items():
        if abs(th - theta) <= 0.006:
            phis.add(ph)
            if vals[2] > -900:
                pts[ph] = vals[2]
    return pts, len(phis)


def elevation_peak_theta(section, phi=0.0):
    """Theta (degrees from the zenith) of maximum total gain along the elevation cut at azimuth phi."""
    best = None
    for (th, ph), vals in section["points"].items():
        if abs(ph - phi) <= 0.006 and vals[2] > -900 and (best is None or vals[2] >= best[1]):
            best = (th, vals[2])   # >= : on a flat maximum keep the lowest elevation
    if best is None:
        raise ValueError("the elevation cut has no valid points")
    return best[0]


def half_power_beamwidth(pattern, drop=3.0):
    """Azimuth beamwidth (degrees) between the -drop dB points around the maximum of a ring
    (phi -> dB); the edges are interpolated linearly in power (a deep null next to the beam does not pull
    them inward, as a dB interpolation would), circular wrap-around. 360 if the pattern never falls drop dB below
    its maximum. The GUI uses the same rule for its elevation cut."""
    phis = sorted(pattern)
    n = len(phis)
    gains = [pattern[p] for p in phis]
    ipk = max(range(n), key=lambda i: gains[i])
    thr = gains[ipk] - drop

    def walk(direction):
        prev_phi, prev_g, total = phis[ipk], gains[ipk], 0.0
        for k in range(1, n):
            i = (ipk + direction * k) % n
            d = ((phis[i] - prev_phi) * direction) % 360.0
            g = gains[i]
            if g < thr:
                p_prev, p_now, p_thr = 10.0 ** (prev_g / 10.0), 10.0 ** (g / 10.0), 10.0 ** (thr / 10.0)
                frac = (p_prev - p_thr) / (p_prev - p_now) if p_prev != p_now else 0.0
                return total + d * frac
            total += d
            prev_phi, prev_g = phis[i], g
        return None

    right, left = walk(1), walk(-1)
    return 360.0 if right is None or left is None else right + left


def ring_metrics(points, n_expected, theta, rear_start=150, rear_end=210, ground_free=True):
    """Pattern metrics of one azimuth ring (phi -> total gain dB) at zenith angle theta."""
    if n_expected == 0:
        raise ValueError(f"no pattern points at theta={theta:g} degrees")
    if len(points) < 0.9 * n_expected:
        if not ground_free:
            raise ValueError(f"pattern ring at theta={theta:g} has no valid points over ground; "
                             "use the take-off angle")
        raise ValueError(f"incomplete pattern table: {len(points)} valid points")
    forward_gain, rear_gain = points.get(0.0), points.get(180.0)
    if forward_gain is None or rear_gain is None:
        raise ValueError("pattern points phi=0 or phi=180 degrees are missing")
    # Robust metric: the worst radiation level in a rear sector, not just the point
    # null at phi=180. This penalizes very narrow nulls.
    rear_samples = [(a, g) for a, g in points.items() if rear_start <= a <= rear_end]
    if not rear_samples:
        raise ValueError("no valid samples in the rear sector")
    worst_rear_angle, worst_rear_gain = max(rear_samples, key=lambda item: item[1])
    # The beam can deviate from phi=0 (squint): forward gain != maximum gain.
    max_angle, max_gain = max(points.items(), key=lambda item: item[1])
    # dBd is only meaningful in free space: over ground the NEC gain includes the ground reflection.
    return {"forward_gain": forward_gain, "rear_gain": rear_gain, "fb": forward_gain - rear_gain,
            "worst_rear_gain": worst_rear_gain, "worst_rear_angle": worst_rear_angle,
            "robust_fb": forward_gain - worst_rear_gain, "max_gain": max_gain, "max_angle": max_angle,
            "hpbw": half_power_beamwidth(points), "gain_dbd": (forward_gain - DBI_TO_DBD) if ground_free else None,
            "theta_deg": theta, "cone_fb": forward_gain - worst_rear_gain, "cone_rear_gain": worst_rear_gain,
            "cone_worst_theta": theta, "cone_worst_phi": worst_rear_angle, "cone_points": 0}


def rear_cone_half_angle(rear_start, rear_end):
    """Half-angle (degrees) of the rear cone: the same angular width as the --rear-sector."""
    return max(abs(180.0 - rear_start), abs(rear_end - 180.0))


def add_cone_metrics(m, section, rear_start=150, rear_end=210):
    """Add the rear-cone F/B to the ring metrics m: forward gain minus the highest gain of every pattern point
    of section within the rear-sector half-angle of the back direction (phi=180 at the elevation of the ring),
    at any elevation. Uses the rear half-sphere grid written by build_deck; the ring sector itself is always
    included, so cone_fb <= robust_fb."""
    half = math.radians(rear_cone_half_angle(rear_start, rear_end))
    th_r = math.radians(m["theta_deg"])
    back = (-math.sin(th_r), 0.0, math.cos(th_r))
    worst, w_th, w_ph, count = m["worst_rear_gain"], m["theta_deg"], m["worst_rear_angle"], 0
    cos_half = math.cos(half) - 1e-9
    for (th, ph), vals in section["points"].items():
        if vals[2] <= -900:
            continue
        t, p_ = math.radians(th), math.radians(ph)
        d = (math.sin(t) * math.cos(p_), math.sin(t) * math.sin(p_), math.cos(t))
        if d[0] * back[0] + d[1] * back[1] + d[2] * back[2] >= cos_half:
            count += 1
            if vals[2] > worst:
                worst, w_th, w_ph = vals[2], th, ph
    m.update(cone_fb=m["forward_gain"] - worst, cone_rear_gain=worst, cone_worst_theta=w_th, cone_worst_phi=w_ph,
             cone_points=count)
    # Cross-polar discrimination in the forward direction: the stronger of the vertical and horizontal
    # components (the ring is written with XNDA=1000) minus the weaker one; inf when the weaker one is a null.
    fwd = next((v for (th, ph), v in section["points"].items() if abs(th - m["theta_deg"]) <= 0.006 and ph == 0.0),
               None)
    m["xpd_db"] = None
    if fwd is not None and max(fwd[0], fwd[1]) > -900:
        m["xpd_db"] = abs(fwd[0] - fwd[1]) if min(fwd[0], fwd[1]) > -900 else math.inf
    return m


def parse_metrics(path, theta=None, rear_start=150, rear_end=210, section=0, ground_free=True):
    """Pattern metrics of the azimuth ring at zenith angle theta of a nec2c .out file.

    theta=None: the azimuth ring of the table (the theta with the most points; the ring at 90 degrees in
    free space).
    section: which frequency of a multi-frequency run (0 = first). The rear-cone F/B is added from the rear
    half-sphere table when the deck has one.
    """
    text = Path(path).read_text(errors="replace")
    fatal, _ = scan_nec_messages(text)
    if fatal:
        raise ValueError(f"NEC2C reported a problem: {fatal[0]}")
    sections = parse_pattern_sections(text)
    if not sections:
        raise ValueError("RADIATION PATTERNS table not found")
    validate_input_table(path)
    try:
        sec = sections[section]
    except IndexError:
        raise ValueError(f"pattern section {section} not found ({len(sections)} present)") from None
    if theta is None:
        counts = {}
        for th, _ in sec["points"]:
            counts[th] = counts.get(th, 0) + 1
        top = max(counts.values())
        ring = [th for th, c in counts.items() if c == top]
        if len(ring) != 1:
            raise ValueError("the pattern table has several azimuth rings; pass theta=")
        theta = ring[0]
    points, n_expected = ring_from_section(sec, theta)
    return add_cone_metrics(ring_metrics(points, n_expected, theta, rear_start, rear_end, ground_free), sec,
                            rear_start, rear_end)


def parse_output(path, rear_start=150, rear_end=210, theta=None, section=0, ground_free=True):
    """Legacy tuple interface of parse_metrics.

    Returns (forward_gain, rear_gain, front_to_back, worst_rear_gain, worst_rear_angle,
    robust_front_to_back, max_gain, max_gain_angle).
    """
    m = parse_metrics(path, theta, rear_start, rear_end, section, ground_free)
    return (m["forward_gain"], m["rear_gain"], m["fb"], m["worst_rear_gain"], m["worst_rear_angle"],
            m["robust_fb"], m["max_gain"], m["max_angle"])


# CURRENTS AND LOCATION table. nec2c prints two leading integers per row:
#   SEG  TAG    X  Y  Z  LENGTH  REAL  IMAGINARY  MAGN  PHASE
# A three-integer layout (SEG, TAG, segment number inside the tag) is also accepted. The table
# starts after a "CURRENTS AND LOCATION" header and ends at the next table or FREQUENCY block.
CURR_ROW2_RE = re.compile(rf"^\s*(\d+)\s+(\d+)((?:\s+{FLOAT}){{8}})\s*$")
CURR_ROW3_RE = re.compile(rf"^\s*(\d+)\s+(\d+)\s+(\d+)((?:\s+{FLOAT}){{8}})\s*$")


def parse_current_rows(lines):
    """Rows of a CURRENTS AND LOCATION table: [{seg_overall, tag, seg_in_tag, current}].

    seg_in_tag comes from the third integer when it is printed, otherwise from the order of the
    rows of each tag (nec2c lists the segments of a tag in numbering order)."""
    rows, count = [], {}
    for line in lines:
        m3 = CURR_ROW3_RE.match(line)
        m = m3 or CURR_ROW2_RE.match(line)
        if not m:
            continue
        tag = int(m.group(2))
        fl = [float(x) for x in m.group(m.lastindex).split()]   # X Y Z LENGTH REAL IMAG MAGN PHASE
        count[tag] = count.get(tag, 0) + 1
        rows.append({"seg_overall": int(m.group(1)), "tag": tag,
                     "seg_in_tag": int(m.group(3)) if m3 else count[tag],
                     "current": complex(fl[4], fl[5])})
    return rows


def parse_element_currents(path, freq_section=0):
    """Rows of the CURRENTS AND LOCATION table of one frequency block (0 = first) of a nec2c .out
    file, as returned by parse_current_rows. Raises ValueError when the table is absent or empty."""
    text = Path(path).read_text(errors="replace")
    blocks = []
    for _freq, lines in nec_blocks(text):
        cur, on = [], False
        for line in lines:
            if "CURRENTS AND LOCATION" in line:
                on = True
                continue
            if on and ("RADIATION PATTERNS" in line or "ANTENNA INPUT PARAMETERS" in line
                       or "POWER BUDGET" in line or "CHARGE DENSITIES" in line):
                on = False
                continue
            if on:
                cur.append(line)
        if cur:
            blocks.append(cur)
    if not blocks:
        raise ValueError("CURRENTS AND LOCATION table not found in the .out file (PT -1 suppresses it)")
    try:
        lines = blocks[freq_section]
    except IndexError:
        raise ValueError(f"frequency section {freq_section} not found ({len(blocks)} present)") from None
    rows = parse_current_rows(lines)
    if not rows:
        raise ValueError("CURRENTS AND LOCATION table contains no readable rows")
    return rows



def segment_current(rows, tag, seg_in_tag=None):
    """Complex current of segment seg_in_tag (1-based, numbered along the tag) of a tag; without
    seg_in_tag the middle segment (the lower one of the two middle segments for an even count)."""
    tag_rows = sorted((r for r in rows if r["tag"] == tag), key=lambda r: r["seg_in_tag"])
    if not tag_rows:
        raise ValueError(f"tag {tag} not found in CURRENTS AND LOCATION table")
    if seg_in_tag is None:
        return tag_rows[(len(tag_rows) - 1) // 2]["current"]
    for r in tag_rows:
        if r["seg_in_tag"] == seg_in_tag:
            return r["current"]
    raise ValueError(f"segment {seg_in_tag} of tag {tag} not found in CURRENTS AND LOCATION table")


def print_element_currents(path, ports=None, freq_section=0):
    """Print |I_rear / I_front| and the phase difference of the currents at the feed ports.

    ports: {"rear": (tag, seg_in_tag), "front": (tag, seg_in_tag)} as returned by build_deck in
    line_info["ports"]; without it, the middle segment of tag 1 (rear) and tag 2 (front).
    Raises ValueError when the table is absent or the segments are missing."""
    ports = ports or {"rear": (1, None), "front": (2, None)}
    rows = parse_element_currents(path, freq_section=freq_section)
    i_front = segment_current(rows, *ports["front"])
    i_rear = segment_current(rows, *ports["rear"])
    ratio = abs(i_rear) / abs(i_front) if abs(i_front) > 1e-30 else float("inf")
    phase_diff = wrap180(math.degrees(math.atan2(i_rear.imag, i_rear.real)
                                      - math.atan2(i_front.imag, i_front.real)))
    print("ELEMENT CURRENTS (at the feed ports)")
    for name, i_x in (("front", i_front), ("rear", i_rear)):
        tag, seg = ports[name]
        where = f"tag {tag}" + (f", segment {seg}" if seg is not None else "")
        print(f"  {name:<5s} ({where}): {i_x.real:+.5e}{i_x.imag:+.5e}j A  "
              f"(|I| = {abs(i_x):.5e}; phase = {math.degrees(math.atan2(i_x.imag, i_x.real)):+.1f} deg)")
    print(f"  |I_rear / I_front| = {ratio:.4f}  (cardioid ideal: 1.000)")
    print(f"  phase(rear) - phase(front) = {phase_diff:+.1f} deg  (cardioid ideal: 180 - kd, +135 deg for "
          "0.125 lambda spacing; mutual coupling shifts the optimum)")
    return ratio, phase_diff


NEC2C_MAX_PATH = 80   # nec2c keeps the -i / -o file names in 81-byte buffers and aborts on longer ones


def run_nec2c(exe, deck, out, timeout=None):
    """Run nec2c on deck, writing out; returns the CompletedProcess (text, captured output).

    nec2c rejects file names longer than 80 characters, so it runs in the folder of the deck with
    bare file names; when those are still too long, or the output goes to another folder, the deck
    is copied to a short temporary name and the result copied back."""
    deck, out = Path(deck).resolve(), Path(out).resolve()
    if deck.parent == out.parent and max(len(deck.name), len(out.name)) <= NEC2C_MAX_PATH:
        return subprocess.run([exe, "-i", deck.name, "-o", out.name], cwd=str(deck.parent), text=True,
                              capture_output=True, timeout=timeout)
    with tempfile.TemporaryDirectory(prefix="nec") as td:
        shutil.copyfile(deck, Path(td) / "in.nec")
        r = subprocess.run([exe, "-i", "in.nec", "-o", "in.out"], cwd=td, text=True, capture_output=True,
                           timeout=timeout)
        if (Path(td) / "in.out").exists():
            shutil.copyfile(Path(td) / "in.out", out)
        return r


def nec2c_version(exe):
    """First line printed by `nec2c -v` (or --version); 'unknown' if it cannot be obtained."""
    for flag in ("-v", "--version"):
        try:
            r = subprocess.run([exe, flag], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        txt = ((r.stdout or "") + (r.stderr or "")).strip()
        if txt:
            return txt.splitlines()[0][:120]
    return "unknown"


# ======================================================================================
# Ideal L-network with typed components (L or C)
#
# Each branch is (kind, value) with kind 'L' (henries) or 'C' (farads) and value >= 0;
# a signed inductance is never used. The reactance of each branch depends on frequency
# according to its kind: L -> jwL, C -> 1/(jwC).
#
# Losses (optional): a network can carry "q_l" and "q_c" (quality factors of inductors
# and capacitors, constant Q over frequency). Each real component is modeled as the
# ideal one plus a series ESR equal to X/Q (X = reactance of the component at that
# frequency); in a shunt branch the ESR is in series with the component. Without Q
# (None) everything is ideal. lnet_design() always computes the lossless ideal network.
# ======================================================================================

EPS = 1e-12

E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]


def _q_of(kind, q_l, q_c):
    return q_l if kind == "L" else q_c


def _branch_z(kind, value, w, q):
    """Series impedance of an L or C component with ESR = |X|/Q (q None or inf: ideal)."""
    x = w * value if kind == "L" else -1.0 / (w * value)
    esr = abs(x) / q if q else 0.0
    return complex(esr, x)


def lnet_series_z(branch, w, q_l=None, q_c=None):
    kind, value = branch
    if value == 0:
        return 0j
    return _branch_z(kind, value, w, _q_of(kind, q_l, q_c))


def lnet_shunt_y(branch, w, q_l=None, q_c=None):
    kind, value = branch
    if value == 0:
        return 0j
    return 1 / _branch_z(kind, value, w, _q_of(kind, q_l, q_c))


def series_from_x(x, w):
    """Series branch with reactance x (ohm) at frequency w: L if x>0, C if x<0."""
    if abs(x) < EPS:
        return ("L", 0.0)
    return ("L", x / w) if x > 0 else ("C", -1.0 / (w * x))


def shunt_from_b(b, w):
    """Shunt branch with susceptance b (S): C if b>0, L if b<0."""
    if abs(b) < EPS:
        return ("C", 0.0)
    return ("C", b / w) if b > 0 else ("L", -1.0 / (w * b))


def lnet_design(z, z0, w):
    """Return dict(topology, series, shunt) that matches z to z0 at frequency w."""
    if z.real <= 0 or z0 <= 0:
        raise ValueError("R > 0 and Z0 > 0 are required")
    if z.real < z0:
        q = math.sqrt(z0 / z.real - 1)
        x_series = z.real * q - z.imag  # can be negative -> series capacitor
        return {"topology": "A", "series": series_from_x(x_series, w),
                "shunt": shunt_from_b(q / z0, w)}
    # R >= Z0: shunt element on the antenna side.
    y = 1 / z
    g, b = y.real, y.imag
    bt = math.sqrt(max(g * (1 / z0 - g), 0.0))  # low-pass solution (series L)
    x_series = bt * z0 / g if g > 0 else 0.0
    return {"topology": "B", "series": series_from_x(x_series, w),
            "shunt": shunt_from_b(bt - b, w)}


def lnet_design_all(z, z0, w):
    """Both L-networks that match z to z0 at w, same topology rule as lnet_design: the solution with the
    positive root first (the one lnet_design returns), then the one with the negative root."""
    if z.real <= 0 or z0 <= 0:
        raise ValueError("R > 0 and Z0 > 0 are required")
    sols = []
    if z.real < z0:
        q = math.sqrt(z0 / z.real - 1)
        for sgn in (+1, -1):
            sols.append({"topology": "A", "series": series_from_x(sgn * q * z.real - z.imag, w),
                         "shunt": shunt_from_b(sgn * q / z0, w)})
    else:
        y = 1 / z
        g, b = y.real, y.imag
        bt = math.sqrt(max(g * (1 / z0 - g), 0.0))
        for sgn in (+1, -1):
            sols.append({"topology": "B", "series": series_from_x(sgn * bt * z0 / g if g > 0 else 0.0, w),
                         "shunt": shunt_from_b(sgn * bt - b, w)})
    if sols[0]["series"] == sols[1]["series"] and sols[0]["shunt"] == sols[1]["shunt"]:
        sols = sols[:1]
    return sols


def lnet_kind_text(net):
    """'low-pass' (series L, shunt C), 'high-pass' (series C, shunt L) or the branch kinds."""
    kinds = (net["series"][0] if net["series"][1] else "-", net["shunt"][0] if net["shunt"][1] else "-")
    return {("L", "C"): "low-pass", ("C", "L"): "high-pass"}.get(kinds, f"series {kinds[0]}, shunt {kinds[1]}")


def lnet_select(z, z0, w, which="plus", table=None):
    """L-network chosen by --lnet-solution: 'plus' = positive root (lnet_design), 'minus' = negative root,
    'lowpass' / 'highpass' = the solution whose branch kinds are series L + shunt C / series C + shunt L (the plus /
    minus root when no solution has them), 'best' = the lower worst VSWR over table [(f_mhz, z_load)] when it has
    two or more points, otherwise the smaller spread between the series and shunt reactances.
    Returns (network, all solutions)."""
    sols = lnet_design_all(z, z0, w)
    if which in ("lowpass", "highpass"):
        want = "low-pass" if which == "lowpass" else "high-pass"
        match = [s for s in sols if lnet_kind_text(s) == want]
        if match:
            return match[0], sols
        which = "plus" if which == "lowpass" else "minus"
    if which == "minus" and len(sols) > 1:
        return sols[1], sols
    if which != "best" or len(sols) == 1:
        return sols[0], sols

    def worst_vswr(net):
        return max(lnet_metrics(lnet_zin(zl, 2 * math.pi * f * 1e6, net), z0)[0] for f, zl in table)

    def spread(net):
        xs = abs(lnet_series_z(net["series"], w).imag) or 1e-12
        ysh = abs(lnet_shunt_y(net["shunt"], w).imag) or 1e-12
        return abs(math.log(xs * ysh))

    key = worst_vswr if table and len(table) >= 2 else spread
    return min(sols, key=key), sols


def component_srf_mhz(kind, value, esl_h, cp_f):
    """Self-resonant frequency (MHz) of a real component: a capacitor C with total lead inductance esl_h, or an
    inductor L with stray capacitance cp_f. None for an absent branch."""
    if value <= 0:
        return None
    lc = esl_h * value if kind == "C" else value * cp_f
    return 1.0 / (2 * math.pi * math.sqrt(lc)) / 1e6 if lc > 0 else None


def srf_warnings(net, fmin_mhz, fmax_mhz, esl_nh_per_lead, coil_cp_pf):
    """Warnings for components whose self-resonance falls inside or near the band fmin..fmax (series and shunt,
    capacitors and inductors). A capacitor has two leads of esl_nh_per_lead each."""
    out = []
    for where, branch in (("series", net["series"]), ("shunt", net["shunt"])):
        kind, value = branch
        srf = component_srf_mhz(kind, value, 2.0 * esl_nh_per_lead * 1e-9, coil_cp_pf * 1e-12)
        if srf is None:
            continue
        what = (f"{where} capacitor {lnet_display(branch):.3f} pF (ESL {2 * esl_nh_per_lead:g} nH)" if kind == "C"
                else f"{where} inductor {lnet_display(branch):.3f} nH (stray C {coil_cp_pf:g} pF)")
        if srf <= fmax_mhz:
            out.append(f"WARNING: {what} self-resonates at about {srf:.1f} MHz, inside or below the band: above "
                       "its SRF a capacitor behaves as an inductor and an inductor as a capacitor. Use a part with a "
                       "higher SRF (smaller value, shorter leads, smaller coil).")
        elif srf < 1.5 * fmax_mhz:
            out.append(f"NOTE: {what} self-resonates at about {srf:.1f} MHz, less than 1.5 x the highest frequency "
                       f"({fmax_mhz:g} MHz); its effective value already differs from the nominal one in band.")
    return out


def lnet_zin(z_load, w, net):
    """Impedance seen from the Z0 side with load z_load at frequency w."""
    q_l, q_c = net.get("q_l"), net.get("q_c")
    zs, ys = lnet_series_z(net["series"], w, q_l, q_c), lnet_shunt_y(net["shunt"], w, q_l, q_c)
    if net["topology"] == "A":
        return 1 / (1 / (z_load + zs) + ys)
    return 1 / (1 / z_load + ys) + zs


def lnet_losses(z_load, w, net, z0):
    """Network losses (dB): (mismatch, dissipation, total) with Pin = accepted power.

    mismatch = -10log10(1-|Gamma|^2); dissipation = -10log10(P_load/P_accepted);
    total = sum = -10log10(P_load/P_available). With an ideal network the dissipation is 0.
    """
    q_l, q_c = net.get("q_l"), net.get("q_c")
    zs = lnet_series_z(net["series"], w, q_l, q_c)
    zi = lnet_zin(z_load, w, net)
    if net["topology"] == "A":
        i_load = 1 / (z_load + zs)          # V = 1 V at the input terminals
    else:
        i_load = (1 - zs / zi) / z_load     # voltage at the load node = 1 - I_in*zs
    p_in = (1 / zi).real
    p_load = abs(i_load) ** 2 * z_load.real
    gamma = abs((zi - z0) / (zi + z0))
    mism = -10 * math.log10(max(1 - gamma ** 2, 1e-12))
    diss = -10 * math.log10(max(p_load / p_in, 1e-12)) if p_in > 0 and p_load > 0 else float("inf")
    mism, diss = max(mism, 0.0) + 0.0, max(diss, 0.0) + 0.0   # avoid -0.000 from rounding
    return mism, diss, mism + diss


def lnet_metrics(z_in, z0):
    """Return (VSWR, return loss in dB, |Gamma|)."""
    gamma = (z_in - z0) / (z_in + z0)
    rho = abs(gamma)
    vswr = (1 + rho) / (1 - rho) if rho < 1 else float("inf")
    rl = -20 * math.log10(rho) if rho > 0 else float("inf")
    return vswr, rl, rho


def lnet_unit(branch):
    return "nH" if branch[0] == "L" else "pF"


def lnet_display(branch):
    """Value in nH (L) or pF (C)."""
    return branch[1] * (1e9 if branch[0] == "L" else 1e12)


def lnet_describe(branch):
    return f"{'inductor' if branch[0] == 'L' else 'capacitor'} {lnet_display(branch):.3f} {lnet_unit(branch)}"


def lnet_with_values(net, series_value, shunt_value):
    """Same topology and kinds as net with values in nH (L) or pF (C)."""
    def conv(branch, v):
        return (branch[0], v * (1e-9 if branch[0] == "L" else 1e-12))
    out = {"topology": net["topology"], "series": conv(net["series"], series_value),
           "shunt": conv(net["shunt"], shunt_value)}
    for key in ("q_l", "q_c"):
        if net.get(key) is not None:
            out[key] = net[key]
    return out


def lnet_with_q(net, q_l=None, q_c=None):
    """Copy of net with quality factors (None = ideal)."""
    out = dict(net)
    out["q_l"], out["q_c"] = q_l, q_c
    return out


def lnet_topology_text(net):
    if net["topology"] == "A":
        return "A: series on the antenna side, shunt on the Z0 side"
    return "B: shunt on the antenna side, series on the Z0 side"


def nearest_e24(value):
    """E24 value closest (on a logarithmic scale) to a positive value."""
    decade = 10 ** math.floor(math.log10(value))
    cands = [m * decade * k for m in E24 for k in (1, 10)] + [10 * decade]
    return min(cands, key=lambda c: abs(math.log(c / value)))


# ======================================================================================
# NEC model generator
# ======================================================================================

TL_MIN_LENGTH_M = 1e-6   # NEC reads a TL of length 0 as the straight-line distance between the ports
TL_MIN_ELEC_DEG = 1e-6

SQUARE_EQ_RADIUS = 0.59   # equivalent radius of a square conductor of side a: 0.59*a

# T-match feed (--feed-style tmatch): taps on both driven elements, bars toward the other element.
TAG_TMATCH_REAR = 3
TAG_TMATCH_FRONT = 4
DEFAULT_TAP_MM = 120.0
DEFAULT_TAP_OFFSET_MM = 50.0
DEFAULT_SEGMENTS = 21
DEFAULT_SEGMENTS_TMATCH = 41    # the T junctions need finer segments for an accurate current solution
# Comment card carrying the series feed capacitor; nec2c echoes it in the .out file, where the Zin readers find it.
SERIES_C_MARK = "HB9CV_SERIES_C_PF="
SERIES_C_RE = re.compile(rf"{SERIES_C_MARK}({FLOAT})")
SERIES_C_Q_MARK = "HB9CV_SERIES_C_Q="
SERIES_C_Q_RE = re.compile(rf"{SERIES_C_Q_MARK}({FLOAT})")
DEFAULT_SERIES_C_Q = 500.0


def series_c_value(args):
    """Series feed capacitor in pF from --series-c-pf when it is a number (> 0); None otherwise ('auto' is
    resolved into a number before the deck is written)."""
    v = getattr(args, "series_c_pf", None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def series_c_reactance(c_pf, freq_mhz):
    """Reactance (ohm, negative) of a c_pf capacitor at freq_mhz."""
    return -1.0 / (2 * math.pi * freq_mhz * 1e6 * c_pf * 1e-12)


def series_c_q(args):
    """Quality factor of the series feed capacitor (--series-c-q; 0 = lossless)."""
    q = getattr(args, "series_c_q", None)
    return DEFAULT_SERIES_C_Q if q is None else float(q)


def series_c_impedance(c_pf, freq_mhz, q=0.0):
    """Series impedance (ohm) of a c_pf capacitor at freq_mhz with ESR = |X|/Q (q <= 0: lossless)."""
    x = series_c_reactance(c_pf, freq_mhz)
    return complex(abs(x) / q if q and q > 0 else 0.0, x)


def series_c_for_x(x_ohm, freq_mhz):
    """Series capacitor (pF) that cancels a positive input reactance x_ohm at freq_mhz; None when x_ohm <= 0."""
    if not x_ohm > 0:
        return None
    return 1e12 / (2 * math.pi * freq_mhz * 1e6 * x_ohm)


def feed_distance_m(args, spacing_m):
    """Straight distance between the two points the phase network connects: the element centers, or the two
    T-bar centers with --feed-style tmatch."""
    if getattr(args, "feed_style", "center") == "tmatch" and args.network in ("tl", "lc"):
        return spacing_m - 2.0 * (getattr(args, "tap_offset_mm", None) or DEFAULT_TAP_OFFSET_MM) / 1000.0
    return spacing_m


def wrap180(x):
    """Normalize an angle to (-180, 180]."""
    x = (x + 180.0) % 360.0 - 180.0
    return 180.0 if x == -180.0 else x


def theory_rear_phase(spacing_factor):
    """Theoretical phase of the rear excitation relative to the front one (degrees)."""
    return wrap180(180.0 - 360.0 * spacing_factor)


def theory_line_lag(feed_element, spacing_factor):
    """Theoretical delay (degrees, 0..360) of the far element's current relative to the
    fed element, for a cardioid with the beam toward +x."""
    kd = 360.0 * spacing_factor
    if feed_element == "front":
        # The rear element must lead the front one by (180 - kd): delay kd + 180.
        return (kd + 180.0) % 360.0
    # Feeding the rear element, the front one must lag by (180 - kd).
    return (180.0 - kd) % 360.0


def line_electrical_deg(target_lag_deg, crossed):
    """Electrical length of the line (degrees, 0..360) that produces the requested
    delay; a crossed line contributes an additional 180 degrees."""
    return (target_lag_deg - (180.0 if crossed else 0.0)) % 360.0


def line_phys_length(elec_deg, lam, vf):
    """Physical length in meters: (degrees/360) * lambda * VF."""
    return (elec_deg / 360.0) * lam * vf


def lc_pi_values(elec_deg, z0, freq_mhz):
    """Series L (H) and C per port (F) of an ideal low-pass pi network that delays by
    elec_deg degrees with characteristic impedance z0; ValueError outside 0..180."""
    if not 0.0 < elec_deg < 180.0:
        raise ValueError(f"a low-pass LC pi network only realizes delays between 0 and 180 degrees "
                         f"(requested: {elec_deg:.1f}); use --crossed (default) to subtract 180 degrees "
                         "from the target delay, or set --line-deg")
    w = 2 * math.pi * freq_mhz * 1e6
    phi = math.radians(elec_deg)
    return z0 * math.sin(phi) / w, math.tan(phi / 2.0) / (z0 * w)


def resolve_line(args, lam):
    """Return (target delay, electrical degrees, physical length in m)."""
    if args.phase_target_deg is not None:
        target = args.phase_target_deg % 360.0
    else:
        target = theory_line_lag(args.feed_element, args.spacing_factor)
    if args.coax_length is not None and args.network == "tl":
        phys = args.coax_length
        elec = 360.0 * (phys / args.coax_vf) / lam
    elif args.line_deg is not None:
        elec = args.line_deg
        phys = line_phys_length(elec, lam, args.coax_vf)
    else:
        elec = line_electrical_deg(target, args.crossed)
        phys = line_phys_length(elec, lam, args.coax_vf)
    return target, elec, phys


def line_min_elec_deg(spacing_m, lam, vf, allowance_m=0.0):
    """Shortest electrical length (degrees) of a straight line that spans the two feed points
    (spacing_m apart) plus allowance_m of connections, at velocity factor vf."""
    return 360.0 * (spacing_m + allowance_m) / (vf * lam)


def line_feasibility_warnings(args, lam, spacing_m, line_elec_deg):
    """Texts for a TL phase line physically shorter than the distance it has to span."""
    if args.network != "tl":
        return []
    allow = (getattr(args, "line_allowance_mm", 0.0) or 0.0) / 1000.0
    need = line_min_elec_deg(spacing_m, lam, args.coax_vf, allow)
    if line_elec_deg + 1e-9 < need:
        return [f"phase line {line_elec_deg:.1f} deg ({line_elec_deg / 360.0 * lam * args.coax_vf * 1000:.1f} mm) is "
                f"shorter than the straight distance between the feed points"
                + (f" plus {allow * 1000:g} mm of connections" if allow else "")
                + f" ({need:.1f} deg, {(spacing_m + allow) * 1000:.1f} mm, needed at VF={args.coax_vf:g}): the model "
                "cannot be built as simulated. Raise the electrical length, raise VF, or reduce the element spacing."]
    return []


def boom_radius(shape, size_mm):
    """Radius (m) of the round conductor equivalent to the boom (size_mm = side or diameter)."""
    return (SQUARE_EQ_RADIUS * size_mm / 1000.0) if shape == "square" else size_mm / 2000.0


def boom_segments(length, seg_target, radius=0.0, lam=None):
    """Segments of a straight span (boom, mast) whose segment length is close to seg_target but never shorter than
    4 radii (the thin-wire kernel needs a segment much longer than the wire radius) nor longer than lambda/10."""
    seg = max(seg_target, 4.0 * radius)
    n = int(math.floor(length / seg + 1e-9)) if radius > 0 else int(math.ceil(length / seg - 1e-9))
    if lam:
        n = max(n, int(math.ceil(length / (lam / 10.0) - 1e-9)))
    return max(1, n)


IDEAL_DRIVE_X = 1.0e6                # ohm: series reactance that turns an EX voltage source into a current source
DEFAULT_BOOM_CLEARANCE_MM = 10.0   # surface-to-surface gap between the elements and a modeled boom
CLEARANCE_WARN_FACTOR = 0.5        # warn when two unconnected wires are closer than this x the larger radius


def _segment_distance(p1, p2, q1, q2):
    """Shortest distance between the 3D segments p1-p2 and q1-q2."""
    return _closest_points(p1, p2, q1, q2)[0]


def _closest_points(p1, p2, q1, q2):
    """(distance, s, t, cp, cq): closest points cp = p1 + s (p2 - p1) and cq = q1 + t (q2 - q1) of two segments."""
    d1 = [b - a for a, b in zip(p1, p2)]
    d2 = [b - a for a, b in zip(q1, q2)]
    r = [a - b for a, b in zip(p1, q1)]
    a = sum(v * v for v in d1)
    e = sum(v * v for v in d2)
    f = sum(x * y for x, y in zip(d2, r))
    if a <= 1e-18 and e <= 1e-18:
        return math.sqrt(sum(v * v for v in r))
    if a <= 1e-18:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = sum(x * y for x, y in zip(d1, r))
        if e <= 1e-18:
            s, t = min(max(-c / a, 0.0), 1.0), 0.0
        else:
            b = sum(x * y for x, y in zip(d1, d2))
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    cp = [p + s * v for p, v in zip(p1, d1)]
    cq = [q + t * v for q, v in zip(q1, d2)]
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(cp, cq))), s, t, cp, cq


def _surface_gap(p1, p2, rp, q1, q2, rq):
    """Surface-to-surface gap between two wires (cylinders with flat ends): the radius of a wire counts fully where
    the closest point is inside it and only by its component across the gap where the closest point is a flat end."""
    dist, s, t, cp, cq = _closest_points(p1, p2, q1, q2)
    if dist <= 1e-12:
        return -(rp + rq)
    v = [b - a for a, b in zip(cp, cq)]

    def eff(r, a, b, par):
        if 1e-9 < par < 1 - 1e-9:
            return r
        d = [y - x for x, y in zip(a, b)]
        dl = math.sqrt(sum(x * x for x in d)) or 1.0
        cosang = abs(sum(x * y for x, y in zip(d, v))) / (dl * dist)
        return r * math.sqrt(max(0.0, 1.0 - cosang * cosang))

    return dist - eff(rp, p1, p2, s) - eff(rq, q1, q2, t)


def wire_clearance_problems(wires, tag_names=None):
    """Check every pair of wires of different tags that do not share an end point.

    wires: [(tag, p1, p2, radius)]. Returns (errors, warnings): an error when the wires overlap (distance between
    axes below the sum of the radii), a warning when the surface-to-surface gap is below CLEARANCE_WARN_FACTOR times
    the larger radius (NEC-2 evaluates the fields on the wire axes, so closely spaced conductors couple wrongly)."""
    tag_names = tag_names or {}
    errors, warnings = [], []
    ends = [(tuple(round(v, 7) for v in w[1]), tuple(round(v, 7) for v in w[2])) for w in wires]
    for i in range(len(wires)):
        ti, p1, p2, ri = wires[i]
        for j in range(i + 1, len(wires)):
            tj, q1, q2, rj = wires[j]
            if ti == tj or set(ends[i]) & set(ends[j]):
                continue
            gap = _surface_gap(p1, p2, ri, q1, q2, rj)
            if gap < 0.0:
                errors.append(f"{tag_names.get(ti, f'tag {ti}')} and {tag_names.get(tj, f'tag {tj}')} overlap "
                              f"(surfaces {-gap * 1000:.1f} mm into each other; radii {ri * 1000:.1f} and "
                              f"{rj * 1000:.1f} mm)")
            elif gap < CLEARANCE_WARN_FACTOR * max(ri, rj):
                warnings.append(f"{tag_names.get(ti, f'tag {ti}')} and {tag_names.get(tj, f'tag {tj}')} are "
                                f"{gap * 1000:.1f} mm apart surface to surface (below {CLEARANCE_WARN_FACTOR:g} x the "
                                f"larger radius): NEC-2 coupling between them is inaccurate")
    return errors, sorted(set(warnings))


def element_sections(args, which, default_mm=None):
    """Sections [(length_mm or None, diameter_mm)] of ONE half of the 'rear' or 'front' element, center outward.
    Priority: --rear/front-sections, then --rear/front-diameter, then the common diameter (default_mm or
    --element-diameter). The last section always runs to the tip."""
    secs = getattr(args, f"{which}_sections", None)
    if secs:
        return list(secs)
    d = getattr(args, f"{which}_diameter", None)
    if not d:
        d = default_mm if default_mm is not None else args.element_diameter
    return [(None, d)]


def sections_text(secs):
    """Human text of a section list: '12 mm for 300 mm, 8 mm to the tip'."""
    if len(secs) == 1:
        return f"{secs[0][1]:g} mm"
    return ", ".join(f"{d:g} mm for {l:g} mm" if l is not None else f"{d:g} mm to the tip" for l, d in secs)


def element_profile(secs, half, name="element"):
    """[(a_from, a_to, radius_m)] of one half (meters from the element center, outward). The last section takes
    whatever is left up to the tip (half = half length in m); ValueError if the fixed sections do not fit."""
    prof, cum = [], 0.0
    for i, (length, diam) in enumerate(secs):
        a1 = half if i == len(secs) - 1 else cum + length / 1000.0
        if a1 - cum < 1e-4:
            raise ValueError(f"{name} element: the stepped sections leave {max(0.0, a1 - cum) * 1000:.2f} mm for "
                             f"section {i + 1} (half length {half * 1000:.1f} mm): shorten the fixed sections")
        prof.append((cum, a1, diam / 2000.0))
        cum = a1
    return prof


def feed_gap_problems(gap_m, radius_m, lam):
    """Reasons why a modeled feed gap (a one-segment bridge wire of length gap_m and the tube radius) breaks
    the thin-wire rules: shorter than 4 radii (2 diameters) or shorter than lambda/200."""
    out = []
    if gap_m < 4.0 * radius_m:
        out.append(f"--model-feed-gap: the {gap_m * 1000:g} mm bridge wire is shorter than 4 tube radii "
                   f"({4000 * radius_m:.1f} mm); widen --feed-gap-mm or model thinner tubes")
    if gap_m < lam / 200.0:
        out.append(f"--model-feed-gap: the {gap_m * 1000:g} mm bridge wire is shorter than lambda/200 "
                   f"({lam / 200.0 * 1000:.1f} mm); widen --feed-gap-mm")
    return out


def wire_chain_warnings(name, chain, lam):
    """Thin-wire checks for the consecutive wires of one element; chain = [(length_m, nseg, radius_m)].
    NEC-2 wants segment length >= ~2 radii (a note below 4 radii, where the thin-wire kernel loses accuracy,
    mostly at junctions), segments <= lambda/10,
    no abrupt radius change (> 2:1) or segment-length jump (> 5:1) between neighbouring wires."""
    out = []
    lbl = name if name in ("boom", "mast") else f"{name} element"
    for i, (ln, k, r) in enumerate(chain):
        seg = ln / k
        if seg < 2.0 * r:
            out.append(f"{lbl}, wire {i + 1}: segment {seg * 1000:.1f} mm is shorter than 2 radii "
                       f"({2 * r * 1000:.1f} mm): thin-wire approximation not valid; reduce --segments or the diameter")
        elif seg < 4.0 * r:
            out.append(f"{lbl}, wire {i + 1}: diameter {2 * r * 1000:.1f} mm is large relative to the "
                       f"segment length {seg * 1000:.1f} mm (segment < 4 radii): the thin-wire kernel loses accuracy, "
                       "mostly at junctions and bends; check the average gain test")
        if seg < lam / 200.0:
            out.append(f"{lbl}, wire {i + 1}: segment {seg * 1000:.1f} mm is shorter than lambda/200 "
                       f"({lam / 200.0 * 1000:.1f} mm): reduce --segments")
        if seg > lam / 10.0:
            out.append(f"{lbl}, wire {i + 1}: segment {seg * 1000:.0f} mm exceeds lambda/10: increase --segments")
        if i:
            lp, kp, rp = chain[i - 1]
            sp = lp / kp
            if max(r, rp) / min(r, rp) > 2.0:
                out.append(f"{lbl}, wires {i}/{i + 1}: radius jump {max(r, rp) / min(r, rp):.1f}:1 (NEC "
                           "loses accuracy above ~2:1)")
            if max(seg, sp) / min(seg, sp) > 5.0:
                out.append(f"{lbl}, wires {i}/{i + 1}: segment length jump {max(seg, sp) / min(seg, sp):.1f}:1 "
                           "(keep neighbouring segments within ~5:1)")
    return out


# ======================================================================================
# Parasitic elements: up to MAX_REFLECTORS reflectors behind the rear element and up to
# MAX_DIRECTORS directors in front of the front element (passive, no source)
# ======================================================================================

MAX_REFLECTORS = 2
MAX_DIRECTORS = 5
TAG_REFLECTOR0 = 20            # reflector i (0-based) -> NEC tag 21, 22
TAG_DIRECTOR0 = 30             # director i (0-based) -> NEC tag 31 .. 35
# Starting ("nominal") values, in length factors of the resonant dipole length (see resonant_length_m) and in
# wavelengths of spacing from the previous element (reflectors: from the rear element, directors: from the front
# element, later ones from the previous parasitic element).
REFLECTOR_NOMINAL_LEN = (1.05, 1.05)
REFLECTOR_NOMINAL_SP = (0.15, 0.20)
DIRECTOR_NOMINAL_LEN = (0.96, 0.95, 0.94, 0.93, 0.92)
DIRECTOR_NOMINAL_SP = (0.15, 0.20, 0.25, 0.30, 0.30)
# Hard limits used as the optimizer's bounds. A reflector must be inductive (not shorter than resonance), a
# director capacitive (not longer than resonance); beyond the lower director bound the element no longer couples.
REFLECTOR_LEN_RANGE = (1.00, 1.15)
DIRECTOR_LEN_RANGE = (0.80, 1.00)
PAR_MIN_SPACING_LAMBDA = 0.05  # electrical floor: closer than this the coupling is too tight and the bandwidth collapses
PAR_MAX_SPACING_LAMBDA = 0.40  # beyond this the element no longer shapes the pattern
DEFAULT_MIN_GAP_MM = 20.0      # surface-to-surface room for the element clamps / insulators
DEFAULT_BOOM_OVERHANG_MM = 50.0


def resonant_length_m(freq_mhz, diam_mm):
    """Tip-to-tip length (m) of a free-space half-wave dipole resonant at freq_mhz, made of a tube of diam_mm.
    Empirical end-effect factor K = 0.97 at d/lambda = 1e-3 falling to 0.945 at 1e-2 (clamped to d/lambda in
    5e-4..3e-2): a starting value for the optimizer, which refines it with NEC."""
    lam = C_LIGHT / (freq_mhz * 1e6)
    ratio = min(max(diam_mm / 1000.0 / lam, 5e-4), 3e-2)
    k = 0.945 + 0.025 * (-math.log10(ratio) - 2.0)
    return 0.5 * lam * k


def parasitic_limits(role, freq_mhz, par_diam_mm, prev_diam_mm, min_gap_mm):
    """Minimum / maximum length factor and spacing (wavelengths) of a parasitic element.
    Minimum spacing = max(electrical floor, mechanical room: radii of both tubes + the clamp gap)."""
    lam = C_LIGHT / (freq_mhz * 1e6)
    lo, hi = REFLECTOR_LEN_RANGE if role == "reflector" else DIRECTOR_LEN_RANGE
    mech_m = (par_diam_mm + prev_diam_mm) / 2000.0 + min_gap_mm / 1000.0
    sp_min = math.ceil(max(PAR_MIN_SPACING_LAMBDA, mech_m / lam) * 1e4) / 1e4
    return {"len_min": lo, "len_max": hi, "sp_min": sp_min, "sp_max": PAR_MAX_SPACING_LAMBDA}


def parasitic_layout(args, freq_mhz, x_rear, x_front, rear_diam_mm, front_diam_mm):
    """Parasitic elements in boom order (reflectors first, then directors) as a list of dicts.
    x_rear / x_front: positions (m) of the two driven elements. Values missing from --reflector-lengths /
    --reflector-spacings / --director-lengths / --director-spacings come from the nominal tables."""
    lam = C_LIGHT / (freq_mhz * 1e6)
    nr = int(getattr(args, "reflectors", 0) or 0)
    nd = int(getattr(args, "directors", 0) or 0)
    par_d = getattr(args, "parasitic_diameter", None) or args.element_diameter
    gap = getattr(args, "min_gap_mm", None)
    gap = DEFAULT_MIN_GAP_MM if gap is None else gap
    l_res = resonant_length_m(freq_mhz, par_d)
    out = []
    for role, n, lens, sps, nom_l, nom_s, x0, d0, sign, tag0 in (
            ("reflector", nr, getattr(args, "reflector_lengths", None), getattr(args, "reflector_spacings", None),
             REFLECTOR_NOMINAL_LEN, REFLECTOR_NOMINAL_SP, x_rear, rear_diam_mm, -1.0, TAG_REFLECTOR0),
            ("director", nd, getattr(args, "director_lengths", None), getattr(args, "director_spacings", None),
             DIRECTOR_NOMINAL_LEN, DIRECTOR_NOMINAL_SP, x_front, front_diam_mm, +1.0, TAG_DIRECTOR0)):
        x, prev_d = x0, d0
        for i in range(n):
            lf = lens[i] if lens and i < len(lens) else nom_l[i]
            sp = sps[i] if sps and i < len(sps) else nom_s[i]
            lim = parasitic_limits(role, freq_mhz, par_d, prev_d, gap)
            x = x + sign * sp * lam
            out.append({
                "name": f"{role} {i + 1}", "role": role, "index": i, "tag": tag0 + 1 + i,
                "x_rear_m": x_rear, "x_front_m": x_front,
                "len_factor": lf, "length_m": lf * l_res, "half_m": lf * l_res / 2.0, "l_res_m": l_res,
                "spacing_lambda": sp, "spacing_m": sp * lam, "x": x, "diam_mm": par_d, "prev_diam_mm": prev_d,
                "len_min_m": lim["len_min"] * l_res, "len_nom_m": nom_l[i] * l_res, "len_max_m": lim["len_max"] * l_res,
                "sp_min_m": lim["sp_min"] * lam, "sp_nom_m": nom_s[i] * lam, "sp_max_m": lim["sp_max"] * lam,
                "limits": lim})
            prev_d = par_d
    return out


def parasitic_warnings(layout):
    """Texts for parasitic elements outside their pre-calculated limits or mechanically impossible."""
    out = []
    for e in layout:
        lim = e["limits"]
        if e["len_factor"] < lim["len_min"] - 1e-9 or e["len_factor"] > lim["len_max"] + 1e-9:
            out.append(f"{e['name']}: length {e['length_m'] * 1000:.1f} mm is outside the pre-calculated range "
                       f"{e['len_min_m'] * 1000:.1f}..{e['len_max_m'] * 1000:.1f} mm "
                       f"({'a reflector must not be shorter than resonance' if e['role'] == 'reflector' else 'a director must not be longer than resonance'} "
                       "for the intended role)")
        if e["spacing_lambda"] < lim["sp_min"] - 1e-9:
            out.append(f"{e['name']}: spacing {e['spacing_m'] * 1000:.1f} mm is below the minimum "
                       f"{e['sp_min_m'] * 1000:.1f} mm (electrical floor {PAR_MIN_SPACING_LAMBDA:g} lambda or tube "
                       "radii + clamp gap)")
        elif e["spacing_lambda"] > lim["sp_max"] + 1e-9:
            out.append(f"{e['name']}: spacing {e['spacing_m'] * 1000:.1f} mm exceeds {PAR_MAX_SPACING_LAMBDA:g} lambda; "
                       "the element has little effect on the pattern")
    return out


def required_boom_mm(layout, x_rear, x_front, overhang_mm):
    """Minimum boom length (mm) holding every element plus the overhang at both ends."""
    xs = [x_rear, x_front] + [e["x"] for e in layout]
    return (max(xs) - min(xs)) * 1000.0 + 2.0 * overhang_mm


def parasitic_report(layout, boom_mm=None):
    """Text lines of the pre-calculated / current dimensions of the parasitic elements (lengths tip to tip)."""
    if not layout:
        return []
    l_res = layout[0]["l_res_m"] * 1000.0
    out = [f"Parasitic elements (tip-to-tip lengths; resonant dipole of a {layout[0]['diam_mm']:g} mm tube = "
           f"{l_res:.1f} mm):",
           f"  {'element':<12s}{'length mm':>10s}{'min':>9s}{'nominal':>9s}{'max':>9s}"
           f"{'spacing mm':>12s}{'min':>8s}{'nominal':>9s}{'pos. mm':>10s}"]
    x_ref = min([e["x"] for e in layout] + [layout[0]["x_rear_m"]])   # the rear-most element of the antenna
    for e in layout:
        out.append(f"  {e['name']:<12s}{e['length_m'] * 1000:10.1f}{e['len_min_m'] * 1000:9.1f}"
                   f"{e['len_nom_m'] * 1000:9.1f}{e['len_max_m'] * 1000:9.1f}{e['spacing_m'] * 1000:12.1f}"
                   f"{e['sp_min_m'] * 1000:8.1f}{e['sp_nom_m'] * 1000:9.1f}{(e['x'] - x_ref) * 1000:10.1f}")
    if boom_mm is not None:
        out.append(f"  pos. = distance from the rear-most element of the antenna. Boom length used (holds every element + overhang): {boom_mm:.0f} mm")
    return out


def sweep_count(fmin, fmax, fstep):
    """Number of points of the uniform sweep (not exceeding fmax) and the last point reached."""
    n = int(math.floor((fmax - fmin) / fstep + 1e-9)) + 1
    return n, fmin + (n - 1) * fstep


def sweep_needs_fmax(fmax, flast):
    """True when fmax is not a point of the uniform sweep and nec2c prints it differently from the last point
    (frequencies closer than the five printed digits would make two indistinguishable blocks)."""
    return fmax - flast > max(1e-6 * max(fmax, 1.0), 2.0 * freq_print_tol(fmax))


def sweep_frequencies(fmin, fmax, fstep):
    """Frequencies (MHz) of the --fsweep range, with fmax appended when it is not a multiple of the step
    (same points as the FR cards of the Zin-only sweep)."""
    n, flast = sweep_count(fmin, fmax, fstep)
    freqs = [fmin + i * fstep for i in range(n)]
    if sweep_needs_fmax(fmax, flast):
        freqs.append(fmax)
    return freqs


GROUND_PRESETS = {"average": (13.0, 0.005), "good": (20.0, 0.02)}   # (relative permittivity, S/m)


def ground_card(args):
    """GN card for --ground perfect / average / good (finite ground: Sommerfeld-Norton, GN 2)."""
    kind = args.ground
    if kind == "perfect":
        return "GN 1"
    eps, sigma = getattr(args, "ground_eps", None), getattr(args, "ground_sigma", None)
    if eps is None or sigma is None:
        eps0, sigma0 = GROUND_PRESETS[kind]
        eps = eps0 if eps is None else eps
        sigma = sigma0 if sigma is None else sigma
    return f"GN 2 0 0 0 {eps:.6g} {sigma:.6g}"


def build_deck(args):
    f = args.freq
    lam = C_LIGHT / (f * 1e6)
    # Initial reference lengths: 150/f (0.50 lambda) and 139/f (~0.4637 lambda).
    # No end-effect or thick-wire correction: the optimizer adjusts them (see --refine).
    rear_len = (150.0 / f) * args.rear_length_factor
    front_len = (139.0 / f) * args.front_length_factor
    spacing = args.spacing_factor * lam
    rear_secs, front_secs = element_sections(args, "rear"), element_sections(args, "front")
    rad_rear, rad_front = rear_secs[0][1] / 2000.0, front_secs[0][1] / 2000.0    # radius at the element center
    rad = max(d for _, d in rear_secs + front_secs) / 2000.0                    # largest element radius (offsets)
    gap_m = (getattr(args, "gap_mm", 0.0) or 0.0) / 1000.0
    model_gap = bool(getattr(args, "model_feed_gap", False)) and gap_m > 0
    wire_warnings = []
    boom_rad = boom_radius(args.boom_shape, args.boom_size)
    boom_isolated = getattr(args, "boom_isolated", False)
    feedline_len = getattr(args, "feedline_length", 0.0) or 0.0
    use_feedline = feedline_len > 0 and args.network in ("tl", "lc")
    clr = getattr(args, "boom_clearance_mm", None)
    clr = (DEFAULT_BOOM_CLEARANCE_MM if clr is None else clr) / 1000.0
    half_rear, half_front = rear_len / 2, front_len / 2
    n = args.segments
    # Parasitic elements (reflectors behind the rear element, directors in front of the front one) and the boom
    # that has to hold them: the boom is centered on the span of all elements (the mast sits at its center).
    par = parasitic_layout(args, f, -spacing / 2, spacing / 2, rear_secs[0][1], front_secs[0][1])
    feed_style = getattr(args, "feed_style", "center") or "center"
    tmatch = feed_style == "tmatch" and args.network in ("tl", "lc")
    contact = args.mount == "contact"
    if contact and not tmatch:
        raise ValueError("--mount contact bonds the element centers to a metallic boom: it needs --feed-style tmatch "
                         "(with a center feed the bond would short the feed points)")
    boom_isolated = boom_isolated or contact
    for e in par:
        if e["spacing_m"] <= (e["prev_diam_mm"] + e["diam_mm"]) / 2000.0 * 1.05 + 0.001:
            raise ValueError(f"{e['name']}: spacing {e['spacing_m'] * 1000:.1f} mm leaves no room between the tubes")
    xs_all = [-spacing / 2, spacing / 2] + [e["x"] for e in par]
    boom_xc = 0.0
    boom_len_m = args.boom_length / 1000.0
    if par:
        boom_xc = 0.5 * (min(xs_all) + max(xs_all))
        overhang_mm = getattr(args, "boom_overhang_mm", None)
        overhang_mm = DEFAULT_BOOM_OVERHANG_MM if overhang_mm is None else overhang_mm
        boom_len_m = max(boom_len_m, (max(xs_all) - min(xs_all)) + 2.0 * overhang_mm / 1000.0)
    half_max = max([half_rear, half_front] + [e["half_m"] for e in par])
    tap_m = (getattr(args, "tap_mm", None) or DEFAULT_TAP_MM) / 1000.0
    tap_off_m = (getattr(args, "tap_offset_mm", None) or DEFAULT_TAP_OFFSET_MM) / 1000.0
    # Per-element T-match dimensions (the Monte-Carlo perturbs each element on its own); default: the common values.
    taps = {w: (getattr(args, f"tap_mm_{w}", None) or tap_m * 1000.0) / 1000.0 for w in ("rear", "front")}
    tap_offs = {w: (getattr(args, f"tap_offset_mm_{w}", None) or tap_off_m * 1000.0) / 1000.0
                for w in ("rear", "front")}
    # The bar and stubs default to the element diameter: NEC-2 loses accuracy at a junction of unequal radii.
    tap_rad = (getattr(args, "tap_wire_mm", None) or min(rear_secs[0][1], front_secs[0][1])) / 2000.0
    if tmatch:
        if len(rear_secs) > 1 or len(front_secs) > 1:
            raise ValueError("--feed-style tmatch needs uniform elements (no --rear-sections/--front-sections)")
        if model_gap:
            raise ValueError("--feed-style tmatch uses continuous elements: do not combine it with --model-feed-gap")
        seg_min = 2.0 * min(half_rear, half_front) / n
        for t in taps.values():
            if t + seg_min > min(half_rear, half_front):
                raise ValueError(f"--tap-mm {t * 1000:g} reaches the element tip (half length "
                                 f"{min(half_rear, half_front) * 1000:.0f} mm)")
            if t < 1.5 * seg_min:
                raise ValueError(f"--tap-mm {t * 1000:g} is shorter than 1.5 element segments "
                                 f"({1.5 * seg_min * 1000:.0f} mm): raise it or use --feed-style center")
        if tap_offs["rear"] + tap_offs["front"] >= spacing - 0.01:
            raise ValueError("--tap-offset-mm leaves no room for the phase line between the two T-bars")
    # Orientation and height. Geometry coordinates: x along the boom (beam toward +x), "a" along the
    # elements, "o" offset from the element plane (boom / feed-line wire side). Horizontal: elements
    # along Y, offsets along Z, element centers at z = height. Vertical: elements along Z, offsets
    # along Y; the LOWEST tip of the longest element (driven or parasitic) is at z = height.
    vertical = getattr(args, "orientation", "horizontal") == "vertical"
    ground = getattr(args, "ground", "free")
    height = getattr(args, "height_m", None) or 0.0
    zc = (height + half_max) if vertical else height
    z_low = []
    all_wires = []
    # Offset of a modeled boom axis from the element plane: the element (or T-bar) and boom surfaces are
    # --boom-clearance-mm apart. Every conductor routed below the elements uses the same reference.
    zb = -(rad + boom_rad + clr)

    def xyz(x, a, o=0.0):
        return (x, o, zc + a) if vertical else (x, a, zc + o)

    def gw(tag, nseg, p1, p2, radius):
        z_low.append(min(p1[2], p2[2]) - radius)
        all_wires.append((tag, p1, p2, radius))
        coords = " ".join(f"{v:.8f}" for v in (*p1, *p2))
        return f"GW {tag} {nseg} {coords} {radius:.8f}"

    def build_element(tag, x, half, secs, name, use_gap=True, split_center=False):
        """GW cards of one element (same tag for every wire; segments are numbered along the tag from -tip to
        +tip). Returns (cards, central_segment). Without sections or modeled gap this is the single n-segment
        wire. Stepped: a central wire (odd segment count, holds the source) plus one wire per section on each
        side. Modeled gap: a 1-segment bridge wire of length g at the center (carries EX/TL; NEC needs a
        connected segment for a source) and the halves from +-g/2 outward. split_center=True splits the central
        wire at the element center (a junction point for the boom bond of --mount contact)."""
        prof = element_profile(secs, half, name)
        r1, a1 = prof[0][2], prof[0][1]
        outer = list(prof[1:])
        g2 = gap_m / 2.0 if (model_gap and use_gap) else 0.0
        if g2 > 0:
            if g2 >= a1 - 1e-4:
                raise ValueError(f"--feed-gap-mm {gap_m * 1000:g} leaves no conductor in the first section of "
                                 f"the {name} element ({a1 * 1000:.1f} mm from the center)")
            for txt in feed_gap_problems(gap_m, r1, lam):
                raise ValueError(f"{name} element: {txt}")
            outer.insert(0, (g2, a1, r1))
            seg_ref = (half - g2) / ((n - 1) / 2.0)
            center = (-g2, g2, 1)
        else:
            seg_ref = 2.0 * half / n
            center = (-a1, a1, max(1, 2 * int(round((2.0 * a1 / seg_ref - 1.0) / 2.0)) + 1))
        nks = [max(1, int(round((b - a) / seg_ref))) for a, b, _ in outer]
        pieces = [(-b, -a, r, k) for (a, b, r), k in reversed(list(zip(outer, nks)))]
        if split_center and g2 == 0:
            kh = (center[2] + 1) // 2
            pieces += [(center[0], 0.0, r1, kh), (0.0, center[1], r1, kh)]
            center = (center[0], center[1], 2 * kh - 1)
        else:
            pieces.append((center[0], center[1], r1, center[2]))
        pieces += [(a, b, r, k) for (a, b, r), k in zip(outer, nks)]
        cards = [gw(tag, k, xyz(x, a), xyz(x, b), r) for a, b, r, k in pieces]
        wire_warnings.extend(wire_chain_warnings(name, [(b - a, k, r) for a, b, r, k in pieces], lam))
        return cards, sum(nks) + center[2] // 2 + 1

    def build_tmatch_element(tag, bar_tag, x, half, r_el, name, toward, split=False, split_center=False):
        """T-match driven element: a continuous tube split at the two tap points (so the stubs join segment ends)
        plus, on bar_tag, a stub from each tap to a bar parallel to the element at the bar offset toward the other
        element (toward = +1 or -1 along x). The bar centre segment is the network/source port; split=True builds
        that segment as its own one-segment wire, so the common-mode shield wire can join its -a end;
        split_center=True splits the element at its center (junction point of the boom bond, --mount contact).
        Returns (cards, element centre segment, bar port segment, bar x, half length of the port segment)."""
        tap_m = taps[name]
        tap_off_m = tap_offs[name]
        seg_ref = 2.0 * half / n
        n_out = max(1, int(round((half - tap_m) / seg_ref)))
        n_c = max(3, 2 * int(round((2.0 * tap_m / seg_ref - 1.0) / 2.0)) + 1)
        if split_center:
            nh = (n_c + 1) // 2
            el = [(-half, -tap_m, n_out), (-tap_m, 0.0, nh), (0.0, tap_m, nh), (tap_m, half, n_out)]
            center_seg = n_out + nh
        else:
            el = [(-half, -tap_m, n_out), (-tap_m, tap_m, n_c), (tap_m, half, n_out)]
            center_seg = n_out + (n_c + 1) // 2
        cards = [gw(tag, k, xyz(x, a), xyz(x, b), r_el) for a, b, k in el]
        wire_warnings.extend(wire_chain_warnings(name, [(b - a, k, r_el) for a, b, k in el], lam))
        xb = x + toward * tap_off_m
        n_s = max(2, int(round(tap_off_m / seg_ref)))
        n_b = max(3, 2 * int(round((2.0 * tap_m / seg_ref - 1.0) / 2.0)) + 1)
        gh = tap_m / n_b
        if split:
            n1 = (n_b - 1) // 2
            bar = [(-tap_m, -gh, n1), (-gh, gh, 1), (gh, tap_m, n1)]
        else:
            bar = [(-tap_m, tap_m, n_b)]
        cards += [gw(bar_tag, n_s, xyz(x, -tap_m), xyz(xb, -tap_m), tap_rad)]
        cards += [gw(bar_tag, k, xyz(xb, a), xyz(xb, b), tap_rad) for a, b, k in bar]
        cards += [gw(bar_tag, n_s, xyz(xb, tap_m), xyz(x, tap_m), tap_rad)]
        wire_warnings.extend(wire_chain_warnings(f"{name} T-match", [(tap_off_m, n_s, tap_rad)]
                                                 + [(b - a, k, tap_rad) for a, b, k in bar]
                                                 + [(tap_off_m, n_s, tap_rad)], lam))
        return cards, center_seg, n_s + (n_b + 1) // 2, xb, gh

    def shield_wires(x_port, gh, seg_len):
        """Common-mode shield: an open-ended wire tied to the -a end of the fed port segment at (x_port, -gh, 0).
        With a modeled boom it slopes down and outward to run rearward beside the boom, --boom-clearance-mm from
        its surface, at the height of the boom axis; without one it drops straight down to where the boom axis
        would be. The slope is pushed outward until it clears the boom by half the clearance."""
        fr = args.feedline_diameter / 2000.0
        if boom_isolated:
            a_side = -(boom_rad + fr + clr)
            while a_side > -1.0:
                dist = _segment_distance((-gh, 0.0, 0.0), (a_side, zb, 0.0), (0.0, zb, 0.0), (0.0, zb, 0.0))
                if dist >= boom_rad + fr + clr:
                    break
                a_side -= 0.002
        else:
            a_side = -gh
        drop = math.hypot(a_side + gh, zb)
        nv = max(1, int(math.ceil(drop / seg_len - 1e-9)))
        nl = max(3, int(math.ceil(feedline_len / seg_len - 1e-9)))
        return [gw(10, nv, xyz(x_port, -gh, 0.0), xyz(x_port, a_side, zb), fr),
                gw(11, nl, xyz(x_port, a_side, zb), xyz(x_port - feedline_len, a_side, zb), fr)]

    rear_phase = args.rear_phase if args.rear_phase is not None else theory_rear_phase(args.spacing_factor)
    rear_ratio = args.rear_ratio
    line_target, line_elec, line_phys = resolve_line(args, lam)
    line_info = {"rear_phase": rear_phase, "line_target_lag_deg": line_target,
                 "line_electrical_deg": line_elec, "line_length_m": line_phys}
    if args.fsweep:
        fmin, fmax, fstep = args.fsweep
        nfreq, flast = sweep_count(fmin, fmax, fstep)

    if args.network == "tl":
        ltype = getattr(args, "line_type", "openwire")
        net_note = (f"NEC TL phase line ({'open-wire balanced' if ltype == 'openwire' else 'coaxial'}, "
                    f"{'CROSSED' if args.crossed else 'straight'}) between centers, "
                    f"fed at the {args.feed_element} element: Z0={args.coax_z0:g} ohm, "
                    f"physical length={line_phys:.5f} m ({line_elec:.1f} electrical degrees) at VF={args.coax_vf:g}, "
                    f"NEC TL length={line_phys / args.coax_vf:.5f} m. "
                    "Ideal balanced model without losses, radiation or common-mode currents.")
    elif args.network == "lc":
        net_note = (f"Ideal lumped LC pi network (NT card, Y parameters) {'CROSSED' if args.crossed else 'straight'} "
                    f"between centers, fed at the {args.feed_element} element. "
                    "No component losses or parasitics.")
    elif getattr(args, "ideal_drive", "current") == "current":
        net_note = (f"IDEAL SOURCES: two EX excitations, each in series with a lossless {IDEAL_DRIVE_X:g} ohm "
                    "reactance (LD 4), so the element CURRENTS have the prescribed amplitude/phase.")
    else:
        net_note = "IDEAL SOURCES: two EX excitations with prescribed VOLTAGE amplitude/phase."

    # NEC-2 only allows CM cards before CE: all comments go in the header.
    extra_cm = []
    if ground != "free":
        gdesc = "perfect ground (GN 1)" if ground == "perfect" else f"{ground} ground ({ground_card(args)})"
        extra_cm.append(f"CM Ground: {gdesc}; elements {'vertical' if vertical else 'horizontal'}; height "
                        f"{height:.4f} m ({'lowest tip of the longest element' if vertical else 'element centers'}).")
    elif vertical:
        extra_cm.append("CM Elements along Z (vertical); free space: same pattern as horizontal.")
    if use_feedline:
        extra_cm.append(f"CM Common-mode test: open-ended wire L={feedline_len:g} m (coax outer conductor) tied to the "
                        f"lower half of the {args.feed_element} element, whose 1-segment center wire carries the "
                        "source; it runs rearward below the elements. Models an unbalanced feed WITHOUT choke.")
    sigma = getattr(args, "conductivity", 0.0) or 0.0
    if sigma > 0:
        extra_cm.append(f"CM Wire conductivity {sigma:.4g} S/m (LD 5, all wires); TL/NT networks are lossless.")
    if len(rear_secs) > 1 or len(front_secs) > 1 or abs(rear_secs[0][1] - front_secs[0][1]) > 1e-9:
        extra_cm.append(f"CM Element diameters (center outward): rear {sections_text(rear_secs)}; "
                        f"front {sections_text(front_secs)}.")
    if par:
        extra_cm.append(f"CM Parasitic elements (passive, no source): {sum(e['role'] == 'reflector' for e in par)} "
                        f"reflector(s) tags 21-22 behind the rear element, {sum(e['role'] == 'director' for e in par)} "
                        "director(s) tags 31-35 in front of the front element.")
        for e in par:
            extra_cm.append(f"CM {e['name']}: L={e['length_m']:.5f} m ({e['len_factor']:.4f} x resonant "
                            f"{e['l_res_m']:.5f} m), x={e['x']:.5f} m, {e['spacing_m'] * 1000:.1f} mm from the "
                            f"previous element, dia {e['diam_mm']:g} mm.")
    if model_gap:
        extra_cm.append(f"CM Feed gap g={gap_m * 1000:g} mm modeled: a 1-segment wire of length g bridges the gap at "
                        "each element center and carries EX/TL (the gap capacitance is not modeled).")
    if tmatch:
        extra_cm.append(f"CM T-match feed on both driven elements: taps {tap_m * 1000:g} mm from each element center, "
                        f"bars {tap_off_m * 1000:g} mm from the element toward the other one, wire "
                        f"{tap_rad * 2000:g} mm (tags {TAG_TMATCH_REAR}/{TAG_TMATCH_FRONT}); EX and the phase network "
                        "connect at the bar centers.")
    c_pf = series_c_value(args)
    if c_pf:
        extra_cm.append(f"CM {SERIES_C_MARK}{c_pf:.6g} {SERIES_C_Q_MARK}{series_c_q(args):.6g} (series capacitor "
                        "between the feed line and the fed port, Q 0 = lossless; hb9cv.py adds it to Zin)")
    if contact:
        boom_note = (f" (metallic, {clr * 1000:g} mm below the elements; element centers bonded to it, tag 7)")
    elif boom_isolated:
        boom_note = f" (isolated conductor {clr * 1000:g} mm below the elements)"
    else:
        boom_note = " (boom not modeled)"
    lines = [
        "CM Exploratory HB9CV generated by hb9cv.py",
        f"CM f={f:.6f} MHz lambda={lam:.8f} m",
        f"CM Lengths: rear={rear_len:.8f} m front={front_len:.8f} m; spacing={spacing:.8f} m",
        f"CM Elements {args.mount}; boom {args.boom_shape}, size={args.boom_size:.3f} mm length={args.boom_length:.1f} mm",
        f"CM Equivalent boom radius in NEC: {boom_rad*1000:.3f} mm" + boom_note,
        f"CM Selected network: {args.network}. {net_note}",
        f"CM Selected objective: {args.objective}.",
        "CM The selected objective defines the evaluation criterion; it does not optimize automatically.",
        "CM The 50/75 ohm target is a design reference; it does not force the computed impedance.",
        ("CM Feed: two independent EX sources (prescribed current or voltage relation, see above)."
         if args.network == "ideal" else
         f"CM Feed: one EX source on the {args.feed_element} element and an ideal phase network "
         f"({'TL' if args.network == 'tl' else 'NT pi LC'}) to the other."),
        "CM Beam toward +x (phi=0): the front element is at x=+d/2.",
        *extra_cm,
        "CE",
    ]

    # Elements parallel to the Y axis (horizontal), separated along X. cseg: source segment of each tag.
    cseg = {}
    feed_front = args.feed_element == "front"
    if tmatch:
        rear_cards, cseg[1], pseg_r, xb_r, gh_r = build_tmatch_element(
            1, TAG_TMATCH_REAR, -spacing / 2, half_rear, rad_rear, "rear", +1.0, split=use_feedline and not feed_front,
            split_center=contact)
        front_cards, cseg[2], pseg_f, xb_f, gh_f = build_tmatch_element(
            2, TAG_TMATCH_FRONT, spacing / 2, half_front, rad_front, "front", -1.0, split=use_feedline and feed_front,
            split_center=contact)
        lines += rear_cards + front_cards
        if use_feedline:
            xb, gh = (xb_f, gh_f) if feed_front else (xb_r, gh_r)
            lines += shield_wires(xb, gh, 2.0 * (half_front if feed_front else half_rear) / n)
    elif use_feedline and model_gap:
        raise ValueError("--model-feed-gap cannot be combined with the common-mode feed-line wire "
                         "(--feedline-length-m / --cm-sensitivity): the wire already splits the fed element "
                         "at a one-segment gap")
    elif use_feedline and len(front_secs if feed_front else rear_secs) > 1:
        raise ValueError("stepped sections on the fed element cannot be combined with the common-mode "
                         "feed-line wire (--feedline-length-m / --cm-sensitivity)")
    elif use_feedline:
        # The fed element is lower half / 1-segment center wire (source) / upper half, all on its own tag so the
        # segments are numbered along the element as for the plain wire; the shield wire is tied to the lower
        # half. The other element is built normally.
        xf = spacing / 2 if feed_front else -spacing / 2
        hf = half_front if feed_front else half_rear
        rad_f = rad_front if feed_front else rad_rear
        seg_len = 2 * hf / n
        gh = seg_len / 2
        nh = (n - 1) // 2
        tag0 = 2 if feed_front else 1
        if feed_front:
            other_cards, cseg[1] = build_element(1, -spacing / 2, half_rear, rear_secs, "rear")
        else:
            other_cards, cseg[2] = build_element(2, spacing / 2, half_front, front_secs, "front")
        fed_cards = [gw(tag0, nh, xyz(xf, -hf), xyz(xf, -gh), rad_f),
                     gw(tag0, 1, xyz(xf, -gh), xyz(xf, gh), rad_f),
                     gw(tag0, nh, xyz(xf, gh), xyz(xf, hf), rad_f)]
        wire_warnings.extend(wire_chain_warnings("rear" if tag0 == 1 else "front",
                                                 [(hf - gh, nh, rad_f), (2 * gh, 1, rad_f), (hf - gh, nh, rad_f)],
                                                 lam))
        cseg[tag0] = nh + 1
        lines += (other_cards + fed_cards) if feed_front else (fed_cards + other_cards)
        lines += shield_wires(xf, gh, seg_len)
    else:
        rear_cards, cseg[1] = build_element(1, -spacing / 2, half_rear, rear_secs, "rear")
        front_cards, cseg[2] = build_element(2, spacing / 2, half_front, front_secs, "front")
        lines += rear_cards + front_cards
    for e in par:   # passive, continuous tube (no feed gap), same plane as the driven elements
        cards, _ = build_element(e["tag"], e["x"], e["half_m"], [(None, e["diam_mm"])], e["name"], use_gap=False,
                                 split_center=contact)
        lines += cards
    rear_tag, front_tag = 1, 2
    # ports: element centre segments (element currents); feed_ports: where EX and the phase network connect
    line_info["ports"] = {"rear": (rear_tag, cseg[rear_tag]), "front": (front_tag, cseg[front_tag])}
    line_info["feed_ports"] = ({"rear": (TAG_TMATCH_REAR, pseg_r), "front": (TAG_TMATCH_FRONT, pseg_f)} if tmatch
                               else dict(line_info["ports"]))
    line_info["feed_distance_m"] = feed_distance_m(args, spacing)
    if boom_isolated:
        boom_len = boom_len_m
        x_left, x_right = boom_xc - boom_len / 2.0, boom_xc + boom_len / 2.0
        if not (x_left < -spacing / 2.0 and spacing / 2.0 < x_right):
            raise ValueError("the boom must be longer than the spacing between elements")
        # Below the elements, --boom-clearance-mm from their surfaces. With --mount contact the boom is split at
        # every element and a short bond wire (tag 7, the clamp) joins each element center to the boom axis.
        seg_target = min(half_rear, half_front) * 2.0 / n
        if contact:
            bonds = [(-spacing / 2, rad_rear), (spacing / 2, rad_front)] + [(e["x"], e["diam_mm"] / 2000.0) for e in par]
            cuts = sorted([x_left, x_right] + [bx for bx, _ in bonds if x_left < bx < x_right])
        else:
            bonds, cuts = [], [x_left, x_right]
        chain = []
        for xa, xc in zip(cuts[:-1], cuts[1:]):
            nb = boom_segments(xc - xa, seg_target, boom_rad, lam)
            chain.append((xc - xa, nb, boom_rad))
            lines.append(gw(5, nb, xyz(xa, 0.0, zb), xyz(xc, 0.0, zb), boom_rad))
        wire_warnings.extend(wire_chain_warnings("boom", chain, lam))
        for bx, br in bonds:
            lines.append(gw(7, 1, xyz(bx, 0.0, 0.0), xyz(bx, 0.0, zb), br))
            wire_warnings.extend(wire_chain_warnings("boom bond", [(abs(zb), 1, br)], lam))

    # Mast (tag 12), never galvanically connected (NEC junctions need coincident wire ends): the --mast-gap-mm air
    # gap sets the coupling distance.
    # Horizontal elements: a vertical conductor perpendicular to the boom and to the elements, hanging below the
    # lowest modeled conductor at the boom center, moved sideways (along the elements) by --mast-offset-mm, the
    # arm of a side clamp. On the axis (offset 0) it lies in the symmetry plane of the balanced elements and is
    # not excited by them.
    # Vertical elements: a side-mounted mast parallel to the elements, behind the rear end of the boom (the clamp
    # end) at --mast-gap-mm + --mast-offset-mm from it, centered on the element centers.
    mast_len = (getattr(args, "mast_length_m", 0.0) or 0.0)
    if mast_len > 0:
        if use_feedline:
            raise ValueError("--mast-length-m cannot be combined with the common-mode feed-line wire "
                             "(--feedline-length-m / --cm-sensitivity): the wires would cross under the boom")
        mr = getattr(args, "mast_diameter_mm", 40.0) / 2000.0
        mgap = (getattr(args, "mast_gap_mm", 20.0) or 0.0) / 1000.0
        moff = (getattr(args, "mast_offset_mm", 0.0) or 0.0) / 1000.0
        seg_m = 2.0 * min(half_rear, half_front) / n
        nm = max(3, boom_segments(mast_len, seg_m, mr, lam))
        if vertical:
            o_boom = zb if boom_isolated else 0.0
            x_end = boom_xc - boom_len_m / 2.0           # rear end of the boom, where the clamp sits
            xm = x_end - mgap - moff - mr
            lines.append(gw(12, nm, xyz(xm, -mast_len / 2.0, o_boom), xyz(xm, mast_len / 2.0, o_boom), mr))
            where = (f"parallel to the elements, {(x_end - xm) * 1000:.0f} mm behind the rear end of the boom, centered "
                     "on the element centers")
        else:
            if boom_isolated:
                o_ref = zb - boom_rad                       # underside of the modeled boom
            else:
                o_ref = -rad                                # boom not modeled: underside of the elements
            o_top = o_ref - (mgap if mgap > 0 else 0.1 * mr)   # the mast end is flat: no radius offset needed
            lines.append(gw(12, nm, xyz(boom_xc, moff, o_top), xyz(boom_xc, moff, o_top - mast_len), mr))
            where = (f"at the boom center{f', {moff * 1000:g} mm to the side' if moff else ''}, top "
                     f"{mgap * 1000:g} mm below the lowest modeled conductor")
        wire_warnings.extend(wire_chain_warnings("mast", [(mast_len, nm, mr)], lam))
        lines.insert(lines.index("CE"), f"CM Mast: {mast_len:g} m x {2000 * mr:g} mm diameter {where}; no galvanic "
                                        "junction.")

    # Thin-wire kernel by default. The extended thin-wire kernel (EK, --extended-kernel) only helps on thick straight
    # wires and NEC-2 turns it off at junctions; with T-match junctions and short segments it can make the current
    # solution unstable (negative radiated power), while the results where it is stable differ by about 1 ohm.
    kernel = ["EK"] if getattr(args, "extended_kernel", False) else []
    names = {1: "rear element", 2: "front element", TAG_TMATCH_REAR: "rear T-match", TAG_TMATCH_FRONT: "front T-match",
             5: "boom", 7: "boom bond", 10: "shield drop", 11: "shield wire", 12: "mast"}
    names.update({e["tag"]: e["name"] for e in par})
    errs, close = wire_clearance_problems(all_wires, names)
    if errs:
        raise ValueError("geometry: " + "; ".join(errs[:3]))
    wire_warnings.extend(close)
    if ground == "free":
        lines += ["GE 0"] + kernel
    else:
        if min(z_low) <= 0:
            raise ValueError(f"the lowest wire edge is at z={min(z_low):.4f} m: raise --height-m / --height-lambda "
                             "(every wire must be above the ground plane)")
        lines += ["GE 1", ground_card(args)] + kernel
    if sigma > 0:
        lines.append(f"LD 5 0 0 0 {sigma:.6g}")      # wire conductivity (type 5), tag 0 = every wire
    # Feed: one EX source plus an ideal TL (tl) or NT pi network (lc) between the two feed points, or two
    # EX sources (ideal). The NEC TL length is physical_length/VF.
    fp = line_info["feed_ports"]
    feed_name, far_name = ("front", "rear") if args.feed_element == "front" else ("rear", "front")
    (feed_tag, feed_seg), (far_tag, far_seg) = fp[feed_name], fp[far_name]
    line_warnings = []
    nt_card = None              # function (frequency in MHz -> NT card) for the LC network
    probe = getattr(args, "probe_port", None)
    if probe and args.network in ("tl", "lc"):
        # Two-port probe: a single 1 V source on one feed port and no phase network (the other port is shorted by
        # the continuous wire); used to measure the port and element admittances for the line synthesis.
        ptag, pseg = fp[probe]
        lines.append(f"EX 0 {ptag} {pseg} 0 1.00000000 0.00000000")
    elif args.network == "tl":
        tl_length = line_phys / args.coax_vf
        if tl_length <= TL_MIN_LENGTH_M:
            raise ValueError(f"the phase line has zero length ({line_elec:.3f} electrical degrees): NEC would read "
                             "a TL of length 0 as the straight-line distance between the two feed points, a different "
                             "line. Change --phase-target-deg / --line-deg / --line-length (a crossed line already "
                             "contributes 180 degrees) or use --no-crossed.")
        line_warnings += line_feasibility_warnings(args, lam, line_info["feed_distance_m"], line_elec)
        if (args.feed_element == "rear" and args.crossed and getattr(args, "phase_target_deg", None) is None
                and getattr(args, "line_deg", None) is None and getattr(args, "coax_length", None) is None):
            line_warnings.append(f"rear-fed crossed line needs {line_elec:.0f} electrical degrees "
                                 f"({line_elec / 360.0:.3f} lambda); --no-crossed gives "
                                 f"{line_electrical_deg(line_target, False):.0f} degrees")
        # NEC: negative Z0 = crossed line (180 degree inversion).
        z0_card = -abs(args.coax_z0) if args.crossed else abs(args.coax_z0)
        lines.append(f"EX 0 {feed_tag} {feed_seg} 0 1.00000000 0.00000000")
        lines.append(f"TL {feed_tag} {feed_seg} {far_tag} {far_seg} {z0_card:.6f} {tl_length:.8f} 0 0 0 0")
    elif args.network == "lc":
        if args.lc_l_nh is not None:
            ind, cap = args.lc_l_nh * 1e-9, args.lc_c_pf * 1e-12
        else:
            ind, cap = lc_pi_values(line_elec, args.lc_z0, f)
        line_info["lc_l_nh"], line_info["lc_c_pf"] = ind * 1e9, cap * 1e12

        def lc_nt_card(fr_mhz):
            """NT card of the pi network at fr_mhz. NEC holds NT admittances fixed, so every frequency group
            gets its own card; a network card that follows an execution card replaces the previous set."""
            w = 2 * math.pi * fr_mhz * 1e6
            ys = 1.0 / (1j * w * ind)       # series branch
            yc = 1j * w * cap               # shunt capacitors at each port
            y11 = y22 = yc + ys
            y12 = ys if args.crossed else -ys  # crossing = inverting the polarity of the far port
            return (f"NT {feed_tag} {feed_seg} {far_tag} {far_seg} "
                    f"{y11.real:.8E} {y11.imag:.8E} {y12.real:.8E} {y12.imag:.8E} {y22.real:.8E} {y22.imag:.8E}")

        nt_card = lc_nt_card
        lines.append(f"EX 0 {feed_tag} {feed_seg} 0 1.00000000 0.00000000")
    elif probe:
        raise ValueError("the two-port probe needs --network tl or lc")
    else:
        # Two sources. --ideal-drive current (default): each source segment carries a lossless series reactance
        # IDEAL_DRIVE_X far above the element impedances (LD 4), so the element currents follow the source voltages
        # to within |Zin|/IDEAL_DRIVE_X and the prescribed amplitude/phase is the CURRENT relation of the cardioid
        # theory, whatever the mutual coupling. --ideal-drive voltage applies it to the source voltages instead.
        drive = complex(rear_ratio * math.cos(math.radians(rear_phase)), rear_ratio * math.sin(math.radians(rear_phase)))
        scale = 1.0
        if getattr(args, "ideal_drive", "current") == "current":
            scale = IDEAL_DRIVE_X
            for tg in (rear_tag, front_tag):
                lines.append(f"LD 4 {tg} {cseg[tg]} {cseg[tg]} 0 {IDEAL_DRIVE_X:.6g}")
        vr, vf = drive * scale, complex(scale, 0.0)
        lines.append(f"EX 0 {rear_tag} {cseg[rear_tag]} 0 {vr.real:.8E} {vr.imag:.8E}")
        lines.append(f"EX 0 {front_tag} {cseg[front_tag]} 0 {vf.real:.8E} {vf.imag:.8E}")

    if args.fsweep and not getattr(args, "fsweep_pattern", False):
        # Frequency sweep: only Zin(f) matters; the RP pattern is omitted (--fsweep-pattern adds it).
        if nt_card is not None:
            # LC network: one FR group per frequency, each with the NT admittances of that frequency.
            for fr in sweep_frequencies(fmin, fmax, fstep):
                lines += [nt_card(fr), f"FR 0 1 0 0 {fr:.8f} 0", "XQ"]
        else:
            lines += [f"FR 0 {nfreq} 0 0 {fmin:.8f} {fstep:.8f}", "XQ"]
            if sweep_needs_fmax(fmax, flast):
                # The range is not a multiple of the step: fmax is added as a separate final point.
                lines += [f"FR 0 1 0 0 {fmax:.8f} 0", "XQ"]
        lines += ["EN", ""]
    else:
        # One FR/RP/XQ group per frequency (EX and TL persist across groups; the LC network gets a new NT card
        # per group). --band evaluates the same geometry at several frequencies in one deck.
        freqs = list(getattr(args, "band_freqs", None) or [f])
        thetas = getattr(args, "ring_theta", 90.0)
        thetas = list(thetas) if isinstance(thetas, (list, tuple)) else [thetas] * len(freqs)
        if len(thetas) != len(freqs):
            raise ValueError("ring_theta needs one value per frequency")
        pattern_mode = getattr(args, "pattern_mode", "ring")
        nphi = int(round(360.0 / args.az_step)) + 1
        for i, fr in enumerate(freqs):
            if nt_card is not None:
                lines.append(nt_card(fr))
            lines.append(f"FR 0 1 0 0 {fr:.8f} 0")
            if i == 0:
                lines.append(f"PT {getattr(args, 'pt_mode', 0)}")   # -1: do not print all currents
            if pattern_mode == "elevation":
                # Elevation cut at phi=0 (the boom plane), theta 0..90 in 1 degree steps; XNDA=1000:
                # vertical/horizontal/total power gains. Used to find the take-off angle over ground.
                lines.append("RP 0 91 1 1000 0 0 1 0")
            else:
                # Azimuth ring at theta=thetas[i] (90 = horizon; over ground use the take-off angle);
                # XNDA=1000: vertical/horizontal/total power gains, no averaging (meaningless on one ring).
                lines.append(f"RP 0 1 {nphi} 1000 {thetas[i]:g} 0 0 {args.az_step:.8g}")
                if getattr(args, "rear_cone_rp", True):
                    # Rear half of the sphere (phi 90..270) on a 5 degree grid, for the rear-cone F/B: theta
                    # 0..180 in free space, 0..90 over ground.
                    lines.append(f"RP 0 {37 if ground == 'free' else 19} 37 1000 0 90 5 5")
            lines.append("XQ")
        lines += ["EN", ""]
    # NEC-2 uses fixed-width cards; split long comments into several CM cards.
    fixed_lines = []
    for line in lines:
        if line.startswith("CM ") and len(line) > 78:
            chunks = textwrap.wrap(line[3:], width=75, break_long_words=False,
                                   break_on_hyphens=False) or [""]
            fixed_lines.extend("CM " + chunk for chunk in chunks)
        else:
            fixed_lines.append(line)
    wire_warnings.extend(parasitic_warnings(par))
    line_info["wire_warnings"] = wire_warnings
    line_info["line_warnings"] = line_warnings
    line_info["parasitics"] = par
    line_info["boom_length_mm"] = boom_len_m * 1000.0
    line_info["boom_xc_m"] = boom_xc
    return "\n".join(fixed_lines), lam, rear_len, front_len, spacing, line_info


OPENWIRE_DEFAULT_VF = 0.95
COAX_DEFAULT_VF = 0.66
OPENWIRE_DEFAULT_WIRE_MM = 3.0
OPENWIRE_DEFAULT_SPACING_MM = 20.0
OPENWIRE_DEFAULT_ALLOWANCE_MM = 20.0   # connections at both ends plus the crossing of an open-wire line
COAX_DEFAULT_ALLOWANCE_MM = 40.0       # connector/pigtail lengths at both ends of a coax section
NEC2C_ENV = "HB9CV_NEC2C"


def find_nec2c(args=None):
    """Path of the nec2c executable: --nec2c (a file or a folder holding nec2c/nec2c.exe), then the HB9CV_NEC2C
    environment variable, then the PATH; None when none of them gives an executable."""
    for cand in (getattr(args, "nec2c", None), os.environ.get(NEC2C_ENV)):
        cand = (cand or "").strip()
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.is_dir():
            for n in ("nec2c", "nec2c.exe"):
                if (p / n).is_file():
                    return str((p / n).resolve())
        elif p.is_file():
            return str(p.resolve())
        found = shutil.which(cand)
        if found:
            return found
    return shutil.which("nec2c") or shutil.which("nec2c.exe")


OPENWIRE_MIN_RATIO = 1.2      # D/d below this leaves almost no air between the wires
OPENWIRE_PROXIMITY_RATIO = 3.0  # D/d below this: proximity effect and spacer dielectric shift Z0 and VF


def openwire_z0(wire_mm, spacing_mm):
    """Z0 (ohm) of a balanced air line of two round wires (diameter d, center spacing D): 120*acosh(D/d), exact
    for round wires in air (it reduces to 276*log10(2D/d) for D >> d)."""
    return 120.0 * math.acosh(spacing_mm / wire_mm)


def openwire_spacing_mm(z0, wire_mm):
    """Inverse of openwire_z0: center spacing D (mm) that gives z0 with wires of diameter d."""
    return wire_mm * math.cosh(z0 / 120.0)


def openwire_ratio_warning(wire_mm, spacing_mm):
    """Warning text for a closely spaced open-wire line (D/d below OPENWIRE_PROXIMITY_RATIO), or None."""
    ratio = spacing_mm / wire_mm
    if ratio < OPENWIRE_PROXIMITY_RATIO:
        return (f"open-wire line D/d = {ratio:.2f} (d {wire_mm:g} mm, D {spacing_mm:.2f} mm): at such close spacing "
                "the spacers and the proximity effect change Z0 and VF noticeably; measure the built line")
    return None


def resolve_line_options(p, args):
    """Resolve everything that depends on --line-type (generate and optimize).

    The internal dests coax_z0 / coax_vf / coax_length are kept for compatibility; they
    hold the Z0, VF and length of the phase line whatever its type (--line-z0, --line-vf
    and --line-length are the preferred option names).
    """
    tl = args.network == "tl"
    if not tl:
        given = [flag for flag, attr in (
            ("--line-type", "line_type"), ("--line-vf", "coax_vf"), ("--line-z0", "coax_z0"),
            ("--line-wire-diameter", "line_wire_diameter"), ("--line-spacing", "line_spacing"),
            ("--z0-values", "z0_values")) if getattr(args, attr, None) is not None]
        if given:
            print(f"WARNING: {', '.join(given)} only apply to --network tl and are ignored "
                  f"with --network {args.network}.", file=sys.stderr)
    args.line_type = args.line_type or "openwire"
    args.line_spacing_given = args.line_spacing is not None
    if tl and args.line_type == "coax":
        if args.crossed:    # True only if --crossed was given explicitly (the default is None)
            p.error("--network tl --line-type coax cannot be --crossed: a coaxial cable has one inner conductor "
                    "and one shield and cannot be crossed. Use --line-type openwire (balanced line, crossed by "
                    "construction) or drop --crossed (straight coax, 225 deg for the classic front-fed HB9CV).")
        args.crossed = False
        if args.line_wire_diameter is not None or args.line_spacing is not None:
            p.error("--line-wire-diameter/--line-spacing only apply to --line-type openwire")
    elif args.crossed is None:
        args.crossed = True
    if args.coax_vf is None:
        args.coax_vf = OPENWIRE_DEFAULT_VF if args.line_type == "openwire" else COAX_DEFAULT_VF
    if getattr(args, "line_allowance_mm", 0.0) is None:
        args.line_allowance_mm = (OPENWIRE_DEFAULT_ALLOWANCE_MM if args.line_type == "openwire"
                                  else COAX_DEFAULT_ALLOWANCE_MM) if tl else 0.0
    if tl and args.line_type == "openwire" and args.coax_vf < 0.85:
        print(f"WARNING: VF={args.coax_vf:g} is low for an air-spaced open-wire line (typically 0.90-0.98).",
              file=sys.stderr)
    if tl and args.line_type == "coax" and args.coax_vf > 0.9:
        print(f"WARNING: VF={args.coax_vf:g} is high for a coaxial cable (solid PE 0.66, foam 0.80-0.85).",
              file=sys.stderr)
    if tl:
        if args.line_type == "openwire":
            if args.coax_z0 is not None:
                p.error("with --line-type openwire Z0 follows from the geometry: do not give --line-z0; set "
                        "--line-wire-diameter and --line-spacing (in optimize also --z0-values)")
            if args.line_wire_diameter is None:
                args.line_wire_diameter = OPENWIRE_DEFAULT_WIRE_MM
            if args.line_spacing is None:
                args.line_spacing = OPENWIRE_DEFAULT_SPACING_MM
            if args.line_spacing <= args.line_wire_diameter:
                p.error("--line-spacing (center to center) must be greater than --line-wire-diameter")
            if args.line_spacing / args.line_wire_diameter < OPENWIRE_MIN_RATIO:
                p.error(f"--line-spacing must be at least {OPENWIRE_MIN_RATIO:g} x --line-wire-diameter")
            w_ratio = openwire_ratio_warning(args.line_wire_diameter, args.line_spacing)
            if w_ratio:
                print(f"WARNING: {w_ratio}.", file=sys.stderr)
            args.coax_z0 = openwire_z0(args.line_wire_diameter, args.line_spacing)
        elif args.coax_z0 is None:
            args.coax_z0 = 50.0


BALUN_REQUIREMENT = (
    "REQUIRED at the feed point: a 1:1 current balun / common-mode choke. The NEC source is an ideal balanced "
    "source; the real antenna is balanced and fed from unbalanced coax, and without a choke the shield carries "
    "common-mode current that radiates, skews the pattern (F/B) and shifts Zin. Target common-mode impedance "
    ">= 500-1000 ohm at {f:g} MHz; validate on the bench. See --feedline-length-m and --cm-sensitivity.")

BALUN_4_1_NOTE = (
    "Balun 4:1: balanced-side reference = {zb:g} ohm (= {zu:g}/4); the balun steps the antenna impedance up by 4 "
    "toward the coax (low-impedance side at the antenna). This is an option to evaluate, not a proven fix: "
    "the reactance must be tuned out first (e.g. 15+15j ohm against 12.5 ohm is still ~2.9:1).")


def balun_n(args):
    return 4.0 if getattr(args, "balun_ratio", "1:1") == "4:1" else 1.0


def apply_balun_ratio(args):
    """match/sweep: the L-network is designed on the balanced side against z0/n."""
    n = balun_n(args)
    if n != 1.0:
        args.z0_unbal = args.z0
        args.z0 = args.z0 / n
        print(BALUN_4_1_NOTE.format(zb=args.z0, zu=args.z0_unbal))
        print(f"The L-network is designed on the balanced side against {args.z0:g} ohm; the balun then "
              f"transforms it to {args.z0_unbal:g} ohm.")


def describe_line(args, elec_deg, phys_m, z0, spacing_mm, lc_l_nh, lc_c_pf):
    """Lines of text describing the phase network for the build sheet."""
    if args.network == "tl":
        kind = "open-wire (balanced)" if args.line_type == "openwire" else "coaxial"
        out = [f"Type: {kind}, {'crossed' if args.crossed else 'straight'}; fed at the {args.feed_element} element",
               f"Z0 = {z0:.1f} ohm; VF of the line = {args.coax_vf:g}",
               f"Electrical length = {elec_deg:.1f} deg; physical length = {phys_m:.4f} m "
               f"(NEC TL card length = {phys_m / args.coax_vf:.4f} m)"]
        if args.line_type == "openwire" and spacing_mm is not None:
            out.append(f"Wire diameter d = {args.line_wire_diameter:g} mm; center spacing D = {spacing_mm:.2f} mm "
                       "(Z0 = 120*acosh(D/d))")
        if args.line_type == "coax":
            out.append("A coax phasing section needs a balun/choke at its element end (the NEC TL does not model it).")
        return out
    if args.network == "lc":
        return [f"Ideal LC pi network, {'crossed' if args.crossed else 'straight'}: series L = {lc_l_nh:.3f} nH, "
                f"C per port = {lc_c_pf:.3f} pF ({elec_deg:.1f} deg). No losses or parasitics."]
    return ["Two ideal sources (no physical feed network)."]


HALF_NOTE = ("'+a' half = the half at the positive element coordinate (NEC +Y for horizontal elements: the LEFT "
             "half looking along the beam; NEC +Z for vertical elements: the UPPER half). '-a' is the other one.")

TUNING_STEPS = [
    "Mount the antenna at its working height and orientation, with the final feed line and choke in place. "
    "Measure with a VNA / antenna analyzer at the feed point (or through a known electrical length of line).",
    "Compare the measured resonance (X = 0) and Zin with the NEC prediction; write every difference down.",
    "(a) Front element: adjust its length (both halves equally) for minimum |X| at the design frequency.",
    "(b) Phasing line: adjust its length / tap for resonance and for the F/B optimum; keep the crossing polarity.",
    "(c) Matching / balun: only after (a) and (b), set the matching network or balun ratio; re-measure Zin and VSWR "
    "over the whole band.",
    "Re-measure the F/B with a field-strength meter or a second station at a distance of several wavelengths, in "
    "the same polarization, with the antenna turned through 180 degrees; check that the coax shield does not change "
    "the reading when it is moved (common-mode current).",
    "Document the final element lengths, line length and matching values versus the NEC prediction.",
]


def _cplx(z):
    return None if z is None else {"r": z.real, "x": z.imag}


def perf_row_from_metrics(m, zin, target, freq):
    """Convert a parse_metrics() dict into the per-frequency row format used by the optimizer."""
    vswr = lnet_metrics(zin, target)[0] if (zin is not None and target) else None
    return {"freq_mhz": freq, "forward_gain": m["forward_gain"], "gain_dbd": m["gain_dbd"], "fb": m["fb"],
            "robust_fb": m["robust_fb"], "cone_fb": m.get("cone_fb"), "takeoff_deg": 90.0 - m["theta_deg"],
            "hpbw": m["hpbw"],
            "zin": zin, "vswr": vswr}


def build_sheet_data(args, geom, line, zin=None, perf=None):
    """Collect everything a builder needs (dict, JSON-serializable after _jsonable)."""
    f = args.freq
    lam_mm = C_LIGHT / (f * 1e6) * 1000.0
    diam = geom.get("diam_mm", args.element_diameter)      # common diameter (optimizer: the best candidate's)
    tmatch = getattr(args, "feed_style", "center") == "tmatch" and args.network in ("tl", "lc")
    gap, cap = (0.0 if tmatch else args.gap_mm), args.end_cap_mm
    feed = getattr(args, "feed_element", "front")
    data = {"design_freq_mhz": f, "network": args.network, "mount": args.mount,
            "reference_impedance_ohm": args.target_impedance, "warnings": list(geom.get("extra_warnings") or [])}

    # 1. Elements ---------------------------------------------------------------------------
    elements = {}
    for name, key in (("rear", "rear_mm"), ("front", "front_mm")):
        length = geom[key]
        half = (length - gap) / 2.0 - cap
        if half <= 0:
            data["warnings"].append(f"{name} element: --feed-gap-mm/--end-cap-mm leave no tube length "
                                    f"(half_cut = {half:.1f} mm).")
        secs = element_sections(args, name, diam)
        tubes = []
        try:
            prof = element_profile(secs, length / 2000.0, name)
        except ValueError as exc:
            data["warnings"].append(str(exc))
            prof = element_profile([(None, secs[0][1])], length / 2000.0, name)
        for i, (a0, a1, r) in enumerate(prof):
            t0 = max(a0, gap / 2000.0)
            t1 = a1 - (cap / 1000.0 if i == len(prof) - 1 else 0.0)
            if t1 - t0 <= 0:
                data["warnings"].append(f"{name} element, tube {i + 1}: no length left after the half gap/end hardware.")
            tubes.append({"tube": i + 1, "nec_from_center_mm": a0 * 1000.0, "nec_to_mm": a1 * 1000.0,
                          "od_mm": 2000.0 * r, "cut_length_mm": (t1 - t0) * 1000.0})
        elements[name] = {"nec_length_mm": length, "feed_gap_mm": gap, "end_cap_extension_mm": cap,
                          "half_cut_mm": half, "halves": 1 if tmatch else 2, "diameter_mm": tubes[0]["od_mm"],
                          "diameter_text": "/".join(f"{t['od_mm']:g}" for t in tubes), "sections": tubes,
                          "wall_mm": args.tube_wall_mm, "fed": args.network in ("tl", "lc") and name == feed}
    data["elements"] = elements
    stepped = any(len(e["sections"]) > 1 for e in elements.values())
    notes = ["NEC length = length between wire end points (flat ends, no end caps), element axis straight."]
    if not tmatch:
        notes.append(f"half_cut = (L_nec - g)/2 - e with g = {gap:g} mm feed gap and e = {cap:g} mm of end hardware "
                     "that extends the conductor beyond the tube end (0 = flat, no caps). L_nec = 2 x half_cut + g + "
                     "2 x e: the gap is inside the NEC length, so if you change g keep the tubes' total length + g "
                     "equal to L_nec (each extra mm of gap with the same tubes lengthens the element by 1 mm).")
    if tmatch:
        notes.append("T-match feed: each driven element is ONE continuous tube (no feed gap; it may be bolted to a "
                     "metallic boom at its center); cut length = NEC length - 2 x e. The 'cut length of each half' "
                     "column is the half of that tube, from the center to the tip.")
    else:
        notes.append(("The feed gap is in the NEC model: a 1-segment wire of length g bridges the gap and carries the "
                      "source/line port; gap capacitance and the insulator are not modeled."
                      if getattr(args, "model_feed_gap", False) and gap > 0 else
                      "The feed gap is not a separate wire in the NEC model (the element is continuous, the source "
                      "sits in its central segment); --model-feed-gap models it."))
    if stepped or elements["rear"]["diameter_text"] != elements["front"]["diameter_text"]:
        for name in ("rear", "front"):
            e = elements[name]
            notes.append(f"{name} element, each half, from the center outward: " + "; ".join(
                f"tube {t['tube']}: {t['od_mm']:g} mm OD, cut {t['cut_length_mm']:.1f} mm" for t in e["sections"])
                + ("" if len(e["sections"]) == 1 else " (exposed lengths as modeled: add the telescopic overlap "
                   "inside the larger tube at every joint)") + ".")
    else:
        notes.append(f"Both elements are modeled with a uniform diameter ({elements['rear']['diameter_text']} mm). "
                     "If the real tubes are telescoped, give the stepped sections (--rear-sections/--front-sections) "
                     "and simulate them before cutting; these lengths are not valid for a stepped element.")
    notes.append("Lengths do not include any mounting/clamp correction: finish on the bench (tuning procedure).")
    notes.append(f"Conductor losses modeled with sigma = {args.conductivity:.3g} S/m (LD 5); check this value for the "
                 "real tube alloy (a different alloy or an anodized/painted surface changes the loss slightly)."
                 if getattr(args, "conductivity", 0) else "Conductors modeled as perfect (--conductivity 0): no losses.")
    data["elements_notes"] = notes

    # 1b. Parasitic elements (reflectors / directors) --------------------------------------------
    pars = geom.get("parasitics") or []
    sp_m = geom["spacing_mm"] / 1000.0
    x_rear_m, x_front_m = -sp_m / 2.0, sp_m / 2.0
    boom_len_mm = geom.get("boom_length_mm", args.boom_length)
    boom_xc_m = geom.get("boom_xc_m", 0.0)
    boom_x0_m = boom_xc_m - boom_len_mm / 2000.0           # rear end of the boom (datum for the positions)
    data["parasitics"] = []
    for e in pars:
        nec_len = e["length_m"] * 1000.0
        cut = nec_len - 2.0 * cap
        if cut <= 0:
            data["warnings"].append(f"{e['name']}: --end-cap-mm leaves no tube length.")
        data["parasitics"].append({
            "name": e["name"], "role": e["role"], "tag": e["tag"], "nec_length_mm": nec_len, "cut_length_mm": cut,
            "diameter_mm": e["diam_mm"], "wall_mm": args.tube_wall_mm, "length_factor": e["len_factor"],
            "spacing_from_previous_mm": e["spacing_m"] * 1000.0, "spacing_lambda": e["spacing_lambda"],
            "position_from_boom_rear_end_mm": (e["x"] - boom_x0_m) * 1000.0,
            "length_min_mm": e["len_min_m"] * 1000.0, "length_nominal_mm": e["len_nom_m"] * 1000.0,
            "length_max_mm": e["len_max_m"] * 1000.0, "spacing_min_mm": e["sp_min_m"] * 1000.0,
            "spacing_nominal_mm": e["sp_nom_m"] * 1000.0})
    if pars:
        notes.append("Reflectors/directors are passive: one continuous tube each (no feed gap), mounted insulated on "
                     "the boom in the same plane as the driven elements. Their lengths and spacings are checked against "
                     "the pre-calculated minimum/maximum values (see the table; optimize keeps them inside); cut length = NEC length - 2 x e.")
        for w in parasitic_warnings(pars):
            data["warnings"].append(w)

    # 2. Spacing, boom -------------------------------------------------------------------
    rad_mm = max(t["od_mm"] for e in elements.values() for t in e["sections"]) / 2.0
    boom_rad_mm = boom_radius(args.boom_shape, args.boom_size) * 1000.0
    clr_mm = DEFAULT_BOOM_CLEARANCE_MM if getattr(args, "boom_clearance_mm", None) is None else args.boom_clearance_mm
    if args.mount == "contact":
        offset = rad_mm + boom_rad_mm + clr_mm
        offset_note = (f"metallic boom below the elements; each element center is bonded to it by its clamp "
                       f"({clr_mm:g} mm between the element and boom surfaces, as modeled)")
    elif getattr(args, "boom_isolated", False):
        offset = rad_mm + boom_rad_mm + clr_mm
        offset_note = (f"metallic boom below the elements, no electrical contact ({clr_mm:g} mm between the element "
                       "and boom surfaces, as modeled)")
    else:
        offset, offset_note = None, "boom not modeled in NEC (simulate with --boom-isolated if it is metallic)"
    all_x = [x_rear_m, x_front_m] + [e["x"] for e in pars]
    span_mm = (max(all_x) - min(all_x)) * 1000.0
    overhang = (boom_len_mm - span_mm) / 2.0
    positions = {"rear": (x_rear_m - boom_x0_m) * 1000.0, "front": (x_front_m - boom_x0_m) * 1000.0}
    positions.update({e["name"]: (e["x"] - boom_x0_m) * 1000.0 for e in pars})
    data["geometry"] = {
        "spacing_center_to_center_mm": geom["spacing_mm"], "spacing_lambda": geom["spacing_mm"] / lam_mm,
        "element_to_boom_axis_offset_mm": offset, "element_to_boom_note": offset_note,
        "boom_shape": args.boom_shape, "boom_size_mm": args.boom_size, "boom_length_mm": boom_len_mm,
        "boom_overhang_each_end_mm": overhang, "element_span_mm": span_mm,
        "positions_from_boom_rear_end_mm": positions,
        "element_tip_distance_from_boom_axis_mm": {"rear": elements["rear"]["nec_length_mm"] / 2.0,
                                                   "front": elements["front"]["nec_length_mm"] / 2.0},
        "notes": ["Elements are perpendicular to the boom; the tips are half an element length from the boom axis "
                  "and well clear of the boom ends.",
                  ("The boom is centered on the span of all elements (as in the model); positions are measured "
                   "from its rear end (the end behind the rear-most element)." if pars else
                   "The boom is assumed centered between the elements, as in the model.")]}
    if overhang < 0:
        data["warnings"].append("boom shorter than the element spacing.")

    # 3. Phasing line ----------------------------------------------------------------------
    elec, phys = line.get("elec_deg"), line.get("phys_m")
    far = "rear" if feed == "front" else "front"
    pl = {"network": args.network}
    if args.network == "tl":
        crossed = bool(args.crossed)
        pl.update({"type": "open-wire (balanced)" if args.line_type == "openwire" else "coaxial",
                   "crossed": crossed, "wires": 2 if args.line_type == "openwire" else "inner conductor + shield",
                   "z0_ohm": line.get("z0"), "vf": args.coax_vf, "electrical_deg": elec, "physical_length_m": phys,
                   "nec_tl_length_m": (phys / args.coax_vf) if phys else None,
                   "fed_element": feed, "far_element": far})
        if args.line_type == "openwire":
            pl.update({"wire_diameter_mm": args.line_wire_diameter, "wire_spacing_mm": line.get("spacing_mm")})
        if crossed:
            att = [f"+a half of the {feed} element <-> -a half of the {far} element",
                   f"-a half of the {feed} element <-> +a half of the {far} element"]
        else:
            att = [f"+a half of the {feed} element <-> +a half of the {far} element",
                   f"-a half of the {feed} element <-> -a half of the {far} element"]
            if args.line_type == "coax":
                att.append("coax: inner conductor on the +a halves, shield on the -a halves, at both ends; the shield "
                           f"end at the {far} element needs a balun/choke (not modeled by the NEC TL)")
        dist_mm = feed_distance_m(args, geom["spacing_mm"] / 1000.0) * 1000.0
        pl["feed_point_distance_mm"] = dist_mm
        if phys:
            pl["slack_mm"] = phys * 1000.0 - dist_mm
            if pl["slack_mm"] < (getattr(args, "line_allowance_mm", 0.0) or 0.0) - 1e-6:
                data["warnings"].append(f"phase line slack {pl['slack_mm']:.1f} mm is below --line-allowance-mm "
                                        f"{args.line_allowance_mm:g}: no room for the connections and the crossing.")
            if phys * 1000.0 > 2.0 * dist_mm:
                data["warnings"].append(f"phase line {phys * 1000:.0f} mm for {dist_mm:.0f} mm between the feed "
                                        "points: it must be folded or coiled, which the ideal NEC TL does not model.")
        if tmatch:
            att = [a.replace("half of the", "T-bar end of the") for a in att]
        pl["attachments"] = att
        pl["physical_length_note"] = ("measured along the line between the two " + ("T-bar centers" if tmatch else
                                      "element feed points") + ", with the line's own VF applied; slack = physical "
                                      "length - straight distance between them")
    elif args.network == "lc":
        pl.update({"type": "lumped LC pi network (ideal)", "crossed": bool(args.crossed), "electrical_deg": elec,
                   "series_l_nh": line.get("lc_l_nh"), "shunt_c_pf_each_port": line.get("lc_c_pf"),
                   "z0_ohm": args.lc_z0,
                   "attachments": [f"between the feed points of the {feed} and {far} elements; "
                                   + ("crossed: reverse the polarity at the far element" if args.crossed
                                      else "straight")],
                   "notes": ["Ideal components: no losses, parasitics or radiation; a physical realization at VHF "
                             "needs its own simulation and bench validation."]})
    else:
        pl.update({"type": "two ideal sources", "notes": ["Not buildable: two ideal excitations."]})
    data["phasing_line"] = pl
    data["half_note"] = HALF_NOTE
    if tmatch:
        tap = geom.get("tap_mm", args.tap_mm)
        data["tmatch"] = {"tap_from_center_mm": tap, "bar_offset_mm": args.tap_offset_mm,
                          "wire_diameter_mm": args.tap_wire_mm or diam, "bar_length_mm": 2.0 * tap,
                          "stub_length_mm": args.tap_offset_mm,
                          "notes": [f"On BOTH driven elements: two stubs of {args.tap_offset_mm:g} mm from the element "
                                    f"at {tap:g} mm each side of the center, joined by a {2 * tap:g} mm bar parallel to "
                                    "the element, on the side facing the other element. The bar is cut at its center: "
                                    "the phase line connects there on both elements and the feed line (through the "
                                    "series capacitor, if any) on the fed one.",
                                    "The stubs and bar are modeled as thin wires; their insulators are not modeled. "
                                    "Slide the taps (both elements alike) during tuning."]}

    # 4. Feed point -----------------------------------------------------------------------
    n = balun_n(args)
    c_pf = geom.get("series_c_pf") if geom.get("series_c_pf") not in (None, "") else series_c_value(args)
    match = {"fed_element": feed if args.network in ("tl", "lc") else None, "balun_ratio": args.balun_ratio,
             "series_c_pf": c_pf,
             "balun_type": "current balun / common-mode choke (MANDATORY)",
             "choke_target": f"common-mode impedance >= 500-1000 ohm at {f:g} MHz (validate on the bench)",
             "reference_impedance_balanced_side_ohm": args.target_impedance, "matching": None}
    if n != 1.0:
        match["balun_note"] = BALUN_4_1_NOTE.format(zb=50.0 / n, zu=50.0)
    if zin is not None:
        vswr = lnet_metrics(zin, args.target_impedance)[0]
        match["zin_design_freq"] = _cplx(zin)
        match["vswr_design_freq"] = vswr
        if geom.get("match_net"):
            net = geom["match_net"]
            match["matching"] = {"topology": lnet_topology_text(net), "series": lnet_describe(net["series"]),
                                 "shunt": lnet_describe(net["shunt"]),
                                 "note": "ideal values of the L-network the optimizer scored the design with; build "
                                         "it on the balanced side and validate with a VNA (see --mode match "
                                         "--balanced)"}
        elif vswr > 1.5:
            try:
                net = lnet_design(zin, args.target_impedance, 2 * math.pi * f * 1e6)
                match["matching"] = {"topology": lnet_topology_text(net),
                                     "series": lnet_describe(net["series"]), "shunt": lnet_describe(net["shunt"]),
                                     "note": "ideal single-frequency L-network on the balanced side; validate "
                                             "with a VNA (see --mode match --balanced)"}
            except ValueError as exc:
                match["matching_error"] = str(exc)
        else:
            match["matching_note"] = f"none needed at {f:g} MHz (VSWR {vswr:.2f}:1 vs {args.target_impedance:g} ohm)"
    else:
        match["matching_note"] = "Zin not available (no single feed point or no nec2c run): no matching values."
    data["feed"] = match

    # 5. Predicted performance --------------------------------------------------------------
    data["performance"] = perf if perf else None
    q_c = series_c_q(args)
    not_inc = ["loss of the balun / common-mode choke and of the feed line and connectors",
               "radiation and coupling of the phasing line (ideal NEC TL/NT network)"]
    if c_pf:
        not_inc.insert(0, f"(the series capacitor is included with Q = {q_c:g})" if q_c > 0 else
                       "loss of the series capacitor (modeled lossless: --series-c-q 0)")
    if args.network == "tl":
        a_db = getattr(args, "line_loss_db_per_m", 0.0) or 0.0
        phys = line.get("phys_m") or 0.0
        if a_db > 0 and phys:
            data["phasing_line"]["loss_upper_bound_db"] = a_db * phys
            not_inc.append(f"phasing-line dissipation, at most {a_db * phys:.3f} dB ({a_db:g} dB/m x "
                           f"{phys:.3f} m, as if all the power crossed the line)")
        else:
            not_inc.append("phasing-line dissipation (give --line-loss-db-per-m for an upper bound)")
    if match.get("matching"):
        not_inc.append("loss of the matching network (see section 4 and --mode match with --q-l/--q-c)")
    data["not_included"] = not_inc

    # 7. Bill of materials ----------------------------------------------------------------
    if args.boom_material:
        boom_mat = args.boom_material
    elif args.mount == "contact":
        boom_mat = "metallic boom; element centers clamped to it (as modeled: bonded)"
    elif getattr(args, "boom_isolated", False):
        boom_mat = "metallic boom with insulating element mounts (as modeled: no electrical contact)"
    else:
        boom_mat = "insulating boom/mounts (the boom is not modeled; if metallic, simulate with --boom-isolated)"
    def tube_txt(name):
        return "; ".join(f"{t['od_mm']:g} mm OD x {t['cut_length_mm']:.1f} mm" for t in elements[name]["sections"])
    bom = [f"Element tubes ({args.tube_wall_mm:g} mm wall), 2 identical halves per element, from the center outward: "
           f"rear {tube_txt('rear')}; front {tube_txt('front')} (cut lengths, telescopic overlaps not included)",
           *([f"Parasitic elements ({args.tube_wall_mm:g} mm wall), one continuous tube each: " + "; ".join(
               f"{q['name']} {q['diameter_mm']:g} mm OD x {q['cut_length_mm']:.1f} mm" for q in data["parasitics"])]
             if data["parasitics"] else []),
           f"Boom: {args.boom_shape} {args.boom_size:g} mm x {boom_len_mm:.0f} mm; {boom_mat}",
           (("Element mounts: metal clamps bonding every element center to the boom (as modeled)"
             if args.mount == "contact" else
             "Element mounts: insulating clamps (as modeled); the continuous T-match elements may instead be bolted "
             "to a metal boom at their centers: simulate that with --mount contact")
            if tmatch else "Insulating element mounts / feed-gap insulators (gap " + f"{gap:g} mm)")]
    if tmatch:
        bom.append(f"T-match: per driven element 2 stubs of {args.tap_offset_mm:g} mm and a {2 * geom.get('tap_mm', args.tap_mm):g} "
                   f"mm bar, {args.tap_wire_mm or diam:g} mm wire or tube, cut at the bar center; insulating support at "
                   "the cut")
    if c_pf:
        bom.append(f"Series feed capacitor {c_pf:.1f} pF (air or PTFE trimmer covering it, rated for the transmit "
                   "voltage), between the feed line and the bar of the fed element")
    if args.network == "tl" and args.line_type == "openwire":
        bom.append(f"Phasing line: 2 wires of {args.line_wire_diameter:g} mm, spacing {line.get('spacing_mm') or 0:.1f} mm, "
                   f"{(phys or 0) * 1000:.0f} mm long between feed points (+ margin for the connections), insulating spacers")
    elif args.network == "tl":
        bom.append(f"Phasing line: coax Z0 = {line.get('z0') or args.coax_z0:g} ohm, VF {args.coax_vf:g}, "
                   f"{(phys or 0) * 1000:.0f} mm between feed points, plus a choke at its element end")
    elif args.network == "lc":
        bom.append("LC pi network components (values above), low-loss VHF parts, in a weatherproof enclosure")
    bom += ["Feed connector suitable for the coax used (impedance: 50 ohm system)",
            f"Ferrite cores / coax choke giving >= 500-1000 ohm common-mode at {f:g} MHz (balun {args.balun_ratio})",
            "Hardware: stainless clamps and bolts, UV-stable cable ties, weather sealing"]
    data["bill_of_materials"] = bom
    data["tuning_procedure"] = list(TUNING_STEPS)
    if tmatch:
        data["tuning_procedure"].insert(4, "(b2) T-match: move the taps of both elements together (same distance from "
                                           "each center) for the F/B optimum, then set the series capacitor for X = 0 "
                                           "at the design frequency.")
    if pars:
        data["tuning_procedure"].insert(3, "(a2) Reflectors and directors: cut them to the NEC lengths and place them at "
                                           "the listed positions; they are passive, so only the front/rear F/B and gain "
                                           "show them. Re-check Zin after fitting each one (they shift it) and trim "
                                           "the directors first, 2-3 mm at a time, for maximum forward signal.")

    # 8. Risks: effects that are NOT in the NEC model, with the sensitivities that bound them ---------------
    mc = (perf or {}).get("mc") if perf else None
    tol = (mc or {}).get("tolerances") or {}
    front_mm, rear_mm = geom["front_mm"], geom["rear_mm"]
    khz_per_mm_front = f * 1000.0 / front_mm          # df/f = dL/L: kHz of resonance shift per mm of element length
    khz_per_mm_rear = f * 1000.0 / rear_mm
    deg_per_mm_spacing = 360.0 / lam_mm               # change of kd (element spacing) per mm
    scale = [f"1 mm of effective element length = {khz_per_mm_front:.0f} kHz of resonance shift (front), "
             f"{khz_per_mm_rear:.0f} kHz (rear)",
             f"1 mm of element spacing = {deg_per_mm_spacing:.3f} deg of spacing phase (kd)"]
    if mc:
        mc_txt = [f"Monte-Carlo ({mc.get('samples', '?')} samples; elements +-{tol.get('length_mm', '?'):g} mm, "
                  f"spacing +-{tol.get('spacing_mm', '?'):g} mm, line +-{tol.get('line_deg', '?'):g} deg): "
                  f"10th percentile F/B sector {mc['fb_p10']:.2f} dB, 10th percentile score {mc['score_p10']:.2f}"
                  + (f", 90th percentile VSWR {mc['vswr_p90']:.2f}" if mc.get("vswr_p90") not in (None, "") else "")]
        tl_mm = tol.get("length_mm")
        if isinstance(tl_mm, (int, float)):
            mc_txt.append(f"+-{tl_mm:g} mm of effective element length is +-{tl_mm * khz_per_mm_front:.0f} kHz "
                          "(front): a detuning of that size is already inside the Monte-Carlo spread")
    else:
        mc_txt = ["Monte-Carlo not run: use --mode optimize --tolerance-mc 100 (or more) to attach the F/B, score "
                  "and VSWR spread under element length, spacing and line tolerances."]
    risks = [
        {"risk": "Insulator dielectric (boom, element mounts, feed-gap insulator, line spacers)",
         "effect": "Not in the NEC model. Dielectric near the element centers and ends adds capacitance across the "
                   "feed gap and loads the elements, so the real resonance is lower than modeled and the phase "
                   "line sees extra shunt capacitance at its ends; lossy plastics add dissipation. Keep the "
                   "dielectric volume small, use low-loss material (PTFE, polyethylene, PP) and keep it away from "
                   "the high-voltage element tips.",
         "nec_sensitivity": scale + mc_txt},
        {"risk": "Ice, rain and wet snow",
         "effect": "Not in the NEC model. A water or ice film on the elements and between the phasing-line wires "
                   "lowers the resonance (the element behaves as longer), lowers the line VF (more electrical "
                   "degrees) and adds loss; the F/B null is the first thing to degrade because it depends on the "
                   "phase relation. Check the antenna wet and iced (or with a dielectric-loaded test) and prefer "
                   "a coated or enclosed phasing line.",
         "nec_sensitivity": scale + mc_txt}]
    if args.network == "tl":
        elec = line.get("elec_deg")
        vf = args.coax_vf
        lines_vf = []
        if elec:
            for s in (0.05, 0.10):
                lo, hi = elec * (1.0 / (1.0 + s) - 1.0), elec * (1.0 / (1.0 - s) - 1.0)
                lines_vf.append(f"real VF {s * 100:.0f} % above / below the {vf:g} used: electrical length "
                                f"{lo:+.1f} / {hi:+.1f} deg on the {elec:.1f} deg line (cut length unchanged)")
        if mc and isinstance(tol.get("line_deg"), (int, float)):
            lines_vf.append(f"Monte-Carlo line tolerance is +-{tol['line_deg']:g} deg: a VF error of "
                            f"{100 * tol['line_deg'] / elec:.1f} % uses all of it" if elec else "")
        typ = ("solid PE coax 0.66, foam PE 0.78-0.85, air-spaced open wire 0.90-0.98 (spacers and "
               "insulation lower it)")
        risks.append({"risk": "Velocity factor of the real cable / line",
                      "effect": f"The phasing line is cut from VF = {vf:g}. Typical real values: {typ}. The "
                                "value printed on the cable is nominal: measure the electrical length of the cut "
                                "line (VNA, open or short at the far end) before connecting it, and trim.",
                      "nec_sensitivity": [x for x in lines_vf if x]})
    data["risks"] = risks
    return data


def _jsonable(o):
    if isinstance(o, complex):
        return _cplx(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _fz(z):
    return "" if z is None else f"{z.real:.2f}{z.imag:+.2f}j"


def render_build_sheet_md(d, args):
    f = d["design_freq_mhz"]
    L = ["# HB9CV build sheet", "",
         f"Generated by hb9cv.py. Design frequency: {f:g} MHz; network: {d['network']}; mount: {d['mount']}.", ""]
    for w in d["warnings"]:
        L.append(f"> WARNING: {w}")
    if d["warnings"]:
        L.append("")
    # 1
    L += ["## 1. Elements", "",
          "| Element | NEC length (mm) | Gap g (mm) | Cut length of each half (mm) | Tubes | OD x wall (mm) | Fed |",
          "|---|---|---|---|---|---|---|"]
    for name in ("rear", "front"):
        e = d["elements"][name]
        L.append(f"| {name} | {e['nec_length_mm']:.1f} | {e['feed_gap_mm']:g} | {e['half_cut_mm']:.1f} | {e['halves']} | "
                 f"{e['diameter_text']} x {e['wall_mm']:g} | {'yes' if e['fed'] else 'no'} |")
    if d.get("parasitics"):
        L += ["", "### 1b. Reflectors and directors (passive)", "",
              "| Element | NEC length (mm) | Cut length (mm) | OD x wall (mm) | Spacing from previous (mm) | "
              "Position from boom rear end (mm) | Allowed length (mm) | Min spacing (mm) |",
              "|---|---|---|---|---|---|---|---|"]
        for q in d["parasitics"]:
            L.append(f"| {q['name']} | {q['nec_length_mm']:.1f} | {q['cut_length_mm']:.1f} | "
                     f"{q['diameter_mm']:g} x {q['wall_mm']:g} | {q['spacing_from_previous_mm']:.1f} | "
                     f"{q['position_from_boom_rear_end_mm']:.1f} | {q['length_min_mm']:.1f} .. {q['length_max_mm']:.1f} | "
                     f"{q['spacing_min_mm']:.1f} |")
    L += [""] + [f"- {s}" for s in d["elements_notes"]] + [""]
    # 2
    g = d["geometry"]
    off = g["element_to_boom_axis_offset_mm"]
    L += ["## 2. Spacing and boom", "",
          f"- Element spacing (center to center): {g['spacing_center_to_center_mm']:.1f} mm "
          f"({g['spacing_lambda']:.4f} lambda)",
          "- Element-to-boom axis offset: " + (f"{off:.1f} mm" if off is not None else "n/a") + f" ({g['element_to_boom_note']})",
          f"- Boom: {g['boom_shape']} {g['boom_size_mm']:g} mm, length {g['boom_length_mm']:g} mm; "
          f"overhang beyond each element: {g['boom_overhang_each_end_mm']:.1f} mm",
          f"- Element tip distance from the boom axis: rear {g['element_tip_distance_from_boom_axis_mm']['rear']:.1f} mm, "
          f"front {g['element_tip_distance_from_boom_axis_mm']['front']:.1f} mm",
          "- Element positions from the rear end of the boom: " + ", ".join(
              f"{k} {v:.1f} mm" for k, v in sorted(g["positions_from_boom_rear_end_mm"].items(), key=lambda kv: kv[1]))
          ] + [f"- {s}" for s in g["notes"]] + [""]
    # 3
    pl = d["phasing_line"]
    L += ["## 3. Phasing line / network", ""]
    for key, label, fmt in (("type", "Type", "{}"), ("wires", "Wires", "{}"),
                            ("wire_diameter_mm", "Wire diameter (mm)", "{:g}"),
                            ("wire_spacing_mm", "Wire spacing, center to center (mm)", "{:.2f}"),
                            ("z0_ohm", "Z0 (ohm)", "{:.1f}"), ("vf", "VF used", "{:g}"),
                            ("electrical_deg", "Electrical length (deg)", "{:.1f}"),
                            ("physical_length_m", "Physical length (m)", "{:.4f}"),
                            ("nec_tl_length_m", "NEC TL card length (m, physical/VF)", "{:.4f}"),
                            ("feed_point_distance_mm", "Straight distance between the line ends (mm)", "{:.1f}"),
                            ("slack_mm", "Slack for connections and crossing (mm)", "{:.1f}"),
                            ("series_l_nh", "Series L (nH)", "{:.3f}"),
                            ("shunt_c_pf_each_port", "Shunt C at each port (pF)", "{:.3f}")):
        if pl.get(key) is not None:
            L.append(f"- {label}: {fmt.format(pl[key])}")
    if "crossed" in pl:
        L.append(f"- Crossed / straight: {'CROSSED' if pl['crossed'] else 'straight'}")
    if "fed_element" in pl:
        L.append(f"- Fed element: {pl['fed_element']}; far element: {pl['far_element']}")
    if pl.get("physical_length_note"):
        L.append(f"- Physical length: {pl['physical_length_note']}")
    L += [f"- Attachment: {a}" for a in pl.get("attachments", [])]
    L += [f"- {n}" for n in pl.get("notes", [])]
    L += [f"- {d['half_note']}", ""]
    tm = d.get("tmatch")
    if tm:
        L += ["### 3b. T-match", "",
              f"- Tap distance from each element center: {tm['tap_from_center_mm']:g} mm (bar {tm['bar_length_mm']:g} mm)",
              f"- Bar offset from the element (toward the other element): {tm['bar_offset_mm']:g} mm",
              f"- Wire diameter of bar and stubs: {tm['wire_diameter_mm']:g} mm"] + [f"- {n}" for n in tm["notes"]] + [""]
    # 4
    fd = d["feed"]
    L += ["## 4. Feed point", "",
          f"- Fed element: {fd['fed_element'] or 'n/a (two ideal sources)'}",
          f"- Balun: {fd['balun_type']}, ratio {fd['balun_ratio']}",
          f"- Choke target: {fd['choke_target']}",
          f"- Reference impedance on the balanced side: {fd['reference_impedance_balanced_side_ohm']:g} ohm"]
    if fd.get("balun_note"):
        L.append(f"- {fd['balun_note']}")
    if fd.get("series_c_pf"):
        L.append(f"- Series feed capacitor: {fd['series_c_pf']:.2f} pF between the feed line and the fed port (included "
                 "in the Zin below)")
    if fd.get("zin_design_freq"):
        z = fd["zin_design_freq"]
        L.append(f"- NEC Zin at {f:g} MHz: {z['r']:.2f}{z['x']:+.2f}j ohm; VSWR {fd['vswr_design_freq']:.2f}:1")
    m = fd.get("matching")
    if m:
        L += [f"- Matching (topology {m['topology']}): series {m['series']}, shunt {m['shunt']}", f"  - {m['note']}"]
    elif fd.get("matching_note"):
        L.append(f"- Matching: {fd['matching_note']}")
    elif fd.get("matching_error"):
        L.append(f"- Matching: not computed ({fd['matching_error']})")
    L.append("")
    # 5
    L += ["## 5. Predicted performance", ""]
    pf = d["performance"]
    if not pf or not (pf.get("design") or pf.get("band")):
        L += ["NOT AVAILABLE: no simulation results in this run (generate needs --run and nec2c; optimize needs a "
              "single feed point).", ""]
    else:
        des = pf.get("design")
        if des:
            L += [f"Design frequency {des['freq_mhz']:.3f} MHz:", "",
                  (f"- Gain: {des['forward_gain']:.2f} dBi = {des['gain_dbd']:.2f} dBd (phi = 0, free space)"
                   if des.get("gain_dbd") is not None else
                   f"- Gain: {des['forward_gain']:.2f} dBi including the ground reflection (phi = 0; no dBd over "
                   "ground)"),
                  f"- F/B: {des['fb']:.2f} dB (point, phi = 180); {des['robust_fb']:.2f} dB (rear sector worst case)"
                  + (f"; {des['cone_fb']:.2f} dB (rear cone, any elevation)" if des.get("cone_fb") is not None
                     else ""),
                  f"- Take-off angle: {des['takeoff_deg']:g} deg; azimuth HPBW: {des['hpbw']:.1f} deg"]
            if des.get("zin") is not None:
                L.append(f"- Zin: {_fz(des['zin'])} ohm; VSWR vs {d['reference_impedance_ohm']:g} ohm: "
                         + (f"{des['vswr']:.2f}:1" if des.get("vswr") is not None else "n/a"))
            L.append("")
        band = pf.get("band") or []
        if band:
            L += [f"Band (VSWR vs {d['reference_impedance_ohm']:g} ohm):", "",
                  "| f (MHz) | Gain dBi | Gain dBd | F/B (dB) | F/B sector (dB) | F/B cone (dB) | Take-off (deg) | "
                  "Zin (ohm) | VSWR |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for x in band:
                def v(key, fmt):
                    return "" if x.get(key) is None else fmt.format(x[key])
                L.append(f"| {x['freq_mhz']:.3f} | {v('forward_gain', '{:.2f}')} | {v('gain_dbd', '{:.2f}')} | "
                         f"{v('fb', '{:.2f}')} | {v('robust_fb', '{:.2f}')} | {v('cone_fb', '{:.2f}')} | "
                         f"{v('takeoff_deg', '{:g}')} | "
                         f"{_fz(x.get('zin'))} | {v('vswr', '{:.2f}')} |")
            L.append("")
        ls = pf.get("loss")
        if ls:
            L += [f"Conductor loss (sigma = {ls['conductivity_S_m']:.3g} S/m, skin depth {ls['skin_depth_um']:.1f} um): "
                  f"gain {ls['delta_gain_dB']:+.3f} dB vs perfect conductors (efficiency "
                  f"{100 * ls['efficiency_estimate']:.2f} %, {ls.get('efficiency_source', 'gain difference')}), "
                  f"F/B {ls['delta_fb_dB']:+.2f} dB"
                  + (f", R_in {ls['delta_r_ohm']:+.2f} ohm" if "delta_r_ohm" in ls else "") + ".", ""]
        ag = pf.get("agt")
        if ag:
            L += [f"Average gain test (free space, perfect conductors): AGT = {ag['agt']:.3f} ({ag['agt_db']:+.2f} dB)"
                  + ("; the NEC gains above are low by about that amount and the input impedance is less reliable "
                     "(see the build sheet risks)" if abs(ag["agt_db"]) > AGT_WARN_DB else "; no correction needed")
                  + ".", ""]
        mc = pf.get("mc")
        L += ["Sensitivity to mechanical tolerances (Monte-Carlo):", ""]
        if mc:
            L += ["| Metric | Value |", "|---|---|",
                  f"| Samples | {mc.get('samples', '')} |",
                  f"| Score, 10th percentile | {mc['score_p10']:.2f} |",
                  f"| F/B sector, 10th percentile | {mc['fb_p10']:.2f} dB |",
                  "| VSWR, 90th percentile | " + (f"{mc['vswr_p90']:.2f}" if mc.get("vswr_p90") not in (None, "") else "n/a") + " |"]
            tol = mc.get("tolerances")
            if tol:
                L.append(f"| Tolerances (uniform) | elements +-{tol['length_mm']:g} mm, spacing +-{tol['spacing_mm']:g} mm, "
                         f"line +-{tol['line_deg']:g} deg; seed {tol['seed']} |")
            L.append("")
        else:
            L += ["Not run: use `--mode optimize --tolerance-mc 100` (or more) to fill this table.", ""]
    L += ["Not included in the figures above: " + "; ".join(d.get("not_included", [])) + ".", ""]
    # 6
    L += ["## 6. Tuning procedure", ""] + [f"{i}. {s}" for i, s in enumerate(d["tuning_procedure"], 1)] + [""]
    # 7
    L += ["## 7. Bill of materials", ""] + [f"- {s}" for s in d["bill_of_materials"]] + [""]
    # 8
    L += ["## 8. Risks not covered by the NEC model", ""]
    for rk in d.get("risks", []):
        L += [f"### {rk['risk']}", "", rk["effect"], ""]
        L += [f"- {s}" for s in rk["nec_sensitivity"]] + [""]
    return "\n".join(L)


def write_build_sheet(path, args, geom, line_lines, zin=None, perf=None, line=None):
    """Write the build sheet: FILE (markdown) and the same data as JSON next to it (.json)."""
    md_path = Path(path)
    if md_path.suffix.lower() == ".json":
        md_path = md_path.with_suffix(".md")
    json_path = md_path.with_suffix(".json")
    data = build_sheet_data(args, geom, line or {}, zin, perf)
    data["phasing_line_summary"] = list(line_lines)
    for w in data["warnings"]:
        print(f"WARNING: build sheet: {w}", file=sys.stderr)
    md_path.write_text(render_build_sheet_md(data, args), encoding="utf-8")
    json_path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Build sheet written: {md_path.resolve()}\nBuild sheet data (JSON): {json_path.resolve()}")


def cm_sensitivity(exe, gargs, test_len, rear_start=150, rear_end=210):
    """Simulate the same model without and with the common-mode feed-line wire and report the
    change in F/B and Zin (over ground: at the take-off angle of each run). Raises
    ValueError/OSError if a run fails."""
    res = []
    with tempfile.TemporaryDirectory(prefix="hb9cv_cm_") as td:
        for i, ln in enumerate((0.0, test_len)):
            a = Namespace(**vars(gargs))
            a.feedline_length, a.fsweep = ln, None
            r = simulate_model(exe, td, f"cm_{i}", a, [a.freq], rear_start, rear_end, pt_mode=-1)
            if not r["ok"]:
                raise ValueError(f"common-mode sensitivity run failed: {r['reason']}")
            res.append(r["per_freq"][0])
            res[-1]["model_warnings"] = r.get("model_warnings", [])
    m0, m1 = res
    z0, z1 = m0["zin"], m1["zin"]
    print(f"\nCOMMON-MODE SENSITIVITY (open-ended shield wire of {test_len:g} m, no choke; rear sector "
          f"{rear_start}..{rear_end} deg)")
    print(f"{'':28s}{'without':>12s}{'with wire':>12s}{'delta':>10s}")
    for label, key in (("gain at phi=0 (dB)", "forward_gain"), ("F/B at phi=180 (dB)", "fb"),
                       ("F/B rear sector (dB)", "robust_fb")):
        print(f"{label:28s}{m0[key]:12.2f}{m1[key]:12.2f}{m1[key] - m0[key]:+10.2f}")
    print(f"{'max gain angle (deg)':28s}{m0['max_angle']:12g}{m1['max_angle']:12g}{m1['max_angle'] - m0['max_angle']:+10g}")
    print(f"Zin without: {z0.real:.2f}{z0.imag:+.2f}j ohm; with wire: {z1.real:.2f}{z1.imag:+.2f}j ohm; "
          f"delta Zin = {(z1 - z0).real:+.2f}{(z1 - z0).imag:+.2f}j ohm (|delta| = {abs(z1 - z0):.2f})")
    print_model_warnings("with shield wire", [w for w in m1["model_warnings"] if w not in m0["model_warnings"]])
    print("Indicative only: the wire is an open-ended shield with no choke and no real coax routing; use it to "
          "judge sensitivity, not to predict the real pattern.")


def mast_sensitivity(exe, gargs, lengths, rear_start=150, rear_end=210):
    """Simulate the model without a mast and with masts of each length in `lengths` (m; diameter and gap from
    gargs) and print a table of gain, F/B and Zin changes. Raises ValueError if the baseline fails; failed
    mast runs are listed as such."""
    rows = []
    with tempfile.TemporaryDirectory(prefix="hb9cv_mast_") as td:
        for i, ln in enumerate((0.0, *lengths)):
            a = Namespace(**vars(gargs))
            a.mast_length_m, a.fsweep, a.feedline_length = ln, None, 0.0
            r = simulate_model(exe, td, f"mast_{i}", a, [a.freq], rear_start, rear_end, pt_mode=-1)
            if not r["ok"]:
                if i == 0:
                    raise ValueError(f"mast sensitivity baseline failed: {r['reason']}")
                rows.append((ln, None, r["reason"]))
                continue
            rows.append((ln, r["per_freq"][0], None))
            rows[-1][1]["model_warnings"] = r.get("model_warnings", [])
    base = rows[0][1]
    zb = base["zin"]
    if getattr(gargs, "orientation", "horizontal") == "vertical":
        where = (f"side-mounted parallel to the elements, {gargs.mast_gap_mm + (gargs.mast_offset_mm or 0.0):g} mm "
                 "behind the rear end of the boom")
    else:
        where = (f"top {gargs.mast_gap_mm:g} mm below the lowest modeled conductor"
                 + (f", {gargs.mast_offset_mm:g} mm to the side of the boom axis" if gargs.mast_offset_mm else
                    " on the boom axis (the symmetry plane of the elements)"))
    print(f"\nMAST SENSITIVITY (mast {gargs.mast_diameter_mm:g} mm diameter, {where}, no galvanic junction; rear "
          f"sector {rear_start}..{rear_end} deg)")
    head = f"{'mast length (m)':>16s}{'gain dBi':>10s}{'dGain':>8s}{'F/B':>8s}{'dF/B':>8s}{'F/B sect':>10s}{'dSect':>8s}"
    if zb is not None:
        head += f"{'Zin (ohm)':>20s}{'|dZin|':>9s}"
    print(head)
    for ln, m, why in rows:
        if m is None:
            print(f"{ln:16.3f}  run failed: {why}")
            continue
        line = (f"{'none' if ln == 0 else format(ln, '.3f'):>16s}{m['forward_gain']:10.2f}"
                f"{m['forward_gain'] - base['forward_gain']:+8.2f}{m['fb']:8.2f}{m['fb'] - base['fb']:+8.2f}"
                f"{m['robust_fb']:10.2f}{m['robust_fb'] - base['robust_fb']:+8.2f}")
        if zb is not None and m["zin"] is not None:
            z = m["zin"]
            line += f"{z.real:11.2f}{z.imag:+8.2f}j{abs(z - zb):9.2f}"
        print(line)
    seen = set(base.get("model_warnings", []))
    for ln, m, _why in rows[1:]:
        if m is not None:
            print_model_warnings(f"mast {ln:g} m", m.get("model_warnings"), seen)
    print("Indicative only: the mast is a single straight conductor, not galvanically connected; real masts, "
          "clamps, guys and nearby structures change the result. Use the spread as a sensitivity estimate.")


def _single_run(exe, gargs, freqs, rear_start=150, rear_end=210, **kw):
    """simulate_model of a variant of gargs in a temporary folder; the per-frequency rows, or ValueError. The first
    row carries the geometry warnings of the variant under "model_warnings"."""
    with tempfile.TemporaryDirectory(prefix="hb9cv_chk_") as td:
        r = simulate_model(exe, td, "chk", _variant(gargs, fsweep=None, **kw), freqs, rear_start, rear_end, pt_mode=-1)
    if not r["ok"]:
        raise ValueError(f"{r['stage']}: {r['reason']}")
    r["per_freq"][0]["model_warnings"] = list(r.get("model_warnings", []))
    return r["per_freq"]


def print_model_warnings(label, warnings, seen=None):
    """Print the geometry warnings of one variant run once each (seen: set shared by the runs of one report)."""
    seen = set() if seen is None else seen
    notes = [w for w in warnings or [] if "is large relative to the segment length" in w]
    for w in warnings or []:
        if w not in seen and w not in notes:
            seen.add(w)
            print(f"  WARNING ({label}): {w}")
    new_notes = [w for w in notes if w not in seen]
    seen.update(new_notes)
    if new_notes:
        print(f"  NOTE ({label}): {len(new_notes)} wire(s) with segments shorter than 4 radii (reduced thin-wire "
              "accuracy; check the average gain test)")


def resonance_estimate(exe, gargs, rear_start=150, rear_end=210, rel_step=0.005):
    """Frequency (MHz) where X_in crosses zero, from a straight-line fit of X_in at f*(1-rel_step), f and
    f*(1+rel_step); None when there is no single feed point or the slope is zero."""
    if gargs.network not in ("tl", "lc"):
        return None
    f0 = gargs.freq
    rows = _single_run(exe, gargs, [f0 * (1 - rel_step), f0, f0 * (1 + rel_step)], rear_start, rear_end)
    fs, xs = [r["freq_mhz"] for r in rows], [r["zin"].imag for r in rows]
    fm, xm = sum(fs) / 3.0, sum(xs) / 3.0
    den = sum((f - fm) ** 2 for f in fs)
    slope = sum((f - fm) * (x - xm) for f, x in zip(fs, xs)) / den if den else 0.0
    if slope == 0.0:
        return None
    return fm - xm / slope, xs[1], slope


def convergence_check(exe, gargs, rear_start=150, rear_end=210):
    """Simulate the model with --segments n and 2n+1 and report the change of gain, F/B and R_in. Large
    changes mean the segmentation is too coarse for the numbers to be trusted."""
    n = gargs.segments
    try:
        a = _single_run(exe, gargs, [gargs.freq], rear_start, rear_end)[0]
        b = _single_run(exe, gargs, [gargs.freq], rear_start, rear_end, segments=2 * n + 1)[0]
    except (ValueError, OSError) as exc:
        print(f"WARNING: convergence check failed: {exc}", file=sys.stderr)
        return None
    dg, dfb = b["forward_gain"] - a["forward_gain"], b["robust_fb"] - a["robust_fb"]
    print(f"\nSEGMENTATION CONVERGENCE at {gargs.freq:g} MHz ({n} vs {2 * n + 1} segments per element)")
    print(f"  gain {a['forward_gain']:.3f} -> {b['forward_gain']:.3f} dBi (delta {dg:+.3f} dB); "
          f"F/B sector {a['robust_fb']:.2f} -> {b['robust_fb']:.2f} dB (delta {dfb:+.2f} dB)")
    bad = abs(dg) > 0.1 or abs(dfb) > 1.0
    out = {"segments": [n, 2 * n + 1], "delta_gain_dB": dg, "delta_fb_sector_dB": dfb}
    if a["zin"] is not None and b["zin"] is not None:
        dr = (b["zin"].real - a["zin"].real) / a["zin"].real * 100.0
        print(f"  Zin {a['zin'].real:.2f}{a['zin'].imag:+.2f}j -> {b['zin'].real:.2f}{b['zin'].imag:+.2f}j ohm "
              f"(R {dr:+.1f} %)")
        out["delta_r_percent"] = dr
        bad = bad or abs(dr) > 5.0
    print("  WARNING: not converged (limits 0.1 dB gain, 1 dB F/B, 5 % R_in): raise --segments and re-run."
          if bad else "  Converged within 0.1 dB gain, 1 dB F/B and 5 % R_in.")
    seen = set()
    print_model_warnings(f"{n} segments", a.get("model_warnings"), seen)
    print_model_warnings(f"{2 * n + 1} segments", b.get("model_warnings"), seen)
    out["converged"] = not bad
    return out


ROBUST_MAST_OFFSET_MM = 60.0   # side-clamp arm of the robustness mast case (horizontal elements)
ROBUST_SIDE_MAST_M = 2.0       # side-mounted mast of the robustness case (vertical elements)


def robustness_check(exe, gargs, rear_start=150, rear_end=210):
    """One table of the change in gain, F/B and Zin caused by the parts the default model leaves out, each simulated
    on its own against the model as given: a metallic boom (when it is not modeled); a mast, placed where it breaks
    the symmetry of the balanced elements (horizontal: 1 m below the boom on a side clamp; vertical: a side-mounted
    mast parallel to the elements behind the boom); and an unchoked coax shield of lambda/2 (single feed point).
    A conductor lying in the symmetry plane of the balanced elements (a boom on their axis, a mast under the boom
    center) is not excited by them; such rows are marked instead of being reported as robustness."""
    lam = C_LIGHT / (gargs.freq * 1e6)
    cases = [("model as given", {})]
    if not getattr(gargs, "boom_isolated", False):
        cases.append(("metallic boom", {"boom_isolated": True}))
    horizontal = getattr(gargs, "orientation", "horizontal") == "horizontal"
    no_feedline = not (getattr(gargs, "feedline_length", 0.0) or 0.0) > 0
    if no_feedline and not (getattr(gargs, "mast_length_m", 0.0) or 0.0) > 0:
        if horizontal:
            cases.append((f"mast 1 m, {ROBUST_MAST_OFFSET_MM:g} mm side clamp",
                          {"mast_length_m": 1.0, "mast_offset_mm": ROBUST_MAST_OFFSET_MM}))
        else:
            cases.append((f"side mast {ROBUST_SIDE_MAST_M:g} m behind the boom",
                          {"mast_length_m": ROBUST_SIDE_MAST_M, "mast_offset_mm": 0.0}))
    fed = element_sections(gargs, "front" if gargs.feed_element == "front" else "rear")
    if (gargs.network in ("tl", "lc") and no_feedline and not getattr(gargs, "model_feed_gap", False)
            and len(fed) == 1 and not (getattr(gargs, "mast_length_m", 0.0) or 0.0) > 0):
        cases.append(("coax shield lambda/2, no choke", {"feedline_length": lam / 2.0}))
    rows = []
    for label, kw in cases:
        try:
            rows.append((label, _single_run(exe, gargs, [gargs.freq], rear_start, rear_end, **kw)[0], None))
        except (ValueError, OSError) as exc:
            if not rows:
                print(f"WARNING: robustness check failed: {exc}", file=sys.stderr)
                return None
            rows.append((label, None, str(exc)))
    base = rows[0][1]
    print(f"\nROBUSTNESS at {gargs.freq:g} MHz (each effect alone; indicative models, see the README)")
    print(f"{'case':<40s}{'gain dBi':>9s}{'dGain':>8s}{'F/B sect':>9s}{'dF/B':>7s}{'F/B cone':>9s}{'|dZin|':>8s}")
    out, symmetric, flagged = [], False, False
    for i, (label, m, err) in enumerate(rows):
        if m is None:
            print(f"{label:<40s}  run failed: {err}")
            continue
        own = [w for w in m.get("model_warnings", []) if i == 0 or w not in base.get("model_warnings", [])]
        if own and i > 0:
            label, flagged = label + " !", True
        dz = abs(m["zin"] - base["zin"]) if (m["zin"] is not None and base["zin"] is not None) else None
        dg, dfb = m["forward_gain"] - base["forward_gain"], m["robust_fb"] - base["robust_fb"]
        sym = i > 0 and abs(dg) < 0.005 and abs(dfb) < 0.005 and (dz is None or dz < 0.01)
        symmetric = symmetric or sym
        print(f"{label + (' *' if sym else ''):<40s}{m['forward_gain']:9.2f}{dg:+8.2f}"
              f"{m['robust_fb']:9.2f}{dfb:+7.2f}{m['cone_fb']:9.2f}"
              + (f"{dz:8.2f}" if dz is not None else f"{'':>8s}"))
        out.append({"case": label, "gain_dBi": m["forward_gain"], "fb_sector_dB": m["robust_fb"],
                    "fb_cone_dB": m["cone_fb"], "zin": m["zin"], "symmetric_no_coupling": sym,
                    "model_warnings": m.get("model_warnings", [])})
    if symmetric:
        print("  * no change at all: the conductor lies in the symmetry plane of the balanced elements and the model "
              "does not excite it. This is a property of the symmetric thin-wire model, not a measured robustness: "
              "real through-boom or plate mounts, clamps and an asymmetric feed couple to it (boom correction of "
              "the element length, detuning); check them on the bench.")
    if flagged:
        print("  ! the model of this case breaks a thin-wire or clearance rule (listed below): its deltas are "
              "less reliable.")
    seen = set()
    for label, m, _err in rows:
        if m is not None:
            print_model_warnings(label, m.get("model_warnings", []), seen)
    return out


def write_json_out(args, data):
    """--json-out FILE: the main results of the run as JSON (used by the GUI instead of parsing the log)."""
    path = getattr(args, "json_out", None)
    if not path:
        return
    data = dict(data, mode=args.mode, written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    try:
        Path(path).write_text(json.dumps(_jsonable(data), indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        print(f"WARNING: could not write {path}: {exc}", file=sys.stderr)


POWER_RE = {key: re.compile(rf"{label}\s*=\s*({FLOAT})", re.IGNORECASE) for key, label in (
    ("input_w", r"INPUT POWER"), ("radiated_w", r"RADIATED POWER"), ("structure_loss_w", r"STRUCTURE LOSS"),
    ("network_loss_w", r"NETWORK LOSS"), ("efficiency_pct", r"EFFICIENCY"))}


def parse_power_budget(text, section=0):
    """POWER BUDGET of one frequency block (0 = first) of a nec2c .out text: dict with input_w, radiated_w,
    structure_loss_w, network_loss_w and efficiency_pct (radiated / input), or None when it is not printed."""
    blocks = ["\n".join(lines) for _f, lines in nec_blocks(text)[1:] if any("POWER BUDGET" in l for l in lines)]
    if len(blocks) <= section:
        return None
    chunk = blocks[section].split("POWER BUDGET", 1)[1][:800]
    out = {}
    for key, rx in POWER_RE.items():
        m = rx.search(chunk)
        if m:
            out[key] = float(m.group(1))
    if "input_w" not in out or "radiated_w" not in out:
        return None
    if "efficiency_pct" not in out and out["input_w"] > 0:
        out["efficiency_pct"] = 100.0 * out["radiated_w"] / out["input_w"]
    return out


def skin_depth_um(freq_mhz, sigma):
    """Skin depth in micrometers of a non-magnetic conductor."""
    return 1e6 / math.sqrt(math.pi * freq_mhz * 1e6 * 4e-7 * math.pi * sigma)


def conductor_loss_check(exe, gargs, rear_start=150, rear_end=210):
    """Simulate the model with perfect conductors and with --conductivity and report the change in gain,
    F/B and R_in. Returns a dict (also stored in the build sheet) or None if there is nothing to compare
    or a run fails."""
    sigma = getattr(gargs, "conductivity", 0.0) or 0.0
    if sigma <= 0:
        print("\nCONDUCTOR LOSS: --conductivity 0 (perfect conductors); losses are not modeled.")
        return None
    res, budget = [], None
    try:
        with tempfile.TemporaryDirectory(prefix="hb9cv_ld_") as td:
            for i, sg in enumerate((0.0, sigma)):
                a = Namespace(**vars(gargs))
                a.conductivity, a.fsweep = sg, None
                r = simulate_model(exe, td, f"ld_{i}", a, [a.freq], rear_start, rear_end, pt_mode=-1)
                if not r["ok"]:
                    raise ValueError(r["reason"])
                res.append(r["per_freq"][0])
                if sg > 0:
                    budget = parse_power_budget(Path(r["out_path"]).read_text(errors="replace"))
    except (ValueError, OSError) as exc:
        print(f"WARNING: conductor-loss check failed: {exc}", file=sys.stderr)
        return None
    m0, m1 = res
    dg = m1["forward_gain"] - m0["forward_gain"]
    eff_gain = 10.0 ** (dg / 10.0)
    # NEC's own power budget (radiated / input power) is exact for the conductor loss; the gain difference is
    # kept as a cross-check (it assumes an unchanged directivity and carries the 0.01 dB rounding of the table).
    eff = min(budget["efficiency_pct"] / 100.0, 1.0) if budget and budget.get("efficiency_pct") is not None \
        else min(eff_gain, 1.0)
    out = {"conductivity_S_m": sigma, "freq_mhz": gargs.freq, "skin_depth_um": skin_depth_um(gargs.freq, sigma),
           "gain_perfect_dBi": m0["forward_gain"], "gain_lossy_dBi": m1["forward_gain"], "delta_gain_dB": dg,
           "efficiency_estimate": eff, "efficiency_source": "nec power budget" if budget else "gain difference",
           "efficiency_from_gain": eff_gain, "power_budget": budget, "delta_fb_dB": m1["fb"] - m0["fb"],
           "delta_fb_sector_dB": m1["robust_fb"] - m0["robust_fb"]}
    z0, z1 = m0["zin"], m1["zin"]
    if z0 is not None and z1 is not None:
        out["zin_perfect"], out["zin_lossy"] = z0, z1
        out["delta_r_ohm"] = z1.real - z0.real
    print(f"\nCONDUCTOR LOSS at {gargs.freq:g} MHz (sigma = {sigma:.3g} S/m; skin depth {out['skin_depth_um']:.1f} um)")
    print(f"{'':28s}{'perfect':>12s}{'lossy':>12s}{'delta':>10s}")
    for label, key in (("gain at phi=0 (dBi)", "forward_gain"), ("F/B at phi=180 (dB)", "fb"),
                       ("F/B rear sector (dB)", "robust_fb")):
        print(f"{label:28s}{m0[key]:12.2f}{m1[key]:12.2f}{m1[key] - m0[key]:+10.2f}")
    if z0 is not None and z1 is not None:
        print(f"{'Zin (ohm)':28s}{z0.real:7.2f}{z0.imag:+.2f}j{z1.real:7.2f}{z1.imag:+.2f}j{z1.real - z0.real:+8.2f} (R)")
    loss_db = max(0.0, -10 * math.log10(eff)) if eff > 0 else float("inf")
    if budget:
        print(f"Estimated efficiency of the conductors: {100 * eff:.2f} % (loss {loss_db:.3f} dB) from the NEC power "
              f"budget (structure loss {budget.get('structure_loss_w', 0.0):.3e} W of {budget['input_w']:.3e} W); "
              f"the gain difference gives {100 * min(eff_gain, 1.0):.1f} %. The TL/NT phase network is lossless in "
              "NEC.")
    else:
        print(f"Estimated efficiency of the conductors: {100 * eff:.1f} % (loss {loss_db:.3f} dB) from the gain "
              "difference (assumes the directivity is unchanged); the TL/NT phase network is lossless in NEC.")
    return out


AGT_STEP_DEG = 5.0
AGT_WARN_DB = 0.5


def sphere_average_gain(points, step):
    """Average of the power gain over the full sphere from a theta 0..180 / phi 0..360 grid of step degrees
    (trapezoid in theta with the sin(theta) weight, rectangle in phi)."""
    nth = int(round(180.0 / step)) + 1
    nph = int(round(360.0 / step))
    total = 0.0
    for i in range(nth):
        th = i * step
        ring = 0.0
        for j in range(nph):
            v = points.get((round(th, 2), round((j * step) % 360.0, 5)))
            if v is not None and v[2] > -900:
                ring += 10.0 ** (v[2] / 10.0)
        w = 0.5 if i in (0, nth - 1) else 1.0
        total += w * ring * math.sin(math.radians(th))
    return total * math.radians(step) * (2.0 * math.pi / nph) / (4.0 * math.pi)


def average_gain_test(exe, gargs, step=AGT_STEP_DEG):
    """NEC average gain test: the model in free space with perfect conductors, its gain averaged over the sphere.
    A correct current solution gives 1; the departure measures the error of the input power NEC normalizes the
    gain with (the reported gains are low by -10*log10(AGT) dB). Returns a dict or None when the run fails."""
    a = _variant(gargs, ground="free", conductivity=0.0, fsweep=None, fsweep_pattern=False, band_freqs=None,
                 pattern_mode="ring", ring_theta=90.0, pt_mode=-1, rear_cone_rp=False, mast_length_m=0.0)
    nth, nph = int(round(180.0 / step)) + 1, int(round(360.0 / step)) + 1
    try:
        deck = build_deck(a)[0].splitlines()
        cut = next(i for i, l in enumerate(deck) if l.startswith("RP"))
        deck = deck[:cut] + [f"RP 0 {nth} {nph} 1000 0 0 {step:g} {step:g}"] + [l for l in deck[cut:]
                                                                                 if not l.startswith("RP")]
        with tempfile.TemporaryDirectory(prefix="hb9cv_agt_") as td:
            out = _run_deck(exe, Path(td), Path(td) / "agt.nec", "\n".join(deck))
            text, _ = _read_checked(out)
        sections = parse_pattern_sections(text)
        if not sections:
            raise ValueError("RADIATION PATTERNS table not found")
        agt = sphere_average_gain(sections[0]["points"], step)
    except (ValueError, OSError, StopIteration) as exc:
        print(f"WARNING: average gain test failed: {exc}", file=sys.stderr)
        return None
    db = 10.0 * math.log10(agt) if agt > 0 else -math.inf
    print(f"\nAVERAGE GAIN TEST (free space, perfect conductors, {step:g} deg sphere): AGT = {agt:.3f} "
          f"({db:+.2f} dB)")
    if abs(db) > AGT_WARN_DB:
        print(f"  WARNING: the current solution does not conserve power: NEC gains are off by {-db:+.2f} dB and the "
              "input impedance is less reliable. Usual causes: junctions of unequal radii (--tap-wire-mm), a T-bar "
              "too close to its element (--tap-offset-mm), too few segments (--segments), very thick tubes.")
    else:
        print(f"  Within {AGT_WARN_DB:g} dB: the reported gains need no correction (corrected = gain {-db:+.2f} dB).")
    return {"agt": agt, "agt_db": db, "step_deg": step, "corrected_gain_offset_db": -db}


def resolve_ground_options(p, args):
    """Validate/resolve --ground, --height-*, --orientation, --ground-eps/sigma (generate and optimize).

    Leaves args.height_m in meters (--height-lambda is converted with the design frequency).
    """
    lam = C_LIGHT / (args.freq * 1e6)
    if args.height_m is not None and args.height_lambda is not None:
        p.error("give --height-m or --height-lambda, not both")
    height = args.height_m if args.height_m is not None else (
        args.height_lambda * lam if args.height_lambda is not None else None)
    if args.ground == "free":
        if height is not None:
            print("WARNING: --height-m/--height-lambda has no effect on the pattern in free space "
                  "(--ground free).", file=sys.stderr)
        if args.ground_eps is not None or args.ground_sigma is not None:
            print("WARNING: --ground-eps/--ground-sigma are ignored with --ground free.", file=sys.stderr)
        if args.orientation == "vertical":
            print("NOTE: in free space the horizontal and vertical orientations give the same pattern.")
        args.height_m = height or 0.0
        return
    if height is None:
        p.error(f"--ground {args.ground} needs --height-m or --height-lambda")
    args.height_m = height
    if args.ground == "perfect":
        if args.ground_eps is not None or args.ground_sigma is not None:
            print("WARNING: --ground-eps/--ground-sigma are ignored with --ground perfect.", file=sys.stderr)
    else:
        eps0, sigma0 = GROUND_PRESETS[args.ground]
        args.ground_eps = eps0 if args.ground_eps is None else args.ground_eps
        args.ground_sigma = sigma0 if args.ground_sigma is None else args.ground_sigma


def check_parasitic_args(p, args):
    """Validate --reflectors/--directors and their value lists (converted in place to tuples of floats)."""
    if not 0 <= args.reflectors <= MAX_REFLECTORS:
        p.error(f"--reflectors must be between 0 and {MAX_REFLECTORS}")
    if not 0 <= args.directors <= MAX_DIRECTORS:
        p.error(f"--directors must be between 0 and {MAX_DIRECTORS}")
    for attr, flag, count, lo, hi in (
            ("reflector_lengths", "--reflector-lengths", args.reflectors, 0.5, 1.5),
            ("reflector_spacings", "--reflector-spacings", args.reflectors, 0.01, 0.5),
            ("director_lengths", "--director-lengths", args.directors, 0.5, 1.5),
            ("director_spacings", "--director-spacings", args.directors, 0.01, 0.5)):
        raw = getattr(args, attr)
        if raw is None or isinstance(raw, tuple):
            continue
        try:
            vals = parse_values(raw, flag, lo, hi)
        except argparse.ArgumentTypeError as exc:
            p.error(str(exc))
        if len(vals) > count:
            p.error(f"{flag} has {len(vals)} values but only {count} element(s) were requested "
                    f"({'--reflectors' if 'reflector' in flag else '--directors'})")
        setattr(args, attr, tuple(vals))
    if args.min_gap_mm < 0 or args.boom_overhang_mm < 0:
        p.error("--min-gap-mm and --boom-overhang-mm must be >= 0")


METAL_BOOM_RE = re.compile(r"metal|alumin|steel|copper|brass|iron", re.IGNORECASE)


def check_common_model_args(p, args):
    """Validation shared by the generate and optimize modes."""
    check_parasitic_args(p, args)
    if args.mount == "contact":
        args.boom_isolated = True
    if not args.boom_isolated:
        if args.boom_material and METAL_BOOM_RE.search(args.boom_material):
            args.boom_isolated = True
            print(f"NOTE: --boom-material '{args.boom_material}' is metallic: the boom is modeled (--boom-isolated).")
        else:
            print("NOTE: the boom is not modeled (insulating boom assumed); for a metallic boom add --boom-isolated "
                  "or --robustness.")
    n_balun = balun_n(args)
    if args.target_impedance is None:
        args.target_impedance = 50.0 / n_balun
    elif n_balun != 1.0 and abs(args.target_impedance - 50.0 / n_balun) > 1e-9:
        print(f"WARNING: --balun-ratio {args.balun_ratio} implies a balanced-side reference of "
              f"{50.0 / n_balun:g} ohm; you gave --target-impedance {args.target_impedance:g}.", file=sys.stderr)
    single = args.network in ("tl", "lc")
    if args.match_score is None:
        args.match_score = args.mode == "optimize" and single
    elif args.match_score and not single:
        print("WARNING: --match-score needs a single feed point (--network tl or lc); ignored.", file=sys.stderr)
        args.match_score = False
    if args.zin_x_max is None:
        args.zin_x_max = MATCHED_ZIN_X_MAX if args.match_score else (30.0 if n_balun == 1.0 else 10.0)
    if args.zin_r_range is None:
        args.zin_r_range = MATCHED_ZIN_R_RANGE if args.match_score else DIRECT_ZIN_R_RANGE
    needs_center = [flag for flag, on in (("--rear-sections/--front-sections", args.rear_sections or args.front_sections),
                                          ("--model-feed-gap", args.model_feed_gap)) if on]
    if args.feed_style is None:
        # Default: the classic HB9CV T-match feed; a center feed when the elements are split or stepped.
        args.feed_style = "tmatch" if single and not needs_center else "center"
        if single and needs_center:
            print(f"NOTE: {', '.join(needs_center)} given: the phase network connects at the element centers "
                  "(--feed-style center).")
    elif args.feed_style == "tmatch":
        if not single:
            print("WARNING: --feed-style tmatch needs --network tl or lc; the element centers are used.",
                  file=sys.stderr)
            args.feed_style = "center"
        elif args.rear_sections or args.front_sections:
            p.error("--feed-style tmatch needs uniform elements: do not give --rear-sections/--front-sections")
        elif args.model_feed_gap:
            p.error("--feed-style tmatch uses continuous elements: do not combine it with --model-feed-gap")
    if args.mount == "contact" and args.feed_style != "tmatch":
        print("WARNING: --mount contact bonds the element centers to the boom, which shorts a center feed point; "
              "with --feed-style center the elements are modeled insulated and the metallic boom as a separate "
              "conductor below them (--mount isolated --boom-isolated).", file=sys.stderr)
        args.mount = "isolated"
    if args.boom_clearance_mm is not None and args.boom_clearance_mm <= 0:
        p.error("--boom-clearance-mm must be greater than zero")
    if args.series_c_pf is not None and not single:
        p.error("--series-c-pf needs a single feed point (--network tl or lc)")
    if args.feedline_length > 0 and args.network not in ("tl", "lc"):
        p.error("--feedline-length-m needs a single feed point: use --network tl or lc")
    if args.cm_sensitivity and args.network not in ("tl", "lc"):
        p.error("--cm-sensitivity needs --network tl or lc")
    if args.cm_sensitivity and getattr(args, "fsweep", None):
        p.error("--cm-sensitivity cannot be combined with --fsweep (it needs the radiation pattern)")
    if args.mast_sensitivity is not None:
        try:
            args.mast_sensitivity = parse_values(args.mast_sensitivity, "--mast-sensitivity", 0.01, 50.0)
        except argparse.ArgumentTypeError as exc:
            p.error(str(exc))
        if getattr(args, "fsweep", None):
            p.error("--mast-sensitivity cannot be combined with --fsweep (it needs the radiation pattern)")
    mast_wanted = args.mast_length_m > 0 or bool(args.mast_sensitivity)
    if mast_wanted and (args.feedline_length > 0 or args.cm_sensitivity):
        p.error("the mast cannot be combined with --feedline-length-m / --cm-sensitivity (the two wires would "
                "cross below the boom); run them separately")
    resolve_line_options(p, args)
    resolve_ground_options(p, args)
    if args.segments is None:
        args.segments = DEFAULT_SEGMENTS_TMATCH if args.feed_style == "tmatch" else DEFAULT_SEGMENTS
    if args.segments < 5 or args.segments % 2 == 0:
        p.error("--segments must be odd and >= 5")
    if args.az_step < 0.01:
        p.error("--az-step must be >= 0.01 degrees (NEC prints angles with two decimals)")
    nsteps = 360.0 / args.az_step
    if abs(nsteps - round(nsteps)) > 1e-7:
        p.error("--az-step must divide 360 exactly; examples: 1, 0.5, 0.2, 0.1")
    for which in ("rear", "front"):
        if getattr(args, f"{which}_sections") and getattr(args, f"{which}_diameter") is not None:
            p.error(f"--{which}-sections already gives the diameters; do not combine it with --{which}-diameter")
    stepped = args.rear_sections or args.front_sections
    if args.model_feed_gap:
        if args.gap_mm <= 0:
            p.error("--model-feed-gap needs --feed-gap-mm greater than zero")
        lam_g = C_LIGHT / (args.freq * 1e6)
        for which in ("rear", "front"):
            probs = feed_gap_problems(args.gap_mm / 1000.0, element_sections(args, which)[0][1] / 2000.0, lam_g)
            if probs:
                p.error(f"{which} element: {probs[0]}")
        if args.feedline_length > 0 or args.cm_sensitivity:
            p.error("--model-feed-gap cannot be combined with --feedline-length-m / --cm-sensitivity (that wire "
                    "already splits the fed element at a one-segment gap)")
    if stepped and (args.feedline_length > 0 or args.cm_sensitivity):
        fed = args.front_sections if args.feed_element == "front" else args.rear_sections
        if fed and len(fed) > 1:
            p.error("stepped sections on the fed element cannot be combined with --feedline-length-m / --cm-sensitivity")
    if not 0.0 < args.coax_vf <= 1.0:
        p.error("--line-vf must be in (0, 1]")


LINE_FOLD_LIMIT = 1.5   # automatic longest line: this many times the straight run it spans (plus allowance)


def auto_max_line_deg(args, theory_deg, min_deg, offsets=(0.0,)):
    """--max-line-deg, or its automatic value for a TL: a line at most LINE_FOLD_LIMIT times the straight distance
    between its ends plus the allowance (min_deg is that distance in degrees), so that it can still be run almost
    straight. A theoretical line beyond 180 degrees (a crossed line fed at the rear element) is long by nature: the
    limit then covers it plus the largest scan offset. Always below 360."""
    if getattr(args, "max_line_deg", None):
        return args.max_line_deg
    limit = LINE_FOLD_LIMIT * min_deg
    if theory_deg > 180.0:
        limit = max(limit, theory_deg + max(offsets))
    return min(359.9, limit)


def choose_default_line(args):
    """generate without --line-deg/--line-length/--phase-target-deg: set args.line_deg to the synthesized line
    (--synthesize-line, needs nec2c) or to the theoretical one, never shorter than the straight distance between
    the feed points plus --line-allowance-mm (TL)."""
    lam = C_LIGHT / (args.freq * 1e6)
    theory = line_electrical_deg(theory_line_lag(args.feed_element, args.spacing_factor), args.crossed)
    if args.network == "tl":
        lo = line_min_elec_deg(feed_distance_m(args, args.spacing_factor * lam), lam, args.coax_vf,
                               args.line_allowance_mm / 1000.0)
        hi = auto_max_line_deg(args, theory, lo)
    else:
        lo, hi = 0.5, 179.5
    if args.synthesize_line:
        exe = find_nec2c(args)
        if exe:
            try:
                objective = None
                if (args.reflectors or 0) + (args.directors or 0) and args.ground == "free":
                    objective = {"objective": args.objective, "w_gain": 1.0, "w_fb": 0.5, "rear_sector": (150, 210),
                                 "fb_metric": "sector"}
                syn = synthesize_line(exe, _variant(args, line_deg=None), lo, hi, objective=objective)
                args.line_deg = max(round(syn["elec_deg"], 3), math.ceil(lo * 1000.0) / 1000.0)
                print(f"NOTE: phase line synthesized from the two-port response of the elements: {synthesis_text(syn)}.")
                if syn["error"] > 0.3:
                    print("NOTE: the cardioid target is far from reach with this line impedance and geometry"
                          + (" (with a centre feed a high-impedance line needs a long, folded line)"
                             if getattr(args, "feed_style", "center") == "center" and args.network == "tl" else "")
                          + "; --mode optimize searches lengths and line together"
                          + ("; --feed-style tmatch moves the line to taps on the elements"
                             if getattr(args, "feed_style", "center") == "center" else "") + ".")
                return
            except (ValueError, OSError) as exc:
                print(f"NOTE: line synthesis failed ({exc}); using the theoretical line.")
        else:
            print("NOTE: --synthesize-line needs nec2c (--nec2c, HB9CV_NEC2C or PATH); using the theoretical line.")
    elec = theory
    if args.network == "tl" and elec < lo:
        elec = math.ceil(lo * 10.0) / 10.0
        print(f"NOTE: the theoretical line ({theory:.1f} deg) is shorter than the distance between the feed points "
              f"plus {args.line_allowance_mm:g} mm ({lo:.1f} deg at VF {args.coax_vf:g}); the shortest buildable line, "
              f"{elec:.1f} deg, is used.")
    if args.network == "lc" and not 0.0 < elec < 180.0:
        return                       # lc_pi_values reports the out-of-range delay
    if elec != theory:
        args.line_deg = elec


def resolve_auto_series_c(p, args):
    """--series-c-pf auto (generate): the capacitor that cancels X_in at --freq, from one nec2c run of the model
    without it; none when X_in is not positive."""
    exe = find_nec2c(args)
    if not exe:
        p.error("--series-c-pf auto needs nec2c (--nec2c, HB9CV_NEC2C or PATH)")
    a = _variant(args, series_c_pf=None, fsweep=None, fsweep_pattern=False, band_freqs=None, pattern_mode="ring",
                 ring_theta=90.0, pt_mode=-1)
    with tempfile.TemporaryDirectory(prefix="hb9cv_sc_") as td:
        try:
            out = _run_deck(exe, Path(td), Path(td) / "sc.nec", build_deck(a)[0])
            z = single_source_zin(out)[0][1]
        except (ValueError, OSError) as exc:
            p.error(f"--series-c-pf auto: {exc}")
    c_pf = series_c_for_x(z.imag, args.freq)
    if c_pf is None:
        print(f"NOTE: --series-c-pf auto: X_in = {z.imag:+.2f} ohm at {args.freq:g} MHz is not inductive; no series "
              "capacitor is used.")
        args.series_c_pf = None
    else:
        args.series_c_pf = round(c_pf, 4)
        print(f"Series feed capacitor (auto): {args.series_c_pf:g} pF cancels X_in = {z.imag:+.2f} ohm at "
              f"{args.freq:g} MHz.")


def run_generate(p, args):
    check_common_model_args(p, args)
    if args.rear_length_factor < 0.5 or args.rear_length_factor > 1.5:
        p.error("--rear-length-factor must be between 0.5 and 1.5")
    if args.front_length_factor < 0.5 or args.front_length_factor > 1.5:
        p.error("--front-length-factor must be between 0.5 and 1.5")
    if args.spacing_factor > 0.5:
        p.error("--spacing-factor must be <= 0.5")
    if args.rear_ratio > 2:
        p.error("--rear-ratio must be <= 2")
    if args.fsweep is not None:
        try:
            fmin, fmax, fstep = [float(x) for x in args.fsweep.split(",")]
        except ValueError:
            p.error("--fsweep requires the format FMIN,FMAX,STEP (MHz)")
        if not all(math.isfinite(x) for x in (fmin, fmax, fstep)) or fmin <= 0 or fstep <= 0 or fmax < fmin:
            p.error("--fsweep requires 0 < FMIN <= FMAX and STEP > 0")
        args.fsweep = (fmin, fmax, fstep)
    single_feed = args.network in ("tl", "lc")
    if not single_feed and args.fsweep:
        print("WARNING: with two EX sources there is no single port; Zin(f) cannot be used to design an L-network.")
    if (args.lc_l_nh is None) != (args.lc_c_pf is None):
        p.error("--lc-l-nh and --lc-c-pf must be given together")
    if args.network != "lc" and args.lc_l_nh is not None:
        p.error("--lc-l-nh/--lc-c-pf are only used with --network lc")
    for name in ("phase_target_deg", "line_deg"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            p.error(f"--{name.replace('_', '-')} must be finite")
    if args.line_deg is not None and not 0.0 < args.line_deg < 360.0:
        p.error("--line-deg must be between 0 and 360 degrees (exclusive)")
    lam_main = C_LIGHT / (args.freq * 1e6)
    # With reflectors/directors the boom is lengthened to hold every element, so only a plain HB9CV can be short.
    if (args.boom_isolated and args.reflectors + args.directors == 0
            and args.boom_length / 1000.0 <= args.spacing_factor * lam_main):
        p.error(f"--boom-length ({args.boom_length:g} mm) must be greater than the spacing between elements "
                f"({args.spacing_factor * lam_main * 1000:.1f} mm)")

    if args.height_sweep is not None:
        try:
            a_, b_, c_ = args.height_sweep.split(",")
            args.height_sweep = (float(a_), float(b_), int(c_))
        except ValueError:
            p.error("--height-sweep requires MIN,MAX,N (meters, meters, count)")
        if args.ground == "free":
            p.error("--height-sweep needs --ground perfect, average or good")
        hmin, hmax, hn = args.height_sweep
        if not (0 < hmin <= hmax and 1 <= hn <= 101 and (hn > 1 or hmin == hmax)):
            p.error("--height-sweep requires 0 < MIN <= MAX and 1 <= N <= 101 (N = 1 only when MIN == MAX)")
    if args.fsweep_pattern and not args.fsweep:
        p.error("--fsweep-pattern needs --fsweep FMIN,FMAX,STEP")
    if args.fsweep_csv and not args.fsweep_pattern:
        print("WARNING: --fsweep-csv only applies with --fsweep-pattern; ignored.", file=sys.stderr)
    over_ground = args.ground != "free"
    pattern_sweep = bool(args.fsweep and args.fsweep_pattern)
    # Over ground the written deck holds the ELEVATION cut: the azimuth ring must be taken at the
    # take-off angle, which is only known after running it (see --run).
    args.pattern_mode = "elevation" if (over_ground and (not args.fsweep or pattern_sweep)) else "ring"
    args.ring_theta = 90.0
    args.band_freqs = sweep_frequencies(*args.fsweep) if pattern_sweep else None
    if pattern_sweep:
        args.pt_mode = -1    # one current table per frequency would make the .out huge
    explicit_line = (args.line_deg is not None or args.coax_length is not None or args.phase_target_deg is not None
                     or args.lc_l_nh is not None)
    if single_feed and not explicit_line:
        choose_default_line(args)
    if single_feed and args.series_c_pf == "auto":
        resolve_auto_series_c(p, args)
    output = Path(args.output or f"hb9cv_{args.freq:g}MHz_{args.network}_{args.mount}.nec")
    try:
        deck, lam, rear, front, spacing, line = build_deck(args)
    except ValueError as exc:
        p.error(str(exc))
    try:
        output.write_text(deck.encode("ascii", errors="replace").decode("ascii"), encoding="ascii")
    except OSError as exc:
        p.error(f"could not write {output}: {exc}")
    print(f"NEC generated: {output.resolve()}")
    print(f"Frequency: {args.freq:g} MHz; lambda={lam:.5f} m; angular step={args.az_step:g} degrees")
    print(f"Rear={rear*1000:.1f} mm; front={front*1000:.1f} mm; spacing={spacing*1000:.1f} mm")
    print(f"Length factors: rear={args.rear_length_factor:.4f}; front={args.front_length_factor:.4f}")
    for txt in parasitic_report(line.get("parasitics", []), line["boom_length_mm"]
                                if line.get("parasitics") else None):
        print(txt)
    rsec, fsec = element_sections(args, "rear"), element_sections(args, "front")
    print(f"Element diameters (from the center outward): rear {sections_text(rsec)}; front {sections_text(fsec)}")
    if args.gap_mm:
        print(f"Feed gap g = {args.gap_mm:g} mm: " + (
            "modeled (1-segment bridge wire at each element center carries EX/TL); the NEC length includes the gap."
            if args.model_feed_gap else
            "cut list only (the NEC element is continuous; use --model-feed-gap to model it)."))
    for w in line.get("wire_warnings", []) + line.get("line_warnings", []):
        print(f"WARNING: {w}")
    if over_ground:
        gtxt = "perfect" if args.ground == "perfect" else f"{args.ground} (eps_r={args.ground_eps:g}, sigma={args.ground_sigma:g} S/m)"
        print(f"Ground: {gtxt}; elements {args.orientation}; height {args.height_m:.3f} m "
              f"({args.height_m / lam:.3f} lambda) of the "
              f"{'lowest tip of the rear element or of the longest reflector' if args.orientation == 'vertical' else 'element centers'}")
        if not args.fsweep or args.fsweep_pattern:
            print("NOTE: over ground the deck contains the elevation cut (RP theta 0..90); with --run the azimuth "
                  "ring at the take-off angle is generated as <name>_azimuth.nec and summarized.")
    elif args.orientation == "vertical":
        print("Elements vertical (along Z); in free space the pattern equals the horizontal one.")
    print(f"Objective={args.objective}; network={args.network}; mount={args.mount}; "
          f"target Z={args.target_impedance:g} ohm")
    if args.network == "tl":
        kind_txt = "open-wire (balanced)" if args.line_type == "openwire" else "coaxial"
        print(f"Feed at the {args.feed_element} element; {kind_txt} line "
              f"{'crossed (Z0<0 in TL)' if args.crossed else 'straight'}; "
              f"target delay of the far element={line['line_target_lag_deg']:.1f} degrees")
        if args.line_type == "openwire":
            print(f"Open-wire line: wire d={args.line_wire_diameter:g} mm, spacing D={args.line_spacing:g} mm "
                  f"-> Z0={args.coax_z0:.1f} ohm (120*acosh(D/d)); VF of the line={args.coax_vf:g}")
        else:
            print(f"Coaxial line: Z0={args.coax_z0:g} ohm; VF of the cable={args.coax_vf:g}")
        print(f"Physical line length (with the line's own VF)={line['line_length_m']:.5f} m; "
              f"electrical length={line['line_electrical_deg']:.1f} degrees; "
              f"NEC TL card length (physical/VF)={line['line_length_m']/args.coax_vf:.6f} m")
        if args.line_type == "coax":
            print("WARNING: a coax phasing section needs a balun/choke at its element end; the NEC TL is a "
                  "balanced two-wire model and does not represent shield/common-mode currents, which change "
                  "the real phase and amplitude of the element currents.")
        print("WARNING: the TL is ideal and includes no losses or radiation; it represents a phase line "
              "between elements, not every detail of a real antenna.")
    elif args.network == "lc":
        print(f"Feed at the {args.feed_element} element; LC pi network {'crossed (Y12 inverted)' if args.crossed else 'straight'}; "
              f"target delay of the far element={line['line_target_lag_deg']:.1f} degrees")
        print(f"Pi network: series L={line['lc_l_nh']:.3f} nH; C at each port={line['lc_c_pf']:.3f} pF; "
              + (f"Zc={args.lc_z0:g} ohm; network phase={line['line_electrical_deg']:.1f} degrees"
                 if args.lc_l_nh is None else "values set by the user"))
        print("WARNING: ideal LC network (no losses or parasitics); the NT model does not include physical circuit coupling.")
    else:
        what = "element currents" if args.ideal_drive == "current" else "source voltages"
        print(f"Ideal rear phase={line['rear_phase']:.1f} degrees; amplitude={args.rear_ratio:g} (relation of the "
              f"{what}, --ideal-drive {args.ideal_drive})")
        print("WARNING: the two-source model is exploratory and does not represent a physical feed network.")
    if args.feedline_length > 0:
        print(f"Common-mode test wire: {args.feedline_length:g} m (open-ended outer conductor, "
              f"{args.feedline_diameter:g} mm) tied to the {args.feed_element} element through a source gap.")
    print("WARNING: " + BALUN_REQUIREMENT.format(f=args.freq))
    if args.network == "ideal" and args.balun_ratio != "1:1":
        print("WARNING: --balun-ratio only changes the target impedance; with --network ideal there is no single port.")
    elif args.balun_ratio == "4:1":
        print("NOTE: " + BALUN_4_1_NOTE.format(zb=args.target_impedance, zu=50.0))
    if args.cm_sensitivity:
        exe_cm = find_nec2c(args)
        if not exe_cm:
            p.error("--cm-sensitivity requested, but nec2c was not found (--nec2c, HB9CV_NEC2C or PATH)")
        test_len = args.feedline_length if args.feedline_length > 0 else lam / 2.0
        try:
            cm_sensitivity(exe_cm, args, test_len)
        except (ValueError, OSError) as exc:
            print(f"WARNING: common-mode sensitivity run failed: {exc}", file=sys.stderr)
    if args.mast_sensitivity:
        exe_mast = find_nec2c(args)
        if not exe_mast:
            p.error("--mast-sensitivity requested, but nec2c was not found (--nec2c, HB9CV_NEC2C or PATH)")
        try:
            mast_sensitivity(exe_mast, args, args.mast_sensitivity)
        except (ValueError, OSError) as exc:
            print(f"WARNING: mast sensitivity run failed: {exc}", file=sys.stderr)
    perf = None
    if args.run:
        exe = find_nec2c(args)
        if not exe:
            p.error("--run requested, but nec2c was not found (--nec2c, HB9CV_NEC2C or PATH)")
        output_file = output.with_suffix(".out").resolve()
        run_nec_cli(exe, output.resolve(), output_file)
        write_freq_sidecar(output_file, args.band_freqs or (sweep_frequencies(*args.fsweep) if args.fsweep
                                                            else [args.freq]))
        if not args.fsweep:
            row = generate_pattern_report(args, exe, output, output_file)
            if row is not None:
                perf = {"design": row, "band": [row], "mc": None}
        elif pattern_sweep:
            rows = fsweep_pattern_report(args, exe, output)
            if rows:
                near = min(rows, key=lambda r: abs(r["freq_mhz"] - args.freq))
                perf = {"design": near if abs(near["freq_mhz"] - args.freq) < 1e-6 else None,
                        "band": rows, "mc": None}
        elif single_feed:
            try:
                perf = {"design": None, "mc": None, "band": [
                    {"freq_mhz": fr, "zin": z, "vswr": lnet_metrics(z, args.target_impedance)[0]}
                    for fr, z in single_source_zin(output_file)]}
                print_bandwidths(perf["band"])
            except (ValueError, OSError) as exc:
                print(f"WARNING: could not read Zin(f) for the build sheet: {exc}", file=sys.stderr)
    if args.run and perf is not None and not (args.fsweep and not args.fsweep_pattern):
        loss = conductor_loss_check(exe, args)
        if loss is not None:
            perf["loss"] = loss
        agt = average_gain_test(exe, args)
        if agt is not None:
            perf["agt"] = agt
    extra = {}
    if args.check_convergence or args.robustness or args.height_sweep:
        exe_x = find_nec2c(args)
        if not exe_x:
            p.error("--check-convergence / --robustness / --height-sweep need nec2c (--nec2c, HB9CV_NEC2C or PATH)")
        a_x = _variant(args, fsweep=None, fsweep_pattern=False, band_freqs=None)
        if args.check_convergence:
            extra["convergence"] = convergence_check(exe_x, a_x)
        if args.robustness:
            extra["robustness"] = robustness_check(exe_x, a_x)
        if args.height_sweep:
            extra["height_sweep"] = height_sweep(exe_x, a_x, args.height_sweep)
    if args.build_sheet:
        zin_bs = None
        if perf and perf.get("design") and perf["design"].get("zin") is not None:
            zin_bs = perf["design"]["zin"]
        elif perf:
            near = [x for x in perf.get("band", []) if x.get("zin") is not None and abs(x["freq_mhz"] - args.freq) < 1e-6]
            zin_bs = near[0]["zin"] if near else None
        write_build_sheet(args.build_sheet, args,
                          {"rear_mm": rear * 1000, "front_mm": front * 1000, "spacing_mm": spacing * 1000,
                           "parasitics": line.get("parasitics", []), "boom_length_mm": line["boom_length_mm"],
                           "boom_xc_m": line["boom_xc_m"]},
                          describe_line(args, line["line_electrical_deg"], line["line_length_m"], args.coax_z0,
                                        args.line_spacing if args.network == "tl" and args.line_type == "openwire"
                                        else None, line.get("lc_l_nh"), line.get("lc_c_pf")),
                          zin_bs, perf,
                          {"elec_deg": line["line_electrical_deg"], "phys_m": line["line_length_m"],
                           "z0": args.coax_z0,
                           "spacing_mm": args.line_spacing if args.network == "tl" and args.line_type == "openwire" else None,
                           "lc_l_nh": line.get("lc_l_nh"), "lc_c_pf": line.get("lc_c_pf")})
    write_json_out(args, dict(extra, deck=str(output.resolve()),
                              out=str(output.with_suffix(".out").resolve()) if args.run else None,
                              freq_mhz=args.freq, lambda_m=lam, rear_mm=rear * 1000, front_mm=front * 1000,
                              spacing_mm=spacing * 1000, line=line, performance=perf,
                              warnings=line.get("wire_warnings", []) + line.get("line_warnings", [])))


def height_sweep(exe, gargs, spec):
    """--height-sweep MIN,MAX,N (m): take-off angle, gain, F/B and Zin of the model at N heights."""
    hmin, hmax, n = spec
    heights = [hmin + (hmax - hmin) * i / (n - 1) for i in range(n)] if n > 1 else [hmin]
    print(f"\nHEIGHT SWEEP at {gargs.freq:g} MHz ({gargs.ground} ground, elements {gargs.orientation})")
    print(f"{'height m':>9s}{'h/lambda':>9s}{'takeoff':>9s}{'gain dBi':>10s}{'F/B sect':>10s}{'F/B cone':>10s}"
          f"{'Zin (ohm)':>22s}")
    lam = C_LIGHT / (gargs.freq * 1e6)
    out, seen = [], set()
    for h in heights:
        try:
            m = _single_run(exe, gargs, [gargs.freq], height_m=h)[0]
        except (ValueError, OSError) as exc:
            print(f"{h:9.3f}  run failed: {exc}")
            continue
        zt = f"{m['zin'].real:.2f}{m['zin'].imag:+.2f}j" if m["zin"] is not None else ""
        print(f"{h:9.3f}{h / lam:9.3f}{m['takeoff_deg']:9.1f}{m['forward_gain']:10.2f}{m['robust_fb']:10.2f}"
              f"{m['cone_fb']:10.2f}{zt:>22s}")
        print_model_warnings(f"height {h:g} m", m.get("model_warnings"), seen)
        out.append({"height_m": h, "takeoff_deg": m["takeoff_deg"], "gain_dBi": m["forward_gain"],
                    "fb_sector_dB": m["robust_fb"], "fb_cone_dB": m["cone_fb"], "zin": m["zin"]})
    return out


def run_nec_cli(exe, input_file, output_file):
    """Run nec2c on a deck, echoing its messages; exits if it fails."""
    print("Running:", " ".join([exe, "-i", str(input_file), "-o", str(output_file)]))
    result = run_nec2c(exe, input_file, output_file)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    if not output_file.exists():
        raise SystemExit(f"NEC2C finished without error but did not create the expected file: {output_file}")
    print(f"Simulation finished. Result: {output_file}")


SQUINT_WARN_DEG = 3.0   # warn when max-gain direction deviates this far from phi=0


def print_pattern_summary(m, zin=None, target=None, ground="free"):
    """Gain (dBi; dBd in free space only), take-off angle, azimuth half-power beamwidth and F/B of a ring.

    The gain is the NEC power gain referred to the power accepted at the input: conductor loss is included,
    source mismatch is not (the realized gain adds it)."""
    takeoff = 90.0 - m["theta_deg"]
    print("PATTERN SUMMARY")
    gtxt = (f"{m['forward_gain']:.2f} dBi = {m['gain_dbd']:.2f} dBd" if m.get("gain_dbd") is not None else
            f"{m['forward_gain']:.2f} dBi (including the ground reflection; no dBd over ground)")
    print(f"  gain at phi=0: {gtxt}; maximum {m['max_gain']:.2f} dBi at phi={m['max_angle']:g}")
    # Squint: the beam direction deviates from the intended +x axis (phi=0).
    squint = min(abs(m["max_angle"]), abs(m["max_angle"] - 360.0))
    if squint > SQUINT_WARN_DEG:
        print(f"  WARNING: beam squints {squint:.1f} deg from phi=0 (max gain at phi={m['max_angle']:g}); "
              f"forward gain is {m['max_gain'] - m['forward_gain']:.2f} dB below the beam peak. "
              "Check element symmetry, phasing-line asymmetry or ground effects.")
    print(f"  take-off angle: {takeoff:g} deg above the horizon (ring at theta={m['theta_deg']:g})")
    print(f"  F/B at the take-off angle: {m['fb']:.2f} dB (phi=180); rear sector worst case: {m['robust_fb']:.2f} dB "
          f"at phi={m['worst_rear_angle']:g}")
    if m.get("xpd_db") is not None:
        print("  cross-polar discrimination at phi=0: "
              + (f"{m['xpd_db']:.1f} dB" if math.isfinite(m["xpd_db"]) else "no cross-polar component (null)"))
    if m.get("cone_points"):
        print(f"  F/B rear cone (any elevation within the sector half-angle of the back direction): "
              f"{m['cone_fb']:.2f} dB, worst at theta={m['cone_worst_theta']:g}, phi={m['cone_worst_phi']:g}")
    print(f"  half-power beamwidth in azimuth: {m['hpbw']:.1f} deg")
    if ground != "free" and takeoff > 60:
        print("  WARNING: the pattern peaks above 60 deg elevation (NVIS-like); F/B at that angle says little "
              "about the horizon performance. Check --height-m/--height-lambda.")
    if zin is not None:
        line = f"  Zin = {zin.real:.2f}{zin.imag:+.2f}j ohm"
        if target:
            vswr, _, rho = lnet_metrics(zin, target)
            line += (f"; VSWR vs {target:g} ohm = {vswr:.2f}:1; realized gain "
                     f"{m['forward_gain'] + 10 * math.log10(max(1 - rho ** 2, 1e-12)):.2f} dBi")
        print(line)


FSWEEP_CSV_FIELDS = ["freq_mhz", "gain_dBi", "gain_dBd", "FB_point_dB", "FB_robust_dB", "FB_cone_dB", "takeoff_deg",
                     "hpbw_deg",
                     "Zin_real", "Zin_imag", "VSWR_vs_target", "realized_gain_dB"]


def fsweep_pattern_report(args, exe, output):
    """--fsweep --fsweep-pattern --run: gain, F/B and Zin at every sweep frequency. Reuses the optimizer's
    ground-aware simulator (free space: ring at theta=90; over ground: elevation cut -> take-off angle per
    frequency -> azimuth ring). Prints a table and writes the CSV; returns the per-frequency rows or None."""
    freqs = list(args.band_freqs)
    try:
        res = simulate_model(exe, output.resolve().parent, output.stem + "_fsp", args, freqs)
    except (ValueError, OSError) as exc:
        res = {"ok": False, "stage": "internal", "reason": str(exc)}
    if not res["ok"]:
        print(f"WARNING: could not compute the pattern sweep ({res['stage']}): {res['reason']}", file=sys.stderr)
        return None
    rows = res["per_freq"]
    csv_path = Path(args.fsweep_csv) if args.fsweep_csv else output.with_name(output.stem + "_fsweep.csv")
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FSWEEP_CSV_FIELDS)
            w.writeheader()
            for r in rows:
                z = r["zin"]
                w.writerow({"freq_mhz": f"{r['freq_mhz']:.6g}", "gain_dBi": r["forward_gain"],
                            "gain_dBd": r["gain_dbd"] if r["gain_dbd"] is not None else "",
                            "FB_point_dB": r["fb"], "FB_robust_dB": r["robust_fb"], "FB_cone_dB": r["cone_fb"],
                            "takeoff_deg": r["takeoff_deg"], "hpbw_deg": r["hpbw"],
                            "Zin_real": z.real if z is not None else "", "Zin_imag": z.imag if z is not None else "",
                            "VSWR_vs_target": r["vswr"] if r["vswr"] is not None else "",
                            "realized_gain_dB": r["realized"] if r["realized"] is not None else ""})
        print(f"Pattern sweep CSV: {csv_path.resolve()}")
    except OSError as exc:
        print(f"WARNING: could not write {csv_path}: {exc}", file=sys.stderr)
    print("PATTERN SWEEP (geometry and phase line fixed at the design frequency)")
    print(f"{'f (MHz)':>9s}{'gain dBi':>10s}{'F/B':>8s}{'F/B sect':>10s}{'F/B cone':>10s}{'Zin (ohm)':>22s}"
          f"{'VSWR':>8s}")
    for r in rows:
        zt = f"{r['zin'].real:.2f}{r['zin'].imag:+.2f}j" if r["zin"] is not None else ""
        vt = f"{r['vswr']:.2f}" if r["vswr"] is not None else ""
        print(f"{r['freq_mhz']:9.3f}{r['forward_gain']:10.2f}{r['fb']:8.2f}{r['robust_fb']:10.2f}{r['cone_fb']:10.2f}"
              f"{zt:>22s}{vt:>8s}")
    print_bandwidths(rows)
    return rows


def contiguous_vswr_span(pairs, limit=2.0):
    """Contiguous band around the minimum-VSWR point where VSWR <= limit, from [(f_mhz, vswr)].

    Returns (f_low, f_high, reaches_sweep_edge) or None when no point meets the limit. Points outside the band that
    contains the minimum are not part of it, even when they meet the limit."""
    pts = sorted(pairs, key=lambda t: t[0])
    if not pts:
        return None
    best = min(range(len(pts)), key=lambda i: pts[i][1])
    if not pts[best][1] <= limit:
        return None
    lo = hi = best
    while lo > 0 and pts[lo - 1][1] <= limit:
        lo -= 1
    while hi < len(pts) - 1 and pts[hi + 1][1] <= limit:
        hi += 1
    return pts[lo][0], pts[hi][0], lo == 0 or hi == len(pts) - 1


def print_bandwidths(rows, drop_db=1.0):
    """2:1 VSWR bandwidth and the band where the gain stays within drop_db of its maximum, from the
    per-frequency rows of a pattern sweep (contiguous run around the best point)."""
    if len(rows) < 2:
        return
    rows = sorted(rows, key=lambda r: r["freq_mhz"])

    def span(ok, best):
        lo = hi = best
        while lo > 0 and ok(rows[lo - 1]):
            lo -= 1
        while hi < len(rows) - 1 and ok(rows[hi + 1]):
            hi += 1
        return rows[lo]["freq_mhz"], rows[hi]["freq_mhz"], lo == 0 or hi == len(rows) - 1

    if all(r.get("vswr") is not None for r in rows):
        best = min(range(len(rows)), key=lambda i: rows[i]["vswr"])
        if rows[best]["vswr"] <= 2.0:
            a, b, edge = span(lambda r: r["vswr"] <= 2.0, best)
            print(f"VSWR <= 2:1 bandwidth: {a:.3f}-{b:.3f} MHz ({b - a:.3f} MHz)"
                  + (" (reaches the end of the sweep: widen it)" if edge else ""))
        else:
            print(f"VSWR never reaches 2:1 in the sweep (minimum {rows[best]['vswr']:.2f}:1 at "
                  f"{rows[best]['freq_mhz']:.3f} MHz)")
    if any(r.get("forward_gain") is None for r in rows):
        return
    gbest = max(range(len(rows)), key=lambda i: rows[i]["forward_gain"])
    gmax = rows[gbest]["forward_gain"]
    a, b, edge = span(lambda r: r["forward_gain"] >= gmax - drop_db, gbest)
    print(f"Gain within {drop_db:g} dB of its maximum ({gmax:.2f} dBi): {a:.3f}-{b:.3f} MHz"
          + (" (reaches the end of the sweep)" if edge else ""))


def generate_pattern_report(args, exe, output, output_file):
    """After --run: summarize the pattern; over ground first find the take-off angle in the elevation
    cut, then write and run the azimuth-ring deck at that angle."""
    try:
        theta = None
        if args.ground != "free":
            sections = parse_pattern_sections(output_file.read_text(errors="replace"))
            if not sections:
                raise ValueError("RADIATION PATTERNS table not found in the elevation run")
            theta = elevation_peak_theta(sections[0])
            a = Namespace(**vars(args))
            a.pattern_mode, a.ring_theta = "ring", theta
            ring_deck = output.with_name(output.stem + "_azimuth.nec").resolve()
            ring_deck.write_text(build_deck(a)[0].encode("ascii", errors="replace").decode("ascii"), encoding="ascii")
            print(f"Take-off angle {90.0 - theta:g} deg above the horizon; azimuth ring deck: {ring_deck}")
            output_file = ring_deck.with_suffix(".out")
            run_nec_cli(exe, ring_deck, output_file)
            write_freq_sidecar(output_file, [args.freq])
        m = parse_metrics(output_file, theta=theta, ground_free=args.ground == "free")
        zin = single_source_zin(output_file)[0][1] if args.network in ("tl", "lc") else None
        print_pattern_summary(m, zin, args.target_impedance if zin is not None else None, args.ground)
        try:
            print_element_currents(output_file, ports=build_deck(args)[5].get("ports"))
        except (ValueError, OSError) as exc:
            print(f"  NOTE: element-current report not available: {exc}", file=sys.stderr)
        return perf_row_from_metrics(m, zin, args.target_impedance if zin is not None else None, args.freq)
    except (ValueError, OSError) as exc:
        print(f"WARNING: could not summarize the pattern: {exc}", file=sys.stderr)
    return None


# ======================================================================================
# Optimizer
# ======================================================================================

def parse_values(raw, name, low=None, high=None):
    try:
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name}: invalid numeric list") from exc
    if not values or any(not math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError(f"{name}: empty or non-finite list")
    if low is not None and any(x < low for x in values):
        raise argparse.ArgumentTypeError(f"{name}: value below {low}")
    if high is not None and any(x > high for x in values):
        raise argparse.ArgumentTypeError(f"{name}: value above {high}")
    return values


def parse_band(text):
    """--band FMIN,FMAX,N -> list of N frequencies (MHz), evenly spaced and inclusive."""
    try:
        a, b, c = text.split(",")
        fmin, fmax, n = float(a), float(b), int(c)
    except ValueError:
        raise argparse.ArgumentTypeError("--band requires FMIN,FMAX,N (for example 144,146,5)") from None
    if not (math.isfinite(fmin) and math.isfinite(fmax)) or fmin <= 0 or fmax < fmin:
        raise argparse.ArgumentTypeError("--band requires 0 < FMIN <= FMAX")
    if fmin == fmax:
        if n != 1:
            raise argparse.ArgumentTypeError("--band with FMIN == FMAX requires N = 1")
        return [fmin]
    if not 2 <= n <= 101:
        raise argparse.ArgumentTypeError("--band requires 2 <= N <= 101 when FMIN < FMAX")
    return [fmin + (fmax - fmin) * i / (n - 1) for i in range(n)]


def percentile(values, q):
    """q-th percentile (0..100) with linear interpolation; nan for an empty list."""
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def _variant(gargs, **kw):
    a = Namespace(**vars(gargs))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _run_deck(exe, td, deck, deck_text):
    """Write a deck, run nec2c and return the .out path (ValueError if it fails)."""
    deck.write_text(deck_text.encode("ascii", errors="replace").decode("ascii"), encoding="ascii")
    out_expected = deck.with_suffix(deck.suffix + ".out")
    run = run_nec2c(exe, deck, out_expected)
    out = next((q for q in (out_expected, deck.with_suffix(".out")) if q.exists()), None)
    if run.returncode or out is None:
        raise OSError("NEC2C failed")
    return out


def _read_checked(out):
    """Text of a .out file; ValueError if nec2c reported a fatal problem. Returns (text, warnings)."""
    text = out.read_text(errors="replace")
    fatal, warns = scan_nec_messages(text)
    if fatal:
        raise ValueError(f"NEC2C reported a problem: {fatal[0]}")
    return text, warns


def simulate_model(exe, td, name, gargs, freqs, rear_start=150, rear_end=210, pt_mode=-1):
    """Run nec2c for the model gargs at every frequency in freqs; ground-aware.

    Free space: one deck with an azimuth ring at theta=90 per frequency.
    Over ground: pass 1 = elevation cut (phi=0, theta 0..90) per frequency -> take-off theta of
    each frequency; pass 2 = azimuth ring at that theta. Every ring also has the rear half-sphere
    grid of the rear-cone F/B. The LC network gets the NT admittances of each frequency in its own
    frequency group (build_deck).

    Returns {"ok": True, "per_freq": [...], "warnings": [...], "deck_path", "out_path"} or
    {"ok": False, "stage": ..., "reason": ...}. Safe to call from worker processes.
    """
    td = Path(td)
    over_ground = getattr(gargs, "ground", "free") != "free"
    single = gargs.network in ("tl", "lc")
    chunks = [list(freqs)]
    per_freq, warnings, deck_path, out_path, model_warnings = [], [], None, None, []
    stage = "generator"
    try:
        for ci, chunk in enumerate(chunks):
            common = dict(band_freqs=chunk, pt_mode=pt_mode, fsweep=None)
            thetas = [90.0] * len(chunk)
            if over_ground:
                stage = "generator"
                deck_text = build_deck(_variant(gargs, pattern_mode="elevation", **common))[0]
                stage = "nec2c"
                out = _run_deck(exe, td, td / f"{name}_c{ci}_el.nec", deck_text)
                stage = "parse"
                text, w = _read_checked(out)
                warnings += w
                sections = parse_pattern_sections(text)
                if len(sections) != len(chunk):
                    raise ValueError(f"elevation cut: expected {len(chunk)} pattern sections, found {len(sections)}")
                thetas = [elevation_peak_theta(sec) for sec in sections]
            stage = "generator"
            deck_text, *_rest, deck_info = build_deck(_variant(gargs, pattern_mode="ring", ring_theta=thetas, **common))
            model_warnings = deck_info.get("wire_warnings", []) + deck_info.get("line_warnings", [])
            stage = "nec2c"
            deck_path = td / f"{name}_c{ci}.nec"
            out_path = _run_deck(exe, td, deck_path, deck_text)
            stage = "parse"
            text, w = _read_checked(out_path)
            warnings += w
            sections = parse_pattern_sections(text)
            if not sections:
                raise ValueError("RADIATION PATTERNS table not found")
            if len(sections) != len(chunk):
                raise ValueError(f"expected {len(chunk)} pattern sections, found {len(sections)}")
            validate_input_table(out_path)
            zins = zin_in_order(out_path, len(chunk)) if single else [None] * len(chunk)
            c_auto = None
            if single and getattr(gargs, "series_c_pf", None) == "auto":
                # The capacitor that cancels X_in at the design frequency, added to the Zin of every frequency.
                k0 = min(range(len(chunk)), key=lambda i: abs(chunk[i] - gargs.freq))
                c_auto = series_c_for_x(zins[k0].imag, chunk[k0])
                if c_auto:
                    zins = [z + series_c_impedance(c_auto, fr, series_c_q(gargs)) for z, fr in zip(zins, chunk)]
            for fr, th, sec, zin in zip(chunk, thetas, sections, zins):
                points, n_exp = ring_from_section(sec, th)
                m = add_cone_metrics(ring_metrics(points, n_exp, th, rear_start, rear_end,
                                                  ground_free=not over_ground), sec, rear_start, rear_end)
                m.update(freq_mhz=fr, takeoff_deg=90.0 - th, zin=zin, vswr=None, realized=None,
                         series_c_pf=c_auto if c_auto else series_c_value(gargs))
                if zin is not None:
                    g = (zin - gargs.target_impedance) / (zin + gargs.target_impedance)
                    m["vswr"] = (1 + abs(g)) / (1 - abs(g)) if abs(g) < 1 else float("inf")
                    m["realized"] = m["forward_gain"] + 10 * math.log10(max(1 - abs(g) ** 2, 1e-12))
                per_freq.append(m)
    except (ValueError, OSError) as exc:
        return {"ok": False, "stage": stage, "reason": str(exc)}
    return {"ok": True, "per_freq": per_freq, "warnings": warnings, "model_warnings": model_warnings,
            "deck_path": str(deck_path), "out_path": str(out_path)}


def _sim_worker(job):
    """ProcessPoolExecutor entry point (module level so it can be pickled)."""
    try:
        return simulate_model(**job)
    except Exception as exc:   # a worker must never take the pool down
        return {"ok": False, "stage": "internal", "reason": f"{type(exc).__name__}: {exc}"}


@contextlib.contextmanager
def make_pool(n_jobs):
    if n_jobs <= 1:
        yield None
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            yield pool


def run_jobs(pool, job_list):
    """Results in submission order (lazily, so progress can be printed while others still run)."""
    if pool is None:
        return (_sim_worker(j) for j in job_list)
    return pool.map(_sim_worker, job_list, chunksize=1)


# ======================================================================================
# Phase-line synthesis from the two-port response of the driven elements
#
# With the network removed, a 1 V source on one feed port (the other one shorted by its continuous wire) gives
# the short-circuit admittances Y[a][b] (current at port a for 1 V at port b) and the element-current responses
# M[a][b] (current at the center of element a). A line of characteristic impedance z0 and electrical length t
# between the ports, crossed (s = -1) or straight (s = +1), with the fed port at 1 V, sets the far-port voltage
#   Vd = (s - j z0 sin t Y[d][f]) / (cos t + j z0 sin t Y[d][d])
# so the element currents are I_e = M[e][f] + M[e][d] Vd and the input admittance is
#   Y[f][f] + Y[f][d] Vd + s (j sin t / z0 Vd + cos t (Y[d][f] + Y[d][d] Vd)).
# An ideal LC pi network of characteristic impedance Zc and phase t behaves as that line at its design
# frequency. The synthesis picks t (z0 fixed by the line) so that the far/fed element-current ratio is the
# cardioid target: equal amplitudes, rear leading front by 180 - kd.
# ======================================================================================

def two_port_response(exe, gargs, td):
    """Y and M of the two feed ports of the model gargs at gargs.freq (see the section comment); ValueError or
    OSError when a probe run fails."""
    out = {}
    td = Path(td)
    for p in ("front", "rear"):
        a = _variant(gargs, probe_port=p, fsweep=None, fsweep_pattern=False, band_freqs=None, pattern_mode="ring",
                     ring_theta=90.0, pt_mode=0, rear_cone_rp=False, series_c_pf=None)
        deck_text, *_, info = build_deck(a)
        out_path = _run_deck(exe, td, td / f"probe_{p}.nec", deck_text)
        _read_checked(out_path)
        rows = parse_element_currents(out_path)
        for q in ("front", "rear"):
            out[("Y", q, p)] = segment_current(rows, *info["feed_ports"][q])
            out[("M", q, p)] = segment_current(rows, *info["ports"][q])
        out[("F", p)] = parse_ring_fields(Path(out_path).read_text(errors="replace"), 90.0)
    return out


def parse_ring_fields(text, theta):
    """{phi: (total gain dB, E_theta, E_phi)} of the pattern points at zenith angle theta of the first frequency
    block, with the complex far-field components (the fields of different runs at the same frequency superpose)."""
    out = {}
    blocks = nec_blocks(text)
    lines = blocks[1][1] if len(blocks) > 1 else blocks[0][1]
    on = False
    for line in lines:
        if "RADIATION PATTERNS" in line:
            on = True
            continue
        if not on:
            continue
        m = PATTERN_FIELD_ROW.match(line)
        if not m:
            continue
        th, ph, _v, _h, tot, eth_m, eth_p, eph_m, eph_p = (float(x) for x in m.groups())
        if abs(th - theta) <= 0.006:
            out[round(ph % 360.0, 5)] = (tot, cmath.rect(eth_m, math.radians(eth_p)),
                                         cmath.rect(eph_m, math.radians(eph_p)))
    return out


def line_pattern_metrics(resp, z0, elec_deg, crossed, feed, rear_sector):
    """Forward gain (dBi, phi = 0) and rear-sector F/B (dB) on the theta = 90 ring of the antenna fed through the
    line z0 / elec_deg, from the superposition of the two probe patterns (fed port at 1 V, far port at V_d) and the
    input power at the fed port; the gain is calibrated on the fed probe's own printed gain. None when the probe
    patterns lack the field columns."""
    f, d = (feed, "rear" if feed == "front" else "front")
    ef, ed = resp.get(("F", f)), resp.get(("F", d))
    if not ef or not ed:
        return None
    s = -1.0 if crossed else 1.0
    t = math.radians(elec_deg)
    den = math.cos(t) + 1j * z0 * math.sin(t) * resp[("Y", d, d)]
    if abs(den) < 1e-15:
        return None
    vd = (s - 1j * z0 * math.sin(t) * resp[("Y", d, f)]) / den
    zin = line_response(resp, z0, elec_deg, crossed, feed)[1]
    if not cmath.isfinite(zin) or zin == 0:
        return None
    p_in = 0.5 * (1.0 / zin).real
    p_f = 0.5 * resp[("Y", f, f)].real
    ref = max((ph for ph in ef if ph in ed), key=lambda ph: ef[ph][0], default=None)
    if ref is None or p_in <= 0 or p_f <= 0:
        return None
    e2_ref = abs(ef[ref][1]) ** 2 + abs(ef[ref][2]) ** 2
    if e2_ref <= 0:
        return None
    k = 10.0 ** (ef[ref][0] / 10.0) * p_f / e2_ref

    def gain(ph):
        eth = ef[ph][1] + vd * ed[ph][1]
        eph = ef[ph][2] + vd * ed[ph][2]
        g = k * (abs(eth) ** 2 + abs(eph) ** 2) / p_in
        return 10.0 * math.log10(g) if g > 1e-30 else -999.0

    phis = [ph for ph in ef if ph in ed]
    if 0.0 not in phis:
        return None
    rear = [gain(ph) for ph in phis if rear_sector[0] <= ph <= rear_sector[1]]
    if not rear:
        return None
    fwd = gain(0.0)
    return {"forward_gain": fwd, "fb_sector": fwd - max(rear), "fb_point": fwd - gain(180.0) if 180.0 in phis else None}


def line_response(resp, z0, elec_deg, crossed, feed="front"):
    """(far/fed element-current ratio, input impedance at the fed port) of a line z0 / elec_deg on resp."""
    f, d = (feed, "rear" if feed == "front" else "front")
    s = -1.0 if crossed else 1.0
    t = math.radians(elec_deg)

    def Y(a, b):
        return resp[("Y", a, b)]

    def M(a, b):
        return resp[("M", a, b)]

    den = math.cos(t) + 1j * z0 * math.sin(t) * Y(d, d)
    if abs(den) < 1e-15:
        return complex("nan"), complex("nan")
    vd = (s - 1j * z0 * math.sin(t) * Y(d, f)) / den
    i_fed = M(f, f) + M(f, d) * vd
    ratio = (M(d, f) + M(d, d) * vd) / i_fed if abs(i_fed) > 1e-30 else complex("nan")
    i_line = s * (1j * math.sin(t) / z0 * vd + math.cos(t) * (Y(d, f) + Y(d, d) * vd))
    y_in = Y(f, f) + Y(f, d) * vd + i_line
    return ratio, (1.0 / y_in if abs(y_in) > 1e-30 else complex("inf"))


def cardioid_target(feed_element, spacing_factor):
    """Far/fed element-current ratio of the ideal cardioid: rear leads front by 180 - kd, equal amplitudes."""
    lead = math.radians(180.0 - 360.0 * spacing_factor)
    return cmath.exp(1j * lead) if feed_element == "front" else cmath.exp(-1j * lead)


def exact_line_solution(resp, target, crossed, feed="front"):
    """Line (elec_deg, z0) that gives the target ratio exactly when z0 is free, or None. Closed form: the far port
    must sit at r = (T M[f][f] - M[d][f]) / (M[d][d] - T M[f][d]) and the line must satisfy
    s = r cos t + j z0 K sin t with K = Y[d][f] + Y[d][d] r."""
    f, d = (feed, "rear" if feed == "front" else "front")
    s = -1.0 if crossed else 1.0
    den = resp[("M", d, d)] - target * resp[("M", f, d)]
    if abs(den) < 1e-30:
        return None
    r = (target * resp[("M", f, f)] - resp[("M", d, f)]) / den
    k = resp[("Y", d, f)] + resp[("Y", d, d)] * r
    q = r.real * k.real + r.imag * k.imag
    if abs(q) < 1e-30 or abs(k.real) < 1e-30:
        return None
    ct = s * k.real / q
    if abs(ct) > 1.0:
        return None
    for t in (math.acos(ct), 2.0 * math.pi - math.acos(ct)):
        if abs(math.sin(t)) < 1e-9:
            continue
        z0 = -r.imag * math.cos(t) / (k.real * math.sin(t))
        if z0 > 0:
            return math.degrees(t), z0
    return None


def synthesize_line(exe, gargs, elec_min, elec_max, td=None, objective=None):
    """Electrical length in [elec_min, elec_max] for the line/network impedance of gargs.

    Without objective: the line whose element-current ratio is closest to the two-element cardioid target.
    With objective (dict: objective, w_gain, w_fb, rear_sector; used with reflectors/directors, where the
    two-element target does not hold): the line that maximizes that objective on the theta = 90 ring, computed from
    the superposed probe patterns (line_pattern_metrics); the cardioid target is the fallback when the patterns are
    not usable. Returns a dict: elec_deg, ratio, zin, error (|ratio - target|), target, z0 (the impedance used),
    exact (the free-z0 solution or None) and, with an objective, pattern (the metrics at the chosen line)."""
    z0 = abs(gargs.coax_z0) if gargs.network == "tl" else gargs.lc_z0
    target = cardioid_target(gargs.feed_element, gargs.spacing_factor)
    with contextlib.ExitStack() as stack:
        tdir = td or stack.enter_context(tempfile.TemporaryDirectory(prefix="hb9cv_syn_"))
        resp = two_port_response(exe, gargs, tdir)

    def cardioid_err(th):
        ratio = line_response(resp, z0, th, gargs.crossed, gargs.feed_element)[0]
        return abs(ratio - target) if cmath.isfinite(ratio) else math.inf

    def objective_err(th):
        m = line_pattern_metrics(resp, z0, th, gargs.crossed, gargs.feed_element, objective["rear_sector"])
        if m is None:
            return math.inf
        fb = m["fb_point"] if objective.get("fb_metric") == "point" and m["fb_point"] is not None else m["fb_sector"]
        if objective["objective"] == "gain":
            return -m["forward_gain"]
        if objective["objective"] == "balance":
            return -(objective["w_gain"] * m["forward_gain"] + objective["w_fb"] * fb)
        return -fb

    err = cardioid_err
    if objective and line_pattern_metrics(resp, z0, 0.5 * (elec_min + elec_max), gargs.crossed, gargs.feed_element,
                                          objective["rear_sector"]) is not None:
        err = objective_err

    lo, hi = max(elec_min, 0.05), min(elec_max, 359.95)
    if hi <= lo:
        raise ValueError(f"no line length between {elec_min:.1f} and {elec_max:.1f} degrees")
    n = max(2, int(math.ceil((hi - lo) / 0.5)))
    grid = [lo + (hi - lo) * i / n for i in range(n + 1)]
    best = min(grid, key=err)
    a, b = max(lo, best - (hi - lo) / n), min(hi, best + (hi - lo) / n)
    g = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):                          # golden-section polish inside the best grid cell
        c1, c2 = b - g * (b - a), a + g * (b - a)
        if err(c1) < err(c2):
            b = c2
        else:
            a = c1
    th = min((best, 0.5 * (a + b)), key=err)
    ratio, zin = line_response(resp, z0, th, gargs.crossed, gargs.feed_element)
    out = {"elec_deg": th, "ratio": ratio, "zin": zin, "error": abs(ratio - target), "target": target, "z0": z0,
           "exact": exact_line_solution(resp, target, gargs.crossed, gargs.feed_element), "range": (lo, hi)}
    if err is objective_err:
        out["pattern"] = line_pattern_metrics(resp, z0, th, gargs.crossed, gargs.feed_element,
                                              objective["rear_sector"])
        out["objective"] = objective["objective"]
    return out


def synthesis_text(syn):
    """One-line summary of a synthesize_line result."""
    r = syn["ratio"]
    txt = (f"line {syn['elec_deg']:.1f} deg at Z0 {syn['z0']:.1f} ohm gives |I_far/I_fed| = {abs(r):.3f}, phase "
           f"{math.degrees(cmath.phase(r)):+.1f} deg (target 1.000, {math.degrees(cmath.phase(syn['target'])):+.1f} "
           f"deg; error {syn['error']:.3f}); Zin at the fed port {syn['zin'].real:.1f}{syn['zin'].imag:+.1f}j ohm")
    ex = syn.get("exact")
    if syn.get("pattern"):
        m = syn["pattern"]
        return (f"line {syn['elec_deg']:.1f} deg at Z0 {syn['z0']:.1f} ohm maximizes the {syn['objective']} "
                f"objective of the probe patterns (gain {m['forward_gain']:.2f} dBi, rear-sector F/B "
                f"{m['fb_sector']:.1f} dB on the horizon ring); the two-element cardioid ratio is not the target with "
                f"reflectors/directors (|I_far/I_fed| = {abs(r):.3f}, phase {math.degrees(cmath.phase(r)):+.1f} deg); "
                f"Zin at the fed port {syn['zin'].real:.1f}{syn['zin'].imag:+.1f}j ohm")
    if ex:
        txt += f". The exact target needs Z0 = {ex[1]:.1f} ohm and {ex[0]:.1f} deg"
    return txt


def run_optimize(p, args, network_given):
    check_common_model_args(p, args)
    single = args.network in ("tl", "lc")
    fixed_d = [bool(getattr(args, f"{w}_sections") or getattr(args, f"{w}_diameter") is not None)
               for w in ("rear", "front")]
    if args.refine_diameter and all(fixed_d):
        if args.refine:
            print("NOTE: --refine-diameter has nothing to move: both elements have fixed diameters/sections.")
        args.refine_diameter = False
    elif args.refine_diameter and any(fixed_d) and args.refine:
        print("NOTE: --refine-diameter moves only the element without --rear/front-diameter or sections.")
    # Defaults that depend on the network (single feed point: Zin exists and matters).
    if args.vswr_weight is None:
        args.vswr_weight = 3.0 if single else 0.0
    if args.realized_gain is None:
        args.realized_gain = single
    elif args.realized_gain and not single:
        print("WARNING: --realized-gain needs a single feed point (--network tl or lc); ignored.", file=sys.stderr)
        args.realized_gain = False
    n_par = args.reflectors + args.directors
    if args.refine is None:
        args.refine = 8 if n_par else (6 if single else 0)
        if args.refine:
            print(f"NOTE: --refine defaults to {args.refine} rounds"
                  + (" (the individual lengths and spacings of the reflectors/directors are tuned by the local "
                     "search)" if n_par else "") + "; use --refine 0 for the grid only.")
    elif n_par and args.refine == 0:
        print("NOTE: --refine 0 with reflectors/directors: their individual lengths and spacings stay at the grid "
              "values.")
    if args.refine < 0 or args.nm_evals < 0:
        p.error("--refine and --nm-evals must be >= 0")
    try:
        par_offsets = parse_values(args.parasitic_length_offsets, "parasitic length offset", -0.3, 0.3) \
            if n_par else [0.0]
        par_scales = parse_values(args.parasitic_spacing_scales, "parasitic spacing scale", 0.3, 3.0) \
            if n_par else [1.0]
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    if args.zin_x_max <= 0 or args.vswr_weight < 0 \
            or not all(math.isfinite(v) for v in (args.w_gain, args.w_fb, args.zin_x_max, args.vswr_weight)):
        p.error("--zin-x-max must be > 0; --vswr-weight >= 0; the weights must be finite")
    try:
        r_min, r_max = [float(x) for x in args.zin_r_range.split(",")]
    except ValueError:
        p.error("--zin-r-range must have the format min,max")
    if not 0 < r_min < r_max:
        p.error("--zin-r-range requires 0 < min < max")
    try:
        d_min, d_max = [float(x) for x in args.diameter_range.split(",")]
    except ValueError:
        p.error("--diameter-range must have the format min,max (mm)")
    if not 0 < d_min < d_max:
        p.error("--diameter-range requires 0 < min < max")
    diam_list = None                 # purchasable diameters: the refinement steps through this list
    if str(args.diameter_values).strip().lower() != "any":
        try:
            diam_list = sorted({v for v in parse_values(args.diameter_values, "--diameter-values", 0.1, 200.0)
                                if d_min - 1e-9 <= v <= d_max + 1e-9})
        except argparse.ArgumentTypeError as exc:
            p.error(str(exc))
        if not diam_list:
            p.error("--diameter-values has no value inside --diameter-range")
        if args.refine_diameter and not any(fixed_d):
            snapped = min(diam_list, key=lambda v: abs(v - args.element_diameter))
            if abs(snapped - args.element_diameter) > 1e-9:
                print(f"NOTE: --element-diameter {args.element_diameter:g} mm snapped to the purchasable "
                      f"{snapped:g} mm (--diameter-values).")
                args.element_diameter = snapped
    # Parasitic tubes: the (snapped) element diameter unless given; a given diameter is snapped to the purchasable
    # list as well, since it is never moved by the refinement.
    par_diam = args.parasitic_diameter or args.element_diameter
    if diam_list and args.parasitic_diameter and n_par:
        snapped = min(diam_list, key=lambda v: abs(v - par_diam))
        if abs(snapped - par_diam) > 1e-9:
            print(f"NOTE: --parasitic-diameter {par_diam:g} mm snapped to the purchasable {snapped:g} mm "
                  "(--diameter-values).")
            par_diam = snapped
    args.parasitic_diameter = par_diam
    tmatch = args.feed_style == "tmatch" and single
    try:
        tap_list = (parse_values(args.tap_values, "--tap-values", 1.0, 2000.0) if args.tap_values
                    else [args.tap_mm]) if tmatch else [None]
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    if args.tap_values and not tmatch:
        print("WARNING: --tap-values only applies with --feed-style tmatch; ignored.", file=sys.stderr)
    match_q = (args.match_q_l, args.match_q_c)
    auto_c = args.series_c_pf == "auto"
    if args.tolerance_mc < 0 or args.tol_top_k < 1:
        p.error("--tolerance-mc must be >= 0 and --tol-top-k >= 1")
    if 0 < args.tolerance_mc < 60:
        print(f"WARNING: --tolerance-mc {args.tolerance_mc}: the 10th percentile rests on "
              f"{max(1, args.tolerance_mc // 10)} sample(s); its sampling error is comparable to the differences "
              "between the best candidates. Use >= 100.", file=sys.stderr)
    n_jobs = args.jobs if args.jobs else (os.cpu_count() or 1)
    if n_jobs < 1:
        p.error("--jobs must be >= 1")
    freqs = args.band_freqs or [args.freq]
    design_i = min(range(len(freqs)), key=lambda i: abs(freqs[i] - args.freq))
    try:
        line_offsets = parse_values(args.line_offset_values, "line offset", -360, 360)
        abs_coax = parse_values(args.coax_length_values, "coaxial length", 0.001, 10.0) if args.coax_length_values else None
        z0_list = parse_values(args.z0_values, "z0", 10.0, 2000.0) if args.z0_values else None
        ratio_list = parse_values(args.ratio_values, "ratio", 0.01, 2.0)  # always validated
        ratios = [1.0] if single else ratio_list
        phase_offsets = parse_values(args.phase_offset_values, "phase offset", -360, 360)
        abs_phases = parse_values(args.phase_values, "phase", -360, 360) if args.phase_values else None
        spacings = parse_values(args.spacing_values, "spacing", 0.001, 0.5)
        if args.network == "lc" and args.coax_length_values:
            p.error("--coax-length-values is only used with --network tl; for lc use --line-offset-values")
        rear_lengths = parse_values(args.rear_length_values, "rear length", 0.5, 1.5)
        front_lengths = parse_values(args.front_length_values, "front length", 0.5, 1.5)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    if args.network == "tl":
        if z0_list is not None and args.line_type == "openwire":
            if args.line_spacing_given:
                p.error("--z0-values on an open-wire line sets the spacing for each Z0; do not give --line-spacing")
            z_min = openwire_z0(1.0, OPENWIRE_MIN_RATIO)
            bad = [z for z in z0_list if z < z_min]
            if bad:
                p.error(f"--z0-values {bad}: Z0 too low for an open-wire line (D/d would be below "
                        f"{OPENWIRE_MIN_RATIO:g}; Z0 must be at least {z_min:.0f} ohm)")
            for z in z0_list:
                w_ratio = openwire_ratio_warning(args.line_wire_diameter,
                                                 openwire_spacing_mm(z, args.line_wire_diameter))
                if w_ratio:
                    print(f"WARNING: Z0 {z:g} ohm: {w_ratio}.", file=sys.stderr)
        z0s = z0_list or [args.coax_z0]
    else:
        z0s = [None]
    try:
        rear_start, rear_end = [int(x.strip()) for x in args.rear_sector.split(",")]
    except Exception:
        p.error("--rear-sector must have the format start,end, for example 150,210")
    if not (0 <= rear_start < rear_end <= 360):
        p.error("--rear-sector requires 0 <= start < end <= 360")
    exe = find_nec2c(args)
    if not exe:
        p.error("nec2c was not found (--nec2c, HB9CV_NEC2C or PATH); install it before running the scan")
    nec_ver = nec2c_version(exe)
    lam = C_LIGHT / (args.freq * 1e6)
    # Pre-calculated parasitic elements: starting values and limits (minimum/maximum length and spacing).
    rd0 = element_sections(args, "rear")[0][1]
    fd0 = element_sections(args, "front")[0][1]
    if args.refine_diameter and args.refine:   # conservative minimum spacing: the driven tubes may grow
        rd0 = rd0 if fixed_d[0] else max(rd0, d_max)
        fd0 = fd0 if fixed_d[1] else max(fd0, d_max)
    base_layout = parasitic_layout(args, args.freq, 0.0, 0.0, rd0, fd0)
    ref_base = [e for e in base_layout if e["role"] == "reflector"]
    dir_base = [e for e in base_layout if e["role"] == "director"]
    ref_lim = [e["limits"] for e in ref_base]
    dir_lim = [e["limits"] for e in dir_base]
    l_res_par = resonant_length_m(args.freq, par_diam)
    allow_m = args.line_allowance_mm / 1000.0
    fb_key = FB_METRIC_KEYS[args.fb_metric]

    def min_line(spacing):
        """Shortest buildable TL line (degrees) for a spacing in wavelengths; 0 for the LC network."""
        if args.network != "tl":
            return 0.0
        return line_min_elec_deg(feed_distance_m(args, spacing * lam), lam, args.coax_vf, allow_m)

    def line_too_short(spacing, elec):
        """True when a TL line of elec degrees cannot span the feed points (spacing in wavelengths)."""
        return args.network == "tl" and elec + 1e-9 < min_line(spacing)

    def max_line(spacing):
        """Longest line (degrees) scanned for a spacing: --max-line-deg or the automatic limit; 179.99 for LC."""
        if args.network != "tl":
            return 179.99
        return auto_max_line_deg(args, line_electrical_deg(theory_line_lag(args.feed_element, spacing), args.crossed),
                                 min_line(spacing), line_offsets)

    def line_valid(spacing, elec):
        """A realizable line/network of elec degrees: LC 0..180 (low-pass pi); TL longer than zero and than the
        distance between the feed points, never beyond --max-line-deg (values are never taken modulo 360)."""
        if args.network == "lc":
            return 0.0 < elec < 180.0
        return TL_MIN_ELEC_DEG <= elec <= max_line(spacing) + 1e-9 and not line_too_short(spacing, elec)

    def clamp_par(c):
        """Candidate with every parasitic length/spacing forced into its pre-calculated limits."""
        d = dict(c)
        for lk, sk, lim in (("ref_len", "ref_sp", ref_lim), ("dir_len", "dir_sp", dir_lim)):
            d[lk] = tuple(round(min(max(v, L["len_min"]), L["len_max"]), 6) for v, L in zip(c[lk], lim))
            d[sk] = tuple(round(min(max(v, L["sp_min"]), L["sp_max"]), 6) for v, L in zip(c[sk], lim))
        return d

    if args.boom_isolated and n_par == 0 and args.boom_length / 1000.0 <= max(spacings) * lam:
        p.error(f"--boom-length ({args.boom_length:g} mm) must be greater than the largest spacing "
                f"({max(spacings) * lam * 1000:.1f} mm)")

    def phase_options(spacing):
        """Rear phases (ideal) or a single marker (tl/lc) for a spacing."""
        if single:
            return [None]
        if abs_phases is not None:
            return abs_phases
        theory = theory_rear_phase(spacing)
        return [wrap180(theory + off) for off in phase_offsets]

    theory_by_sp = {sp: line_electrical_deg(theory_line_lag(args.feed_element, sp), args.crossed) for sp in spacings}

    gen_base = dict(objective=args.objective, mount=args.mount, network=args.network, freq=args.freq,
                    segments=args.segments, az_step=args.az_step, coax_vf=args.coax_vf, line_type=args.line_type,
                    line_wire_diameter=args.line_wire_diameter,
                    feedline_length=args.feedline_length, feedline_diameter=args.feedline_diameter,
                    mast_length_m=args.mast_length_m, mast_diameter_mm=args.mast_diameter_mm,
                    mast_gap_mm=args.mast_gap_mm, mast_offset_mm=args.mast_offset_mm,
                    lc_z0=args.lc_z0, boom_length=args.boom_length, boom_size=args.boom_size,
                    boom_shape=args.boom_shape, boom_isolated=args.boom_isolated,
                    element_diameter=args.element_diameter, target_impedance=args.target_impedance,
                    feed_element=args.feed_element, crossed=args.crossed, fsweep=None, phase_target_deg=None,
                    lc_l_nh=None, lc_c_pf=None, ground=args.ground, ground_eps=args.ground_eps,
                    ground_sigma=args.ground_sigma, height_m=args.height_m, orientation=args.orientation,
                    conductivity=args.conductivity, rear_diameter=args.rear_diameter,
                    front_diameter=args.front_diameter, rear_sections=args.rear_sections,
                    front_sections=args.front_sections, gap_mm=args.gap_mm, model_feed_gap=args.model_feed_gap,
                    reflectors=args.reflectors, directors=args.directors, parasitic_diameter=par_diam,
                    min_gap_mm=args.min_gap_mm, boom_overhang_mm=args.boom_overhang_mm,
                    line_allowance_mm=args.line_allowance_mm, feed_style=args.feed_style,
                    tap_offset_mm=args.tap_offset_mm, tap_wire_mm=args.tap_wire_mm, series_c_pf=args.series_c_pf,
                    series_c_q=args.series_c_q, boom_clearance_mm=args.boom_clearance_mm,
                    ideal_drive=args.ideal_drive, extended_kernel=args.extended_kernel, nec2c=args.nec2c)

    def gargs_from(c):
        z0c = c["z0"]
        spmm = (openwire_spacing_mm(z0c, args.line_wire_diameter)
                if z0c is not None and args.line_type == "openwire" else None)
        d = dict(gen_base)
        d.update(spacing_factor=c["spacing"], rear_ratio=c["ratio"], rear_length_factor=c["rear"],
                 front_length_factor=c["front"], rear_phase=c["phase"], coax_length=c["coax"],
                 coax_z0=z0c if z0c is not None else args.coax_z0, line_spacing=spmm,
                 line_deg=c["line_elec"] if (c["coax"] is None and args.network == "lc") else None,
                 element_diameter=c["diam"], tap_mm=c.get("tap") or args.tap_mm,
                 reflector_lengths=tuple(c["ref_len"]) or None, reflector_spacings=tuple(c["ref_sp"]) or None,
                 director_lengths=tuple(c["dir_len"]) or None, director_spacings=tuple(c["dir_sp"]) or None)
        return Namespace(**d)

    def final_gargs(c):
        """gargs of a candidate with its own series capacitor when one was chosen for it (--series-c-pf auto)."""
        g = gargs_from(c)
        if "series_c" in c:
            g.series_c_pf = c["series_c"]
        return g

    def nominal_candidate(spacing, z0, tap, front=None):
        """Median lengths (or the given front factor), nominal parasitics and the given tap: the geometry the line
        synthesis uses."""
        return {"rear": sorted(rear_lengths)[len(rear_lengths) // 2],
                "front": front if front is not None else sorted(front_lengths)[len(front_lengths) // 2],
                "spacing": spacing, "ratio": 1.0,
                "phase": None, "line_off": None, "line_elec": None, "coax": None, "z0": z0,
                "diam": args.element_diameter, "tap": tap,
                "ref_len": tuple(e["len_factor"] for e in ref_base), "ref_sp": tuple(e["spacing_lambda"] for e in ref_base),
                "dir_len": tuple(e["len_factor"] for e in dir_base), "dir_sp": tuple(e["spacing_lambda"] for e in dir_base)}

    # Centre of the line scan of every (spacing, Z0, tap): the synthesized line (or the theory), moved up when the
    # lowest offset would fall below the shortest buildable line. Computed for the median front length and again
    # after the pre-tune has moved the front-length grid.
    line_bases, line_syn = {}, {}

    def compute_line_bases(front=None, quiet=False):
        line_bases.clear()
        line_syn.clear()
        if not (single and abs_coax is None):
            return
        for sp in spacings:
            for z0 in z0s:
                for tap in tap_list:
                    lo = max(min_line(sp), TL_MIN_ELEC_DEG) if args.network == "tl" else 0.5
                    hi = max_line(sp) if args.network == "tl" else 179.5
                    base = theory_by_sp[sp]
                    if args.synthesize_line:
                        try:
                            syn = synthesize_line(exe, gargs_from(nominal_candidate(sp, z0, tap, front)), lo, hi,
                                                  objective=syn_objective)
                            base = syn["elec_deg"]
                            line_syn[(sp, z0, tap)] = syn
                        except (ValueError, OSError) as exc:
                            print(f"NOTE: line synthesis failed at spacing {sp:g} lambda ({exc}); the scan is "
                                  "centered on the theoretical line.")
                    if base + min(line_offsets) < lo:
                        base = lo - min(line_offsets)
                    line_bases[(sp, z0, tap)] = base
        if not quiet:
            for (sp, z0, tap), syn in sorted(line_syn.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0.0,
                                                                               kv[0][2] or 0.0)):
                print(f"Line synthesis, spacing {sp:g} lambda" + (f", tap {tap:g} mm" if tap is not None else "")
                      + f": {synthesis_text(syn)}; scan centered on {line_bases[(sp, z0, tap)]:.1f} deg.")

    # With reflectors/directors in free space the synthesis maximizes the selected objective computed from the
    # superposed probe patterns instead of aiming at the two-element cardioid.
    syn_objective = None
    if n_par and args.ground == "free":
        syn_objective = {"objective": args.objective, "w_gain": args.w_gain, "w_fb": args.w_fb,
                         "rear_sector": (rear_start, rear_end), "fb_metric": args.fb_metric}
    elif n_par and args.synthesize_line and single:
        print("NOTE: over ground the line synthesis aims at the two-element cardioid relation; with "
              "reflectors/directors that is only a starting point, the scan and refinement find the line.")
    compute_line_bases()

    def line_options(spacing, z0=None, tap=None):
        """Tuples (offset, electrical degrees, physical length) for a spacing, a line Z0 and a tap."""
        if not single:
            return [(None, None, None)]
        z0 = z0 if z0 is not None else z0s[0]
        tap = tap if tap is not None else tap_list[0]
        if abs_coax is not None:
            cand = [(None, 360.0 * (L / args.coax_vf) / lam, L) for L in abs_coax if L > TL_MIN_LENGTH_M]
        else:
            base = line_bases[(spacing, z0, tap)]
            cand = [(off, base + off, line_phys_length(base + off, lam, args.coax_vf) if args.network == "tl" else None)
                    for off in line_offsets]
        out = [c for c in cand if line_valid(spacing, c[1])]
        line_dropped[(spacing, z0, tap)] = (len(cand) - len(out), len(cand))
        return out

    def line_phys(elec):
        return line_phys_length(elec, lam, args.coax_vf) if args.network == "tl" else None

    line_dropped = {}
    if single and all(not line_options(sp, z0, tap) for sp in spacings for z0 in z0s for tap in tap_list):
        p.error("no line value gives a realizable line (LC: 0..180 electrical degrees; TL: length > 0, not "
                "shorter than the distance between the feed points and not above --max-line-deg); adjust "
                "--line-offset-values / --line-length-values / --spacing-values / --max-line-deg")
    for (sp, z0, tap), (dropped, total_l) in sorted(line_dropped.items(),
                                                    key=lambda kv: (kv[0][0], kv[0][1] or 0.0, kv[0][2] or 0.0)):
        if dropped:
            print(f"NOTE: spacing {sp:g} lambda" + (f", tap {tap:g} mm" if tap is not None else "")
                  + f": {dropped} of {total_l} line values dropped (shorter than the "
                  f"distance between the feed points, {min_line(sp):.1f} deg at VF {args.coax_vf:g}"
                  + (f" with {args.line_allowance_mm:g} mm of connections" if allow_m else "")
                  + f", or longer than {max_line(sp):.0f} deg)")

    total = sum(len(ratios) * len(rear_lengths) * len(front_lengths) * len(phase_options(sp))
                * sum(len(line_options(sp, z0, tap)) for z0 in z0s for tap in tap_list) * len(par_offsets)
                * len(par_scales) for sp in spacings)
    runs_per = 2 if args.ground != "free" else 1
    print(f"Scan: {total} candidates; objective={args.objective}; mount={args.mount}; "
          f"network={args.network}{'' if network_given else ' (default value)'}; "
          f"{n_jobs} parallel job(s); nec2c: {nec_ver}")
    if n_par:
        print(f"Parasitic elements: {args.reflectors} reflector(s) + {args.directors} director(s), "
              f"{par_diam:g} mm tubes; lengths and spacings are limited to the pre-calculated minimum/maximum "
              "values below and tuned by the grid offsets/scales and the refinement.")
        for txt in parasitic_report(parasitic_layout(args, args.freq, -0.5 * spacings[0] * lam, 0.5 * spacings[0] * lam,
                                                     rd0, fd0)):
            print(txt)
    if single:
        print(f"{'Crossed' if args.crossed else 'Straight'} line fed at the {args.feed_element} element"
              + (f" through T-matches (taps {', '.join(f'{t:g}' for t in tap_list)} mm)" if tmatch else "")
              + "; scan around the " + ("synthesized" if line_syn else "theoretical") + " line of each spacing.")
        if args.match_score:
            print(f"Score through an ideal L-network designed at {args.freq:g} MHz against {args.target_impedance:g} "
                  f"ohm (Q_L {args.match_q_l:g}, Q_C {args.match_q_c:g}): matched VSWR over the band, gain after the "
                  "network loss and mismatch" + (f"; the front-back objective pays {args.loss_weight:g} dB of score "
                                                  "per dB of that loss (efficiency penalty; the F/B itself is "
                                                  "unchanged)" if args.objective == "front-back" else "") + ".")
    if args.band_freqs:
        print(f"Band: {len(freqs)} points {freqs[0]:g}..{freqs[-1]:g} MHz; score = worst case over the band"
              + (f" minus {args.vswr_weight:g} dB per unit of (max VSWR - 1)" if single and args.vswr_weight else "")
              + ("; the LC network gets its NT admittances per frequency" if args.network == "lc" else ""))
    if args.ground != "free":
        print(f"Ground {args.ground}, elements {args.orientation}, height {args.height_m:.3f} m: elevation cut -> "
              f"take-off angle -> azimuth ring ({runs_per} nec2c run(s) per candidate).")
    if not single:
        print("Note: with two EX sources there is no single port; the Zin constraints, VSWR_vs_target and "
              "realized gain only apply with --network tl or lc.")

    def layout_for(c):
        """Parasitic layout (and boom) of a candidate, with the same geometry the deck uses."""
        sp_m = c["spacing"] * lam
        ga = gargs_from(c)
        return parasitic_layout(ga, args.freq, -sp_m / 2, sp_m / 2, element_sections(ga, "rear")[0][1],
                                element_sections(ga, "front")[0][1])

    def apply_match(per_freq, net=None):
        """--match-score: design the L-network at the design frequency (both roots tried, the lower worst VSWR over
        the band kept), build it with components of Q --match-q-l/--match-q-c and store in every per-frequency row
        the matched VSWR (m_vswr), the network dissipation plus residual mismatch (m_loss, dB) and the gain after
        them (m_gain). A given net (the nominal design's network, in the Monte-Carlo) is used as built instead.
        Returns the network, or None when Zin cannot be matched (R <= 0)."""
        z_d = per_freq[design_i]["zin"]
        if net is None and z_d is not None and z_d.real > 0:
            w0 = 2 * math.pi * args.freq * 1e6
            table = [(x["freq_mhz"], x["zin"]) for x in per_freq]
            try:
                net = lnet_with_q(lnet_select(z_d, args.target_impedance, w0, "best", table)[0], *match_q)
            except ValueError:
                net = None
        for x in per_freq:
            if net is None or x["zin"] is None or x["zin"].real <= 0:
                x.update(m_vswr=math.inf, m_loss=math.inf, m_gain=-math.inf)
                continue
            w = 2 * math.pi * x["freq_mhz"] * 1e6
            zi = lnet_zin(x["zin"], w, net)
            tot = lnet_losses(x["zin"], w, net, args.target_impedance)[2]
            x.update(m_vswr=lnet_metrics(zi, args.target_impedance)[0], m_loss=tot, m_gain=x["forward_gain"] - tot)
        per_freq[design_i]["m_net"] = net
        return net

    def score_candidate(per_freq, net=None):
        """Worst case over the band of the objective term, minus the VSWR penalty at the worst VSWR.
        With --match-score the VSWR is the one after the matching network (net: a fixed network, see apply_match)
        and the gain is the gain after its loss; the front-back objective keeps its pure F/B and pays
        --loss-weight dB of score per dB of network loss (an efficiency penalty: the loss does not change the F/B).
        Returns (score, index of the worst frequency, Zin-constraint violation text, max VSWR)."""
        terms, vswrs, violation = [], [], ""
        if single and args.match_score:
            apply_match(per_freq, net)
        for pf in per_freq:
            if single and args.match_score:
                g_used, loss = pf["m_gain"], pf["m_loss"]
            else:
                g_used = pf["realized"] if (args.realized_gain and pf["realized"] is not None) else pf["forward_gain"]
                loss = 0.0
            rfb = pf[fb_key]
            terms.append(rfb - args.loss_weight * loss if args.objective == "front-back"
                         else g_used if args.objective == "gain" else args.w_gain * g_used + args.w_fb * rfb)
            if single:
                vswrs.append(pf["m_vswr"] if args.match_score else pf["vswr"])
                z = pf["zin"]
                if not violation:
                    if not r_min <= z.real <= r_max:
                        violation = f"R_in={z.real:.1f} outside {r_min:g}..{r_max:g} at {pf['freq_mhz']:.3f} MHz"
                    elif abs(z.imag) > args.zin_x_max:
                        violation = f"|X_in|={abs(z.imag):.1f} > {args.zin_x_max:g} at {pf['freq_mhz']:.3f} MHz"
        terms = [t if math.isfinite(t) else -1e9 for t in terms]
        worst = min(range(len(terms)), key=lambda i: terms[i])
        score, vmax = terms[worst], None
        if single:
            vmax = max(vswrs)
            score = score - args.vswr_weight * (vmax - 1) if math.isfinite(vmax) else -1e9
        return score, worst, violation, vmax

    def penalty_of(vmax):
        """VSWR penalty subtracted from the objective term (dB); 0 without a single feed point."""
        if vmax is None or not math.isfinite(vmax):
            return 0.0
        return args.vswr_weight * (vmax - 1)

    def zin_violation(per_freq):
        """Total ohms by which Zin leaves the admissible window (R range, |X| limit) over the band; 0 = feasible."""
        if not single:
            return 0.0
        v = 0.0
        for pf_ in per_freq:
            z = pf_["zin"]
            v += max(0.0, r_min - z.real) + max(0.0, z.real - r_max) + max(0.0, abs(z.imag) - args.zin_x_max)
        return v

    def rank_of(score, viol):
        """Sort key (smaller = better): feasible by score, infeasible by violation then score."""
        return (0, -score, 0.0) if viol <= 0 else (1, viol, -score)

    def rank_better(a, b):
        if a[0] != b[0]:
            return a[0] < b[0]
        if a[1] < b[1] - 1e-9:
            return True
        return abs(a[1] - b[1]) <= 1e-9 and a[2] < b[2] - 1e-9

    evals = []
    st = {"idx": 0, "total": 0, "warn_runs": 0, "warn_total": 0, "best": None}
    rejected, warn_samples = {}, []

    with tempfile.TemporaryDirectory(prefix="hb9cv_opt_") as td_name, make_pool(n_jobs) as pool:
        td = Path(td_name)

        def job_for(name, c, fr, pt_mode=-1, gargs=None):
            return dict(exe=exe, td=str(td), name=name, gargs=gargs if gargs is not None else gargs_from(c), freqs=fr,
                        rear_start=rear_start, rear_end=rear_end, pt_mode=pt_mode)

        def cand_key(c):
            """Canonical key of a candidate: two candidates with the same key build the same deck."""
            def r(v):
                return None if v is None else round(float(v), 6)
            return (r(c["rear"]), r(c["front"]), r(c["spacing"]), r(c["ratio"]), r(c["phase"]), r(c["line_elec"]),
                    r(c["coax"]), r(c["z0"]), r(c["diam"]), r(c.get("tap")), tuple(map(r, c["ref_len"])),
                    tuple(map(r, c["ref_sp"])), tuple(map(r, c["dir_len"])), tuple(map(r, c["dir_sp"])))

        sim_cache = {}      # candidate key -> CSV row (None: the run failed); every geometry is simulated once

        def absorb(idx, cand, res, stage):
            """Turn one simulation result into a CSV row; update the best candidate."""
            if not res["ok"]:
                key = f"{res['stage']}: {str(res['reason'])[:90]}"
                rejected[key] = rejected.get(key, 0) + 1
                print(f"[{idx}/{st['total']}] result discarded ({res['stage']}): {res['reason']}")
                return None
            if res["warnings"]:
                st["warn_runs"] += 1
                st["warn_total"] += len(res["warnings"])
                for w_line in res["warnings"]:
                    if w_line not in warn_samples and len(warn_samples) < 5:
                        warn_samples.append(w_line)
            pf = res["per_freq"]
            d = pf[design_i]
            score, worst_i, violation, vmax = score_candidate(pf)
            z0, line_off, line_elec, coax_length = cand["z0"], cand["line_off"], cand["line_elec"], cand["coax"]
            sp_mm = (openwire_spacing_mm(z0, args.line_wire_diameter)
                     if z0 is not None and args.line_type == "openwire" else None)
            lc_l = lc_c = None
            if args.network == "lc":
                lc_l, lc_c = lc_pi_values(line_elec, args.lc_z0, args.freq)
            zin = d["zin"]
            row = {"objective": args.objective, "network": args.network, "mount": args.mount,
                   "ground": args.ground, "orientation": args.orientation,
                   "height_m": args.height_m if args.ground != "free" else "",
                   "element_diameter_mm": cand["diam"], "conductivity_S_m": args.conductivity,
                   "boom_shape": args.boom_shape,
                   "boom_size_mm": args.boom_size, "boom_length_mm": args.boom_length,
                   "target_impedance_ohm": args.target_impedance,
                   "balun_ratio": args.balun_ratio,
                   "feedline_length_m": args.feedline_length,
                   "feed_element": args.feed_element if single else "",
                   "crossed_line": args.crossed if single else "",
                   "line_type": args.line_type if args.network == "tl" else "",
                   "line_z0_ohm": z0 if z0 is not None else "",
                   "line_wire_diameter_mm": args.line_wire_diameter if sp_mm is not None else "",
                   "line_spacing_mm": sp_mm if sp_mm is not None else "",
                   "line_vf": args.coax_vf if args.network == "tl" else "",
                   "line_length_m": coax_length if args.network == "tl" else "",
                   "nec_tl_length_m": coax_length / args.coax_vf if (args.network == "tl" and coax_length) else "",
                   "lc_z0_ohm": args.lc_z0 if args.network == "lc" else "",
                   "lc_l_nh": lc_l * 1e9 if lc_l is not None else "",
                   "lc_c_pf": lc_c * 1e12 if lc_c is not None else "",
                   "line_electrical_deg": line_elec if single else "",
                   "line_theory_offset_deg": line_off if line_off is not None else "",
                   "line_min_electrical_deg": min_line(cand["spacing"]) if args.network == "tl" else "",
                   "line_slack_mm": ((coax_length - feed_distance_m(args, cand["spacing"] * lam)) * 1000.0
                                     if (args.network == "tl" and coax_length) else ""),
                   "feed_style": args.feed_style if single else "",
                   "tap_mm": cand.get("tap") if tmatch else "",
                   "series_c_pf": d.get("series_c_pf") or "",
                   "rear_length_factor": cand["rear"],
                   "front_length_factor": cand["front"],
                   "rear_ratio": cand["ratio"] if not single else "",
                   "rear_phase_deg": cand["phase"] if cand["phase"] is not None else "",
                   "spacing_lambda": cand["spacing"], "forward_gain_dB": d["forward_gain"],
                   "forward_gain_dBd": d["gain_dbd"] if d["gain_dbd"] is not None else "",
                   "takeoff_deg": d["takeoff_deg"],
                   "hpbw_az_deg": d["hpbw"],
                   "max_gain_dB": d["max_gain"], "max_gain_angle_deg": d["max_angle"],
                   "realized_gain_dB": d["realized"] if d["realized"] is not None else "",
                   "rear_gain_dB": d["rear_gain"], "front_to_back_dB": d["fb"],
                   "rear_sector_max_dB": d["worst_rear_gain"], "rear_sector_max_angle_deg": d["worst_rear_angle"],
                   "front_to_back_sector_dB": d["robust_fb"], "front_to_back_cone_dB": d["cone_fb"],
                   "Zin_real": zin.real if zin is not None else "",
                   "Zin_imag": zin.imag if zin is not None else "",
                   "VSWR_vs_target": d["vswr"] if d["vswr"] is not None else "",
                   "band_points": len(freqs),
                   "band_worst_freq_MHz": pf[worst_i]["freq_mhz"],
                   "band_min_gain_dB": min(x["forward_gain"] for x in pf),
                   "band_min_fb_sector_dB": min(x["robust_fb"] for x in pf),
                   "band_min_fb_cone_dB": min(x["cone_fb"] for x in pf),
                   "band_max_vswr": vmax if vmax is not None else "",
                   "zin_constraint": violation,
                   "match_network": (f"{lnet_topology_text(d['m_net'])}: series {lnet_describe(d['m_net']['series'])}, "
                                     f"shunt {lnet_describe(d['m_net']['shunt'])}") if d.get("m_net") else "",
                   "matched_vswr_max": max(x["m_vswr"] for x in pf) if args.match_score else "",
                   "match_loss_dB": max(x["m_loss"] for x in pf) if args.match_score else "",
                   "matched_gain_dB": d["m_gain"] if args.match_score else "",
                   "loss_penalty_dB": (args.loss_weight * max(x["m_loss"] for x in pf)
                                       if args.match_score and args.objective == "front-back" else ""),
                   "nec_warnings": len(res["warnings"]),
                   "search": stage,
                   "fb_metric": args.fb_metric,
                   "objective_term": score + penalty_of(vmax) if score > -1e8 else "",
                   "vswr_penalty": penalty_of(vmax) if vmax is not None else "",
                   "score": score}
            lay = layout_for(cand)
            sp_c = cand["spacing"] * lam
            row.update({
                "reflectors": args.reflectors, "directors": args.directors,
                "reflector_length_mm": ";".join(f"{e['length_m'] * 1000:.1f}" for e in lay if e["role"] == "reflector"),
                "reflector_spacing_mm": ";".join(f"{e['spacing_m'] * 1000:.1f}" for e in lay if e["role"] == "reflector"),
                "director_length_mm": ";".join(f"{e['length_m'] * 1000:.1f}" for e in lay if e["role"] == "director"),
                "director_spacing_mm": ";".join(f"{e['spacing_m'] * 1000:.1f}" for e in lay if e["role"] == "director"),
                "parasitic_length_factors": ";".join(f"{e['len_factor']:.4f}" for e in lay),
                "boom_required_mm": (round(required_boom_mm(lay, -sp_c / 2, sp_c / 2, args.boom_overhang_mm), 1)
                                     if lay else "")})
            if args.tolerance_mc > 0:
                row.update(score_p10="", score_p10_lo="", score_p10_hi="", fb_p10="", vswr_p90="", mc_samples="")
            viol = zin_violation(pf)
            row["zin_violation_ohm"] = round(viol, 3)
            ev = {"row": row, "cand": dict(cand), "per_freq": pf, "score": score, "ok": not violation, "viol": viol}
            evals.append(ev)
            if ev["ok"] and (st["best"] is None or score > st["best"]["score"]):
                st["best"] = ev
            return row

        def eval_batch(cands, stage):
            """Simulate candidates (in parallel); rows come back in the order of cands. A candidate already
            simulated (same key, see cand_key) is not run again: its earlier row is returned."""
            keys = [cand_key(c) for c in cands]
            new, seen_now = [], set()
            for c, key in zip(cands, keys):
                if key not in sim_cache and key not in seen_now:
                    seen_now.add(key)
                    new.append((c, key))
            st["total"] += len(new)
            jobs, meta = [], []
            for c, _key in new:
                st["idx"] += 1
                meta.append(st["idx"])
                jobs.append(job_for(f"candidate_{st['idx']:04d}", c, freqs))
            for k, (idx, res) in enumerate(zip(meta, run_jobs(pool, jobs))):
                c, key = new[k]
                sim_cache[key] = absorb(idx, c, res, stage)
                if idx % 10 == 0 or k == len(new) - 1:
                    print(f"[{idx}/{st['total']}] valid={len(evals)}; best score="
                          + (f"{st['best']['score']:.2f} dB" if st["best"] else "none"))
            return [sim_cache[key] for key in keys]

        def line_offset_of(c):
            """Offset of the candidate's line from the scan centre of its spacing (None off the grid spacings)."""
            base = line_bases.get((c["spacing"], c["z0"], c.get("tap")))
            return None if base is None or c["line_elec"] is None else c["line_elec"] - base

        def tap_valid(c):
            """The T-match tap keeps 1.5 segments from the center and one segment from the tip of both elements."""
            length = min(150.0 / args.freq * c["rear"], 139.0 / args.freq * c["front"])
            seg = length / args.segments
            return 1.5 * seg <= c["tap"] / 1000.0 <= length / 2.0 - seg

        def next_diameter(d, sign, step):
            """Neighbouring purchasable diameter (--diameter-values), or d +- step in --diameter-range; None at
            the end of the list or range."""
            if diam_list:
                i = min(range(len(diam_list)), key=lambda k: abs(diam_list[k] - d))
                j = i + (1 if sign > 0 else -1)
                return diam_list[j] if 0 <= j < len(diam_list) else None
            v = round(d + sign * step, 4)
            return v if d_min <= v <= d_max else None

        def center_candidate():
            """Middle point of the grid: median lengths and spacing, line offset closest to the scan centre."""
            sp = sorted(spacings)[len(spacings) // 2]
            lopts = line_options(sp, z0s[0], tap_list[len(tap_list) // 2])
            if single and not lopts:
                return None
            lo = min(lopts, key=lambda t: abs(t[0]) if t[0] is not None else 0.0)
            ph = phase_options(sp)
            return clamp_par({"rear": sorted(rear_lengths)[len(rear_lengths) // 2],
                              "front": sorted(front_lengths)[len(front_lengths) // 2], "spacing": sp,
                              "ratio": ratios[len(ratios) // 2], "phase": ph[len(ph) // 2], "line_off": lo[0],
                              "line_elec": lo[1], "coax": lo[2], "z0": z0s[0], "diam": args.element_diameter,
                              "tap": tap_list[len(tap_list) // 2],
                              "ref_len": tuple(e["len_factor"] for e in ref_base),
                              "ref_sp": tuple(e["spacing_lambda"] for e in ref_base),
                              "dir_len": tuple(e["len_factor"] for e in dir_base),
                              "dir_sp": tuple(e["spacing_lambda"] for e in dir_base)})

        # ---------------- pre-tune: front length for X_in ~ 0 at --freq, then center the front grid on it
        if args.pretune and single:
            c0 = center_candidate()
            if c0 is not None:
                f_mid = c0["front"]

                def x_of(front):
                    c = dict(c0, front=front)
                    row = eval_batch([c], "pretune")[0]
                    return None if not row or row["Zin_imag"] == "" else row["Zin_imag"]

                pts, f_a = [], f_mid
                x_a = x_of(f_a)
                if x_a is not None:
                    pts.append((f_a, x_a))
                    f_b = f_a * (0.99 if x_a > 0 else 1.01)     # X grows with the element length
                    for _ in range(5):
                        if abs(pts[-1][1]) < 2.0:
                            break
                        x_b = x_of(f_b)
                        if x_b is None:
                            break
                        pts.append((f_b, x_b))
                        (f0_, x0_), (f1_, x1_) = pts[-2], pts[-1]
                        if abs(x1_) < 2.0 or x1_ == x0_:
                            break
                        f_new = f1_ - x1_ * (f1_ - f0_) / (x1_ - x0_)
                        f_b = min(max(f_new, 0.7 * f_mid, 0.5), 1.3 * f_mid, 1.5)
                    f_best, x_best = min(pts, key=lambda t: abs(t[1]))
                    shift = f_best - f_mid
                    front_lengths = sorted({round(min(max(v + shift, 0.5), 1.5), 6) for v in front_lengths})
                    print(f"Pre-tune: front length factor {f_best:.4f} gives X_in = {x_best:+.1f} ohm at "
                          f"{args.freq:g} MHz ({len(pts)} nec2c run(s)); front grid centered on it: "
                          + ",".join(f"{v:g}" for v in front_lengths))
                    if args.synthesize_line and abs(shift) > 1e-9 and abs_coax is None:
                        print("Line synthesis repeated for the pre-tuned front length:")
                        compute_line_bases(front=f_best)

        # ---------------- grid
        grid = []
        for rear_length, front_length, spacing in itertools.product(rear_lengths, front_lengths, spacings):
            line_sets = [(z0, tap, lo_) for z0 in z0s for tap in tap_list for lo_ in line_options(spacing, z0, tap)]
            for ratio, phase, (z0, tap, (line_off, line_elec, coax_length)), p_off, p_scale in itertools.product(
                    ratios, phase_options(spacing), line_sets, par_offsets, par_scales):
                cand0 = {"rear": rear_length, "front": front_length, "spacing": spacing, "ratio": ratio,
                         "phase": phase, "line_off": line_off, "line_elec": line_elec, "coax": coax_length,
                         "z0": z0, "diam": args.element_diameter, "tap": tap,
                         "ref_len": tuple(e["len_factor"] + p_off for e in ref_base),
                         "ref_sp": tuple(e["spacing_lambda"] * p_scale for e in ref_base),
                         "dir_len": tuple(e["len_factor"] + p_off for e in dir_base),
                         "dir_sp": tuple(e["spacing_lambda"] * p_scale for e in dir_base)}
                grid.append(clamp_par(cand0))
        eval_batch(grid, "grid")

        # ---------------- local refinement (all moves of a round run in parallel; best improvement wins)
        start_ev = st["best"]
        if start_ev is None and args.refine and single and evals:
            start_ev = min(evals, key=lambda e: (e["viol"], -e["score"]))
            print(f"NOTE: no grid candidate meets the Zin window (R {r_min:g}..{r_max:g}, |X| <= {args.zin_x_max:g} "
                  f"over the band). The refinement starts from the least-violating one (Zin out of the window by "
                  f"{start_ev['viol']:.1f} ohm summed over the band) and steers Zin into the window first, "
                  "then maximizes the score.")
        if args.refine and start_ev is not None:
            cur, cur_rank = dict(start_ev["cand"]), rank_of(start_ev["score"], start_ev["viol"])
            steps = {"len": 0.01, "ang": 5.0, "sp": 0.01, "diam": 1.0, "plen": 0.01, "psp": 0.01, "tap": 10.0}
            what = ["lengths"] + (["reflector/director lengths and spacings"] if n_par else []) \
                + (["phase"] if not single and abs_phases is None else []) \
                + (["line"] if single and abs_coax is None else []) + (["tap"] if tmatch else []) \
                + (["spacing"] if args.refine_spacing else []) + (["element diameter"] if args.refine_diameter else [])
            diam_txt = ("the next purchasable diameter" if diam_list else f"diameter {steps['diam']:g} mm")
            print(f"Local refinement: {args.refine} rounds from the best grid candidate over {', '.join(what)} "
                  f"(initial steps: lengths {steps['len']:g}, angles {steps['ang']:g} deg, spacing {steps['sp']:g} "
                  f"lambda, {diam_txt}, reflector/director length {steps['plen']:g} and "
                  f"spacing {steps['psp']:g} lambda" + (f", tap {steps['tap']:g} mm" if tmatch else "")
                  + "; halved when a round does not improve).")
            for rnd in range(args.refine):
                moves = [("rear", "len"), ("front", "len")]
                if cur["phase"] is not None and abs_phases is None:
                    moves.append(("phase", "ang"))
                if single and abs_coax is None:
                    moves.append(("line_elec", "ang"))
                if tmatch:
                    moves.append(("tap", "tap"))
                if args.refine_spacing:
                    moves.append(("spacing", "sp"))
                if args.refine_diameter:
                    moves.append(("diam", "diam"))
                for fld, sk_par in (("ref_len", "plen"), ("ref_sp", "psp"), ("dir_len", "plen"), ("dir_sp", "psp")):
                    moves += [(("par", fld, i), sk_par) for i in range(len(cur[fld]))]
                cands = []
                for key, sk in moves:
                    step = steps[sk]
                    for sign in (1, -1):
                        c = dict(cur)
                        if isinstance(key, tuple):
                            _, fld, i = key
                            vals = list(cur[fld])
                            vals[i] = round(vals[i] + sign * step, 6)
                            c[fld] = tuple(vals)
                            c = clamp_par(c)
                            if c[fld] == tuple(cur[fld]):
                                continue            # already at the pre-calculated limit
                        elif key in ("rear", "front"):
                            c[key] = round(cur[key] + sign * step, 6)
                            if not 0.5 <= c[key] <= 1.5:
                                continue
                        elif key == "phase":
                            c["phase"] = wrap180(cur["phase"] + sign * step)
                        elif key == "line_elec":
                            c["line_elec"] = round(cur["line_elec"] + sign * step, 6)
                            if not line_valid(cur["spacing"], c["line_elec"]):
                                continue
                            c["coax"] = line_phys(c["line_elec"])
                            c["line_off"] = line_offset_of(c)
                        elif key == "tap":
                            c["tap"] = round(cur["tap"] + sign * step, 3)
                            if not tap_valid(c):
                                continue
                        elif key == "spacing":
                            c["spacing"] = round(cur["spacing"] + sign * step, 6)
                            if not 0.03 <= c["spacing"] <= 0.30:
                                continue
                            if single and not line_valid(c["spacing"], cur["line_elec"]):
                                continue                  # the line no longer spans the new spacing
                            c["line_off"] = line_offset_of(c)
                        else:
                            c["diam"] = next_diameter(cur["diam"], sign, step)
                            if c["diam"] is None:
                                continue
                        cands.append(c)
                rows = eval_batch(cands, f"refined{rnd + 1}")
                best_i, best_rank = None, cur_rank
                for i, row in enumerate(rows):
                    if not row:
                        continue
                    rk = rank_of(row["score"], row["zin_violation_ohm"] if row["zin_constraint"] else 0.0)
                    if rank_better(rk, best_rank):
                        best_i, best_rank = i, rk
                if best_i is not None:
                    cur, cur_rank = cands[best_i], best_rank
                else:
                    steps = {k: v / 2 for k, v in steps.items()}
            start_ev = {"cand": cur, "rank": cur_rank}
        elif start_ev is not None:
            start_ev = {"cand": dict(start_ev["cand"]), "rank": rank_of(start_ev["score"], start_ev["viol"])}

        # ---------------- Nelder-Mead polish of the continuous parameters (handles correlated moves, e.g.
        # front length against line length, that a one-axis-at-a-time search cannot follow)
        if args.nm_evals and start_ev is not None:
            base_c = start_ev["cand"]
            keys = [("rear", 0.01), ("front", 0.01)]
            if args.refine_spacing:
                keys.append(("spacing", 0.01))
            if single and abs_coax is None:
                keys.append(("line_elec", 5.0))
            if tmatch:
                keys.append(("tap", 10.0))
            if base_c["phase"] is not None and abs_phases is None:
                keys.append(("phase", 5.0))
            if args.refine_diameter and diam_list is None:
                keys.append(("diam", 1.0))

            def nm_cand(x):
                c = dict(base_c)
                for (k, _), v in zip(keys, x):
                    c[k] = round(v, 2) if k == "diam" else round(v, 6)
                if not (0.5 <= c["rear"] <= 1.5 and 0.5 <= c["front"] <= 1.5):
                    return None
                if args.refine_spacing and not 0.03 <= c["spacing"] <= 0.30:
                    return None
                if args.refine_diameter and not d_min - 1e-9 <= c["diam"] <= d_max + 1e-9:
                    return None
                if c["phase"] is not None:
                    c["phase"] = wrap180(c["phase"])
                if tmatch and not tap_valid(c):
                    return None
                if single and not line_valid(c["spacing"], c["line_elec"]):
                    return None
                if single and abs_coax is None:
                    c["coax"] = line_phys(c["line_elec"])
                c["line_off"] = line_offset_of(c)
                return c

            def nm_value(rank):
                """Scalar to minimize: -score when Zin is in its window, otherwise 1000 + violation."""
                return rank[1] if rank[0] == 0 else 1e3 + rank[1] + 1e-3 * rank[2]

            budget = {"left": args.nm_evals}
            cache = {}

            def nm_eval_many(xs):
                todo, out = [], [None] * len(xs)
                for i, x in enumerate(xs):
                    key = tuple(round(v, 6) for v in x)
                    if key in cache:
                        out[i] = cache[key]
                        continue
                    c = nm_cand(x)
                    if c is None:
                        cache[key] = out[i] = math.inf
                        continue
                    todo.append((i, key, c))
                todo = todo[:max(0, budget["left"])]
                if todo:
                    budget["left"] -= len(todo)
                    rows = eval_batch([c for _, _, c in todo], "nelder-mead")
                    for (i, key, _), row in zip(todo, rows):
                        val = math.inf if not row else nm_value(
                            rank_of(row["score"], row["zin_violation_ohm"] if row["zin_constraint"] else 0.0))
                        cache[key] = out[i] = val
                return [math.inf if v is None else v for v in out]

            x0 = [base_c[k] for k, _ in keys]
            simplex = [x0] + [[v + (st_ if j == i else 0.0) for j, (v, (_, st_)) in enumerate(zip(x0, keys))]
                              for i in range(len(keys))]
            cache[tuple(round(v, 6) for v in x0)] = nm_value(start_ev["rank"])
            print(f"Nelder-Mead polish over {', '.join(k for k, _ in keys)} (budget {args.nm_evals} nec2c runs).")
            fvals = nm_eval_many(simplex)
            while budget["left"] > 0:
                order = sorted(range(len(simplex)), key=lambda i: fvals[i])
                simplex, fvals = [simplex[i] for i in order], [fvals[i] for i in order]
                if not math.isfinite(fvals[0]) or abs(fvals[-1] - fvals[0]) < 1e-3:
                    break
                cen = [sum(v[j] for v in simplex[:-1]) / (len(simplex) - 1) for j in range(len(keys))]
                worst_x = simplex[-1]
                xr = [c_ + (c_ - w_) for c_, w_ in zip(cen, worst_x)]
                fr = nm_eval_many([xr])[0]
                if fr < fvals[0]:
                    xe = [c_ + 2.0 * (c_ - w_) for c_, w_ in zip(cen, worst_x)]
                    fe = nm_eval_many([xe])[0]
                    simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    xc = [c_ + 0.5 * (w_ - c_) for c_, w_ in zip(cen, worst_x)]
                    fc = nm_eval_many([xc])[0]
                    if fc < fvals[-1]:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        shrunk = [[b_ + 0.5 * (v - b_) for v, b_ in zip(x, simplex[0])] for x in simplex[1:]]
                        simplex[1:] = shrunk
                        fvals[1:] = nm_eval_many(shrunk)
            print(f"Nelder-Mead finished ({args.nm_evals - budget['left']} runs).")

        # ---------------- Monte-Carlo tolerance analysis of the best K candidates. The perturbations are drawn
        # once and applied to every candidate (common random numbers), so differences between candidates come
        # from the designs, not from different random draws.
        mc_done = False
        if args.tolerance_mc > 0 and st["best"] is not None:
            rng = random.Random(args.tol_seed)
            top = sorted((e for e in evals if e["ok"]), key=lambda e: e["score"], reverse=True)[:args.tol_top_k]
            dist_txt = ("uniform" if args.tol_dist == "uniform" else
                        "normal, tolerance = 2 sigma, clipped at 3 sigma")
            tol_txt = [f"elements +-{args.tol_length_mm:g} mm", f"spacing +-{args.tol_spacing_mm:g} mm",
                       f"line/phase +-{args.tol_line_deg:g} deg"]
            if tmatch:
                tol_txt += [f"taps +-{args.tol_tap_mm:g} mm", f"T-bar offsets +-{args.tol_bar_mm:g} mm"]
            if args.network == "tl":
                tol_txt += [f"line VF +-{args.tol_vf_pct:g} %", f"line Z0 +-{args.tol_z0_pct:g} %"]
            elif args.network == "lc":
                tol_txt.append(f"LC network Z0 +-{args.tol_z0_pct:g} %")
            if args.series_c_pf is not None:
                tol_txt.append(f"series C +-{args.tol_c_pct:g} %")
            print(f"Tolerance Monte-Carlo: {len(top)} candidates x {args.tolerance_mc} common samples "
                  f"({dist_txt}: {', '.join(tol_txt)}; seed {args.tol_seed}).")
            print("The series capacitor and the matching network stay as designed for the nominal candidate"
                  if not args.tol_retune else
                  "The series capacitor (auto) and the matching network are re-tuned for every sample "
                  "(--tol-retune: as if adjusted on the bench after the build)")

            def unit():
                """One normalized deviation: the tolerance is 1."""
                if args.tol_dist == "uniform":
                    return rng.uniform(-1.0, 1.0)
                return max(-1.5, min(1.5, rng.gauss(0.0, 0.5)))

            n_pe = args.reflectors + args.directors
            draws = [{"dr": unit(), "df": unit(), "ds": unit(), "dl": unit(),
                      "pl": [unit() for _ in range(n_pe)], "ps": [unit() for _ in range(n_pe)],
                      "tr": unit(), "tf": unit(), "br": unit(), "bf": unit(), "vf": unit(), "z0": unit(),
                      "c": unit()} for _ in range(args.tolerance_mc)]

            def sample_gargs(e, d, u):
                """Model of one perturbed sample: geometry from d plus the build tolerances that are not part of a
                candidate (taps, T-bar offsets, line VF and Z0, series capacitor), the series capacitor and the
                matching network frozen at the nominal design unless --tol-retune."""
                g = gargs_from(d)
                if tmatch:
                    for w, kt, kb in (("rear", "tr", "br"), ("front", "tf", "bf")):
                        setattr(g, f"tap_mm_{w}", max(1.0, d["tap"] + u[kt] * args.tol_tap_mm))
                        setattr(g, f"tap_offset_mm_{w}", max(1.0, args.tap_offset_mm + u[kb] * args.tol_bar_mm))
                if args.network == "tl":
                    g.coax_vf = min(1.0, args.coax_vf * (1.0 + u["vf"] * args.tol_vf_pct / 100.0))
                    g.coax_z0 = g.coax_z0 * (1.0 + u["z0"] * args.tol_z0_pct / 100.0)
                elif args.network == "lc":
                    g.lc_z0 = args.lc_z0 * (1.0 + u["z0"] * args.tol_z0_pct / 100.0)
                if args.series_c_pf is not None and not (auto_c and args.tol_retune):
                    c_nom = e["per_freq"][design_i].get("series_c_pf") if auto_c else series_c_value(args)
                    g.series_c_pf = c_nom * (1.0 + u["c"] * args.tol_c_pct / 100.0) if c_nom else None
                return g

            def perturb(c, u):
                d = dict(c)
                d["rear"] = c["rear"] + u["dr"] * args.tol_length_mm / 1000.0 / (150.0 / args.freq)
                d["front"] = c["front"] + u["df"] * args.tol_length_mm / 1000.0 / (139.0 / args.freq)
                d["spacing"] = max(0.005, c["spacing"] + u["ds"] * args.tol_spacing_mm / 1000.0 / lam)
                k = 0
                for lk, sk in (("ref_len", "ref_sp"), ("dir_len", "dir_sp")):
                    pl, ps = u["pl"][k:k + len(c[lk])], u["ps"][k:k + len(c[lk])]
                    k += len(c[lk])
                    d[lk] = tuple(v + du * args.tol_length_mm / 1000.0 / l_res_par for v, du in zip(c[lk], pl))
                    d[sk] = tuple(max(0.005, v + du * args.tol_spacing_mm / 1000.0 / lam) for v, du in zip(c[sk], ps))
                dl = u["dl"] * args.tol_line_deg
                if single:
                    elec = c["line_elec"] + dl
                    if args.network == "lc":
                        elec = min(max(elec, 0.01), 179.99)
                    else:
                        # a built line is never shorter than the distance it spans, and the error never wraps
                        elec = min(max(elec, 0.01, min_line(d["spacing"])), max(max_line(d["spacing"]), c["line_elec"]))
                    d["line_elec"] = elec
                    d["coax"] = line_phys_length(elec, lam, args.coax_vf) if args.network == "tl" else None
                    d["line_off"] = None
                else:
                    d["phase"] = wrap180(c["phase"] + dl)
                return d

            boot_rng = random.Random(args.tol_seed + 1)

            def p10_interval(vals, n_boot=200):
                """10..90 % bootstrap interval of the 10th percentile."""
                bs = sorted(percentile([vals[boot_rng.randrange(len(vals))] for _ in vals], 10)
                            for _ in range(n_boot))
                return percentile(bs, 10), percentile(bs, 90)

            for k, e in enumerate(top):
                jobs = []
                for j, u in enumerate(draws):
                    d = perturb(e["cand"], u)
                    jobs.append(job_for(f"mc_{k:02d}_{j:03d}", d, freqs, gargs=sample_gargs(e, d, u)))
                net_nom = (e["per_freq"][design_i].get("m_net")
                           if (single and args.match_score and not args.tol_retune) else None)
                sc, fbs, vsw, failed = [], [], [], 0
                for res in run_jobs(pool, jobs):
                    if not res["ok"]:
                        failed += 1
                        continue
                    s_, _, _, vmax = score_candidate(res["per_freq"], net_nom)
                    sc.append(s_)
                    fbs.append(min(x[fb_key] for x in res["per_freq"]))
                    if vmax is not None:
                        vsw.append(vmax)
                if len(sc) < max(3, args.tolerance_mc // 2):
                    print(f"  candidate {k + 1}: only {len(sc)}/{args.tolerance_mc} samples simulated; ranked last.")
                    e["p10"] = -1e9
                    continue
                e["p10"] = percentile(sc, 10)
                lo_b, hi_b = p10_interval(sc)
                e["row"].update(score_p10=e["p10"], score_p10_lo=lo_b, score_p10_hi=hi_b,
                                fb_p10=percentile(fbs, 10), vswr_p90=percentile(vsw, 90) if vsw else "",
                                mc_samples=len(sc))
                print(f"  candidate {k + 1}: nominal score {e['score']:.2f}; p10 score {e['p10']:.2f} "
                      f"(bootstrap {lo_b:.2f}..{hi_b:.2f}); p10 F/B ({args.fb_metric}) {percentile(fbs, 10):.2f} dB"
                      + (f"; p90 VSWR {percentile(vsw, 90):.2f}" if vsw else "")
                      + (f"; {failed} failed" if failed else ""))
            ranked = [e for e in top if "p10" in e]
            if ranked:
                new_best = max(ranked, key=lambda e: e["p10"])
                if new_best is not st["best"]:
                    print("The robust (10th percentile) ranking prefers a different candidate than the nominal score.")
                overlap = [e for e in ranked if e is not new_best and "score_p10_hi" in e["row"]
                           and e["row"]["score_p10_hi"] >= new_best["row"].get("score_p10_lo", math.inf)]
                if overlap:
                    print(f"NOTE: the bootstrap interval of the winner overlaps {len(overlap)} other candidate(s): "
                          "the robust ranking between them is not significant; raise --tolerance-mc.")
                st["best"] = new_best
                mc_done = True

        zin_window_met = st["best"] is not None
        if st["best"] is None and evals and single and not args.strict_zin:
            # No candidate meets the Zin window: the closest one is reported and written, flagged as such.
            st["best"] = min(evals, key=lambda e: (e["viol"], -e["score"]))
        best_ev = st["best"]
        final = None
        if best_ev is not None and auto_c:
            # The written deck carries the capacitor chosen for the best candidate as a number.
            c_best = best_ev["per_freq"][design_i].get("series_c_pf")
            best_ev["cand"]["series_c"] = c_best
            gen_base["series_c_pf"] = None
        if best_ev is not None:
            final = simulate_model(exe, td, "best_final", final_gargs(best_ev["cand"]), [args.freq],
                                   rear_start, rear_end, pt_mode=0)
            if final["ok"]:
                final["deck_text"] = Path(final["deck_path"]).read_text(errors="replace")
                final["out_text"] = Path(final["out_path"]).read_text(errors="replace")

    # ---------------- report
    warn_runs, warn_total = st["warn_runs"], st["warn_total"]
    if rejected:
        print(f"Runs discarded because of NEC2C messages or validation: {sum(rejected.values())}")
        for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {count} x {reason}")
    if warn_total:
        print(f"NEC2C warnings accepted: {warn_total} in {warn_runs} runs; examples:")
        for w_line in warn_samples:
            print(f"  {w_line}")
    if not evals:
        p.error("there were no valid simulations; no optimized model was generated")
    csv_path = Path(args.csv).resolve()
    ordered = sorted((e for e in evals), key=lambda e: (
        isinstance(e["row"].get("score_p10"), float), e["row"].get("score_p10") if isinstance(e["row"].get("score_p10"), float) else 0.0,
        e["score"]), reverse=True)
    meta_lines = [f"hb9cv.py optimize; written {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}",
                  f"nec2c: {nec_ver}", f"python: {sys.version.split()[0]}",
                  f"argv: {shlex.join(getattr(args, '_argv', []))}",
                  "args: " + json.dumps({k: v for k, v in vars(args).items() if not k.startswith('_')},
                                        sort_keys=True, default=str)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        for line in meta_lines:
            f.write("# " + line.replace("\n", " ") + "\n")   # comment header: skip lines starting with '#'
        writer = csv.DictWriter(f, fieldnames=list(evals[0]["row"].keys()))
        writer.writeheader()
        writer.writerows(e["row"] for e in ordered)
    if best_ev is None or (not zin_window_met and args.strict_zin):
        least = min(evals, key=lambda e: (e["viol"], -e["score"]))
        zl = least["per_freq"][design_i]["zin"]
        p.error("no candidate satisfies the Zin constraints; the closest one has Zin "
                f"{zl.real:.1f}{zl.imag:+.1f}j ohm at {freqs[design_i]:g} MHz (violation {least['viol']:.3g} ohm over "
                "the band). Widen --zin-r-range / --zin-x-max, lower --vswr-weight, give --refine more rounds, or "
                "design a matching network afterwards (--mode match). The candidates and their exclusion reason are "
                f"in the zin_constraint and zin_violation_ohm columns of {csv_path}")
    best = dict(best_ev["row"])
    window_note = None
    if not zin_window_met:
        z_d = best_ev["per_freq"][design_i]["zin"]
        window_note = (f"no candidate meets the Zin window (R {r_min:g}..{r_max:g} ohm, |X| <= {args.zin_x_max:g} ohm "
                       f"at every band frequency); the closest one is reported and written: Zin {z_d.real:.1f}"
                       f"{z_d.imag:+.1f}j ohm at {freqs[design_i]:g} MHz, {best['zin_constraint']} (violation "
                       f"{best_ev['viol']:.3g} ohm summed over the band). Its score ignores the window. Widen "
                       "--zin-r-range / --zin-x-max or narrow --band to search inside a reachable window; "
                       "--strict-zin stops with an error instead")
        bar = "!" * 78
        print(f"\n{bar}\nWARNING: {window_note}.\n{bar}")
    best_pf = best_ev["per_freq"]
    output = Path(args.output or "hb9cv_optimized.nec").resolve()
    out_path = output.with_suffix(".out")
    if final and final["ok"]:
        output.write_text(final["deck_text"], encoding="ascii", errors="replace")
        out_path.write_text(final["out_text"], encoding="utf-8")
        write_freq_sidecar(out_path, [args.freq])
    else:
        why = final["reason"] if final else "no best candidate"
        print(f"WARNING: could not re-run the best model for the final files ({why}); writing the deck only.",
              file=sys.stderr)
        deck_text = build_deck(_variant(final_gargs(best_ev["cand"]), pattern_mode="ring", ring_theta=90.0,
                                        band_freqs=None, pt_mode=0))[0]
        output.write_text(deck_text, encoding="ascii", errors="replace")
    print("\nBEST CONFIGURATION OF THE SCAN" + (" (ranked by the 10th percentile score)" if mc_done else "")
          + ("" if zin_window_met else " (OUTSIDE THE Zin WINDOW, see the warning above)"))
    for k, v in best.items():
        print(f"{k}: {v}")
    print(f"Model: {output}\nNEC2C output of the best model: {out_path}\nCSV results: {csv_path}")
    freq = args.freq
    d_rear = element_sections(args, "rear", best["element_diameter_mm"])
    d_front = element_sections(args, "front", best["element_diameter_mm"])
    d_txt = (f"elements of {d_rear[0][1]:g} mm" if d_rear == d_front and len(d_rear) == 1
             else f"rear {sections_text(d_rear)}, front {sections_text(d_front)}")
    best_layout = layout_for(best_ev["cand"])
    best_sp_m = best_ev["cand"]["spacing"] * lam
    best_boom_mm = (max(args.boom_length, required_boom_mm(best_layout, -best_sp_m / 2, best_sp_m / 2,
                                                           args.boom_overhang_mm)) if best_layout else args.boom_length)
    print(f"Physical parameters of the model: {d_txt}; boom {args.boom_shape} "
          f"{args.boom_size:g} mm x {best_boom_mm:g} mm"
          + (" (lengthened to hold the reflectors/directors)" if best_boom_mm > args.boom_length + 1e-6 else ""))
    if args.model_feed_gap:
        print(f"Feed gap g = {args.gap_mm:g} mm modeled (bridge wire carries EX/TL).")
    li_best = {}
    try:
        li_best = build_deck(_variant(final_gargs(best_ev["cand"]), pattern_mode="ring", ring_theta=90.0,
                                      band_freqs=None, pt_mode=0))[5]
        for w in li_best.get("wire_warnings", []) + li_best.get("line_warnings", []):
            print(f"WARNING: {w}")
    except ValueError:
        pass
    print(f"Estimated rear length: {(150.0/freq)*best['rear_length_factor']*1000:.1f} mm")
    print(f"Estimated front length: {(139.0/freq)*best['front_length_factor']*1000:.1f} mm")
    print(f"Estimated spacing: {best['spacing_lambda']*C_LIGHT/(freq*1e6)*1000:.1f} mm")
    if tmatch:
        print(f"T-match on both driven elements: taps {best['tap_mm']:g} mm from each element center, bars "
              f"{args.tap_offset_mm:g} mm from the element toward the other one, "
              f"{args.tap_wire_mm or best['element_diameter_mm']:g} mm wire; the "
              "phase line and the feed connect at the bar centers.")
    if best.get("series_c_pf") not in ("", None):
        print(f"Series feed capacitor: {best['series_c_pf']:.2f} pF between the feed line and the fed port"
              + (" (chosen to cancel X_in at the design frequency)" if auto_c else "") + ".")
    if args.network == "tl" and best["line_length_m"] != "":
        dist_mm = feed_distance_m(args, best_sp_m) * 1000.0
        phys_mm = best["line_length_m"] * 1000.0
        print(f"Phase line: {phys_mm:.1f} mm for {dist_mm:.1f} mm between the feed points (slack {phys_mm - dist_mm:.1f} "
              f"mm for the connections and the crossing; --line-allowance-mm {args.line_allowance_mm:g}).")
        if phys_mm > 2.0 * dist_mm:
            print(f"WARNING: the phase line is {phys_mm / dist_mm:.1f} times the distance it spans: it has to be folded "
                  "or coiled, which the ideal NEC TL does not model (coupling to the elements, radiation). Lower "
                  "--max-line-deg" + (" or widen --tap-values" if tmatch else " or use --feed-style tmatch")
                  + " for a straight line.")
    if best_layout:
        print("\nREAL DIMENSIONS TO BUILD for the reflectors/directors (tip-to-tip NEC lengths; positions measured "
              "from the rear-most element):")
        for txt in parasitic_report(best_layout, li_best.get("boom_length_mm")):
            print(txt)
        print("The optimizer kept every length and spacing inside the minimum/maximum columns above.\n")
    gain_txt = (f"{best['forward_gain_dB']:.2f} dBi = {best['forward_gain_dBd']:.2f} dBd"
                if best["forward_gain_dBd"] != "" else
                f"{best['forward_gain_dB']:.2f} dBi including the ground reflection")
    print(f"Maximum gain of the best model: {best['max_gain_dB']:.2f} dB at phi={best['max_gain_angle_deg']:g} degrees "
          f"(gain at phi=0: {gain_txt}).")
    print(f"Take-off angle: {best['takeoff_deg']:g} deg above the horizon; azimuth half-power beamwidth: "
          f"{best['hpbw_az_deg']:.1f} deg; F/B at the take-off angle: {best['front_to_back_dB']:.2f} dB; "
          f"rear sector {best['front_to_back_sector_dB']:.2f} dB; rear cone {best['front_to_back_cone_dB']:.2f} dB.")
    print(f"F/B metrics: sector = gain(phi=0) minus the maximum of the rear sector {rear_start}..{rear_end} degrees on "
          f"the ring; cone = the same against every direction within {rear_cone_half_angle(rear_start, rear_end):g} "
          f"degrees of the back direction, at any elevation. The score uses: {args.fb_metric}.")
    if best["objective_term"] != "":
        if args.match_score:
            print(f"Score {best['score']:.2f} = objective term {best['objective_term']:.2f} ({args.objective}, after "
                  f"the matching network: loss up to {best['match_loss_dB']:.2f} dB over the band) - VSWR penalty "
                  f"{best['vswr_penalty'] or 0.0:.2f} dB (matched VSWR up to {best['matched_vswr_max']:.2f}:1).")
            if best["match_network"]:
                print(f"Matching network used by the score (ideal values, Q_L {args.match_q_l:g}, Q_C "
                      f"{args.match_q_c:g}): {best['match_network']}; matched gain {best['matched_gain_dB']:.2f} dBi "
                      f"at {freq:g} MHz. Design it with --mode match --nec-out on the output below.")
        else:
            print(f"Score {best['score']:.2f} = objective term {best['objective_term']:.2f} ({args.objective}"
                  + (", realized gain" if args.realized_gain and args.objective != "front-back" else "")
                  + f") - VSWR penalty {best['vswr_penalty'] or 0.0:.2f} dB"
                  + (" (the realized gain already discounts the mismatch once; the penalty weights matching further)"
                     if args.realized_gain and args.vswr_weight and args.objective != "front-back" else "") + ".")
        if single and best["zin_violation_ohm"] == 0 and best["Zin_real"] != "" and (
                abs(best["Zin_real"] - r_min) < 0.05 * r_min or abs(best["Zin_real"] - r_max) < 0.05 * r_max
                or abs(abs(best["Zin_imag"]) - args.zin_x_max) < 0.05 * args.zin_x_max):
            print(f"NOTE: the best Zin ({best['Zin_real']:.1f}{best['Zin_imag']:+.1f}j ohm) sits on the edge of the "
                  f"admissible window (R {r_min:g}..{r_max:g}, |X| <= {args.zin_x_max:g}): the optimum may lie "
                  "outside; widen --zin-r-range / --zin-x-max to check.")
    if args.band_freqs:
        print("\nBAND RESULTS of the best model (the worst frequency drives the score):")
        print(f"{'f (MHz)':>9s}{'gain dBi':>10s}{'gain dBd':>10s}{'F/B':>8s}{'F/B sect':>10s}{'F/B cone':>10s}"
              f"{'takeoff':>9s}{'HPBW':>7s}" + (f"{'VSWR':>8s}{'Zin (ohm)':>20s}" if single else "")
              + (f"{'matched':>9s}" if single and args.match_score else ""))
        for x in best_pf:
            dbd = f"{x['gain_dbd']:10.2f}" if x["gain_dbd"] is not None else f"{'-':>10s}"
            line = (f"{x['freq_mhz']:9.3f}{x['forward_gain']:10.2f}{dbd}{x['fb']:8.2f}"
                    f"{x['robust_fb']:10.2f}{x['cone_fb']:10.2f}{x['takeoff_deg']:9.1f}{x['hpbw']:7.1f}")
            if single:
                line += f"{x['vswr']:8.2f}{x['zin'].real:11.2f}{x['zin'].imag:+9.2f}j"
                if args.match_score:
                    line += f"{x.get('m_vswr', math.inf):9.2f}"
            print(line)
    if single and abs_coax is None and best["line_theory_offset_deg"] != "" and not args.refine \
            and best["line_theory_offset_deg"] in (min(line_offsets), max(line_offsets)) and len(line_offsets) > 1:
        print("WARNING: the best result is at the edge of the line scan "
              f"({best['line_theory_offset_deg']:+g} degrees from theory); the optimum may be outside. "
              "Widen --line-offset-values.")
    if not single and abs_phases is None and len(phase_offsets) > 1 \
            and best["rear_phase_deg"] != "":
        theory = theory_rear_phase(best["spacing_lambda"])
        off = wrap180(best["rear_phase_deg"] - theory)
        if abs(off - min(phase_offsets)) < 1e-6 or abs(off - max(phase_offsets)) < 1e-6:
            print("WARNING: the best phase is at the edge of the scan; widen --phase-offset-values.")
    if args.network == "lc":
        print(f"Ideal LC pi network {'crossed' if args.crossed else 'straight'} fed at the {args.feed_element} element: "
              f"Zc={args.lc_z0:g} ohm, series L={best['lc_l_nh']:.3f} nH, C per port={best['lc_c_pf']:.3f} pF "
              f"({best['line_electrical_deg']:.1f} degrees).")
    if args.network == "tl":
        kind_txt = "open-wire (balanced)" if args.line_type == "openwire" else "coaxial"
        print(f"Ideal {kind_txt} TL line {'crossed' if args.crossed else 'straight'} fed at the {args.feed_element} "
              f"element: |Z0|={best['line_z0_ohm']:g} ohm, VF of the line={args.coax_vf:g}, "
              f"physical length={best['line_length_m']:.5f} m ({best['line_electrical_deg']:.1f} electrical degrees), "
              f"NEC TL card length (physical/VF)={best['nec_tl_length_m']:.5f} m.")
        if args.line_type == "openwire":
            print(f"Open-wire geometry: wire d={args.line_wire_diameter:g} mm, center spacing "
                  f"D={best['line_spacing_mm']:.2f} mm.")
        else:
            print("WARNING: a coax phasing section needs a balun/choke at its element end; the NEC TL does not "
                  "model shield/common-mode currents.")
    if single:
        if best["Zin_real"] != "":
            print(f"NEC Zin of the best model at {freq:g} MHz: {best['Zin_real']:.3f}{best['Zin_imag']:+.3f}j ohm; "
                  f"VSWR vs {args.target_impedance:g} ohm = {best['VSWR_vs_target']:.2f}:1; "
                  f"realized gain = {best['realized_gain_dB']:.2f} dB")
        try:
            res_est = resonance_estimate(exe, final_gargs(best_ev["cand"]), rear_start, rear_end)
        except (ValueError, OSError) as exc:
            res_est = None
            print(f"WARNING: resonance estimate failed: {exc}", file=sys.stderr)
        if res_est is not None:
            f_res, x_in, slope = res_est
            print(f"|X_in| = {abs(x_in):.2f} ohm at {freq:g} MHz; series resonance (X_in = 0) estimated at "
                  f"{f_res:.3f} MHz ({f_res - freq:+.3f} MHz from the design frequency; dX/df {slope:.1f} ohm/MHz).")
        vmax = max(x["vswr"] for x in best_pf)
        if vmax > 2.0 and args.match_score:
            print(f"NOTE: the raw VSWR of the best model is {vmax:.2f}:1 vs {args.target_impedance:g} ohm: it is meant to "
                  "be fed through the matching network above (scored with its loss); --no-match-score searches "
                  "for a direct connection instead.")
        elif vmax > 2.0:
            bar = "!" * 78
            print(f"\n{bar}\nWARNING: the best result has VSWR {vmax:.2f}:1 vs {args.target_impedance:g} ohm"
                  + (" somewhere in the band" if args.band_freqs else "") + " (> 2:1).\n"
                  "It needs a matching network, or loosen --zin-r-range/--zin-x-max, raise --vswr-weight, widen the\n"
                  f"scan or enable --refine. Mismatch loss at that VSWR: "
                  f"{-10 * math.log10(1 - ((vmax - 1) / (vmax + 1)) ** 2):.2f} dB.\n{bar}\n")
        print((("Note: ideal LC pi network between elements; no component losses or parasitics. "
                if args.network == "lc" else
                "Note: ideal TL between elements; it includes no cable losses or radiation. ")
               + "For the matching network use --mode match --nec-out with the NEC2C output indicated above."))
    else:
        print("Note: optimization with ideal sources; it does not replace the design and simulation of a physical "
              "network, and with two EX sources there is no single Zin to design a 50 ohm network.")
    print("WARNING: " + BALUN_REQUIREMENT.format(f=freq))
    if single:
        if args.balun_ratio == "4:1":
            print("NOTE: " + BALUN_4_1_NOTE.format(zb=args.target_impedance, zu=50.0))
        if args.vswr_weight == 0:
            print("NOTE: the optimizer is not driving Zin to the target (--vswr-weight 0 = report only); a weight "
                  "above 0 (and a tight --zin-x-max) makes the search prefer designs that need little or no "
                  "matching network.")

    if out_path.exists() and li_best.get("ports"):
        try:
            print_element_currents(out_path, ports=li_best["ports"])
        except (ValueError, OSError) as exc:
            print(f"NOTE: element-current report not available: {exc}", file=sys.stderr)
    loss_bs = conductor_loss_check(exe, final_gargs(best_ev["cand"]), rear_start, rear_end)
    agt_bs = average_gain_test(exe, final_gargs(best_ev["cand"]))
    conv = convergence_check(exe, final_gargs(best_ev["cand"]), rear_start, rear_end) if args.check_convergence else None
    robust = robustness_check(exe, final_gargs(best_ev["cand"]), rear_start, rear_end) if args.robustness else None
    if args.build_sheet and single:
        zbest = (complex(best["Zin_real"], best["Zin_imag"]) if best["Zin_real"] != "" else None)
        write_build_sheet(args.build_sheet, args,
                          {"rear_mm": (150.0 / freq) * best["rear_length_factor"] * 1000,
                           "front_mm": (139.0 / freq) * best["front_length_factor"] * 1000,
                           "spacing_mm": best["spacing_lambda"] * C_LIGHT / (freq * 1e6) * 1000,
                           "diam_mm": best["element_diameter_mm"],
                           "parasitics": best_layout, "boom_length_mm": li_best.get("boom_length_mm", args.boom_length),
                           "boom_xc_m": li_best.get("boom_xc_m", 0.0),
                           "tap_mm": best["tap_mm"] if best["tap_mm"] != "" else args.tap_mm,
                           "series_c_pf": best["series_c_pf"],
                           "match_net": best_pf[design_i].get("m_net"),
                           "extra_warnings": [window_note] if window_note else []},
                          describe_line(args, best["line_electrical_deg"], best["line_length_m"] or 0.0,
                                        best["line_z0_ohm"] or None,
                                        best["line_spacing_mm"] if best["line_spacing_mm"] != "" else None,
                                        best["lc_l_nh"] if best["lc_l_nh"] != "" else None,
                                        best["lc_c_pf"] if best["lc_c_pf"] != "" else None), zbest,
                          {"design": best_pf[design_i], "band": list(best_pf) if args.band_freqs else [best_pf[design_i]],
                           "mc": ({"samples": best.get("mc_samples"), "score_p10": best["score_p10"],
                                   "fb_p10": best["fb_p10"], "vswr_p90": best.get("vswr_p90"),
                                   "tolerances": {"length_mm": args.tol_length_mm, "spacing_mm": args.tol_spacing_mm,
                                                  "line_deg": args.tol_line_deg, "seed": args.tol_seed}}
                                  if isinstance(best.get("score_p10"), float) else None),
                           "loss": loss_bs, "agt": agt_bs},
                          {"elec_deg": best["line_electrical_deg"], "phys_m": best["line_length_m"] or 0.0,
                           "z0": best["line_z0_ohm"] or None,
                           "spacing_mm": best["line_spacing_mm"] if best["line_spacing_mm"] != "" else None,
                           "lc_l_nh": best["lc_l_nh"] if best["lc_l_nh"] != "" else None,
                           "lc_c_pf": best["lc_c_pf"] if best["lc_c_pf"] != "" else None})
    elif args.build_sheet:
        print("WARNING: --build-sheet needs --network tl or lc; not written.", file=sys.stderr)
    if args.cm_sensitivity:
        test_len = args.feedline_length if args.feedline_length > 0 else lam / 2.0
        try:
            cm_sensitivity(exe, final_gargs(best_ev["cand"]), test_len, rear_start, rear_end)
        except (ValueError, OSError) as exc:
            print(f"WARNING: common-mode sensitivity run failed: {exc}", file=sys.stderr)
    if args.mast_sensitivity:
        try:
            mast_sensitivity(exe, final_gargs(best_ev["cand"]), args.mast_sensitivity, rear_start, rear_end)
        except (ValueError, OSError) as exc:
            print(f"WARNING: mast sensitivity run failed: {exc}", file=sys.stderr)
    write_json_out(args, {"deck": str(output), "out": str(out_path), "csv": str(csv_path), "freq_mhz": freq,
                          "zin_window_met": zin_window_met,
                          "best": best, "design": best_pf[design_i], "band": best_pf if args.band_freqs else None,
                          "ports": li_best.get("ports"), "conductor_loss": loss_bs, "average_gain_test": agt_bs,
                          "convergence": conv,
                          "robustness": robust,
                          "warnings": li_best.get("wire_warnings", []) + li_best.get("line_warnings", [])})


# ======================================================================================
# L-network matching at a single frequency
# ======================================================================================

def lnet_solution_note(which, net):
    """Note when --lnet-solution lowpass/highpass found no solution with those branch kinds."""
    want = {"lowpass": "low-pass", "highpass": "high-pass"}.get(which)
    if want and lnet_kind_text(net) != want:
        return (f"NOTE: no {want} solution exists for this load; the {'plus' if which == 'lowpass' else 'minus'} root "
                f"({lnet_kind_text(net)}) is used.")
    return None


def lnet_balanced_lines(net):
    """Describe an L-network as a balanced network: series elements split between the two legs,
    shunt element across the line."""
    s_kind, s_val = net["series"]
    h_kind, h_val = net["shunt"]
    if s_val == 0:
        series = "Series branch: none (straight through in both legs)"
    elif s_kind == "L":
        series = f"Series branch: 2 x inductor {s_val * 1e9 / 2:.3f} nH, one in each leg (total {s_val * 1e9:.3f} nH)"
    else:
        series = (f"Series branch: 2 x capacitor {s_val * 1e12 * 2:.3f} pF, one in each leg "
                  f"(total {s_val * 1e12:.3f} pF in series)")
    if h_val == 0:
        shunt = "Shunt branch: none"
    else:
        shunt = f"Shunt branch: 1 x {lnet_describe(net['shunt'])} connected across the two legs"
    place = ("series halves next to the antenna terminals, shunt element across the line on the balun side"
             if net["topology"] == "A" else
             "shunt element across the antenna terminals, then the series halves in each leg towards the balun")
    return [series, shunt, f"Placement: {place}; the balun/choke follows the network"]


def run_match(p, args):
    manual = args.r is not None or args.x is not None
    apply_balun_ratio(args)
    if args.nec_out and manual:
        p.error("use --nec-out or --r/--x, not both")
    if not args.nec_out and not (args.r is not None and args.x is not None):
        p.error("give --nec-out file.out, or --r and --x together; there is no default impedance")

    table = []
    if args.nec_out:
        try:
            table = single_source_zin(args.nec_out)
            if args.freq is None:
                if len(table) != 1:
                    p.error("the file contains several frequencies; give --freq")
                freq, z = table[0]
            else:
                freq, z = pick_frequency(table, args.freq)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        source = f"NEC ({Path(args.nec_out).name})"
    else:
        freq = args.freq if args.freq is not None else 145.0
        z = complex(args.r, args.x)
        source = "entered manually"
    if freq <= 0:
        p.error("the frequency must be greater than zero")

    w = 2 * math.pi * freq * 1e6
    gamma = (z - args.z0) / (z + args.z0)
    rho = abs(gamma)
    vswr = (1 + rho) / (1 - rho) if rho < 1 else float("inf")
    rl = -20 * math.log10(rho) if rho else float("inf")
    reflected = rho ** 2

    try:
        net, sols = lnet_select(z, args.z0, w, args.lnet_solution, table)
    except ValueError as exc:
        p.error(str(exc))
    z_input = lnet_zin(z, w, net)
    vswr_after, _, _ = lnet_metrics(z_input, args.z0)

    print(f"Frequency: {freq:.6f} MHz")
    print(f"Impedance {source}: {z.real:.4f} {z.imag:+.4f}j ohm")
    print(f"Before: |Gamma|={rho:.6f}; VSWR={vswr:.3f}:1; return loss={rl:.2f} dB; reflected power={reflected*100:.1f}%")
    print("\nIdeal L-network computed (mathematical values, not final components):")
    print(f"  Topology {lnet_topology_text(net)}")
    print(f"  Series branch: {lnet_describe(net['series'])}")
    print(f"  Shunt branch: {lnet_describe(net['shunt'])}")
    print(f"  Solution: {lnet_kind_text(net)} (--lnet-solution {args.lnet_solution})")
    if lnet_solution_note(args.lnet_solution, net):
        print(f"  {lnet_solution_note(args.lnet_solution, net)}")
    for alt in sols:
        if alt is not net:
            print(f"  Other solution ({lnet_kind_text(alt)}): series {lnet_describe(alt['series'])}, "
                  f"shunt {lnet_describe(alt['shunt'])}")
    for w_txt in srf_warnings(net, freq, freq, args.esl_nh, args.coil_cp_pf):
        print(f"  {w_txt}")
    if args.balanced:
        print("\nBalanced version (symmetric legs):")
        for line in lnet_balanced_lines(net):
            print(f"  {line}")
        print("WARNING: above ~100 MHz lead inductance, pad capacitance, self-resonance and layout dominate "
              "component values of this size; treat these as a starting point and validate/tune with a VNA on "
              "the built network.")
    print(f"After, ideal: Zin={z_input.real:.4f} {z_input.imag:+.4f}j ohm; VSWR={vswr_after:.4f}:1")
    if args.q_l is not None or args.q_c is not None:
        real = lnet_with_q(net, args.q_l, args.q_c)
        z_real = lnet_zin(z, w, real)
        vswr_real, _, _ = lnet_metrics(z_real, args.z0)
        mism, diss, _ = lnet_losses(z, w, real, args.z0)
        print(f"With ESR (Q_L={args.q_l}, Q_C={args.q_c}) without readjusting values: "
              f"Zin={z_real.real:.4f} {z_real.imag:+.4f}j ohm; VSWR={vswr_real:.4f}:1; "
              f"mismatch={mism:.3f} dB; dissipation={diss:.3f} dB ({(1 - 10 ** (-diss / 10)) * 100:.1f}% of the power)")
    print("\nIMPORTANT: single-frequency lumped calculation. It does not include parasitics, physical circuit")
    print("coupling, nor does it recompute the NEC pattern. To see the bandwidth use --mode sweep with a NEC")
    print("sweep (--mode generate --fsweep) and validate with an antenna analyzer/VNA.")

    # When the .out file holds a frequency sweep, show the VSWR of the network across it.
    preview = []
    if args.nec_out and len(table) >= 2:
        print("\nBANDWIDTH PREVIEW (ideal L-network values, Zin from the NEC sweep):")
        vswr_rows = []
        for f_mhz, z_load in table:
            w_f = 2 * math.pi * f_mhz * 1e6
            zi_f = lnet_zin(z_load, w_f, net)
            vswr_f, _, _ = lnet_metrics(zi_f, args.z0)
            vswr_rows.append((f_mhz, vswr_f))
        preview = vswr_rows
        print(f"  {'f (MHz)':>9s}  {'VSWR':>7s}  {'Zload (ohm)':>22s}  {'Zin (ohm)':>22s}")
        for f_mhz, z_load in table:
            w_f = 2 * math.pi * f_mhz * 1e6
            zi_f = lnet_zin(z_load, w_f, net)
            vswr_f, _, _ = lnet_metrics(zi_f, args.z0)
            print(f"  {f_mhz:9.3f}  {vswr_f:7.3f}  "
                  f"{z_load.real:10.3f}{z_load.imag:+.3f}j  "
                  f"{zi_f.real:10.3f}{zi_f.imag:+.3f}j")
        # Contiguous bandwidth at SWR <= 2:1 around the best point.
        span2 = contiguous_vswr_span(vswr_rows)
        if span2:
            a, b, edge = span2
            print(f"  VSWR <= 2:1 bandwidth: {a:.3f}–{b:.3f} MHz ({b - a:.3f} MHz span)"
                  + (" (reaches the end of the sweep: widen it)" if edge else ""))
        else:
            vmin = min(v for _, v in vswr_rows)
            print(f"  VSWR never reaches 2:1 within the sweep (min VSWR = {vmin:.2f}:1); "
                  "use --mode sweep for a wider range.")
    write_json_out(args, {"freq_mhz": freq, "zload": z, "z0_ohm": args.z0, "source": source,
                          "topology": net["topology"], "topology_text": lnet_topology_text(net),
                          "series": lnet_describe(net["series"]), "shunt": lnet_describe(net["shunt"]),
                          "solution": lnet_kind_text(net),
                          "alternatives": [{"series": lnet_describe(a["series"]), "shunt": lnet_describe(a["shunt"]),
                                            "solution": lnet_kind_text(a)} for a in sols if a is not net],
                          "vswr_before": vswr, "vswr_after": vswr_after, "zin_after": z_input,
                          "preview": [{"freq_mhz": f_, "vswr": v_} for f_, v_ in preview]})


# ======================================================================================
# L-network frequency sweep
# ======================================================================================

def parse_pairs(items):
    out = []
    for item in items or []:
        try:
            v1, v2 = [float(x) for x in item.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(f"--commercial must be SERIES,SHUNT (received: {item})")
        if v1 < 0 or v2 < 0:
            raise argparse.ArgumentTypeError("--commercial requires non-negative values")
        out.append((v1, v2))
    return out


def run_sweep(p, args):
    apply_balun_ratio(args)
    try:
        pairs = parse_pairs(args.commercial)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))

    if args.nec_out and args.assume_constant_z:
        p.error("use --nec-out or --assume-constant-z, not both")
    if args.assume_constant_z:
        if args.r is None or args.x is None:
            p.error("--assume-constant-z requires --r and --x")
        if min(args.fmin, args.step) <= 0 or args.fmax < args.fmin:
            p.error("invalid frequency parameters")
        if not args.fmin <= args.design_freq <= args.fmax:
            p.error("--design-freq must be within --fmin/--fmax")
        n = math.floor((args.fmax - args.fmin) / args.step + 1e-9)
        z_const = complex(args.r, args.x)
        freqs = [args.fmin + i * args.step for i in range(n + 1)]
        if args.fmax - freqs[-1] > 1e-6 * args.fmax:
            freqs.append(args.fmax)   # the range is not a multiple of the step: fmax is included
        table = [(f, z_const) for f in freqs]
        design_z = z_const
        print("WARNING: constant antenna Z over the whole sweep. The curve reflects only the L-network;")
        print("         do not use it to judge bandwidth or the frequency of minimum VSWR.")
    elif args.nec_out:
        try:
            table = single_source_zin(args.nec_out)
            _, design_z = pick_frequency(table, args.design_freq)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        if len(table) < 2:
            p.error("the .out contains a single frequency; generate the model with --fsweep FMIN,FMAX,STEP")
    else:
        p.error("give --nec-out file.out (recommended) or --assume-constant-z with --r and --x")

    w0 = 2 * math.pi * args.design_freq * 1e6
    try:
        ideal, sols = lnet_select(design_z, args.z0, w0, args.lnet_solution, table)
    except ValueError as exc:
        p.error(str(exc))
    if not pairs:
        pairs = [(nearest_e24(lnet_display(ideal["series"])) if ideal["series"][1] > 0 else 0.0,
                  nearest_e24(lnet_display(ideal["shunt"])) if ideal["shunt"][1] > 0 else 0.0)]
    networks = [("ideal", ideal)]
    for v1, v2 in pairs:
        net = lnet_with_q(lnet_with_values(ideal, v1, v2), args.q_l, args.q_c)
        networks.append((f"{v1:g}{lnet_unit(net['series'])}_{v2:g}{lnet_unit(net['shunt'])}", net))

    print(f"Antenna Z at {args.design_freq:g} MHz used for the design: "
          f"{design_z.real:.4f}{design_z.imag:+.4f}j ohm")
    print(f"Topology {lnet_topology_text(ideal)}")
    print(f"Ideal values: series branch = {lnet_describe(ideal['series'])}; "
          f"shunt branch = {lnet_describe(ideal['shunt'])} ({lnet_kind_text(ideal)}, --lnet-solution "
          f"{args.lnet_solution})")
    if lnet_solution_note(args.lnet_solution, ideal):
        print(lnet_solution_note(args.lnet_solution, ideal))
    for alt in sols:
        if alt is not ideal:
            print(f"Other solution ({lnet_kind_text(alt)}): series {lnet_describe(alt['series'])}, "
                  f"shunt {lnet_describe(alt['shunt'])}")
    for label, net in networks:
        zi = lnet_zin(design_z, w0, net)
        vswr, rl, _ = lnet_metrics(zi, args.z0)
        mism, diss, _ = lnet_losses(design_z, w0, net, args.z0)
        print(f"{label}: Zin={zi.real:.3f}{zi.imag:+.3f}j ohm; VSWR={vswr:.3f}:1; return loss={rl:.2f} dB; "
              f"mismatch loss={mism:.3f} dB; dissipation={diss:.3f} dB")

    rows = []
    for f_mhz, z_load in table:
        w = 2 * math.pi * f_mhz * 1e6
        for label, net in networks:
            zi = lnet_zin(z_load, w, net)
            vswr, rl, rho = lnet_metrics(zi, args.z0)
            mism, diss, tot = lnet_losses(z_load, w, net, args.z0)
            rows.append({
                "frequency_MHz": f"{f_mhz:.3f}",
                "network": label,
                "topology": net["topology"],
                "series_kind": net["series"][0],
                "series_value": f"{lnet_display(net['series']):.4f}",
                "series_unit": lnet_unit(net["series"]),
                "shunt_kind": net["shunt"][0],
                "shunt_value": f"{lnet_display(net['shunt']):.4f}",
                "shunt_unit": lnet_unit(net["shunt"]),
                "Zload_real_ohm": f"{z_load.real:.5f}",
                "Zload_imag_ohm": f"{z_load.imag:.5f}",
                "Zin_real_ohm": f"{zi.real:.5f}",
                "Zin_imag_ohm": f"{zi.imag:.5f}",
                "VSWR": f"{vswr:.5f}",
                "return_loss_dB": f"{rl:.3f}",
                "gamma_magnitude": f"{rho:.6f}",
                "mismatch_loss_dB": f"{mism:.4f}",
                "dissipation_loss_dB": f"{diss:.4f}",
                "total_loss_dB": f"{tot:.4f}",
            })
    with open(args.sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for label, _ in networks:
        sel = [r for r in rows if r["network"] == label]
        best = min(sel, key=lambda r: float(r["VSWR"]))
        print(f"Minimum VSWR of {label}: {best['VSWR']}:1 at {best['frequency_MHz']} MHz")
    print(f"CSV saved: {Path(args.sweep_csv).resolve()}")
    if args.q_l is None and args.q_c is None:
        print("Warning: this neither changes nor simulates the NEC file; the L-network is ideal (no losses or "
              "parasitics); use --q-l/--q-c to include the ESR of the components.")
    else:
        print("Warning: this neither changes nor simulates the NEC file; only the ESR (constant Q) of the "
              "commercial networks is modeled; the 'ideal' network is still free of losses and parasitics.")

    # ---- Tolerance box: worst-case VSWR over the four corners of the component tolerances ----
    tol_l = getattr(args, "tol_l", 0.0) or 0.0
    tol_c = getattr(args, "tol_c", 0.0) or 0.0
    if tol_l > 0 or tol_c > 0:
        print(f"\nTOLERANCE ANALYSIS (L ±{tol_l:g}%, C ±{tol_c:g}%)")
        corners = list(itertools.product((-1, +1), repeat=2))   # (sign_series, sign_shunt)
        for label, net in networks:
            worst_by_freq = []
            for f_mhz, z_load in table:
                w_f = 2 * math.pi * f_mhz * 1e6
                worst_vswr = 0.0
                for ds, dh in corners:
                    s_kind, s_val = net["series"]
                    h_kind, h_val = net["shunt"]
                    # Apply tolerance: L values in H, C values in F.
                    tol_s = (tol_l if s_kind == "L" else tol_c) / 100.0
                    tol_h = (tol_l if h_kind == "L" else tol_c) / 100.0
                    s_val_c = s_val * (1.0 + ds * tol_s) if s_val > 0 else s_val
                    h_val_c = h_val * (1.0 + dh * tol_h) if h_val > 0 else h_val
                    net_c = lnet_with_q({"topology": net["topology"],
                                         "series": (s_kind, s_val_c),
                                         "shunt": (h_kind, h_val_c)},
                                        net.get("q_l"), net.get("q_c"))
                    zi_c = lnet_zin(z_load, w_f, net_c)
                    vswr_c, _, _ = lnet_metrics(zi_c, args.z0)
                    worst_vswr = max(worst_vswr, vswr_c)
                worst_by_freq.append((f_mhz, worst_vswr))
            # Print a compact summary.
            wf_max = max(worst_by_freq, key=lambda x: x[1])
            span2 = contiguous_vswr_span(worst_by_freq)
            bw_str = (f"{span2[0]:.3f}–{span2[1]:.3f} MHz" + (" (reaches the end of the sweep)" if span2[2] else "")
                      if span2 else "none")
            print(f"  {label}: worst-case VSWR = {wf_max[1]:.2f}:1 at {wf_max[0]:.3f} MHz; "
                  f"VSWR<=2 window: {bw_str}")


    fmin_t, fmax_t = min(f for f, _ in table), max(f for f, _ in table)
    for label, net in networks:
        for w_txt in srf_warnings(net, fmin_t, fmax_t, args.esl_nh, args.coil_cp_pf):
            print(f"  {label}: {w_txt}")
    write_json_out(args, {"design_freq_mhz": args.design_freq, "z0_ohm": args.z0, "zload_design": design_z,
                          "csv": str(Path(args.sweep_csv).resolve()),
                          "networks": [{"label": label, "topology": net["topology"],
                                        "series": lnet_describe(net["series"]), "shunt": lnet_describe(net["shunt"])}
                                       for label, net in networks]})


# ======================================================================================
# Command line
# ======================================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Each option lists the modes in which it is used; options given in a mode that "
               "does not use them are reported and ignored.\n"
               "Example: python3 hb9cv.py --mode generate --network tl --mount isolated "
               "--freq 145 -o hb9cv_145.nec")
    table = {}   # option string -> (dest, modes)

    def opt(modes, *names, **kw):
        if kw.get("help"):
            kw["help"] = f"{kw['help']} [modes: {', '.join(modes)}]"
        action = p.add_argument(*names, **kw)
        for name in names:
            table[name] = (action.dest, set(modes))
        return action

    G, O, M, S = MODES
    p.add_argument("--mode", choices=MODES, required=True,
                   help="generate: write a NEC model; optimize: scan parameters with nec2c; "
                        "match: single-frequency L-network; sweep: L-network frequency sweep")

    # ---- shared by generate / optimize / match
    opt((G, O, M), "--freq", type=positive_float, default=None,
        help="frequency in MHz. generate/optimize: default 145. match: with --nec-out it must exist in the "
             "file (default: the only one it contains), without it the default is 145")
    # ---- model options (generate / optimize)
    opt((G, O), "--objective", choices=["front-back", "gain", "balance"], default="front-back",
        help="generate: INFORMATIVE ONLY (written to a CM comment; no other effect on the deck); "
             "optimize: scoring criterion")
    opt((G, O), "--mount", choices=["isolated", "contact"], default="isolated",
        help="isolated (default): elements insulated from the boom. contact: element centers bonded to a metallic "
             "boom (modeled, with a short bond wire per element standing for the clamp); only with --feed-style "
             "tmatch, where the element centers are at a voltage null. With a center feed the bond would short the "
             "feed points, so contact falls back to '--mount isolated --boom-isolated'")
    opt((G, O), "--boom-clearance-mm", type=nonneg_float, default=None, metavar="MM",
        help=f"surface-to-surface gap in mm between the elements (and T-bars) and a modeled boom (default "
             f"{DEFAULT_BOOM_CLEARANCE_MM:g}); the common-mode shield wire runs beside the boom at the same gap. "
             "Conductors much closer than their radii couple inaccurately in NEC-2")
    opt((G, O), "--element-diameter", type=positive_float, default=10.0,
        help="diameter in mm of both elements (optimize --refine-diameter moves it); --rear-diameter, "
             "--front-diameter and the section lists override it per element")
    opt((G, O), "--rear-diameter", type=positive_float, default=None, metavar="MM",
        help="rear element diameter in mm (default: --element-diameter)")
    opt((G, O), "--front-diameter", type=positive_float, default=None, metavar="MM",
        help="front element diameter in mm (default: --element-diameter)")
    opt((G, O), "--rear-sections", type=sections_list, default=None, metavar="L:D,...,D",
        help="stepped rear element (telescoped tubes), ONE half, from the center outward: LENGTH_MM:DIAMETER_MM per "
             "section, the last section runs to the tip and takes only its DIAMETER_MM, e.g. 300:12,200:10,8; both "
             "halves are alike; the lengths are measured from the element center (the half gap included); "
             "needs --mount isolated; excludes --rear-diameter")
    opt((G, O), "--front-sections", type=sections_list, default=None, metavar="L:D,...,D",
        help="stepped front element, same format as --rear-sections")
    opt((G, O), "--reflectors", type=int, default=0, metavar="N",
        help=f"number of passive reflectors behind the rear element (0..{MAX_REFLECTORS}); their length and "
             "spacing are pre-calculated (minimum / nominal / maximum) and, in optimize, searched within those "
             "limits")
    opt((G, O), "--directors", type=int, default=0, metavar="N",
        help=f"number of passive directors in front of the front element (0..{MAX_DIRECTORS}); same treatment "
             "as --reflectors")
    opt((G, O), "--reflector-lengths", default=None, metavar="F1,F2",
        help="reflector lengths as factors of the resonant dipole length of the parasitic tube (see "
             "--parasitic-diameter); missing values use the pre-calculated nominal (1.05). generate: the value "
             "used; optimize: the starting point of the search")
    opt((G, O), "--reflector-spacings", default=None, metavar="S1,S2",
        help="reflector spacings in wavelengths, each from the previous element (the first from the rear "
             "element); missing values use the nominal (0.15, 0.20)")
    opt((G, O), "--director-lengths", default=None, metavar="F1,...,F5",
        help="director lengths as factors of the resonant dipole length; missing values use the nominal "
             "(0.96, 0.95, 0.94, 0.93, 0.92)")
    opt((G, O), "--director-spacings", default=None, metavar="S1,...,S5",
        help="director spacings in wavelengths, each from the previous element (the first from the front "
             "element); missing values use the nominal (0.15, 0.20, 0.25, 0.30, 0.30)")
    opt((G, O), "--parasitic-diameter", type=positive_float, default=None, metavar="MM",
        help="tube diameter in mm of the reflectors and directors (default: --element-diameter at the start; "
             "it is not moved by --refine-diameter)")
    opt((G, O), "--min-gap-mm", type=nonneg_float, default=DEFAULT_MIN_GAP_MM,
        help="smallest surface-to-surface gap in mm between neighbouring elements (room for clamps and "
             "insulators); with the 0.05 lambda electrical floor it sets the minimum spacing of the parasitics")
    opt((G, O), "--boom-overhang-mm", type=nonneg_float, default=DEFAULT_BOOM_OVERHANG_MM,
        help="boom length beyond the outer elements at each end, in mm; with reflectors/directors the boom is "
             "lengthened automatically to hold every element (the longer of --boom-length and the span + 2 x overhang)")
    opt((G, O), "--boom-shape", choices=["square", "round"], default="square", help="nominal boom section")
    opt((G, O), "--boom-size", type=positive_float, default=25.0, help="boom dimension/diameter in mm")
    opt((G, O), "--boom-length", type=positive_float, default=700.0, help="boom length in mm")
    opt((G, O), "--boom-isolated", action="store_true",
        help="with --mount isolated, model the boom as a metallic conductor with no electrical connection "
             "(below the elements); by default it is not modeled")
    opt((G, O), "-o", "--output", default=None,
        help="output NEC file (default: generate: derived from the parameters; optimize: hb9cv_optimized.nec)")
    opt((G, O), "--network", choices=["tl", "lc", "ideal", "coaxial"], default="tl",
        help="tl: one EX + ideal balanced TL between centers (see --line-type); coaxial: deprecated alias of tl; lc: one EX + ideal LC pi network (NT card); "
             "ideal: two EX sources with prescribed (generate) or scanned (optimize) amplitude/phase")
    opt((G, O), "--feed-element", choices=["front", "rear"], default="front",
        help="element that receives the single EX source in --network tl/lc (classic HB9CV: front)")
    opt((G, O), "--crossed", action=argparse.BooleanOptionalAction, default=None,
        help="crossed phase line or network (negative Z0 in TL; inverted Y12 in NT); --no-crossed for straight. "
             "Default: crossed, except --line-type coax (always straight; --crossed there is an error)")
    opt((G, O), "--line-type", choices=["openwire", "coax"], default=None,
        help="physical phase line of --network tl. openwire (default): balanced air line, crossed by construction, "
             "VF 0.95, Z0 computed from geometry; coax: coaxial cable, always straight, VF 0.66, Z0 from --line-z0")
    opt((G, O), "--line-wire-diameter", type=positive_float, default=None,
        help="open-wire line: wire diameter d in mm (default 3)")
    opt((G, O), "--line-spacing", type=positive_float, default=None,
        help="open-wire line: wire center spacing D in mm (default 20); Z0 = 120*acosh(D/d)")
    opt((G, O), "--line-z0", "--coax-z0", dest="coax_z0", type=positive_float, default=None,
        help="Z0 of the TL in ohms (coax line only, default 50; for openwire it is computed from the geometry). "
             "--coax-z0 is a deprecated alias of --line-z0")
    opt((G, O), "--line-vf", "--coax-vf", dest="coax_vf", type=positive_float, default=None,
        help="velocity factor of the phase line (0<VF<=1); default 0.95 openwire, 0.66 coax. "
             "--coax-vf is a deprecated alias of --line-vf")
    opt((G, O), "--lc-z0", type=positive_float, default=50.0,
        help="characteristic impedance of the LC pi network in ohms (--network lc)")
    opt((G, O), "--segments", type=int, default=None,
        help=f"segments per element (odd, >= 5); default {DEFAULT_SEGMENTS}, {DEFAULT_SEGMENTS_TMATCH} with "
             "--feed-style tmatch")
    opt((G, O), "--az-step", type=positive_float, default=1.0,
        help="azimuth pattern step in degrees (>= 0.01: NEC prints 2 decimals); must divide 360 exactly; "
             "small values increase run time")
    opt((G, O), "--target-impedance", type=positive_float, default=None,
        help="target impedance in ohms (balanced side). Default 50, or 12.5 with --balun-ratio 4:1. "
             "generate: INFORMATIVE ONLY; optimize: reference for VSWR_vs_target")
    opt((G, O), "--feedline-length-m", dest="feedline_length", type=nonneg_float, default=0.0,
        help="common-mode test: length in m of a wire for the coax outer conductor, tied to the feed element "
             "through a source gap and running rearward below the elements (0 = off; --network tl/lc, isolated)")
    opt((G, O), "--feedline-diameter", type=positive_float, default=5.0,
        help="diameter in mm of the feed-line wire (RG-58 is about 5 mm)")
    opt((G, O), "--cm-sensitivity", action="store_true",
        help="simulate the model without and with the feed-line wire (length --feedline-length-m, default "
             "lambda/2) and report the change in F/B and Zin (needs nec2c; generate: the model given; "
             "optimize: the best model)")
    opt((G, O), "--mast-length-m", type=nonneg_float, default=0.0, metavar="M",
        help="model a mast: straight conductor of this length (m). Horizontal elements: hanging below the boom "
             "center, perpendicular to the boom and to the elements (moved sideways by --mast-offset-mm). Vertical "
             "elements: side mount, parallel to the elements behind the rear end of the boom, centered on the element "
             "centers. 0 = off; not combined with the feed-line wire. Not galvanically connected: --mast-gap-mm sets "
             "the coupling distance")
    opt((G, O), "--mast-diameter-mm", type=positive_float, default=40.0,
        help="mast diameter in mm (default 40)")
    opt((G, O), "--mast-offset-mm", type=nonneg_float, default=0.0, metavar="MM",
        help="horizontal elements: sideways offset of the mast from the boom axis along the elements (a side clamp "
             "arm; 0 = on the axis, where the balanced elements do not excite it). Vertical elements: extra distance "
             "of the side-mounted mast behind the rear end of the boom")
    opt((G, O), "--mast-gap-mm", type=nonneg_float, default=20.0,
        help="air gap in mm between the lowest modeled conductor (boom if modeled, else the elements) and the "
             "top of the mast (default 20; 0 puts the mast surface 10%% of its radius from the conductor)")
    opt((G, O), "--mast-sensitivity", default=None, metavar="L1,L2,...",
        help="mast lengths in m (e.g. 0.5,1,2); simulate without a mast and with each length (diameter and gap "
             "from --mast-diameter-mm/--mast-gap-mm) and print a table of gain, F/B and Zin changes "
             "(needs nec2c; generate: the model given; optimize: the best model)")
    opt((G, O), "--build-sheet", default=None, metavar="FILE",
        help="write the build sheet FILE (markdown) and FILE with .json suffix: cut list, spacing/boom, phasing "
             "line, feed/balun, predicted performance, tuning procedure, bill of materials. generate: performance "
             "needs --run; optimize: needs --network tl/lc")
    opt((G, O), "--conductivity", type=nonneg_float, default=2.5e7, metavar="S/M",
        help="conductivity of all wires in S/m (LD 5 card). Default 2.5e7 (6061 aluminium; check your tube: 6063 "
             "is about 3.0e7, copper 5.8e7, brass 1.5e7). 0 = perfect conductors (no LD card)")
    opt((G, O), "--line-allowance-mm", type=nonneg_float, default=None, metavar="MM",
        help="--network tl: extra physical line length in mm for connections, the crossing and element stubs "
             f"(default {OPENWIRE_DEFAULT_ALLOWANCE_MM:g} open-wire, {COAX_DEFAULT_ALLOWANCE_MM:g} coax). A phase line "
             "shorter than the distance between the feed points plus this allowance cannot be built: generate warns "
             "(and never picks such a line by default), optimize skips it")
    opt((G, O), "--max-line-deg", type=positive_float, default=None, metavar="DEG",
        help="longest phase line in electrical degrees that generate picks and optimize scans. Default: a line "
             f"at most {LINE_FOLD_LIMIT:g} times the straight distance between its ends (plus --line-allowance-mm), "
             "so it can be run almost straight; when the theoretical line is beyond 180 degrees (a crossed line fed "
             "at the rear element) the limit covers it plus the largest --line-offset-values. The LC network is "
             "always 0..180; values never wrap around 360")
    opt((G, O), "--synthesize-line", action=argparse.BooleanOptionalAction, default=True,
        help="--network tl/lc: compute the phase line from the two-port response of the driven elements (two nec2c "
             "probe runs) so that the element currents approach the cardioid target (equal amplitudes, rear leading "
             "by 180 - kd). generate: the default line when none is given; optimize: the centre of the line scan of "
             "every spacing and Z0. --no-synthesize-line uses the cardioid theory (kd + 180 crossed), never shorter "
             "than the buildable minimum")
    opt((G, O), "--feed-style", choices=["center", "tmatch"], default=None,
        help="tmatch (default with --network tl/lc): the classic HB9CV feed: continuous elements with a T-match on "
             "each driven element (taps at --tap-mm from the center, bar at --tap-offset-mm toward the other "
             "element); EX and the phase network connect at the bar centers, which transforms the element impedances "
             "so that a short straight line reaches the cardioid currents. center: EX and the phase network connect "
             "at the element centers (split, insulated elements); used by default with stepped sections or "
             "--model-feed-gap")
    opt((G, O), "--tap-mm", type=positive_float, default=DEFAULT_TAP_MM, metavar="MM",
        help="--feed-style tmatch: distance in mm from the element center to each tap (optimize moves it)")
    opt((G, O), "--tap-offset-mm", type=positive_float, default=DEFAULT_TAP_OFFSET_MM, metavar="MM",
        help="--feed-style tmatch: distance in mm from the element axis to the T-bar, toward the other element "
             "(NEC-2 gets less accurate as the bar comes closer than ~5 element diameters: see the average gain test)")
    opt((G, O), "--tap-wire-mm", type=positive_float, default=None, metavar="MM",
        help="--feed-style tmatch: diameter in mm of the T-bar and its stubs (default: the element diameter; "
             "unequal radii at the T junctions lower the NEC-2 accuracy)")
    opt((G, O), "--series-c-pf", type=series_c_arg, default=None, metavar="PF|auto",
        help="--network tl/lc: series capacitor in pF between the feed line and the fed port (the HB9CV trimmer); "
             "it is written as a comment card and added to every Zin read from the .out file. auto: the value that "
             "cancels X_in at --freq (none when X_in is capacitive); in optimize it is chosen per candidate")
    opt((G, O), "--series-c-q", type=nonneg_float, default=None, metavar="Q",
        help=f"quality factor of the series feed capacitor: ESR = |X|/Q is added to Zin with the reactance "
             f"(default {DEFAULT_SERIES_C_Q:g}, a good air or PTFE trimmer; 0 = lossless)")
    opt((G, O), "--line-loss-db-per-m", type=nonneg_float, default=0.0, metavar="DB",
        help="--network tl, build sheet only: matched loss of the phasing line in dB/m; the build sheet reports "
             "the upper bound loss x physical length (as if all the power crossed the line). The NEC TL itself is "
             "lossless")
    opt((G, O), "--extended-kernel", action="store_true",
        help="use NEC-2's extended thin-wire kernel (EK card). Off by default: it only helps on thick straight wires, "
             "NEC-2 turns it off at junctions, and with T-match junctions and short segments it can make the current "
             "solution unstable (negative radiated power); every run is checked for a physical power budget")
    opt((G, O), "--ideal-drive", choices=["current", "voltage"], default="current",
        help="--network ideal: current (default) prescribes the amplitude/phase of the element CURRENTS, as the "
             "cardioid theory does (each source gets a lossless series reactance of 1e6 ohm, so the currents follow "
             "the source voltages whatever the mutual coupling); voltage prescribes the source voltages")
    opt((G, O), "--nec2c", default=None, metavar="PATH",
        help=f"nec2c executable, or the folder that holds it (default: ${NEC2C_ENV}, then the PATH)")
    opt((G, O), "--check-convergence", action="store_true",
        help="simulate the model (optimize: the best model) with --segments n and 2n+1 and report the change of "
             "gain, F/B and R_in (warns above 0.1 dB, 1 dB, 5 %%); needs nec2c")
    opt((G, O), "--robustness", action="store_true",
        help="one table of the change in gain, F/B and Zin caused by a metallic boom (when not modeled), a mast "
             f"placed where it breaks the symmetry of the elements (horizontal: 1 m on a {ROBUST_MAST_OFFSET_MM:g} mm "
             f"side clamp; vertical: a {ROBUST_SIDE_MAST_M:g} m side-mounted mast) and an unchoked lambda/2 coax "
             "shield, each alone (optimize: on the best model); cases the symmetric model cannot excite are marked; "
             "needs nec2c")
    opt((G, O, M, S), "--json-out", default=None, metavar="FILE",
        help="also write the main results as JSON to FILE (deck/out paths, dimensions, metrics, Zin, network "
             "values, warnings)")
    opt((G, O), "--feed-gap-mm", "--gap-mm", dest="gap_mm", type=nonneg_float, default=10.0,
        help="feed gap g between the two halves of each element, in mm. Cut list: half cut = (L_nec - g)/2 - e, "
             "L_nec = tip-to-tip NEC length (the gap is inside it, so the electrical length of the model does not "
             "depend on g; with the same tubes, each extra mm of gap lengthens the element by 1 mm). "
             "--model-feed-gap also puts the gap in the NEC model. (--gap-mm is the older name)")
    opt((G, O), "--model-feed-gap", action="store_true",
        help="model the feed gap g (--feed-gap-mm) in NEC: each element is two half wires from +-g/2 to the tips plus "
             "a 1-segment wire of length g at the center that carries EX/TL (NEC needs a connected segment for a "
             "source/line port; an EX on the end segment of one unconnected half would drive only that half). "
             "The gap capacitance and the insulator are not modeled. The bridge must be at least 4 tube radii and "
             "lambda/200 long (thin-wire rules), so it suits thin tubes or wide gaps; not combined with the feed-line "
             "wire")
    opt((G, O), "--end-cap-mm", type=nonneg_float, default=0.0,
        help="build sheet: conductor length e added beyond each tube end by end hardware, in mm (0 = flat ends)")
    opt((G, O), "--tube-wall-mm", type=positive_float, default=1.0, help="build sheet: element tube wall in mm")
    opt((G, O), "--boom-material", default=None, help="build sheet: boom/mount material text for the bill of "
                                                      "materials (default derived from --mount)")
    opt((G, O, M, S), "--balun-ratio", choices=["1:1", "4:1"], default="1:1",
        help="1:1 current balun (default) or 4:1: balanced-side reference becomes z0/4 = 12.5 ohm (optimize: "
             "default --target-impedance 12.5 and --zin-x-max 10; match/sweep: L-network designed against 12.5). "
             "The balun steps the low antenna impedance up by 4 toward the 50 ohm coax (a 4:1 balun with its "
             "low-impedance side at the antenna). An option to evaluate, not a proven fix")

    # ---- ground, height and polarization (generate / optimize)
    opt((G, O), "--ground", choices=["free", "perfect", "average", "good"], default="free",
        help="free: free space (default, as before); perfect: GN 1; average: eps_r=13, sigma=0.005 S/m; good: "
             "eps_r=20, sigma=0.02 S/m (finite ground: GN 2, Sommerfeld). Over ground the pattern is taken at the "
             "take-off angle (elevation cut -> azimuth ring) instead of the horizon")
    opt((G, O), "--height-m", type=positive_float, default=None,
        help="height above ground in meters: horizontal = element centers; vertical = lowest tip of the longest "
             "element (driven or parasitic). Required when --ground is not free")
    opt((G, O), "--height-lambda", type=positive_float, default=None,
        help="same as --height-m in wavelengths of --freq")
    opt((G, O), "--orientation", choices=["horizontal", "vertical"], default="horizontal",
        help="horizontal: elements along Y; vertical: elements along Z (spacing along X in both). "
             "Identical in free space")
    opt((G, O), "--ground-eps", type=positive_float, default=None,
        help="relative permittivity of the finite ground (overrides the --ground preset)")
    opt((G, O), "--ground-sigma", type=nonneg_float, default=None,
        help="conductivity in S/m of the finite ground (overrides the --ground preset)")

    # ---- generate only
    opt((G,), "--line-deg", type=finite_float, default=None,
        help="electrical length of the line or LC network in degrees (0..360), by default derived from theory; "
             "with --network tl, --coax-length takes priority")
    opt((G,), "--lc-l-nh", type=positive_float, default=None,
        help="series inductor of the pi network in nH (--network lc); requires --lc-c-pf and replaces the derived value")
    opt((G,), "--lc-c-pf", type=positive_float, default=None,
        help="shunt capacitor of each port in pF (--network lc); requires --lc-l-nh")
    opt((G,), "--phase-target-deg", type=finite_float, default=None,
        help="desired delay of the far element's current relative to the fed one, in degrees; by default computed "
             "with the cardioid theory (kd+180 if feed=front, 180-kd if feed=rear)")
    opt((G,), "--fsweep", default=None, metavar="FMIN,FMAX,STEP",
        help="frequency sweep in MHz to obtain Zin(f); omits the RP pattern unless --fsweep-pattern is given")
    opt((G,), "--fsweep-pattern", action="store_true",
        help="with --fsweep: also write FR/RP/XQ for every frequency (geometry and line fixed at --freq) and, with "
             "--run, summarize gain, F/B and Zin vs frequency (ground-aware: take-off angle per frequency) and write "
             "the CSV --fsweep-csv")
    opt((G,), "--fsweep-csv", default=None, metavar="FILE",
        help="CSV of the --fsweep-pattern summary (default: <output>_fsweep.csv)")
    opt((G,), "--line-length", "--coax-length", dest="coax_length", type=positive_float, default=None,
        help="physical length of the phase line in meters; by default derived from --phase-target-deg: "
             "(degrees/360)*lambda*VF. --coax-length is a deprecated alias of --line-length")
    opt((G,), "--spacing-factor", type=positive_float, default=0.125, help="spacing in wavelengths")
    opt((G,), "--rear-length-factor", type=positive_float, default=1.0,
        help="rear element length factor relative to 150/f meters")
    opt((G,), "--front-length-factor", type=positive_float, default=1.0,
        help="front element length factor relative to 139/f meters")
    opt((G,), "--rear-ratio", type=positive_float, default=0.92, help="ideal rear amplitude relative to the front one")
    opt((G,), "--rear-phase", type=phase_float, default=None,
        help="ideal rear phase in degrees (ideal/lc network); default 180-360*spacing (+135 for 0.125 lambda)")
    opt((G,), "--run", action="store_true", help="try to run nec2c after generating the file")
    opt((G,), "--height-sweep", default=None, metavar="MIN,MAX,N",
        help="over ground: simulate the model at N heights from MIN to MAX meters and print take-off angle, gain, "
             "F/B and Zin (needs nec2c)")

    # ---- optimize only
    opt((O,), "--line-offset-values", default="-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30",
        help="offsets in electrical degrees from the theoretical line length; used with --network tl/lc")
    opt((O,), "--z0-values", default=None,
        help="Z0 values (ohm) of the TL phase line to scan, e.g. 50,75,300; on an open-wire line each Z0 sets the "
             "wire spacing (d fixed by --line-wire-diameter); default: the single Z0 of the line (--network tl)")
    opt((O,), "--line-length-values", "--coax-length-values", dest="coax_length_values", default=None,
        help="absolute physical line lengths in meters; if given, replaces --line-offset-values. "
             "--coax-length-values is a deprecated alias of --line-length-values")
    opt((O,), "--ratio-values", default="0.7,1.0,1.3", help="relative rear source amplitude; ideal network only")
    opt((O,), "--phase-offset-values", default="-60,-45,-30,-15,0,15,30,45,60",
        help="offsets in degrees from the theoretical rear phase (180 - kd) for each spacing; ideal network")
    opt((O,), "--phase-values", default=None,
        help="absolute rear phases in degrees; if given, replaces --phase-offset-values")
    opt((O,), "--parasitic-length-offsets", default="0",
        help="grid offsets added to EVERY reflector/director length factor (e.g. -0.02,0,0.02); the result is "
             "clamped to the pre-calculated limits. Needs --reflectors/--directors; the refinement tunes each "
             "element separately")
    opt((O,), "--parasitic-spacing-scales", default="1",
        help="grid multipliers of EVERY reflector/director spacing (e.g. 0.9,1,1.1); clamped to the "
             "pre-calculated minimum/maximum spacing. Needs --reflectors/--directors")
    opt((O,), "--spacing-values", default="0.08,0.125,0.17", help="spacing in wavelengths")
    opt((O,), "--rear-length-values", default="0.98,1.0,1.02",
        help="rear element length factors relative to 150/f (1 %% of length is about 1 %% of frequency)")
    opt((O,), "--front-length-values", default="0.98,1.0,1.02",
        help="front element length factors relative to 139/f (with --pretune the grid is re-centered on the tuned "
             "front length)")
    opt((O,), "--zin-r-range", default=None,
        help="admissible R_in in ohms, min,max; candidates outside are excluded from the best (tl/lc only; "
             f"with --band: at every band frequency). Default {MATCHED_ZIN_R_RANGE} with --match-score (the "
             f"matching-network loss already penalizes extreme values), {DIRECT_ZIN_R_RANGE} without it")
    opt((O,), "--zin-x-max", type=float, default=None,
        help=f"maximum admissible |X_in| in ohms (tl/lc only); default {MATCHED_ZIN_X_MAX:g} with --match-score, "
             "otherwise 30, or 10 with --balun-ratio 4:1")
    opt((O,), "--match-score", action=argparse.BooleanOptionalAction, default=None,
        help="--network tl/lc: score every candidate through the ideal L-network designed at --freq against "
             "--target-impedance, built with components of Q --match-q-l / --match-q-c: the VSWR penalty uses the "
             "matched VSWR over the band, the gain is the gain after the network loss and mismatch, and the "
             "front-back objective is lowered by that loss. Default on; --no-match-score scores the raw Zin "
             "(direct connection, no matching network)")
    opt((O,), "--match-q-l", type=positive_float, default=150.0,
        help="--match-score: Q of the matching inductor (ESR = X/Q)")
    opt((O,), "--match-q-c", type=positive_float, default=500.0,
        help="--match-score: Q of the matching capacitor (ESR = X/Q)")
    opt((O,), "--diameter-values", default=DEFAULT_DIAMETER_VALUES, metavar="D1,D2,...|any",
        help="purchasable element tube diameters in mm that --refine-diameter may choose (the start value is "
             "snapped to the nearest one); 'any' moves the diameter continuously within --diameter-range")
    opt((O,), "--tap-values", default=None, metavar="MM1,MM2,...",
        help="--feed-style tmatch: tap distances in mm scanned by the grid (default: --tap-mm); the refinement "
             "moves the tap in 10 mm steps")
    opt((O,), "--vswr-weight", type=float, default=None,
        help="dB subtracted from the score for each unit of VSWR above 1 relative to --target-impedance "
             "(worst VSWR over the band). Default 3 with --network tl/lc, 0 otherwise; 0 = report only. With "
             "--realized-gain the gain term already contains the mismatch loss once; this penalty weights matching "
             "further (the CSV columns objective_term and vswr_penalty show both parts)")
    opt((O,), "--realized-gain", action=argparse.BooleanOptionalAction, default=None,
        help="in the gain objectives use realized gain (discounts mismatch loss; tl/lc only). Default on with "
             "--network tl/lc; --no-realized-gain for the plain gain")
    opt((O,), "--band", default=None, metavar="FMIN,FMAX,N",
        help="score every candidate at N evenly spaced frequencies (e.g. 144,146,5) with one deck per candidate "
             "and keep the WORST case: min over f of the objective term, minus --vswr-weight*(max VSWR - 1). "
             "--freq (default: band center) fixes the geometry and the line length. Without --band: single frequency")
    opt((O,), "--tolerance-mc", type=int, default=0, metavar="N",
        help="after the grid/refine, simulate the same N random perturbations (common random numbers) on each of "
             "the top --tol-top-k candidates and rank them by the 10th percentile score, with a bootstrap interval; "
             "adds score_p10, score_p10_lo/hi, fb_p10, vswr_p90 to the CSV. 0 = off (use >= 100)")
    opt((O,), "--tol-dist", choices=["uniform", "normal"], default="uniform",
        help="Monte-Carlo distribution: uniform within +-tolerance (worst-case style), or normal with the "
             "tolerance as 2 sigma, clipped at 3 sigma")
    opt((O,), "--tol-length-mm", type=nonneg_float, default=2.0, help="Monte-Carlo: element length tolerance, +- mm (uniform)")
    opt((O,), "--tol-spacing-mm", type=nonneg_float, default=3.0, help="Monte-Carlo: element spacing tolerance, +- mm (uniform)")
    opt((O,), "--tol-line-deg", type=nonneg_float, default=2.0,
        help="Monte-Carlo: phase line (or rear phase, ideal network) tolerance, +- electrical degrees (uniform)")
    opt((O,), "--tol-top-k", type=int, default=5, help="Monte-Carlo: number of best candidates to analyze")
    opt((O,), "--tol-seed", type=int, default=12345, help="Monte-Carlo: random seed (reproducible)")
    opt((O,), "--tol-tap-mm", type=nonneg_float, default=2.0,
        help="Monte-Carlo: T-match tap position tolerance, +- mm, drawn for each driven element on its own")
    opt((O,), "--tol-bar-mm", type=nonneg_float, default=1.0,
        help="Monte-Carlo: T-bar offset tolerance, +- mm, drawn for each driven element on its own")
    opt((O,), "--tol-vf-pct", type=nonneg_float, default=2.0,
        help="Monte-Carlo: phasing-line velocity factor tolerance, +- percent (spacers, insulation, moisture); the "
             "cut length stays, its electrical length changes")
    opt((O,), "--tol-z0-pct", type=nonneg_float, default=3.0,
        help="Monte-Carlo: phasing-line (or LC network) Z0 tolerance, +- percent (wire spacing, cable tolerance)")
    opt((O,), "--tol-c-pct", type=nonneg_float, default=2.0,
        help="Monte-Carlo: series feed capacitor setting tolerance, +- percent")
    opt((O,), "--strict-zin", action="store_true",
        help="stop with an error when no candidate meets the Zin window (--zin-r-range / --zin-x-max at every band "
             "frequency). Default: the closest candidate is reported and written, flagged as outside the window")
    opt((O,), "--tol-retune", action="store_true",
        help="Monte-Carlo: re-tune the series capacitor (--series-c-pf auto) and re-design the matching network "
             "(--match-score) for every sample, as if they were adjusted on the bench after the build. Default: "
             "they stay as designed for the nominal candidate, so their mistuning counts in the spread")
    opt((O,), "--loss-weight", type=nonneg_float, default=1.0,
        help="--match-score with --objective front-back: dB of score subtracted per dB of matching-network loss "
             "(an efficiency penalty; the loss does not change the F/B). 0 = pure F/B")
    opt((O,), "--jobs", type=int, default=None, metavar="N",
        help="parallel nec2c processes (default: number of CPUs)")
    opt((O,), "--refine-spacing", action=argparse.BooleanOptionalAction, default=True,
        help="--refine also moves the element spacing (step 0.01 lambda, halved)")
    opt((O,), "--refine-diameter", action=argparse.BooleanOptionalAction, default=True,
        help="--refine also moves the element diameter (step 1 mm, halved, within --diameter-range)")
    opt((O,), "--diameter-range", default="6,16", help="element diameter limits in mm for --refine, min,max")
    opt((O,), "--w-gain", type=float, default=1.0, help="gain weight in the balance objective")
    opt((O,), "--w-fb", type=float, default=0.5, help="robust F/B weight in the balance objective")
    opt((O,), "--rear-sector", default="150,210",
        help="rear sector in degrees, start,end; F/B is maximized against the worst point of the sector")
    opt((O,), "--refine", type=int, default=None, metavar="ROUNDS",
        help="after the grid, rounds of local search around the best point (every round simulates all moves in "
             "parallel and keeps the best improvement; the step is halved when none improves) over element lengths, "
             "phase/line, spacing, element diameter and the reflector/director lengths and spacings; 0 = grid only. "
             "Default 6 with --network tl/lc, 8 with reflectors/directors, 0 with --network ideal")
    opt((O,), "--nm-evals", type=int, default=40, metavar="N",
        help="after the local search, a Nelder-Mead polish of the continuous parameters (element lengths, spacing, "
             "line or rear phase, element diameter) with at most N nec2c runs; it follows correlated moves such as "
             "front length against line length. 0 = off")
    opt((O,), "--pretune", action=argparse.BooleanOptionalAction, default=True,
        help="--network tl/lc: before the grid, tune the front length of the central candidate for X_in ~ 0 at "
             "--freq (secant search, up to 6 nec2c runs) and center the --front-length-values grid on it")
    opt((O,), "--fb-metric", choices=["point", "sector", "cone"], default="sector",
        help="F/B used by the front-back and balance objectives and by the Monte-Carlo: point (phi=180 on the ring), "
             "sector (worst point of --rear-sector on the ring) or cone (worst direction within the sector "
             "half-angle of the back direction at any elevation, from a 5 degree rear half-sphere grid)")
    opt((O,), "--csv", default="hb9cv_scan.csv",
        help="CSV file with all scanned candidates; the first lines ('# ...') record the arguments and the nec2c "
             "version (read with comment='#' in pandas, or skip them)")

    # ---- match / sweep (L-network)
    opt((M, S), "--nec-out", help="nec2c .out file from which Zin is read (a single EX); for sweep it must "
                                   "contain a frequency sweep (--mode generate --fsweep FMIN,FMAX,STEP --run)")
    opt((M, S), "--r", type=finite_float, help="load R in ohms (manual; requires --x; sweep: only with --assume-constant-z)")
    opt((M, S), "--x", type=finite_float, help="load X in ohms (manual; requires --r; sweep: only with --assume-constant-z)")
    opt((M, S), "--z0", type=positive_float, default=50.0, help="reference (unbalanced) impedance in ohms")
    opt((M, S), "--lnet-solution", choices=["plus", "minus", "lowpass", "highpass", "best"], default="plus",
        help="which of the two L-networks to use: plus = positive root, minus = negative root, lowpass = the "
             "solution with a series inductor and a shunt capacitor, highpass = series capacitor and shunt inductor "
             "(when neither solution has those kinds, the root closest to it: plus for lowpass, minus for highpass), "
             "best = lower worst VSWR over the NEC sweep (match: when the .out holds a sweep; otherwise the smaller "
             "spread between the two reactances). The other one is printed too")
    opt((M, S), "--esl-nh", type=nonneg_float, default=2.0,
        help="lead inductance per lead in nH of a capacitor, for the self-resonance check (two leads)")
    opt((M, S), "--coil-cp-pf", type=nonneg_float, default=0.7,
        help="stray capacitance in pF of an inductor, for the self-resonance check")
    opt((M,), "--balanced", action="store_true",
        help="also print the L-network as a balanced network (series elements split between the two legs) "
             "with a warning about parasitics above ~100 MHz")
    opt((M, S), "--q-l", type=positive_float, default=None, help="inductor Q (ESR = X/Q); default ideal")
    opt((M, S), "--q-c", type=positive_float, default=None, help="capacitor Q (ESR = X/Q); default ideal")

    # ---- sweep only
    opt((S,), "--design-freq", type=positive_float, default=145.0, help="design frequency of the network in MHz")
    opt((S,), "--assume-constant-z", action="store_true",
        help="use constant --r/--x and --fmin/--fmax/--step (does not represent the real antenna)")
    opt((S,), "--fmin", type=float, default=140.0, help="MHz; only with --assume-constant-z")
    opt((S,), "--fmax", type=float, default=150.0, help="MHz; only with --assume-constant-z")
    opt((S,), "--step", type=float, default=0.25, help="MHz; only with --assume-constant-z")
    opt((S,), "--commercial", action="append", metavar="SERIES,SHUNT",
        help="commercial values of the series branch and of the shunt branch (repeatable), in nH if the branch "
             "is an inductor or pF if it is a capacitor, according to the ideal network; default: nearest E24")
    opt((S,), "--sweep-csv", default="lnetwork_sweep.csv", help="output CSV of the L-network sweep")
    opt((S,), "--tol-l", type=nonneg_float, default=0.0,
        help="inductor tolerance as a percentage (e.g. 5 for 5 %%, 0 = off); reports the worst-case VSWR over the "
             "four corners of the tolerance box at every frequency of the sweep, for every network (ideal and "
             "commercial) around its own nominal values")
    opt((S,), "--tol-c", type=nonneg_float, default=0.0,
        help="capacitor tolerance as a percentage (e.g. 5 for 5 %%, 0 = off); see --tol-l")
    return p, table


# Comma-list options whose first value may be negative ("-20,0,20"): argparse treats such a token as an option.
LIST_OPTS = {"--line-offset-values", "--line-length-values", "--coax-length-values", "--z0-values",
             "--ratio-values", "--phase-offset-values", "--phase-values", "--spacing-values",
             "--rear-length-values", "--front-length-values", "--zin-r-range", "--diameter-range",
             "--rear-sector", "--fsweep", "--band", "--commercial", "--parasitic-length-offsets",
             "--parasitic-spacing-scales", "--diameter-values", "--tap-values"}
_NEG_LIST = re.compile(r"-\.?\d")


def fix_negative_lists(argv):
    """Rewrite '--opt -20,0,20' as '--opt=-20,0,20' for the comma-list options (argparse would read the
    value as an option and fail with 'expected one argument'). '--opt=value' forms are left untouched."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if t in LIST_OPTS and nxt is not None and _NEG_LIST.match(nxt):
            out.append(f"{t}={nxt}")
            i += 2
        else:
            out.append(t)
            i += 1
    return out


def warn_ignored_options(args, table, argv):
    """Report options explicitly given on the command line that the chosen mode does not use."""
    seen = set()
    for token in argv:
        if token.startswith("-"):
            key = token.split("=", 1)[0]
            if key in table and table[key][0] not in seen:
                dest, modes = table[key]
                seen.add(dest)
                if args.mode not in modes:
                    print(f"WARNING: option {key} is not used in --mode {args.mode} and is ignored "
                          f"(used in: {', '.join(m for m in MODES if m in modes)}).", file=sys.stderr)


# Mapping from deprecated --coax-* option names to their preferred --line-* replacements.
_DEPRECATED_LINE_OPTS = {
    "--coax-z0":            "--line-z0",
    "--coax-vf":            "--line-vf",
    "--coax-length":        "--line-length",
    "--coax-length-values": "--line-length-values",
}


def warn_deprecated_coax_opts(argv):
    """Emit a deprecation warning for every --coax-* option found in argv."""
    for token in argv:
        key = token.split("=", 1)[0]
        if key in _DEPRECATED_LINE_OPTS:
            print(f"WARNING: {key} is deprecated; use {_DEPRECATED_LINE_OPTS[key]} instead "
                  "(--coax-* names will be removed in a future version).", file=sys.stderr)


def main(argv=None):
    argv = fix_negative_lists(sys.argv[1:] if argv is None else list(argv))
    warn_deprecated_coax_opts(argv)
    p, table = build_parser()
    args = p.parse_args(argv)
    warn_ignored_options(args, table, argv)
    if args.mode in ("generate", "optimize") and args.network == "coaxial":
        print("WARNING: --network coaxial is deprecated; use --network tl. Note the default phase line is now "
              "--line-type openwire (balanced, crossed, Z0 from geometry); use --line-type coax for a coaxial "
              "cable (straight).", file=sys.stderr)
        args.network = "tl"


    args._argv = argv
    if args.mode in ("generate", "optimize"):
        network_given = any(a.split("=", 1)[0] == "--network" for a in argv)
        args.band_freqs = None
        if args.mode == "optimize" and args.band:
            try:
                args.band_freqs = parse_band(args.band)
            except argparse.ArgumentTypeError as exc:
                p.error(str(exc))
        if args.freq is None:
            args.freq = 0.5 * (args.band_freqs[0] + args.band_freqs[-1]) if args.band_freqs else 145.0
        if args.mode == "generate":
            run_generate(p, args)
        else:
            run_optimize(p, args, network_given)
    elif args.mode == "match":
        run_match(p, args)
    else:
        for name in ("fmin", "fmax", "step"):
            if not math.isfinite(getattr(args, name)):
                p.error(f"--{name} must be finite")
        run_sweep(p, args)


if __name__ == "__main__":
    main()
