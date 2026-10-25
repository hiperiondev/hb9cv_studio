# test_hb9cv.py - regression tests for hb9cv.py (part of HB9CV Studio)
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
"""Regression tests for hb9cv.py (run with: python -m pytest tests).

Tests marked nec2c need the nec2c executable (--nec2c, HB9CV_NEC2C or the PATH) and are skipped without it.
"""
import cmath
import contextlib
import importlib.util
import io
import math
import random
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hb9cv", ROOT / "hb9cv.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

NEC2C = h.find_nec2c()
needs_nec2c = pytest.mark.skipif(NEC2C is None, reason="nec2c not found")


def model(*extra, freq=145.0):
    """Arguments of a generate model, validated as the command line does."""
    p, _ = h.build_parser()
    a = p.parse_args(["--mode", "generate", "--no-synthesize-line", *extra])
    a.freq = a.freq or freq
    a.band_freqs, a._argv = None, []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        h.check_common_model_args(p, a)
    a.pattern_mode, a.ring_theta = "ring", 90.0
    return a


VARIANTS = [
    [], ["--boom-isolated"], ["--boom-isolated", "--feedline-length-m", "1"], ["--feedline-length-m", "1"],
    ["--feed-style", "center", "--boom-isolated", "--feedline-length-m", "1"],
    ["--mast-length-m", "1", "--mast-offset-mm", "60"], ["--boom-isolated", "--mast-length-m", "1"],
    ["--mount", "contact"], ["--mount", "contact", "--reflectors", "1", "--directors", "2"],
    ["--mount", "contact", "--feedline-length-m", "1"],
    ["--orientation", "vertical", "--mount", "contact", "--feedline-length-m", "1"],
    ["--orientation", "vertical", "--boom-isolated", "--mast-length-m", "2"],
    ["--network", "ideal", "--boom-isolated"],
    ["--ground", "average", "--height-m", "5", "--boom-isolated", "--feedline-length-m", "1"],
]


def test_lnet_roundtrip():
    rng = random.Random(1)
    w = 2 * math.pi * 145e6
    for _ in range(200):
        z = complex(rng.uniform(2, 400), rng.uniform(-300, 300))
        for net in h.lnet_design_all(z, 50.0, w):
            assert abs(h.lnet_zin(z, w, net) - 50.0) < 1e-6


@pytest.mark.parametrize("extra", VARIANTS, ids=lambda v: " ".join(v) or "default")
def test_geometry_is_valid(extra):
    """No overlapping conductors, no segment below 2 radii, no close-clearance warning on default variants."""
    *_, info = h.build_deck(model(*extra))
    bad = [w for w in info["wire_warnings"] if "shorter than 2 radii" in w or "apart surface to surface" in w]
    assert not bad, bad


def test_overlap_is_an_error():
    wires = [(1, (0, -0.5, 0), (0, 0.5, 0), 0.005), (5, (-0.3, 0.004, 0), (0.3, 0.004, 0), 0.005)]
    errors, _ = h.wire_clearance_problems(wires)
    assert errors


def test_openwire_inverse_and_exact():
    for z in (150.0, 200.0, 310.0, 450.0):
        assert abs(h.openwire_z0(3.0, h.openwire_spacing_mm(z, 3.0)) - z) < 1e-9
    assert abs(h.openwire_z0(3.0, 20.0) - 120.0 * math.acosh(20.0 / 3.0)) < 1e-12


def test_hpbw_power_interpolation():
    pts = {float(p): 10 * math.log10(max(math.cos(math.radians(p / 2)) ** 2, 1e-9)) for p in range(0, 360, 10)}
    bw = h.half_power_beamwidth(pts)
    assert 170.0 < bw < 190.0


def test_repeated_frequency_block_is_folded():
    def block(z, pat, inp=True):
        s = "\n FREQUENCY :  1.4500E+02 MHz\n"
        if inp:
            s += " ANTENNA INPUT PARAMETERS\n"
            s += (f"    4    20  1.0E+00  0.0E+00  2.0E-02  1.0E-03  {z.real:.4E}  {z.imag:.4E}  2.0E-02  1.0E-03"
                  "  1.0E-02\n")
        s += " RADIATION PATTERNS\n"
        for th, ph in pat:
            s += f"  {th:7.2f} {ph:7.2f}   1.00  2.00  3.00  0.00000  0.00 LINEAR 1.0E-01 10.0 1.0E-01 20.0\n"
        return s
    text = "STRUCTURE SPECIFICATION\n" + block(50 + 3j, [(90, 0)]) + block(50 + 3j, [(5, 90)])
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.out"
        out.write_text(text)
        assert len(h.parse_input_parameters(out)) == 1
    secs = h.parse_pattern_sections(text)
    assert len(secs) == 1 and len(secs[0]["points"]) == 2


def test_series_c_esr():
    z = h.series_c_impedance(10.0, 145.0, 500.0)
    assert z.imag < 0 and abs(z.real - abs(z.imag) / 500.0) < 1e-12


@needs_nec2c
def test_ideal_drive_currents():
    a = model("--network", "ideal", "--rear-ratio", "0.8", "--rear-phase", "120")
    with tempfile.TemporaryDirectory() as td:
        r = h.simulate_model(NEC2C, td, "ideal", a, [145.0], pt_mode=0)
        assert r["ok"], r
        rows = h.parse_element_currents(r["out_path"])
        info = h.build_deck(a)[5]
        ratio = h.segment_current(rows, *info["ports"]["rear"]) / h.segment_current(rows, *info["ports"]["front"])
    assert abs(abs(ratio) - 0.8) < 1e-3
    assert abs(math.degrees(cmath.phase(ratio)) - 120.0) < 0.1


@needs_nec2c
def test_probe_superposition_matches_nec():
    a = model("--reflectors", "1", "--directors", "1")
    with tempfile.TemporaryDirectory() as td:
        resp = h.two_port_response(NEC2C, a, td)
        for th in (40.0, 90.0):
            pred = h.line_pattern_metrics(resp, abs(a.coax_z0), th, a.crossed, a.feed_element, (150, 210))
            m = h.simulate_model(NEC2C, td, f"v{int(th)}", h._variant(a, line_deg=th), [145.0])["per_freq"][0]
            assert abs(pred["forward_gain"] - m["forward_gain"]) < 0.05
            assert abs(pred["fb_sector"] - m["robust_fb"]) < 0.1


def test_unphysical_power_budget_is_rejected():
    text = ("STRUCTURE SPECIFICATION\n FREQUENCY :  1.4500E+02 MHz\n ANTENNA INPUT PARAMETERS\n"
            "    4   102  1.0E+00  0.0E+00  1.4E-02  9.1E-04  6.8711E+01 -4.3194E+00  1.4E-02  9.1E-04  7.2E-03\n"
            " POWER BUDGET\n INPUT POWER   =  7.2483E-03 Watts\n RADIATED POWER=  7.4000E-03 Watts\n"
            " STRUCTURE LOSS=  1.0E-05 Watts\n EFFICIENCY    =  102.10 Percent\n")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.out"
        out.write_text(text)
        with pytest.raises(ValueError, match="not physical"):
            h.validate_input_table(out)


def test_thin_wire_kernel_by_default():
    assert "EK" not in h.build_deck(model())[0].splitlines()
    assert "EK" in h.build_deck(model("--extended-kernel"))[0].splitlines()


def test_find_nec2c_uses_bundled_bin(tmp_path, monkeypatch):
    """Without --nec2c, HB9CV_NEC2C or a nec2c in the PATH, the bin folder next to hb9cv.py is used."""
    exe = tmp_path / "bin" / ("nec2c.exe" if h.os.name == "nt" else "nec2c")
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(h, "__file__", str(tmp_path / "hb9cv.py"))
    monkeypatch.setattr(h.shutil, "which", lambda name: None)
    monkeypatch.delenv(h.NEC2C_ENV, raising=False)
    assert h.find_nec2c() == str(exe)


def test_find_nec2c_uses_exe_next_to_program(tmp_path, monkeypatch):
    """On Windows a nec2c.exe shipped in the project folder is found when nothing else provides nec2c; elsewhere
    it is ignored, since a Windows executable cannot run there."""
    exe = tmp_path / "nec2c.exe"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(h, "__file__", str(tmp_path / "hb9cv.py"))
    monkeypatch.setattr(h.shutil, "which", lambda name: None)
    monkeypatch.delenv(h.NEC2C_ENV, raising=False)
    assert h.find_nec2c() == (str(exe) if h.os.name == "nt" else None)
