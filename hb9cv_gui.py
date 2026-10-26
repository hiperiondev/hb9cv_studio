#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hb9cv_gui.py - HB9CV Studio, graphical front end for hb9cv.py
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
"""HB9CV Studio - graphical front end for hb9cv.py (NEC2C toolkit).

Requirements:  pip install customtkinter matplotlib numpy     (tkinter ships with Python)
               nec2c in PATH (or set its location in the GUI) for optimize / --run / pattern plots.
Put this file in the same folder as hb9cv.py and run:   python hb9cv_gui.py

What it does
  * Every option of the four modes (generate, optimize, match, sweep) is available in the tabs on the left.
    Options that the selected mode does not use are greyed out; the exact command line is shown live.
  * The program runs as a separate process (live log, Stop button).
  * At the end, for generate/optimize it shows:
      - Antenna: element / spacing / boom dimensions, phase line, feed point, ground and height
      - Pattern 2D: azimuth and elevation polar cuts, rectangular cut, key figures
      - Pattern 3D: interactive surface of the full radiation sphere (radius in dB or linear in power)
    The patterns come from extra nec2c runs on the model produced by hb9cv.py, at the deck frequency closest
    to the design frequency (with the NT card of that frequency for an LC network), following the procedure of
    hb9cv.py: over ground an elevation cut at 1 degree gives the take-off angle; a 1 degree azimuth ring at that
    angle (with the rear half-sphere grid) gives the gain, F/B (point, --rear-sector, rear cone), HPBW and Zin,
    so the figures agree with the log; a full sphere on the selected grid feeds the 3D plot, the elevation cut
    and, in free space, the sphere-averaged gain (= radiation efficiency) with the directivity.
  * The results are read from the JSON file that hb9cv.py writes with --json-out (the log is the fallback).
  * For match / sweep it shows the L-network schematic and the VSWR / impedance versus frequency.

Display scale (normal and high-DPI screens)
  * "Scale" menu (top right): Auto follows the system DPI; fixed values 75 % ... 300 % override it.
    Ctrl + / Ctrl - / Ctrl 0 change it from the keyboard.  Everything (widgets, fonts, plots, tooltips,
    window size) follows the scale, and the window is never larger than the screen.
  * Linux: Auto uses Xft.dpi / Tk's DPI / GDK_SCALE / QT_SCALE_FACTOR.  Set HB9CV_GUI_SCALE=1.5 to force a value.

Languages
  * English, Spanish and Italian ("Language" menu, top right; the system language is used the first time).
    The choice and the scale are stored in ~/.hb9cv_gui.json.  The text printed by hb9cv.py itself
    (log, warnings) is not translated.
"""
from __future__ import annotations

__author__ = "Emiliano Gonzalez - LU3VEA"
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (C) 2026 Emiliano Gonzalez - LU3VEA"

import argparse
import csv
import importlib.util
import json
import locale
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import webbrowser
from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

APP_TITLE = "HB9CV Studio"
APP_AUTHOR = "Emiliano Gonzalez - LU3VEA"
APP_LICENSE = "GPL-3.0-or-later"
APP_LICENSE_URL = "https://www.gnu.org/licenses/gpl-3.0.html"
PROJECT_URL = "https://github.com/hiperiondev/hb9cv_studio"
SCRIPT_DIR = Path(__file__).resolve().parent
IS_WIN = sys.platform.startswith("win")
# On Windows, child processes of the windowless GUI (pythonw) would each open a console window.
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WIN else {}


def console_python() -> str:
    """Interpreter for hb9cv.py: python.exe instead of pythonw.exe on Windows, so the engine has real stdout/stderr."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").is_file():
        return str(exe.parent / "python.exe")
    return sys.executable
CONFIG_PATH = Path.home() / ".hb9cv_gui.json"

# Light palette used in the plots
COL_REAR = "#c0392b"
COL_FRONT = "#1f6fb2"
COL_BOOM = "#6b6f76"
COL_LINE = "#2e7d32"
COL_OTHER = "#8a6d3b"
COL_REFL = "#8e44ad"     # passive reflectors
COL_DIR = "#16a085"      # passive directors
COL_GROUND = "#8d6e63"

# Current total UI scale (1.0 = 96 dpi).  Updated by the App; used by the raw-tk pieces (tooltips).
UI = {"scale": 1.0}
SCALE_CHOICES = [75, 90, 100, 110, 125, 150, 175, 200, 250, 300]   # percent


# ======================================================================================
# Internationalisation (English / Spanish / Italian)
# ======================================================================================

LANGS = {"en": "English", "es": "Español", "it": "Italiano"}
_LANG = "en"
_LANG_IDX = {"en": 0, "es": 1, "it": 2}


def set_language(code: str) -> None:
    global _LANG
    _LANG = code if code in LANGS else "en"


def get_language() -> str:
    return _LANG


def tr(key: str, **kw) -> str:
    """Translate a key into the current language (falls back to English, then to the key itself)."""
    entry = STR.get(key)
    if entry is None:
        return key
    text = entry[_LANG_IDX[_LANG]] or entry[0]
    return text.format(**kw) if kw else text


# key: (English, Spanish, Italian)
STR = {
    # ---- window / startup
    "app.subtitle": ("HB9CV toolkit for NEC2C", "herramientas HB9CV para NEC2C", "strumenti HB9CV per NEC2C"),
    "core.notfound": ("hb9cv.py was not found next to this program.\nPlease select it.",
                      "No se encontró hb9cv.py junto a este programa.\nPor favor, selecciónalo.",
                      "hb9cv.py non è stato trovato accanto a questo programma.\nSelezionalo, per favore."),
    "core.select": ("Select hb9cv.py", "Seleccionar hb9cv.py", "Seleziona hb9cv.py"),
    "core.required": ("hb9cv.py is required", "hb9cv.py es necesario", "hb9cv.py è necessario"),
    # ---- file dialogs
    "ft.all": ("All files", "Todos los archivos", "Tutti i file"),
    "ft.necout": ("NEC output", "Salida NEC", "Output NEC"),
    "ft.necdeck": ("NEC deck", "Modelo NEC", "Modello NEC"),
    "ft.preset": ("Preset", "Preajuste", "Preset"),
    "dlg.nec2c": ("Select the nec2c executable", "Selecciona el ejecutable de nec2c",
                  "Seleziona l'eseguibile di nec2c"),
    # ---- modes
    "mode.generate": ("generate", "generar", "genera"),
    "mode.optimize": ("optimize", "optimizar", "ottimizza"),
    "mode.match": ("match", "adaptar", "adatta"),
    "mode.sweep": ("sweep", "barrido", "scansione"),
    "mode_info.generate": (
        "Write a NEC2C model of the HB9CV (optionally run nec2c) and plot geometry and patterns.",
        "Escribe un modelo NEC2C de la HB9CV (opcionalmente ejecuta nec2c) y dibuja la geometría y los diagramas.",
        "Scrive un modello NEC2C della HB9CV (eventualmente esegue nec2c) e disegna geometria e diagrammi."),
    "mode_info.optimize": (
        "Grid search (plus refinement) of the geometry with nec2c; plots the best design. Needs nec2c.",
        "Búsqueda en rejilla (más refinamiento) de la geometría con nec2c; dibuja el mejor diseño. Requiere nec2c.",
        "Ricerca a griglia (più raffinamento) della geometria con nec2c; disegna il progetto migliore. Richiede nec2c."),
    "mode_info.match": (
        "Ideal L-network matching the antenna impedance (from a NEC .out file or typed R + jX).",
        "Red en L ideal que adapta la impedancia de la antena (de un archivo .out de NEC o R + jX escritos a mano).",
        "Rete a L ideale che adatta l'impedenza dell'antenna (da un file .out di NEC o R + jX digitati)."),
    "mode_info.sweep": (
        "Frequency sweep of an L-network designed at one frequency (Zin(f) from NEC or constant).",
        "Barrido en frecuencia de una red en L diseñada a una frecuencia (Zin(f) de NEC o constante).",
        "Scansione in frequenza di una rete a L progettata a una frequenza (Zin(f) da NEC o costante)."),
    # ---- option groups (left tabs)
    "group.Main": ("Main", "Principal", "Principale"),
    "group.Antenna": ("Antenna", "Antena", "Antenna"),
    "group.Feed": ("Feed", "Alimentación", "Alimentazione"),
    "group.Ground": ("Ground", "Tierra", "Terreno"),
    "group.Generate": ("Generate", "Generar", "Genera"),
    "group.Optimize": ("Optimize", "Optimizar", "Ottimizza"),
    "group.Match": ("Match", "Adaptación", "Adattamento"),
    # ---- result tabs (right)
    "rtab.Antenna": ("Antenna", "Antena", "Antenna"),
    "rtab.Pattern 2D": ("Pattern 2D", "Diagrama 2D", "Diagramma 2D"),
    "rtab.Pattern 3D": ("Pattern 3D", "Diagrama 3D", "Diagramma 3D"),
    "rtab.Matching": ("Matching", "Adaptación", "Adattamento"),
    "rtab.Summary": ("Summary", "Resumen", "Riepilogo"),
    "rtab.Log": ("Log", "Registro", "Registro"),
    # ---- placeholders
    "ph.auto": ("(automatic)", "(automático)", "(automatico)"),
    "ph.none": ("(none)", "(ninguno)", "(nessuno)"),
    # ---- main window widgets
    "ui.language": ("Language", "Idioma", "Lingua"),
    "ui.author": ("Author", "Autor", "Autore"),
    "ui.license": ("License", "Licencia", "Licenza"),
    "ui.project": ("Project", "Proyecto", "Progetto"),
    "ui.open_link": ("Open in the web browser", "Abrir en el navegador", "Apri nel browser"),
    "ui.scale": ("Scale", "Escala", "Scala"),
    "ui.language.tip": ("Interface language.", "Idioma de la interfaz.", "Lingua dell'interfaccia."),
    "ui.scale.tip": (
        "Interface scale. Auto follows the system DPI; a fixed percentage overrides it.\n"
        "Keyboard: Ctrl +  /  Ctrl -  /  Ctrl 0 (auto).",
        "Escala de la interfaz. Auto sigue los DPI del sistema; un porcentaje fijo lo sustituye.\n"
        "Teclado: Ctrl +  /  Ctrl -  /  Ctrl 0 (auto).",
        "Scala dell'interfaccia. Auto segue i DPI del sistema; una percentuale fissa la sostituisce.\n"
        "Tastiera: Ctrl +  /  Ctrl -  /  Ctrl 0 (auto)."),
    "scale.auto": ("Auto", "Auto", "Auto"),
    "ui.workdir": ("Work folder", "Carpeta de trabajo", "Cartella di lavoro"),
    "ui.nec2c_ph": ("(HB9CV_NEC2C or PATH)", "(se busca en HB9CV_NEC2C o en el PATH)",
                    "(cercato in HB9CV_NEC2C o nel PATH)"),
    "ui.patterns": ("Compute patterns for the plots (nec2c)", "Calcular diagramas para los gráficos (nec2c)",
                    "Calcola i diagrammi per i grafici (nec2c)"),
    "ui.step": ("step", "paso", "passo"),
    "ui.deg": ("deg", "grados", "gradi"),
    "ui.run": ("\u25b6  Run", "\u25b6  Ejecutar", "\u25b6  Esegui"),
    "ui.stop": ("\u25a0  Stop", "\u25a0  Detener", "\u25a0  Ferma"),
    "ui.copy": ("Copy cmd", "Copiar comando", "Copia comando"),
    "ui.save_preset": ("Save preset", "Guardar preajuste", "Salva preset"),
    "ui.load_preset": ("Load preset", "Cargar preajuste", "Carica preset"),
    "ui.open_nec": ("Open .nec...", "Abrir .nec...", "Apri .nec..."),
    "ui.open_folder": ("Open folder", "Abrir carpeta", "Apri cartella"),
    "ui.drag3d": ("drag with the mouse to rotate", "arrastra con el ratón para girar",
                  "trascina con il mouse per ruotare"),
    "view.iso": ("Iso", "Iso", "Iso"),
    "view.top": ("Top", "Planta", "Alto"),
    "view.side": ("Side", "Perfil", "Lato"),
    "view.front": ("Front", "Alzado", "Fronte"),
    # ---- status bar
    "st.ready": ("Ready", "Listo", "Pronto"),
    "st.running": ("{mode} running...", "{mode} en ejecución...", "{mode} in esecuzione..."),
    "st.stopping": ("Stopping...", "Deteniendo...", "Arresto in corso..."),
    "st.copied": ("Command copied to the clipboard", "Comando copiado al portapapeles",
                  "Comando copiato negli appunti"),
    "st.preset_saved": ("Preset saved: {p}", "Preajuste guardado: {p}", "Preset salvato: {p}"),
    "st.preset_loaded": ("Preset loaded: {p}", "Preajuste cargado: {p}", "Preset caricato: {p}"),
    "st.post": ("Post-processing results...", "Procesando los resultados...", "Elaborazione dei risultati..."),
    "st.failed": ("Finished with errors (exit code {rc}) - see the Log tab",
                  "Terminó con errores (código de salida {rc}) - consulta la pestaña Registro",
                  "Terminato con errori (codice di uscita {rc}) - vedi la scheda Registro"),
    "st.stopped": ("Stopped by the user", "Detenido por el usuario", "Interrotto dall'utente"),
    "st.error": ("Error", "Error", "Errore"),
    "st.done": ("Done", "Hecho", "Fatto"),
    "st.done_notes": ("Done (with notes - see Summary)", "Hecho (con notas - ver Resumen)",
                      "Fatto (con note - vedi Riepilogo)"),
    "st.sim_deck": ("Simulating patterns for the selected deck...",
                    "Simulando los diagramas del modelo seleccionado...",
                    "Simulazione dei diagrammi del modello selezionato..."),
    # ---- message boxes / errors
    "msg.fix": ("Please fix:\n\n{items}", "Por favor, corrige:\n\n{items}", "Correggi, per favore:\n\n{items}"),
    "msg.bad_number": ("{label}: '{value}' is not a valid number", "{label}: '{value}' no es un número válido",
                       "{label}: '{value}' non è un numero valido"),
    "msg.bad_int": ("{label}: '{value}' is not a valid integer", "{label}: '{value}' no es un entero válido",
                    "{label}: '{value}' non è un intero valido"),
    "msg.no_workdir": ("Work folder does not exist:\n{p}", "La carpeta de trabajo no existe:\n{p}",
                       "La cartella di lavoro non esiste:\n{p}"),
    "msg.need_nec2c": (
        "optimize needs nec2c, which was not found.\nInstall it or set its location in the 'nec2c' field.",
        "optimize necesita nec2c y no se encontró.\nInstálalo o indica su ubicación en el campo 'nec2c'.",
        "optimize richiede nec2c, che non è stato trovato.\nInstallalo o indica la sua posizione nel campo 'nec2c'."),
    "msg.no_nec2c_continue": (
        "nec2c was not found: the model will be generated and its geometry drawn, but no radiation patterns "
        "can be computed.\n\nContinue?",
        "No se encontró nec2c: se generará el modelo y se dibujará su geometría, pero no se podrán calcular "
        "los diagramas de radiación.\n\n¿Continuar?",
        "nec2c non è stato trovato: il modello verrà generato e la sua geometria disegnata, ma non sarà "
        "possibile calcolare i diagrammi di radiazione.\n\nContinuare?"),
    "msg.start_fail": ("Could not start the program: {exc}", "No se pudo iniciar el programa: {exc}",
                       "Impossibile avviare il programma: {exc}"),
    "msg.exit_code": ("The program stopped with exit code {rc}.\n\n{tail}",
                      "El programa terminó con el código de salida {rc}.\n\n{tail}",
                      "Il programma è terminato con codice di uscita {rc}.\n\n{tail}"),
    "msg.preset_fail": ("Could not load the preset:\n{exc}", "No se pudo cargar el preajuste:\n{exc}",
                        "Impossibile caricare il preset:\n{exc}"),
    "msg.folder_fail": ("Could not open the folder:\n{exc}", "No se pudo abrir la carpeta:\n{exc}",
                        "Impossibile aprire la cartella:\n{exc}"),
    "err.no_nec_file": ("could not find the generated .nec file in the program output",
                        "no se encontró el archivo .nec generado en la salida del programa",
                        "file .nec generato non trovato nell'output del programma"),
    "err.no_fr": ("the .nec file has no FR card", "el archivo .nec no tiene tarjeta FR",
                  "il file .nec non ha la scheda FR"),
    "err.nec2c_failed": ("nec2c failed ({rc}): {msg}", "nec2c falló ({rc}): {msg}", "nec2c è fallito ({rc}): {msg}"),
    "err.nec2c_problem": ("nec2c reported a problem: {msg}", "nec2c informó de un problema: {msg}",
                          "nec2c ha segnalato un problema: {msg}"),
    "err.no_rp": ("no RADIATION PATTERNS table in the nec2c output",
                  "no hay tabla RADIATION PATTERNS en la salida de nec2c",
                  "nessuna tabella RADIATION PATTERNS nell'output di nec2c"),
    "err.no_valid_pts": ("the radiation pattern contains no valid points",
                         "el diagrama de radiación no contiene puntos válidos",
                         "il diagramma di radiazione non contiene punti validi"),
    "err.no_tag1": ("the deck has no element with tag 1", "el modelo no tiene ningún elemento con etiqueta 1",
                    "il modello non ha alcun elemento con tag 1"),
    "err.no_front": ("the deck has no front element", "el modelo no tiene elemento delantero",
                     "il modello non ha l'elemento anteriore"),
    "pat.off": ("Pattern computation is switched off (checkbox on the left).",
                "El cálculo de diagramas está desactivado (casilla de la izquierda).",
                "Il calcolo dei diagrammi è disattivato (casella a sinistra)."),
    "pat.no_nec2c": ("nec2c was not found: set its location in the 'nec2c' field to get the patterns.",
                     "No se encontró nec2c: indica su ubicación en el campo 'nec2c' para obtener los diagramas.",
                     "nec2c non è stato trovato: indica la sua posizione nel campo 'nec2c' per ottenere i diagrammi."),
    "pat.failed": ("Pattern simulation failed: {exc}", "La simulación del diagrama falló: {exc}",
                   "La simulazione del diagramma è fallita: {exc}"),
    # ---- result tabs: placeholders
    "plot.run_hint": ("Run a mode to see the result here.", "Ejecuta un modo para ver aquí el resultado.",
                      "Esegui una modalità per vedere qui il risultato."),
    "plot.draw_fail": ("Could not draw this plot:\n{exc}", "No se pudo dibujar este gráfico:\n{exc}",
                       "Impossibile disegnare questo grafico:\n{exc}"),
    "res.no_geom": ("No geometry available.", "No hay geometría disponible.", "Nessuna geometria disponibile."),
    "res.no_pattern": ("No pattern data.", "No hay datos de diagrama.", "Nessun dato di diagramma."),
    "res.match_hint": ("Matching plots appear after running the match or sweep mode.",
                       "Los gráficos de adaptación aparecen tras ejecutar el modo adaptar o barrido.",
                       "I grafici di adattamento compaiono dopo aver eseguito la modalità adatta o scansione."),
    "res.tab_unused": ("This tab is used by the generate and optimize modes.",
                       "Esta pestaña se usa en los modos generar y optimizar.",
                       "Questa scheda è usata dalle modalità genera e ottimizza."),
    "res.no_plot": ("No result to plot.", "No hay resultado que dibujar.", "Nessun risultato da disegnare."),
    # ---- antenna plot
    "el.rear": ("Rear", "Trasero", "Posteriore"),
    "el.front": ("Front", "Delantero", "Anteriore"),
    "ant.crossed": ("crossed", "cruzada", "incrociata"),
    "ant.straight": ("straight", "recta", "diritta"),
    "ant.phase_line": ("phase line ({kind})\n|Z0| {z0:.0f} \u03a9, NEC length {length:.0f} mm",
                       "línea de fase ({kind})\n|Z0| {z0:.0f} \u03a9, longitud NEC {length:.0f} mm",
                       "linea di fase ({kind})\n|Z0| {z0:.0f} \u03a9, lunghezza NEC {length:.0f} mm"),
    "ant.lc_pi": ("LC \u03c0 network", "red LC en \u03c0", "rete LC a \u03c0"),
    "ant.spacing_dim": ("spacing {mm:.1f} mm  ({lam:.3f} \u03bb)", "separación {mm:.1f} mm  ({lam:.3f} \u03bb)",
                        "spaziatura {mm:.1f} mm  ({lam:.3f} \u03bb)"),
    "ant.spacing_mm": ("spacing {mm:.1f} mm", "separación {mm:.1f} mm", "spaziatura {mm:.1f} mm"),
    "ant.boom_dim": ("boom {mm:.0f} mm", "boom {mm:.0f} mm", "boom {mm:.0f} mm"),
    "ant.beam": ("beam (+x)", "haz (+x)", "fascio (+x)"),
    "ant.xlabel": ("x along the boom (mm)", "x a lo largo del boom (mm)", "x lungo il boom (mm)"),
    "ant.ylabel": ("{ax} along the elements (mm)", "{ax} a lo largo de los elementos (mm)",
                   "{ax} lungo gli elementi (mm)"),
    "ant.title": ("HB9CV - element plane (dimensions are NEC tip-to-tip)",
                  "HB9CV - plano de los elementos (cotas NEC de punta a punta)",
                  "HB9CV - piano degli elementi (quote NEC da punta a punta)"),
    "ant.ground": (" ground", " tierra", " terreno"),
    "ant.tap_dim": ("tap {mm:.0f} mm", "toma {mm:.0f} mm", "presa {mm:.0f} mm"),
    "ant.bar_dim": ("bar {mm:.0f} mm", "barra {mm:.0f} mm", "barra {mm:.0f} mm"),
    "ant.ground_desc": ("ground ({d})", "tierra ({d})", "terreno ({d})"),
    "ant.side_view": ("Side view (x-z)", "Vista lateral (x-z)", "Vista laterale (x-z)"),
    "ant.end_view": ("End view (y-z)", "Vista frontal (y-z)", "Vista frontale (y-z)"),
    "ant.spacing": ("Spacing", "Separación", "Spaziatura"),
    "ant.gain": ("Gain", "Ganancia", "Guadagno"),
    "ant.sector": ("sector", "sector", "settore"),
    # ---- 2D patterns
    "p2.az_title": ("Azimuth cut at elevation {el:.0f}\u00b0   (beam at 0\u00b0 = +x)",
                    "Corte en azimut a elevación {el:.0f}\u00b0   (haz en 0\u00b0 = +x)",
                    "Taglio in azimut a elevazione {el:.0f}\u00b0   (fascio a 0\u00b0 = +x)"),
    "p2.el_title": ("Elevation cut (vertical plane through the boom)",
                    "Corte en elevación (plano vertical por el boom)",
                    "Taglio in elevazione (piano verticale lungo il boom)"),
    "p2.ground_note": ("over ground: only the upper half-space is valid",
                       "sobre tierra: solo es válido el semiespacio superior",
                       "sul terreno: è valido solo il semispazio superiore"),
    "p2.azimuth": ("azimuth", "azimut", "azimut"),
    "p2.elevation": ("elevation", "elevación", "elevazione"),
    "p2.rect_title": ("Rectangular cuts", "Cortes rectangulares", "Tagli rettangolari"),
    "p2.xlabel": ("angle from the beam direction (deg)", "ángulo desde la dirección del haz (grados)",
                  "angolo dalla direzione del fascio (gradi)"),
    "p2.ylabel": ("gain (dBi)", "ganancia (dBi)", "guadagno (dBi)"),
    "tbl.freq": ("Frequency", "Frecuencia", "Frequenza"),
    "tbl.peak": ("Peak gain", "Ganancia máxima", "Guadagno di picco"),
    "tbl.peak_val": ("{g:.2f} dBi  (az {az:g}\u00b0, el {el:g}\u00b0)", "{g:.2f} dBi  (az {az:g}\u00b0, el {el:g}\u00b0)",
                     "{g:.2f} dBi  (az {az:g}\u00b0, el {el:g}\u00b0)"),
    "tbl.fwd": ("Forward gain", "Ganancia frontal", "Guadagno in avanti"),
    "tbl.takeoff": ("Take-off angle", "Ángulo de radiación", "Angolo di radiazione"),
    "tbl.fb_point": ("F/B (point)", "F/B (puntual)", "F/B (puntuale)"),
    "tbl.fb_sector": ("F/B ({a:g}-{b:g}\u00b0)", "F/B ({a:g}-{b:g}\u00b0)", "F/B ({a:g}-{b:g}\u00b0)"),
    "tbl.fb_cone": ("F/B (rear cone)", "F/B (cono trasero)", "F/B (cono posteriore)"),
    "tbl.incl_ground": ("dBi incl. ground reflection", "dBi con reflexión del suelo", "dBi con riflessione del suolo"),
    "tbl.avg_gain": ("Average gain (efficiency)", "Ganancia media (eficiencia)", "Guadagno medio (efficienza)"),
    "tbl.directivity": ("Directivity", "Directividad", "Direttività"),
    "warn.sphere_avg": ("sphere-averaged gain {a:.3f} differs from the expected efficiency {e:.3f}: the NEC current "
                        "solution does not conserve power (average gain test: gains off by about "
                        "10*log10 of the ratio) or the pattern grid is too coarse",
                        "la ganancia media en la esfera {a:.3f} difiere de la eficiencia esperada {e:.3f}: la solución "
                        "de corrientes de NEC no conserva la potencia (prueba de ganancia media: ganancias desviadas "
                        "unos 10*log10 del cociente) o la rejilla del diagrama es demasiado gruesa",
                        "il guadagno medio sulla sfera {a:.3f} differisce dall'efficienza attesa {e:.3f}: la soluzione "
                        "delle correnti di NEC non conserva la potenza (test del guadagno medio: guadagni sbagliati di "
                        "circa 10*log10 del rapporto) o la griglia del diagramma è troppo grossolana"),
    "warn.lc_nt": ("LC network: the NT admittances of the {f:g} MHz frequency group are used",
                   "Red LC: se usan las admitancias NT del grupo de frecuencia de {f:g} MHz",
                   "Rete LC: si usano le ammettenze NT del gruppo di frequenza a {f:g} MHz"),
    "tbl.hpbw_az": ("HPBW azimuth", "HPBW azimut", "HPBW azimut"),
    "tbl.hpbw_el": ("HPBW elevation", "HPBW elevación", "HPBW elevazione"),
    "tbl.vswr_vs": ("VSWR vs {z:g} \u03a9", "VSWR vs {z:g} \u03a9", "VSWR vs {z:g} \u03a9"),
    # ---- 3D pattern
    "p3.xlabel": ("x (boom, beam +x)", "x (boom, haz +x)", "x (boom, fascio +x)"),
    "p3.cbar": ("gain (dBi)  -  radius scaled over a 40 dB range",
                "ganancia (dBi)  -  radio escalado en un rango de 40 dB",
                "guadagno (dBi)  -  raggio scalato su un intervallo di 40 dB"),
    "p3.cbar_lin": ("gain (dBi)  -  radius proportional to the power gain",
                    "ganancia (dBi)  -  radio proporcional a la ganancia de potencia",
                    "guadagno (dBi)  -  raggio proporzionale al guadagno di potenza"),
    "ui.radius_lin": ("linear radius", "radio lineal", "raggio lineare"),
    "p3.title": ("3D radiation pattern at {f:g} MHz", "Diagrama de radiación 3D a {f:g} MHz",
                 "Diagramma di radiazione 3D a {f:g} MHz"),
    "p3.inset": ("antenna", "antena", "antenna"),
    # ---- matching plots
    "m.title": ("Ideal L-network", "Red en L ideal", "Rete a L ideale"),
    "m.source": ("Source\nZ0\n{z:g} \u03a9", "Fuente\nZ0\n{z:g} \u03a9", "Sorgente\nZ0\n{z:g} \u03a9"),
    "m.antenna": ("Antenna", "Antena", "Antenna"),
    "m.series": ("series", "serie", "serie"),
    "m.shunt": ("shunt", "paralelo", "parallelo"),
    "m.topology": ("Topology {t}:", "Topología {t}:", "Topologia {t}:"),
    "m.vswr_label": ("VSWR with the L-network", "VSWR con la red en L", "VSWR con la rete a L"),
    "m.freq": ("frequency (MHz)", "frecuencia (MHz)", "frequenza (MHz)"),
    "m.preview_title": ("Bandwidth preview (NEC sweep through the ideal network)",
                        "Vista previa del ancho de banda (barrido NEC a través de la red ideal)",
                        "Anteprima della banda passante (scansione NEC attraverso la rete ideale)"),
    "s.title": ("L-network sweep", "Barrido de la red en L", "Scansione della rete a L"),
    "s.rload": ("R load", "R carga", "R carico"),
    "s.xload": ("X load", "X carga", "X carico"),
    "s.rin": ("R in ({n})", "R entrada ({n})", "R ingresso ({n})"),
    "s.xin": ("X in ({n})", "X entrada ({n})", "X ingresso ({n})"),
    "s.imp": ("impedance (\u03a9)", "impedancia (\u03a9)", "impedenza (\u03a9)"),
    # ---- summary tab
    "sum.model": ("Model file", "Archivo del modelo", "File del modello"),
    "sum.rear": ("Rear element", "Elemento trasero", "Elemento posteriore"),
    "sum.front": ("Front element", "Elemento delantero", "Elemento anteriore"),
    "sum.spacing": ("Spacing", "Separación", "Spaziatura"),
    "sum.boom": ("Boom", "Boom", "Boom"),
    "sum.elem_val": ("{len:.1f} mm   diameter(s) {d} mm", "{len:.1f} mm   diámetro(s) {d} mm",
                     "{len:.1f} mm   diametro/i {d} mm"),
    "sum.pattern_hdr": ("Radiation pattern at {f:g} MHz (full sphere simulated with nec2c)",
                        "Diagrama de radiación a {f:g} MHz (esfera completa simulada con nec2c)",
                        "Diagramma di radiazione a {f:g} MHz (sfera completa simulata con nec2c)"),
    "sum.fwd": ("forward gain", "ganancia frontal", "guadagno in avanti"),
    "sum.peak": ("peak gain", "ganancia máxima", "guadagno di picco"),
    "sum.peak_val": ("{g:.2f} dBi at azimuth {az:g} deg, elevation {el:g} deg",
                     "{g:.2f} dBi en azimut {az:g}\u00b0, elevación {el:g}\u00b0",
                     "{g:.2f} dBi ad azimut {az:g}\u00b0, elevazione {el:g}\u00b0"),
    "sum.takeoff": ("take-off angle", "ángulo de radiación", "angolo di radiazione"),
    "sum.takeoff_val": ("{v:g} deg", "{v:g}\u00b0", "{v:g}\u00b0"),
    "sum.fb": ("F/B", "F/B", "F/B"),
    "sum.fb_val": ("{fb}   (rear sector {a:g}-{b:g}: {rb}; rear cone: {cb})",
                   "{fb}   (sector trasero {a:g}-{b:g}: {rb}; cono trasero: {cb})",
                   "{fb}   (settore posteriore {a:g}-{b:g}: {rb}; cono posteriore: {cb})"),
    "sum.avg": ("sphere average", "media en la esfera", "media sulla sfera"),
    "sum.avg_val": ("{a:.3f} of the input power radiated (efficiency); directivity {d:.2f} dBi",
                    "{a:.3f} de la potencia de entrada radiada (eficiencia); directividad {d:.2f} dBi",
                    "{a:.3f} della potenza d'ingresso irradiata (efficienza); direttività {d:.2f} dBi"),
    "sum.hpbw": ("HPBW", "HPBW", "HPBW"),
    "sum.hpbw_val": ("azimuth {a}, elevation {e}", "azimut {a}, elevación {e}", "azimut {a}, elevazione {e}"),
    "sum.zin_val": ("{r:.2f}{x:+.2f}j ohm   VSWR vs {z:g} ohm = {v}:1",
                    "{r:.2f}{x:+.2f}j ohm   VSWR vs {z:g} ohm = {v}:1",
                    "{r:.2f}{x:+.2f}j ohm   VSWR vs {z:g} ohm = {v}:1"),
    "sum.sweep_rows": ("Sweep rows: {n}", "Filas del barrido: {n}", "Righe della scansione: {n}"),
    "sum.warnings": ("Warnings from the program:", "Avisos del programa:", "Avvisi del programma:"),
    "sum.notes": ("Notes:", "Notas:", "Note:"),
}


# ---- translations of the option labels / tooltips: flag -> ((label_es, tip_es), (label_it, tip_it))
OPT_TR = {
    "--freq": (("Frecuencia (MHz)", "generate/optimize: frecuencia de diseño (por defecto 145). match: debe existir en el archivo NEC."),
               ("Frequenza (MHz)", "generate/optimize: frequenza di progetto (predefinita 145). match: deve esistere nel file NEC.")),
    "--network": (("Red de alimentación", "tl: una EX + línea de transmisión ideal; lc: una EX + red LC en pi ideal; ideal: dos fuentes EX."),
                  ("Rete di alimentazione", "tl: una EX + linea di trasmissione ideale; lc: una EX + rete LC a pi ideale; ideal: due sorgenti EX.")),
    "--feed-element": (("Elemento alimentado", "Elemento que recibe la única fuente EX (HB9CV clásica: el delantero)."),
                       ("Elemento alimentato", "Elemento che riceve l'unica sorgente EX (HB9CV classica: l'anteriore).")),
    "--objective": (("Objetivo", "optimize: criterio de puntuación. generate: solo informativo."),
                    ("Obiettivo", "optimize: criterio di valutazione. generate: solo informativo.")),
    "--mount": (("Montaje", "isolated: elementos aislados del boom (por defecto). contact: centros de los elementos unidos a un boom metálico (solo con T-match)."),
                ("Montaggio", "isolated: elementi isolati dal boom (predefinito). contact: centri degli elementi collegati a un boom metallico (solo con T-match).")),
    "--balun-ratio": (("Relación del balun", "Balun de corriente 1:1, o 4:1 (referencia del lado balanceado 12,5 ohm)."),
                      ("Rapporto del balun", "Balun di corrente 1:1, oppure 4:1 (riferimento lato bilanciato 12,5 ohm).")),
    "--target-impedance": (("Z objetivo (ohm)", "Referencia del lado balanceado. Por defecto 50, o 12,5 con balun 4:1."),
                           ("Z obiettivo (ohm)", "Riferimento lato bilanciato. Predefinito 50, oppure 12,5 con balun 4:1.")),
    "--segments": (("Segmentos / elemento", "Impar, >= 5. Por defecto 21, o 41 con el T-match."),
                   ("Segmenti / elemento", "Dispari, >= 5. Predefinito 21, oppure 41 con il T-match.")),
    "--az-step": (("Paso en azimut (grados)", "Paso del diagrama escrito en el modelo; debe dividir 180 exactamente."),
                  ("Passo in azimut (gradi)", "Passo del diagramma scritto nel modello; deve dividere esattamente 180.")),
    "--conductivity": (("Conductividad (S/m)", "Conductividad del hilo (LD 5). 2.5e7 = aluminio 6061, 5.8e7 = cobre, 0 = conductores perfectos."),
                       ("Conducibilità (S/m)", "Conducibilità del filo (LD 5). 2.5e7 = alluminio 6061, 5.8e7 = rame, 0 = conduttori perfetti.")),
    "--output": (("Archivo .nec de salida", "Por defecto: derivado de los parámetros (generate) o hb9cv_optimized.nec (optimize)."),
                 ("File .nec di uscita", "Predefinito: derivato dai parametri (generate) o hb9cv_optimized.nec (optimize).")),
    "--build-sheet": (("Hoja de construcción (.md)", "Escribe la hoja de construcción (markdown + json). generate requiere 'Ejecutar nec2c'; optimize requiere red tl/lc."),
                      ("Scheda di costruzione (.md)", "Scrive la scheda di costruzione (markdown + json). generate richiede 'Esegui nec2c'; optimize richiede rete tl/lc.")),
    "--run": (("Ejecutar nec2c tras generar", "Ejecuta nec2c e imprime el resumen del diagrama, las corrientes de los elementos y la comprobación de pérdidas en los conductores."),
              ("Esegui nec2c dopo la generazione", "Esegue nec2c e stampa il riepilogo del diagramma, le correnti degli elementi e la verifica delle perdite nei conduttori.")),
    # ---- Antenna
    "--spacing-factor": (("Separación (longitudes de onda)", "Separación entre elementos en longitudes de onda."),
                         ("Spaziatura (lunghezze d'onda)", "Spaziatura tra gli elementi in lunghezze d'onda.")),
    "--rear-length-factor": (("Factor de longitud trasero", "Relativo a 150/f metros (0,5 .. 1,5)."),
                             ("Fattore di lunghezza posteriore", "Relativo a 150/f metri (0,5 .. 1,5).")),
    "--front-length-factor": (("Factor de longitud delantero", "Relativo a 139/f metros (0,5 .. 1,5)."),
                              ("Fattore di lunghezza anteriore", "Relativo a 139/f metri (0,5 .. 1,5).")),
    "--element-diameter": (("Diámetro de elemento (mm)", "Ambos elementos."), ("Diametro elemento (mm)", "Entrambi gli elementi.")),
    "--rear-diameter": (("Diámetro trasero (mm)", "Por defecto: diámetro de elemento."),
                        ("Diametro posteriore (mm)", "Predefinito: diametro elemento.")),
    "--front-diameter": (("Diámetro delantero (mm)", "Por defecto: diámetro de elemento."),
                         ("Diametro anteriore (mm)", "Predefinito: diametro elemento.")),
    "--rear-sections": (("Secciones traseras", "Tubos telescópicos, UNA mitad, del centro hacia fuera: LONGITUD_MM:DIAMETRO_MM,...,DIAMETRO_MM  p. ej. 300:12,200:10,8"),
                        ("Sezioni posteriori", "Tubi telescopici, UNA metà, dal centro verso l'esterno: LUNGHEZZA_MM:DIAMETRO_MM,...,DIAMETRO_MM  es. 300:12,200:10,8")),
    "--front-sections": (("Secciones delanteras", "Mismo formato que las secciones traseras."),
                         ("Sezioni anteriori", "Stesso formato delle sezioni posteriori.")),
    "--boom-shape": (("Forma del boom", ""), ("Forma del boom", "")),
    "--boom-size": (("Tamaño del boom (mm)", "Lado (cuadrado) o diámetro (redondo)."),
                    ("Dimensione del boom (mm)", "Lato (quadrato) o diametro (tondo).")),
    "--boom-length": (("Longitud del boom (mm)", ""), ("Lunghezza del boom (mm)", "")),
    "--boom-isolated": (("Modelar el boom (aislado)", "Boom como conductor metálico sin conexión eléctrica, situado bajo los elementos."),
                        ("Modellare il boom (isolato)", "Boom come conduttore metallico senza collegamento elettrico, sotto gli elementi.")),
    "--feed-gap-mm": (("Hueco de alimentación (mm)", "Separación entre las dos mitades de cada elemento."),
                      ("Gap di alimentazione (mm)", "Spazio tra le due metà di ciascun elemento.")),
    "--model-feed-gap": (("Modelar el hueco de alimentación en NEC",
                          "Hilo puente de longitud g; requiere g >= 4 radios del tubo y >= lambda/200."),
                         ("Modellare il gap di alimentazione in NEC",
                          "Filo ponte di lunghezza g; richiede g >= 4 raggi del tubo e >= lambda/200.")),
    "--end-cap-mm": (("Tapón final (mm)", "Solo hoja de construcción."), ("Tappo terminale (mm)", "Solo scheda di costruzione.")),
    "--tube-wall-mm": (("Pared del tubo (mm)", "Solo hoja de construcción."), ("Spessore del tubo (mm)", "Solo scheda di costruzione.")),
    "--boom-material": (("Texto del material del boom", "Lista de materiales de la hoja de construcción."),
                        ("Testo del materiale del boom", "Distinta materiali della scheda di costruzione.")),
    # ---- Feed
    "--line-type": (("Tipo de línea de fase", "openwire (por defecto): línea balanceada en aire, cruzada, VF 0,95. coax: recta, VF 0,66."),
                    ("Tipo di linea di fase", "openwire (predefinito): linea bilanciata in aria, incrociata, VF 0,95. coax: diritta, VF 0,66.")),
    "--crossed": (("Línea / red cruzada", "Por defecto: cruzada, salvo coax (siempre recta)."),
                  ("Linea / rete incrociata", "Predefinito: incrociata, tranne coax (sempre diritta).")),
    "--line-wire-diameter": (("Diámetro del hilo abierto (mm)", "Por defecto 3."), ("Diametro filo linea aperta (mm)", "Predefinito 3.")),
    "--line-spacing": (("Separación de la línea abierta (mm)", "Por defecto 20. Z0 = 119,9*acosh(D/d)."),
                       ("Spaziatura linea aperta (mm)", "Predefinito 20. Z0 = 119,9*acosh(D/d).")),
    "--line-z0": (("Z0 de la línea (ohm, coax)", "Solo coax, por defecto 50."), ("Z0 della linea (ohm, coax)", "Solo coax, predefinito 50.")),
    "--line-vf": (("Factor de velocidad de la línea", "0 < VF <= 1."), ("Fattore di velocità della linea", "0 < VF <= 1.")),
    "--lc-z0": (("Z0 de la red LC (ohm)", "Solo con red lc."), ("Z0 della rete LC (ohm)", "Solo con rete lc.")),
    "--line-deg": (("Longitud eléctrica de la línea (grados)", "0..360. Por defecto: la línea sintetizada (o la teórica), nunca más corta de lo construible."),
                   ("Lunghezza elettrica della linea (gradi)", "0..360. Predefinito: la linea sintetizzata (o quella teorica), mai più corta del costruibile.")),
    "--phase-target-deg": (("Fase objetivo (grados)", "Retardo de la corriente del elemento lejano respecto al alimentado."),
                           ("Fase obiettivo (gradi)", "Ritardo della corrente dell'elemento lontano rispetto a quello alimentato.")),
    "--line-length": (("Longitud física de la línea (m)", "Sustituye la longitud calculada."),
                      ("Lunghezza fisica della linea (m)", "Sostituisce la lunghezza calcolata.")),
    "--lc-l-nh": (("L serie de la red LC (nH)", "Requiere la C de la red LC."), ("L serie della rete LC (nH)", "Richiede la C della rete LC.")),
    "--lc-c-pf": (("C paralelo de la red LC (pF)", "Requiere la L de la red LC."), ("C parallelo della rete LC (pF)", "Richiede la L della rete LC.")),
    "--rear-ratio": (("Relación de amplitud trasera", "Solo red ideal (1,0 = cardioide)."),
                     ("Rapporto di ampiezza posteriore", "Solo rete ideale (1,0 = cardioide).")),
    "--rear-phase": (("Fase trasera (grados)", "Ideal / lc. Por defecto 180 - 360*separación."),
                     ("Fase posteriore (gradi)", "Ideale / lc. Predefinito 180 - 360*spaziatura.")),
    "--feedline-length-m": (("Long. línea alim. MC (m)", "Hilo de prueba de modo común para el conductor exterior del coaxial (0 = desactivado)."),
                            ("Lungh. linea alim. MC (m)", "Filo di prova di modo comune per il conduttore esterno del coassiale (0 = disattivato).")),
    "--feedline-diameter": (("Diám. línea alim. MC (mm)", ""), ("Diam. linea alim. MC (mm)", "")),
    "--cm-sensitivity": (("Sensibilidad al modo común", "Requiere nec2c."), ("Sensibilità al modo comune", "Richiede nec2c.")),
    # ---- Ground
    "--ground": (("Tierra", "free (espacio libre) / perfect (perfecta) / average (media) / good (buena)."),
                 ("Terreno", "free (spazio libero) / perfect (perfetto) / average (medio) / good (buono).")),
    "--height-m": (("Altura (m)", "Obligatoria sobre tierra."), ("Altezza (m)", "Obbligatoria sul terreno.")),
    "--height-lambda": (("Altura (longitudes de onda)", "Alternativa a la altura en m."),
                        ("Altezza (lunghezze d'onda)", "Alternativa all'altezza in m.")),
    "--orientation": (("Orientación", ""), ("Orientamento", "")),
    "--ground-eps": (("eps_r del suelo", "Sustituye el valor predefinido."), ("eps_r del terreno", "Sostituisce il valore predefinito.")),
    "--ground-sigma": (("sigma del suelo (S/m)", "Sustituye el valor predefinido."), ("sigma del terreno (S/m)", "Sostituisce il valore predefinito.")),
    "--mast-length-m": (("Longitud del mástil (m)", "Horizontal: bajo el centro del boom. Vertical: montaje lateral, paralelo a los elementos (0 = desactivado)."),
                        ("Lunghezza del palo (m)", "Orizzontale: sotto il centro del boom. Verticale: montaggio laterale, parallelo agli elementi (0 = disattivato).")),
    "--mast-diameter-mm": (("Diámetro del mástil (mm)", ""), ("Diametro del palo (mm)", "")),
    "--mast-gap-mm": (("Separación del mástil (mm)", ""), ("Distanza dal palo (mm)", "")),
    "--mast-sensitivity": (("Sensibilidad al mástil (m)", "Lista de longitudes de mástil separadas por comas, p. ej. 0.5,1,2. Requiere nec2c."),
                           ("Sensibilità al palo (m)", "Elenco di lunghezze del palo separate da virgole, es. 0.5,1,2. Richiede nec2c.")),
    # ---- Generate
    "--fsweep": (("Barrido de frecuencia", "FMIN,FMAX,PASO en MHz (da Zin(f))."), ("Scansione di frequenza", "FMIN,FMAX,PASSO in MHz (fornisce Zin(f)).")),
    "--fsweep-pattern": (("Diagrama vs frecuencia", "Con el barrido: ganancia / F/B / Zin en función de la frecuencia y CSV (requiere Ejecutar nec2c)."),
                         ("Diagramma vs frequenza", "Con la scansione: guadagno / F/B / Zin in funzione della frequenza e CSV (richiede Esegui nec2c).")),
    "--fsweep-csv": (("CSV del barrido de frecuencia", ""), ("CSV scansione di frequenza", "")),
    # ---- Optimize
    "--band": (("Banda FMIN,FMAX,N", "Puntúa cada candidato en toda la banda (peor caso)."),
               ("Banda FMIN,FMAX,N", "Valuta ogni candidato sull'intera banda (caso peggiore).")),
    "--refine": (("Rondas de refinamiento", "Rondas de búsqueda local tras la rejilla (0 = solo rejilla). Por defecto 6 con tl/lc, 8 con reflectores/directores."),
                 ("Cicli di raffinamento", "Cicli di ricerca locale dopo la griglia (0 = solo griglia). Predefinito 6 con tl/lc, 8 con riflettori/direttori.")),
    "--nm-evals": (("Simulaciones Nelder-Mead", "Presupuesto del pulido Nelder-Mead tras la búsqueda local (0 = desactivado)."),
                   ("Simulazioni Nelder-Mead", "Budget della rifinitura Nelder-Mead dopo la ricerca locale (0 = disattivata).")),
    "--pretune": (("Presintonizar longitud delantera", "Ajusta la longitud delantera para X_in ~ 0 en el puerto NEC (sin el condensador serie) antes de la rejilla (activado por defecto)."),
                  ("Pre-sintonia lunghezza anteriore", "Regola la lunghezza anteriore per X_in ~ 0 alla porta NEC (senza il condensatore serie) prima della griglia (attivo per impostazione predefinita).")),
    "--fb-metric": (("Métrica F/B", "F/B que usa la puntuación: point, sector (anillo) o cone (cualquier elevación)."),
                    ("Metrica F/B", "F/B usato dal punteggio: point, sector (anello) o cone (qualsiasi elevazione).")),
    "--jobs": (("Trabajos en paralelo", "Por defecto: número de CPU."), ("Processi in parallelo", "Predefinito: numero di CPU.")),
    "--csv": (("CSV de la exploración", ""), ("CSV della ricerca", "")),
    "--spacing-values": (("Valores de separación", "Longitudes de onda."), ("Valori di spaziatura", "Lunghezze d'onda.")),
    "--rear-length-values": (("Factores de longitud trasera", ""), ("Fattori di lunghezza posteriore", "")),
    "--front-length-values": (("Factores de longitud delantera", "Se recentran en la longitud delantera presintonizada."),
                              ("Fattori di lunghezza anteriore", "Ricentrati sulla lunghezza anteriore pre-sintonizzata.")),
    "--line-offset-values": (("Desfases de línea (grados)", "Desviaciones respecto al centro de la exploración de línea: la línea sintetizada o la teórica (tl/lc)."),
                             ("Offset della linea (gradi)", "Scostamenti rispetto al centro dell'esplorazione della linea: la linea sintetizzata o quella teorica (tl/lc).")),
    "--line-length-values": (("Longitudes de línea (m)", "Sustituye los desfases."), ("Lunghezze della linea (m)", "Sostituisce gli offset.")),
    "--z0-values": (("Valores de Z0 (ohm)", "p. ej. 50,75,300"), ("Valori di Z0 (ohm)", "es. 50,75,300")),
    "--ratio-values": (("Relaciones de amplitud", "Solo red ideal."), ("Rapporti di ampiezza", "Solo rete ideale.")),
    "--phase-offset-values": (("Desfases de fase (grados)", "Red ideal."), ("Offset di fase (gradi)", "Rete ideale.")),
    "--phase-values": (("Fases absolutas (grados)", "Sustituye los desfases."), ("Fasi assolute (gradi)", "Sostituisce gli offset.")),
    "--zin-r-range": (("R_in admisible (mín,máx)", "ohm. Por defecto 2,500 con la puntuación con red L, 8,80 sin ella."),
                      ("R_in ammessa (min,max)", "ohm. Predefinito 2,500 con il punteggio con rete L, 8,80 senza.")),
    "--zin-x-max": (("|X_in| máx. (ohm)", "Por defecto 500 con la puntuación con red L; si no, 30 (10 con balun 4:1)."),
                    ("|X_in| max (ohm)", "Predefinito 500 con il punteggio con rete L; altrimenti 30 (10 con balun 4:1).")),
    "--match-score": (("Puntuar a través de la red L", "Puntúa cada candidato tras una red L con pérdidas (por defecto con tl/lc); no = conexión directa."),
                      ("Punteggio attraverso la rete L", "Valuta ogni candidato dopo una rete L con perdite (predefinito con tl/lc); no = collegamento diretto.")),
    "--match-q-l": (("Red L: Q del inductor", "Q del inductor de la red L de puntuación."), ("Rete L: Q dell'induttore", "Q dell'induttore della rete L di valutazione.")),
    "--match-q-c": (("Red L: Q del condensador", "Q del condensador de la red L de puntuación."), ("Rete L: Q del condensatore", "Q del condensatore della rete L di valutazione.")),
    "--diameter-values": (("Diámetros de tubo (mm)", "Diámetros comerciales que recorre el refinamiento; 'any' = continuo."),
                          ("Diametri dei tubi (mm)", "Diametri commerciali percorsi dal raffinamento; 'any' = continuo.")),
    "--tap-values": (("Tomas del T-match (mm)", "Distancias de toma exploradas por la rejilla (por defecto: la toma del T-match)."),
                     ("Prese del T-match (mm)", "Distanze di presa esplorate dalla griglia (predefinito: la presa del T-match).")),
    "--feed-style": (("Tipo de alimentación", "tmatch (por defecto con tl/lc): T-match en ambos elementos activos, línea y alimentación en las barras. center: en los centros de los elementos."),
                     ("Tipo di alimentazione", "tmatch (predefinito con tl/lc): T-match su entrambi gli elementi attivi, linea e alimentazione sulle barre. center: ai centri degli elementi.")),
    "--tap-mm": (("Toma del T-match (mm)", "Distancia del centro del elemento a cada toma (optimize la mueve)."),
                 ("Presa del T-match (mm)", "Distanza dal centro dell'elemento a ciascuna presa (optimize la sposta).")),
    "--tap-offset-mm": (("Separación de la barra T (mm)", "Distancia del elemento a la barra T, hacia el otro elemento (barras más cercanas reducen la precisión de NEC)."),
                        ("Distanza della barra T (mm)", "Distanza dall'elemento alla barra T, verso l'altro elemento (barre più vicine riducono la precisione di NEC).")),
    "--tap-wire-mm": (("Hilo del T-match (mm)", "Diámetro de la barra T y de sus tramos (por defecto: el diámetro del elemento)."),
                      ("Filo del T-match (mm)", "Diametro della barra T e dei suoi tratti (predefinito: il diametro dell'elemento).")),
    "--max-line-deg": (("Longitud máx. de línea (grados)", "La línea más larga que elige generate y explora optimize. Por defecto: 1,5 x el tramo recto."),
                       ("Lunghezza max della linea (gradi)", "La linea più lunga scelta da generate ed esplorata da optimize. Predefinito: 1,5 x il tratto diritto.")),
    "--synthesize-line": (("Sintetizar la línea", "Línea que maximiza el objetivo sobre los diagramas de prueba superpuestos (dos ejecuciones de nec2c); por defecto activado."),
                          ("Sintetizzare la linea", "Linea che massimizza l'obiettivo sui diagrammi di prova sovrapposti (due esecuzioni di nec2c); predefinito attivo.")),
    "--series-c-pf": (("C serie de alimentación (pF)", "Condensador serie entre la línea de alimentación y el puerto alimentado, en pF, o auto (anula X_in a la frecuencia)."),
                      ("C serie di alimentazione (pF)", "Condensatore serie tra la linea di alimentazione e la porta alimentata, in pF, oppure auto (annulla X_in alla frequenza).")),
    "--mast-offset-mm": (("Desplazamiento lateral del mástil (mm)", "Horizontal: desplazamiento del mástil respecto al eje del boom (abrazadera lateral). Vertical: distancia extra detrás del boom."),
                         ("Spostamento laterale del palo (mm)", "Orizzontale: spostamento del palo dall'asse del boom (morsetto laterale). Verticale: distanza extra dietro il boom.")),
    "--vswr-weight": (("Peso del VSWR (dB)", "Por defecto 3 con tl/lc, si no 0."), ("Peso del VSWR (dB)", "Predefinito 3 con tl/lc, altrimenti 0.")),
    "--realized-gain": (("Ganancia realizada", ""), ("Guadagno realizzato", "")),
    "--w-gain": (("Peso: ganancia", "Objetivo balance."), ("Peso: guadagno", "Obiettivo balance.")),
    "--w-fb": (("Peso: F/B", "Objetivo balance."), ("Peso: F/B", "Obiettivo balance.")),
    "--rear-sector": (("Sector trasero (grados)", "inicio,fin con 90 < inicio < 180 < fin < 270"),
                      ("Settore posteriore (gradi)", "inizio,fine con 90 < inizio < 180 < fine < 270")),
    "--refine-spacing": (("Refinar separación", ""), ("Raffinare spaziatura", "")),
    "--refine-diameter": (("Refinar diámetro", ""), ("Raffinare diametro", "")),
    "--diameter-range": (("Rango de diámetros (mm)", "mín,máx"), ("Intervallo diametri (mm)", "min,max")),
    "--tolerance-mc": (("Muestras Monte Carlo", "0 = desactivado (usar >= 100)."), ("Campioni Monte Carlo", "0 = disattivato (usare >= 100).")),
    "--tol-dist": (("MC: distribución", "uniform dentro de +-tolerancia, o normal (tolerancia = 2 sigma)."),
                   ("MC: distribuzione", "uniform entro +-tolleranza, oppure normal (tolleranza = 2 sigma).")),
    "--tol-top-k": (("MC: mejores K candidatos", ""), ("MC: migliori K candidati", "")),
    "--tol-length-mm": (("MC: tol. de longitud (mm)", ""), ("MC: tol. lunghezza (mm)", "")),
    "--tol-spacing-mm": (("MC: tol. de separación (mm)", ""), ("MC: tol. spaziatura (mm)", "")),
    "--tol-line-deg": (("MC: tol. de línea (grados)", ""), ("MC: tol. linea (gradi)", "")),
    "--tol-seed": (("MC: semilla", ""), ("MC: seme", "")),
    # ---- Match / sweep
    "--nec-out": (("Archivo .out de NEC", "Salida de nec2c de la que se lee Zin (una sola EX). Sweep necesita un barrido de frecuencia dentro."),
                  ("File .out di NEC", "Output di nec2c da cui si legge Zin (una sola EX). Sweep richiede una scansione di frequenza al suo interno.")),
    "--r": (("R de carga (ohm)", "Impedancia manual (necesita X)."), ("R del carico (ohm)", "Impedenza manuale (richiede X).")),
    "--x": (("X de carga (ohm)", "Impedancia manual (necesita R)."), ("X del carico (ohm)", "Impedenza manuale (richiede R).")),
    "--z0": (("Z0 de referencia (ohm)", "Impedancia de la fuente no balanceada."), ("Z0 di riferimento (ohm)", "Impedenza della sorgente sbilanciata.")),
    "--q-l": (("Q del inductor", "ESR = X/Q (por defecto ideal)."), ("Q dell'induttore", "ESR = X/Q (predefinito ideale).")),
    "--q-c": (("Q del condensador", "ESR = X/Q (por defecto ideal)."), ("Q del condensatore", "ESR = X/Q (predefinito ideale).")),
    "--balanced": (("Mostrar también la versión balanceada", ""), ("Mostra anche la versione bilanciata", "")),
    "--design-freq": (("Frecuencia de diseño (MHz)", "Sweep: frecuencia de la red."), ("Frequenza di progetto (MHz)", "Sweep: frequenza della rete.")),
    "--assume-constant-z": (("Suponer Z constante", "Usa R/X y fmin/fmax/paso."), ("Supporre Z costante", "Usa R/X e fmin/fmax/passo.")),
    "--fmin": (("Fmín (MHz)", "Solo con Z constante."), ("Fmin (MHz)", "Solo con Z costante.")),
    "--fmax": (("Fmáx (MHz)", "Solo con Z constante."), ("Fmax (MHz)", "Solo con Z costante.")),
    "--step": (("Paso (MHz)", "Solo con Z constante."), ("Passo (MHz)", "Solo con Z costante.")),
    "--commercial": (("Valores comerciales", "SERIE,PARALELO en nH/pF; varios pares separados por ';'. Por defecto: el E24 más cercano."),
                     ("Valori commerciali", "SERIE,PARALLELO in nH/pF; più coppie separate da ';'. Predefinito: E24 più vicino.")),
    "--sweep-csv": (("CSV del barrido", ""), ("CSV della scansione", "")),
    "--tol-l": (("Tolerancia del inductor (%)", "0 = desactivado."), ("Tolleranza induttore (%)", "0 = disattivato.")),
    "--tol-c": (("Tolerancia del condensador (%)", "0 = desactivado."), ("Tolleranza condensatore (%)", "0 = disattivato.")),
    "--line-allowance-mm": (("Margen de línea (mm)", "Longitud extra de línea para conexiones y el cruce (por defecto 20 bifilar, 40 coaxial); una línea más corta que la distancia entre puntos de alimentación más este margen no se puede construir."),
                            ("Margine della linea (mm)", "Lunghezza extra di linea per collegamenti e incrocio (predefinito 20 bifilare, 40 coassiale); una linea più corta della distanza tra i punti di alimentazione più questo margine non si può costruire.")),
    "--height-sweep": (("Barrido de altura (m)", "MIN,MAX,N sobre tierra: ángulo de salida, ganancia, F/B y Zin en N alturas. Requiere nec2c."),
                       ("Scansione di altezza (m)", "MIN,MAX,N sul terreno: angolo di partenza, guadagno, F/B e Zin a N altezze. Richiede nec2c.")),
    "--robustness": (("Tabla de robustez", "Efecto de un boom metálico, un mástil que rompe la simetría y una malla de coaxial sin choke, cada uno por separado. Requiere nec2c."),
                     ("Tabella di robustezza", "Effetto di un boom metallico, un palo che rompe la simmetria e una calza del coassiale senza choke, ciascuno da solo. Richiede nec2c.")),
    "--check-convergence": (("Convergencia de segmentación", "Vuelve a simular con 2n+1 segmentos e informa del cambio de ganancia, F/B y Zin (la incertidumbre de Zin). Por defecto: activado en optimize, desactivado en generate. Requiere nec2c."),
                            ("Convergenza della segmentazione", "Risimula con 2n+1 segmenti e riporta la variazione di guadagno, F/B e Zin (l'incertezza di Zin). Predefinito: attivo in optimize, disattivo in generate. Richiede nec2c.")),
    "--lnet-solution": (("Solución de la red L", "plus / minus (raíz), lowpass (L serie, C paralelo), highpass (C serie, L paralelo) o best (menor VSWR en el peor punto del barrido, ambas topologías)."),
                        ("Soluzione della rete L", "plus / minus (radice), lowpass (L serie, C parallelo), highpass (C serie, L parallelo) o best (VSWR peggiore più basso nella scansione, entrambe le topologie).")),
    "--esl-nh": (("ESL del condensador por patilla (nH)", "Comprobación de autorresonancia."),
                 ("ESL del condensatore per reoforo (nH)", "Verifica di autorisonanza.")),
    "--coil-cp-pf": (("C parásita de la bobina (pF)", "Comprobación de autorresonancia."),
                     ("C parassita della bobina (pF)", "Verifica di autorisonanza.")),
}


STR.update({
    "el.reflector": ("Reflector", "Reflector", "Riflettore"),
    "el.director": ("Director", "Director", "Direttore"),
    "sum.parasitics": ("Reflectors / directors", "Reflectores / directores", "Riflettori / direttori"),
})

OPT_TR.update({
    "--kernel": (("Núcleo de NEC", "auto: núcleo de hilo fino extendido (EK) cuando un segmento es más corto que 8 radios del hilo; thin / extended fuerzan uno."),
                 ("Nucleo di NEC", "auto: nucleo a filo sottile esteso (EK) quando un segmento è più corto di 8 raggi del filo; thin / extended ne forzano uno.")),
    "--strict-zin": (("Detener si no hay ajuste de Zin", "Termina con error si ningún candidato cumple la ventana de Zin (por defecto: escribe el más cercano, marcado)."),
                     ("Fermare se Zin non rientra", "Termina con errore se nessun candidato rispetta la finestra di Zin (predefinito: scrive il più vicino, segnalato).")),
    "--boom-clearance-mm": (("Separación del boom (mm)", "Hueco entre las superficies de los elementos y del boom modelado (por defecto 10)."),
                            ("Distanza dal boom (mm)", "Spazio tra le superfici degli elementi e del boom modellato (predefinito 10).")),
    "--ideal-drive": (("Excitación ideal", "Red ideal: fija las corrientes de los elementos (por defecto) o las tensiones de las fuentes."),
                      ("Eccitazione ideale", "Rete ideale: fissa le correnti degli elementi (predefinito) o le tensioni delle sorgenti.")),
    "--series-c-q": (("Q del C serie", "Q del condensador serie (ESR = |X|/Q sumada a Zin); por defecto 500, 0 = sin pérdidas."),
                     ("Q del C serie", "Q del condensatore serie (ESR = |X|/Q sommata a Zin); predefinito 500, 0 = senza perdite.")),
    "--line-loss-db-per-m": (("Pérdida de la línea (dB/m)", "Hoja de construcción: pérdida de la línea de fase; se informa como cota superior (0 = no se informa)."),
                             ("Perdita della linea (dB/m)", "Scheda di costruzione: perdita della linea di fase; riportata come limite superiore (0 = non riportata).")),
    "--tol-tap-mm": (("MC: tol. de toma (mm)", "Posición de la toma del T-match, cada elemento."),
                     ("MC: toll. presa (mm)", "Posizione della presa del T-match, ogni elemento.")),
    "--tol-bar-mm": (("MC: tol. de barra T (mm)", "Separación de la barra T, cada elemento."),
                     ("MC: toll. barra T (mm)", "Distanza della barra T, ogni elemento.")),
    "--tol-vf-pct": (("MC: tol. VF de línea (%)", "Factor de velocidad de la línea de fase."),
                     ("MC: toll. VF linea (%)", "Fattore di velocità della linea di fase.")),
    "--tol-z0-pct": (("MC: tol. Z0 de línea (%)", "Z0 de la línea de fase o de la red LC."),
                     ("MC: toll. Z0 linea (%)", "Z0 della linea di fase o della rete LC.")),
    "--tol-c-pct": (("MC: tol. C serie (%)", "Ajuste del condensador serie."),
                    ("MC: toll. C serie (%)", "Regolazione del condensatore serie.")),
    "--tol-retune": (("MC: reajustar por muestra", "Reajusta el C serie (auto) y rediseña la red de adaptación en cada muestra."),
                     ("MC: ritarare per campione", "Ritara il C serie (auto) e riprogetta la rete di adattamento per ogni campione.")),
    "--loss-weight": (("Peso de la pérdida de red", "front-back con la puntuación con red L: dB de puntuación por dB de pérdida de la red (0 = F/B puro)."),
                      ("Peso della perdita di rete", "front-back con il punteggio con rete L: dB di punteggio per dB di perdita della rete (0 = F/B puro).")),
    "--reflectors": (("Reflectores (0-2)", "Reflectores pasivos detrás del elemento trasero. Longitud y separación se precalculan (mínimo / nominal / máximo)."),
                     ("Riflettori (0-2)", "Riflettori passivi dietro l'elemento posteriore. Lunghezza e spaziatura sono precalcolate (minimo / nominale / massimo).")),
    "--directors": (("Directores (0-5)", "Directores pasivos delante del elemento delantero. Longitud y separación se precalculan (mínimo / nominal / máximo)."),
                    ("Direttori (0-5)", "Direttori passivi davanti all'elemento anteriore. Lunghezza e spaziatura sono precalcolate (minimo / nominale / massimo).")),
    "--reflector-lengths": (("Longitudes de reflectores", "Factores de la longitud resonante F1,F2. Vacío = nominal precalculado (1.05)."),
                            ("Lunghezze dei riflettori", "Fattori della lunghezza risonante F1,F2. Vuoto = nominale precalcolato (1.05).")),
    "--reflector-spacings": (("Separaciones de reflectores", "En longitudes de onda, cada una respecto al elemento anterior (el primero, del trasero). Vacío = 0.15, 0.20."),
                             ("Spaziature dei riflettori", "In lunghezze d'onda, ciascuna rispetto all'elemento precedente (il primo, dal posteriore). Vuoto = 0.15, 0.20.")),
    "--director-lengths": (("Longitudes de directores", "Factores de la longitud resonante F1..F5. Vacío = nominal precalculado."),
                           ("Lunghezze dei direttori", "Fattori della lunghezza risonante F1..F5. Vuoto = nominale precalcolato.")),
    "--director-spacings": (("Separaciones de directores", "En longitudes de onda, cada una respecto al elemento anterior (el primero, del delantero). Vacío = 0.15, 0.20, 0.25, 0.30, 0.30."),
                            ("Spaziature dei direttori", "In lunghezze d'onda, ciascuna rispetto all'elemento precedente (il primo, dall'anteriore). Vuoto = 0.15, 0.20, 0.25, 0.30, 0.30.")),
    "--parasitic-diameter": (("Diámetro parásitos (mm)", "Diámetro del tubo de reflectores y directores. Por defecto: el diámetro de elemento."),
                             ("Diametro parassiti (mm)", "Diametro del tubo di riflettori e direttori. Predefinito: il diametro dell'elemento.")),
    "--min-gap-mm": (("Hueco mínimo entre elementos (mm)", "Hueco mínimo entre superficies de elementos vecinos (abrazaderas y aisladores). Fija, junto con 0.05 lambda, la separación mínima."),
                     ("Spazio minimo tra elementi (mm)", "Spazio minimo tra le superfici di elementi vicini (morsetti e isolatori). Fissa, insieme a 0.05 lambda, la spaziatura minima.")),
    "--boom-overhang-mm": (("Saliente del boom (mm)", "Boom a cada lado más allá de los elementos exteriores. Con reflectores/directores el boom se alarga solo."),
                           ("Sporgenza del boom (mm)", "Boom oltre gli elementi esterni a ciascuna estremità. Con riflettori/direttori il boom si allunga da solo.")),
    "--parasitic-length-offsets": (("Desfases de longitud de parásitos", "Desfases de rejilla sumados a TODOS los factores de longitud, p. ej. -0.02,0,0.02. Se limitan a los límites precalculados."),
                                   ("Scostamenti lunghezza parassiti", "Scostamenti di griglia sommati a TUTTI i fattori di lunghezza, es. -0.02,0,0.02. Limitati ai valori precalcolati.")),
    "--parasitic-spacing-scales": (("Escalas de separación de parásitos", "Multiplicadores de rejilla de TODAS las separaciones, p. ej. 0.9,1,1.1. Se limitan a los límites precalculados."),
                                   ("Scale di spaziatura parassiti", "Moltiplicatori di griglia di TUTTE le spaziature, es. 0.9,1,1.1. Limitati ai valori precalcolati.")),
})


def opt_label(o) -> str:
    if _LANG == "en" or o.flag not in OPT_TR:
        return o.label
    return OPT_TR[o.flag][_LANG_IDX[_LANG] - 1][0]


def opt_tip(o) -> str:
    if _LANG == "en" or o.flag not in OPT_TR:
        return o.tip
    return OPT_TR[o.flag][_LANG_IDX[_LANG] - 1][1]


def opt_ph(o) -> str:
    return {"(automatic)": tr("ph.auto"), "(none)": tr("ph.none")}.get(o.ph, o.ph)


# ---- extended help shown by the orange "?" badge: flag -> (English, Spanish, Italian).  Max 2048 characters
# per language including the automatic "modes / default" lines (see opt_help()).
HELP_MAX = 2048

STR.update({
    "help.modes": ("Used in modes: {m}", "Se usa en los modos: {m}", "Usata nelle modalità: {m}"),
    "help.default": ("Default value: {d}", "Valor por defecto: {d}", "Valore predefinito: {d}"),
    "help.badge": ("Hover for help", "Pasa el ratón para ver la ayuda", "Passa il mouse per l'aiuto"),
})

HELP = {
    "--freq": (
        "Design frequency in MHz. In generate/optimize it sets the wavelength used to size the elements (rear about 150/f metres, front about 139/f metres) and the element spacing, and it is the frequency at which nec2c computes the pattern. In optimize, if you give a band, the geometry is still fixed at this frequency (default: band centre). In match mode with a NEC output file it must be one of the frequencies stored in that file; leave it empty to use the only one it contains.",
        "Frecuencia de diseño en MHz. En generate/optimize fija la longitud de onda con la que se dimensionan los elementos (trasero unos 150/f metros, delantero unos 139/f metros) y la separación, y es la frecuencia a la que nec2c calcula el diagrama. En optimize, si indicas una banda, la geometría sigue fijada a esta frecuencia (por defecto: el centro de la banda). En match con un archivo de salida de NEC debe ser una de las frecuencias que contiene; déjala vacía para usar la única que haya.",
        "Frequenza di progetto in MHz. In generate/optimize determina la lunghezza d'onda con cui si dimensionano gli elementi (posteriore circa 150/f metri, anteriore circa 139/f metri) e la spaziatura, ed è la frequenza a cui nec2c calcola il diagramma. In optimize, se indichi una banda, la geometria resta fissata a questa frequenza (predefinita: centro banda). In match con un file di uscita NEC deve essere una delle frequenze presenti nel file; lasciala vuota per usare l'unica contenuta."),
    "--network": (
        "How the two elements are fed. tl: a single EX source on one element and an ideal two-wire transmission line (TL card) carrying the signal to the other element; it is the classic HB9CV phasing line. lc: a single source and an ideal lumped pi network of L and C between the feed points. ideal: two sources with chosen amplitude and phase, useful as a theoretical reference; by default ('Ideal drive' current) they prescribe the element currents. The TL and LC networks are lossless and ideal: they model the phase shift, not every detail of a real line.",
        "Cómo se alimentan los dos elementos. tl: una sola fuente EX en un elemento y una línea de transmisión bifilar ideal (tarjeta TL) que lleva la señal al otro elemento; es la línea de fase clásica de la HB9CV. lc: una sola fuente y una red ideal en pi con L y C entre los puntos de alimentación. ideal: dos fuentes con amplitud y fase elegidas, útil como referencia teórica; por defecto ('Excitación ideal' current) fijan las corrientes de los elementos. Las redes TL y LC son ideales y sin pérdidas: modelan el desfase, no todos los detalles de una línea real.",
        "Come vengono alimentati i due elementi. tl: una sola sorgente EX su un elemento e una linea di trasmissione bifilare ideale (scheda TL) che porta il segnale all'altro elemento; è la classica linea di fase della HB9CV. lc: una sola sorgente e una rete ideale a pi con L e C tra i punti di alimentazione. ideal: due sorgenti con ampiezza e fase scelte, utile come riferimento teorico; per impostazione predefinita ('Eccitazione ideale' current) fissano le correnti degli elementi. Le reti TL e LC sono ideali e senza perdite: modellano lo sfasamento, non tutti i dettagli di una linea reale."),
    "--feed-element": (
        "Chooses which element receives the single EX source when the network is tl or lc. In the classic HB9CV the front (director side) element is fed and a crossed line of about 45 electrical degrees carries the signal to the rear one, giving the 225 degree delay needed for a cardioid pattern. Feeding the rear element instead changes the required delay (180 minus the spacing in degrees) and is mainly an alternative to evaluate.",
        "Elige qué elemento recibe la única fuente EX cuando la red es tl o lc. En la HB9CV clásica se alimenta el elemento delantero y una línea cruzada de unos 45 grados eléctricos lleva la señal al trasero, dando el retardo de 225 grados necesario para un diagrama en cardioide. Alimentar el trasero cambia el retardo necesario (180 menos la separación en grados) y es sobre todo una alternativa para evaluar.",
        "Sceglie quale elemento riceve l'unica sorgente EX quando la rete è tl o lc. Nella HB9CV classica si alimenta l'elemento anteriore e una linea incrociata di circa 45 gradi elettrici porta il segnale al posteriore, ottenendo il ritardo di 225 gradi necessario per un diagramma a cardioide. Alimentare il posteriore cambia il ritardo richiesto (180 meno la spaziatura in gradi) ed è soprattutto un'alternativa da valutare."),
    "--objective": (
        "Scoring criterion used by the optimizer. front-back maximizes the front-to-back ratio, gain maximizes forward gain, balance combines both using the weights 'Weight: gain' and 'Weight: F/B'. In generate mode it is informative only: it is written to the comments of the NEC file but does not change the geometry or the excitation.",
        "Criterio de puntuación del optimizador. front-back maximiza la relación delante/atrás, gain maximiza la ganancia frontal y balance combina ambas con los pesos 'Peso: ganancia' y 'Peso: F/B'. En el modo generate es solo informativo: se escribe en los comentarios del archivo NEC pero no cambia la geometría ni la excitación.",
        "Criterio di valutazione usato dall'ottimizzatore. front-back massimizza il rapporto avanti/indietro, gain massimizza il guadagno in avanti, balance combina entrambi usando i pesi 'Peso: guadagno' e 'Peso: F/B'. In generate è solo informativo: viene scritto nei commenti del file NEC ma non cambia geometria né eccitazione."),
    "--mount": (
        "How the elements are fixed to the boom. isolated (default): the elements are insulated from the boom; the boom is a separate conductor (modelled only with 'Model the boom'). contact: the element centres (driven, reflectors and directors) are bonded to a metallic boom by their clamps; the boom is modelled with a short bond wire per element. It needs the T-match feed, where the element centres are at a voltage null, which is the classic HB9CV built on a metal boom. With a centre feed the bond would short the feed points, so contact falls back to isolated plus a modelled boom.",
        "Cómo se fijan los elementos al boom. isolated (por defecto): los elementos están aislados del boom; el boom es un conductor aparte (solo se modela con 'Modelar el boom'). contact: los centros de los elementos (activos, reflectores y directores) se unen a un boom metálico mediante sus abrazaderas; el boom se modela con un hilo corto de unión por elemento. Requiere la alimentación T-match, en la que los centros de los elementos están en un nulo de tensión: la HB9CV clásica sobre boom metálico. Con alimentación central la unión cortocircuitaría los puntos de alimentación, así que contact pasa a isolated más un boom modelado.",
        "Come sono fissati gli elementi al boom. isolated (predefinito): gli elementi sono isolati dal boom; il boom è un conduttore a parte (modellato solo con 'Modellare il boom'). contact: i centri degli elementi (attivi, riflettori e direttori) sono collegati a un boom metallico dai loro morsetti; il boom è modellato con un breve filo di collegamento per elemento. Richiede l'alimentazione T-match, in cui i centri degli elementi sono a un nullo di tensione: la HB9CV classica su boom metallico. Con l'alimentazione centrale il collegamento cortocircuiterebbe i punti di alimentazione, quindi contact diventa isolated più un boom modellato."),
    "--balun-ratio": (
        "Ratio of the balun at the feed point. 1:1 (default) is a current balun or common-mode choke between the unbalanced coax and the balanced antenna; one is mandatory in a real build because the NEC source is an ideal balanced source. 4:1 changes the balanced-side reference impedance to 50/4 = 12.5 ohm (default target 12.5 ohm and maximum |X_in| of 10 ohm in optimize; the L-network in match/sweep is designed against 12.5 ohm). It is an option to evaluate, not a proven fix.",
        "Relación del balun en el punto de alimentación. 1:1 (por defecto) es un balun de corriente o choke de modo común entre el coaxial no balanceado y la antena balanceada; en una construcción real es obligatorio porque la fuente de NEC es una fuente balanceada ideal. 4:1 cambia la impedancia de referencia del lado balanceado a 50/4 = 12,5 ohm (objetivo por defecto 12,5 ohm y |X_in| máximo de 10 ohm en optimize; la red en L de match/sweep se diseña contra 12,5 ohm). Es una opción para evaluar, no una solución probada.",
        "Rapporto del balun nel punto di alimentazione. 1:1 (predefinito) è un balun di corrente o choke di modo comune tra il coassiale sbilanciato e l'antenna bilanciata; in una costruzione reale è obbligatorio perché la sorgente NEC è una sorgente bilanciata ideale. 4:1 porta l'impedenza di riferimento lato bilanciato a 50/4 = 12,5 ohm (obiettivo predefinito 12,5 ohm e |X_in| massimo 10 ohm in optimize; la rete a L in match/sweep è progettata su 12,5 ohm). È un'opzione da valutare, non una soluzione dimostrata."),
    "--target-impedance": (
        "Impedance, in ohm, that the balanced side of the antenna should present (the reference for VSWR). The default is 50 ohm, or 12.5 ohm when the balun ratio is 4:1. In generate mode it is informative only (written to the comments); in optimize it is the reference used to compute VSWR_vs_target and the VSWR penalty of each candidate.",
        "Impedancia, en ohm, que debería presentar el lado balanceado de la antena (la referencia del VSWR). Por defecto 50 ohm, o 12,5 ohm con balun 4:1. En generate es solo informativa (se escribe en los comentarios); en optimize es la referencia con la que se calcula VSWR_vs_target y la penalización por VSWR de cada candidato.",
        "Impedenza, in ohm, che il lato bilanciato dell'antenna dovrebbe presentare (il riferimento per il VSWR). Predefinita 50 ohm, oppure 12,5 ohm con balun 4:1. In generate è solo informativa (scritta nei commenti); in optimize è il riferimento con cui si calcolano VSWR_vs_target e la penalità VSWR di ogni candidato."),
    "--segments": (
        "Number of wire segments used for each element in the NEC model. It must be odd (so a segment sits exactly at the centre, where the source goes) and at least 5. More segments give a finer current distribution and more accurate results but increase the nec2c run time, especially in optimize. The program warns if the segments break the thin-wire rules (segment length at least 2 radii and at most lambda/10). Default 21, or 41 with the T-match feed, whose junctions need finer segments; the average gain test reports the accuracy.",
        "Número de segmentos de hilo de cada elemento en el modelo NEC. Debe ser impar (para que haya un segmento justo en el centro, donde va la fuente) y al menos 5. Más segmentos dan una distribución de corriente más fina y resultados más precisos, pero aumentan el tiempo de nec2c, sobre todo en optimize. El programa avisa si los segmentos incumplen las reglas de hilo fino (longitud de segmento de al menos 2 radios y como máximo lambda/10). Por defecto 21, o 41 con la alimentación T-match, cuyas uniones necesitan segmentos más finos; la prueba de ganancia media informa de la precisión.",
        "Numero di segmenti di filo per ogni elemento nel modello NEC. Deve essere dispari (così un segmento è esattamente al centro, dove va la sorgente) e almeno 5. Più segmenti danno una distribuzione di corrente più fine e risultati più precisi, ma aumentano il tempo di nec2c, soprattutto in optimize. Il programma avvisa se i segmenti violano le regole del filo sottile (lunghezza segmento almeno 2 raggi e al massimo lambda/10). Predefinito 21, oppure 41 con l'alimentazione T-match, le cui giunzioni richiedono segmenti più fini; il test del guadagno medio riporta la precisione."),
    "--az-step": (
        "Step in degrees of the azimuth radiation pattern written in the NEC file (RP card). It must divide 180 exactly (1, 2, 5, 10...), so that the ring holds the back direction, and be at least 0.01. A smaller step gives a smoother pattern and a finer front/back and beamwidth reading but increases run time. The GUI plots use their own full-sphere step (the 'step' selector on the left).",
        "Paso en grados del diagrama de radiación en azimut escrito en el archivo NEC (tarjeta RP). Debe dividir 180 exactamente (1, 2, 5, 10...), para que el anillo contenga la dirección trasera, y ser al menos 0,01. Un paso menor da un diagrama más suave y una lectura más fina de delante/atrás y de la anchura de haz, pero aumenta el tiempo. Los gráficos de la interfaz usan su propio paso de esfera completa (el selector 'paso' de la izquierda).",
        "Passo in gradi del diagramma di radiazione in azimut scritto nel file NEC (scheda RP). Deve dividere esattamente 180 (1, 2, 5, 10...), così che l'anello contenga la direzione posteriore, ed essere almeno 0,01. Un passo minore dà un diagramma più liscio e una lettura più fine di avanti/indietro e apertura, ma aumenta il tempo. I grafici dell'interfaccia usano un proprio passo sulla sfera completa (il selettore 'passo' a sinistra)."),
    "--conductivity": (
        "Electrical conductivity of all wires, in siemens per metre, written as an LD 5 card so nec2c includes conductor losses. Default 2.5e7 (6061 aluminium); 6063 aluminium is about 3.0e7, brass 1.5e7, copper 5.8e7. 0 means perfect conductors (no LD card). With 'Run nec2c' and at the end of optimize the model is simulated again with perfect conductors and the change in gain, F/B and R_in is reported as an efficiency estimate.",
        "Conductividad eléctrica de todos los hilos, en siemens por metro, escrita como tarjeta LD 5 para que nec2c incluya las pérdidas del conductor. Por defecto 2.5e7 (aluminio 6061); el aluminio 6063 ronda 3.0e7, el latón 1.5e7 y el cobre 5.8e7. 0 significa conductores perfectos (sin tarjeta LD). Con 'Ejecutar nec2c' y al final de optimize el modelo se simula de nuevo con conductores perfectos y se informa del cambio de ganancia, F/B y R_in como estimación de eficiencia.",
        "Conducibilità elettrica di tutti i fili, in siemens al metro, scritta come scheda LD 5 così nec2c include le perdite dei conduttori. Predefinita 2.5e7 (alluminio 6061); l'alluminio 6063 è circa 3.0e7, l'ottone 1.5e7, il rame 5.8e7. 0 significa conduttori perfetti (nessuna scheda LD). Con 'Esegui nec2c' e al termine di optimize il modello viene simulato di nuovo con conduttori perfetti e si riporta la variazione di guadagno, F/B e R_in come stima di efficienza."),
    "--kernel": (
        "NEC-2 kernel written to the deck. auto (default) uses the extended thin-wire kernel (EK card) when any wire has segments shorter than 8 wire radii, which happens with thick tubes and at the T-match junctions; there the plain thin-wire kernel loses accuracy and the input impedance keeps moving as the segmentation is refined, while with EK it converges (for the default 2 m model |delta Zin| between n and 2n+1 segments drops from 7 % to under 4 %). thin and extended force one kernel. Every run is checked for a physical power budget and discarded if it fails. The command-line option --extended-kernel is an older name of extended.",
        "Núcleo de NEC-2 escrito en el modelo. auto (por defecto) usa el núcleo de hilo fino extendido (tarjeta EK) cuando algún hilo tiene segmentos más cortos que 8 radios del hilo, lo que ocurre con tubos gruesos y en las uniones del T-match; allí el núcleo de hilo fino pierde precisión y la impedancia de entrada sigue moviéndose al refinar la segmentación, mientras que con EK converge (en el modelo de 2 m por defecto |delta Zin| entre n y 2n+1 segmentos baja del 7 % a menos del 4 %). thin y extended fuerzan un núcleo. Cada ejecución se comprueba con un balance de potencia físico y se descarta si no lo cumple. La opción de línea de comandos --extended-kernel es un nombre antiguo de extended.",
        "Nucleo di NEC-2 scritto nel modello. auto (predefinito) usa il nucleo a filo sottile esteso (scheda EK) quando un filo ha segmenti più corti di 8 raggi del filo, come avviene con tubi spessi e alle giunzioni del T-match; lì il nucleo a filo sottile perde precisione e l'impedenza d'ingresso continua a muoversi raffinando la segmentazione, mentre con EK converge (nel modello predefinito per i 2 m |delta Zin| tra n e 2n+1 segmenti scende dal 7 % a meno del 4 %). thin ed extended forzano un nucleo. Ogni esecuzione viene verificata con un bilancio di potenza fisico e scartata se non lo rispetta. L'opzione da riga di comando --extended-kernel è un vecchio nome di extended."),
    "--output": (
        "Name of the NEC input file to write. If left empty, generate derives a name from the parameters (for example hb9cv_145MHz_tl_isolated.nec) and optimize writes hb9cv_optimized.nec. Relative names are created inside the work folder. The GUI reads this file back to draw the antenna and compute the radiation patterns.",
        "Nombre del archivo de entrada NEC que se escribe. Si se deja vacío, generate deduce un nombre de los parámetros (por ejemplo hb9cv_145MHz_tl_isolated.nec) y optimize escribe hb9cv_optimized.nec. Los nombres relativos se crean dentro de la carpeta de trabajo. La interfaz vuelve a leer este archivo para dibujar la antena y calcular los diagramas de radiación.",
        "Nome del file di ingresso NEC da scrivere. Se vuoto, generate ricava un nome dai parametri (per esempio hb9cv_145MHz_tl_isolated.nec) e optimize scrive hb9cv_optimized.nec. I nomi relativi vengono creati nella cartella di lavoro. L'interfaccia rilegge questo file per disegnare l'antenna e calcolare i diagrammi di radiazione."),
    "--build-sheet": (
        "Writes a build sheet: a markdown file with the name you give and the same data as JSON next to it. It contains the cut list of the elements (NEC and cut lengths), spacing and boom, the phasing line, feed point and balun, the predicted performance, a tuning procedure and a bill of materials. In generate the performance section needs 'Run nec2c'; in optimize the network must be tl or lc.",
        "Escribe una hoja de construcción: un archivo markdown con el nombre indicado y los mismos datos en JSON a su lado. Contiene la lista de cortes de los elementos (longitudes NEC y de corte), separación y boom, la línea de fase, el punto de alimentación y el balun, el rendimiento previsto, un procedimiento de ajuste y una lista de materiales. En generate la sección de rendimiento necesita 'Ejecutar nec2c'; en optimize la red debe ser tl o lc.",
        "Scrive una scheda di costruzione: un file markdown con il nome indicato e gli stessi dati in JSON accanto. Contiene l'elenco dei tagli degli elementi (lunghezze NEC e di taglio), spaziatura e boom, la linea di fase, il punto di alimentazione e il balun, le prestazioni previste, una procedura di taratura e una distinta materiali. In generate la sezione prestazioni richiede 'Esegui nec2c'; in optimize la rete deve essere tl o lc."),
    "--run": (
        "After writing the NEC file, run nec2c on it and print a summary: gain in dBi and dBd, take-off angle, half-power beamwidth, front-to-back ratio, element currents and the conductor-loss check (the model is simulated again with perfect conductors). Without it only the model file is generated. nec2c must be installed and reachable (PATH or the 'nec2c' field).",
        "Tras escribir el archivo NEC, ejecuta nec2c sobre él e imprime un resumen: ganancia en dBi y dBd, ángulo de radiación, anchura de haz a media potencia, relación delante/atrás, corrientes de los elementos y la comprobación de pérdidas de los conductores (el modelo se simula de nuevo con conductores perfectos). Sin ella solo se genera el archivo del modelo. nec2c debe estar instalado y accesible (PATH o el campo 'nec2c').",
        "Dopo aver scritto il file NEC, esegue nec2c e stampa un riepilogo: guadagno in dBi e dBd, angolo di radiazione, apertura a metà potenza, rapporto avanti/indietro, correnti degli elementi e verifica delle perdite dei conduttori (il modello viene simulato di nuovo con conduttori perfetti). Senza di essa si genera solo il file del modello. nec2c deve essere installato e raggiungibile (PATH o campo 'nec2c')."),
    "--spacing-factor": (
        "Distance between the rear and the front element, in wavelengths of the design frequency. The classic HB9CV uses 0.125 lambda (45 electrical degrees, about 258 mm at 145 MHz). The theoretical phase lead of the rear current is 180 minus the spacing in degrees (135 degrees for 0.125 lambda). Larger spacing usually adds gain bandwidth but lengthens the boom; the optimizer scans several values.",
        "Distancia entre el elemento trasero y el delantero, en longitudes de onda de la frecuencia de diseño. La HB9CV clásica usa 0,125 lambda (45 grados eléctricos, unos 258 mm a 145 MHz). El adelanto de fase teórico de la corriente trasera es 180 menos la separación en grados (135 grados para 0,125 lambda). Una separación mayor suele dar más ancho de banda de ganancia pero alarga el boom; el optimizador explora varios valores.",
        "Distanza tra l'elemento posteriore e quello anteriore, in lunghezze d'onda della frequenza di progetto. La HB9CV classica usa 0,125 lambda (45 gradi elettrici, circa 258 mm a 145 MHz). L'anticipo di fase teorico della corrente posteriore è 180 meno la spaziatura in gradi (135 gradi per 0,125 lambda). Una spaziatura maggiore di solito dà più banda di guadagno ma allunga il boom; l'ottimizzatore esplora diversi valori."),
    "--rear-length-factor": (
        "Scales the rear element length relative to the nominal 150/f metres (f in MHz), i.e. 1.0 gives about 1034 mm at 145 MHz. Allowed range 0.5 to 1.5. The rear element is normally a little longer than the front one. Mutual coupling and the element diameter shift the resonance, so the optimizer scans factors around 1.0.",
        "Escala la longitud del elemento trasero respecto a los 150/f metros nominales (f en MHz); es decir, 1,0 da unos 1034 mm a 145 MHz. Rango permitido de 0,5 a 1,5. El elemento trasero suele ser algo más largo que el delantero. El acoplamiento mutuo y el diámetro desplazan la resonancia, por lo que el optimizador explora factores alrededor de 1,0.",
        "Scala la lunghezza dell'elemento posteriore rispetto ai 150/f metri nominali (f in MHz), cioè 1,0 dà circa 1034 mm a 145 MHz. Intervallo ammesso da 0,5 a 1,5. L'elemento posteriore è di solito un po' più lungo dell'anteriore. L'accoppiamento mutuo e il diametro spostano la risonanza, quindi l'ottimizzatore esplora fattori intorno a 1,0."),
    "--front-length-factor": (
        "Scales the front element length relative to the nominal 139/f metres (f in MHz), i.e. 1.0 gives about 959 mm at 145 MHz. Allowed range 0.5 to 1.5. Shortening the front element moves its resonance up, which changes the input impedance and the F/B; the optimizer scans factors around 1.0.",
        "Escala la longitud del elemento delantero respecto a los 139/f metros nominales (f en MHz); es decir, 1,0 da unos 959 mm a 145 MHz. Rango permitido de 0,5 a 1,5. Acortar el elemento delantero sube su resonancia, lo que cambia la impedancia de entrada y el F/B; el optimizador explora factores alrededor de 1,0.",
        "Scala la lunghezza dell'elemento anteriore rispetto ai 139/f metri nominali (f in MHz), cioè 1,0 dà circa 959 mm a 145 MHz. Intervallo ammesso da 0,5 a 1,5. Accorciare l'elemento anteriore alza la sua risonanza, cambiando impedenza d'ingresso e F/B; l'ottimizzatore esplora fattori intorno a 1,0."),
    "--element-diameter": (
        "Diameter in mm of both elements (the default for each). It sets the wire radius in the model and therefore the electrical length and bandwidth: thicker tubes are shorter electrically and broader. The per-element diameters and the section lists override it. In optimize, 'Refine diameter' can move it within the diameter range.",
        "Diámetro en mm de ambos elementos (valor por defecto de cada uno). Fija el radio del hilo en el modelo y, por tanto, la longitud eléctrica y el ancho de banda: los tubos más gruesos son eléctricamente más cortos y de banda más ancha. Los diámetros por elemento y las listas de secciones lo sustituyen. En optimize, 'Refinar diámetro' puede moverlo dentro del rango de diámetros.",
        "Diametro in mm di entrambi gli elementi (il valore predefinito per ciascuno). Determina il raggio del filo nel modello e quindi lunghezza elettrica e banda: tubi più spessi sono elettricamente più corti e più larghi in banda. I diametri per elemento e gli elenchi di sezioni lo sostituiscono. In optimize 'Raffinare diametro' può variarlo entro l'intervallo dei diametri."),
    "--rear-diameter": (
        "Diameter in mm of the rear element only. If empty, the general element diameter is used. Cannot be combined with the rear sections list, which defines the diameters section by section.",
        "Diámetro en mm solo del elemento trasero. Si está vacío se usa el diámetro general de elemento. No se puede combinar con la lista de secciones traseras, que define los diámetros sección a sección.",
        "Diametro in mm solo dell'elemento posteriore. Se vuoto si usa il diametro generale degli elementi. Non può essere combinato con l'elenco delle sezioni posteriori, che definisce i diametri sezione per sezione."),
    "--front-diameter": (
        "Diameter in mm of the front element only. If empty, the general element diameter is used. Cannot be combined with the front sections list, which defines the diameters section by section.",
        "Diámetro en mm solo del elemento delantero. Si está vacío se usa el diámetro general de elemento. No se puede combinar con la lista de secciones delanteras, que define los diámetros sección a sección.",
        "Diametro in mm solo dell'elemento anteriore. Se vuoto si usa il diametro generale degli elementi. Non può essere combinato con l'elenco delle sezioni anteriori, che definisce i diametri sezione per sezione."),
    "--rear-sections": (
        "Models a stepped (telescoped) rear element. Give ONE half only, from the centre outward, as LENGTH_MM:DIAMETER_MM pairs separated by commas; the last item has only a diameter and runs to the tip. Example 300:12,200:10,8 means 300 mm of 12 mm tube, then 200 mm of 10 mm, then 8 mm to the tip. Both halves are identical. Lengths are measured from the element centre (half the feed gap included). Needs the isolated mount and excludes the rear diameter.",
        "Modela un elemento trasero escalonado (telescópico). Indica SOLO una mitad, del centro hacia fuera, como pares LONGITUD_MM:DIAMETRO_MM separados por comas; el último elemento solo lleva diámetro y llega hasta la punta. Ejemplo 300:12,200:10,8: 300 mm de tubo de 12 mm, luego 200 mm de 10 mm y después 8 mm hasta la punta. Ambas mitades son iguales. Las longitudes se miden desde el centro del elemento (incluida la mitad del hueco de alimentación). Requiere el montaje isolated y excluye el diámetro trasero.",
        "Modella un elemento posteriore a gradini (telescopico). Indica UNA SOLA metà, dal centro verso l'esterno, come coppie LUNGHEZZA_MM:DIAMETRO_MM separate da virgole; l'ultima voce ha solo il diametro e arriva fino alla punta. Esempio 300:12,200:10,8: 300 mm di tubo da 12 mm, poi 200 mm da 10 mm, poi 8 mm fino alla punta. Le due metà sono uguali. Le lunghezze si misurano dal centro dell'elemento (inclusa metà del gap di alimentazione). Richiede il montaggio isolated ed esclude il diametro posteriore."),
    "--front-sections": (
        "Same as the rear sections but for the front element: ONE half, from the centre outward, LENGTH_MM:DIAMETER_MM pairs separated by commas and a final diameter that runs to the tip (for example 280:12,180:10,8). Every model is checked against the thin-wire rules (neighbouring radius ratio up to 2:1) and warnings are printed.",
        "Igual que las secciones traseras pero para el elemento delantero: UNA mitad, del centro hacia fuera, pares LONGITUD_MM:DIAMETRO_MM separados por comas y un diámetro final que llega hasta la punta (por ejemplo 280:12,180:10,8). Cada modelo se comprueba con las reglas de hilo fino (relación de radios vecinos hasta 2:1) y se imprimen avisos.",
        "Come le sezioni posteriori ma per l'elemento anteriore: UNA metà, dal centro verso l'esterno, coppie LUNGHEZZA_MM:DIAMETRO_MM separate da virgole e un diametro finale che arriva fino alla punta (per esempio 280:12,180:10,8). Ogni modello è verificato con le regole del filo sottile (rapporto tra raggi vicini fino a 2:1) e vengono stampati gli avvisi."),
    "--boom-shape": (
        "Cross-section of the boom: square or round. It only matters when the boom is modelled (isolated boom) and in the build sheet. A square boom of side a is represented in NEC by an equivalent wire of radius 0.59 times a; a round boom uses its true radius.",
        "Sección del boom: cuadrada o redonda. Solo importa cuando el boom se modela (boom aislado) y en la hoja de construcción. Un boom cuadrado de lado a se representa en NEC con un hilo equivalente de radio 0,59 veces a; uno redondo usa su radio real.",
        "Sezione del boom: quadrata o tonda. Conta solo quando il boom è modellato (boom isolato) e nella scheda di costruzione. Un boom quadrato di lato a è rappresentato in NEC da un filo equivalente di raggio 0,59 volte a; uno tondo usa il suo raggio reale."),
    "--boom-size": (
        "Boom dimension in mm: the side for a square boom or the diameter for a round one. It determines the equivalent radius of the modelled boom and is listed in the build sheet. Default 25 mm.",
        "Dimensión del boom en mm: el lado si es cuadrado o el diámetro si es redondo. Determina el radio equivalente del boom modelado y figura en la hoja de construcción. Por defecto 25 mm.",
        "Dimensione del boom in mm: il lato se quadrato o il diametro se tondo. Determina il raggio equivalente del boom modellato e compare nella scheda di costruzione. Predefinita 25 mm."),
    "--boom-length": (
        "Total length of the boom in mm. It sets how far the modelled boom conductor extends along the beam axis and is reported in the build sheet; it should be at least the element spacing plus some margin at both ends. Default 700 mm.",
        "Longitud total del boom en mm. Fija hasta dónde se extiende el conductor del boom modelado a lo largo del eje del haz y se indica en la hoja de construcción; debería ser al menos la separación entre elementos más algo de margen en ambos extremos. Por defecto 700 mm.",
        "Lunghezza totale del boom in mm. Determina fin dove si estende il conduttore del boom modellato lungo l'asse del fascio e compare nella scheda di costruzione; dovrebbe essere almeno la spaziatura tra gli elementi più un po' di margine alle estremità. Predefinita 700 mm."),
    "--boom-isolated": (
        "Includes the boom in the NEC model as a metallic conductor with no electrical connection to the elements, 'Boom clearance' below their surfaces. By default the boom is not modelled at all. Turning it on shows its coupling effect (small shift of resonance and impedance). The boom is segmented by its own radius (segments of at least 4 radii), as the thin-wire kernel requires.",
        "Incluye el boom en el modelo NEC como un conductor metálico sin conexión eléctrica con los elementos, a la 'Separación del boom' por debajo de sus superficies. Por defecto el boom no se modela. Activarlo muestra su efecto de acoplamiento (pequeño desplazamiento de resonancia e impedancia). El boom se segmenta según su propio radio (segmentos de al menos 4 radios), como exige el núcleo de hilo fino.",
        "Include il boom nel modello NEC come conduttore metallico senza collegamento elettrico con gli elementi, alla 'Distanza dal boom' sotto le loro superfici. Per impostazione predefinita il boom non è modellato. Attivarlo mostra il suo effetto di accoppiamento (piccolo spostamento di risonanza e impedenza). Il boom è segmentato secondo il proprio raggio (segmenti di almeno 4 raggi), come richiede il nucleo a filo sottile."),
    "--boom-clearance-mm": (
        'Gap in mm between the surfaces of the elements (and T-bars) and the surface of a modelled boom (default 10). The common-mode shield wire runs beside the boom at the same gap. NEC-2 evaluates the fields on the wire axes, so conductors much closer than their radii couple inaccurately; every model is checked and conductors closer than half the larger radius are reported.',
        'Hueco en mm entre las superficies de los elementos (y de las barras T) y la del boom modelado (por defecto 10). El hilo de malla de modo común corre junto al boom con el mismo hueco. NEC-2 evalúa los campos en los ejes de los hilos, así que conductores mucho más cercanos que sus radios se acoplan con poca precisión; cada modelo se comprueba y se informa de los conductores más cercanos que medio radio mayor.',
        'Spazio in mm tra le superfici degli elementi (e delle barre T) e quella del boom modellato (predefinito 10). Il filo della calza di modo comune corre accanto al boom con lo stesso spazio. NEC-2 valuta i campi sugli assi dei fili, quindi conduttori molto più vicini dei loro raggi si accoppiano in modo impreciso; ogni modello viene verificato e si segnalano i conduttori più vicini di metà del raggio maggiore.'),
    "--feed-gap-mm": (
        "Gap in mm between the two halves of each element at the feed point (where the insulator is). In the cut list each half is (L_nec - g)/2 minus the end hardware; the gap is inside the tip-to-tip NEC length, so changing it does not change the electrical length of the model, but with the same tubes each extra millimetre of gap lengthens the element by 1 mm. See 'Model the feed gap in NEC'.",
        "Hueco en mm entre las dos mitades de cada elemento en el punto de alimentación (donde está el aislador). En la lista de cortes cada mitad es (L_nec - g)/2 menos el herraje de los extremos; el hueco está dentro de la longitud NEC de punta a punta, así que cambiarlo no altera la longitud eléctrica del modelo, pero con los mismos tubos cada milímetro extra de hueco alarga el elemento 1 mm. Véase 'Modelar el hueco de alimentación en NEC'.",
        "Spazio in mm tra le due metà di ogni elemento nel punto di alimentazione (dove c'è l'isolatore). Nell'elenco dei tagli ogni metà è (L_nec - g)/2 meno la minuteria di estremità; il gap è compreso nella lunghezza NEC da punta a punta, quindi cambiarlo non altera la lunghezza elettrica del modello, ma con gli stessi tubi ogni millimetro in più di gap allunga l'elemento di 1 mm. Vedi 'Modellare il gap di alimentazione in NEC'."),
    "--model-feed-gap": (
        "Also puts the feed gap in the NEC geometry: each element becomes two half wires running from plus/minus g/2 to the tips, plus a one-segment wire of length g at the centre that carries the source or the line port (NEC needs a connected segment for a port). The gap capacitance and the insulator are not modelled. The one-segment bridge must be at least 4 tube radii and lambda/200 long (thin-wire rules), so the option suits thin tubes or wide gaps. Not together with the common-mode feed-line wire.",
        "Pone también el hueco de alimentación en la geometría NEC: cada elemento pasa a ser dos medios hilos que van de más/menos g/2 hasta las puntas, más un hilo de un segmento de longitud g en el centro que lleva la fuente o el puerto de la línea (NEC necesita un segmento conectado para un puerto). No se modelan la capacidad del hueco ni el aislador. El puente de un segmento debe medir al menos 4 radios del tubo y lambda/200 (reglas de hilo delgado), así que la opción sirve para tubos finos o huecos anchos. No se combina con el hilo de línea de alimentación de modo común.",
        "Inserisce anche il gap di alimentazione nella geometria NEC: ogni elemento diventa due mezzi fili da più/meno g/2 alle punte, più un filo di un segmento di lunghezza g al centro che porta la sorgente o la porta della linea (NEC richiede un segmento collegato per una porta). La capacità del gap e l'isolatore non sono modellati. Il ponte di un segmento deve essere lungo almeno 4 raggi del tubo e lambda/200 (regole del filo sottile), quindi l'opzione è adatta a tubi sottili o gap ampi. Non si combina con il filo di linea di alimentazione di modo comune."),
    "--end-cap-mm": (
        "Build sheet only. Length in mm of conductor added beyond each tube end by end hardware (caps, bolts); the cut list subtracts it so the finished element has the NEC length. 0 means flat ends. It does not change the NEC model.",
        "Solo hoja de construcción. Longitud en mm de conductor que añade el herraje de cada extremo del tubo (tapones, tornillos); la lista de cortes la resta para que el elemento terminado tenga la longitud NEC. 0 significa extremos planos. No cambia el modelo NEC.",
        "Solo scheda di costruzione. Lunghezza in mm di conduttore aggiunta oltre ogni estremità del tubo dalla minuteria (tappi, bulloni); l'elenco dei tagli la sottrae in modo che l'elemento finito abbia la lunghezza NEC. 0 significa estremità piatte. Non cambia il modello NEC."),
    "--tube-wall-mm": (
        "Build sheet only. Wall thickness in mm of the element tube, used in the bill of materials and to describe the telescoped sections. It does not change the NEC model, which uses solid wires.",
        "Solo hoja de construcción. Espesor de pared en mm del tubo de los elementos, usado en la lista de materiales y para describir las secciones telescópicas. No cambia el modelo NEC, que usa hilos macizos.",
        "Solo scheda di costruzione. Spessore di parete in mm del tubo degli elementi, usato nella distinta materiali e per descrivere le sezioni telescopiche. Non cambia il modello NEC, che usa fili pieni."),
    "--boom-material": (
        "Build sheet only. Free text describing the boom and mount material (for example '25x25x2 mm aluminium square tube'), printed in the bill of materials. If empty, a text derived from the mount type is used.",
        "Solo hoja de construcción. Texto libre que describe el material del boom y del soporte (por ejemplo 'tubo cuadrado de aluminio 25x25x2 mm'), impreso en la lista de materiales. Si está vacío se usa un texto derivado del tipo de montaje.",
        "Solo scheda di costruzione. Testo libero che descrive il materiale di boom e supporto (per esempio 'tubo quadro di alluminio 25x25x2 mm'), stampato nella distinta materiali. Se vuoto si usa un testo ricavato dal tipo di montaggio."),
    "--line-type": (
        "Physical type of the phase line used with network tl. openwire (default): a balanced air line, crossed by construction, velocity factor 0.95, with Z0 computed from the wire diameter and spacing (119.9*acosh(D/d), exact for round wires). coax: a coaxial cable, always straight because a coax cannot be crossed, velocity factor 0.66 and Z0 from 'Line Z0'; it needs a balun or choke at its element end, which the NEC TL card does not model. auto picks openwire.",
        "Tipo físico de la línea de fase usada con la red tl. openwire (por defecto): línea de aire balanceada, cruzada por construcción, factor de velocidad 0,95 y Z0 calculada con el diámetro y la separación de los hilos (119.9*acosh(D/d), exacta para hilos redondos). coax: cable coaxial, siempre recto porque un coaxial no se puede cruzar, factor de velocidad 0,66 y Z0 de 'Z0 de la línea'; necesita un balun o choke en su extremo del elemento, que la tarjeta TL de NEC no modela. auto elige openwire.",
        "Tipo fisico della linea di fase usata con la rete tl. openwire (predefinito): linea bilanciata in aria, incrociata per costruzione, fattore di velocità 0,95 e Z0 calcolata da diametro e distanza dei fili (119,9*acosh(D/d), esatta per fili tondi). coax: cavo coassiale, sempre diritto perché un coassiale non si può incrociare, fattore di velocità 0,66 e Z0 da 'Z0 della linea'; richiede un balun o choke al suo estremo sull'elemento, che la scheda TL di NEC non modella. auto sceglie openwire."),
    "--crossed": (
        "Whether the phase line or LC network is crossed (the two conductors swap sides), which adds the extra 180 degrees of the classic HB9CV delay (180 + 45 = 225 degrees). In NEC a crossed TL is a negative Z0 and a crossed NT has an inverted Y12. default = crossed, except for coax lines which are always straight; yes with coax is an error. no gives a straight line.",
        "Indica si la línea de fase o la red LC está cruzada (los dos conductores intercambian lado), lo que añade los 180 grados extra del retardo clásico de la HB9CV (180 + 45 = 225 grados). En NEC una TL cruzada es una Z0 negativa y una NT cruzada tiene Y12 invertida. default = cruzada, salvo las líneas coaxiales, que siempre son rectas; yes con coax es un error. no da una línea recta.",
        "Indica se la linea di fase o la rete LC è incrociata (i due conduttori si scambiano di lato), il che aggiunge i 180 gradi extra del classico ritardo della HB9CV (180 + 45 = 225 gradi). In NEC una TL incrociata ha Z0 negativa e una NT incrociata ha Y12 invertita. default = incrociata, tranne le linee coassiali che sono sempre diritte; yes con coax è un errore. no dà una linea diritta."),
    "--line-wire-diameter": (
        'Open-wire line only: diameter d in mm of each of the two wires of the air line. Together with the spacing it sets the line impedance Z0 = 119.9*acosh(D/d). Default 3 mm. Below D/d = 3 the spacers and the proximity effect change Z0 and VF noticeably and a warning is printed.',
        'Solo línea abierta: diámetro d en mm de cada uno de los dos hilos de la línea de aire. Junto con la separación fija la impedancia de la línea Z0 = 119,9*acosh(D/d). Por defecto 3 mm. Por debajo de D/d = 3 los separadores y el efecto de proximidad cambian Z0 y VF de forma apreciable y se imprime un aviso.',
        "Solo linea aperta: diametro d in mm di ciascuno dei due fili della linea in aria. Insieme alla spaziatura determina l'impedenza della linea Z0 = 119,9*acosh(D/d). Predefinito 3 mm. Sotto D/d = 3 i distanziatori e l'effetto di prossimità cambiano Z0 e VF in modo apprezzabile e viene stampato un avviso."),
    "--line-spacing": (
        'Open-wire line only: centre-to-centre spacing D in mm between the two wires. Z0 = 119.9*acosh(D/d) (eta0/pi = 119.9 ohm), exact for round wires in air. The default of 20 mm with 3 mm wire gives about 310 ohm. Wider spacing raises Z0; the line length in the model is not affected, only the impedance. D must be at least 1.2 d.',
        'Solo línea abierta: separación D en mm entre los centros de los dos hilos. Z0 = 119,9*acosh(D/d) (eta0/pi = 119,9 ohm), exacta para hilos redondos en aire. Los 20 mm por defecto con hilo de 3 mm dan unos 310 ohm. Una separación mayor sube Z0; la longitud de la línea en el modelo no cambia, solo la impedancia. D debe ser al menos 1,2 d.',
        "Solo linea aperta: distanza D in mm tra i centri dei due fili. Z0 = 119,9*acosh(D/d) (eta0/pi = 119,9 ohm), esatta per fili tondi in aria. I 20 mm predefiniti con filo da 3 mm danno circa 310 ohm. Una distanza maggiore aumenta Z0; la lunghezza della linea nel modello non cambia, solo l'impedenza. D deve essere almeno 1,2 d."),
    "--line-z0": (
        "Characteristic impedance in ohm of the phase line when it is a coaxial cable (default 50 ohm; use 75 for 75-ohm cable). Ignored for open-wire lines, where Z0 is computed from the wire diameter and spacing.",
        "Impedancia característica en ohm de la línea de fase cuando es un cable coaxial (por defecto 50 ohm; usa 75 para cable de 75 ohm). Se ignora en líneas abiertas, donde Z0 se calcula con el diámetro y la separación de los hilos.",
        "Impedenza caratteristica in ohm della linea di fase quando è un cavo coassiale (predefinita 50 ohm; usa 75 per cavo da 75 ohm). Ignorata per le linee aperte, dove Z0 si calcola da diametro e distanza dei fili."),
    "--line-vf": (
        "Velocity factor of the phase line, greater than 0 and at most 1. Default 0.95 for an open-wire air line and 0.66 for solid-polyethylene coax. The physical length of the line is the electrical length divided by this factor, so a lower factor gives a shorter cable for the same phase shift.",
        "Factor de velocidad de la línea de fase, mayor que 0 y como máximo 1. Por defecto 0,95 para línea abierta de aire y 0,66 para coaxial de polietileno macizo. La longitud física de la línea es la longitud eléctrica dividida por este factor, así que un factor menor da un cable más corto para el mismo desfase.",
        "Fattore di velocità della linea di fase, maggiore di 0 e al massimo 1. Predefinito 0,95 per linea aperta in aria e 0,66 per coassiale in polietilene pieno. La lunghezza fisica della linea è la lunghezza elettrica divisa per questo fattore, quindi un fattore minore dà un cavo più corto per lo stesso sfasamento."),
    "--lc-z0": (
        "Characteristic impedance in ohm of the ideal pi network (network lc). It sets the L and C values that produce the required phase shift: L = Z0*sin(theta)/w and C = tan(theta/2)/(Z0*w). Default 50 ohm.",
        "Impedancia característica en ohm de la red en pi ideal (red lc). Fija los valores de L y C que producen el desfase necesario: L = Z0*sen(theta)/w y C = tan(theta/2)/(Z0*w). Por defecto 50 ohm.",
        "Impedenza caratteristica in ohm della rete a pi ideale (rete lc). Determina i valori di L e C che producono lo sfasamento richiesto: L = Z0*sen(theta)/w e C = tan(theta/2)/(Z0*w). Predefinita 50 ohm."),
    "--line-deg": (
        "Electrical length of the phase line or LC network in degrees (0 to 360). By default it is synthesized from the response of the elements (or, without synthesis, derived from theory) so that the far element current has the right amplitude and phase, and it is never shorter than the line can physically be. Typing a value forces it, which is useful to test sensitivity. With a transmission line, 'Line physical length' takes priority if both are given.",
        "Longitud eléctrica de la línea de fase o de la red LC en grados (0 a 360). Por defecto se sintetiza a partir de la respuesta de los elementos (o, sin síntesis, se deduce de la teoría) para que la corriente del elemento lejano tenga la amplitud y la fase correctas, y nunca es más corta de lo que la línea puede ser físicamente. Escribir un valor lo fuerza, lo que sirve para probar la sensibilidad. Con una línea de transmisión, 'Longitud física de la línea' tiene prioridad si se dan ambas.",
        "Lunghezza elettrica della linea di fase o della rete LC in gradi (0-360). Per impostazione predefinita è sintetizzata dalla risposta degli elementi (o, senza sintesi, ricavata dalla teoria) in modo che la corrente dell'elemento lontano abbia l'ampiezza e la fase giuste, e non è mai più corta di quanto la linea possa essere fisicamente. Digitare un valore la forza, utile per provare la sensibilità. Con una linea di trasmissione, 'Lunghezza fisica della linea' ha priorità se entrambe sono indicate."),
    "--phase-target-deg": (
        "Desired delay in degrees of the current in the far element relative to the fed one. By default it follows cardioid theory: spacing in degrees plus 180 when the front element is fed (225 for 0.125 lambda), or 180 minus the spacing when the rear is fed. Mutual coupling shifts the real value, so the optimizer scans around it.",
        "Retardo deseado, en grados, de la corriente del elemento lejano respecto al alimentado. Por defecto sigue la teoría del cardioide: la separación en grados más 180 si se alimenta el delantero (225 para 0,125 lambda), o 180 menos la separación si se alimenta el trasero. El acoplamiento mutuo desplaza el valor real, por lo que el optimizador explora a su alrededor.",
        "Ritardo desiderato, in gradi, della corrente nell'elemento lontano rispetto a quello alimentato. Per impostazione predefinita segue la teoria della cardioide: la spaziatura in gradi più 180 se si alimenta l'anteriore (225 per 0,125 lambda), oppure 180 meno la spaziatura se si alimenta il posteriore. L'accoppiamento mutuo sposta il valore reale, quindi l'ottimizzatore esplora intorno ad esso."),
    "--line-length": (
        "Physical length in metres of the phase line. If empty it is derived from the phase target: (degrees/360) * wavelength * velocity factor. Typing a length overrides the derived one, handy to model a line cut to a specific size. The NEC TL card receives the physical length divided by the velocity factor.",
        "Longitud física en metros de la línea de fase. Si está vacía se deduce del desfase objetivo: (grados/360) * longitud de onda * factor de velocidad. Escribir una longitud sustituye la calculada, útil para modelar una línea cortada a una medida concreta. La tarjeta TL de NEC recibe la longitud física dividida por el factor de velocidad.",
        "Lunghezza fisica in metri della linea di fase. Se vuota si ricava dallo sfasamento obiettivo: (gradi/360) * lunghezza d'onda * fattore di velocità. Digitare una lunghezza sostituisce quella ricavata, utile per modellare una linea tagliata a una misura precisa. La scheda TL di NEC riceve la lunghezza fisica divisa per il fattore di velocità."),
    "--lc-l-nh": (
        "Series inductor of the pi network in nH. Must be given together with the shunt capacitor; both replace the values derived from the target delay. Use it to test real component values. The network is ideal: no losses, parasitics or component radiation.",
        "Inductor serie de la red en pi en nH. Debe darse junto con el condensador paralelo; ambos sustituyen los valores deducidos del retardo objetivo. Úsalo para probar valores de componentes reales. La red es ideal: sin pérdidas, parásitos ni radiación de los componentes.",
        "Induttore serie della rete a pi in nH. Va indicato insieme al condensatore parallelo; entrambi sostituiscono i valori ricavati dal ritardo obiettivo. Usalo per provare valori di componenti reali. La rete è ideale: nessuna perdita, parassita o radiazione dei componenti."),
    "--lc-c-pf": (
        "Shunt capacitor at each port of the pi network in pF. Must be given together with the series inductor; both replace the values derived from the target delay. The network is ideal.",
        "Condensador paralelo en cada puerto de la red en pi en pF. Debe darse junto con el inductor serie; ambos sustituyen los valores deducidos del retardo objetivo. La red es ideal.",
        "Condensatore parallelo a ciascuna porta della rete a pi in pF. Va indicato insieme all'induttore serie; entrambi sostituiscono i valori ricavati dal ritardo obiettivo. La rete è ideale."),
    "--rear-ratio": (
        "Amplitude of the rear excitation relative to the front one when the network is ideal (two sources): of the element currents with 'Ideal drive' current (default), of the source voltages with voltage. 1.0 (default) means equal amplitudes, the cardioid relation. In optimize the 'Amplitude ratios' list scans several values.",
        "Amplitud de la excitación trasera relativa a la delantera cuando la red es ideal (dos fuentes): de las corrientes de los elementos con 'Excitación ideal' current (por defecto), de las tensiones de las fuentes con voltage. 1,0 (por defecto) significa amplitudes iguales, la relación de la cardioide. En optimize la lista 'Relaciones de amplitud' explora varios valores.",
        "Ampiezza dell'eccitazione posteriore relativa a quella anteriore quando la rete è ideale (due sorgenti): delle correnti degli elementi con 'Eccitazione ideale' current (predefinito), delle tensioni delle sorgenti con voltage. 1,0 (predefinito) significa ampiezze uguali, la relazione della cardioide. In optimize l'elenco 'Rapporti di ampiezza' esplora diversi valori."),
    "--ideal-drive": (
        'Ideal network only. current (default): the amplitude ratio and the phase prescribe the element CURRENTS, as the cardioid theory does: each source gets a lossless series reactance of 1e6 ohm, so the currents follow the source voltages whatever the mutual coupling between the elements. voltage: they prescribe the source voltages; with mutual coupling the currents then differ strongly from the requested relation.',
        'Solo red ideal. current (por defecto): la relación de amplitud y la fase fijan las CORRIENTES de los elementos, como hace la teoría del cardioide: cada fuente lleva una reactancia serie sin pérdidas de 1e6 ohm, así que las corrientes siguen a las tensiones de las fuentes sea cual sea el acoplamiento mutuo entre los elementos. voltage: fijan las tensiones de las fuentes; con acoplamiento mutuo las corrientes difieren entonces mucho de la relación pedida.',
        "Solo rete ideale. current (predefinito): il rapporto di ampiezza e la fase fissano le CORRENTI degli elementi, come fa la teoria della cardioide: ogni sorgente ha una reattanza serie senza perdite di 1e6 ohm, così le correnti seguono le tensioni delle sorgenti qualunque sia l'accoppiamento mutuo tra gli elementi. voltage: fissano le tensioni delle sorgenti; con l'accoppiamento mutuo le correnti differiscono allora molto dalla relazione richiesta."),
    "--rear-phase": (
        "Phase in degrees of the rear excitation for the ideal network (element current with 'Ideal drive' current, source voltage with voltage). Default 180 minus 360 times the spacing in wavelengths, i.e. 135 degrees for 0.125 lambda, the current relation that gives a cardioid with a null at the back for equal-amplitude currents.",
        "Fase en grados de la excitación trasera para la red ideal (corriente del elemento con 'Excitación ideal' current, tensión de la fuente con voltage). Por defecto 180 menos 360 veces la separación en longitudes de onda, es decir, 135 grados para 0,125 lambda, la relación de corrientes que da un cardioide con nulo atrás para corrientes de igual amplitud.",
        "Fase in gradi dell'eccitazione posteriore per la rete ideale (corrente dell'elemento con 'Eccitazione ideale' current, tensione della sorgente con voltage). Predefinita 180 meno 360 volte la spaziatura in lunghezze d'onda, cioè 135 gradi per 0,125 lambda, la relazione di correnti che dà una cardioide con nullo dietro per correnti di uguale ampiezza."),
    "--feedline-length-m": (
        "Common-mode test. Adds a wire of this length in metres representing the outer conductor of the coax, tied to the fed port and running rearward below the elements (beside a modelled boom, at 'Boom clearance' from it). It shows how current on the feed line can disturb the pattern when there is no good balun. 0 turns it off. The wire is an open-ended shield: an upper-bound sensitivity test, not a model of a real coax run or choke.",
        "Prueba de modo común. Añade un hilo de esta longitud en metros que representa el conductor exterior del coaxial, unido al puerto alimentado y tendido hacia atrás bajo los elementos (junto a un boom modelado, a la 'Separación del boom' de él). Muestra cómo la corriente en la línea de alimentación puede alterar el diagrama cuando no hay un buen balun. 0 lo desactiva. El hilo es una malla de extremo abierto: una prueba de sensibilidad de cota superior, no un modelo de un tendido de coaxial real ni de un choke.",
        "Prova di modo comune. Aggiunge un filo di questa lunghezza in metri che rappresenta il conduttore esterno del coassiale, collegato alla porta alimentata e steso all'indietro sotto gli elementi (accanto a un boom modellato, alla 'Distanza dal boom' da esso). Mostra come la corrente sulla linea di alimentazione possa disturbare il diagramma in assenza di un buon balun. 0 lo disattiva. Il filo è una calza a estremità aperta: una prova di sensibilità di limite superiore, non un modello di un coassiale reale né di un choke."),
    "--feedline-diameter": (
        "Diameter in mm of the common-mode test wire that represents the coax outer conductor. RG-58 is about 5 mm. Only used when the common-mode feed-line length is greater than 0.",
        "Diámetro en mm del hilo de prueba de modo común que representa el conductor exterior del coaxial. El RG-58 ronda los 5 mm. Solo se usa cuando la longitud de la línea de alimentación de modo común es mayor que 0.",
        "Diametro in mm del filo di prova di modo comune che rappresenta il conduttore esterno del coassiale. L'RG-58 è circa 5 mm. Usato solo quando la lunghezza della linea di alimentazione di modo comune è maggiore di 0."),
    "--cm-sensitivity": (
        "Simulates the model without and with the common-mode feed-line wire (length from 'CM feed-line length', default half a wavelength) and reports the change in front-to-back ratio and input impedance. A large change means the antenna depends on a good balun/choke. Needs nec2c. In generate it uses the model you give; in optimize, the best model found.",
        "Simula el modelo sin y con el hilo de línea de alimentación de modo común (longitud de 'Long. línea alim. MC', por defecto media longitud de onda) e informa del cambio en la relación delante/atrás y en la impedancia de entrada. Un cambio grande indica que la antena depende de un buen balun/choke. Requiere nec2c. En generate usa el modelo dado; en optimize, el mejor modelo encontrado.",
        "Simula il modello senza e con il filo di linea di alimentazione di modo comune (lunghezza da 'Lungh. linea alim. MC', predefinita mezza lunghezza d'onda) e riporta la variazione di rapporto avanti/indietro e impedenza d'ingresso. Una variazione grande indica che l'antenna dipende da un buon balun/choke. Richiede nec2c. In generate usa il modello dato; in optimize il miglior modello trovato."),
    "--ground": (
        "Ground model. free: free space (default); the pattern is a single azimuth ring at the horizon. perfect: perfectly conducting ground. average: finite ground with eps_r 13 and sigma 0.005 S/m. good: eps_r 20 and sigma 0.02 S/m. Over ground you must give a height; the pattern is then taken in two steps, an elevation cut to find the take-off angle and an azimuth ring at that angle, because horizontal polarization has no radiation at the horizon over ground.",
        "Modelo de tierra. free: espacio libre (por defecto); el diagrama es un solo anillo de azimut en el horizonte. perfect: tierra perfectamente conductora. average: tierra finita con eps_r 13 y sigma 0,005 S/m. good: eps_r 20 y sigma 0,02 S/m. Sobre tierra hay que dar una altura; entonces el diagrama se toma en dos pasos, un corte de elevación para hallar el ángulo de radiación y un anillo de azimut a ese ángulo, porque la polarización horizontal no radia en el horizonte sobre tierra.",
        "Modello del terreno. free: spazio libero (predefinito); il diagramma è un solo anello in azimut all'orizzonte. perfect: terreno perfettamente conduttore. average: terreno finito con eps_r 13 e sigma 0,005 S/m. good: eps_r 20 e sigma 0,02 S/m. Sul terreno occorre indicare un'altezza; il diagramma viene poi preso in due passi, un taglio in elevazione per trovare l'angolo di radiazione e un anello in azimut a quell'angolo, perché la polarizzazione orizzontale non irradia all'orizzonte sul terreno."),
    "--height-m": (
        "Height above ground in metres. For horizontal elements it is the height of the element centres; for vertical it is the lowest tip of the longest element (driven or parasitic). Required whenever the ground is not free space. Height strongly affects the take-off angle, the gain and the input impedance because of the ground reflection.",
        "Altura sobre el suelo en metros. Para elementos horizontales es la altura de los centros de los elementos; para verticales, la punta más baja del elemento más largo (excitado o parásito). Obligatoria siempre que la tierra no sea espacio libre. La altura afecta mucho al ángulo de radiación, la ganancia y la impedancia de entrada por la reflexión en el suelo.",
        "Altezza dal suolo in metri. Per elementi orizzontali è l'altezza dei centri degli elementi; per verticali è la punta più bassa dell'elemento più lungo (alimentato o parassita). Obbligatoria quando il terreno non è spazio libero. L'altezza influenza molto angolo di radiazione, guadagno e impedenza d'ingresso a causa della riflessione del suolo."),
    "--height-lambda": (
        "Same as the height in metres but expressed in wavelengths of the design frequency (for example 1.0 = one wavelength, about 2.07 m at 145 MHz). Use one or the other.",
        "Igual que la altura en metros pero expresada en longitudes de onda de la frecuencia de diseño (por ejemplo 1,0 = una longitud de onda, unos 2,07 m a 145 MHz). Usa una u otra.",
        "Come l'altezza in metri ma espressa in lunghezze d'onda della frequenza di progetto (per esempio 1,0 = una lunghezza d'onda, circa 2,07 m a 145 MHz). Usa l'una o l'altra."),
    "--orientation": (
        "Polarization of the antenna. horizontal: the elements lie along Y and the spacing along X (beam toward +X). vertical: the elements lie along Z, with the spacing again along X. The two are identical in free space; over ground they behave very differently.",
        "Polarización de la antena. horizontal: los elementos están a lo largo de Y y la separación a lo largo de X (haz hacia +X). vertical: los elementos están a lo largo de Z, con la separación otra vez a lo largo de X. Son idénticas en espacio libre; sobre tierra se comportan de forma muy distinta.",
        "Polarizzazione dell'antenna. horizontal: gli elementi giacciono lungo Y e la spaziatura lungo X (fascio verso +X). vertical: gli elementi giacciono lungo Z, con la spaziatura sempre lungo X. Sono identiche nello spazio libero; sul terreno si comportano in modo molto diverso."),
    "--ground-eps": (
        "Relative permittivity (eps_r) of the finite ground. Typing a value overrides the one from the ground preset (average 13, good 20). Together with the conductivity it defines the ground used by the Sommerfeld model (GN 2).",
        "Permitividad relativa (eps_r) de la tierra finita. Escribir un valor sustituye el del preajuste de tierra (average 13, good 20). Junto con la conductividad define la tierra usada por el modelo de Sommerfeld (GN 2).",
        "Permittività relativa (eps_r) del terreno finito. Digitare un valore sostituisce quello del preset del terreno (average 13, good 20). Insieme alla conducibilità definisce il terreno usato dal modello di Sommerfeld (GN 2)."),
    "--ground-sigma": (
        "Conductivity in S/m of the finite ground. Typing a value overrides the one from the ground preset (average 0.005, good 0.02). Wet soil is about 0.01 to 0.03; dry sand and rock are lower.",
        "Conductividad en S/m de la tierra finita. Escribir un valor sustituye el del preajuste de tierra (average 0,005, good 0,02). El suelo húmedo ronda 0,01 a 0,03; la arena seca y la roca son menores.",
        "Conducibilità in S/m del terreno finito. Digitare un valore sostituisce quello del preset del terreno (average 0,005, good 0,02). Il suolo umido è circa 0,01-0,03; sabbia asciutta e roccia sono inferiori."),
    "--mast-length-m": (
        "Models a mast as a straight conductor of this length in metres. 0 turns it off. With horizontal elements it hangs below the boom centre, perpendicular to the boom and the elements; on the boom axis it lies in the symmetry plane of the balanced elements and is not excited, so use 'Mast side offset' for a side clamp. With vertical elements it is a side mount: parallel to the elements, behind the rear end of the boom, centred on the element centres. Not together with the feed-line wire. It is not galvanically connected: the 'Mast gap' sets the coupling distance.",
        "Modela un mástil como un conductor recto de esta longitud en metros. 0 lo desactiva. Con elementos horizontales cuelga bajo el centro del boom, perpendicular al boom y a los elementos; en el eje del boom queda en el plano de simetría de los elementos balanceados y no se excita, así que usa el 'Desplazamiento lateral del mástil' para una abrazadera lateral. Con elementos verticales es un montaje lateral: paralelo a los elementos, detrás del extremo trasero del boom, centrado en los centros de los elementos. No junto con el hilo de línea de alimentación. No está conectado galvánicamente: la 'Separación del mástil' fija la distancia de acoplamiento.",
        "Modella un palo come conduttore rettilineo di questa lunghezza in metri. 0 lo disattiva. Con elementi orizzontali pende sotto il centro del boom, perpendicolare a boom ed elementi; sull'asse del boom giace nel piano di simmetria degli elementi bilanciati e non viene eccitato, quindi usa lo 'Spostamento laterale del palo' per un morsetto laterale. Con elementi verticali è un montaggio laterale: parallelo agli elementi, dietro l'estremità posteriore del boom, centrato sui centri degli elementi. Non insieme al filo di linea di alimentazione. Non è collegato galvanicamente: la 'Distanza dal palo' fissa la distanza di accoppiamento."),
    "--mast-diameter-mm": (
        "Diameter in mm of the modelled mast. Default 40 mm. A thicker mast couples more strongly to the antenna. Only used when the mast length (or the mast sensitivity list) is set.",
        "Diámetro en mm del mástil modelado. Por defecto 40 mm. Un mástil más grueso se acopla con más fuerza a la antena. Solo se usa cuando se indica la longitud del mástil (o la lista de sensibilidad al mástil).",
        "Diametro in mm del palo modellato. Predefinito 40 mm. Un palo più spesso si accoppia più fortemente all'antenna. Usato solo quando è indicata la lunghezza del palo (o l'elenco di sensibilità al palo)."),
    "--mast-gap-mm": (
        "Air gap in mm between the lowest modelled conductor (the boom if modelled, otherwise the elements) and the top of the mast. Default 20 mm. 0 places the mast surface 10% of its radius from the conductor. A smaller gap means stronger coupling.",
        "Hueco de aire en mm entre el conductor modelado más bajo (el boom si se modela, si no los elementos) y la parte alta del mástil. Por defecto 20 mm. 0 sitúa la superficie del mástil al 10 % de su radio del conductor. Un hueco menor significa un acoplamiento más fuerte.",
        "Distanza d'aria in mm tra il conduttore modellato più basso (il boom se modellato, altrimenti gli elementi) e la sommità del palo. Predefinita 20 mm. 0 colloca la superficie del palo al 10% del suo raggio dal conduttore. Una distanza minore significa un accoppiamento più forte."),
    "--mast-sensitivity": (
        "Comma-separated list of mast lengths in metres (for example 0.5,1,2). The model is simulated without a mast and with each length, using the mast diameter and gap above, and a table with the changes in gain, F/B and Zin is printed. Needs nec2c. In generate it uses the model you give; in optimize, the best model.",
        "Lista de longitudes de mástil en metros separadas por comas (por ejemplo 0.5,1,2). El modelo se simula sin mástil y con cada longitud, usando el diámetro y el hueco de mástil indicados, y se imprime una tabla con los cambios de ganancia, F/B y Zin. Requiere nec2c. En generate usa el modelo dado; en optimize, el mejor modelo.",
        "Elenco di lunghezze del palo in metri separate da virgole (per esempio 0.5,1,2). Il modello viene simulato senza palo e con ciascuna lunghezza, usando diametro e distanza del palo indicati, e viene stampata una tabella con le variazioni di guadagno, F/B e Zin. Richiede nec2c. In generate usa il modello dato; in optimize il modello migliore."),
    "--fsweep": (
        "Frequency sweep written as FMIN,FMAX,STEP in MHz (for example 140,150,0.5). nec2c then computes the input impedance Zin(f) at every frequency, which the match and sweep modes read to study bandwidth. The geometry and the line stay fixed at the design frequency. Without the pattern option the radiation pattern is omitted from the sweep.",
        "Barrido de frecuencia escrito como FMIN,FMAX,PASO en MHz (por ejemplo 140,150,0.5). nec2c calcula entonces la impedancia de entrada Zin(f) en cada frecuencia, que los modos match y sweep leen para estudiar el ancho de banda. La geometría y la línea siguen fijadas a la frecuencia de diseño. Sin la opción de diagrama se omite el diagrama de radiación del barrido.",
        "Scansione di frequenza scritta come FMIN,FMAX,PASSO in MHz (per esempio 140,150,0.5). nec2c calcola allora l'impedenza d'ingresso Zin(f) a ogni frequenza, che le modalità match e sweep leggono per studiare la banda. Geometria e linea restano fissate alla frequenza di progetto. Senza l'opzione diagramma il diagramma di radiazione è omesso dalla scansione."),
    "--fsweep-pattern": (
        "With the frequency sweep, also computes the radiation pattern at every frequency. With 'Run nec2c' it summarizes gain, front-to-back ratio and Zin versus frequency (taking the take-off angle per frequency over ground) and writes a CSV. Takes longer than an impedance-only sweep.",
        "Con el barrido de frecuencia, calcula también el diagrama de radiación en cada frecuencia. Con 'Ejecutar nec2c' resume ganancia, relación delante/atrás y Zin en función de la frecuencia (tomando el ángulo de radiación de cada frecuencia sobre tierra) y escribe un CSV. Tarda más que un barrido solo de impedancia.",
        "Con la scansione di frequenza, calcola anche il diagramma di radiazione a ogni frequenza. Con 'Esegui nec2c' riassume guadagno, rapporto avanti/indietro e Zin in funzione della frequenza (prendendo l'angolo di radiazione per ogni frequenza sul terreno) e scrive un CSV. Richiede più tempo di una scansione di sola impedenza."),
    "--fsweep-csv": (
        "Name of the CSV file that receives the pattern-versus-frequency summary. If empty it is <output file>_fsweep.csv. Only used together with the frequency sweep and the pattern-versus-frequency option.",
        "Nombre del archivo CSV que recibe el resumen del diagrama en función de la frecuencia. Si está vacío es <archivo de salida>_fsweep.csv. Solo se usa junto con el barrido de frecuencia y la opción de diagrama en función de la frecuencia.",
        "Nome del file CSV che riceve il riepilogo del diagramma in funzione della frequenza. Se vuoto è <file di uscita>_fsweep.csv. Usato solo insieme alla scansione di frequenza e all'opzione diagramma in funzione della frequenza."),
    "--band": (
        "Scores every candidate at N evenly spaced frequencies, written FMIN,FMAX,N (for example 144,146,5), and keeps the worst case: the minimum over frequency of the objective minus the VSWR penalty. This favours designs that hold up across the whole band instead of one frequency. The geometry and line length stay fixed at the design frequency, which must lie inside the band and is always scored (it is added when it is not one of the N points). Empty means a single frequency. Costs N times more simulations.",
        "Puntúa cada candidato en N frecuencias equiespaciadas, escritas FMIN,FMAX,N (por ejemplo 144,146,5), y conserva el peor caso: el mínimo en frecuencia del objetivo menos la penalización por VSWR. Favorece diseños que se mantienen en toda la banda en vez de a una sola frecuencia. La geometría y la longitud de línea siguen fijadas a la frecuencia de diseño, que debe estar dentro de la banda y siempre se puntúa (se añade si no es uno de los N puntos). Vacío significa una sola frecuencia. Cuesta N veces más simulaciones.",
        "Valuta ogni candidato a N frequenze equispaziate, scritte FMIN,FMAX,N (per esempio 144,146,5), e conserva il caso peggiore: il minimo in frequenza dell'obiettivo meno la penalità VSWR. Favorisce progetti che reggono su tutta la banda invece che a una sola frequenza. Geometria e lunghezza della linea restano fissate alla frequenza di progetto, che deve stare dentro la banda ed è sempre valutata (viene aggiunta se non è uno degli N punti). Vuoto significa una sola frequenza. Costa N volte più simulazioni."),
    "--refine": (
        "Rounds of local search after the grid scan. Each round simulates all small moves around the best point in parallel (element lengths, phase or line length and, if enabled, spacing and diameter), keeps the best improvement and halves the step when nothing improves. 0 means grid only. Empty uses the default: 6 rounds with the tl/lc networks, 8 with reflectors or directors, 0 with ideal sources. A Nelder-Mead polish follows (see Nelder-Mead runs).",
        "Rondas de búsqueda local tras el barrido en rejilla. Cada ronda simula en paralelo todos los pequeños movimientos alrededor del mejor punto (longitudes de elementos, fase o longitud de línea y, si se activan, separación y diámetro), conserva la mejor mejora y reduce a la mitad el paso cuando nada mejora. 0 significa solo rejilla. Vacío usa el valor por defecto: 6 rondas con las redes tl/lc, 8 con reflectores o directores, 0 con fuentes ideales. Después sigue un pulido Nelder-Mead (ver Simulaciones Nelder-Mead).",
        "Cicli di ricerca locale dopo la scansione a griglia. Ogni ciclo simula in parallelo tutte le piccole mosse intorno al punto migliore (lunghezze degli elementi, fase o lunghezza della linea e, se abilitati, spaziatura e diametro), conserva il miglior miglioramento e dimezza il passo quando nulla migliora. 0 significa solo griglia. Vuoto usa il valore predefinito: 6 cicli con le reti tl/lc, 8 con riflettori o direttori, 0 con sorgenti ideali. Segue una rifinitura Nelder-Mead (vedi Simulazioni Nelder-Mead)."),
    "--nm-evals": (
        "Maximum number of nec2c runs of the Nelder-Mead polish that follows the local search (default 40; 0 turns it off). It moves the continuous parameters together (element lengths, spacing, line length or rear phase, diameter), so it can follow correlated changes such as a longer front element with a shorter line, which a one-axis-at-a-time search cannot.",
        "Número máximo de ejecuciones de nec2c del pulido Nelder-Mead que sigue a la búsqueda local (por defecto 40; 0 lo desactiva). Mueve juntos los parámetros continuos (longitudes de elementos, separación, longitud de línea o fase trasera, diámetro), así que puede seguir cambios correlacionados como un elemento delantero más largo con una línea más corta, que una búsqueda de un eje cada vez no puede seguir.",
        "Numero massimo di esecuzioni di nec2c della rifinitura Nelder-Mead che segue la ricerca locale (predefinito 40; 0 la disattiva). Muove insieme i parametri continui (lunghezze degli elementi, spaziatura, lunghezza della linea o fase posteriore, diametro), quindi può seguire cambiamenti correlati come un elemento anteriore più lungo con una linea più corta, che una ricerca un asse alla volta non può seguire."),
    "--pretune": (
        "With the tl/lc networks, before the grid the front element length of the central candidate is tuned for an input reactance near zero at the NEC port (without the series capacitor) at the design frequency (a secant search of up to six nec2c runs), and the front-length grid is re-centred on that value. 1 % of element length moves the resonance by about 1 % of frequency, so a grid that starts detuned wastes most of its points. Default on.",
        "Con las redes tl/lc, antes de la rejilla se ajusta la longitud del elemento delantero del candidato central para que la reactancia de entrada en el puerto NEC (sin el condensador serie) sea casi cero a la frecuencia de diseño (búsqueda por secante de hasta seis ejecuciones de nec2c), y la rejilla de longitudes delanteras se recentra en ese valor. Un 1 % de longitud mueve la resonancia alrededor de un 1 % de frecuencia, así que una rejilla que empieza desintonizada desperdicia la mayoría de sus puntos. Activado por defecto.",
        "Con le reti tl/lc, prima della griglia la lunghezza dell'elemento anteriore del candidato centrale viene regolata per una reattanza d'ingresso vicina a zero alla porta NEC (senza il condensatore serie) alla frequenza di progetto (ricerca a secante di al massimo sei esecuzioni di nec2c) e la griglia delle lunghezze anteriori viene ricentrata su quel valore. L'1 % di lunghezza sposta la risonanza di circa l'1 % in frequenza, quindi una griglia che parte disaccordata spreca la maggior parte dei punti. Attivo per impostazione predefinita."),
    "--fb-metric": (
        "Which front-to-back ratio the front-back and balance objectives (and the Monte-Carlo) use. point: forward gain minus the gain at exactly 180 degrees on the ring. sector: forward gain minus the worst point of the rear sector on the ring (default). cone: forward gain minus the worst direction within the sector half-angle of the back direction at any elevation, taken from a 5 degree rear half-sphere grid; it catches rear lobes above or below the ring.",
        "Qué relación delante-detrás usan los objetivos front-back y balance (y el Monte Carlo). point: ganancia delantera menos la ganancia exactamente a 180 grados en el anillo. sector: ganancia delantera menos el peor punto del sector trasero en el anillo (por defecto). cone: ganancia delantera menos la peor dirección dentro del semiángulo del sector alrededor de la dirección trasera, a cualquier elevación, tomada de una rejilla de 5 grados de la semiesfera trasera; detecta lóbulos traseros por encima o por debajo del anillo.",
        "Quale rapporto avanti-indietro usano gli obiettivi front-back e balance (e il Monte Carlo). point: guadagno anteriore meno il guadagno esattamente a 180 gradi sull'anello. sector: guadagno anteriore meno il punto peggiore del settore posteriore sull'anello (predefinito). cone: guadagno anteriore meno la direzione peggiore entro il semiangolo del settore attorno alla direzione posteriore, a qualsiasi elevazione, presa da una griglia di 5 gradi della semisfera posteriore; rileva lobi posteriori sopra o sotto l'anello."),
    "--jobs": (
        "Number of nec2c processes run in parallel while scoring candidates. By default it equals the number of CPU cores. Lower it if the computer becomes unresponsive; raising it above the core count rarely helps.",
        "Número de procesos nec2c que se ejecutan en paralelo al puntuar candidatos. Por defecto es el número de núcleos de la CPU. Redúcelo si el equipo deja de responder; subirlo por encima del número de núcleos rara vez ayuda.",
        "Numero di processi nec2c eseguiti in parallelo durante la valutazione dei candidati. Per impostazione predefinita è pari al numero di core della CPU. Riducilo se il computer non risponde; aumentarlo oltre il numero di core raramente aiuta."),
    "--csv": (
        "CSV file that receives every scanned candidate with its score, gain, F/B, input impedance and VSWR. The first lines start with '#' and record the arguments and the nec2c version; skip them (comment='#' in pandas) when reading. Default hb9cv_scan.csv in the work folder.",
        "Archivo CSV que recibe cada candidato explorado con su puntuación, ganancia, F/B, impedancia de entrada y VSWR. Las primeras líneas empiezan por '#' y registran los argumentos y la versión de nec2c; sáltalas (comment='#' en pandas) al leerlo. Por defecto hb9cv_scan.csv en la carpeta de trabajo.",
        "File CSV che riceve ogni candidato esplorato con punteggio, guadagno, F/B, impedenza d'ingresso e VSWR. Le prime righe iniziano con '#' e registrano gli argomenti e la versione di nec2c; saltale (comment='#' in pandas) in lettura. Predefinito hb9cv_scan.csv nella cartella di lavoro."),
    "--spacing-values": (
        "List of element spacings to scan, in wavelengths, separated by commas (default 0.08,0.125,0.17). Every value is combined with the other lists, so the number of simulations is the product of the list sizes; keep the lists short.",
        "Lista de separaciones entre elementos que se exploran, en longitudes de onda, separadas por comas (por defecto 0.08,0.125,0.17). Cada valor se combina con las demás listas, así que el número de simulaciones es el producto de los tamaños de las listas; mantenlas cortas.",
        "Elenco delle spaziature tra gli elementi da esplorare, in lunghezze d'onda, separate da virgole (predefinito 0.08,0.125,0.17). Ogni valore è combinato con gli altri elenchi, quindi il numero di simulazioni è il prodotto delle dimensioni degli elenchi; tienili brevi."),
    "--rear-length-values": (
        "List of rear element length factors to scan, relative to 150/f metres (default 0.98,1.0,1.02). Combined with the other lists, so the number of simulations is the product of the list sizes.",
        "Lista de factores de longitud del elemento trasero que se exploran, relativos a 150/f metros (por defecto 0.98,1.0,1.02). Se combina con las demás listas, así que el número de simulaciones es el producto de los tamaños de las listas.",
        "Elenco dei fattori di lunghezza dell'elemento posteriore da esplorare, relativi a 150/f metri (predefinito 0.98,1.0,1.02). Combinato con gli altri elenchi, quindi il numero di simulazioni è il prodotto delle dimensioni degli elenchi."),
    "--front-length-values": (
        "List of front element length factors to scan, relative to 139/f metres (default 0.98,1.0,1.02). With the pre-tune on, the list is shifted so that its middle value is the tuned front length. Combined with the other lists, so the number of simulations is the product of the list sizes.",
        "Lista de factores de longitud del elemento delantero que se exploran, relativos a 139/f metros (por defecto 0.98,1.0,1.02). Con la presintonía activada, la lista se desplaza para que su valor central sea la longitud delantera ajustada. Se combina con las demás listas, así que el número de simulaciones es el producto de los tamaños de las listas.",
        "Elenco dei fattori di lunghezza dell'elemento anteriore da esplorare, relativi a 139/f metri (predefinito 0.98,1.0,1.02). Con la pre-sintonia attiva, l'elenco viene spostato in modo che il valore centrale sia la lunghezza anteriore regolata. Combinato con gli altri elenchi, quindi il numero di simulazioni è il prodotto delle dimensioni degli elenchi."),
    "--line-offset-values": (
        "Offsets in electrical degrees, added to the centre of the line scan of each spacing (the synthesized line, or the theoretical one without synthesis, moved up when the lowest offset would fall below the shortest buildable line), that the optimizer tries with the tl and lc networks (default -30 to +30 in 5 degree steps). Values outside the buildable range are dropped, never wrapped around 360. Replaced by 'Line lengths' if that is given.",
        "Desfases en grados eléctricos, sumados al centro de la exploración de línea de cada separación (la línea sintetizada, o la teórica sin síntesis, desplazado hacia arriba si el desfase menor quedara por debajo de la línea más corta construible), que el optimizador prueba con las redes tl y lc (por defecto de -30 a +30 en pasos de 5 grados). Los valores fuera del rango construible se descartan, nunca se dan la vuelta en 360. Se sustituye por 'Longitudes de línea' si se indica.",
        "Offset in gradi elettrici, sommati al centro dell'esplorazione della linea di ogni spaziatura (la linea sintetizzata, o quella teorica senza sintesi, spostato in alto se l'offset minore cadesse sotto la linea più corta costruibile), che l'ottimizzatore prova con le reti tl e lc (predefinito da -30 a +30 a passi di 5 gradi). I valori fuori dall'intervallo costruibile sono scartati, mai riportati attorno a 360. Sostituito da 'Lunghezze della linea' se indicato."),
    "--line-length-values": (
        "Absolute physical line lengths in metres to scan. If given, it replaces the list of degree offsets. Useful when the cable length is the quantity you actually cut.",
        "Longitudes físicas absolutas de línea en metros que se exploran. Si se indica, sustituye la lista de desfases en grados. Útil cuando la longitud del cable es la magnitud que realmente se corta.",
        "Lunghezze fisiche assolute della linea in metri da esplorare. Se indicato, sostituisce l'elenco degli offset in gradi. Utile quando la lunghezza del cavo è la grandezza che si taglia realmente."),
    "--z0-values": (
        "Characteristic impedances in ohm of the TL phase line to scan, for example 50,75,300. On an open-wire line each value sets the wire spacing (the wire diameter stays fixed). Empty means the single Z0 of the configured line.",
        "Impedancias características en ohm de la línea de fase TL que se exploran, por ejemplo 50,75,300. En una línea abierta cada valor fija la separación de los hilos (el diámetro del hilo permanece fijo). Vacío significa la única Z0 de la línea configurada.",
        "Impedenze caratteristiche in ohm della linea di fase TL da esplorare, per esempio 50,75,300. In una linea aperta ogni valore determina la distanza dei fili (il diametro del filo resta fisso). Vuoto significa l'unica Z0 della linea configurata."),
    "--ratio-values": (
        "Relative amplitudes of the rear source to scan (default 0.7,1.0,1.3). Only for the ideal network with two sources; the tl and lc networks fix the amplitude by their own physics.",
        "Amplitudes relativas de la fuente trasera que se exploran (por defecto 0.7,1.0,1.3). Solo para la red ideal de dos fuentes; las redes tl y lc fijan la amplitud con su propia física.",
        "Ampiezze relative della sorgente posteriore da esplorare (predefinito 0.7,1.0,1.3). Solo per la rete ideale a due sorgenti; le reti tl e lc fissano l'ampiezza con la propria fisica."),
    "--phase-offset-values": (
        "Offsets in degrees from the theoretical rear phase (180 minus the spacing in degrees) to scan for each spacing, for the ideal network (default -60 to +60 in 15 degree steps). Replaced by 'Absolute phases' if given.",
        "Desfases en grados respecto a la fase trasera teórica (180 menos la separación en grados) que se exploran para cada separación, en la red ideal (por defecto de -60 a +60 en pasos de 15 grados). Se sustituye por 'Fases absolutas' si se indican.",
        "Offset in gradi rispetto alla fase posteriore teorica (180 meno la spaziatura in gradi) da esplorare per ogni spaziatura, per la rete ideale (predefinito da -60 a +60 a passi di 15 gradi). Sostituito da 'Fasi assolute' se indicate."),
    "--phase-values": (
        "Absolute rear source phases in degrees to scan with the ideal network. If given, it replaces the list of phase offsets.",
        "Fases absolutas de la fuente trasera en grados que se exploran con la red ideal. Si se indica, sustituye la lista de desfases de fase.",
        "Fasi assolute della sorgente posteriore in gradi da esplorare con la rete ideale. Se indicato, sostituisce l'elenco degli offset di fase."),
    "--zin-r-range": (
        'Admissible range of the input resistance in ohm, written min,max. Default 2,500 when candidates are scored through the L-network (the network loss already penalizes extreme values) and 8,80 for a direct connection. Candidates outside the range are excluded from the best choice (tl and lc only; with a band, at every band frequency).',
        'Rango admisible de la resistencia de entrada en ohm, escrito mín,máx. Por defecto 2,500 cuando los candidatos se puntúan a través de la red L (la pérdida de la red ya penaliza los valores extremos) y 8,80 para una conexión directa. Los candidatos fuera del rango se excluyen de la mejor elección (solo tl y lc; con banda, en cada frecuencia de la banda).',
        "Intervallo ammesso della resistenza d'ingresso in ohm, scritto min,max. Predefinito 2,500 quando i candidati sono valutati attraverso la rete L (la perdita della rete penalizza già i valori estremi) e 8,80 per un collegamento diretto. I candidati fuori intervallo sono esclusi dalla scelta del migliore (solo tl e lc; con una banda, a ogni frequenza della banda)."),
    "--zin-x-max": (
        'Maximum admissible absolute input reactance |X_in| in ohm (tl and lc only). Default 500 when candidates are scored through the L-network; for a direct connection 30, or 10 with a 4:1 balun.',
        'Reactancia de entrada absoluta máxima admisible |X_in| en ohm (solo tl y lc). Por defecto 500 cuando los candidatos se puntúan a través de la red L; para una conexión directa 30, o 10 con balun 4:1.',
        "Reattanza d'ingresso assoluta massima ammessa |X_in| in ohm (solo tl e lc). Predefinita 500 quando i candidati sono valutati attraverso la rete L; per un collegamento diretto 30, oppure 10 con balun 4:1."),
    "--strict-zin": (
        'What optimize does when no candidate meets the Zin window (admissible R_in and max |X_in| at every band frequency). Off (default): the closest candidate is still written (model, build sheet, JSON) and reported with a clear warning, so you can see how far the window is from reach. On: stop with an error and write no model.',
        'Qué hace optimize cuando ningún candidato cumple la ventana de Zin (R_in admisible y |X_in| máxima en cada frecuencia de la banda). Desactivado (por defecto): el candidato más cercano se escribe igualmente (modelo, hoja de construcción, JSON) y se informa con un aviso claro, para ver lo lejos que está la ventana. Activado: termina con un error y no escribe ningún modelo.',
        'Cosa fa optimize quando nessun candidato rispetta la finestra di Zin (R_in ammessa e |X_in| massima a ogni frequenza della banda). Disattivato (predefinito): il candidato più vicino viene comunque scritto (modello, scheda di costruzione, JSON) e segnalato con un avviso chiaro, per vedere quanto la finestra è lontana. Attivato: termina con un errore e non scrive alcun modello.'),
    "--vswr-weight": (
        "dB subtracted from the score for each unit of VSWR above 1 (worst VSWR over the band, against the target impedance). Default 3 with tl/lc and 0 otherwise. 0 only reports VSWR without penalizing it. Increase it to prefer well-matched designs over slightly better pattern.",
        "dB que se restan de la puntuación por cada unidad de VSWR por encima de 1 (peor VSWR en la banda, frente a la impedancia objetivo). Por defecto 3 con tl/lc y 0 en los demás casos. 0 solo informa del VSWR sin penalizarlo. Súbelo para preferir diseños bien adaptados frente a un diagrama algo mejor.",
        "dB sottratti dal punteggio per ogni unità di VSWR sopra 1 (peggior VSWR sulla banda, rispetto all'impedenza obiettivo). Predefinito 3 con tl/lc e 0 negli altri casi. 0 riporta solo il VSWR senza penalizzarlo. Aumentalo per preferire progetti ben adattati a un diagramma leggermente migliore."),
    "--realized-gain": (
        "In the gain-based objectives use realized gain, which discounts the mismatch loss against the target impedance, instead of plain gain. Only meaningful with the tl and lc networks. default is on for tl/lc; no uses the plain gain.",
        "En los objetivos basados en ganancia usa la ganancia realizada, que descuenta la pérdida por desadaptación frente a la impedancia objetivo, en lugar de la ganancia simple. Solo tiene sentido con las redes tl y lc. default está activada para tl/lc; no usa la ganancia simple.",
        "Negli obiettivi basati sul guadagno usa il guadagno realizzato, che sconta la perdita per disadattamento rispetto all'impedenza obiettivo, invece del guadagno semplice. Ha senso solo con le reti tl e lc. default è attivo per tl/lc; no usa il guadagno semplice."),
    "--w-gain": (
        "Weight of the gain term in the balance objective. The balance score is w_gain times the gain plus w_fb times the robust front-to-back ratio. Default 1.0. Increase it to favour gain over rejection.",
        "Peso del término de ganancia en el objetivo balance. La puntuación balance es w_gain por la ganancia más w_fb por la relación delante/atrás robusta. Por defecto 1,0. Súbelo para favorecer la ganancia frente al rechazo.",
        "Peso del termine di guadagno nell'obiettivo balance. Il punteggio balance è w_gain per il guadagno più w_fb per il rapporto avanti/indietro robusto. Predefinito 1,0. Aumentalo per favorire il guadagno rispetto alla reiezione."),
    "--w-fb": (
        "Weight of the robust front-to-back term in the balance objective (default 0.5). Increase it to favour rear rejection over forward gain.",
        "Peso del término de relación delante/atrás robusta en el objetivo balance (por defecto 0,5). Súbelo para favorecer el rechazo trasero frente a la ganancia frontal.",
        "Peso del termine di rapporto avanti/indietro robusto nell'obiettivo balance (predefinito 0,5). Aumentalo per favorire la reiezione posteriore rispetto al guadagno in avanti."),
    "--rear-sector": (
        "Rear sector in degrees of azimuth, written start,end (default 150,210). The robust front-to-back ratio is the forward gain minus the worst (highest) point inside this sector, so a design that only has a null exactly at 180 degrees is not rewarded. The sector must contain 180 degrees and stay in the rear half: 90 < start < 180 < end < 270.",
        "Sector trasero en grados de azimut, escrito inicio,fin (por defecto 150,210). La relación delante/atrás robusta es la ganancia frontal menos el peor punto (el más alto) dentro de este sector, de modo que no se premia un diseño que solo tiene un nulo justo a 180 grados. El sector debe contener 180 grados y quedar en la mitad trasera: 90 < inicio < 180 < fin < 270.",
        "Settore posteriore in gradi di azimut, scritto inizio,fine (predefinito 150,210). Il rapporto avanti/indietro robusto è il guadagno in avanti meno il punto peggiore (il più alto) nel settore, così un progetto che ha un nullo solo esattamente a 180 gradi non viene premiato. Il settore deve contenere 180 gradi e restare nella metà posteriore: 90 < inizio < 180 < fine < 270."),
    "--refine-spacing": (
        "When refining, also move the element spacing (step 0.01 wavelength, halved when no move improves). default follows the program setting, yes forces it and no keeps the spacing fixed. Letting it move can improve the result but changes the boom length.",
        "Al refinar, mueve también la separación entre elementos (paso de 0,01 longitud de onda, reducido a la mitad cuando ningún movimiento mejora). default sigue el ajuste del programa, yes lo fuerza y no mantiene la separación fija. Permitir que se mueva puede mejorar el resultado pero cambia la longitud del boom.",
        "Durante il raffinamento sposta anche la spaziatura tra gli elementi (passo 0,01 lunghezze d'onda, dimezzato quando nessuna mossa migliora). default segue l'impostazione del programma, yes la forza e no mantiene la spaziatura fissa. Lasciarla muovere può migliorare il risultato ma cambia la lunghezza del boom."),
    "--refine-diameter": (
        "When refining, also move the element diameter: to the next or previous purchasable value of 'Tube diameters', or in 1 mm steps (halved when no move improves) when that list is 'any', always inside the diameter range. default follows the program setting, yes forces it and no keeps the diameter fixed.",
        "Al refinar, mueve también el diámetro de los elementos: al valor comercial siguiente o anterior de 'Diámetros de tubo', o en pasos de 1 mm (reducidos a la mitad cuando ningún movimiento mejora) si esa lista es 'any', siempre dentro del rango de diámetros. default sigue el ajuste del programa, yes lo fuerza y no mantiene el diámetro fijo.",
        "Durante il raffinamento sposta anche il diametro degli elementi: al valore commerciale successivo o precedente di 'Diametri dei tubi', oppure a passi di 1 mm (dimezzati quando nessuna mossa migliora) se l'elenco è 'any', sempre nell'intervallo dei diametri. default segue l'impostazione del programma, yes lo forza e no mantiene il diametro fisso."),
    "--diameter-range": (
        "Lower and upper limits in mm for the element diameter during refinement, written min,max (default 6,16). Use the tube sizes you can really buy.",
        "Límites inferior y superior en mm del diámetro de los elementos durante el refinamiento, escritos mín,máx (por defecto 6,16). Usa los tamaños de tubo que realmente puedas comprar.",
        "Limiti inferiore e superiore in mm del diametro degli elementi durante il raffinamento, scritti min,max (predefinito 6,16). Usa le misure di tubo che puoi davvero acquistare."),
    "--tolerance-mc": (
        "Monte-Carlo tolerance analysis. After the scan, the same N random perturbations are applied to each of the best candidates: element lengths and spacings, line phase, and with their own tolerances the T-match taps and bar offsets of each element, the line velocity factor and Z0 and the series capacitor. The series capacitor and the matching network stay as designed for the nominal candidate unless 'MC: re-tune per sample' is on. The candidates are re-ranked by the 10th percentile score, so you pick a design that tolerates build errors; a bootstrap interval of that percentile is printed. 0 turns it off; use 100 or more. It multiplies the run time.",
        "Análisis de tolerancias por Monte Carlo. Tras la exploración se aplican las mismas N perturbaciones aleatorias a cada uno de los mejores candidatos: longitudes y separaciones de los elementos, fase de la línea y, con sus propias tolerancias, las tomas y barras del T-match de cada elemento, el factor de velocidad y la Z0 de la línea y el condensador serie. El condensador serie y la red de adaptación se mantienen como se diseñaron para el candidato nominal salvo que se active 'MC: reajustar por muestra'. Los candidatos se reordenan por la puntuación del percentil 10, de modo que eliges un diseño que tolera errores de construcción; se imprime un intervalo bootstrap de ese percentil. 0 lo desactiva; usa 100 o más. Multiplica el tiempo de ejecución.",
        "Analisi di tolleranza Monte Carlo. Dopo la scansione le stesse N perturbazioni casuali vengono applicate a ciascuno dei migliori candidati: lunghezze e spaziature degli elementi, fase della linea e, con le proprie tolleranze, prese e barre del T-match di ogni elemento, fattore di velocità e Z0 della linea e condensatore serie. Il condensatore serie e la rete di adattamento restano come progettati per il candidato nominale salvo che sia attivo 'MC: ritarare per campione'. I candidati vengono riordinati per punteggio al 10° percentile, così scegli un progetto che tollera gli errori di costruzione; viene stampato un intervallo bootstrap di quel percentile. 0 lo disattiva; usa 100 o più. Moltiplica il tempo di esecuzione."),
    "--tol-dist": (
        "Distribution of the Monte-Carlo errors. uniform: any value within plus or minus the tolerance is equally likely (a worst-case style assumption). normal: Gaussian with the tolerance as two standard deviations, clipped at three.",
        "Distribución de los errores del Monte Carlo. uniform: cualquier valor dentro de más o menos la tolerancia es igual de probable (una suposición de tipo peor caso). normal: gaussiana con la tolerancia como dos desviaciones típicas, recortada a tres.",
        "Distribuzione degli errori del Monte Carlo. uniform: qualsiasi valore entro più o meno la tolleranza è ugualmente probabile (un'ipotesi da caso peggiore). normal: gaussiana con la tolleranza pari a due deviazioni standard, troncata a tre."),
    "--tol-top-k": (
        "Monte-Carlo: how many of the best candidates of the scan are analysed with random perturbations (default 5). More candidates give a better chance to find a robust design but cost proportionally more simulations.",
        "Monte Carlo: cuántos de los mejores candidatos de la exploración se analizan con perturbaciones aleatorias (por defecto 5). Más candidatos dan más opciones de encontrar un diseño robusto pero cuestan proporcionalmente más simulaciones.",
        "Monte Carlo: quanti dei migliori candidati della scansione vengono analizzati con perturbazioni casuali (predefinito 5). Più candidati danno maggiori possibilità di trovare un progetto robusto ma costano proporzionalmente più simulazioni."),
    "--tol-length-mm": (
        "Monte-Carlo: element length tolerance, plus or minus this many mm with uniform distribution (default 2). Set it to the accuracy you can really achieve when cutting the tubes.",
        "Monte Carlo: tolerancia de la longitud de los elementos, más o menos estos mm con distribución uniforme (por defecto 2). Ponla en la precisión que realmente puedas lograr al cortar los tubos.",
        "Monte Carlo: tolleranza della lunghezza degli elementi, più o meno questi mm con distribuzione uniforme (predefinita 2). Impostala sulla precisione che puoi davvero ottenere tagliando i tubi."),
    "--tol-spacing-mm": (
        "Monte-Carlo: element spacing tolerance, plus or minus this many mm with uniform distribution (default 3). It reflects how accurately the elements are positioned on the boom.",
        "Monte Carlo: tolerancia de la separación entre elementos, más o menos estos mm con distribución uniforme (por defecto 3). Refleja con qué precisión se colocan los elementos en el boom.",
        "Monte Carlo: tolleranza della spaziatura tra gli elementi, più o meno questi mm con distribuzione uniforme (predefinita 3). Riflette la precisione con cui gli elementi sono posizionati sul boom."),
    "--tol-line-deg": (
        "Monte-Carlo: tolerance of the phase line (or of the rear phase with the ideal network), plus or minus this many electrical degrees with uniform distribution (default 2). It reflects the accuracy of the cable or line length.",
        "Monte Carlo: tolerancia de la línea de fase (o de la fase trasera con la red ideal), más o menos estos grados eléctricos con distribución uniforme (por defecto 2). Refleja la precisión de la longitud del cable o de la línea.",
        "Monte Carlo: tolleranza della linea di fase (o della fase posteriore con la rete ideale), più o meno questi gradi elettrici con distribuzione uniforme (predefinita 2). Riflette la precisione della lunghezza del cavo o della linea."),
    "--tol-seed": (
        "Monte-Carlo: seed of the random number generator (default 12345). Using the same seed repeats exactly the same perturbations, so results are reproducible; change it to try a different random set.",
        "Monte Carlo: semilla del generador de números aleatorios (por defecto 12345). Con la misma semilla se repiten exactamente las mismas perturbaciones, así que los resultados son reproducibles; cámbiala para probar otro conjunto aleatorio.",
        "Monte Carlo: seme del generatore di numeri casuali (predefinito 12345). Con lo stesso seme si ripetono esattamente le stesse perturbazioni, quindi i risultati sono riproducibili; cambialo per provare un altro insieme casuale."),
    "--loss-weight": (
        "With 'Score through the L-network' and the front-back objective: dB of score subtracted per dB of matching-network loss (default 1). It is an efficiency penalty, the loss does not change the F/B; 0 ranks by the pure F/B.",
        "Con 'Puntuar a través de la red L' y el objetivo front-back: dB de puntuación restados por dB de pérdida de la red de adaptación (por defecto 1). Es una penalización de eficiencia, la pérdida no cambia la F/B; 0 ordena por la F/B pura.",
        "Con 'Punteggio attraverso la rete L' e l'obiettivo front-back: dB di punteggio sottratti per dB di perdita della rete di adattamento (predefinito 1). È una penalità di efficienza, la perdita non cambia il F/B; 0 ordina per il F/B puro."),
    "--tol-retune": (
        'Monte-Carlo: re-tune the series capacitor (auto) and re-design the matching network for every sample, as if they were adjusted on the bench after the build. Off by default: they stay as designed for the nominal candidate, so their mistuning counts in the spread.',
        'Monte Carlo: reajusta el condensador serie (auto) y rediseña la red de adaptación en cada muestra, como si se ajustaran en el banco tras la construcción. Desactivado por defecto: se mantienen como se diseñaron para el candidato nominal, así que su desajuste cuenta en la dispersión.',
        'Monte Carlo: ritara il condensatore serie (auto) e riprogetta la rete di adattamento per ogni campione, come se venissero regolati al banco dopo la costruzione. Disattivato per impostazione predefinita: restano come progettati per il candidato nominale, quindi il loro disaccordo conta nella dispersione.'),
    "--tol-c-pct": (
        'Monte-Carlo: tolerance of the series feed capacitor setting in percent (default 2). Only when a series capacitor is used.',
        'Monte Carlo: tolerancia del ajuste del condensador serie en porcentaje (por defecto 2). Solo cuando se usa un condensador serie.',
        'Monte Carlo: tolleranza della regolazione del condensatore serie in percentuale (predefinita 2). Solo quando si usa un condensatore serie.'),
    "--tol-z0-pct": (
        'Monte-Carlo: tolerance of the phasing-line Z0 (or of the LC network characteristic impedance) in percent (default 3): wire spacing or cable tolerance.',
        'Monte Carlo: tolerancia de la Z0 de la línea de fase (o de la impedancia característica de la red LC) en porcentaje (por defecto 3): separación de los hilos o tolerancia del cable.',
        "Monte Carlo: tolleranza della Z0 della linea di fase (o dell'impedenza caratteristica della rete LC) in percentuale (predefinita 3): distanza dei fili o tolleranza del cavo."),
    "--tol-vf-pct": (
        'Monte-Carlo: tolerance of the phasing-line velocity factor in percent (default 2): spacers, insulation and moisture change it. The cut length stays, so its electrical length changes.',
        'Monte Carlo: tolerancia del factor de velocidad de la línea de fase en porcentaje (por defecto 2): los separadores, el aislamiento y la humedad lo cambian. La longitud cortada se mantiene, así que cambia su longitud eléctrica.',
        'Monte Carlo: tolleranza del fattore di velocità della linea di fase in percentuale (predefinita 2): distanziatori, isolamento e umidità lo cambiano. La lunghezza tagliata resta, quindi cambia la sua lunghezza elettrica.'),
    "--tol-bar-mm": (
        'Monte-Carlo: tolerance of the T-bar offset from its element, plus or minus this many mm, drawn for each driven element on its own (default 1). Only with the T-match feed.',
        'Monte Carlo: tolerancia de la separación de la barra T respecto a su elemento, más o menos estos mm, sorteada para cada elemento activo por separado (por defecto 1). Solo con alimentación T-match.',
        'Monte Carlo: tolleranza della distanza della barra T dal suo elemento, più o meno questi mm, estratta per ogni elemento attivo separatamente (predefinita 1). Solo con alimentazione T-match.'),
    "--tol-tap-mm": (
        'Monte-Carlo: tolerance of the T-match tap position, plus or minus this many mm, drawn for each driven element on its own (default 2). Only with the T-match feed.',
        'Monte Carlo: tolerancia de la posición de la toma del T-match, más o menos estos mm, sorteada para cada elemento activo por separado (por defecto 2). Solo con alimentación T-match.',
        'Monte Carlo: tolleranza della posizione della presa del T-match, più o meno questi mm, estratta per ogni elemento attivo separatamente (predefinita 2). Solo con alimentazione T-match.'),
    "--nec-out": (
        "nec2c output file (.out) from which the antenna input impedance is read. It must contain a single EX source. For the sweep mode it must contain a frequency sweep, which you obtain with generate, a frequency sweep and 'Run nec2c'. Alternatively, type R and X by hand in match mode.",
        "Archivo de salida de nec2c (.out) del que se lee la impedancia de entrada de la antena. Debe contener una sola fuente EX. Para el modo sweep debe contener un barrido de frecuencia, que se obtiene con generate, un barrido de frecuencia y 'Ejecutar nec2c'. Alternativamente, escribe R y X a mano en el modo match.",
        "File di uscita di nec2c (.out) da cui si legge l'impedenza d'ingresso dell'antenna. Deve contenere una sola sorgente EX. Per la modalità sweep deve contenere una scansione di frequenza, che si ottiene con generate, una scansione di frequenza e 'Esegui nec2c'. In alternativa, digita R e X a mano in modalità match."),
    "--r": (
        "Resistive part R of the load impedance in ohm, typed manually instead of reading a NEC file. It must be accompanied by the reactance X. In sweep mode it is only used with 'Assume constant Z'.",
        "Parte resistiva R de la impedancia de carga en ohm, escrita a mano en lugar de leer un archivo NEC. Debe ir acompañada de la reactancia X. En el modo sweep solo se usa con 'Suponer Z constante'.",
        "Parte resistiva R dell'impedenza di carico in ohm, digitata a mano invece di leggere un file NEC. Deve essere accompagnata dalla reattanza X. In modalità sweep si usa solo con 'Supporre Z costante'."),
    "--x": (
        "Reactive part X of the load impedance in ohm (positive inductive, negative capacitive), typed manually. It must be accompanied by the resistance R. In sweep mode it is only used with 'Assume constant Z'.",
        "Parte reactiva X de la impedancia de carga en ohm (positiva inductiva, negativa capacitiva), escrita a mano. Debe ir acompañada de la resistencia R. En el modo sweep solo se usa con 'Suponer Z constante'.",
        "Parte reattiva X dell'impedenza di carico in ohm (positiva induttiva, negativa capacitiva), digitata a mano. Deve essere accompagnata dalla resistenza R. In modalità sweep si usa solo con 'Supporre Z costante'."),
    "--z0": (
        "Reference (unbalanced) impedance of the source or cable the L-network must match to, in ohm. Default 50. With a 4:1 balun the balanced-side reference becomes 12.5 ohm instead.",
        "Impedancia de referencia (no balanceada) de la fuente o cable a la que debe adaptar la red en L, en ohm. Por defecto 50. Con balun 4:1 la referencia del lado balanceado pasa a ser 12,5 ohm.",
        "Impedenza di riferimento (sbilanciata) della sorgente o del cavo a cui la rete a L deve adattare, in ohm. Predefinita 50. Con balun 4:1 il riferimento lato bilanciato diventa 12,5 ohm."),
    "--q-l": (
        "Quality factor Q of the inductor. It adds a series loss resistance ESR = X/Q so the loss in the matching network is included. Empty means an ideal inductor without loss. Typical air coils at VHF have Q of 100 to 200.",
        "Factor de calidad Q del inductor. Añade una resistencia de pérdidas serie ESR = X/Q para incluir la pérdida de la red de adaptación. Vacío significa un inductor ideal sin pérdidas. Las bobinas de aire típicas en VHF tienen Q de 100 a 200.",
        "Fattore di qualità Q dell'induttore. Aggiunge una resistenza di perdita serie ESR = X/Q per includere la perdita della rete di adattamento. Vuoto significa induttore ideale senza perdite. Le tipiche bobine in aria in VHF hanno Q da 100 a 200."),
    "--q-c": (
        "Quality factor Q of the capacitor. It adds a series loss resistance ESR = X/Q to account for loss in the capacitor. Empty means an ideal capacitor. Good RF ceramic capacitors have Q of several hundred.",
        "Factor de calidad Q del condensador. Añade una resistencia de pérdidas serie ESR = X/Q para tener en cuenta la pérdida del condensador. Vacío significa un condensador ideal. Los buenos condensadores cerámicos de RF tienen Q de varios cientos.",
        "Fattore di qualità Q del condensatore. Aggiunge una resistenza di perdita serie ESR = X/Q per tener conto della perdita del condensatore. Vuoto significa condensatore ideale. I buoni condensatori ceramici RF hanno Q di diverse centinaia."),
    "--balanced": (
        "Also prints the L-network as a balanced network, with the series elements split between the two legs, together with a warning about parasitics above about 100 MHz. Use it when the matching network sits between a balanced antenna feed point and the coax.",
        "Imprime también la red en L como red balanceada, con los elementos serie repartidos entre las dos ramas, junto con un aviso sobre parásitos por encima de unos 100 MHz. Úsalo cuando la red de adaptación esté entre un punto de alimentación balanceado de la antena y el coaxial.",
        "Stampa anche la rete a L come rete bilanciata, con gli elementi serie ripartiti tra i due rami, insieme a un avviso sui parassiti sopra circa 100 MHz. Usalo quando la rete di adattamento si trova tra un punto di alimentazione bilanciato dell'antenna e il coassiale."),
    "--design-freq": (
        "Frequency in MHz at which the L-network is designed in sweep mode (default 145). The component values are fixed at this frequency, and then VSWR and impedance are evaluated across the swept range.",
        "Frecuencia en MHz a la que se diseña la red en L en el modo sweep (por defecto 145). Los valores de los componentes quedan fijados a esta frecuencia y luego se evalúan el VSWR y la impedancia en todo el rango barrido.",
        "Frequenza in MHz alla quale si progetta la rete a L in modalità sweep (predefinita 145). I valori dei componenti sono fissati a questa frequenza e poi si valutano VSWR e impedenza sull'intero intervallo scansionato."),
    "--assume-constant-z": (
        "Instead of reading Zin(f) from a NEC sweep, assume the typed R and X are constant over the range given by Fmin, Fmax and Step. It is a quick idealization that does not represent the real antenna, whose impedance changes with frequency.",
        "En lugar de leer Zin(f) de un barrido de NEC, supone que R y X escritas son constantes en el rango dado por Fmín, Fmáx y Paso. Es una idealización rápida que no representa la antena real, cuya impedancia cambia con la frecuencia.",
        "Invece di leggere Zin(f) da una scansione NEC, assume che R e X digitate siano costanti nell'intervallo dato da Fmin, Fmax e Passo. È un'idealizzazione rapida che non rappresenta l'antenna reale, la cui impedenza cambia con la frequenza."),
    "--fmin": (
        "Lowest frequency of the sweep in MHz (default 140). Only used with 'Assume constant Z'; otherwise the frequencies come from the NEC file.",
        "Frecuencia más baja del barrido en MHz (por defecto 140). Solo se usa con 'Suponer Z constante'; en otro caso las frecuencias vienen del archivo NEC.",
        "Frequenza più bassa della scansione in MHz (predefinita 140). Usata solo con 'Supporre Z costante'; altrimenti le frequenze provengono dal file NEC."),
    "--fmax": (
        "Highest frequency of the sweep in MHz (default 150). Only used with 'Assume constant Z'; otherwise the frequencies come from the NEC file.",
        "Frecuencia más alta del barrido en MHz (por defecto 150). Solo se usa con 'Suponer Z constante'; en otro caso las frecuencias vienen del archivo NEC.",
        "Frequenza più alta della scansione in MHz (predefinita 150). Usata solo con 'Supporre Z costante'; altrimenti le frequenze provengono dal file NEC."),
    "--step": (
        "Frequency step of the sweep in MHz (default 0.25). Only used with 'Assume constant Z'. A smaller step gives a smoother curve and more rows in the CSV.",
        "Paso de frecuencia del barrido en MHz (por defecto 0,25). Solo se usa con 'Suponer Z constante'. Un paso menor da una curva más suave y más filas en el CSV.",
        "Passo di frequenza della scansione in MHz (predefinito 0,25). Usato solo con 'Supporre Z costante'. Un passo minore dà una curva più liscia e più righe nel CSV."),
    "--commercial": (
        "Commercial component values to evaluate, written SERIES,SHUNT in nH for an inductor or pF for a capacitor, according to the ideal network. Separate several pairs with a semicolon, for example 12,3.3;10,3.9. If empty, the nearest E24 values are used. Each pair is plotted as its own curve so you can compare how real values degrade the match.",
        "Valores comerciales de componentes a evaluar, escritos SERIE,PARALELO en nH si es inductor o pF si es condensador, según la red ideal. Separa varios pares con punto y coma, por ejemplo 12,3.3;10,3.9. Si está vacío se usan los valores E24 más cercanos. Cada par se dibuja como su propia curva para comparar cómo los valores reales degradan la adaptación.",
        "Valori commerciali dei componenti da valutare, scritti SERIE,PARALLELO in nH per un induttore o pF per un condensatore, secondo la rete ideale. Separa più coppie con un punto e virgola, per esempio 12,3.3;10,3.9. Se vuoto si usano i valori E24 più vicini. Ogni coppia è disegnata come curva a sé per confrontare come i valori reali peggiorano l'adattamento."),
    "--sweep-csv": (
        "CSV file that receives the L-network sweep: frequency, load impedance, input impedance with the network, VSWR and the network name for every point. Default lnetwork_sweep.csv in the work folder. The GUI reads this file to draw the sweep plot.",
        "Archivo CSV que recibe el barrido de la red en L: frecuencia, impedancia de carga, impedancia de entrada con la red, VSWR y nombre de la red en cada punto. Por defecto lnetwork_sweep.csv en la carpeta de trabajo. La interfaz lee este archivo para dibujar el gráfico del barrido.",
        "File CSV che riceve la scansione della rete a L: frequenza, impedenza di carico, impedenza d'ingresso con la rete, VSWR e nome della rete per ogni punto. Predefinito lnetwork_sweep.csv nella cartella di lavoro. L'interfaccia legge questo file per disegnare il grafico della scansione."),
    "--tol-l": (
        "Inductor tolerance as a percentage (for example 5 for plus or minus 5 %; 0 turns it off). The worst-case VSWR over the corners and edge midpoints of the tolerance box is reported at every frequency of the sweep, for every network (ideal and commercial) around its own nominal values.",
        "Tolerancia del inductor en porcentaje (por ejemplo 5 para más o menos 5 %; 0 la desactiva). Se informa del peor VSWR en las esquinas y los puntos medios de los lados de la caja de tolerancias en cada frecuencia del barrido, para cada red (ideal y comercial) alrededor de sus propios valores nominales.",
        "Tolleranza dell'induttore in percentuale (per esempio 5 per più o meno 5 %; 0 la disattiva). Si riporta il peggior VSWR negli angoli e nei punti medi dei lati del riquadro di tolleranza a ogni frequenza della scansione, per ogni rete (ideale e commerciale) attorno ai propri valori nominali."),
    "--tol-c": (
        "Capacitor tolerance as a percentage (for example 5 for plus or minus 5 %; 0 turns it off). Reports the worst-case VSWR over the tolerance range together with the inductor tolerance.",
        "Tolerancia del condensador en porcentaje (por ejemplo 5 para más o menos 5 %; 0 la desactiva). Informa del peor VSWR dentro del rango de tolerancia junto con la tolerancia del inductor.",
        "Tolleranza del condensatore in percentuale (per esempio 5 per più o meno 5 %; 0 la disattiva). Riporta il peggior VSWR nell'intervallo di tolleranza insieme alla tolleranza dell'induttore."),
    "--lnet-solution": (
        'Every load can be matched by two L-networks of each topology that exists, one per root of the design equation: topology A (series branch on the antenna side) when R < Z0, topology B (shunt branch on the antenna side) when R^2 + X^2 >= R*Z0, so for R < Z0 with a large |X| there are four. plus (default) and minus pick a root of topology A when R < Z0, of B otherwise; lowpass picks a solution with a series inductor and a shunt capacitor and highpass one with a series capacitor and a shunt inductor, of either topology (when no solution has those kinds, the plus or minus root is used and a note is printed); best picks, among all of them, the one with the lower worst VSWR over the NEC sweep (match: when the file holds a sweep; otherwise the one with the smaller spread between the two reactances). The other solutions are always printed.',
        'Toda carga se puede adaptar con dos redes L por cada topología que exista, una por cada raíz de la ecuación de diseño: la topología A (rama serie del lado de la antena) cuando R < Z0 y la B (rama paralelo del lado de la antena) cuando R^2 + X^2 >= R*Z0, así que con R < Z0 y |X| grande hay cuatro. plus (por defecto) y minus eligen una raíz de la topología A si R < Z0, de la B en otro caso; lowpass elige una solución con inductor serie y condensador paralelo y highpass una con condensador serie e inductor paralelo, de cualquier topología (si ninguna tiene esos tipos se usa la raíz plus o minus y se imprime una nota); best elige, entre todas, la de menor VSWR en el peor punto del barrido NEC (match: cuando el archivo contiene un barrido; si no, la de menor dispersión entre las dos reactancias). Las demás soluciones siempre se imprimen.',
        "Ogni carico si può adattare con due reti L per ciascuna topologia esistente, una per radice dell'equazione di progetto: la topologia A (ramo serie dal lato dell'antenna) quando R < Z0 e la B (ramo parallelo dal lato dell'antenna) quando R^2 + X^2 >= R*Z0, quindi con R < Z0 e |X| grande ce ne sono quattro. plus (predefinito) e minus scelgono una radice della topologia A se R < Z0, della B altrimenti; lowpass sceglie una soluzione con induttore serie e condensatore parallelo e highpass una con condensatore serie e induttore parallelo, di qualunque topologia (se nessuna ha quei tipi si usa la radice plus o minus e si stampa una nota); best sceglie, tra tutte, quella con il VSWR peggiore più basso nella scansione NEC (match: quando il file contiene una scansione; altrimenti quella con minore dispersione tra le due reattanze). Le altre soluzioni vengono sempre stampate."),
    "--esl-nh": (
        "Lead inductance in nH of each lead of a capacitor (default 2; a capacitor has two leads). Used to estimate the self-resonant frequency of every capacitor of the L-network: a warning is printed when it falls inside or below the band, a note when it is below 1.5 times the highest frequency.",
        "Inductancia en nH de cada patilla de un condensador (por defecto 2; un condensador tiene dos patillas). Se usa para estimar la frecuencia de autorresonancia de cada condensador de la red L: se avisa cuando cae dentro o por debajo de la banda y se da una nota cuando está por debajo de 1,5 veces la frecuencia más alta.",
        "Induttanza in nH di ciascun reoforo di un condensatore (predefinito 2; un condensatore ha due reofori). Serve a stimare la frequenza di autorisonanza di ogni condensatore della rete L: si avvisa quando cade dentro o sotto la banda e si dà una nota quando è sotto 1,5 volte la frequenza più alta."),
    "--coil-cp-pf": (
        "Stray (inter-turn) capacitance in pF of an inductor (default 0.7, typical of a small air coil). Used for the self-resonance check of every inductor of the L-network, as for the capacitors.",
        "Capacidad parásita (entre espiras) en pF de un inductor (por defecto 0,7, típica de una bobina de aire pequeña). Se usa para comprobar la autorresonancia de cada inductor de la red L, como con los condensadores.",
        "Capacità parassita (tra le spire) in pF di un induttore (predefinito 0,7, tipica di una piccola bobina in aria). Serve per la verifica di autorisonanza di ogni induttore della rete L, come per i condensatori."),
    "--line-allowance-mm": (
        'Extra physical line length in mm for the connections at both ends, the crossing and the stubs (default 20 for an open-wire line, 40 for coax). A straight line cannot be shorter than the distance between its two ends; with this allowance the minimum becomes that distance plus the allowance. generate never picks a shorter line by default and warns when one is given; optimize skips such lines and reports how many it dropped.',
        'Longitud física extra de línea en mm para las conexiones en ambos extremos, el cruce y los tramos (por defecto 20 en línea bifilar, 40 en coaxial). Una línea recta no puede ser más corta que la distancia entre sus dos extremos; con este margen el mínimo pasa a ser esa distancia más el margen. generate nunca elige por defecto una línea más corta y avisa si se indica una; optimize descarta esas líneas e informa de cuántas eliminó.',
        "Lunghezza fisica extra della linea in mm per i collegamenti alle due estremità, l'incrocio e i tratti (predefinito 20 per la linea bifilare, 40 per il coassiale). Una linea dritta non può essere più corta della distanza tra le sue due estremità; con questo margine il minimo diventa quella distanza più il margine. generate non sceglie mai per default una linea più corta e avvisa se ne viene indicata una; optimize scarta tali linee e riporta quante ne ha eliminate."),
    "--height-sweep": (
        "Over ground, generate mode: simulates the model at N heights from MIN to MAX metres, written MIN,MAX,N (for example 4,12,5), and prints the take-off angle, gain, F/B and input impedance at each height. At one or two wavelengths the height changes the pattern more than any build tolerance. Needs nec2c.",
        "Sobre tierra, modo generate: simula el modelo en N alturas de MIN a MAX metros, escrito MIN,MAX,N (por ejemplo 4,12,5), e imprime el ángulo de salida, la ganancia, la F/B y la impedancia de entrada en cada altura. A una o dos longitudes de onda la altura cambia el diagrama más que cualquier tolerancia de construcción. Requiere nec2c.",
        "Sul terreno, modalità generate: simula il modello a N altezze da MIN a MAX metri, scritto MIN,MAX,N (per esempio 4,12,5), e stampa angolo di partenza, guadagno, F/B e impedenza d'ingresso a ogni altezza. A una o due lunghezze d'onda l'altezza cambia il diagramma più di qualsiasi tolleranza di costruzione. Richiede nec2c."),
    "--robustness": (
        'Prints one table with the change of gain, F/B and input impedance caused by parts the default model leaves out, each simulated alone against the model as given: a metallic boom (when the boom is not modelled), a mast placed where it breaks the symmetry of the elements (horizontal: 1 m on a 60 mm side clamp; vertical: a 2 m side-mounted mast) and an unchoked coax shield of half a wavelength. A conductor in the symmetry plane of the balanced elements is not excited by the model; such rows are marked with * and explained instead of being read as robustness. In optimize it runs on the best model. Needs nec2c.',
        'Imprime una tabla con el cambio de ganancia, F/B e impedancia de entrada que producen partes que el modelo por defecto omite, cada una simulada por separado frente al modelo tal como está: un boom metálico (si el boom no se modela), un mástil colocado donde rompe la simetría de los elementos (horizontal: 1 m en una abrazadera lateral de 60 mm; vertical: un mástil lateral de 2 m) y una malla de coaxial sin choke de media longitud de onda. Un conductor en el plano de simetría de los elementos balanceados no lo excita el modelo; esas filas se marcan con * y se explican en vez de leerse como robustez. En optimize se ejecuta sobre el mejor modelo. Requiere nec2c.',
        "Stampa una tabella con la variazione di guadagno, F/B e impedenza d'ingresso causata da parti che il modello predefinito tralascia, ciascuna simulata da sola rispetto al modello così com'è: un boom metallico (se il boom non è modellato), un palo posto dove rompe la simmetria degli elementi (orizzontale: 1 m su un morsetto laterale di 60 mm; verticale: un palo laterale di 2 m) e una calza del coassiale senza choke di mezza lunghezza d'onda. Un conduttore nel piano di simmetria degli elementi bilanciati non viene eccitato dal modello; tali righe sono marcate con * e spiegate invece di essere lette come robustezza. In optimize viene eseguita sul modello migliore. Richiede nec2c."),
    "--check-convergence": (
        "Simulates the model again with 2n+1 segments per element instead of n and prints the change of gain, F/B and input impedance. A warning is printed above 0.1 dB of gain, 1 dB of F/B, 5 % of resistance or 5 % of |Zin|: the segmentation is then too coarse for the numbers to be trusted. |delta Zin| is reported as the Zin uncertainty in the summary, the build sheet and the JSON. Default: on in optimize, where it runs on the best model; off in generate. Needs nec2c.",
        "Vuelve a simular el modelo con 2n+1 segmentos por elemento en vez de n e imprime el cambio de ganancia, F/B e impedancia de entrada. Se avisa por encima de 0,1 dB de ganancia, 1 dB de F/B, 5 % de resistencia o 5 % de |Zin|: entonces la segmentación es demasiado gruesa para fiarse de los números. |delta Zin| se informa como incertidumbre de Zin en el resumen, la hoja de construcción y el JSON. Por defecto: activado en optimize, donde se ejecuta sobre el mejor modelo; desactivado en generate. Requiere nec2c.",
        "Risimula il modello con 2n+1 segmenti per elemento invece di n e stampa la variazione di guadagno, F/B e impedenza d'ingresso. Si avvisa sopra 0,1 dB di guadagno, 1 dB di F/B, 5 % di resistenza o 5 % di |Zin|: in tal caso la segmentazione è troppo grossolana per fidarsi dei numeri. |delta Zin| viene riportato come incertezza di Zin nel riepilogo, nella scheda di costruzione e nel JSON. Predefinito: attivo in optimize, dove viene eseguita sul modello migliore; disattivo in generate. Richiede nec2c."),
    "--feed-style": (
        'Where the phase line and the feed connect. tmatch (the default with the tl and lc networks) is the classic HB9CV: each driven element is a continuous tube with a T-match (two stubs at the tap distance from the centre joined by a bar parallel to the element, on the side facing the other element); the line and the feed connect at the bar centres. The T-match transforms the element impedances so that a short, straight phase line can give the cardioid currents. center connects at the element centres (split, insulated elements) and is used automatically with stepped sections or a modelled feed gap; with a high-impedance open-wire line it cannot reach the cardioid currents with a short line.',
        'Dónde se conectan la línea de fase y la alimentación. tmatch (por defecto con las redes tl y lc) es la HB9CV clásica: cada elemento activo es un tubo continuo con un T-match (dos tramos a la distancia de toma del centro unidos por una barra paralela al elemento, del lado que mira al otro elemento); la línea y la alimentación se conectan en los centros de las barras. El T-match transforma las impedancias de los elementos de modo que una línea de fase corta y recta pueda dar las corrientes de cardioide. center conecta en los centros de los elementos (elementos partidos y aislados) y se usa automáticamente con secciones escalonadas o con la separación de alimentación modelada; con una línea bifilar de alta impedancia no alcanza las corrientes de cardioide con una línea corta.',
        "Dove si collegano la linea di fase e l'alimentazione. tmatch (predefinito con le reti tl e lc) è la HB9CV classica: ogni elemento attivo è un tubo continuo con un T-match (due tratti alla distanza di presa dal centro uniti da una barra parallela all'elemento, dal lato rivolto all'altro elemento); la linea e l'alimentazione si collegano ai centri delle barre. Il T-match trasforma le impedenze degli elementi in modo che una linea di fase corta e diritta possa dare le correnti a cardioide. center collega ai centri degli elementi (elementi divisi e isolati) e si usa automaticamente con sezioni a gradini o con il gap di alimentazione modellato; con una linea bifilare ad alta impedenza non raggiunge le correnti a cardioide con una linea corta."),
    "--series-c-pf": (
        "Series capacitor in pF between the feed line and the fed port: the trimmer of the classic HB9CV. It is written to the NEC file as a comment and added, with its loss (ESR = |X|/Q, see 'Series C quality factor'), to every input impedance read from the output file (also by match and sweep, and by the pattern plots). auto chooses the value that cancels the input reactance at the design frequency (no capacitor when the reactance is capacitive); in optimize it is chosen for every candidate and then kept fixed in the Monte-Carlo.",
        "Condensador serie en pF entre la línea de alimentación y el puerto alimentado: el trimmer de la HB9CV clásica. Se escribe en el archivo NEC como comentario y se suma, con su pérdida (ESR = |X|/Q, ver 'Q del C serie'), a toda impedancia de entrada leída del archivo de salida (también en match y sweep, y en los gráficos de diagrama). auto elige el valor que anula la reactancia de entrada a la frecuencia de diseño (sin condensador si la reactancia es capacitiva); en optimize se elige para cada candidato y luego se mantiene fijo en el Monte Carlo.",
        "Condensatore serie in pF tra la linea di alimentazione e la porta alimentata: il trimmer della HB9CV classica. Viene scritto nel file NEC come commento e sommato, con la sua perdita (ESR = |X|/Q, vedi 'Q del C serie'), a ogni impedenza d'ingresso letta dal file di uscita (anche in match e sweep e nei grafici del diagramma). auto sceglie il valore che annulla la reattanza d'ingresso alla frequenza di progetto (nessun condensatore se la reattanza è capacitiva); in optimize viene scelto per ogni candidato e poi mantenuto fisso nel Monte Carlo."),
    "--series-c-q": (
        'Quality factor of the series feed capacitor. Its loss resistance ESR = |X|/Q is added to the input impedance together with its reactance. Default 500 (a good air or PTFE trimmer); 0 makes it lossless.',
        'Factor de calidad del condensador serie de alimentación. Su resistencia de pérdidas ESR = |X|/Q se suma a la impedancia de entrada junto con su reactancia. Por defecto 500 (un buen trimmer de aire o PTFE); 0 lo hace sin pérdidas.',
        "Fattore di qualità del condensatore serie di alimentazione. La sua resistenza di perdita ESR = |X|/Q viene sommata all'impedenza d'ingresso insieme alla sua reattanza. Predefinito 500 (un buon trimmer in aria o PTFE); 0 lo rende senza perdite."),
    "--line-loss-db-per-m": (
        'Build sheet only, network tl: matched loss of the phasing line in dB per metre. The NEC TL card is lossless, so the build sheet reports the loss times the physical length as an upper bound (as if all the power crossed the line). 0 leaves it out of the report.',
        'Solo hoja de construcción, red tl: pérdida adaptada de la línea de fase en dB por metro. La tarjeta TL de NEC no tiene pérdidas, así que la hoja informa la pérdida por la longitud física como cota superior (como si toda la potencia atravesara la línea). 0 la deja fuera del informe.',
        'Solo scheda di costruzione, rete tl: perdita adattata della linea di fase in dB per metro. La scheda TL di NEC è senza perdite, quindi la scheda riporta la perdita per la lunghezza fisica come limite superiore (come se tutta la potenza attraversasse la linea). 0 la esclude dal rapporto.'),
    "--synthesize-line": (
        'Computes the phase line from the driven elements themselves: two nec2c probe runs (1 V on one feed port, no network) give how each port, each element current and the far field respond. Superposing the two probe patterns gives the exact pattern of the antenna for any line, and the line that maximizes the objective (optimize: with the F/B metric, the rear sector and the gain/F/B weights) is chosen: on the horizon ring in free space, on the ring at the take-off angle over ground. The ratio of the element-centre currents to the point-source cardioid (equal amplitudes, rear leading by 180 minus the spacing in degrees) is reported as a diagnostic: with unequal elements and T-matches it is not the F/B optimum. generate uses it as the default line; optimize centres the line scan of every spacing, Z0 and tap on it and repeats it after the pre-tune. no uses the cardioid theory, never shorter than the buildable minimum.',
        'Calcula la línea de fase a partir de los propios elementos activos: dos ejecuciones de prueba de nec2c (1 V en un puerto, sin red) dan cómo responden cada puerto, la corriente de cada elemento y el campo lejano. Superponiendo los dos diagramas de prueba se obtiene el diagrama exacto de la antena con cualquier línea, y se elige la línea que maximiza el objetivo (optimize: con la métrica de F/B, el sector trasero y los pesos de ganancia y F/B): en el anillo del horizonte en espacio libre, en el anillo del ángulo de despegue sobre suelo. La relación de las corrientes en el centro de los elementos frente a la cardioide de fuentes puntuales (amplitudes iguales, el trasero adelantado 180 menos la separación en grados) se informa como diagnóstico: con elementos desiguales y T-match no es el óptimo de F/B. generate la usa como línea por defecto; optimize centra en ella la exploración de línea de cada separación, Z0 y toma, y la repite tras la presintonía. no usa la teoría de cardioide, nunca más corta que el mínimo construible.',
        "Calcola la linea di fase dagli elementi attivi stessi: due esecuzioni di prova di nec2c (1 V su una porta, nessuna rete) danno come rispondono ogni porta, la corrente di ogni elemento e il campo lontano. Sovrapponendo i due diagrammi di prova si ottiene il diagramma esatto dell'antenna con qualunque linea, e si sceglie la linea che massimizza l'obiettivo (optimize: con la metrica F/B, il settore posteriore e i pesi di guadagno e F/B): sull'anello dell'orizzonte nello spazio libero, sull'anello all'angolo di partenza sopra il suolo. Il rapporto delle correnti al centro degli elementi rispetto alla cardioide di sorgenti puntiformi (ampiezze uguali, il posteriore in anticipo di 180 meno la spaziatura in gradi) viene riportato come diagnostica: con elementi diversi e T-match non è l'ottimo di F/B. generate la usa come linea predefinita; optimize vi centra l'esplorazione della linea di ogni spaziatura, Z0 e presa e la ripete dopo la pre-sintonia. no usa la teoria della cardioide, mai più corta del minimo costruibile."),
    "--match-score": (
        "Scores every optimize candidate after the ideal L-network designed at the design frequency against the target impedance, built with components of the given Q: the VSWR penalty uses the matched VSWR over the band and the gain objectives use the gain after the network loss and residual mismatch. The front-back objective keeps the pure F/B and pays 'Network loss weight' dB of score per dB of network loss (an efficiency penalty). It lets the optimizer use the low or high input resistances where the best patterns are, paying for the matching network. Default on with tl/lc; no searches for a direct connection.",
        "Puntúa cada candidato de optimize tras la red L ideal diseñada a la frecuencia de diseño contra la impedancia objetivo, construida con componentes del Q indicado: la penalización por VSWR usa el VSWR adaptado en la banda y los objetivos de ganancia usan la ganancia tras la pérdida de la red y la desadaptación residual. El objetivo front-back conserva la F/B pura y paga 'Peso de la pérdida de red' dB de puntuación por dB de pérdida de la red (una penalización de eficiencia). Permite al optimizador usar las resistencias de entrada bajas o altas donde están los mejores diagramas, pagando la red de adaptación. Por defecto activado con tl/lc; no busca una conexión directa.",
        "Valuta ogni candidato di optimize dopo la rete L ideale progettata alla frequenza di progetto sull'impedenza obiettivo, costruita con componenti del Q indicato: la penalità VSWR usa il VSWR adattato sulla banda e gli obiettivi di guadagno usano il guadagno dopo la perdita della rete e il disadattamento residuo. L'obiettivo front-back conserva il F/B puro e paga 'Peso della perdita di rete' dB di punteggio per dB di perdita della rete (una penalità di efficienza). Permette all'ottimizzatore di usare le resistenze d'ingresso basse o alte dove si trovano i diagrammi migliori, pagando la rete di adattamento. Predefinito attivo con tl/lc; no cerca un collegamento diretto."),
    "--diameter-values": (
        "Element tube diameters in mm that you can buy, comma separated (default 6,8,10,12,14,16). The starting element diameter is snapped to the nearest one and 'Refine diameter' steps through this list, so the result is always a real tube size. 'any' moves the diameter continuously within the diameter range.",
        "Diámetros de tubo en mm que puedes comprar, separados por comas (por defecto 6,8,10,12,14,16). El diámetro inicial se ajusta al más cercano y 'Refinar diámetro' recorre esta lista, de modo que el resultado es siempre un tubo real. 'any' mueve el diámetro de forma continua dentro del rango de diámetros.",
        "Diametri dei tubi in mm che puoi acquistare, separati da virgole (predefinito 6,8,10,12,14,16). Il diametro iniziale viene portato al più vicino e 'Raffinare diametro' percorre questo elenco, così il risultato è sempre un tubo reale. 'any' muove il diametro in modo continuo nell'intervallo dei diametri."),
}


HELP.update({
    "--reflectors": (
        "Number of passive reflectors placed behind the rear element (0, 1 or 2; NEC tags 21-22). Each one is a continuous tube in the plane of the driven elements, with no source. The program pre-calculates for each reflector a minimum, nominal and maximum length (1.00 to 1.15 times the resonant dipole length of the tube) and spacing (from 0.05 lambda or the tube radii plus the clamp gap, up to 0.40 lambda), prints them as a table and lengthens the boom to hold every element. optimize searches inside those limits.",
        "Número de reflectores pasivos detrás del elemento trasero (0, 1 o 2; etiquetas NEC 21-22). Cada uno es un tubo continuo en el plano de los elementos excitados, sin fuente. El programa precalcula para cada reflector una longitud mínima, nominal y máxima (1,00 a 1,15 veces la longitud del dipolo resonante del tubo) y una separación (desde 0,05 lambda o los radios de los tubos más el hueco de la abrazadera, hasta 0,40 lambda), las muestra en una tabla y alarga el boom para sostener todos los elementos. optimize busca dentro de esos límites.",
        "Numero di riflettori passivi dietro l'elemento posteriore (0, 1 o 2; tag NEC 21-22). Ognuno è un tubo continuo nel piano degli elementi alimentati, senza sorgente. Il programma precalcola per ogni riflettore una lunghezza minima, nominale e massima (da 1,00 a 1,15 volte la lunghezza del dipolo risonante del tubo) e una spaziatura (da 0,05 lambda o dai raggi dei tubi più lo spazio del morsetto, fino a 0,40 lambda), le mostra in una tabella e allunga il boom per sostenere tutti gli elementi. optimize cerca entro questi limiti."),
    "--directors": (
        "Number of passive directors placed in front of the front element (0 to 5; NEC tags 31-35). They are treated like the reflectors: one continuous tube each, lengths pre-calculated between 0.80 and 1.00 times the resonant dipole length of the tube, spacings between the electrical/mechanical minimum and 0.40 lambda, and the boom grows to hold them. Directors raise the gain and change the input impedance a lot; re-check Zin and the matching after adding them.",
        "Número de directores pasivos delante del elemento delantero (0 a 5; etiquetas NEC 31-35). Se tratan como los reflectores: un tubo continuo cada uno, longitudes precalculadas entre 0,80 y 1,00 veces la longitud del dipolo resonante del tubo, separaciones entre el mínimo eléctrico/mecánico y 0,40 lambda, y el boom crece para sostenerlos. Los directores aumentan la ganancia y cambian mucho la impedancia de entrada; vuelve a comprobar Zin y la adaptación después de añadirlos.",
        "Numero di direttori passivi davanti all'elemento anteriore (da 0 a 5; tag NEC 31-35). Sono trattati come i riflettori: un tubo continuo ciascuno, lunghezze precalcolate tra 0,80 e 1,00 volte la lunghezza del dipolo risonante del tubo, spaziature tra il minimo elettrico/meccanico e 0,40 lambda, e il boom si allunga per sostenerli. I direttori aumentano il guadagno e cambiano molto l'impedenza d'ingresso; ricontrolla Zin e l'adattamento dopo averli aggiunti."),
    "--reflector-lengths": (
        "Lengths of the reflectors as factors of the resonant dipole length of the parasitic tube, separated by commas (one value per reflector, e.g. 1.05,1.06). Missing values use the nominal 1.05. In generate this is the length used; in optimize it is the starting point of the search, which stays between 1.00 and 1.15.",
        "Longitudes de los reflectores como factores de la longitud del dipolo resonante del tubo parásito, separadas por comas (un valor por reflector, p. ej. 1.05,1.06). Los valores que falten usan el nominal 1,05. En generate es la longitud usada; en optimize es el punto de partida de la búsqueda, que se mantiene entre 1,00 y 1,15.",
        "Lunghezze dei riflettori come fattori della lunghezza del dipolo risonante del tubo parassita, separate da virgole (un valore per riflettore, es. 1.05,1.06). I valori mancanti usano il nominale 1,05. In generate è la lunghezza usata; in optimize è il punto di partenza della ricerca, che resta tra 1,00 e 1,15."),
    "--reflector-spacings": (
        "Spacing of each reflector in wavelengths, measured from the previous element: the first from the rear element, the second from the first reflector. Comma-separated, e.g. 0.15,0.20 (the nominal values). The minimum is the larger of 0.05 lambda and the tube radii plus the clamp gap (--min-gap-mm); the maximum is 0.40 lambda.",
        "Separación de cada reflector en longitudes de onda, medida desde el elemento anterior: el primero desde el elemento trasero, el segundo desde el primer reflector. Separadas por comas, p. ej. 0.15,0.20 (los valores nominales). El mínimo es el mayor entre 0,05 lambda y los radios de los tubos más el hueco de la abrazadera (--min-gap-mm); el máximo es 0,40 lambda.",
        "Spaziatura di ogni riflettore in lunghezze d'onda, misurata dall'elemento precedente: il primo dall'elemento posteriore, il secondo dal primo riflettore. Separate da virgole, es. 0.15,0.20 (i valori nominali). Il minimo è il maggiore tra 0,05 lambda e i raggi dei tubi più lo spazio del morsetto (--min-gap-mm); il massimo è 0,40 lambda."),
    "--director-lengths": (
        "Lengths of the directors as factors of the resonant dipole length of the parasitic tube, comma-separated (one per director). Missing values use the nominal 0.96, 0.95, 0.94, 0.93, 0.92. In generate this is the length used; in optimize the starting point of a search limited to 0.80..0.99 (a director must stay shorter than resonance; the resonant length is fitted to NEC-2 thin-wire results).",
        "Longitudes de los directores como factores de la longitud del dipolo resonante del tubo parásito, separadas por comas (una por director). Los valores que falten usan los nominales 0,96, 0,95, 0,94, 0,93, 0,92. En generate es la longitud usada; en optimize el punto de partida de una búsqueda limitada a 0,80..0,99 (un director debe quedar más corto que la resonancia; la longitud resonante está ajustada a resultados de hilo fino de NEC-2).",
        "Lunghezze dei direttori come fattori della lunghezza del dipolo risonante del tubo parassita, separate da virgole (una per direttore). I valori mancanti usano i nominali 0,96, 0,95, 0,94, 0,93, 0,92. In generate è la lunghezza usata; in optimize il punto di partenza di una ricerca limitata a 0,80..0,99 (un direttore deve restare più corto della risonanza; la lunghezza risonante è adattata ai risultati a filo sottile di NEC-2)."),
    "--director-spacings": (
        "Spacing of each director in wavelengths from the previous element: the first from the front element, each next one from the director before it. Comma-separated; the nominal values are 0.15, 0.20, 0.25, 0.30, 0.30. Limits as for the reflectors (0.05 lambda or the mechanical minimum, up to 0.40 lambda).",
        "Separación de cada director en longitudes de onda desde el elemento anterior: el primero desde el elemento delantero, cada uno de los siguientes desde el director anterior. Separadas por comas; los valores nominales son 0,15, 0,20, 0,25, 0,30, 0,30. Límites como en los reflectores (0,05 lambda o el mínimo mecánico, hasta 0,40 lambda).",
        "Spaziatura di ogni direttore in lunghezze d'onda dall'elemento precedente: il primo dall'elemento anteriore, ciascuno dei successivi dal direttore precedente. Separate da virgole; i valori nominali sono 0,15, 0,20, 0,25, 0,30, 0,30. Limiti come per i riflettori (0,05 lambda o il minimo meccanico, fino a 0,40 lambda)."),
    "--parasitic-diameter": (
        "Tube diameter in mm of all reflectors and directors. Empty: the element diameter at the start of the run. The resonant length that the length factors refer to depends on it. optimize snaps it to the nearest purchasable diameter (--diameter-values) but never moves it during the refinement.",
        "Diámetro en mm del tubo de todos los reflectores y directores. Vacío: el diámetro de los elementos al comienzo de la ejecución. De él depende la longitud resonante a la que se refieren los factores de longitud. optimize lo ajusta al diámetro comercial más cercano (--diameter-values) pero nunca lo mueve durante el refinamiento.",
        "Diametro in mm del tubo di tutti i riflettori e direttori. Vuoto: il diametro degli elementi all'inizio dell'esecuzione. Da esso dipende la lunghezza risonante a cui si riferiscono i fattori di lunghezza. optimize lo porta al diametro commerciale più vicino (--diameter-values) ma non lo muove mai durante il raffinamento."),
    "--min-gap-mm": (
        "Smallest surface-to-surface gap in mm between neighbouring elements, the room needed for clamps and insulators (default 20). Together with the 0.05 lambda electrical floor it sets the minimum spacing of the reflectors and directors.",
        "Hueco mínimo en mm entre superficies de elementos vecinos, el espacio necesario para abrazaderas y aisladores (por defecto 20). Junto con el mínimo eléctrico de 0,05 lambda fija la separación mínima de reflectores y directores.",
        "Spazio minimo in mm tra le superfici di elementi vicini, lo spazio necessario per morsetti e isolatori (predefinito 20). Insieme al minimo elettrico di 0,05 lambda fissa la spaziatura minima di riflettori e direttori."),
    "--boom-overhang-mm": (
        "Boom length in mm beyond the outer elements at each end (default 50). With reflectors or directors the modelled boom is the longer of --boom-length and the span of all elements plus twice this overhang; the build sheet gives the element positions from the rear end of that boom.",
        "Longitud del boom en mm más allá de los elementos exteriores en cada extremo (por defecto 50). Con reflectores o directores el boom modelado es el mayor entre --boom-length y la distancia entre todos los elementos más dos veces este sobrante; la hoja de construcción da las posiciones de los elementos desde el extremo trasero de ese boom.",
        "Lunghezza del boom in mm oltre gli elementi esterni a ciascuna estremità (predefinito 50). Con riflettori o direttori il boom modellato è il maggiore tra --boom-length e l'estensione di tutti gli elementi più due volte questa sporgenza; il foglio di costruzione dà le posizioni degli elementi dall'estremità posteriore di quel boom."),
    "--tap-mm": (
        "T-match feed: distance in mm from the element centre to each tap, the point where a stub joins the element (default 120). The same value is used on both driven elements. Larger taps raise the impedance seen at the bar centre and change the phase relation of the element currents; optimize moves the tap in 10 mm steps (and scans --tap-values in the grid). The tap must stay at least 1.5 segments from the centre and one segment from the tip.",
        "Alimentación T-match: distancia en mm del centro del elemento a cada toma, el punto donde un brazo se une al elemento (por defecto 120). Se usa el mismo valor en los dos elementos excitados. Tomas más alejadas elevan la impedancia vista en el centro de la barra y cambian la relación de fase de las corrientes; optimize mueve la toma en pasos de 10 mm (y recorre --tap-values en la rejilla). La toma debe quedar al menos a 1,5 segmentos del centro y a un segmento de la punta.",
        "Alimentazione T-match: distanza in mm dal centro dell'elemento a ciascuna presa, il punto in cui un braccio si unisce all'elemento (predefinito 120). Lo stesso valore vale per entrambi gli elementi alimentati. Prese più lontane alzano l'impedenza vista al centro della barra e cambiano la relazione di fase delle correnti; optimize sposta la presa a passi di 10 mm (e percorre --tap-values nella griglia). La presa deve restare almeno a 1,5 segmenti dal centro e a un segmento dalla punta."),
    "--tap-offset-mm": (
        "T-match feed: distance in mm from the element axis to the T-bar, on the side facing the other element (default 50). The phase line runs between the two bar centres, so a larger offset shortens the straight distance it has to span. NEC-2 loses accuracy as the bar comes closer than about 5 element diameters; check the average gain test.",
        "Alimentación T-match: distancia en mm del eje del elemento a la barra T, del lado que mira al otro elemento (por defecto 50). La línea de fase va entre los centros de las dos barras, así que una separación mayor acorta la distancia recta que debe cubrir. NEC-2 pierde precisión cuando la barra se acerca a menos de unos 5 diámetros del elemento; comprueba la prueba de ganancia media.",
        "Alimentazione T-match: distanza in mm dall'asse dell'elemento alla barra T, sul lato rivolto all'altro elemento (predefinito 50). La linea di fase corre tra i centri delle due barre, quindi una distanza maggiore accorcia il tratto rettilineo da coprire. NEC-2 perde precisione quando la barra si avvicina a meno di circa 5 diametri dell'elemento; controlla il test del guadagno medio."),
    "--tap-wire-mm": (
        "T-match feed: diameter in mm of the T-bar and its two stubs. Empty: the element diameter, which is the most accurate choice in NEC-2 (junctions of unequal radii lower the accuracy; a thin bar on a thick element can make the average gain test fail).",
        "Alimentación T-match: diámetro en mm de la barra T y de sus dos brazos. Vacío: el diámetro del elemento, que es la opción más precisa en NEC-2 (las uniones de radios distintos reducen la precisión; una barra fina sobre un elemento grueso puede hacer fallar la prueba de ganancia media).",
        "Alimentazione T-match: diametro in mm della barra T e dei suoi due bracci. Vuoto: il diametro dell'elemento, la scelta più precisa in NEC-2 (le giunzioni di raggi diversi riducono la precisione; una barra sottile su un elemento spesso può far fallire il test del guadagno medio)."),
    "--max-line-deg": (
        "Longest phase line, in electrical degrees, that generate picks and optimize scans. Empty: a line at most 1.5 times the straight distance between its ends plus --line-allowance-mm, so that it can be run almost straight; when the theoretical line is beyond 180 degrees (a crossed line fed at the rear element) the limit covers it plus the largest line offset. A line more than twice the distance it spans is reported as one that must be folded or coiled. The LC network is always 0..180 degrees.",
        "Línea de fase más larga, en grados eléctricos, que elige generate y que recorre optimize. Vacío: una línea de como mucho 1,5 veces la distancia recta entre sus extremos más --line-allowance-mm, para que pueda tenderse casi recta; cuando la línea teórica supera 180 grados (una línea cruzada alimentada en el elemento trasero) el límite la cubre más el mayor desplazamiento de línea. Una línea de más del doble de la distancia que cubre se informa como una que debe plegarse o enrollarse. La red LC es siempre de 0..180 grados.",
        "Linea di fase più lunga, in gradi elettrici, che generate sceglie e optimize percorre. Vuoto: una linea al massimo 1,5 volte la distanza rettilinea tra i suoi estremi più --line-allowance-mm, così da poterla stendere quasi dritta; quando la linea teorica supera 180 gradi (una linea incrociata alimentata sull'elemento posteriore) il limite la comprende più il maggiore scostamento di linea. Una linea più lunga del doppio della distanza che copre viene segnalata come da piegare o avvolgere. La rete LC è sempre 0..180 gradi."),
    "--mast-offset-mm": (
        "Horizontal elements: sideways offset in mm of the mast from the boom axis, along the elements (the arm of a side clamp). With 0 the mast lies in the symmetry plane of the balanced elements and the model does not excite it at all. Vertical elements: extra distance of the side-mounted mast behind the rear end of the boom.",
        "Elementos horizontales: desplazamiento lateral en mm del mástil respecto al eje del boom, a lo largo de los elementos (el brazo de una abrazadera lateral). Con 0 el mástil queda en el plano de simetría de los elementos balanceados y el modelo no lo excita en absoluto. Elementos verticales: distancia adicional del mástil lateral detrás del extremo trasero del boom.",
        "Elementi orizzontali: spostamento laterale in mm del palo rispetto all'asse del boom, lungo gli elementi (il braccio di un morsetto laterale). Con 0 il palo giace nel piano di simmetria degli elementi bilanciati e il modello non lo eccita affatto. Elementi verticali: distanza aggiuntiva del palo laterale dietro l'estremità posteriore del boom."),
    "--parasitic-length-offsets": (
        "Grid offsets added to the length factor of EVERY reflector and director at the same time, comma-separated (e.g. -0.02,0,0.02). Each value multiplies the number of grid candidates. The result is clamped to the pre-calculated limits; the refinement then tunes each element on its own. Needs --reflectors or --directors.",
        "Desplazamientos de rejilla sumados a la vez al factor de longitud de TODOS los reflectores y directores, separados por comas (p. ej. -0.02,0,0.02). Cada valor multiplica el número de candidatos de la rejilla. El resultado se limita a los límites precalculados; después el refinamiento ajusta cada elemento por separado. Requiere --reflectors o --directors.",
        "Scostamenti di griglia sommati contemporaneamente al fattore di lunghezza di TUTTI i riflettori e direttori, separati da virgole (es. -0.02,0,0.02). Ogni valore moltiplica il numero di candidati della griglia. Il risultato è limitato ai valori precalcolati; poi il raffinamento regola ogni elemento separatamente. Richiede --reflectors o --directors."),
    "--parasitic-spacing-scales": (
        "Grid multipliers applied to the spacing of EVERY reflector and director at the same time, comma-separated (e.g. 0.9,1,1.1). Each value multiplies the number of grid candidates. The result is clamped to the pre-calculated minimum and maximum spacing; the refinement then tunes each spacing on its own. Needs --reflectors or --directors.",
        "Multiplicadores de rejilla aplicados a la vez a la separación de TODOS los reflectores y directores, separados por comas (p. ej. 0.9,1,1.1). Cada valor multiplica el número de candidatos de la rejilla. El resultado se limita a la separación mínima y máxima precalculadas; después el refinamiento ajusta cada separación por separado. Requiere --reflectors o --directors.",
        "Moltiplicatori di griglia applicati contemporaneamente alla spaziatura di TUTTI i riflettori e direttori, separati da virgole (es. 0.9,1,1.1). Ogni valore moltiplica il numero di candidati della griglia. Il risultato è limitato alla spaziatura minima e massima precalcolate; poi il raffinamento regola ogni spaziatura separatamente. Richiede --reflectors o --directors."),
    "--match-q-l": (
        "With the matched score: quality factor Q of the inductor of the ideal L-network every candidate is scored through (default 150). Its ESR = X/Q is included, so the score pays the real dissipation of the network; a low Q penalizes designs whose impedance is far from the target.",
        "Con la puntuación adaptada: factor de calidad Q de la bobina de la red en L ideal con la que se puntúa cada candidato (por defecto 150). Se incluye su ESR = X/Q, así que la puntuación paga la disipación real de la red; una Q baja penaliza los diseños cuya impedancia está lejos del objetivo.",
        "Con il punteggio adattato: fattore di merito Q dell'induttore della rete a L ideale con cui si valuta ogni candidato (predefinito 150). È inclusa la sua ESR = X/Q, quindi il punteggio paga la dissipazione reale della rete; un Q basso penalizza i progetti la cui impedenza è lontana dall'obiettivo."),
    "--match-q-c": (
        "With the matched score: quality factor Q of the capacitor of the ideal L-network (default 500, a good air or PTFE trimmer). Its ESR = |X|/Q is included in the network loss of every candidate.",
        "Con la puntuación adaptada: factor de calidad Q del condensador de la red en L ideal (por defecto 500, un buen trimmer de aire o PTFE). Su ESR = |X|/Q se incluye en la pérdida de la red de cada candidato.",
        "Con il punteggio adattato: fattore di merito Q del condensatore della rete a L ideale (predefinito 500, un buon compensatore ad aria o in PTFE). La sua ESR = |X|/Q è inclusa nella perdita della rete di ogni candidato."),
    "--tap-values": (
        "T-match feed: tap distances in mm scanned by the grid, comma-separated (e.g. 100,120,140). Empty: only --tap-mm. Each value multiplies the number of grid candidates and gets its own synthesized phase line; the refinement then moves the tap of the best candidate in 10 mm steps.",
        "Alimentación T-match: distancias de toma en mm que recorre la rejilla, separadas por comas (p. ej. 100,120,140). Vacío: solo --tap-mm. Cada valor multiplica el número de candidatos de la rejilla y tiene su propia línea de fase sintetizada; después el refinamiento mueve la toma del mejor candidato en pasos de 10 mm.",
        "Alimentazione T-match: distanze delle prese in mm percorse dalla griglia, separate da virgole (es. 100,120,140). Vuoto: solo --tap-mm. Ogni valore moltiplica il numero di candidati della griglia e ha la propria linea di fase sintetizzata; poi il raffinamento sposta la presa del miglior candidato a passi di 10 mm."),
})


def opt_help(o) -> str:
    """Full help text of an option (explanation + modes + default), never longer than HELP_MAX characters."""
    entry = HELP.get(o.flag)
    body = (entry[_LANG_IDX[_LANG]] or entry[0]) if entry else opt_tip(o)
    modes = ", ".join(tr("mode." + m) for m in MODES if m[0].upper() in o.modes)
    suffix = "\n\n" + tr("help.modes", m=modes)
    if o.kind != "bool" and o.default != "":
        suffix += "\n" + tr("help.default", d=o.default)
    room = HELP_MAX - len(suffix)
    if len(body) > room:
        body = body[:room - 1].rstrip() + "\u2026"
    return body + suffix


# ======================================================================================
# Persistent settings (language, scale)
# ======================================================================================

def detect_language() -> str:
    cands = []
    try:
        if sys.platform.startswith("win"):
            import ctypes
            lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()   # type: ignore[attr-defined]
            cands.append(locale.windows_locale.get(lid, ""))
    except Exception:
        pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"):
        cands.append(os.environ.get(var, ""))
    try:
        cands.append(locale.getlocale()[0] or "")
    except Exception:
        pass
    for c in cands:
        c = (c or "").lower()
        for code in ("es", "it", "en"):
            if c.startswith(code):
                return code
    return "en"


def load_config() -> dict:
    cfg = {"language": None, "scale": "auto"}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if data.get("language") in LANGS:
            cfg["language"] = data["language"]
        s = data.get("scale", "auto")
        if s == "auto" or (isinstance(s, (int, float)) and not isinstance(s, bool) and 0.5 <= float(s) <= 4.0):
            cfg["scale"] = s
    except Exception:
        pass
    if cfg["language"] is None:
        cfg["language"] = detect_language()
    return cfg


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except OSError:
        pass


# ======================================================================================
# Option table (mirrors the argparse definition of hb9cv.py)
# ======================================================================================

class Opt:
    __slots__ = ("flag", "label", "kind", "default", "modes", "group", "tip", "choices", "ph")

    def __init__(self, flag, label, kind, default, modes, group, tip="", choices=None, ph=""):
        self.flag, self.label, self.kind, self.default = flag, label, kind, default
        self.modes, self.group, self.tip, self.choices, self.ph = modes, group, tip, choices, ph


def _o(*a, **k):
    return Opt(*a, **k)


# modes letters: G generate, O optimize, M match, S sweep
OPTS = [
    # ---------------------------------------------------------------- Main
    _o("--freq", "Frequency (MHz)", "float", "", "GOM", "Main",
       "generate/optimize: design frequency (default 145). match: must exist in the NEC file.", ph="145"),
    _o("--network", "Feed network", "choice", "tl", "GO", "Main",
       "tl: one EX + ideal transmission line; lc: one EX + ideal pi LC network; ideal: two EX sources.",
       choices=["tl", "lc", "ideal"]),
    _o("--feed-element", "Fed element", "choice", "front", "GO", "Main",
       "Element that receives the single EX source (classic HB9CV: front).", choices=["front", "rear"]),
    _o("--objective", "Objective", "choice", "front-back", "GO", "Main",
       "optimize: scoring criterion. generate: informative only.", choices=["front-back", "gain", "balance"]),
    _o("--mount", "Mount", "choice", "isolated", "GO", "Main",
       "isolated: elements insulated from the boom (default). contact: element centres bonded to a metallic boom "
       "(T-match feed only).", choices=["isolated", "contact"]),
    _o("--balun-ratio", "Balun ratio", "choice", "1:1", "GOMS", "Main",
       "1:1 current balun, or 4:1 (balanced-side reference 12.5 ohm).", choices=["1:1", "4:1"]),
    _o("--target-impedance", "Target Z (ohm)", "float", "", "GO", "Main",
       "Balanced-side reference. Default 50, or 12.5 with balun 4:1.", ph="50"),
    _o("--segments", "Segments / element", "int", "", "GO", "Main", "Odd, >= 5. Default 21, or 41 with the T-match feed.",
       ph="21 / 41"),
    _o("--az-step", "Azimuth step (deg)", "float", "1.0", "GO", "Main",
       "Pattern step written to the deck; must divide 180 exactly."),
    _o("--kernel", "NEC kernel", "choice", "auto", "GO", "Main",
       "auto: extended thin-wire kernel (EK) when a segment is shorter than 8 wire radii; thin / extended force one.",
       choices=["auto", "thin", "extended"]),
    _o("--conductivity", "Conductivity (S/m)", "float", "2.5e7", "GO", "Main",
       "Wire conductivity (LD 5). 2.5e7 = 6061 aluminium, 5.8e7 copper, 0 = perfect conductors."),
    _o("--output", "Output .nec file", "file_save", "", "GO", "Main",
       "Default: derived from the parameters (generate) or hb9cv_optimized.nec (optimize).", ph="(automatic)"),
    _o("--build-sheet", "Build sheet (.md)", "file_save", "", "GO", "Main",
       "Writes the build sheet (markdown + json). generate needs 'Run nec2c'; optimize needs network tl/lc.",
       ph="(none)"),
    _o("--run", "Run nec2c after generating", "bool", "0", "G", "Main",
       "Runs nec2c and prints the pattern summary, element currents and conductor-loss check."),

    # ---------------------------------------------------------------- Antenna
    _o("--spacing-factor", "Spacing (wavelengths)", "float", "0.125", "G", "Antenna", "Element spacing in wavelengths."),
    _o("--rear-length-factor", "Rear length factor", "float", "1.0", "G", "Antenna",
       "Relative to 150/f meters (0.5 .. 1.5)."),
    _o("--front-length-factor", "Front length factor", "float", "1.0", "G", "Antenna",
       "Relative to 139/f meters (0.5 .. 1.5)."),
    _o("--element-diameter", "Element diameter (mm)", "float", "10.0", "GO", "Antenna", "Both elements."),
    _o("--rear-diameter", "Rear diameter (mm)", "float", "", "GO", "Antenna", "Default: element diameter."),
    _o("--front-diameter", "Front diameter (mm)", "float", "", "GO", "Antenna", "Default: element diameter."),
    _o("--rear-sections", "Rear sections", "text", "", "GO", "Antenna",
       "Telescoped tubes, ONE half, centre outward: LENGTH_MM:DIAMETER_MM,...,DIAMETER_MM  e.g. 300:12,200:10,8",
       ph="300:12,200:10,8"),
    _o("--front-sections", "Front sections", "text", "", "GO", "Antenna", "Same format as rear sections."),
    _o("--reflectors", "Reflectors (0-2)", "choice", "0", "GO", "Antenna",
       "Passive reflectors behind the rear element. Length and spacing are pre-calculated (minimum / nominal / maximum); "
       "optimize keeps them inside those limits.", choices=["0", "1", "2"]),
    _o("--directors", "Directors (0-5)", "choice", "0", "GO", "Antenna",
       "Passive directors in front of the front element. Length and spacing are pre-calculated (minimum / nominal / maximum); "
       "optimize keeps them inside those limits.", choices=["0", "1", "2", "3", "4", "5"]),
    _o("--reflector-lengths", "Reflector lengths (x resonant)", "text", "", "GO", "Antenna",
       "Factors of the resonant dipole length, F1,F2. Empty = pre-calculated nominal (1.05). In optimize: the starting point.",
       ph="1.05,1.05"),
    _o("--reflector-spacings", "Reflector spacings (wavelengths)", "text", "", "GO", "Antenna",
       "Each from the previous element (the first from the rear element). Empty = 0.15, 0.20.", ph="0.15,0.20"),
    _o("--director-lengths", "Director lengths (x resonant)", "text", "", "GO", "Antenna",
       "Factors of the resonant dipole length, F1..F5. Empty = pre-calculated nominal.", ph="0.96,0.95,0.94,0.93,0.92"),
    _o("--director-spacings", "Director spacings (wavelengths)", "text", "", "GO", "Antenna",
       "Each from the previous element (the first from the front element). Empty = 0.15, 0.20, 0.25, 0.30, 0.30.",
       ph="0.15,0.20,0.25,0.30,0.30"),
    _o("--parasitic-diameter", "Parasitic tube dia (mm)", "float", "", "GO", "Antenna",
       "Tube diameter of reflectors and directors. Default: the element diameter.", ph="(element diameter)"),
    _o("--min-gap-mm", "Min gap between elements (mm)", "float", "20.0", "GO", "Antenna",
       "Smallest surface-to-surface gap between neighbouring elements (clamps, insulators). With 0.05 lambda it sets the "
       "minimum spacing of reflectors and directors."),
    _o("--boom-overhang-mm", "Boom overhang (mm)", "float", "50.0", "GO", "Antenna",
       "Boom beyond the outer elements at each end. With reflectors/directors the boom is lengthened automatically."),
    _o("--boom-shape", "Boom shape", "choice", "square", "GO", "Antenna", "", choices=["square", "round"]),
    _o("--boom-size", "Boom size (mm)", "float", "25.0", "GO", "Antenna", "Side (square) or diameter (round)."),
    _o("--boom-length", "Boom length (mm)", "float", "700.0", "GO", "Antenna", ""),
    _o("--boom-isolated", "Model the boom (isolated)", "bool", "0", "GO", "Antenna",
       "Boom as a metallic conductor with no electrical connection, below the elements."),
    _o("--boom-clearance-mm", "Boom clearance (mm)", "float", "", "GO", "Antenna",
       "Gap between the element and boom surfaces of a modeled boom (default 10).", ph="10"),
    _o("--feed-gap-mm", "Feed gap (mm)", "float", "10.0", "GO", "Antenna", "Gap between the two halves of each element."),
    _o("--model-feed-gap", "Model the feed gap in NEC", "bool", "0", "GO", "Antenna",
       "Bridge wire of length g; needs g >= 4 tube radii and >= lambda/200."),
    _o("--end-cap-mm", "End cap (mm)", "float", "0.0", "GO", "Antenna", "Build sheet only."),
    _o("--tube-wall-mm", "Tube wall (mm)", "float", "1.0", "GO", "Antenna", "Build sheet only."),
    _o("--boom-material", "Boom material text", "text", "", "GO", "Antenna", "Build sheet bill of materials."),
    _o("--feed-style", "Feed style", "choice", "auto", "GO", "Antenna",
       "tmatch (default with tl/lc): T-match on both driven elements, line and feed at the bars. center: at the "
       "element centres (default with stepped sections or a modelled feed gap).", choices=["auto", "tmatch", "center"]),
    _o("--tap-mm", "T-match tap (mm)", "float", "120.0", "GO", "Antenna",
       "Distance from the element centre to each tap (optimize moves it)."),
    _o("--tap-offset-mm", "T-bar offset (mm)", "float", "50.0", "GO", "Antenna",
       "Distance from the element to the T-bar, toward the other element (closer bars lower the NEC accuracy)."),
    _o("--tap-wire-mm", "T-match wire (mm)", "float", "", "GO", "Antenna",
       "Diameter of the T-bar and its stubs (default: the element diameter).", ph="(element diameter)"),

    # ---------------------------------------------------------------- Feed network
    _o("--line-type", "Phase line type", "choice", "auto", "GO", "Feed",
       "openwire (default): balanced air line, crossed, VF 0.95. coax: straight, VF 0.66.",
       choices=["auto", "openwire", "coax"]),
    _o("--crossed", "Crossed line / network", "tri", "default", "GO", "Feed",
       "Default: crossed except coax (always straight).", choices=["default", "yes", "no"]),
    _o("--line-wire-diameter", "Open-wire diameter (mm)", "float", "", "GO", "Feed", "Default 3.", ph="3"),
    _o("--line-spacing", "Open-wire spacing (mm)", "float", "", "GO", "Feed", "Default 20. Z0 = 119.9*acosh(D/d).", ph="20"),
    _o("--line-z0", "Line Z0 (ohm, coax)", "float", "", "GO", "Feed", "Coax only, default 50.", ph="50"),
    _o("--line-vf", "Line velocity factor", "float", "", "GO", "Feed", "0 < VF <= 1.", ph="0.95 / 0.66"),
    _o("--lc-z0", "LC network Z0 (ohm)", "float", "50.0", "GO", "Feed", "Only with network lc."),
    _o("--line-deg", "Line electrical length (deg)", "float", "", "G", "Feed", "0..360. Default: synthesized line (or theory), never shorter than buildable."),
    _o("--phase-target-deg", "Phase target (deg)", "float", "", "G", "Feed",
       "Delay of the far element current relative to the fed one."),
    _o("--line-length", "Line physical length (m)", "float", "", "G", "Feed", "Overrides the derived length."),
    _o("--lc-l-nh", "LC series L (nH)", "float", "", "G", "Feed", "Requires LC C."),
    _o("--lc-c-pf", "LC shunt C (pF)", "float", "", "G", "Feed", "Requires LC L."),
    _o("--rear-ratio", "Rear amplitude ratio", "float", "1.0", "G", "Feed", "Ideal network only (1.0 = cardioid)."),
    _o("--ideal-drive", "Ideal drive", "choice", "current", "GO", "Feed",
       "Ideal network: prescribe the element currents (default) or the source voltages.",
       choices=["current", "voltage"]),
    _o("--rear-phase", "Rear phase (deg)", "float", "", "G", "Feed", "Ideal / lc. Default 180 - 360*spacing."),
    _o("--feedline-length-m", "CM feed-line length (m)", "float", "0.0", "GO", "Feed",
       "Common-mode test wire for the coax outer conductor (0 = off)."),
    _o("--feedline-diameter", "CM feed-line dia (mm)", "float", "5.0", "GO", "Feed", ""),
    _o("--cm-sensitivity", "Common-mode sensitivity", "bool", "0", "GO", "Feed", "Needs nec2c."),
    _o("--line-allowance-mm", "Line allowance (mm)", "float", "", "GO", "Feed",
       "Extra line length for connections and the crossing (default 20 open-wire, 40 coax); a line shorter than the "
       "feed-point distance plus this cannot be built.", ph="20 / 40"),
    _o("--max-line-deg", "Max line length (deg)", "float", "", "GO", "Feed",
       "Longest line generate picks and optimize scans. Default: 1.5 x the straight run (or theory + largest offset).",
       ph="(automatic)"),
    _o("--synthesize-line", "Synthesize the line", "tri", "default", "GO", "Feed",
       "Line that maximizes the objective on the superposed probe patterns (two nec2c probe runs); default on.",
       choices=["default", "yes", "no"]),
    _o("--series-c-pf", "Series feed C (pF)", "text", "", "GO", "Feed",
       "Series capacitor between the feed line and the fed port, in pF, or auto (cancels X_in at the frequency).",
       ph="auto"),
    _o("--series-c-q", "Series C quality factor", "float", "", "GO", "Feed",
       "Q of the series feed capacitor (ESR = |X|/Q added to Zin); default 500, 0 = lossless.", ph="500"),
    _o("--line-loss-db-per-m", "Line loss (dB/m)", "float", "0.0", "GO", "Feed",
       "Build sheet: matched loss of the phasing line; reported as an upper bound (0 = not reported)."),

    # ---------------------------------------------------------------- Ground
    _o("--ground", "Ground", "choice", "free", "GO", "Ground", "free / perfect / average / good.",
       choices=["free", "perfect", "average", "good"]),
    _o("--height-m", "Height (m)", "float", "", "GO", "Ground", "Required over ground."),
    _o("--height-lambda", "Height (wavelengths)", "float", "", "GO", "Ground", "Alternative to height in m."),
    _o("--orientation", "Orientation", "choice", "horizontal", "GO", "Ground", "",
       choices=["horizontal", "vertical"]),
    _o("--ground-eps", "Ground eps_r", "float", "", "GO", "Ground", "Overrides the preset."),
    _o("--ground-sigma", "Ground sigma (S/m)", "float", "", "GO", "Ground", "Overrides the preset."),
    _o("--mast-length-m", "Mast length (m)", "float", "0.0", "GO", "Ground",
       "Horizontal: below the boom centre. Vertical: side mount, parallel to the elements (0 = off)."),
    _o("--mast-diameter-mm", "Mast diameter (mm)", "float", "40.0", "GO", "Ground", ""),
    _o("--mast-gap-mm", "Mast gap (mm)", "float", "20.0", "GO", "Ground", ""),
    _o("--mast-offset-mm", "Mast side offset (mm)", "float", "0.0", "GO", "Ground",
       "Horizontal: sideways offset of the mast from the boom axis (side clamp). Vertical: extra distance behind the boom."),
    _o("--mast-sensitivity", "Mast sensitivity (m)", "text", "", "GO", "Ground",
       "Comma list of mast lengths, e.g. 0.5,1,2. Needs nec2c.", ph="0.5,1,2"),
    _o("--height-sweep", "Height sweep (m)", "text", "", "G", "Ground",
       "MIN,MAX,N over ground: take-off, gain, F/B and Zin at N heights. Needs nec2c.", ph="4,12,5"),
    _o("--robustness", "Robustness table", "bool", "0", "GO", "Ground",
       "Effect of a metallic boom, a mast that breaks the symmetry and an unchoked coax shield, each alone. Needs nec2c."),
    _o("--check-convergence", "Segmentation convergence", "tri", "default", "GO", "Ground",
       "Re-simulate with 2n+1 segments and report the change of gain, F/B and Zin (the Zin uncertainty). Default: on "
       "in optimize, off in generate. Needs nec2c.", choices=["default", "yes", "no"]),

    # ---------------------------------------------------------------- Generate (sweep)
    _o("--fsweep", "Frequency sweep", "text", "", "G", "Generate", "FMIN,FMAX,STEP in MHz (gives Zin(f)).",
       ph="140,150,0.5"),
    _o("--fsweep-pattern", "Pattern vs frequency", "bool", "0", "G", "Generate",
       "With the sweep: gain / F/B / Zin vs frequency and CSV (needs Run nec2c)."),
    _o("--fsweep-csv", "Sweep CSV", "file_save", "", "G", "Generate", "", ph="(automatic)"),

    # ---------------------------------------------------------------- Optimize
    _o("--band", "Band FMIN,FMAX,N", "text", "", "O", "Optimize", "Score every candidate over the band (worst case).",
       ph="144,146,5"),
    _o("--refine", "Refine rounds", "int", "", "O", "Optimize",
       "Local search rounds after the grid (0 = grid only). Default 6 with tl/lc, 8 with reflectors/directors.",
       ph="6"),
    _o("--nm-evals", "Nelder-Mead runs", "int", "40", "O", "Optimize",
       "Budget of the Nelder-Mead polish after the local search (0 = off)."),
    _o("--pretune", "Pre-tune front length", "tri", "default", "O", "Optimize",
       "Tune the front length for X_in ~ 0 at the NEC port (without the series capacitor) before the grid (default "
       "on).", choices=["default", "yes", "no"]),
    _o("--fb-metric", "F/B metric", "choice", "sector", "O", "Optimize",
       "F/B used by the score: point, sector (ring) or cone (any elevation).", choices=["point", "sector", "cone"]),
    _o("--jobs", "Parallel jobs", "int", "", "O", "Optimize", "Default: number of CPUs."),
    _o("--csv", "Scan CSV", "file_save", "hb9cv_scan.csv", "O", "Optimize", ""),
    _o("--parasitic-length-offsets", "Parasitic length offsets", "text", "0", "O", "Optimize",
       "Grid offsets added to EVERY reflector/director length factor, e.g. -0.02,0,0.02. Clamped to the pre-calculated limits.",
       ph="-0.02,0,0.02"),
    _o("--parasitic-spacing-scales", "Parasitic spacing scales", "text", "1", "O", "Optimize",
       "Grid multipliers of EVERY reflector/director spacing, e.g. 0.9,1,1.1. Clamped to the pre-calculated limits.",
       ph="0.9,1,1.1"),
    _o("--spacing-values", "Spacing values", "text", "0.08,0.125,0.17", "O", "Optimize", "Wavelengths."),
    _o("--rear-length-values", "Rear length factors", "text", "0.98,1.0,1.02", "O", "Optimize", ""),
    _o("--front-length-values", "Front length factors", "text", "0.98,1.0,1.02", "O", "Optimize",
       "Re-centred on the pre-tuned front length."),
    _o("--line-offset-values", "Line offsets (deg)", "text", "-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30", "O",
       "Optimize", "Offsets from the centre of the line scan: the synthesized or theoretical line (tl/lc)."),
    _o("--line-length-values", "Line lengths (m)", "text", "", "O", "Optimize", "Replaces the offsets."),
    _o("--z0-values", "Z0 values (ohm)", "text", "", "O", "Optimize", "e.g. 50,75,300", ph="50,75,300"),
    _o("--ratio-values", "Amplitude ratios", "text", "0.7,1.0,1.3", "O", "Optimize", "Ideal network only."),
    _o("--phase-offset-values", "Phase offsets (deg)", "text", "-60,-45,-30,-15,0,15,30,45,60", "O", "Optimize",
       "Ideal network."),
    _o("--phase-values", "Absolute phases (deg)", "text", "", "O", "Optimize", "Replaces the offsets."),
    _o("--match-score", "Score through the L-network", "tri", "default", "O", "Optimize",
       "Score every candidate after an L-network with component losses (default on with tl/lc); no = direct feed.",
       choices=["default", "yes", "no"]),
    _o("--match-q-l", "Match: inductor Q", "float", "150.0", "O", "Optimize", "Q of the scoring L-network inductor."),
    _o("--match-q-c", "Match: capacitor Q", "float", "500.0", "O", "Optimize", "Q of the scoring L-network capacitor."),
    _o("--zin-r-range", "Admissible R_in (min,max)", "text", "", "O", "Optimize",
       "ohm. Default 2,500 with the L-network score, 8,80 without it.", ph="2,500 / 8,80"),
    _o("--strict-zin", "Stop if no Zin match", "bool", "0", "O", "Optimize",
       "Stop with an error when no candidate meets the Zin window (default: write the closest one, flagged)."),
    _o("--zin-x-max", "Max |X_in| (ohm)", "float", "", "O", "Optimize",
       "Default 500 with the L-network score; otherwise 30 (10 with balun 4:1).", ph="500 / 30"),
    _o("--vswr-weight", "VSWR weight (dB)", "float", "", "O", "Optimize", "Default 3 with tl/lc, else 0.", ph="3"),
    _o("--realized-gain", "Realized gain", "tri", "default", "O", "Optimize", "", choices=["default", "yes", "no"]),
    _o("--w-gain", "Weight: gain", "float", "1.0", "O", "Optimize", "Balance objective."),
    _o("--w-fb", "Weight: F/B", "float", "0.5", "O", "Optimize", "Balance objective."),
    _o("--rear-sector", "Rear sector (deg)", "text", "150,210", "O", "Optimize", "start,end with 90 < start < 180 < end < 270"),
    _o("--refine-spacing", "Refine spacing", "tri", "default", "O", "Optimize", "", choices=["default", "yes", "no"]),
    _o("--refine-diameter", "Refine diameter", "tri", "default", "O", "Optimize", "", choices=["default", "yes", "no"]),
    _o("--diameter-range", "Diameter range (mm)", "text", "6,16", "O", "Optimize", "min,max"),
    _o("--diameter-values", "Tube diameters (mm)", "text", "6,8,10,12,14,16", "O", "Optimize",
       "Purchasable diameters the refinement steps through; 'any' = continuous."),
    _o("--tap-values", "T-match taps (mm)", "text", "", "O", "Optimize",
       "Tap distances scanned by the grid (default: the T-match tap).", ph="80,120,160"),
    _o("--tolerance-mc", "Monte-Carlo samples", "int", "0", "O", "Optimize", "0 = off (use >= 100)."),
    _o("--tol-dist", "MC: distribution", "choice", "uniform", "O", "Optimize",
       "uniform within +-tolerance, or normal (tolerance = 2 sigma).", choices=["uniform", "normal"]),
    _o("--tol-top-k", "MC: top-K candidates", "int", "5", "O", "Optimize", ""),
    _o("--tol-length-mm", "MC: length tol (mm)", "float", "2.0", "O", "Optimize", ""),
    _o("--tol-spacing-mm", "MC: spacing tol (mm)", "float", "3.0", "O", "Optimize", ""),
    _o("--tol-line-deg", "MC: line tol (deg)", "float", "2.0", "O", "Optimize", ""),
    _o("--tol-seed", "MC: seed", "int", "12345", "O", "Optimize", ""),
    _o("--tol-tap-mm", "MC: tap tol (mm)", "float", "2.0", "O", "Optimize", "T-match tap position, each element."),
    _o("--tol-bar-mm", "MC: T-bar tol (mm)", "float", "1.0", "O", "Optimize", "T-bar offset, each element."),
    _o("--tol-vf-pct", "MC: line VF tol (%)", "float", "2.0", "O", "Optimize", "Phasing-line velocity factor."),
    _o("--tol-z0-pct", "MC: line Z0 tol (%)", "float", "3.0", "O", "Optimize", "Phasing-line or LC network Z0."),
    _o("--tol-c-pct", "MC: series C tol (%)", "float", "2.0", "O", "Optimize", "Series feed capacitor setting."),
    _o("--tol-retune", "MC: re-tune per sample", "bool", "0", "O", "Optimize",
       "Re-tune the series C (auto) and re-design the matching network for every sample."),
    _o("--loss-weight", "Network loss weight", "float", "1.0", "O", "Optimize",
       "front-back with the L-network score: dB of score per dB of network loss (0 = pure F/B)."),

    # ---------------------------------------------------------------- Match / Sweep
    _o("--nec-out", "NEC .out file", "file_open", "", "MS", "Match",
       "nec2c output from which Zin is read (single EX). Sweep needs a frequency sweep inside.", ph="hb9cv_145MHz_tl_isolated.out"),
    _o("--r", "Load R (ohm)", "float", "", "MS", "Match", "Manual impedance (needs X)."),
    _o("--x", "Load X (ohm)", "float", "", "MS", "Match", "Manual impedance (needs R)."),
    _o("--z0", "Reference Z0 (ohm)", "float", "50.0", "MS", "Match", "Unbalanced source impedance."),
    _o("--q-l", "Inductor Q", "float", "", "MS", "Match", "ESR = X/Q (default ideal)."),
    _o("--q-c", "Capacitor Q", "float", "", "MS", "Match", "ESR = X/Q (default ideal)."),
    _o("--balanced", "Also print balanced version", "bool", "0", "M", "Match", ""),
    _o("--lnet-solution", "L-network solution", "choice", "plus", "MS", "Match",
       "plus / minus root, lowpass (series L, shunt C), highpass (series C, shunt L), or best (lower worst VSWR "
       "over the sweep, both topologies).", choices=["plus", "minus", "lowpass", "highpass", "best"]),
    _o("--esl-nh", "Capacitor ESL per lead (nH)", "float", "2.0", "MS", "Match", "Self-resonance check."),
    _o("--coil-cp-pf", "Coil stray C (pF)", "float", "0.7", "MS", "Match", "Self-resonance check."),
    _o("--design-freq", "Design frequency (MHz)", "float", "145.0", "S", "Match", "Sweep: frequency of the network."),
    _o("--assume-constant-z", "Assume constant Z", "bool", "0", "S", "Match", "Use R/X and fmin/fmax/step."),
    _o("--fmin", "Fmin (MHz)", "float", "140.0", "S", "Match", "Only with constant Z."),
    _o("--fmax", "Fmax (MHz)", "float", "150.0", "S", "Match", "Only with constant Z."),
    _o("--step", "Step (MHz)", "float", "0.25", "S", "Match", "Only with constant Z."),
    _o("--commercial", "Commercial values", "text", "", "S", "Match",
       "SERIES,SHUNT in nH/pF; several pairs separated by ';'. Default: nearest E24.", ph="12,3.3;10,3.9"),
    _o("--sweep-csv", "Sweep CSV", "file_save", "lnetwork_sweep.csv", "S", "Match", ""),
    _o("--tol-l", "Inductor tolerance (%)", "float", "0.0", "S", "Match", "0 = off."),
    _o("--tol-c", "Capacitor tolerance (%)", "float", "0.0", "S", "Match", "0 = off."),
]
OPT_BY_FLAG = {o.flag: o for o in OPTS}
GROUP_ORDER = ["Main", "Antenna", "Feed", "Ground", "Generate", "Optimize", "Match"]
MODES = ["generate", "optimize", "match", "sweep"]

# --------------------------------------------------------------------------------------
# --load-default-example : the most common 2 m HB9CV (146 MHz) found in amateur-radio builds.
#   * lambda at 146 MHz = 2053 mm  ->  rear element ~1027 mm (150/f), front ~952 mm (139/f)
#   * spacing 0.125 lambda (45 deg, ~257 mm), horizontal polarisation, free space
#   * front element fed, crossed open-wire phase line (3 mm wire, 20 mm spacing, VF 0.95),
#     1:1 current balun, 10 mm aluminium tubes (6061), 25 mm square boom
# The example starts in OPTIMIZE mode: it searches the geometry and the phase line for the best
# forward gain while the antenna is matched to a real 50 ohm line with no matching network
# (direct connection: no L-network score; target Z 50 ohm, 1:1 balun, realized gain, heavy VSWR penalty,
# R_in / X_in limits, worst case over 145-147 MHz).  Run it, then use the generate mode on the result if you want the plain model.
# Every flag listed here is preloaded in the GUI AND always written to the command line,
# even when its value equals the program default (see App.force_flags).
# --------------------------------------------------------------------------------------
EXAMPLE_MODE = "optimize"
EXAMPLE_VALUES = {
    "--freq": "146",
    "--network": "tl",
    "--feed-element": "front",
    "--mount": "isolated",
    "--balun-ratio": "1:1",
    "--segments": "41",
    "--conductivity": "2.5e7",
    "--spacing-factor": "0.125",
    "--rear-length-factor": "1.0",
    "--front-length-factor": "1.0",
    "--element-diameter": "10",
    "--boom-shape": "square",
    "--boom-size": "25",
    "--boom-length": "700",
    "--feed-gap-mm": "10",
    "--line-type": "openwire",
    "--line-wire-diameter": "3",
    # (no --line-spacing: hb9cv.py derives it from each value of --z0-values)
    "--ground": "free",
    "--orientation": "horizontal",
    "--run": True,
    # ---- optimisation: best gain + direct match to a real 50 ohm line
    "--objective": "gain",
    "--target-impedance": "50",
    "--realized-gain": "yes",
    "--vswr-weight": "10",
    "--match-score": "no",
    "--zin-r-range": "30,75",
    "--zin-x-max": "10",
    "--band": "145,147,3",
    "--spacing-values": "0.10,0.125,0.15",
    "--rear-length-values": "0.97,1.0,1.03",
    "--front-length-values": "0.97,1.0,1.03",
    "--line-offset-values": "-20,-10,0,10,20",
    "--z0-values": "200,250,310",
    "--refine": "6",
    "--refine-spacing": "yes",
    "--build-sheet": "hb9cv_146MHz_build.md",
    # match / sweep tabs: same antenna, so the L-network is designed at 146 MHz too
    "--z0": "50",
    "--design-freq": "146",
    "--fmin": "144",
    "--fmax": "148",
}


# ======================================================================================
# hb9cv.py loader
# ======================================================================================

def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("hb9cv_core", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hb9cv_core"] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================================
# NEC deck / output helpers (no GUI code below this line until the widgets)
# ======================================================================================

def parse_nec_geometry(text: str) -> dict:
    wires, ex, tl, nt, ground = [], [], None, None, None
    for line in text.splitlines():
        s = line.split()
        if not s:
            continue
        c = s[0].upper()
        try:
            if c == "GW" and len(s) >= 10:
                wires.append({"tag": int(s[1]), "nseg": int(s[2]),
                              "p1": np.array([float(v) for v in s[3:6]]),
                              "p2": np.array([float(v) for v in s[6:9]]),
                              "rad": float(s[9])})
            elif c == "EX" and len(s) >= 7:
                ex.append({"tag": int(s[2]), "v": complex(float(s[5]), float(s[6]))})
            elif c == "TL" and len(s) >= 7:
                tl = {"t1": int(s[1]), "t2": int(s[3]), "z0": float(s[5]), "length": float(s[6])}
            elif c == "NT" and len(s) >= 5:
                nt = {"t1": int(s[1]), "t2": int(s[3])}
            elif c == "GN":
                ground = s
        except ValueError:
            continue
    return {"wires": wires, "ex": ex, "tl": tl, "nt": nt, "ground": ground}


def _tag_extent(wires, tag):
    pts = np.array([p for w in wires if w["tag"] == tag for p in (w["p1"], w["p2"])])
    return pts.min(axis=0), pts.max(axis=0)


def analyse_geometry(geom: dict) -> dict:
    wires = geom["wires"]
    tags = sorted({w["tag"] for w in wires})
    if 1 not in tags:
        raise ValueError(tr("err.no_tag1"))
    front_tag = 2 if 2 in tags else None
    if front_tag is None:
        raise ValueError(tr("err.no_front"))
    lo1, hi1 = _tag_extent(wires, 1)
    axis = 1 if (hi1[1] - lo1[1]) >= (hi1[2] - lo1[2]) else 2   # element axis: y (horizontal) or z (vertical)
    elems = {}
    for name, tag in (("rear", 1), ("front", front_tag)):
        lo, hi = _tag_extent(wires, tag)
        diam = sorted({round(w["rad"] * 2000.0, 2) for w in wires if w["tag"] == tag}, reverse=True)
        elems[name] = {"tag": tag, "x": 0.5 * (lo[0] + hi[0]), "lo": lo[axis], "hi": hi[axis],
                       "len": hi[axis] - lo[axis], "diam": diam, "center": 0.5 * (lo[axis] + hi[axis])}
    boom = None
    bw = [w for w in wires if w["tag"] in (5, 6, 7)]
    if bw:
        xs = [v for w in bw for v in (w["p1"][0], w["p2"][0])]
        boom = {"x0": min(xs), "x1": max(xs), "length": max(xs) - min(xs)}
    pars = []
    for tag in tags:
        if 20 < tag < 30 or 30 < tag < 40:
            lo, hi = _tag_extent(wires, tag)
            diam = sorted({round(w["rad"] * 2000.0, 2) for w in wires if w["tag"] == tag}, reverse=True)
            role = "reflector" if tag < 30 else "director"
            pars.append({"tag": tag, "role": role, "name": f"{'R' if role == 'reflector' else 'D'}{tag % 10}",
                         "x": 0.5 * (lo[0] + hi[0]), "lo": lo[axis], "hi": hi[axis], "len": hi[axis] - lo[axis],
                         "diam": diam, "center": 0.5 * (lo[axis] + hi[axis])})
    pars.sort(key=lambda e: e["x"])
    # T-match (tags 3 rear, 4 front): the bar is the piece at constant x, the stubs join it to the element.
    tmatch = {}
    for tag, name in ((3, "rear"), (4, "front")):
        bar = [w for w in wires if w["tag"] == tag and abs(w["p1"][0] - w["p2"][0]) < 1e-9]
        if not bar:
            continue
        vals = [p[axis] for w in bar for p in (w["p1"], w["p2"])]
        tmatch[tag] = {"element": name, "x": bar[0]["p1"][0], "center": 0.5 * (min(vals) + max(vals)),
                       "tap": 0.5 * (max(vals) - min(vals)), "offset": abs(bar[0]["p1"][0] - elems[name]["x"])}
    # Height of the antenna, as hb9cv.py defines it: element centres (horizontal) or the lowest tip of the
    # elements (vertical).
    el_pts = [p for w in wires if w["tag"] in (1, front_tag) or 20 < w["tag"] < 40 for p in (w["p1"], w["p2"])]
    if axis == 1:
        z1 = [p[2] for w in wires if w["tag"] == 1 for p in (w["p1"], w["p2"])]
        height = float(sum(z1) / len(z1))
    else:
        height = float(min(p[2] for p in el_pts))
    return {"axis": axis, "elems": elems, "boom": boom, "front_tag": front_tag, "pars": pars, "tmatch": tmatch,
            "height": height,
            "has_ground": geom["ground"] is not None,
            "ground_desc": (" ".join(geom["ground"][:2]) if geom["ground"] else "")}


def wire_class(tag, front_tag):
    if tag == 1:
        return "rear"
    if tag == front_tag:
        return "front"
    if tag in (5, 6, 7):
        return "boom"
    if 20 < tag < 30:
        return "reflector"
    if 30 < tag < 40:
        return "director"
    if tag == 12:
        return "mast"
    if tag in (3, 4) or 8 <= tag <= 11:      # T-match bars and stubs, common-mode feed-line wires
        return "feed"
    return "other"


WIRE_STYLE = {"rear": (COL_REAR, 3.0), "front": (COL_FRONT, 3.0), "boom": (COL_BOOM, 5.0),
              "mast": (COL_OTHER, 6.0), "feed": (COL_LINE, 1.5), "other": (COL_OTHER, 2.0),
              "reflector": (COL_REFL, 3.0), "director": (COL_DIR, 3.0)}


def elevation_cut(core, text: str, over_ground: bool):
    """Elevation plane through the boom from the 1 degree cuts at phi=0 and phi=180 of a nec2c output text.

    Returns (psi, gain_dB): psi from -180 to 180 degrees in 1 degree steps, measured from the forward horizon
    (+x) through the zenith (psi = 90) to the back horizon (psi = 180 = -180) and below the horizon for negative
    psi; NaN where there is no radiation (over ground: below the horizon)."""
    secs = core.parse_pattern_sections(text)
    pts = {}
    for sec in secs:
        pts.update(sec["points"])
    psi = np.arange(-180.0, 181.0, 1.0)
    el = np.full(psi.shape, np.nan)
    for k, ps in enumerate(psi):
        if abs(ps) <= 90.0:
            key = (round(90.0 - ps, 2), 0.0)
        elif ps > 90.0:
            key = (round(ps - 90.0, 2), 180.0)
        else:
            key = (round(270.0 - abs(ps), 2), 180.0)
        if over_ground and key[0] > 90.0 + 1e-6:
            continue
        v = pts.get(key)
        if v is not None and v[2] > -900:
            el[k] = v[2]
    return psi, el


def beamwidth_lin(x, y, drop=3.0, circular=False):
    """Width between the -drop dB points around the maximum of y(x) (x ascending, 1 value per step), interpolated
    in linear power so that a deep null next to the beam does not pull the edge in. circular=True wraps around
    (x spans a full turn; a repeated end point is ignored). None when an edge is not found (a NaN or the end of a
    non-circular range comes first)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if circular and len(x) > 1 and abs((x[-1] - x[0]) - 360.0) < 1e-6:
        x, y = x[:-1], y[:-1]
    ok = np.isfinite(y)
    if not ok.any():
        return None
    n = len(y)
    i = int(np.argmax(np.where(ok, y, -np.inf)))
    p = np.where(ok, 10.0 ** (y / 10.0), np.nan)
    thr = p[i] * 10.0 ** (-drop / 10.0)

    def walk(step):
        j, dist = i, 0.0
        for _ in range(n - 1):
            k = j + step
            if circular:
                k %= n
            elif k < 0 or k >= n:
                return None
            if not ok[k]:
                return None
            dx = abs(x[k] - x[j]) if not circular else ((x[k] - x[j]) * step) % 360.0
            if p[k] < thr:
                return dist + dx * (p[j] - thr) / (p[j] - p[k])
            dist += dx
            j = k
        return None

    a, b = walk(-1), walk(1)
    return None if a is None or b is None else a + b


def find_nec2c(custom: str = "", core=None):
    """Path of nec2c in the lookup order of hb9cv.find_nec2c: the GUI field (a file or a folder), the HB9CV_NEC2C
    environment variable, the PATH, the bin folder next to this program, then a nec2c.exe shipped next to it.
    With core (the loaded hb9cv.py) its own lookup is used."""
    custom = (custom or "").strip()
    if core is not None and hasattr(core, "find_nec2c"):
        return core.find_nec2c(types.SimpleNamespace(nec2c=custom or None))
    for cand in (custom, os.environ.get("HB9CV_NEC2C", "").strip()):
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.is_dir():
            for n in ("nec2c", "nec2c.exe"):
                if (p / n).is_file():
                    return str(p / n)
        elif p.is_file():
            return str(p)
        found = shutil.which(cand)
        if found:
            return found
    found = shutil.which("nec2c") or shutil.which("nec2c.exe")
    if found:
        return found
    names = ("bin/nec2c.exe", "nec2c.exe") if IS_WIN else ("bin/nec2c",)
    for p in (SCRIPT_DIR / n for n in names):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def deck_head(nec_path: Path, freq_hint: float):
    """Geometry, ground, loads and excitation of a deck, for a single-frequency re-simulation.

    Returns (head lines, frequency, network line, over_ground). head holds every card before the first FR card
    except NT cards; the frequency is the one of all the FR cards (every point of a stepped FR) closest to
    freq_hint; network line is the NT card in force for that frequency (each frequency group of an LC deck carries
    its own) or None."""
    head, groups, nt, found = [], [], None, False
    for line in nec_path.read_text(errors="replace").splitlines():
        card = line[:2].upper()
        if card == "NT":
            nt = line
            continue
        if card == "FR":
            found = True
            parts = line.split()
            try:
                n, f0 = int(parts[2]), float(parts[5])
                df = float(parts[6]) if len(parts) > 6 else 0.0
                groups += [(f0 + i * df, nt) for i in range(max(1, n))]
            except (ValueError, IndexError):
                pass
            continue
        if not found:
            head.append(line)
    if not found:
        raise ValueError(tr("err.no_fr"))
    if groups:
        freq, nt_line = min(groups, key=lambda g: abs(g[0] - freq_hint))
    else:
        freq, nt_line = freq_hint, nt
    over_ground = any(l[:2].upper() == "GN" for l in head)
    return head, freq, nt_line, over_ground


def _run_deck(core, exe, td: Path, name: str, lines) -> tuple[Path, str]:
    """Write a deck, run nec2c on it and return (output path, output text); RuntimeError on failure."""
    inp, out = td / f"{name}.nec", td / f"{name}.out"
    inp.write_text("\n".join(lines).encode("ascii", errors="replace").decode("ascii"), encoding="ascii")
    if hasattr(core, "run_nec2c"):
        r = core.run_nec2c(exe, inp, out, timeout=1200)
    else:
        r = subprocess.run([exe, "-i", inp.name, "-o", out.name], cwd=str(td), capture_output=True, text=True,
                           timeout=1200, **NO_WINDOW)
    if r.returncode or not out.exists():
        raise RuntimeError(tr("err.nec2c_failed", rc=r.returncode, msg=(r.stderr or r.stdout).strip()[:300]))
    text = out.read_text(errors="replace")
    fatal, _ = core.scan_nec_messages(text)
    if fatal:
        raise RuntimeError(tr("err.nec2c_problem", msg=fatal[0]))
    return out, text


def sphere_avg_tolerance(step: float) -> float:
    """Expected error of the sphere-averaged gain of a step-degree grid (trapezoid in theta, pole rows included):
    about 0.01 up to 3 degrees, 0.03 at 5, 0.08 at 10 and 0.15 at 15 degrees."""
    return float(np.interp(step, [1.0, 3.0, 5.0, 10.0, 15.0], [0.01, 0.01, 0.03, 0.08, 0.15]))


def sphere_average(an: dict):
    """Average power gain over the full sphere = radiation efficiency of the model (free space only)."""
    if not an["full_sphere"]:
        return None
    th, ph, g = np.radians(an["thetas"]), np.asarray(an["phis"]), an["g"]
    if len(ph) > 1 and ph[-1] >= 360.0 - 1e-6 and abs(ph[0]) < 1e-6:
        ph, g = ph[:-1], g[:, :-1]                    # 360 is the same meridian as 0
    trap = getattr(np, "trapezoid", None) or np.trapz
    lin = np.nan_to_num(10.0 ** (g / 10.0), nan=0.0)
    dphi = 2.0 * np.pi / len(ph)
    return float(trap((lin * np.sin(th)[:, None]).sum(axis=1) * dphi, th) / (4.0 * np.pi))


def simulate_pattern(core, nec_path: Path, freq_hint: float, step: float, exe: str, target_z: float,
                     rear_sector=(150.0, 210.0), expected_eff=None) -> dict:
    """Pattern of a model at the frequency closest to freq_hint, with the procedure of hb9cv.py: over ground an
    elevation cut at phi=0 (1 degree) gives the take-off angle; a 1 degree azimuth ring at that angle (with the
    rear half-sphere grid) gives the gain, F/B, rear-sector and rear-cone F/B, HPBW and Zin; a full-sphere run on
    the step grid feeds the 3D plot, the elevation polar and the sphere average."""
    head, freq, nt, over_ground = deck_head(nec_path, freq_hint)
    base = head + ([nt] if nt else []) + [f"FR 0 1 0 0 {freq:.8f} 0", "PT -1"]
    rs, re_ = rear_sector
    notes = []
    if nt and sum(1 for l in nec_path.read_text(errors="replace").splitlines() if l[:2].upper() == "NT") > 1:
        notes.append(tr("warn.lc_nt", f=freq))
    with tempfile.TemporaryDirectory(prefix="hb9") as td_name:
        td = Path(td_name)
        th_cut = 90.0
        # Elevation plane through the boom at 1 degree: phi=0 and phi=180 (over ground theta 0..90 only).
        n_el = 91 if over_ground else 181
        _, el_text = _run_deck(core, exe, td, "el", base + [f"RP 0 {n_el} 1 1000 0 0 1 0",
                                                            f"RP 0 {n_el} 1 1000 0 180 1 0", "XQ", "EN", ""])
        secs = core.parse_pattern_sections(el_text)
        if not secs:
            raise RuntimeError(tr("err.no_rp"))
        if over_ground:
            th_cut = core.elevation_peak_theta(secs[0])
        ring_out, _ = _run_deck(core, exe, td, "ring", base + [
            f"RP 0 1 361 1000 {th_cut:g} 0 0 1", f"RP 0 {19 if over_ground else 37} 37 1000 0 90 5 5", "XQ", "EN", ""])
        m = core.parse_metrics(ring_out, theta=th_cut, rear_start=rs, rear_end=re_, ground_free=not over_ground)
        ring_text = ring_out.read_text(errors="replace")
        ring_pts, _n = core.ring_from_section(core.parse_pattern_sections(ring_text)[0], th_cut)
        zin = None
        try:
            zin = core.single_source_zin(ring_out)[0][1]
        except Exception:
            pass
        nth, nph = int(round(180.0 / step)) + 1, int(round(360.0 / step)) + 1
        _, sph_text = _run_deck(core, exe, td, "sphere", base + [f"RP 0 {nth} {nph} 1000 0 0 {step:g} {step:g}",
                                                                  "XQ", "EN", ""])
        sections = core.parse_pattern_sections(sph_text)
        if not sections:
            raise RuntimeError(tr("err.no_rp"))
    pts = sections[0]["points"]
    if over_ground:      # nec2c prints the directions below a perfect ground too; they are not radiation
        pts = {k: v for k, v in pts.items() if k[0] <= 90.0 + 1e-6}
    an = analyse_pattern(core, pts, freq, zin, target_z)
    if over_ground:
        an["full_sphere"] = False
    psi, el = elevation_cut(core, el_text, over_ground)
    an["psi"], an["el_g"] = psi, el
    an["hpbw_el"] = beamwidth_lin(psi, el, circular=not over_ground)
    az_phis = np.array(sorted(ring_pts))
    an.update(theta_cut=th_cut, takeoff=90.0 - th_cut, fwd=m["forward_gain"], rear=m["rear_gain"], fb=m["fb"],
              fb_robust=m["robust_fb"], fb_cone=m.get("cone_fb"), hpbw_az=m["hpbw"], gain_dbd=m["gain_dbd"],
              az_phis=az_phis, az_g=np.array([ring_pts[p] for p in az_phis]), rear_sector=(rs, re_),
              over_ground=over_ground, notes=notes)
    avg = sphere_average(an)
    an["sphere_avg"] = avg
    an["directivity"] = an["gmax"] - 10.0 * math.log10(avg) if avg and avg > 0 else None
    exp = 1.0 if expected_eff is None else expected_eff
    if hasattr(core, "agt_needs_warning"):
        warn_avg = core.agt_needs_warning(avg, exp, sphere_avg_tolerance(step))
    else:
        warn_avg = avg is not None and abs(avg - exp) > sphere_avg_tolerance(step)
    if warn_avg:
        an["notes"].append(tr("warn.sphere_avg", a=avg, e=exp))
    return an


def analyse_pattern(core, points: dict, freq: float, zin, target_z: float) -> dict:
    thetas = np.array(sorted({k[0] for k in points}))
    phis = np.array(sorted({k[1] for k in points}))
    ti = {t: i for i, t in enumerate(thetas)}
    pi = {p: j for j, p in enumerate(phis)}
    g = np.full((len(thetas), len(phis)), np.nan)
    for (t, p), v in points.items():
        if v[2] > -900:
            g[ti[t], pi[p]] = v[2]
    if not np.isfinite(g).any():
        raise RuntimeError(tr("err.no_valid_pts"))
    j0 = int(np.argmin(np.abs(phis - 0.0)))
    j180 = int(np.argmin(np.abs(phis - 180.0)))
    gmax = float(np.nanmax(g))
    ip, jp = np.unravel_index(int(np.nanargmax(g)), g.shape)

    col0 = g[:, j0]
    i_cut = int(np.nanargmax(col0)) if np.isfinite(col0).any() else int(ip)
    az_g = g[i_cut, :]
    fwd = float(g[i_cut, j0]) if np.isfinite(g[i_cut, j0]) else float(np.nanmax(az_g))
    rear = float(g[i_cut, j180]) if np.isfinite(g[i_cut, j180]) else float("nan")
    sector = [(phis[j], az_g[j]) for j in range(len(phis)) if 150 <= phis[j] <= 210 and np.isfinite(az_g[j])]
    robust = fwd - max(v for _, v in sector) if sector else float("nan")
    az_dict = {float(phis[j]): float(az_g[j]) for j in range(len(phis)) if np.isfinite(az_g[j])}
    try:
        hpbw_az = core.half_power_beamwidth(az_dict) if len(az_dict) > 3 else None
    except Exception:
        hpbw_az = None

    # elevation plane through the boom from the sphere grid: psi from -180..180 measured from +x through the
    # zenith (simulate_pattern replaces it with the 1 degree cuts)
    psi = np.arange(-180.0, 181.0, 1.0)
    el = np.full(psi.shape, np.nan)
    for k, ps in enumerate(psi):
        if abs(ps) <= 90:
            th, col = 90.0 - ps, g[:, j0]
        elif ps > 90:
            th, col = ps - 90.0, g[:, j180]
        else:
            th, col = 270.0 - abs(ps), g[:, j180]
        m = np.isfinite(col)
        if m.sum() >= 2:
            el[k] = np.interp(th, thetas[m], col[m], left=np.nan, right=np.nan)
    full = bool(np.isfinite(g[-1]).any())
    hpbw_el = beamwidth_lin(psi, el, circular=full)

    vswr = None
    if zin is not None and target_z:
        try:
            vswr = float(core.lnet_metrics(zin, target_z)[0])
        except Exception:
            vswr = None
    return {"freq": freq, "thetas": thetas, "phis": phis, "g": g, "gmax": gmax,
            "theta_peak": float(thetas[ip]), "phi_peak": float(phis[jp]),
            "i_cut": i_cut, "theta_cut": float(thetas[i_cut]), "takeoff": 90.0 - float(thetas[i_cut]),
            "az_g": az_g, "fwd": fwd, "rear": rear, "fb": fwd - rear if np.isfinite(rear) else float("nan"),
            "fb_robust": robust, "fb_cone": None, "gain_dbd": (fwd - 2.15) if full else None,
            "hpbw_az": hpbw_az, "hpbw_el": hpbw_el, "az_phis": phis, "rear_sector": (150.0, 210.0),
            "psi": psi, "el_g": el, "zin": zin, "vswr": vswr, "zref": target_z, "full_sphere": full,
            "over_ground": not full, "notes": [], "sphere_avg": None, "directivity": None}


def fmt_opt(v, fmt="{:.1f}", unit=""):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt.format(v) + unit


def extract_log_info(log: str) -> list[str]:
    prefixes = ("Feed at the", "Open-wire line", "Open-wire geometry", "Coaxial line", "Pi network",
                "Physical line length", "Ideal rear phase", "Ideal open-wire", "Ideal coaxial", "Ideal LC",
                "Ideal TL", "Element diameters", "Ground:")
    out, par_rows = [], []
    for line in log.splitlines():
        s = line.strip()
        if s.startswith("Parasitic elements (tip-to-tip"):
            par_rows = [s]                       # keep only the LAST table (the optimizer prints it twice)
        elif par_rows and (s.startswith(("reflector ", "director ", "element ", "pos. = "))):
            par_rows.append(s)
        elif s.startswith(prefixes) and s not in out:
            out.append(s)
    return out[:7] + par_rows


# ======================================================================================
# Plot drawing (pure matplotlib; testable without a display)
# ======================================================================================

def _table(rows, sep=" : "):
    """(label, value) pairs -> lines with the labels padded to the same width (monospace panels)."""
    w = max(len(l) for l, _ in rows)
    return [f"{l.ljust(w)}{sep}{v}" for l, v in rows]


def _dim(ax, p1, p2, text, tpos=None, rot=0.0, color="#222222", fs=8):
    ax.annotate("", xy=p1, xytext=p2, arrowprops=dict(arrowstyle="<->", color=color, lw=1.0,
                                                      shrinkA=0, shrinkB=0), zorder=5)
    mid = tpos if tpos is not None else ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    ax.text(mid[0], mid[1], text, ha="center", va="center", rotation=rot, fontsize=fs, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))


def _placeholder(fig, msg):
    fig.clear()
    fig.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="#667085", wrap=True)


def draw_antenna(fig: Figure, geom: dict, an: dict | None = None, info: list[str] | None = None,
                 freq: float | None = None):
    fig.clear()
    ai = analyse_geometry(geom)
    axis, E = ai["axis"], ai["elems"]
    rear, front = E["rear"], E["front"]
    pars = ai["pars"]
    mm = 1000.0
    gs = fig.add_gridspec(2, 2, width_ratios=[2.15, 1.0], height_ratios=[1.35, 1.0], hspace=0.22, wspace=0.12,
                          left=0.04, right=0.985, top=0.93, bottom=0.05)
    ax = fig.add_subplot(gs[:, 0])
    axs = fig.add_subplot(gs[0, 1])
    axt = fig.add_subplot(gs[1, 1])
    axt.axis("off")

    lmax = max([rear["len"], front["len"]] + [e["len"] for e in pars]) * mm
    off = max(25.0, 0.05 * lmax)
    xr, xf = rear["x"] * mm, front["x"] * mm
    spacing = abs(xf - xr)

    # --- elements
    for name, e, col in (("rear", rear, COL_REAR), ("front", front, COL_FRONT)):
        ax.plot([e["x"] * mm] * 2, [e["lo"] * mm, e["hi"] * mm], color=col, lw=4.0, solid_capstyle="round", zorder=4)
        ax.text(e["x"] * mm, e["hi"] * mm + 0.035 * lmax,
                f"{tr('el.' + name).upper()}" + ("\n" if pars else "  ") + "\u00d8 " + "/".join(f"{d:g}" for d in e["diam"]) + " mm",
                ha="center", va="bottom", fontsize=7 if pars else 8, color=col, fontweight="bold")
    for e in pars:      # passive reflectors / directors
        col = COL_REFL if e["role"] == "reflector" else COL_DIR
        ax.plot([e["x"] * mm] * 2, [e["lo"] * mm, e["hi"] * mm], color=col, lw=3.2, solid_capstyle="round", zorder=4)
        ax.text(e["x"] * mm, e["hi"] * mm + 0.035 * lmax, f"{e['name']}\n{e['len'] * mm:.0f} mm", ha="center",
                va="bottom", fontsize=7, color=col, fontweight="bold")
    # --- boom
    lo_all = min([rear["lo"], front["lo"]] + [e["lo"] for e in pars]) * mm
    if ai["boom"]:
        b = ai["boom"]
        ac = 0.5 * (rear["center"] + front["center"]) * mm
        ax.plot([b["x0"] * mm, b["x1"] * mm], [ac, ac], color=COL_BOOM, lw=7, alpha=0.55, solid_capstyle="butt",
                zorder=2)
        x_lo, x_hi = min(b["x0"] * mm, xr - 2.2 * off), max(b["x1"] * mm, xf + 2.2 * off)
    else:
        ac = 0.5 * (rear["center"] + front["center"]) * mm
        ax.plot([xr, xf], [ac, ac], color=COL_BOOM, lw=3, alpha=0.4, ls="--", zorder=2)
        x_lo, x_hi = xr - 2.2 * off, xf + 2.2 * off
    if pars:
        xp = [e["x"] * mm for e in pars]
        x_lo, x_hi = min(x_lo, min(xp) - 2.2 * off), max(x_hi, max(xp) + 2.2 * off)
        if not ai["boom"]:
            ax.plot([min(xp + [xr]), max(xp + [xf])], [ac, ac], color=COL_BOOM, lw=3, alpha=0.4, ls="--", zorder=2)

    # --- feed / phase line / sources
    d = 0.035 * lmax
    tl, nt = geom["tl"], geom["nt"]
    ex = geom["ex"]
    fed_tag = tl["t1"] if tl else (nt["t1"] if nt else None)
    centers = {rear["tag"]: (xr, rear["center"] * mm), front["tag"]: (xf, front["center"] * mm)}
    tm = ai.get("tmatch") or {}
    for w in geom["wires"]:          # T-match stubs and bars in the element plane
        if w["tag"] in tm:
            ax.plot([w["p1"][0] * mm, w["p2"][0] * mm], [w["p1"][axis] * mm, w["p2"][axis] * mm], color=COL_LINE,
                    lw=2.0, solid_capstyle="round", zorder=4)
    for tag, b in tm.items():
        centers[tag] = (b["x"] * mm, b["center"] * mm)
    # The phase network connects the two feed ports: the T-bar centres with a T-match, else the element centres.
    lx0 = centers[tl["t1"] if tl else nt["t1"]][0] if (tl or nt) and (tl or nt)["t1"] in centers else xr
    lx1 = centers[tl["t2"] if tl else nt["t2"]][0] if (tl or nt) and (tl or nt)["t2"] in centers else xf
    lx0, lx1 = min(lx0, lx1), max(lx0, lx1)
    if tl or nt:
        crossed = bool(tl and tl["z0"] < 0)
        if tl:
            ya, yb = ac + d, ac - d
            if crossed:
                ax.plot([lx0, lx1], [ya, yb], color=COL_LINE, lw=1.8, zorder=3)
                ax.plot([lx0, lx1], [yb, ya], color=COL_LINE, lw=1.8, zorder=3)
            else:
                ax.plot([lx0, lx1], [ya, ya], color=COL_LINE, lw=1.8, zorder=3)
                ax.plot([lx0, lx1], [yb, yb], color=COL_LINE, lw=1.8, zorder=3)
            label = tr("ant.phase_line", kind=tr("ant.crossed" if crossed else "ant.straight"),
                       z0=abs(tl["z0"]), length=tl["length"] * mm)
        else:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (lx0 + 0.2 * (lx1 - lx0), ac - 2 * d), 0.6 * (lx1 - lx0), 4 * d, fc="#e8f5e9", ec=COL_LINE, lw=1.5,
                zorder=3))
            label = tr("ant.lc_pi")
        if pars:
            y_lab = max([rear["hi"], front["hi"]] + [q["hi"] for q in pars]) * mm + 0.22 * lmax
            ax.annotate(label, xy=((xr + xf) / 2.0, ac + d), xytext=((xr + xf) / 2.0, y_lab), ha="center", va="bottom",
                        fontsize=6.5, color=COL_LINE, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=COL_LINE, lw=0.6, ls=":"),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        else:
            ax.text((xr + xf) / 2.0, ac + 3.2 * d, label, ha="center", va="bottom", fontsize=7.5, color=COL_LINE,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=6)
    for e in ex:
        tag = e["tag"] if e["tag"] in centers else (fed_tag if fed_tag in centers else front["tag"])
        cx, cy = centers[tag]
        ax.plot(cx, cy, "o", ms=11, mfc="#ffd54f", mec="black", zorder=7)
        txt = "EX" if len(ex) == 1 else f"EX {abs(e['v']):.2f}\u2220{math.degrees(math.atan2(e['v'].imag, e['v'].real)):.0f}\u00b0"
        ax.text(cx, cy - 2.4 * d, txt, ha="center", va="top", fontsize=8, fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    # --- dimensions
    for tag, b in tm.items():        # T-match: tap distance from the element centre and bar offset
        xbm, side = b["x"] * mm, (1 if b["element"] == "rear" else -1)
        cy, tap_mm = b["center"] * mm, b["tap"] * mm
        # below the boom axis, clear of the phase line and of its label above it
        _dim(ax, (xbm + side * 0.6 * off, cy - tap_mm), (xbm + side * 0.6 * off, cy), tr("ant.tap_dim", mm=tap_mm),
             tpos=(xbm + side * 0.6 * off, cy - 0.55 * tap_mm), rot=90, color=COL_LINE, fs=7)
        xe = (rear if b["element"] == "rear" else front)["x"] * mm
        yb_dim = cy - tap_mm - 0.05 * lmax
        _dim(ax, (min(xe, xbm), yb_dim), (max(xe, xbm), yb_dim), tr("ant.bar_dim", mm=b["offset"] * mm),
             tpos=(0.5 * (xe + xbm), yb_dim - 0.035 * lmax), color=COL_LINE, fs=7)
    for e, side in ((rear, -1), (front, +1)):
        xd = e["x"] * mm + side * 1.5 * off
        for yy in (e["lo"] * mm, e["hi"] * mm):
            ax.plot([e["x"] * mm, xd + side * 0.3 * off], [yy, yy], color="#888", lw=0.7, ls=":", zorder=1)
        _dim(ax, (xd, e["lo"] * mm), (xd, e["hi"] * mm), f"{e['len'] * mm:.1f} mm",
             tpos=(xd, e["hi"] * mm - 0.14 * e["len"] * mm), rot=90)
    y1 = lo_all - 0.10 * lmax
    chain = sorted([(xr, rear), (xf, front)] + [(e["x"] * mm, e) for e in pars], key=lambda t: t[0])
    for xx, e in chain:
        ax.plot([xx, xx], [e["lo"] * mm, y1 - 0.02 * lmax], color="#888", lw=0.7, ls=":", zorder=1)
    for k in range(len(chain) - 1):
        xa, xb = chain[k][0], chain[k + 1][0]
        if chain[k][1] is rear and chain[k + 1][1] is front:
            txt = (f"{spacing:.0f} mm" if pars else
                   tr("ant.spacing_dim", mm=spacing, lam=spacing / (299792458.0 / (an["freq"] * 1e6) * mm))
                   if an else tr("ant.spacing_mm", mm=spacing))
        else:
            txt = f"{xb - xa:.0f} mm"
        _dim(ax, (xa, y1), (xb, y1), txt, tpos=((xa + xb) / 2.0, y1 - 0.035 * lmax - (0.05 * lmax if k % 2 else 0.0)),
             fs=8 if len(chain) <= 4 else 7)
    y_min = y1 - (0.14 if len(chain) > 2 else 0.09) * lmax
    if ai["boom"]:
        b = ai["boom"]
        y2 = y1 - 0.14 * lmax
        _dim(ax, (b["x0"] * mm, y2), (b["x1"] * mm, y2), tr("ant.boom_dim", mm=b["length"] * mm),
             tpos=((b["x0"] + b["x1"]) * mm / 2.0, y2 - 0.035 * lmax))
        y_min = y2 - 0.09 * lmax
    y_top = max([rear["hi"], front["hi"]] + [e["hi"] for e in pars]) * mm
    ax.annotate("", xy=(x_hi - 0.2 * off, y_top + 0.17 * lmax), xytext=(x_hi - 2.6 * off, y_top + 0.17 * lmax),
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.6))
    ax.text(x_hi - 1.4 * off, y_top + 0.19 * lmax, tr("ant.beam"), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(x_lo - 0.5 * off, x_hi + 0.5 * off)
    ax.set_ylim(y_min, y_top + (0.40 if pars else 0.28) * lmax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(tr("ant.xlabel"), fontsize=8)
    ax.set_ylabel(tr("ant.ylabel", ax="y" if axis == 1 else "z"), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, ls=":", alpha=0.35)
    ax.set_title(tr("ant.title"), fontsize=10, fontweight="bold")
    if axis == 2 and ai["has_ground"]:
        ax.axhline(0, color=COL_GROUND, lw=3)
        ax.text(x_lo, 0, tr("ant.ground"), color=COL_GROUND, fontsize=8, va="bottom")

    # --- side / end view
    hx = 0 if axis == 1 else 1
    ff = ai["front_tag"]
    zs = []
    for w in geom["wires"]:
        kind = wire_class(w["tag"], ff)
        col, lw = WIRE_STYLE[kind]
        pa, pb = w["p1"], w["p2"]
        axs.plot([pa[hx] * mm, pb[hx] * mm], [pa[2] * mm, pb[2] * mm], color=col,
                 lw=max(1.0, lw * (0.7 if kind in ("rear", "front") else 0.8)), solid_capstyle="round",
                 alpha=0.9 if kind != "rear" else 0.6, zorder=3)
        if kind in ("rear", "front"):
            axs.plot(pa[hx] * mm, pa[2] * mm, "o", ms=4, color=col, zorder=4)
        zs += [pa[2] * mm, pb[2] * mm]
    zmin, zmax = min(zs), max(zs)
    hz = max(zmax - zmin, 1.0)
    if ai["has_ground"]:
        axs.axhline(0, color=COL_GROUND, lw=3, zorder=1)
        axs.text(0.02, 0.02, tr("ant.ground_desc", d=ai["ground_desc"]), transform=axs.transAxes, color=COL_GROUND, fontsize=7)
        ref = ai["height"] * mm
        xm = min(min(w["p1"][hx], w["p2"][hx]) for w in geom["wires"]) * mm
        _dim(axs, (xm - 0.05 * hz, 0), (xm - 0.05 * hz, ref), f"H = {ref:.0f} mm",
             tpos=(xm - 0.05 * hz, ref / 2.0), rot=90)
        zmin = min(zmin, 0.0)
    axs.set_aspect("equal", adjustable="datalim")
    axs.margins(0.2)
    axs.tick_params(labelsize=7)
    axs.grid(True, ls=":", alpha=0.35)
    axs.set_title(tr("ant.side_view") if axis == 1 else tr("ant.end_view"), fontsize=9, fontweight="bold")

    # --- text panel
    lam = (299792458.0 / (an["freq"] * 1e6)) if an else (299792458.0 / (freq * 1e6) if freq else None)
    lines = []
    if lam:
        f_mhz = an["freq"] if an else freq
        lines.append(f"f = {f_mhz:g} MHz     \u03bb = {lam * mm:.1f} mm")
    rows = [(tr("el.rear"), f"{rear['len'] * mm:7.1f} mm   \u00d8 " + "/".join(f"{x:g}" for x in rear["diam"])),
            (tr("el.front"), f"{front['len'] * mm:7.1f} mm   \u00d8 " + "/".join(f"{x:g}" for x in front["diam"])),
            (tr("ant.spacing"), f"{spacing:.1f} mm" + (f"   Boom: {ai['boom']['length'] * mm:.0f} mm" if ai["boom"] else ""))]
    for k, (xx, e) in enumerate(chain):      # reflectors / directors: length and distance to the neighbour toward the driven pair
        if e is rear or e is front:
            continue
        nb = chain[k + 1][0] if e["role"] == "reflector" else chain[k - 1][0]
        rows.append((e["name"], f"{e['len'] * mm:7.1f} mm   \u00d8 " + "/".join(f"{x:g}" for x in e["diam"])
                     + f"   \u0394 {abs(xx - nb):.0f} mm"))
    lines += _table(rows)
    if an:
        lines.append("")
        rows = [(tr("ant.gain"), (f"{an['fwd']:.2f} dBi = {an['gain_dbd']:.2f} dBd" if an.get("gain_dbd") is not None
                                  else f"{an['fwd']:.2f} {tr('tbl.incl_ground')}")),
                ("F/B", f"{fmt_opt(an['fb'], '{:.1f}', ' dB')}   {tr('ant.sector')}: {fmt_opt(an['fb_robust'], '{:.1f}', ' dB')}"),
                ("HPBW", f"az {fmt_opt(an['hpbw_az'], '{:.0f}', chr(176))}   el {fmt_opt(an['hpbw_el'], '{:.0f}', chr(176))}")]
        if an["zin"] is not None:
            z = an["zin"]
            rows.append(("Zin", f"{z.real:.1f}{z.imag:+.1f}j \u03a9   VSWR {fmt_opt(an['vswr'], '{:.2f}')}:1 "
                                f"(Z ref {an['zref']:g} \u03a9)"))
        lines += _table(rows)
    for t in (info or []):
        lines.append("")
        lines += textwrap.wrap(t, 52)
    axt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.6,
             transform=axt.transAxes, color="#222")
    fig.suptitle("")


def _polar_axes(ax, gmax):
    rmax = 5.0 * math.ceil(gmax / 5.0)
    rmin = rmax - 40
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(rmin, rmax)
    ticks = np.arange(rmin + 10, rmax + 0.1, 10)
    ax.set_rticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.5)
    return rmin, rmax


def draw_pattern_2d(fig: Figure, an: dict):
    fig.clear()
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28, left=0.06, right=0.97, top=0.93, bottom=0.08)
    gmax = an["gmax"]
    axA = fig.add_subplot(gs[0, 0], projection="polar")
    axE = fig.add_subplot(gs[0, 1], projection="polar")
    axC = fig.add_subplot(gs[1, 0])
    axT = fig.add_subplot(gs[1, 1])
    axT.axis("off")

    rmin, rmax = _polar_axes(axA, gmax)
    aph = np.asarray(an.get("az_phis", an["phis"]))
    phis = np.append(aph, 360.0) if aph[-1] < 360 else aph
    azg = np.append(an["az_g"], an["az_g"][0]) if len(phis) > len(an["az_g"]) else an["az_g"]
    azp = np.where(np.isfinite(azg), azg, rmin)
    axA.plot(np.radians(phis), np.clip(azp, rmin, rmax), color=COL_FRONT, lw=1.8)
    axA.fill(np.radians(phis), np.clip(azp, rmin, rmax), color=COL_FRONT, alpha=0.15)
    axA.set_title(tr("p2.az_title", el=an["takeoff"]), fontsize=9,
                  fontweight="bold", pad=14)

    rmin, rmax = _polar_axes(axE, gmax)
    elp = np.where(np.isfinite(an["el_g"]), an["el_g"], rmin)
    axE.plot(np.radians(an["psi"]), np.clip(elp, rmin, rmax), color=COL_REAR, lw=1.8)
    axE.fill(np.radians(an["psi"]), np.clip(elp, rmin, rmax), color=COL_REAR, alpha=0.15)
    axE.set_title(tr("p2.el_title"), fontsize=9, fontweight="bold", pad=14)
    if not an["full_sphere"]:
        axE.text(0.5, -0.12, tr("p2.ground_note"), transform=axE.transAxes,
                 ha="center", fontsize=7, color=COL_GROUND)

    # rectangular
    ph = np.where(aph > 180, aph - 360, aph)
    order = np.argsort(ph)
    axC.plot(ph[order], an["az_g"][order], color=COL_FRONT, lw=1.6, label=tr("p2.azimuth"))
    axC.plot(an["psi"], an["el_g"], color=COL_REAR, lw=1.6, label=tr("p2.elevation"))
    axC.axhline(gmax - 3, color="#888", ls="--", lw=0.8)
    axC.text(-178, gmax - 2.6, "-3 dB", fontsize=7, color="#666")
    axC.set_xlim(-180, 180)
    axC.set_ylim(gmax - 40, gmax + 1)
    axC.set_xticks(np.arange(-180, 181, 45))
    axC.set_xlabel(tr("p2.xlabel"), fontsize=8)
    axC.set_ylabel(tr("p2.ylabel"), fontsize=8)
    axC.tick_params(labelsize=7)
    axC.grid(True, ls=":", alpha=0.5)
    axC.legend(fontsize=7, loc="lower center", ncol=2)
    axC.set_title(tr("p2.rect_title"), fontsize=9, fontweight="bold")

    rows = [(tr("tbl.freq"), f"{an['freq']:g} MHz"),
            (tr("tbl.peak"), tr("tbl.peak_val", g=gmax, az=an["phi_peak"], el=90 - an["theta_peak"])),
            (tr("tbl.fwd"), (f"{an['fwd']:.2f} dBi = {an['gain_dbd']:.2f} dBd" if an.get("gain_dbd") is not None
                             else f"{an['fwd']:.2f} {tr('tbl.incl_ground')}")),
            (tr("tbl.takeoff"), f"{an['takeoff']:g}\u00b0"),
            (tr("tbl.fb_point"), fmt_opt(an["fb"], "{:.2f}", " dB")),
            (tr("tbl.fb_sector", a=an["rear_sector"][0], b=an["rear_sector"][1]),
             fmt_opt(an["fb_robust"], "{:.2f}", " dB")),
            (tr("tbl.fb_cone"), fmt_opt(an.get("fb_cone"), "{:.2f}", " dB")),
            (tr("tbl.hpbw_az"), fmt_opt(an["hpbw_az"], "{:.1f}", chr(176))),
            (tr("tbl.hpbw_el"), fmt_opt(an["hpbw_el"], "{:.1f}", chr(176)))]
    if an.get("sphere_avg") is not None:
        rows.append((tr("tbl.avg_gain"), f"{an['sphere_avg']:.3f}"))
        rows.append((tr("tbl.directivity"), fmt_opt(an.get("directivity"), "{:.2f}", " dBi")))
    if an["zin"] is not None:
        z = an["zin"]
        rows.append(("Zin", f"{z.real:.2f}{z.imag:+.2f}j \u03a9"))
        rows.append((tr("tbl.vswr_vs", z=an["zref"]), f"{fmt_opt(an['vswr'], '{:.2f}')}:1"))
    t = _table(rows, sep="  ")
    axT.text(0.02, 0.98, "\n".join(t), va="top", ha="left", family="monospace", fontsize=9,
             transform=axT.transAxes, bbox=dict(boxstyle="round", fc="#f6f8fb", ec="#d0d7e2"))


def draw_pattern_3d(fig: Figure, an: dict, geom: dict | None = None, linear: bool = False):
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    th, ph, g = an["thetas"], an["phis"], an["g"]
    if ph[-1] < 359.999:
        ph = np.append(ph, 360.0)
        g = np.concatenate([g, g[:, :1]], axis=1)
    rows = [i for i in range(len(th)) if np.isfinite(g[i]).any()]
    th, g = th[rows], g[rows]
    gmax = an["gmax"]
    floor = gmax - 40.0
    gg = np.where(np.isfinite(g), g, floor)
    if linear:      # radius proportional to the power gain
        r = np.clip(10.0 ** ((gg - gmax) / 10.0), 0.0, 1.0)
    else:           # radius linear in dB over the 40 dB range
        r = np.clip((gg - floor) / (gmax - floor), 0.0, 1.0)
    TH, PH = np.meshgrid(np.radians(th), np.radians(ph), indexing="ij")
    X, Y, Z = r * np.sin(TH) * np.cos(PH), r * np.sin(TH) * np.sin(PH), r * np.cos(TH)
    norm = mcolors.Normalize(vmin=floor, vmax=gmax)
    stride = max(1, int(math.ceil(math.sqrt(gg.size / 12000.0))))   # step 5 deg -> 1; very dense grids -> coarser
    ax.plot_surface(X, Y, Z, facecolors=cm.turbo(norm(gg)), rstride=stride, cstride=stride, linewidth=0,
                    antialiased=True, shade=False)
    upper_only = not an["full_sphere"]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0 if upper_only else -1, 1)
    ax.set_box_aspect((1, 1, 0.55 if upper_only else 1))
    ax.set_xlabel(tr("p3.xlabel"), fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.set_zlabel("z", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    # beam marker
    ip = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(g), g, -1e9))), g.shape)
    pk = (X[ip], Y[ip], Z[ip])
    ax.scatter([pk[0]], [pk[1]], [pk[2]], color="black", s=22, depthshade=False)
    ax.text(pk[0] * 1.12, pk[1] * 1.12, pk[2] * 1.12 + 0.04, f"{gmax:.1f} dBi", fontsize=8, fontweight="bold")
    ax.quiver(0, 0, 0, 1.25, 0, 0, color="#222", arrow_length_ratio=0.08, lw=1.2)
    ax.text(1.3, 0, 0, "+x", fontsize=8)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.turbo)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.04)
    cb.set_label(tr("p3.cbar_lin" if linear else "p3.cbar"), fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(tr("p3.title", f=an["freq"]), fontsize=10, fontweight="bold")
    ax.view_init(elev=24, azim=-58)

    if geom and geom["wires"]:   # small inset with the antenna wires
        ins = fig.add_axes([0.01, 0.03, 0.23, 0.26], projection="3d")
        ff = 2
        allp = np.array([p for w in geom["wires"] for p in (w["p1"], w["p2"])])
        c = allp.mean(axis=0)
        span = float(np.abs(allp - c).max()) or 1.0
        for w in geom["wires"]:
            if wire_class(w["tag"], ff) in ("rear", "front", "boom", "reflector", "director"):
                col = WIRE_STYLE[wire_class(w["tag"], ff)][0]
                a, b = (w["p1"] - c) / span, (w["p2"] - c) / span
                ins.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=col, lw=2)
        ins.set_xlim(-1, 1), ins.set_ylim(-1, 1), ins.set_zlim(-1, 1)
        ins.set_box_aspect((1, 1, 1))
        ins.set_xticks([]), ins.set_yticks([]), ins.set_zticks([])
        ins.set_title(tr("p3.inset"), fontsize=7, pad=0)
        ins.view_init(elev=24, azim=-58)


def draw_match(fig: Figure, d: dict):
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.35, left=0.07, right=0.97, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(tr("m.title"), fontsize=10, fontweight="bold")
    topo = d.get("topology", "A")
    ser = tr("m.series") + ("\n" + d["series_val"] if d.get("series_val") else "")
    shu = tr("m.shunt") + ("\n" + d["shunt_val"] if d.get("shunt_val") else "")

    def box(x, y, w, h, text, col):
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (x, y), w, h, fc="#eef3fb", ec=col, lw=1.6, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, zorder=4)

    y = 2.6
    ax.plot([0.9, 9.1], [y, y], color="#333", lw=1.4, zorder=1)
    ax.plot([0.9, 9.1], [0.5, 0.5], color="#333", lw=1.4, zorder=1)
    box(0.0, 0.9, 0.9, 3.0 - 0.9 - 0.2 + 0.0, "", "#333")
    ax.text(0.45, 2.4, tr("m.source", z=d.get("z0", 50)), ha="center", va="center", fontsize=8, zorder=5)
    box(9.1, 0.9, 0.9, 1.9, "", "#333")
    ax.text(9.55, 2.0, (tr("m.antenna") + "\n" + f"{d['zload_val'][0]:.1f}{d['zload_val'][1]:+.1f}j \u03a9"
                  if d.get("zload_val") else tr("m.antenna") + "\nZ"), ha="center", va="center", fontsize=7.5, zorder=5)
    # series element in the top rail, shunt element to the bottom rail
    if topo == "A":     # series next to the antenna, shunt on the Z0 side
        xs, xh = 6.1, 3.0
    else:               # shunt next to the antenna, series on the Z0 side
        xs, xh = 3.2, 6.0
    box(xs, y - 0.35, 1.9, 0.7, ser, "#c0392b")
    ax.plot([xh, xh], [y, 1.7], color="#333", lw=1.4, zorder=1)
    ax.plot([xh, xh], [0.5, 1.1], color="#333", lw=1.4, zorder=1)
    box(xh - 0.95, 1.1, 1.9, 0.6, shu, "#1f6fb2")
    ax.plot(xh, y, "o", color="#333", ms=4, zorder=5)
    ax.plot(xh, 0.5, "o", color="#333", ms=4, zorder=5)
    ax.text(5.0, 3.6, (tr("m.topology", t=d["topology"]) + d.get("topology_rest", "") if d.get("topology") else ""), ha="center", fontsize=8, color="#444")

    axv = fig.add_subplot(gs[1])
    rows = d.get("preview") or []
    if rows:
        f = [r[0] for r in rows]
        v = [r[1] for r in rows]
        axv.plot(f, v, "-o", ms=3, color=COL_FRONT, label=tr("m.vswr_label"))
        axv.axhline(2.0, color="#c0392b", ls="--", lw=0.9)
        axv.text(f[0], 2.03, "2:1", color="#c0392b", fontsize=7)
        axv.set_xlabel(tr("m.freq"), fontsize=8)
        axv.set_ylabel("VSWR", fontsize=8)
        axv.set_ylim(1, max(3.0, min(max(v) * 1.1, 10)))
        axv.grid(True, ls=":", alpha=0.5)
        axv.legend(fontsize=7)
        axv.set_title(tr("m.preview_title"), fontsize=9, fontweight="bold")
    else:
        axv.axis("off")
        txt = "\n".join(d.get("summary", []))
        axv.text(0.01, 0.98, txt, va="top", family="monospace", fontsize=8, transform=axv.transAxes)
    axv.tick_params(labelsize=7)


def draw_sweep(fig: Figure, rows: list[dict]):
    fig.clear()
    gs = fig.add_gridspec(2, 1, hspace=0.3, left=0.08, right=0.97, top=0.93, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    labels = []
    for r in rows:
        if r["network"] not in labels:
            labels.append(r["network"])
    palette = ["#1f6fb2", "#c0392b", "#2e7d32", "#8e44ad", "#d68910"]
    for i, lab in enumerate(labels):
        sel = [r for r in rows if r["network"] == lab]
        f = [float(r["frequency_MHz"]) for r in sel]
        ax1.plot(f, [float(r["VSWR"]) for r in sel], "-o", ms=3, color=palette[i % 5], label=lab)
    ax1.axhline(2.0, color="#888", ls="--", lw=0.9)
    ax1.set_ylabel("VSWR", fontsize=8)
    ax1.set_ylim(1, max(3.0, min(max(float(r["VSWR"]) for r in rows) * 1.1, 12)))
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend(fontsize=7)
    ax1.set_title(tr("s.title"), fontsize=10, fontweight="bold")
    first = [r for r in rows if r["network"] == labels[0]]
    f = [float(r["frequency_MHz"]) for r in first]
    ax2.plot(f, [float(r["Zload_real_ohm"]) for r in first], color="#1f6fb2", label=tr("s.rload"))
    ax2.plot(f, [float(r["Zload_imag_ohm"]) for r in first], color="#1f6fb2", ls="--", label=tr("s.xload"))
    ax2.plot(f, [float(r["Zin_real_ohm"]) for r in first], color="#c0392b", label=tr("s.rin", n=labels[0]))
    ax2.plot(f, [float(r["Zin_imag_ohm"]) for r in first], color="#c0392b", ls="--", label=tr("s.xin", n=labels[0]))
    ax2.set_xlabel(tr("m.freq"), fontsize=8)
    ax2.set_ylabel(tr("s.imp"), fontsize=8)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(fontsize=7, ncol=2)
    for a in (ax1, ax2):
        a.tick_params(labelsize=7)


# ======================================================================================
# Log parsing for match mode
# ======================================================================================

def match_from_json(js: dict, log: str) -> dict:
    """Matching result from the --json-out file of hb9cv.py --mode match (log only for the summary lines)."""
    d = {"summary": [l for l in log.splitlines() if l.strip()][:30], "preview": []}
    d["topology"] = js.get("topology", "")
    tt = js.get("topology_text", "")
    d["topology_rest"] = tt.split(":", 1)[1] if ":" in tt else ""
    d["series_val"], d["shunt_val"] = js.get("series", ""), js.get("shunt", "")
    z = js.get("zload")
    if isinstance(z, dict):
        d["zload_val"] = (float(z["r"]), float(z["x"]))
    d["preview"] = [(float(p["freq_mhz"]), float(p["vswr"])) for p in js.get("preview", [])]
    return d


def parse_match_log(log: str) -> dict:
    d = {"summary": [l for l in log.splitlines() if l.strip()][:30], "preview": []}
    m = re.search(r"Topology\s+([AB]):(.*)", log)
    if m:
        d["topology"], d["topology_rest"] = m.group(1), m.group(2)
    m = re.search(r"Series branch:\s*(.+)", log)
    d["series_val"] = m.group(1).strip() if m else ""
    m = re.search(r"Shunt branch:\s*(.+)", log)
    d["shunt_val"] = m.group(1).strip() if m else ""
    m = re.search(r"Impedance .*?:\s*([-\d.]+)\s*([+-][\d.]+)j", log)
    if m:
        d["zload_val"] = (float(m.group(1)), float(m.group(2)))
    in_prev = False
    for line in log.splitlines():
        if "BANDWIDTH PREVIEW" in line:
            in_prev = True
            continue
        if in_prev:
            mm_ = re.match(r"^\s+(\d+\.\d+)\s+(\d+\.\d+)\s", line)
            if mm_:
                d["preview"].append((float(mm_.group(1)), float(mm_.group(2))))
            elif d["preview"] and not line.strip():
                break
    return d



# ======================================================================================
# GUI widgets
# ======================================================================================

def _ui_font_family() -> str:
    try:
        import tkinter.font as tkfont
        return tkfont.nametofont("TkDefaultFont").actual("family")
    except Exception:
        return "Helvetica"


class ToolTip:
    """Hover tooltip made of raw tk widgets: sizes are in pixels and follow the UI scale."""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip, self.job = widget, text, None, None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")

    def _enter(self, _e=None):
        try:
            self.job = self.widget.after(450, self._show)
        except tk.TclError:
            self.job = None

    def _leave(self, _e=None):
        try:
            if self.job:
                self.widget.after_cancel(self.job)
        except tk.TclError:
            pass
        self.job = None
        if self.tip:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None

    def _show(self):
        if self.tip or not self.text:
            return
        s = UI["scale"]
        try:
            x = self.widget.winfo_rootx() + int(16 * s)
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + int(4 * s)
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            tk.Label(self.tip, text=self.text, justify="left", background="#fffbe6", foreground="#222",
                     relief="solid", borderwidth=1, wraplength=int(380 * s),
                     font=(_ui_font_family(), -max(10, int(round(12 * s))))
                     ).pack(ipadx=int(4 * s), ipady=int(2 * s))
        except tk.TclError:
            self.tip = None


class HelpBadge(tk.Canvas):
    """Small orange circle with a '?'.  Hovering it opens a popup with the explanation of the option
    (title + help text of at most HELP_MAX characters).  Sizes are in pixels and follow the UI scale.
    A plain tk.Canvas (two items) instead of a rounded CTkLabel: 105 of these are created on every
    build, and the customtkinter version was the most expensive single widget of the option rows."""

    COL, COL_HOT = "#f08c00", "#d9480f"

    def __init__(self, master, title: str, body_fn):
        s = UI["scale"]
        d = max(12, int(round(18 * s)))
        super().__init__(master, width=d, height=d, bg="white", highlightthickness=0, bd=0)
        self._oval = self.create_oval(1, 1, d - 1, d - 1, fill=self.COL, outline=self.COL)
        self.create_text(d / 2, d / 2, text="?", fill="white",
                         font=(_ui_font_family(), -max(9, int(round(12 * s))), "bold"))
        self.title_text, self.body_fn, self.pop, self.job = title, body_fn, None, None
        self.bind("<Enter>", self._enter, add="+")
        self.bind("<Leave>", self._leave, add="+")

    def _enter(self, _e=None):
        try:
            self.itemconfigure(self._oval, fill=self.COL_HOT, outline=self.COL_HOT)
            self.job = self.after(120, self._show)
        except tk.TclError:
            self.job = None

    def _leave(self, _e=None):
        try:
            if self.job:
                self.after_cancel(self.job)
            self.itemconfigure(self._oval, fill=self.COL, outline=self.COL)
        except tk.TclError:
            pass
        self.job = None
        self._hide()

    def _hide(self):
        if self.pop is not None:
            try:
                self.pop.destroy()
            except tk.TclError:
                pass
            self.pop = None

    def _show(self):
        if self.pop is not None:
            return
        s = UI["scale"]
        try:
            body = self.body_fn()[:HELP_MAX]
            fam, px = _ui_font_family(), -max(10, int(round(12 * s)))
            top = tk.Toplevel(self)
            top.wm_overrideredirect(True)
            try:
                top.attributes("-topmost", True)
            except tk.TclError:
                pass
            frame = tk.Frame(top, background="#fffbe6", highlightbackground=self.COL,
                             highlightcolor=self.COL, highlightthickness=max(1, int(round(1.5 * s))))
            frame.pack()
            wrap = int(480 * s)
            pad = int(8 * s)
            tk.Label(frame, text=self.title_text, font=(fam, px, "bold"), background="#fffbe6", foreground="#b34700",
                     justify="left", anchor="w", wraplength=wrap).pack(fill="x", padx=pad, pady=(pad, int(4 * s)))
            tk.Label(frame, text=body, font=(fam, px), background="#fffbe6", foreground="#222",
                     justify="left", anchor="w", wraplength=wrap).pack(fill="x", padx=pad, pady=(0, pad))
            top.update_idletasks()
            w, h = top.winfo_reqwidth(), top.winfo_reqheight()
            sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
            bx, by = self.winfo_rootx(), self.winfo_rooty()
            x = bx + int(22 * s)
            y = by + int(22 * s)
            if x + w > sw - 8:                 # not enough room on the right: open to the left of the badge
                x = max(0, bx - w - int(6 * s))
            if y + h > sh - 8:                 # not enough room below: open above, or at the screen bottom
                y = by - h - int(6 * s) if by - h - int(6 * s) >= 0 else max(0, sh - h - 8)
            top.wm_geometry(f"+{x}+{y}")
            self.pop = top
        except tk.TclError:
            self.pop = None


class PlotPane(ctk.CTkFrame):
    """A matplotlib figure embedded in a customtkinter frame (light theme).

    `dpi` is the dpi the Figure is created with; the caller computes it from the UI scale so that
    text and lines in the plots have the same apparent size as the widgets on any screen.
    """

    def __init__(self, master, figsize=(8, 6), toolbar=True, dpi=100.0):
        super().__init__(master, fg_color="white")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.extra = ctk.CTkFrame(self, fg_color="transparent", height=0)   # grows only when it holds widgets
        self.extra.pack(side="top", fill="x")
        if toolbar:
            tb = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
        self._tkw = self.canvas.get_tk_widget()
        self._tkw.pack(side="top", fill="both", expand=True)
        # matplotlib applies its HiDPI ratio when the widget is first mapped and then asks the canvas for
        # ratio x its old size; if the layout cannot grant it, no <Configure> follows and the figure stays
        # too large (cropped plot).  Re-fit the figure to the real widget size after that.
        self._tkw.bind("<Map>", lambda _e: self.after(120, self._refit), add="+")
        self._pending = None          # (fn, args, kwargs) of a plot not drawn yet because its tab is hidden
        self.message(tr("plot.run_hint"))

    def _refit(self):
        try:
            w, h = self._tkw.winfo_width(), self._tkw.winfo_height()
            if w < 20 or h < 20:
                return
            fw, fh = self.fig.get_size_inches() * self.fig.dpi
            if abs(fw - w) > 2 or abs(fh - h) > 2:
                self._tkw.configure(width=w, height=h)
                self.canvas.resize(types.SimpleNamespace(width=w, height=h))
        except tk.TclError:
            pass

    def message(self, text):
        self._pending = None
        _placeholder(self.fig, text)
        self.canvas.draw_idle()
        self.after_idle(self._refit)

    def flush(self):
        """Draw the plot that was deferred while this pane's tab was hidden."""
        if self._pending is not None:
            fn, a, k = self._pending
            self._pending = None
            self.redraw(fn, *a, **k)

    def redraw(self, fn, *a, defer=False, **k):
        if defer:                     # hidden tab: remember it, draw it when the tab is shown
            self._pending = (fn, a, k)
            return
        self._pending = None
        try:
            fn(self.fig, *a, **k)
        except Exception as exc:   # never let a plotting problem kill the GUI
            _placeholder(self.fig, tr("plot.draw_fail", exc=exc))
        self.canvas.draw_idle()
        self.after_idle(self._refit)


# ======================================================================================
# Main application
# ======================================================================================

RTAB_IDS = ("Antenna", "Pattern 2D", "Pattern 3D", "Matching", "Summary", "Log")
IS_MAC = sys.platform == "darwin"


class App(ctk.CTk):
    def __init__(self, cfg: dict, load_example: bool = False):
        super().__init__()
        self.cfg = cfg
        self.force_flags: set[str] = set()   # flags always written to the command line (even at default value)
        set_language(cfg["language"])

        # ---- display scale.  ctk_dpi: what customtkinter detects by itself (Windows per-monitor DPI;
        # 1.0 elsewhere).  sys_total: the scale "Auto" stands for.  Total scale = ctk_dpi * user factor.
        try:
            self.ctk_dpi = float(self._get_widget_scaling()) or 1.0
        except Exception:
            self.ctk_dpi = 1.0
        self.sys_total = self._detect_system_scale()
        self.cur_scale = self.ctk_dpi
        self._rebuilding = False
        self._apply_scale()

        self.title(f"{APP_TITLE} - {tr('app.subtitle')}")
        icon = SCRIPT_DIR / "assets" / "hb9cv-studio.png"
        if icon.is_file():
            try:
                self._icon = tk.PhotoImage(file=str(icon))
                self.iconphoto(True, self._icon)
            except tk.TclError:
                pass
        self.configure(fg_color="#eef1f6")
        self._apply_geometry()

        self.core_path = self._locate_core()
        self.core = load_core(self.core_path)
        self.q: queue.Queue = queue.Queue()
        self.proc = None
        self.running = False
        self.stopped = False
        self.log_chunks: list[str] = []
        self.last_result = None

        # variables that live as long as the application (the widgets are rebuilt on language/scale change)
        self.mode_var = tk.StringVar(value="generate")
        self.workdir_var = tk.StringVar(value=str(Path.cwd()))
        self.nec2c_var = tk.StringVar(value=shutil.which("nec2c") or "")
        self.step_var = tk.StringVar(value="5")
        self.patterns_var = tk.BooleanVar(value=True)
        self.p3_linear_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value=tr("st.ready"))
        self.lang_var = tk.StringVar()
        self.scale_var = tk.StringVar()
        self.vars: dict[str, tk.Variable] = {}
        self._pv_job = None                  # pending (debounced) command-line preview refresh
        self._busy_ui = False                # True while the widgets are torn down / rebuilt
        self._wstate: dict[str, bool] = {}   # flag -> enabled state last applied to its widgets
        self._lframes: dict[str, object] = {}
        self._built_groups: set[str] = set()
        for o in OPTS:
            var = tk.BooleanVar(value=(o.default == "1")) if o.kind == "bool" else tk.StringVar(value=o.default)
            var.trace_add("write", lambda *_: self._schedule_preview())
            self.vars[o.flag] = var
        self.rows: dict[str, tuple] = {}
        self.pane: dict[str, PlotPane] = {}
        self.preview = None

        self._build_layout()
        self._refresh_modes()
        if load_example:
            self.load_default_example()
        self._update_preview()

        for seq in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            self.bind(seq, lambda _e: self._step_scale(+1))
        for seq in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.bind(seq, lambda _e: self._step_scale(-1))
        self.bind("<Control-Key-0>", lambda _e: self._step_scale(0))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick = 0
        self.after(80, self._poll)

    # ---------------------------------------------------------------- scale handling
    def _detect_system_scale(self) -> float:
        """The total scale that 'Auto' means (1.0 = 96 dpi)."""
        if IS_WIN or IS_MAC:
            return self.ctk_dpi          # Windows: customtkinter follows the monitor DPI; macOS: Tk uses points
        env = os.environ.get("HB9CV_GUI_SCALE", "").strip()
        if env:
            try:
                return min(max(float(env), 0.5), 4.0)
            except ValueError:
                pass
        dpi = None
        try:
            out = subprocess.run(["xrdb", "-query"], capture_output=True, text=True, timeout=1.5).stdout
            m = re.search(r"Xft\.dpi:\s*([\d.]+)", out)
            if m:
                dpi = float(m.group(1))
        except Exception:
            pass
        if dpi is None:
            try:
                dpi = float(self.winfo_fpixels("1i"))
            except Exception:
                dpi = 96.0
        f = dpi / 96.0
        for var in ("GDK_SCALE", "QT_SCALE_FACTOR"):
            try:
                f = max(f, float(os.environ[var]))
            except (KeyError, ValueError):
                pass
        return min(max(round(f * 4) / 4, 1.0), 4.0)

    def _cfg_total(self) -> float:
        s = self.cfg.get("scale", "auto")
        total = self.sys_total if s == "auto" else float(s)
        return min(max(total, 0.6), 4.0)

    def _apply_scale(self) -> None:
        total = self._cfg_total()
        user = total / self.ctk_dpi
        ctk.set_widget_scaling(user)
        ctk.set_window_scaling(user)
        if not (IS_WIN or IS_MAC):
            # raw Tk things (matplotlib toolbar icons, point-sized fonts, matplotlib's own dpi ratio)
            try:
                self.tk.call("tk", "scaling", total * 96.0 / 72.0)
            except tk.TclError:
                pass
        self._sync_scale()

    def _sync_scale(self) -> None:
        try:
            s = float(self._get_widget_scaling())
        except Exception:
            s = self._cfg_total()
        self.cur_scale = UI["scale"] = s

    def _window_scale(self) -> float:
        try:
            return float(self._get_window_scaling())
        except Exception:
            return self.cur_scale

    def _fig_dpi(self) -> float:
        """dpi for new matplotlib Figures so that the final rendering dpi is 100 * UI scale."""
        ratio = 1.0
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if hasattr(FigureCanvasTkAgg, "_update_device_pixel_ratio"):   # matplotlib >= 3.5 multiplies by it
                ratio = round(float(self.tk.call("tk", "scaling")) / (96.0 / 72.0), 2) or 1.0
        except Exception:
            pass
        return 100.0 * UI["scale"] / ratio

    def _apply_geometry(self) -> None:
        """Window size in logical units, never larger than the screen; centred."""
        ws = self._window_scale()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = int(min(1620, 0.94 * sw / ws))
        h = int(min(940, 0.90 * sh / ws))
        try:
            self.minsize(int(min(1180, w)), int(min(700, h)))
            x = max(0, (sw - int(w * ws)) // 2)
            y = max(0, (sh - int(h * ws)) // 2 - int(20 * ws))
            self.geometry(f"{w}x{h}+{x}+{y}")
        except tk.TclError:
            pass

    def _left_width(self) -> int:
        lw = self.winfo_screenwidth() / self._window_scale()
        return int(max(520, min(640, 0.42 * lw)))

    def _scale_labels(self):
        auto = f"{tr('scale.auto')} ({int(round(self.sys_total * 100))}%)"
        pcts = set(SCALE_CHOICES)
        cur = self.cfg.get("scale", "auto")
        if cur != "auto":
            pcts.add(int(round(float(cur) * 100)))
        values = [auto] + [f"{p}%" for p in sorted(pcts)]
        current = auto if cur == "auto" else f"{int(round(float(cur) * 100))}%"
        return values, current

    def _on_scale(self, label: str):
        if label.startswith(tr("scale.auto")):
            self.cfg["scale"] = "auto"
        else:
            try:
                self.cfg["scale"] = float(label.rstrip("%")) / 100.0
            except ValueError:
                return
        save_config(self.cfg)
        self._rescale()

    def _step_scale(self, delta: int):
        cur = self.cur_scale * 100.0
        if delta == 0:
            self.cfg["scale"] = "auto"
        elif delta > 0:
            c = [v for v in SCALE_CHOICES if v > cur + 1]
            self.cfg["scale"] = (c[0] if c else SCALE_CHOICES[-1]) / 100.0
        else:
            c = [v for v in SCALE_CHOICES if v < cur - 1]
            self.cfg["scale"] = (c[-1] if c else SCALE_CHOICES[0]) / 100.0
        save_config(self.cfg)
        self._rescale()

    def _rescale(self):
        snap = self._snapshot()
        self._busy_ui = True
        self._teardown()              # destroy first: set_widget_scaling() would redraw every live widget
        self._apply_scale()
        self.after(30, lambda: self._after_rescale(snap))

    def _after_rescale(self, snap=None):
        self._apply_geometry()
        self._rebuild_ui(snap)

    def _check_scale(self):
        """Detect a DPI change done by the system (window moved to another monitor) and redraw."""
        if self._rebuilding or self._busy_ui:
            return
        try:
            s = float(self._get_widget_scaling())
        except Exception:
            return
        if abs(s - self.cur_scale) > 0.01:
            self._sync_scale()
            self.after(50, self._rebuild_ui)

    # ---------------------------------------------------------------- language
    def _on_language(self, name: str):
        code = next((c for c, n in LANGS.items() if n == name), None)
        if code is None or code == get_language():
            return
        set_language(code)
        self.cfg["language"] = code
        save_config(self.cfg)
        self.after(10, self._rebuild_ui)

    def _retranslate_status(self, old_values):
        cur = self.status_var.get()
        for key in ("st.ready", "st.done", "st.done_notes", "st.stopped", "st.error", "st.stopping"):
            if cur in STR[key]:
                self.status_var.set(tr(key))
                return

    # ---------------------------------------------------------------- setup
    def _locate_core(self) -> Path:
        for p in (SCRIPT_DIR / "hb9cv.py", Path.cwd() / "hb9cv.py"):
            if p.exists():
                return p
        messagebox.showinfo(APP_TITLE, tr("core.notfound"))
        sel = filedialog.askopenfilename(title=tr("core.select"), filetypes=[("Python", "*.py")])
        if not sel:
            raise SystemExit(tr("core.required"))
        return Path(sel)

    def _ltab_id(self, shown: str):
        return next((g for g, n in self.ltab.items() if n == shown), None)

    def _rtab_id(self, shown: str):
        return next((k for k, n in self.rtab.items() if n == shown), None)

    def _rset(self, ident: str):
        try:
            self.rtabs.set(self.rtab[ident])
        except Exception:
            pass
        self._flush_visible()

    def _visible_rid(self):
        try:
            return self._rtab_id(self.rtabs.get())
        except Exception:
            return None

    def _flush_visible(self):
        """Draw the deferred plot of the tab that is on screen (called when a result tab is shown)."""
        pane = self.pane.get(self._visible_rid())
        if pane is not None:
            pane.flush()

    def _plot(self, ident, fn, *a):
        """Draw a plot now if its tab is visible, otherwise on first display (hidden 3D plots are costly)."""
        self.pane[ident].redraw(fn, *a, defer=(self._visible_rid() != ident))

    def _snapshot(self):
        """State that must survive a rebuild: log text, selected right/left tab."""
        log_txt, rid, lid = "", None, None
        try:
            self.update_idletasks()
            log_txt = self.log.get("1.0", "end-1c")
            rid = self._rtab_id(self.rtabs.get())
            lid = self._ltab_id(self.tabs.get())
        except Exception:
            pass
        return log_txt, rid, lid

    def _teardown(self):
        self.preview = None
        self.rows = {}
        self.pane = {}
        for w in self.winfo_children():
            w.destroy()

    def _rebuild_ui(self, snap=None):
        """Recreate every widget (new language / scale).  All state lives in the tk variables."""
        if self._rebuilding:
            return
        self._rebuilding = True
        self._busy_ui = True
        try:
            log_txt, rid, lid = snap if snap is not None else self._snapshot()
            self.title(f"{APP_TITLE} - {tr('app.subtitle')}")
            self._build_layout()
            if log_txt:
                self.log.insert("1.0", log_txt)
                self.log.see("end")
            self._retranslate_status(None)
            self._refresh_modes()
            if lid in self.ltab:
                self._set_ltab(lid)
            self._update_preview()
            if rid:
                self._rset(rid)                      # select first, so only this tab is drawn now
            if self.last_result is not None:
                self._render_result(self.last_result, select=False)
            if self.running:
                self._set_running_ui(True)
        finally:
            self._rebuilding = False
            self._busy_ui = False
            self._sync_scale()

    # ---- left option tabs are built lazily: only the visible one costs time
    def _set_ltab(self, gid: str):
        try:
            self._ensure_group(gid)
            self.tabs.set(self.ltab[gid])
        except Exception:
            pass

    def _on_ltab_click(self):
        gid = self._ltab_id(self.tabs.get())
        if gid:
            self._ensure_group(gid)

    def _ensure_group(self, gid: str):
        if gid in self._built_groups or gid not in self._lframes:
            return
        self._built_groups.add(gid)
        sf = self._lframes[gid]
        n = 0
        for o in OPTS:
            if o.group == gid:
                self._make_option(sf, o, n)
                n += 1

    def _build_layout(self):
        self.preview = None
        for w in self.winfo_children():
            w.destroy()
        self.rows = {}
        self.pane = {}
        self._wstate = {}
        self._lframes = {}
        self._built_groups = set()
        self.ltab = {g: tr("group." + g) for g in GROUP_ORDER}
        self.rtab = {k: tr("rtab." + k) for k in RTAB_IDS}
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        left = ctk.CTkFrame(self, width=self._left_width(), fg_color="white", corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 4))
        left.grid_propagate(False)
        right = ctk.CTkFrame(self, fg_color="white", corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 4))
        self._build_left(left)
        self._build_right(right)
        self._build_footer()

    def _build_footer(self):
        """Bottom line of the window: author, license and project URL (the last two open in the web browser)."""
        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 6))
        font = ctk.CTkFont(size=11)
        link_font = ctk.CTkFont(size=11, underline=True)
        grey, blue = "#667085", "#1f6fb2"
        items = [(f"{APP_TITLE}  \u00b7  {tr('ui.author')}: {APP_AUTHOR}  \u00b7  \u00a9 2026", None),
                 (f"{tr('ui.license')}: {APP_LICENSE}", APP_LICENSE_URL),
                 (f"{tr('ui.project')}: {PROJECT_URL}", PROJECT_URL)]
        for col, (text, url) in enumerate(items):
            if col:
                ctk.CTkLabel(foot, text="\u00b7", font=font, text_color=grey).grid(row=0, column=2 * col - 1, padx=6)
            lab = ctk.CTkLabel(foot, text=text, font=link_font if url else font, text_color=blue if url else grey,
                               cursor="hand2" if url else "")
            lab.grid(row=0, column=2 * col, sticky="w")
            if url:
                lab.bind("<Button-1>", lambda _e, u=url: self._open_url(u))
                ToolTip(lab, tr("ui.open_link"))

    def _open_url(self, url):
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _build_left(self, left):
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)
        head = ctk.CTkFrame(left, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text=APP_TITLE, font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=0, column=0, rowspan=2, sticky="w")
        lab_l = ctk.CTkLabel(head, text=tr("ui.language"), font=ctk.CTkFont(size=10), text_color="#667085")
        lab_l.grid(row=0, column=1, sticky="w", padx=(6, 0))
        lab_s = ctk.CTkLabel(head, text=tr("ui.scale"), font=ctk.CTkFont(size=10), text_color="#667085")
        lab_s.grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.lang_var.set(LANGS[get_language()])
        lang_menu = ctk.CTkOptionMenu(head, values=list(LANGS.values()), variable=self.lang_var, width=96,
                                      height=24, dynamic_resizing=False, font=ctk.CTkFont(size=12),
                                      command=self._on_language)
        lang_menu.grid(row=1, column=1, padx=(6, 0))
        vals, cur = self._scale_labels()
        self.scale_var.set(cur)
        scale_menu = ctk.CTkOptionMenu(head, values=vals, variable=self.scale_var, width=104, height=24,
                                       dynamic_resizing=False, font=ctk.CTkFont(size=12),
                                       command=self._on_scale)
        scale_menu.grid(row=1, column=2, padx=(6, 0))
        ToolTip(lang_menu, tr("ui.language.tip"))
        ToolTip(lab_l, tr("ui.language.tip"))
        ToolTip(scale_menu, tr("ui.scale.tip"))
        ToolTip(lab_s, tr("ui.scale.tip"))

        self.mode_btn = ctk.CTkSegmentedButton(head, values=[tr("mode." + m) for m in MODES],
                                               command=self._on_mode_display)
        self.mode_btn.set(tr("mode." + self.mode_var.get()))
        self.mode_btn.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.mode_info = ctk.CTkLabel(head, text="", wraplength=self._left_width() - 40, justify="left",
                                      text_color="#555", font=ctk.CTkFont(size=12))
        self.mode_info.grid(row=3, column=0, columnspan=3, sticky="w")

        env = ctk.CTkFrame(left, fg_color="#f6f8fb", corner_radius=8)
        env.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        env.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(env, text=tr("ui.workdir"), font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkEntry(env, textvariable=self.workdir_var, height=26).grid(row=0, column=1, sticky="ew", padx=4)
        ctk.CTkButton(env, text="...", width=30, height=26, command=self._pick_workdir).grid(row=0, column=2, padx=(0, 8))
        ctk.CTkLabel(env, text="nec2c", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", padx=8, pady=3)
        e2 = ctk.CTkEntry(env, textvariable=self.nec2c_var, height=26, placeholder_text=tr("ui.nec2c_ph"))
        e2.grid(row=1, column=1, sticky="ew", padx=4)
        ctk.CTkButton(env, text="...", width=30, height=26, command=self._pick_nec2c).grid(row=1, column=2, padx=(0, 8))
        opts = ctk.CTkFrame(env, fg_color="transparent")
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 6))
        ctk.CTkCheckBox(opts, text=tr("ui.patterns"), variable=self.patterns_var,
                        checkbox_width=18, checkbox_height=18, font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkLabel(opts, text="   " + tr("ui.step"), font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkOptionMenu(opts, values=["2", "3", "5", "10"], variable=self.step_var, width=60, height=24).pack(side="left", padx=4)
        ctk.CTkLabel(opts, text=tr("ui.deg"), font=ctk.CTkFont(size=12)).pack(side="left")

        self.tabs = ctk.CTkTabview(left, fg_color="white", segmented_button_selected_color="#2b6cb0",
                                   command=self._on_ltab_click)
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        for g in GROUP_ORDER:
            tab = self.tabs.add(self.ltab[g])
            sf = ctk.CTkScrollableFrame(tab, fg_color="white")
            sf.pack(fill="both", expand=True)
            sf.grid_columnconfigure((0, 1), weight=1, uniform="c")
            self._lframes[g] = sf
        # option rows are created lazily (_ensure_group): the first tab now, the others when first shown
        self._ensure_group(GROUP_ORDER[0])

        bottom = ctk.CTkFrame(left, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 12))
        bottom.grid_columnconfigure((0, 1, 2), weight=1, uniform="b")
        self.preview = ctk.CTkTextbox(bottom, height=58, wrap="word", font=ctk.CTkFont(family="Courier", size=11),
                                      fg_color="#f6f8fb", border_width=1, border_color="#d0d7e2")
        self.preview.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.run_btn = ctk.CTkButton(bottom, text=tr("ui.run"), height=36, fg_color="#2b8a3e",
                                     hover_color="#237032", font=ctk.CTkFont(size=14, weight="bold"), command=self._run)
        self.run_btn.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.stop_btn = ctk.CTkButton(bottom, text=tr("ui.stop"), height=36, fg_color="#c92a2a",
                                      hover_color="#a61e1e", state="disabled", command=self._stop)
        self.stop_btn.grid(row=1, column=1, sticky="ew", padx=3)
        ctk.CTkButton(bottom, text=tr("ui.copy"), height=36, command=self._copy_cmd).grid(
            row=1, column=2, sticky="ew", padx=(3, 0))
        ctk.CTkButton(bottom, text=tr("ui.save_preset"), height=32, command=self._save_preset).grid(
            row=2, column=0, sticky="ew", padx=(0, 3), pady=(6, 0))
        ctk.CTkButton(bottom, text=tr("ui.load_preset"), height=32, command=self._load_preset).grid(
            row=2, column=1, sticky="ew", padx=3, pady=(6, 0))
        ctk.CTkButton(bottom, text=tr("ui.open_nec"), height=32, fg_color="#5f6b7a", hover_color="#4a5360",
                      command=self._open_nec).grid(row=2, column=2, sticky="ew", padx=(3, 0), pady=(6, 0))

    def _build_right(self, right):
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self.rtabs = ctk.CTkTabview(right, fg_color="white", segmented_button_selected_color="#2b6cb0",
                                    command=self._flush_visible)
        self.rtabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 2))
        dpi = self._fig_dpi()
        for ident in ("Antenna", "Pattern 2D", "Pattern 3D", "Matching"):
            t = self.rtabs.add(self.rtab[ident])
            self.pane[ident] = PlotPane(t, figsize=(9, 6.4), dpi=dpi)
            self.pane[ident].pack(fill="both", expand=True)
        # 3D view buttons
        for key, el, az in (("view.iso", 24, -58), ("view.top", 90, -90), ("view.side", 0, -90), ("view.front", 0, 0)):
            ctk.CTkButton(self.pane["Pattern 3D"].extra, text=tr(key), width=64, height=24,
                          command=lambda e=el, a=az: self._view3d(e, a)).pack(side="left", padx=3, pady=3)
        ctk.CTkCheckBox(self.pane["Pattern 3D"].extra, text=tr("ui.radius_lin"), variable=self.p3_linear_var,
                        command=self._redraw3d, checkbox_width=18, checkbox_height=18).pack(side="left", padx=6)
        ctk.CTkLabel(self.pane["Pattern 3D"].extra, text=tr("ui.drag3d"),
                     text_color="#667085", font=ctk.CTkFont(size=11)).pack(side="left", padx=8)
        t = self.rtabs.add(self.rtab["Summary"])
        self.summary = ctk.CTkTextbox(t, wrap="word", font=ctk.CTkFont(family="Courier", size=12), fg_color="#fbfcfe")
        self.summary.pack(fill="both", expand=True)
        t = self.rtabs.add(self.rtab["Log"])
        self.log = ctk.CTkTextbox(t, wrap="none", font=ctk.CTkFont(family="Courier", size=11), fg_color="#fbfcfe")
        self.log.pack(fill="both", expand=True)

        bar = ctk.CTkFrame(right, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 8))
        bar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(bar, textvariable=self.status_var, anchor="w", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self.progress = ctk.CTkProgressBar(bar, mode="indeterminate", width=200)
        self.progress.grid(row=0, column=1, padx=8)
        self.progress.set(0)
        ctk.CTkButton(bar, text=tr("ui.open_folder"), width=110, height=26, command=self._open_folder).grid(row=0, column=2)

    # ---------------------------------------------------------------- option widgets
    def _make_option(self, parent, o: Opt, index: int):
        cell = ctk.CTkFrame(parent, fg_color="transparent")
        cell.grid(row=index // 2, column=index % 2, sticky="ew", padx=5, pady=3)
        cell.grid_columnconfigure(0, weight=1)
        var = self.vars[o.flag]
        widgets = []
        enabled = self.mode_var.get()[0].upper() in o.modes
        st = "normal" if enabled else "disabled"
        self._wstate[o.flag] = enabled
        if o.kind == "bool":
            lab = None
            w = ctk.CTkCheckBox(cell, text=opt_label(o), variable=var, checkbox_width=18, checkbox_height=18,
                                font=ctk.CTkFont(size=12), state=st)
            w.grid(row=0, column=0, sticky="w", pady=(14, 0))
            widgets.append(w)
        else:
            lab = ctk.CTkLabel(cell, text=opt_label(o), anchor="w", font=ctk.CTkFont(size=11),
                               text_color="#344054" if enabled else "#b0b7c3")
            lab.grid(row=0, column=0, sticky="w")
            if o.kind in ("choice", "tri"):
                w = ctk.CTkOptionMenu(cell, values=o.choices, variable=var, height=26, dynamic_resizing=False,
                                      font=ctk.CTkFont(size=12), state=st)
                w.grid(row=1, column=0, sticky="ew")
                widgets.append(w)
            elif o.kind.startswith("file"):
                fr = ctk.CTkFrame(cell, fg_color="transparent")
                fr.grid(row=1, column=0, sticky="ew")
                fr.grid_columnconfigure(0, weight=1)
                w = ctk.CTkEntry(fr, textvariable=var, height=26, placeholder_text=opt_ph(o), font=ctk.CTkFont(size=12),
                                 state=st)
                w.grid(row=0, column=0, sticky="ew")
                b = ctk.CTkButton(fr, text="...", width=28, height=26, state=st,
                                  command=lambda v=var, k=o.kind: self._pick_file(v, k))
                b.grid(row=0, column=1, padx=(3, 0))
                widgets += [w, b]
            else:
                w = ctk.CTkEntry(cell, textvariable=var, height=26, placeholder_text=opt_ph(o), font=ctk.CTkFont(size=12),
                                 state=st)
                w.grid(row=1, column=0, sticky="ew")
                widgets.append(w)
        tip = f"{o.flag}\n{opt_tip(o)}".strip()
        for w in widgets + ([lab] if lab else []):
            ToolTip(w, tip)
        # orange "?" at the end of the element: opens the extended help of the option
        badge = HelpBadge(cell, f"{opt_label(o)}   [{o.flag}]", lambda o=o: opt_help(o))
        if o.kind == "bool":
            badge.grid(row=0, column=1, sticky="w", padx=(6, 0), pady=(14, 0))
        else:
            badge.grid(row=1, column=1, sticky="w", padx=(6, 0))
        widgets.append(badge)   # (so it is the last widget of the row; its state is never disabled, see _refresh_modes)
        self.rows[o.flag] = (lab, widgets)

    def _pick_file(self, var, kind):
        if kind == "file_open":
            p = filedialog.askopenfilename(initialdir=self.workdir_var.get(),
                                           filetypes=[(tr("ft.necout"), "*.out"), (tr("ft.all"), "*.*")])
        else:
            p = filedialog.asksaveasfilename(initialdir=self.workdir_var.get())
        if p:
            try:
                p = os.path.relpath(p, self.workdir_var.get()) if Path(p).is_relative_to(self.workdir_var.get()) else p
            except Exception:
                pass
            var.set(p)

    def _pick_workdir(self):
        p = filedialog.askdirectory(initialdir=self.workdir_var.get())
        if p:
            self.workdir_var.set(p)

    def _pick_nec2c(self):
        p = filedialog.askopenfilename(title=tr("dlg.nec2c"))
        if p:
            self.nec2c_var.set(p)

    # ---------------------------------------------------------------- mode handling
    def _on_mode_display(self, shown: str):
        for m in MODES:
            if tr("mode." + m) == shown:
                self.mode_var.set(m)
                break
        self._on_mode()

    def _on_mode(self):
        self._refresh_modes()
        self._update_preview()

    def _refresh_modes(self):
        mode = self.mode_var.get()
        letter = mode[0].upper()
        self.mode_info.configure(text=tr("mode_info." + mode))
        for o in OPTS:
            enabled = letter in o.modes
            row = self.rows.get(o.flag)
            if row is None or self._wstate.get(o.flag) == enabled:   # tab not built yet, or already in that state
                continue
            self._wstate[o.flag] = enabled
            lab, widgets = row
            state = "normal" if enabled else "disabled"
            for w in widgets:
                if not isinstance(w, HelpBadge):      # the "?" stays usable even for options a mode ignores
                    w.configure(state=state)
            if lab is not None:
                lab.configure(text_color="#344054" if enabled else "#b0b7c3")
        if mode in ("match", "sweep"):
            self._set_ltab("Match")
        elif self._ltab_id(self.tabs.get()) == "Match":
            self._set_ltab("Main")

    # ---------------------------------------------------------------- default example
    def load_default_example(self):
        """Preload the common 146 MHz HB9CV (EXAMPLE_VALUES) in the widgets and force the flags on the command line."""
        self.force_flags = set(EXAMPLE_VALUES)
        self.mode_var.set(EXAMPLE_MODE)
        self.mode_btn.set(tr("mode." + EXAMPLE_MODE))
        self.patterns_var.set(True)
        for flag, value in EXAMPLE_VALUES.items():
            if flag in self.vars:
                self.vars[flag].set(value)
        self._refresh_modes()
        self._update_preview()

    # ---------------------------------------------------------------- command line
    def build_argv(self):
        mode = self.mode_var.get()
        letter = mode[0].upper()
        argv, errors = ["--mode", mode], []
        for o in OPTS:
            if letter not in o.modes:
                continue
            v = self.vars[o.flag].get()
            if o.kind == "bool":
                if v:
                    argv.append(o.flag)
                continue
            s = str(v).strip()
            if o.kind == "tri":
                if s == "yes":
                    argv.append(o.flag)
                elif s == "no":
                    argv.append("--no-" + o.flag[2:])
                continue
            if s == "" or (o.kind == "choice" and s == "auto"):
                continue
            if o.kind in ("float", "int"):
                try:
                    num = float(s) if o.kind == "float" else int(s)
                    if (o.default != "" and o.flag not in self.force_flags
                            and num == (float(o.default) if o.kind == "float" else int(o.default))):
                        continue
                except ValueError:
                    errors.append(tr("msg.bad_number" if o.kind == "float" else "msg.bad_int",
                                     label=opt_label(o), value=s))
                    continue
            elif s == o.default and o.flag not in self.force_flags:
                continue
            if o.flag == "--commercial":
                for pair in [p.strip() for p in s.split(";") if p.strip()]:
                    argv.append(f"{o.flag}={pair}")
            else:
                argv.append(f"{o.flag}={s}")
        return argv, errors

    def _schedule_preview(self):
        """Coalesce the many variable-trace callbacks (e.g. loading an example/preset) into one refresh."""
        if self._pv_job is None:
            try:
                self._pv_job = self.after_idle(self._run_preview)
            except tk.TclError:
                self._pv_job = None

    def _run_preview(self):
        self._pv_job = None
        self._update_preview()

    def _update_preview(self):
        if getattr(self, "preview", None) is None:
            return
        argv, _ = self.build_argv()
        cmd = " ".join(shlex.quote(a) for a in ["python", self.core_path.name, *argv])
        try:
            self.preview.configure(state="normal")
            self.preview.delete("1.0", "end")
            self.preview.insert("1.0", cmd)
            self.preview.configure(state="disabled")
        except tk.TclError:
            pass

    def _copy_cmd(self):
        self.clipboard_clear()
        self.clipboard_append(self.preview.get("1.0", "end").strip())
        self.status_var.set(tr("st.copied"))

    # ---------------------------------------------------------------- presets
    def _save_preset(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[(tr("ft.preset"), "*.json")])
        if not p:
            return
        data = {"mode": self.mode_var.get(), "workdir": self.workdir_var.get(), "nec2c": self.nec2c_var.get(),
                "values": {k: (v.get() if not isinstance(v, tk.BooleanVar) else bool(v.get()))
                           for k, v in self.vars.items()}}
        Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.status_var.set(tr("st.preset_saved", p=p))

    def _load_preset(self):
        p = filedialog.askopenfilename(filetypes=[(tr("ft.preset"), "*.json")])
        if not p:
            return
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            mode = data.get("mode", "generate")
            self.mode_var.set(mode if mode in MODES else "generate")
            self.mode_btn.set(tr("mode." + self.mode_var.get()))
            self.workdir_var.set(data.get("workdir", self.workdir_var.get()))
            self.nec2c_var.set(data.get("nec2c", self.nec2c_var.get()))
            for k, v in data.get("values", {}).items():
                if k in self.vars:
                    self.vars[k].set(v)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, tr("msg.preset_fail", exc=exc))
            return
        self._refresh_modes()
        self._update_preview()
        self.status_var.set(tr("st.preset_loaded", p=p))

    # ---------------------------------------------------------------- running
    def _design_freq(self) -> float:
        s = str(self.vars["--freq"].get()).strip()
        try:
            return float(s)
        except ValueError:
            pass
        band = str(self.vars["--band"].get()).strip()
        if self.mode_var.get() == "optimize" and band:
            try:
                a = [float(x) for x in band.split(",")]
                return 0.5 * (a[0] + a[1])
            except (ValueError, IndexError):
                pass
        return 145.0

    def _target_z(self) -> float:
        try:
            return float(str(self.vars["--target-impedance"].get()).strip())
        except ValueError:
            return 12.5 if self.vars["--balun-ratio"].get() == "4:1" else 50.0

    def _rear_sector(self):
        """Rear sector of the F/B figures: --rear-sector in optimize mode, where hb9cv.py uses it; generate always
        uses 150..210, so the plots agree with its log."""
        if self.mode_var.get() != "optimize":
            return 150.0, 210.0
        try:
            a, b = [float(x) for x in str(self.vars["--rear-sector"].get()).split(",")]
            if 0 <= a < b <= 360:
                return a, b
        except (ValueError, KeyError):
            pass
        return 150.0, 210.0

    def _ctx(self):
        return {"mode": self.mode_var.get(), "cwd": Path(self.workdir_var.get() or "."),
                "freq": self._design_freq(), "target": self._target_z(), "patterns": bool(self.patterns_var.get()),
                "step": float(self.step_var.get()), "nec2c": self.nec2c_var.get(),
                "rear_sector": self._rear_sector(), "json": Path(self.workdir_var.get() or ".") / ".hb9cv_gui_result.json",
                "sweep_csv": str(self.vars["--sweep-csv"].get()).strip() or "lnetwork_sweep.csv"}

    def _run(self):
        if self.running:
            return
        argv, errors = self.build_argv()
        if errors:
            messagebox.showerror(APP_TITLE, tr("msg.fix", items="\n".join(errors)))
            return
        ctx = self._ctx()
        if not ctx["cwd"].is_dir():
            messagebox.showerror(APP_TITLE, tr("msg.no_workdir", p=ctx["cwd"]))
            return
        if ctx["mode"] in ("generate", "optimize"):
            exe = find_nec2c(ctx["nec2c"], self.core)
            needs = ctx["mode"] == "optimize" or self.vars["--run"].get() or ctx["patterns"]
            if not exe and ctx["mode"] == "optimize":
                messagebox.showerror(APP_TITLE, tr("msg.need_nec2c"))
                return
            if not exe and needs:
                if not messagebox.askyesno(APP_TITLE, tr("msg.no_nec2c_continue")):
                    return
        self._clear_for_run(tr("st.running", mode=tr("mode." + ctx["mode"])))
        self.log_chunks = []
        try:
            ctx["json"].unlink()
        except OSError:
            pass
        argv = argv + [f"--json-out={ctx['json']}"]
        threading.Thread(target=self._worker_run, args=(argv, ctx), daemon=True).start()

    def _set_running_ui(self, running: bool):
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        if running:
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.set(0)

    def _clear_for_run(self, status):
        self.running, self.stopped = True, False
        self._set_running_ui(True)
        self.status_var.set(status)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self._rset("Log")

    def _stop(self):
        self.stopped = True
        p = self.proc
        if p and p.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, **NO_WINDOW)
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.terminate()
        self.status_var.set(tr("st.stopping"))

    def _on_close(self):
        if self.running:
            self._stop()
        self.destroy()

    # ---- worker threads (no tkinter calls here, only queue puts)
    def _worker_run(self, argv, ctx):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        exe = find_nec2c(ctx["nec2c"], self.core)
        if exe:
            env["PATH"] = str(Path(exe).resolve().parent) + os.pathsep + env.get("PATH", "")
            if ctx["mode"] in ("generate", "optimize"):
                argv = argv + [f"--nec2c={exe}"]
        cmd = [console_python(), "-u", str(self.core_path), *argv]
        self.q.put(("log", "$ " + " ".join(shlex.quote(a) for a in cmd[2:]) + "\n\n"))
        kw = {"start_new_session": True} if os.name != "nt" else dict(NO_WINDOW)
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(ctx["cwd"]), env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **kw)
            for line in self.proc.stdout:
                self.log_chunks.append(line)
                self.q.put(("log", line))
            rc = self.proc.wait()
        except Exception as exc:
            self.q.put(("error", tr("msg.start_fail", exc=exc)))
            return
        log = "".join(self.log_chunks)
        if self.stopped:
            self.q.put(("stopped", None))
            return
        if rc != 0:
            self.q.put(("failed", (rc, log)))
            return
        self.q.put(("status", tr("st.post")))
        self.q.put(("done", self._postprocess(ctx, log)))

    def _postprocess(self, ctx, log):
        res = {"mode": ctx["mode"], "log": log, "errors": []}
        js = None
        try:
            js = json.loads(Path(ctx["json"]).read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            js = None
        res["json"] = js
        try:
            if ctx["mode"] in ("generate", "optimize"):
                deck = js.get("deck") if js else None
                if not deck:
                    m = re.search(r"^NEC generated:\s*(.+)$", log, re.M) or re.search(r"^Model:\s*(.+)$", log, re.M)
                    if not m:
                        raise ValueError(tr("err.no_nec_file"))
                    deck = m.group(1).strip()
                self._fill_model_result(res, ctx, self._resolve(ctx, deck))
            elif ctx["mode"] == "match":
                res["match"] = match_from_json(js, log) if js else parse_match_log(log)
                z0 = (js or {}).get("z0_ohm")
                if z0 is None:
                    try:
                        z0 = float(str(self.vars["--z0"].get()).strip())
                        if str(self.vars["--balun-ratio"].get()).strip() == "4:1":
                            z0 /= 4.0     # the network is designed on the balanced side of a 4:1 balun
                    except ValueError:
                        z0 = 50.0
                res["match"]["z0"] = float(z0)
            else:
                p = self._resolve(ctx, ctx["sweep_csv"])
                with open(p, newline="", encoding="utf-8") as fh:
                    res["sweep"] = list(csv.DictReader(fh))
        except Exception as exc:
            res["errors"].append(str(exc))
        return res

    def _resolve(self, ctx, raw) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (ctx["cwd"] / p)

    def _fill_model_result(self, res, ctx, nec_path: Path):
        res["nec"] = str(nec_path)
        text = nec_path.read_text(errors="replace")
        res["geom"] = parse_nec_geometry(text)
        res["info"] = extract_log_info(res.get("log", ""))
        res["freq"] = ctx["freq"]
        # pattern_error is kept as (translation key, arguments) so it can be re-translated if the language changes
        if not ctx["patterns"]:
            res["pattern_error"] = ("pat.off", {})
            return
        exe = find_nec2c(ctx["nec2c"], self.core)
        if not exe:
            res["pattern_error"] = ("pat.no_nec2c", {})
            return
        js = res.get("json") or {}
        loss = js.get("conductor_loss") or (js.get("performance") or {}).get("loss")
        if isinstance(loss, dict) and loss.get("efficiency_estimate") is not None:
            eff = float(loss["efficiency_estimate"])
        else:
            m = re.search(r"Estimated efficiency of the conductors:\s*([\d.]+)\s*%", res.get("log", ""))
            eff = float(m.group(1)) / 100.0 if m else None
        try:
            res["an"] = simulate_pattern(self.core, nec_path, ctx["freq"], ctx["step"], exe, ctx["target"],
                                         ctx["rear_sector"], eff)
            res["errors"] += res["an"].get("notes", [])
        except Exception as exc:
            res["pattern_error"] = ("pat.failed", {"exc": str(exc)})

    def _open_nec(self):
        if self.running:
            return
        p = filedialog.askopenfilename(initialdir=self.workdir_var.get(),
                                       filetypes=[(tr("ft.necdeck"), "*.nec *.txt"), (tr("ft.all"), "*.*")])
        if not p:
            return
        ctx = self._ctx()
        self._clear_for_run(tr("st.sim_deck"))
        self.log_chunks = []

        def work():
            res = {"mode": "generate", "log": "", "errors": []}
            try:
                self._fill_model_result(res, ctx, Path(p))
            except Exception as exc:
                res["errors"].append(str(exc))
            self.q.put(("done", res))

        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------------- UI updates from the queue
    def _poll(self):
        if self._busy_ui:                      # widgets are being rebuilt: keep queued items for the next tick
            self.after(80, self._poll)
            return
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", payload)
                    self.log.see("end")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self._finish(payload)
                elif kind == "failed":
                    rc, log = payload
                    self._end_run(tr("st.failed", rc=rc))
                    tail = "\n".join([l for l in log.strip().splitlines() if l.strip()][-6:])
                    messagebox.showerror(APP_TITLE, tr("msg.exit_code", rc=rc, tail=tail))
                elif kind == "stopped":
                    self._end_run(tr("st.stopped"))
                elif kind == "error":
                    self._end_run(tr("st.error"))
                    messagebox.showerror(APP_TITLE, payload)
        except queue.Empty:
            pass
        self._tick += 1
        if self._tick % 12 == 0:
            self._check_scale()
        self.after(80, self._poll)

    def _end_run(self, status):
        self.running = False
        self.proc = None
        self._set_running_ui(False)
        self.status_var.set(status)

    def _finish(self, res):
        self.last_result = res
        notes = self._render_result(res, select=True)
        self._end_run(tr("st.done_notes") if notes else tr("st.done"))

    def _render_result(self, res, select=True):
        """Draw the plots and fill the summary from a result dict.  Returns the list of notes."""
        mode = res["mode"]
        notes = list(res.get("errors", []))
        perr = res.get("pattern_error")
        perr_text = tr(perr[0], **perr[1]) if perr else None
        for pn in self.pane.values():    # drop plots still waiting from a previous result
            pn._pending = None
        if select:                       # show the target tab first: only visible plots are drawn right away
            if mode in ("generate", "optimize"):
                self._rset("Antenna" if res.get("geom") else "Log")
            else:
                self._rset("Matching")
        if mode in ("generate", "optimize"):
            geom, an = res.get("geom"), res.get("an")
            if geom:
                self._plot("Antenna", draw_antenna, geom, an, res.get("info"), res.get("freq"))
            else:
                self.pane["Antenna"].message(tr("res.no_geom") + "\n" + "\n".join(notes))
            if an:
                self._plot("Pattern 2D", draw_pattern_2d, an)
                self._plot("Pattern 3D", draw_pattern_3d, an, geom, bool(self.p3_linear_var.get()))
            else:
                msg = perr_text or tr("res.no_pattern")
                self.pane["Pattern 2D"].message(msg)
                self.pane["Pattern 3D"].message(msg)
            self.pane["Matching"].message(tr("res.match_hint"))
            if perr_text:
                notes.append(perr_text)
        else:
            for n in ("Antenna", "Pattern 2D", "Pattern 3D"):
                self.pane[n].message(tr("res.tab_unused"))
            if mode == "match" and "match" in res:
                self._plot("Matching", draw_match, res["match"])
            elif mode == "sweep" and res.get("sweep"):
                self._plot("Matching", draw_sweep, res["sweep"])
            else:
                self.pane["Matching"].message(tr("res.no_plot") + "\n" + "\n".join(notes))
        self._write_summary(res, notes)
        return notes

    def _write_summary(self, res, notes):
        lines = []
        an, geom = res.get("an"), res.get("geom")
        if geom:
            try:
                ai = analyse_geometry(geom)
                r, f = ai["elems"]["rear"], ai["elems"]["front"]
                rows = [(tr("sum.model"), res.get("nec", "")),
                        (tr("sum.rear"), tr("sum.elem_val", len=r["len"] * 1000, d=", ".join(f"{d:g}" for d in r["diam"]))),
                        (tr("sum.front"), tr("sum.elem_val", len=f["len"] * 1000, d=", ".join(f"{d:g}" for d in f["diam"]))),
                        (tr("sum.spacing"), f"{abs(f['x'] - r['x']) * 1000:.1f} mm")]
                xs_sorted = sorted([(r["x"], r)] + [(q["x"], q) for q in ai["pars"]] + [(f["x"], f)], key=lambda t: t[0])
                for k, (xx, q) in enumerate(xs_sorted):
                    if q is r or q is f:
                        continue
                    nb = xs_sorted[k + 1][0] if q["role"] == "reflector" else xs_sorted[k - 1][0]
                    rows.append((f"{tr('el.' + q['role'])} {q['name'][1:]}",
                                 tr("sum.elem_val", len=q["len"] * 1000, d=", ".join(f"{d:g}" for d in q["diam"]))
                                 + f"   \u0394 {abs(xx - nb) * 1000:.0f} mm"))
                if ai["boom"]:
                    rows.append((tr("sum.boom"), f"{ai['boom']['length'] * 1000:.0f} mm"))
                lines += _table(rows) + [""]
            except Exception:
                pass
        if an:
            lines.append(tr("sum.pattern_hdr", f=an["freq"]))
            rows = [(tr("sum.fwd"), (f"{an['fwd']:.2f} dBi = {an['gain_dbd']:.2f} dBd" if an.get("gain_dbd") is not None
                                     else f"{an['fwd']:.2f} {tr('tbl.incl_ground')}")),
                    (tr("sum.peak"), tr("sum.peak_val", g=an["gmax"], az=an["phi_peak"], el=90 - an["theta_peak"])),
                    (tr("sum.takeoff"), tr("sum.takeoff_val", v=an["takeoff"])),
                    (tr("sum.fb"), tr("sum.fb_val", fb=fmt_opt(an["fb"], "{:.2f}", " dB"),
                                      rb=fmt_opt(an["fb_robust"], "{:.2f}", " dB"),
                                      cb=fmt_opt(an.get("fb_cone"), "{:.2f}", " dB"),
                                      a=an["rear_sector"][0], b=an["rear_sector"][1])),
                    (tr("sum.hpbw"), tr("sum.hpbw_val", a=fmt_opt(an["hpbw_az"], "{:.1f}", " deg"),
                                        e=fmt_opt(an["hpbw_el"], "{:.1f}", " deg")))]
            if an["zin"] is not None:
                z = an["zin"]
                rows.append(("Zin", tr("sum.zin_val", r=z.real, x=z.imag, z=an["zref"],
                                       v=fmt_opt(an["vswr"], "{:.2f}"))))
            if an.get("sphere_avg") is not None and an.get("directivity") is not None:
                rows.append((tr("sum.avg"), tr("sum.avg_val", a=an["sphere_avg"], d=an["directivity"])))
            lines += ["  " + l for l in _table(rows)] + [""]
        if res.get("mode") == "match" and res.get("match"):
            lines += res["match"]["summary"]
        if res.get("mode") == "sweep" and res.get("sweep"):
            lines.append(tr("sum.sweep_rows", n=len(res["sweep"])))
        for l in extract_log_info(res.get("log", "")):
            lines.append(l)
        warn = [l.strip() for l in res.get("log", "").splitlines() if l.strip().startswith("WARNING")]
        if warn:
            lines += ["", tr("sum.warnings")] + [f"  {w}" for w in warn[:12]]
        if notes:
            lines += ["", tr("sum.notes")] + [f"  {n}" for n in notes]
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(lines))

    def _redraw3d(self):
        res = self.last_result
        if res and res.get("an"):
            self._plot("Pattern 3D", draw_pattern_3d, res["an"], res.get("geom"), bool(self.p3_linear_var.get()))

    def _view3d(self, elev, azim):
        fig = self.pane["Pattern 3D"].fig
        if fig.axes:
            fig.axes[0].view_init(elev=elev, azim=azim)
            self.pane["Pattern 3D"].canvas.draw_idle()

    def _open_folder(self):
        p = self.workdir_var.get()
        try:
            if os.name == "nt":
                os.startfile(p)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as exc:
            messagebox.showerror(APP_TITLE, tr("msg.folder_fail", exc=exc))


def parse_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="HB9CV Studio - graphical front end for hb9cv.py",
        epilog="Copyright (C) 2026 Emiliano Gonzalez - LU3VEA. License GPLv3+: GNU GPL version 3 or later "
               "<https://www.gnu.org/licenses/gpl.html>. This is free software: you are free to change and "
               "redistribute it. There is NO WARRANTY, to the extent permitted by law.")
    ap.add_argument("--load-default-example", action="store_true",
                    help="preload the most common 2 m HB9CV (146 MHz, 0.125 lambda spacing, crossed open-wire "
                         "phase line, front element fed, 1:1 balun, free space, horizontal) in every GUI field "
                         "and on the command line, in optimize mode (best realized gain matched to 50 ohm over "
                         "145-147 MHz)")
    args, _unknown = ap.parse_known_args(argv)
    return args


def main():
    args = parse_cli()
    cfg = load_config()
    set_language(cfg["language"])
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = App(cfg, load_example=args.load_default_example)
    app.mainloop()


if __name__ == "__main__":
    main()
