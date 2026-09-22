"""Native window for drum2midi.

Launch without a console:  drum2midi.bat  (or pythonw drum2midi_gui.pyw)

One window: settings on the left; conversion, a timeline you can listen to, and the
results on the right. Built on PySide6 (Qt) so the design's toggles, segmented controls
and timeline can be drawn directly instead of being assembled from Tk canvases, and
so playback can seek.

Conversion runs in a separate process. Its output is read line by line on a background
thread and handed to the window through a Qt signal, so the window stays responsive and
the run can be cancelled.

Everything that decides *what* runs -- the command line, the overrides, the estimate,
the log parsing -- is plain functions at module level, so the smoke tests can check it
without a display.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import copy
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "gui_settings.json"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _console_python(exe: Path) -> Path:
    """The console interpreter beside a windowless one, for running child scripts.

    The window runs under pythonw.exe (or drum2midi.exe, its branded copy), which is
    a GUI-subsystem program, and Windows ignores CREATE_NO_WINDOW for those: the
    pipeline then ran with no console at all. Anything *it* started that is a console
    program -- audio-separator.exe, and FluidSynth for the listen renders -- found no
    console to inherit and got a fresh visible one. Measured: a Windows Terminal
    window opened for every MDX23C separation. python.exe with CREATE_NO_WINDOW gets
    a hidden console instead, and every descendant inherits that.
    """
    if sys.platform != "win32":
        return exe
    sibling = exe.with_name("python.exe")
    return sibling if sibling.exists() else exe


PY = str(_console_python(Path(sys.executable)))

sys.path.insert(0, str(ROOT))

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".opus"}

STAGES = {"[0/4]": 10, "[1/4]": 30, "[2/4]": 60, "[3/4]": 85, "[4/4]": 95}

# How much of the bar each stage owns, so a stage that reports its own progress can
# fill its slice smoothly instead of jumping.
STAGE_SPANS = {"[0/4]": (10, 28), "[1/4]": (30, 58), "[2/4]": (60, 84),
               "[3/4]": (85, 94), "[4/4]": (95, 99)}

# The chips under the progress bar, one per pipeline stage. "[0/4]" only runs when the
# drums are pulled out of a full song first.
STAGE_CHIPS = [("[0/4]", "Extract drums"), ("[1/4]", "Transcribe"),
               ("[2/4]", "Separate"), ("[3/4]", "Articulate"), ("[4/4]", "Export")]

# the pipeline forwards the separator's percentage as "      separating: 42%"
_SEPARATING = re.compile(r"separating:\s*(\d{1,3})%")

# Findings the pipeline already prints; surfaced live instead of left in the log.
_LIVE = {
    "tempo": re.compile(r"Writing MIDI at ([\d.]+) BPM"),
    "notes": re.compile(r"(\d+) hits detected"),
    "fused": re.compile(r"stem fusion: \+(\d+)"),
    "ride": re.compile(r"ride/crash: (\d+) hits reassigned"),
}

_NOTE_LETTERS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Cubase, Logic, Ableton and Reaper all call MIDI 60 "C3"; this follows them, as the
# rest of the project does.
_MIDDLE_C_OCTAVE = 3


def note_name(pitch: int) -> str:
    """36 -> "C1". The number is what gets written; the name is for the human."""
    if not 0 <= pitch <= 127:
        return "—"
    return f"{_NOTE_LETTERS[pitch % 12]}{pitch // 12 - (5 - _MIDDLE_C_OCTAVE)}"


def _count(n: int, word: str) -> str:
    """1 -> "1 note", 2 -> "2 notes"."""
    return f"{n} {word}{'s' * (n != 1)}"


def _device_plan() -> tuple[str | None, str | None]:
    """What each stage will actually run on: (separator name, transcriber name).

    Both come from `pick_devices`, because the split is not fixed — CUDA takes the
    transcriber too, where cuDNN's fused recurrent kernels beat the CPU, while XPU and
    MPS do not and leave it behind. Reporting a hardcoded split was wrong on any NVIDIA
    machine, which is most of them.

    Imported lazily and defensively: the GUI must still open when torch is missing
    or a driver is broken.
    """
    try:
        from devices import SEPARATOR, TRANSCRIBER, describe_device, pick_devices
        plan = pick_devices("auto")
        sep = plan[SEPARATOR]
        tra = plan[TRANSCRIBER]
        return (None if sep == "cpu" else describe_device(sep),
                None if tra == "cpu" else describe_device(tra))
    except Exception:
        return (None, None)


def device_line(plan: tuple[str | None, str | None]) -> tuple[str, bool]:
    """The sentence shown under the separator choices, and whether it is good news."""
    sep, tra = plan
    if sep is None:
        return ("No supported GPU found — everything runs on the CPU", False)
    if tra is None:
        return (f"Separation runs on {sep}; transcription stays on the CPU, "
                f"where recurrent layers are faster", True)
    if tra == sep:
        return (f"Separation and transcription both run on {sep}", True)
    return (f"Separation runs on {sep}; transcription on {tra}", True)


@dataclass(frozen=True)
class Separator:
    id: str
    name: str
    stems: str
    desc: str
    quality: int        # 1..5, for the little meter
    speed: int          # 1..5
    more: bool = False  # hidden behind "Show 2 more"


SEPARATORS = [
    Separator("uvr", "MDX23C", "6 stems",
              "Separate ride and crash. Best quality.", 5, 1),
    Separator("larsnet", "LarsNet", "5 stems",
              "No ride. About 10 s per minute of audio, weaker on ghost notes "
              "and cymbals.", 3, 4),
    Separator("none", "No separation", "—",
              "Fastest, but velocity is a flat 100.", 2, 5),
    Separator("drumsep", "DrumSep", "4 stems", "Not recommended.", 2, 2, more=True),
    Separator("hybrid", "Hybrid", "—", "Not recommended.", 2, 1, more=True),
]

# The quantize grid as a segmented control, plus a triplet switch for the two grids
# that have a triplet form. The values are what --quantize takes.
QUANT_STEPS = [("", "Off"), ("4", "1/4"), ("8", "1/8"), ("16", "1/16"), ("32", "1/32")]
TRIPLETS = {"8": "12", "16": "24"}

# Everything the pipeline can emit, as the note number that actually reaches the file.
# The kick is 36 here rather than ADTOF's internal 35, because this table is what the
# user compares against their sampler.
KIT = [
    (36, "Kick"),
    (38, "Snare"),
    (43, "Floor tom"),
    (47, "Mid tom"),
    (50, "High tom"),
    (42, "Hi-hat closed"),
    (44, "Hi-hat pedal"),
    (46, "Hi-hat open"),
    (49, "Crash"),
    (51, "Ride"),
]
DRUM_CHANNEL = 9

# The pipeline prints General MIDI names; map them back to pitches so a row can be
# given the right colour. Both kick numbers appear because 35 is the internal class
# and 36 is what gets written.
LABEL_TO_PITCH = {
    "kick": 36, "snare": 38, "snare rim": 40,
    "floor tom": 43, "floor tom hi": 43, "low tom": 45, "mid tom": 47,
    "hi-mid tom": 48, "high tom": 50,
    "hi-hat closed": 42, "hi-hat pedal": 44, "hi-hat open": 46,
    "crash": 49, "ride": 51,
    # coarse class names used by the early preview, before articulation is decided
    "toms": 47, "hi-hat": 42, "cymbals": 49,
}

# Timeline lanes, low to high, and which pitches land on each.
LANES = [("Kick", "kick"), ("Snare", "snare"), ("Hi-hat", "hat"),
         ("Toms", "tom"), ("Crash", "crash"), ("Ride", "ride")]
_LANE_OF = {35: 0, 36: 0, 37: 1, 38: 1, 40: 1, 42: 2, 44: 2, 46: 2,
            41: 3, 43: 3, 45: 3, 47: 3, 48: 3, 50: 3,
            49: 4, 52: 4, 55: 4, 57: 4, 51: 5, 53: 5, 59: 5}


def lane_of(pitch: int) -> int | None:
    return _LANE_OF.get(pitch)


def colour_key(pitch: int) -> str:
    """The token name a drum is drawn in: the same colour in every part of the window."""
    lane = lane_of(pitch)
    return LANES[lane][1] if lane is not None else "muted"


# --------------------------------------------------------------------- what runs
@dataclass
class Choices:
    """Everything the window lets you set, independent of any widget."""
    input: str = ""
    output: str = ""
    separator: str = "uvr"
    song: bool = False
    fuse: bool = True
    learned: bool = True
    rescue: bool = True
    hats: bool = True
    split: bool = False
    quant: str = ""
    triplets: bool = False
    tempo: str = ""
    thresholds: str = ""
    # source pitch -> written note, and source pitch -> channel (0-based, as the CLI
    # takes it). Rows equal to General MIDI are not passed on.
    notes: dict = field(default_factory=lambda: {p: p for p, _ in KIT})
    channels: dict = field(default_factory=lambda: {p: DRUM_CHANNEL for p, _ in KIT})


def fuse_available(sep: str) -> bool:
    return sep == "uvr"


def rescue_available(sep: str) -> bool:
    # rescue_toms only helps when the tom stem is dirty, which is LarsNet's case;
    # on MDX23C's clean stems it adds false positives (toms 0.605 -> 0.398)
    return sep in ("larsnet", "drumsep", "hybrid")


def quant_value(c: Choices) -> str:
    if c.triplets and c.quant in TRIPLETS:
        return TRIPLETS[c.quant]
    return c.quant


def overrides(c: Choices) -> tuple[dict, dict]:
    """Only rows that differ from the default; out-of-range values are ignored."""
    notes = {p: n for p, n in c.notes.items() if n != p and 0 <= n <= 127}
    chans = {p: ch for p, ch in c.channels.items()
             if ch != DRUM_CHANNEL and 0 <= ch <= 15}
    return notes, chans


def tempo_value(text: str) -> float | None:
    """Returns the tempo to force, None to let the pipeline estimate it.
    Raises ValueError if the field holds something that is not a usable BPM."""
    raw = text.strip().replace(",", ".")
    if not raw:
        return None
    value = float(raw)
    if not 20.0 <= value <= 400.0:
        raise ValueError("tempo outside the 20–400 range")
    return value


def build_command(c: Choices) -> list[str]:
    cmd = [PY, str(ROOT / "drum2midi.py"), c.input, "-o", c.output, "--device", "auto"]
    if c.separator == "none":
        cmd.append("--no-separate")
    else:
        cmd += ["--separator", c.separator]
        fuse = c.fuse and fuse_available(c.separator)
        cmd += ["--fuse-stem-onsets", "on" if fuse else "off"]
    if c.song:
        cmd.append("--from-song")
    if not c.learned:
        cmd.append("--no-learned")
    if not (c.rescue and rescue_available(c.separator)):
        cmd.append("--no-rescue-toms")
    if not c.hats:
        cmd.append("--no-split-hats")
    if c.split:
        cmd.append("--split-tracks")
    notes, chans = overrides(c)
    if notes:
        cmd += ["--notes", ",".join(f"{k}={v}" for k, v in sorted(notes.items()))]
    if chans:
        cmd += ["--channels", ",".join(f"{k}={v}" for k, v in sorted(chans.items()))]
    q = quant_value(c)
    if q:
        cmd += ["--quantize", q]
    tempo = tempo_value(c.tempo)
    if tempo is not None:
        cmd += ["--tempo", f"{tempo:g}"]
    if c.thresholds.strip():
        cmd += ["--thresholds", c.thresholds.strip()]
    return cmd


# Measured on the development machine: seconds of processing per second of audio.
# Measured with the separator on a GPU and the transcriber on the CPU. On CUDA the
# transcriber moves to the GPU too, which makes these an over-estimate there rather
# than a wrong shape.
def estimate_seconds(sep: str, audio_seconds: float, gpu: bool) -> float:
    rate = {"uvr": 1.9 if gpu else 15.0, "larsnet": 0.6, "none": 0.35,
            "drumsep": 1.2, "hybrid": 2.9 if gpu else 16.0}.get(sep, 1.0)
    return audio_seconds * rate


def human_duration(secs: float) -> str:
    return (f"{secs / 3600:.1f} h" if secs >= 5400
            else f"{secs / 60:.0f} min" if secs >= 90 else f"{secs:.0f} s")


def clock(t: float, tenths: bool = True) -> str:
    """72.4 -> "1:12.4"."""
    t = max(0.0, t)
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:04.1f}" if tenths else f"{m}:{int(s):02d}"


def summary_row(line: str) -> tuple[str, int, int, int] | None:
    """One row of the pipeline's final table: "Kick   270   15  127"."""
    parts = line.rstrip("\n").rsplit(None, 3)
    if len(parts) != 4 or not all(p.isdigit() for p in parts[1:]):
        return None
    return parts[0].strip(), int(parts[1]), int(parts[2]), int(parts[3])


def source_pitch(written: int, c: Choices) -> int:
    """The kit piece a written note came from, undoing the user's note overrides."""
    notes, _ = overrides(c)
    inverse = {v: k for k, v in notes.items()}
    return inverse.get(written, written if written not in notes else -1)


def row_pitch(label: str, c: Choices) -> int | None:
    """Maps a name from the summary table back to the kit piece it describes."""
    label = label.strip()
    if label.isdigit():
        written = int(label)
    else:
        written = LABEL_TO_PITCH.get(label.lower())
        if written is None:
            return None
    return source_pitch(written, c)


def render_missing() -> str | None:
    """What stops the MIDI from being rendered for listening, or None if nothing does."""
    try:
        import render_midi
        return render_midi.missing()
    except Exception as exc:  # noqa: BLE001 -- the window must open regardless
        return f"render_midi.py could not be loaded: {exc}"


def audio_files(folder: Path) -> list[Path]:
    return sorted(f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in AUDIO_EXTS)


def read_hits(midi: Path, c: Choices) -> list[tuple[int, float, int]]:
    """(lane, seconds, velocity) for every note in a written MIDI file."""
    import mido
    hits = []
    t = 0.0
    for msg in mido.MidiFile(str(midi)):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            lane = lane_of(source_pitch(msg.note, c))
            if lane is not None:
                hits.append((lane, t, msg.velocity))
    return hits


def envelope(path: Path, bins: int = 1600) -> list[float]:
    """A peak envelope of the recording, normalised to 0..1, for the timeline."""
    import numpy as np
    import soundfile as sf
    info = sf.info(str(path))
    hop = max(1, info.frames // bins)
    peaks = []
    for block in sf.blocks(str(path), blocksize=hop, always_2d=True):
        peaks.append(float(np.abs(block).max()) if block.size else 0.0)
    top = max(peaks) if peaks else 0.0
    return [p / top for p in peaks] if top > 0 else peaks


# =============================================================================== Qt
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal  # noqa: E402
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QIcon, QImage, QPainter,  # noqa: E402
                           QPainterPath, QPen, QPolygonF)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup,  # noqa: E402
                               QComboBox, QFileDialog, QFrame, QGraphicsOpacityEffect,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                               QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

# Exact values from the design (THEMES in "drum2midi App.dc.html"); its oklch colours
# are converted to sRGB here. The six drum colours share one lightness and chroma per
# theme and differ only in hue, in the kit's low-to-high order.
#
# One deliberate departure: the light theme's hi-hat is oklch(0.66 0.14 75), not the
# design's 0.70. At 0.70 amber on white is 2.7:1, under the 3:1 a mark needs to be seen
# at all, and the hi-hat is the busiest lane on the timeline. A smoke test holds this.
THEMES = {
    "dark": {
        "bg": "#0e1014", "panel": "#15181e", "raised": "#1c2028", "sunken": "#0b0d10",
        "line": "#232831", "line2": "#333a46", "text": "#e8eaef", "muted": "#959cab",
        "faint": "#4d5461", "knob": "#f3f4f7", "toggleOff": "#3a414d", "wave": "#5c6472",
        "accent": "#5ea8f9", "accentInk": "#0b0d10", "accentSoft": "rgba(94,168,249,26)",
        "ok": "#61cb7c", "warn": "#e7b551",
        "kick": "#9499fa", "snare": "#ef7e7c", "tom": "#5bbd74",
        "hat": "#e7b551", "crash": "#c987dd", "ride": "#24c1c9",
    },
    "light": {
        "bg": "#eef0f3", "panel": "#ffffff", "raised": "#f6f7f9", "sunken": "#eceef2",
        "line": "#e2e5ea", "line2": "#cfd4db", "text": "#15181d", "muted": "#5d6471",
        "faint": "#b3b8c1", "knob": "#ffffff", "toggleOff": "#c9ced6", "wave": "#a4abb6",
        "accent": "#1c6fd2", "accentInk": "#ffffff", "accentSoft": "rgba(28,111,210,18)",
        "ok": "#25984d", "warn": "#b07509",
        "kick": "#6262cc", "snare": "#cb4644", "tom": "#2a904b",
        "hat": "#c38300", "crash": "#9c4db4", "ride": "#00929f",
    },
}

# Geist is the design's face; the native fallbacks it names come next.
UI_FAMILIES = ["Geist", "Segoe UI Variable Text", "Segoe UI", "SF Pro Text",
               "Inter", "Cantarell", "DejaVu Sans"]
MONO_FAMILIES = ["Geist Mono", "Cascadia Mono", "Consolas", "SF Mono", "Menlo",
                 "DejaVu Sans Mono"]


def _first_installed(families: list[str], fallback: str) -> str:
    have = set(QFontDatabase.families())
    return next((f for f in families if f in have), fallback)


class Tokens:
    """The active theme, shared by the stylesheet and every hand-painted widget."""
    name = "dark"
    t = THEMES["dark"]
    ui = "sans-serif"
    mono = "monospace"

    @classmethod
    def c(cls, key: str) -> QColor:
        value = cls.t[key]
        if value.startswith("rgba"):
            r, g, b, a = (int(x) for x in value[5:-1].split(","))
            return QColor(r, g, b, a)
        return QColor(value)


def font(size: float, weight: int = 400, mono: bool = False) -> QFont:
    f = QFont(Tokens.mono if mono else Tokens.ui)
    f.setPixelSize(round(size))
    # QFont.Weight is a strict enum in PySide6, so the design's 550/650 round to the
    # nearest weight Qt names
    names = {400: QFont.Weight.Normal, 500: QFont.Weight.Medium,
             600: QFont.Weight.DemiBold, 700: QFont.Weight.Bold}
    f.setWeight(names[min(names, key=lambda w: abs(w - weight - 1))])
    return f


def stylesheet() -> str:
    t = dict(Tokens.t, ui=Tokens.ui, mono=Tokens.mono)
    return """
    * {{ color: {text}; }}
    QMainWindow, #root {{ background: {bg}; }}
    #header {{ background: {panel}; border-bottom: 1px solid {line}; }}
    #sidebar, #sidebarBody {{ background: {panel}; }}
    #sidebar {{ border: 0; border-right: 1px solid {line}; }}
    #card {{ background: {panel}; border: 1px solid {line}; border-radius: 12px; }}
    #cardHead {{ border: 0; border-bottom: 1px solid {line}; border-radius: 0; }}
    QLabel {{ background: transparent; }}
    QLabel[role="section"] {{ color: {muted}; font-size: 11px; font-weight: 600; }}
    QLabel[role="muted"] {{ color: {muted}; font-size: 12px; }}
    QLabel[role="hint"] {{ color: {muted}; font-size: 11px; }}
    QLabel[role="mono"] {{ color: {muted}; font-family: "{mono}"; font-size: 11px; }}
    QLabel[role="ok"] {{ color: {ok}; font-size: 12px; }}
    QLabel[role="chip"] {{ background: {sunken}; color: {muted}; font-family: "{mono}";
        font-size: 10px; border-radius: 4px; padding: 1px 5px; }}
    QLabel[role="colhead"] {{ color: {muted}; font-size: 10px; }}
    QPushButton {{ background: {raised}; border: 1px solid {line2}; border-radius: 6px;
        padding: 0 10px; min-height: 24px; font-size: 12px; font-weight: 500; }}
    QPushButton:hover {{ border-color: {muted}; }}
    QPushButton:disabled {{ color: {faint}; border-color: {line}; }}
    QPushButton#primary {{ background: {accent}; color: {accentInk}; border: 0;
        border-radius: 8px; min-height: 40px; padding: 0 22px; font-size: 14px;
        font-weight: 600; }}
    QPushButton#primary:disabled {{ background: {line2}; color: {muted}; }}
    QPushButton#big {{ min-height: 38px; border-radius: 8px; padding: 0 16px;
        font-size: 13px; }}
    QPushButton#link {{ background: transparent; border: 0; padding: 2px 0;
        color: {muted}; text-align: left; }}
    QPushButton#link:hover {{ color: {text}; }}
    QPushButton#round {{ border-radius: 16px; padding: 0; min-height: 0; }}
    QFrame#seg {{ background: {sunken}; border: 1px solid {line}; border-radius: 8px; }}
    QPushButton#segbtn {{ background: transparent; border: 0; border-radius: 6px;
        color: {muted}; min-height: 24px; padding: 0 10px; }}
    QPushButton#segbtn[mono="true"] {{ font-family: "{mono}"; }}
    QPushButton#segbtn:checked {{ background: {panel}; color: {text};
        border: 1px solid {line}; }}
    QPushButton#segbtn:disabled {{ color: {faint}; }}
    QPushButton#tab {{ background: transparent; border: 0; border-radius: 0;
        border-bottom: 2px solid transparent; color: {muted}; min-height: 40px;
        font-size: 13px; font-weight: 600; padding: 0 10px; }}
    QPushButton#tab:checked {{ color: {text}; border-bottom-color: {accent}; }}
    QLineEdit {{ background: {sunken}; border: 1px solid {line}; border-radius: 6px;
        min-height: 28px; padding: 0 10px; font-family: "{mono}"; font-size: 12px;
        selection-background-color: {accent}; selection-color: {accentInk}; }}
    QLineEdit:focus {{ border-color: {accent}; }}
    QLineEdit[path="true"] {{ color: {muted}; font-size: 11px; }}
    QFrame#sepcard {{ border: 1px solid {line}; border-radius: 9px;
        background: transparent; }}
    QFrame#sepcard:hover {{ border-color: {line2}; }}
    QFrame#sepcard[selected="true"] {{ border-color: {accent}; background: {accentSoft}; }}
    QFrame#filecard {{ background: {raised}; border: 1px solid {line};
        border-radius: 10px; }}
    QFrame#drop {{ border: 2px dashed {line2}; border-radius: 10px;
        background: transparent; }}
    QFrame#dropBig {{ border: 2px dashed {line2}; border-radius: 14px;
        background: {panel}; }}
    QFrame#devchip {{ background: {sunken}; border: 1px solid {line};
        border-radius: 13px; }}
    QFrame#row {{ border: 0; border-bottom: 1px solid {line}; background: transparent; }}
    QFrame#row:hover {{ background: {raised}; }}
    QFrame#colheads {{ border: 0; border-bottom: 1px solid {line}; }}
    QFrame#stat {{ border: 0; border-left: 1px solid {line}; }}
    QComboBox {{ background: transparent; border: 1px solid {line}; border-radius: 5px;
        padding: 0 6px; min-height: 20px; font-family: "{mono}"; font-size: 11px; }}
    QComboBox:hover {{ border-color: {line2}; }}
    QComboBox::drop-down {{ border: 0; width: 14px; }}
    QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
    QComboBox QAbstractItemView {{ background: {panel}; border: 1px solid {line2};
        selection-background-color: {accentSoft}; selection-color: {text};
        font-family: "{mono}"; font-size: 11px; outline: 0; }}
    QPlainTextEdit#log {{ background: {sunken}; border: 0; border-radius: 8px;
        padding: 8px 10px; font-family: "{mono}"; font-size: 11px; color: {muted};
        selection-background-color: {accent}; selection-color: {accentInk}; }}
    QScrollArea {{ background: transparent; border: 0; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {line2}; border-radius: 3px;
        min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {muted}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QToolTip {{ background: {raised}; color: {text}; border: 1px solid {line2};
        padding: 4px 6px; }}
    QMessageBox {{ background: {panel}; }}
    """.format(**t)


def _label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    lab.setWordWrap(wrap)
    return lab


def _section(text: str) -> QLabel:
    lab = _label(text.upper(), "section")
    f = lab.font()
    f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
    lab.setFont(f)
    return lab


def _repolish(w: QWidget) -> None:
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


# ------------------------------------------------------------------ small widgets
class ElidedLabel(QLabel):
    """A one-line label that ends in "…" instead of being clipped."""

    def __init__(self, text: str = ""):
        super().__init__()
        self.full = text
        self.setMinimumWidth(10)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text: str) -> None:
        self.full = text
        self._elide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self) -> None:
        QLabel.setText(self, self.fontMetrics().elidedText(
            self.full, Qt.TextElideMode.ElideMiddle, max(10, self.width())))


class NoteBox(QComboBox):
    """A combo box whose arrow is drawn here, since QSS cannot draw a triangle.

    It also ignores the mouse wheel until clicked. Qt's combo box takes the wheel on
    hover, so scrolling the sidebar or the table over the twenty note and channel
    pickers changed whichever one passed under the pointer -- three notches moved
    the snare from 38 to 41 without a click. The wheel now scrolls the panel.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)   # not WheelFocus

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()                                     # to the scroll area

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(Tokens.c("muted"))
        x, y = self.width() - 12, self.height() / 2 - 1.5
        p.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x + 6, y), QPointF(x + 3, y + 4)]))


def _note_box(note: int, width: int) -> NoteBox:
    box = NoteBox()
    box.setFixedWidth(width)
    box.addItems([f"{n} · {note_name(n)}" for n in range(128)])
    box.setMaxVisibleItems(16)
    box.setCurrentIndex(note)
    box.setToolTip("The note this drum is written as (--notes)")
    return box


def _channel_box(channel: int, width: int) -> NoteBox:
    box = NoteBox()
    box.setFixedWidth(width)
    # shown 1-16 like every DAW; the CLI takes 0-15
    box.addItems([str(n + 1) for n in range(16)])
    box.setCurrentIndex(channel)
    box.setToolTip("MIDI channel (--channels). 10 is the General MIDI drum channel.")
    return box


# ------------------------------------------------------------------ kit picture
_KIT_CACHE: dict[tuple, QImage | None] = {}


def kit_image(width: float, theme: str, alpha: int) -> QImage | None:
    """The drawn kit (drum_kit_art) in this theme's drum colours; None without Pillow.

    Rendered at a width rounded up to 32 px and scaled down when painted, so dragging
    the window's edge does not redraw the kit for every pixel.
    """
    width = max(64, -(-int(width) // 32) * 32)
    key = (width, theme, alpha)
    if key not in _KIT_CACHE:
        if len(_KIT_CACHE) >= 8:          # a resize walks through sizes; keep few
            _KIT_CACHE.clear()
        try:
            import drum_kit_art
            t = THEMES[theme]
            img = drum_kit_art.render(width, dark=theme == "dark", alpha=alpha,
                                      colours=lambda p: t[colour_key(p)])
            _KIT_CACHE[key] = QImage(img.tobytes("raw", "RGBA"), img.width, img.height,
                                     img.width * 4, QImage.Format.Format_RGBA8888).copy()
        except Exception:
            # optional, as the icons are: without Pillow the background stays plain
            _KIT_CACHE[key] = None
    return _KIT_CACHE[key]


def paint_kit(p: QPainter, area: QRectF, alpha: int, max_width: float) -> bool:
    """Draws the kit centred in `area`, as large as fits up to `max_width`."""
    w = min(area.width(), area.height() / 0.76, max_width)
    if w < 96:
        return False
    dpr = p.device().devicePixelRatioF()
    img = kit_image(w * dpr, Tokens.name, alpha)
    if img is None:
        return False
    h = w * img.height() / img.width()
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.drawImage(QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h), img)
    return True


class KitPicture(QWidget):
    """The kit on its own, above the empty state's caption: every drum that will be
    found, each in the colour its row and lane will use."""

    def __init__(self, alpha: int, max_width: int):
        super().__init__()
        self.alpha, self.max_width = alpha, max_width
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(0)

    def sizeHint(self) -> QSize:
        return QSize(self.max_width, int(self.max_width * 0.76))

    def paintEvent(self, _):
        paint_kit(QPainter(self), QRectF(self.rect()), self.alpha, self.max_width)


class KitScroll(QScrollArea):
    """A scroll area with the kit painted on its viewport, behind the rows.

    The viewport stays put when the rows scroll, so the kit does too and the rows
    pass over it, as they did in the Tk window's hand-drawn table.
    """

    def __init__(self, alpha: int, max_width: int):
        super().__init__()
        self.alpha, self.max_width = alpha, max_width

    def paintEvent(self, e):
        super().paintEvent(e)
        area = QRectF(self.viewport().rect()).adjusted(12, 12, -12, -12)
        paint_kit(QPainter(self.viewport()), area, self.alpha, self.max_width)


# ------------------------------------------------------------------- drum icons
# The drum drawn beside its name wherever the window lists instruments: the drawings
# the Tk window used, previewed in docs/kit_icons.png (drum_icons).
_ICON_CACHE: dict[tuple, QImage | None] = {}

# The drum a timeline lane's caption shows, one per lane in LANES order.
LANE_ICON = (36, 38, 42, 47, 49, 51)


def icon_image(pitch: int, px: int, theme: str) -> QImage | None:
    """A drum's icon, px square, in this theme's colour for it; None without Pillow."""
    key = (pitch, px, theme)
    if key not in _ICON_CACHE:
        try:
            import drum_icons
            img = drum_icons.draw_icon(pitch, px, ink=THEMES[theme][colour_key(pitch)])
            _ICON_CACHE[key] = QImage(img.tobytes("raw", "RGBA"), img.width, img.height,
                                      img.width * 4, QImage.Format.Format_RGBA8888).copy()
        except Exception:
            _ICON_CACHE[key] = None
    return _ICON_CACHE[key]


def paint_icon(p: QPainter, pitch: int, rect: QRectF) -> bool:
    """Draws a drum's icon into `rect` at the screen's own pixel density."""
    px = max(8, round(rect.width() * p.device().devicePixelRatioF()))
    img = icon_image(pitch, px, Tokens.name)
    if img is None:
        return False
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.drawImage(rect, img)
    return True


def paint_dot(p: QPainter, key: str, rect: QRectF, size: float, radius: float) -> None:
    """The design's plain colour mark, centred in `rect`: what shows without Pillow."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(Tokens.c(key))
    c = rect.center()
    p.drawRoundedRect(QRectF(c.x() - size / 2, c.y() - size / 2, size, size),
                      radius, radius)


class DrumIcon(QWidget):
    """A drum's icon, in the colour its lane and velocity bar are drawn in."""

    def __init__(self, pitch: int, size: int):
        super().__init__()
        self.pitch = pitch
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QPainter(self)
        if not paint_icon(p, self.pitch, QRectF(self.rect())):
            paint_dot(p, colour_key(self.pitch), QRectF(self.rect()), 8, 4)


class Toggle(QAbstractButton):
    """A 30x18 switch, the design's replacement for a checkbox."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(30, 18)

    def sizeHint(self) -> QSize:
        return QSize(30, 18)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked() and self.isEnabled()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(Tokens.c("accent") if on else Tokens.c("toggleOff"))
        p.drawRoundedRect(QRectF(0, 0, 30, 18), 9, 9)
        p.setBrush(QColor(0, 0, 0, 60))
        x = 14 if on else 2
        p.drawEllipse(QRectF(x, 3, 14, 14))
        p.setBrush(Tokens.c("knob"))
        p.drawEllipse(QRectF(x, 2, 14, 14))


class OptionRow(QWidget):
    """A label, an optional hint, and a toggle; the whole row is clickable."""

    def __init__(self, label: str, hint: str = ""):
        super().__init__()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(_label(label))
        if hint:
            text.addWidget(_label(hint, "hint"))
        lay.addLayout(text, 1)
        self.toggle = Toggle()
        lay.addWidget(self.toggle)
        self._fx = QGraphicsOpacityEffect(self)
        self._fx.setOpacity(1.0)
        self.setGraphicsEffect(self._fx)

    def mousePressEvent(self, e):
        if self.toggle.isEnabled():
            self.toggle.click()

    def set_available(self, ok: bool) -> None:
        self.toggle.setEnabled(ok)
        self._fx.setOpacity(1.0 if ok else 0.45)
        self.setCursor(Qt.CursorShape.PointingHandCursor if ok
                       else Qt.CursorShape.ArrowCursor)


class Segmented(QFrame):
    """A row of mutually exclusive pills on a sunken track."""
    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], mono: bool = False,
                 stretch: bool = False):
        super().__init__()
        self.setObjectName("seg")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for value, label in options:
            b = QPushButton(label)
            b.setObjectName("segbtn")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if mono:
                b.setProperty("mono", "true")
            if stretch:
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[value] = b
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))

    def value(self) -> str:
        return next((v for v, b in self.buttons.items() if b.isChecked()), "")

    def set_value(self, value: str) -> None:
        if value in self.buttons:
            self.buttons[value].setChecked(True)


class Meter(QWidget):
    """Five little segments, n of them lit."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.active = False
        self.setFixedSize(5 * 12 + 4 * 2, 6)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(Qt.PenStyle.NoPen)
        on = Tokens.c("accent" if self.active else "muted")
        for i in range(5):
            p.setBrush(on if i < self.n else Tokens.c("line2"))
            p.drawRoundedRect(QRectF(i * 14, 1, 12, 4), 1, 1)


class RadioDot(QWidget):
    def __init__(self):
        super().__init__()
        self.on = False
        self.setFixedSize(16, 16)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(Tokens.c("accent" if self.on else "line2"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(1, 1, 14, 14))
        if self.on:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(Tokens.c("accent"))
            p.drawEllipse(QRectF(5, 5, 6, 6))


class SeparatorCard(QFrame):
    clicked = Signal(str)

    def __init__(self, sep: Separator):
        super().__init__()
        self.sep = sep
        self.setObjectName("sepcard")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        grid = QGridLayout(self)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.dot = RadioDot()
        grid.addWidget(self.dot, 0, 0, Qt.AlignmentFlag.AlignTop)
        head = QVBoxLayout()
        head.setSpacing(2)
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name = _label(sep.name)
        name.setFont(font(13, 600))
        name_row.addWidget(name)
        name_row.addWidget(_label(sep.stems, "chip"))
        name_row.addStretch(1)
        self.est = _label("", "mono")
        name_row.addWidget(self.est)
        head.addLayout(name_row)
        head.addWidget(_label(sep.desc, "muted", wrap=True))
        grid.addLayout(head, 0, 1, 1, 2)
        meters = QHBoxLayout()
        meters.setSpacing(16)
        self.meters = []
        for caption, n in (("Quality", sep.quality), ("Speed", sep.speed)):
            box = QHBoxLayout()
            box.setSpacing(6)
            cap = _label(caption, "hint")
            cap.setFont(font(10))
            box.addWidget(cap)
            m = Meter(n)
            self.meters.append(m)
            box.addWidget(m)
            meters.addLayout(box)
        meters.addStretch(1)
        grid.addLayout(meters, 1, 1, 1, 2)
        grid.setColumnStretch(1, 1)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", "true" if on else "false")
        self.dot.on = on
        for m in self.meters:
            m.active = on
            m.update()
        self.dot.update()
        _repolish(self)

    def mousePressEvent(self, e):
        self.clicked.emit(self.sep.id)


class LogoMark(QWidget):
    """The 3x2 grid of drum colours from the design's title bar."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(26, 26)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(Tokens.c("line2"), 1))
        p.setBrush(Tokens.c("sunken"))
        p.drawRoundedRect(QRectF(0.5, 0.5, 25, 25), 7, 7)
        p.setPen(Qt.PenStyle.NoPen)
        keys = ["kick", "snare", "tom", "hat", "crash", "ride"]
        for i, k in enumerate(keys):
            col, row = i % 3, i // 3
            p.setBrush(Tokens.c(k))
            p.drawRoundedRect(QRectF(3.5 + col * 7, 7 + row * 7, 5, 5), 1, 1)


class WaveGlyph(QWidget):
    """The five-bar audio glyph in the file card."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(38, 38)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(Tokens.c("sunken"))
        p.drawRoundedRect(QRectF(0, 0, 38, 38), 8, 8)
        p.setBrush(Tokens.c("accent"))
        for i, h in enumerate((10, 20, 14, 24, 8)):
            p.drawRoundedRect(QRectF(11 + i * 3.4, 19 - h / 2, 3, h), 1.5, 1.5)


class PlayButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setObjectName("round")
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.playing = False

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(Tokens.c("text" if self.isEnabled() else "faint"))
        if self.playing:
            p.drawRoundedRect(QRectF(11, 10.5, 3, 11), 1, 1)
            p.drawRoundedRect(QRectF(17, 10.5, 3, 11), 1, 1)
        else:
            p.drawPolygon(QPolygonF([QPointF(13, 10), QPointF(23, 16), QPointF(13, 22)]))


class VelocityBar(QWidget):
    """min .. max velocity as a coloured span over 0..127."""

    def __init__(self):
        super().__init__()
        self.lo = self.hi = None
        self.key = "muted"
        self.setFixedHeight(10)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        w = self.width()
        p.setBrush(Tokens.c("sunken"))
        p.drawRoundedRect(QRectF(0, 3, w, 4), 2, 2)
        if self.lo is None:
            return
        x0 = self.lo / 127 * w
        span = max(0.015 * w, (self.hi - self.lo) / 127 * w)
        p.setBrush(Tokens.c(self.key))
        p.drawRoundedRect(QRectF(x0, 3, min(span, w - x0), 4), 2, 2)


class Dot(QWidget):
    def __init__(self, key: str, size: int = 8, radius: float = 4):
        super().__init__()
        self.key, self.r = key, radius
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(Tokens.c(self.key))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), self.r, self.r)


# --------------------------------------------------------------------- timeline
class Timeline(QWidget):
    """Time ruler, the recording's envelope, and one lane per drum.

    Clicking seeks. While a conversion runs, the part already processed is tinted.
    """
    seek = Signal(float)

    LABEL_W = 78
    TOP = 18
    AUDIO_H = 52
    LANE_H = 22

    def __init__(self):
        super().__init__()
        self.duration = 0.0
        self.env: list[float] = []
        self.hits: list[tuple[int, float, int]] = []
        self.position = 0.0
        self.progress: float | None = None     # 0..1 while converting
        self.mode = "orig"
        self.setMinimumHeight(self.TOP + self.AUDIO_H + 6 * self.LANE_H + 12)
        self.setCursor(Qt.CursorShape.IBeamCursor)

    def _area(self) -> tuple[float, float]:
        x0 = self.LABEL_W
        return x0, max(1.0, self.width() - x0 - 14)

    def mousePressEvent(self, e):
        if self.duration <= 0:
            return
        x0, w = self._area()
        t = (e.position().x() - x0) / w * self.duration
        self.seek.emit(max(0.0, min(self.duration, t)))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.mousePressEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        x0, w = self._area()
        dur = self.duration or 1.0
        muted, line = Tokens.c("muted"), Tokens.c("line")

        # lane captions
        p.setFont(font(11))
        y_audio = self.TOP
        y_lanes = self.TOP + self.AUDIO_H
        p.setPen(muted)
        p.drawText(QRectF(14, y_audio, x0 - 14, self.AUDIO_H),
                   Qt.AlignmentFlag.AlignVCenter, "Audio")
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i, (name, key) in enumerate(LANES):
            cy = y_lanes + i * self.LANE_H + self.LANE_H / 2
            mark = QRectF(10, cy - 7, 14, 14)
            if not paint_icon(p, LANE_ICON[i], mark):
                paint_dot(p, key, mark, 7, 2)
            p.setPen(muted)
            p.drawText(QRectF(29, cy - 9, x0 - 29, 18), Qt.AlignmentFlag.AlignVCenter, name)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # ruler
        p.setFont(font(10, mono=True))
        if self.duration > 0:
            step = next(s for s in (5, 10, 15, 30, 60, 120, 300, 600, 1e9)
                        if w / (self.duration / s) >= 60)
            s = 0.0
            while s <= self.duration:
                px = x0 + s / dur * w
                p.fillRect(QRectF(px, 12, 1, 5), muted)
                if 0 < s < self.duration - step / 3:
                    p.setPen(muted)
                    p.drawText(QPointF(px + 4, 11), clock(s, tenths=False))
                s += step

        # envelope
        played = x0 + self.position / dur * w
        mid = y_audio + self.AUDIO_H / 2 - 4
        p.setOpacity(0.35 if self.mode == "midi" else 1.0)
        if self.env:
            n = len(self.env)
            px = 0
            while px < w:
                a = self.env[min(n - 1, int(px / w * n))]
                hh = max(1.0, a * 21)
                p.fillRect(QRectF(x0 + px, mid - hh, 1.2, hh * 2),
                           Tokens.c("accent") if x0 + px < played else Tokens.c("wave"))
                px += 2
        else:
            p.fillRect(QRectF(x0, mid, w, 1), Tokens.c("wave"))
        p.setOpacity(1.0)

        # lane rules
        for i in range(7):
            p.fillRect(QRectF(x0, y_lanes + i * self.LANE_H - (1 if i == 6 else 0), w, 1),
                       line)

        # hits
        dim = 0.3 if self.mode == "orig" else 1.0
        cols = [Tokens.c(key) for _, key in LANES]
        for lane, t, v in self.hits:
            px = x0 + t / dur * w
            cy = y_lanes + lane * self.LANE_H + self.LANE_H / 2
            hh = 3 + v / 127 * 15
            p.setOpacity((0.35 + 0.65 * v / 127) * dim)
            p.fillRect(QRectF(px, cy - hh / 2, 1.5, hh), cols[lane])
        p.setOpacity(1.0)

        if self.progress is not None:
            p.fillRect(QRectF(x0, y_lanes, self.progress * w, 6 * self.LANE_H),
                       Tokens.c("accentSoft"))

        # playhead
        if self.duration > 0:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(QRectF(played, self.TOP, 1.5, self.height() - self.TOP),
                       Tokens.c("text"))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(Tokens.c("text"))
            p.drawPolygon(QPolygonF([QPointF(played - 4, 12), QPointF(played + 5.5, 12),
                                     QPointF(played + 0.75, 18)]))


# ------------------------------------------------------------------ result table
COLS = (("Instrument", None), ("Note", 92), ("Ch", 60), ("Hits", 56), ("Velocity", None))


class ResultRow(QFrame):
    """One kit piece: its colour, the note and channel it is written to, and what the
    last run found. The note and channel cells are the overrides."""
    edited = Signal()

    def __init__(self, pitch: int, name: str, note: int, channel: int):
        super().__init__()
        self.pitch = pitch
        self.setObjectName("row")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(31)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 0, 18, 0)
        lay.setSpacing(16)

        inst = QHBoxLayout()
        inst.setSpacing(9)
        self.icon = DrumIcon(pitch, 18)
        inst.addWidget(self.icon)
        inst.addWidget(_label(name))
        inst.addStretch(1)
        lay.addLayout(inst, 13)

        self.note = _note_box(note, COLS[1][1])
        self.note.currentIndexChanged.connect(lambda _: self.edited.emit())
        lay.addWidget(self.note)

        self.chan = _channel_box(channel, COLS[2][1])
        self.chan.currentIndexChanged.connect(lambda _: self.edited.emit())
        lay.addWidget(self.chan)

        self.hits = _label("—")
        self.hits.setFont(font(12, 600, mono=True))
        self.hits.setFixedWidth(COLS[3][1])
        self.hits.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.hits)

        vel = QHBoxLayout()
        vel.setSpacing(8)
        self.vmin = _label("", "mono")
        self.vmin.setFixedWidth(26)
        self.vmin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bar = VelocityBar()
        self.bar.key = colour_key(pitch)
        self.vmax = _label("", "mono")
        self.vmax.setFixedWidth(26)
        vel.addWidget(self.vmin)
        vel.addWidget(self.bar, 1)
        vel.addWidget(self.vmax)
        lay.addLayout(vel, 20)

    def set_result(self, hits: int | None, lo: int | None, hi: int | None) -> None:
        self.hits.setText("—" if hits is None else f"{hits:,}")
        self.vmin.setText("" if lo is None else str(lo))
        self.vmax.setText("" if hi is None else str(hi))
        self.bar.lo, self.bar.hi = lo, hi
        self.bar.update()


class MapRow(QWidget):
    """One drum in the sidebar's channel and note map: the same two overrides as the
    result row, there before any file is chosen."""
    edited = Signal()

    def __init__(self, pitch: int, name: str, note: int, channel: int):
        super().__init__()
        self.pitch = pitch
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.icon = DrumIcon(pitch, 16)
        lay.addWidget(self.icon)
        lay.addWidget(_label(name), 1)
        self.note = _note_box(note, COLS[1][1])
        self.note.currentIndexChanged.connect(lambda _: self.edited.emit())
        lay.addWidget(self.note)
        self.chan = _channel_box(channel, 52)
        self.chan.currentIndexChanged.connect(lambda _: self.edited.emit())
        lay.addWidget(self.chan)


# ------------------------------------------------------------------ the window
class Bridge(QObject):
    """Signals emitted from worker threads; Qt delivers them on the UI thread."""
    line = Signal(str)
    done = Signal(int)
    env = Signal(str, object)
    rendered = Signal(str, str, str)      # mode, path, error
    devices = Signal(object)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("drum2midi — drums to MIDI")
        self.c = Choices()
        self.run_c = self.c        # a snapshot of it once a run starts; see _start
        self.theme = "dark"
        self.phase = "empty"                 # empty | ready | converting | done
        self.proc: subprocess.Popen | None = None
        self.out_path: Path | None = None
        self.results: dict[int, tuple[int, int, int]] = {}
        self.in_reading_table = False
        self.stage_span = (0, 100)
        self.stage = ""
        self.stage_marker = ""
        self._stopping = False
        self.progress = 0.0
        self.started = 0.0
        self.took = 0.0
        self.status_override = ""
        self.live: dict[str, str] = {}
        self.audio_seconds = 0.0
        self.plan: tuple | None = None
        self.renders: dict[str, Path] = {}
        self.listen = "orig"
        self.player = None
        # true between setSource and LoadedMedia, when the player cannot seek yet
        self._loading = False
        self._want_play = False

        self.bridge = Bridge()
        self.bridge.line.connect(self._handle)
        self.bridge.done.connect(self._finish)
        self.bridge.env.connect(self._got_envelope)
        self.bridge.rendered.connect(self._got_render)
        self.bridge.devices.connect(self._got_devices)

        self.setAcceptDrops(True)
        self._build()
        self._load_settings()
        self._apply_theme()
        self._sync()
        threading.Thread(target=lambda: self.bridge.devices.emit(_device_plan()),
                         daemon=True).start()

        self.tick = QTimer(self)
        self.tick.setInterval(33)
        self.tick.timeout.connect(self._tick)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_main(), 1)
        outer.addLayout(body, 1)

    def _build_header(self) -> QWidget:
        head = QFrame()
        head.setObjectName("header")
        head.setFixedHeight(48)
        lay = QHBoxLayout(head)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(12)
        self.logo = LogoMark()
        lay.addWidget(self.logo)
        title = _label("drum2midi")
        title.setFont(font(15, 650))
        lay.addWidget(title)
        lay.addWidget(_label("drum recordings to General MIDI — measured, not guessed",
                             "muted"))
        lay.addStretch(1)

        chip = QFrame()
        chip.setObjectName("devchip")
        chip.setFixedHeight(26)
        cl = QHBoxLayout(chip)
        cl.setContentsMargins(10, 0, 10, 0)
        cl.setSpacing(7)
        self.dev_dot = Dot("faint", 6, 3)
        cl.addWidget(self.dev_dot)
        self.dev_text = _label("detecting device…", "mono")
        cl.addWidget(self.dev_text)
        lay.addWidget(chip)

        self.theme_seg = Segmented([("light", "Light"), ("dark", "Dark")])
        self.theme_seg.changed.connect(self._set_theme)
        lay.addWidget(self.theme_seg)
        return head

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("sidebar")
        scroll.setFixedWidth(344)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("sidebarBody")
        scroll.setWidget(body)
        col = QVBoxLayout(body)
        col.setContentsMargins(18, 18, 18, 24)
        col.setSpacing(26)

        # -- source
        src = QVBoxLayout()
        src.setSpacing(10)
        top = QHBoxLayout()
        top.addWidget(_section("Source"))
        top.addStretch(1)
        b_file = QPushButton("File…")
        b_file.clicked.connect(self._pick_input)
        b_folder = QPushButton("Folder…")
        b_folder.clicked.connect(self._pick_folder)
        top.addWidget(b_file)
        top.addWidget(b_folder)
        src.addLayout(top)

        self.file_card = QFrame()
        self.file_card.setObjectName("filecard")
        fc = QHBoxLayout(self.file_card)
        fc.setContentsMargins(12, 12, 12, 12)
        fc.setSpacing(12)
        fc.addWidget(WaveGlyph())
        names = QVBoxLayout()
        names.setSpacing(3)
        self.file_name = ElidedLabel()
        self.file_name.setFont(font(13, 550))
        self.file_name.setMinimumWidth(10)
        self.file_meta = _label("", "mono")
        names.addWidget(self.file_name)
        names.addWidget(self.file_meta)
        fc.addLayout(names, 1)
        src.addWidget(self.file_card)

        self.drop_small = QFrame()
        self.drop_small.setObjectName("drop")
        self.drop_small.setCursor(Qt.CursorShape.PointingHandCursor)
        dl = QVBoxLayout(self.drop_small)
        dl.setContentsMargins(12, 18, 12, 18)
        lab = _label("Drop a drum stem, a song or a folder", "muted")
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dl.addWidget(lab)
        self.drop_small.mousePressEvent = lambda e: self._pick_input()
        src.addWidget(self.drop_small)

        self.opt_song = OptionRow("This is a whole song — extract drums first")
        self.opt_song.toggle.toggled.connect(lambda v: self._set("song", v))
        src.addWidget(self.opt_song)

        save = QVBoxLayout()
        save.setSpacing(6)
        save.addWidget(_label("Save to", "muted"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.out_edit = QLineEdit()
        self.out_edit.setProperty("path", "true")
        self.out_edit.setPlaceholderText("next to the input")
        self.out_edit.textEdited.connect(lambda v: self._set("output", v, sync=False))
        row.addWidget(self.out_edit, 1)
        b_out = QPushButton("Browse…")
        b_out.setMinimumHeight(30)
        b_out.clicked.connect(self._pick_output)
        row.addWidget(b_out)
        save.addLayout(row)
        src.addLayout(save)
        col.addLayout(src)

        # -- separator
        sep = QVBoxLayout()
        sep.setSpacing(8)
        sep.addWidget(_section("Separator"))
        self.sep_cards: dict[str, SeparatorCard] = {}
        for s in SEPARATORS:
            card = SeparatorCard(s)
            card.clicked.connect(lambda v: self._set("separator", v))
            sep.addWidget(card)
            self.sep_cards[s.id] = card
        self.more_btn = QPushButton()
        self.more_btn.setObjectName("link")
        self.more_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.more_btn.clicked.connect(self._toggle_more)
        self._more = False
        sep.addWidget(self.more_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.dev_line = _label("", "muted", wrap=True)
        self.dev_line.hide()                     # until the device check reports
        sep.addWidget(self.dev_line)
        col.addLayout(sep)

        # -- options
        opt = QVBoxLayout()
        opt.setSpacing(12)
        opt.addWidget(_section("Options"))
        self.opt_rows: dict[str, OptionRow] = {}
        for key, label, hint in (
                ("fuse", "Fuse onsets found in stems", "MDX23C only"),
                ("learned", "Learned models", "velocity, pedal hi-hat"),
                ("rescue", "Rescue toms from the stem", "LarsNet, DrumSep, Hybrid"),
                ("hats", "Split hi-hat open / closed", ""),
                ("split", "One MIDI track per drum", "")):
            r = OptionRow(label, hint)
            r.toggle.toggled.connect(lambda v, k=key: self._set(k, v))
            opt.addWidget(r)
            self.opt_rows[key] = r

        qbox = QVBoxLayout()
        qbox.setSpacing(6)
        qbox.addWidget(_label("Quantize", "muted"))
        self.quant_seg = Segmented(QUANT_STEPS, mono=True, stretch=True)
        self.quant_seg.changed.connect(lambda v: self._set("quant", v))
        qbox.addWidget(self.quant_seg)
        self.opt_trip = OptionRow("Triplets", "1/8 and 1/16 only")
        self.opt_trip.toggle.toggled.connect(lambda v: self._set("triplets", v))
        qbox.addWidget(self.opt_trip)
        opt.addSpacing(6)
        opt.addLayout(qbox)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.addWidget(_label("Tempo", "muted"), 0, 0)
        grid.addWidget(_label("Thresholds", "muted"), 0, 1)
        self.tempo_edit = QLineEdit()
        self.tempo_edit.setPlaceholderText("auto  (BPM)")
        self.tempo_edit.textEdited.connect(lambda v: self._set("tempo", v, sync=False))
        self.thr_edit = QLineEdit()
        self.thr_edit.setPlaceholderText("tuned defaults")
        self.thr_edit.textEdited.connect(lambda v: self._set("thresholds", v, sync=False))
        grid.addWidget(self.tempo_edit, 1, 0)
        grid.addWidget(self.thr_edit, 1, 1)
        grid.addWidget(_label("empty = detect automatically", "hint", wrap=True), 2, 0)
        grid.addWidget(_label("e.g. kick=0.3,snare=0.25", "hint", wrap=True), 2, 1)
        opt.addLayout(grid)
        col.addLayout(opt)

        # -- channels & notes: a setting made before converting, so it sits with the
        # others. The result table edits the same map once a file is chosen.
        mp = QVBoxLayout()
        mp.setSpacing(6)
        mp.addWidget(_section("Channels & notes"))
        mp.addWidget(_label("Everything goes to channel 10, the General MIDI drum "
                            "channel. Remap notes for a sampler that does not follow "
                            "General MIDI, or give each drum its own channel.",
                            "hint", wrap=True))
        heads = QHBoxLayout()
        heads.setSpacing(8)
        heads.addWidget(_label("INSTRUMENT", "colhead"), 1)
        for title, width in (("NOTE", COLS[1][1]), ("CH", 52)):
            lab = _label(title, "colhead")
            lab.setFixedWidth(width)
            heads.addWidget(lab)
        mp.addSpacing(4)
        mp.addLayout(heads)
        self.map_rows: dict[int, MapRow] = {}
        for pitch, name in KIT:
            r = MapRow(pitch, name, self.c.notes.get(pitch, pitch),
                       self.c.channels.get(pitch, DRUM_CHANNEL))
            r.edited.connect(lambda r=r: self._row_edited(r))
            mp.addWidget(r)
            self.map_rows[pitch] = r
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.btn_map_reset = QPushButton("Reset to General MIDI")
        self.btn_map_reset.clicked.connect(self._reset_overrides)
        self.btn_spread = QPushButton("One channel each")
        self.btn_spread.setToolTip("Kick on channel 1, snare on 2 … ride on 10, "
                                   "for a sampler with one output per channel")
        self.btn_spread.clicked.connect(self._spread_channels)
        buttons.addWidget(self.btn_map_reset)
        buttons.addWidget(self.btn_spread)
        buttons.addStretch(1)
        mp.addSpacing(6)
        mp.addLayout(buttons)
        self.map_hint = _label("", "hint", wrap=True)
        mp.addWidget(self.map_hint)
        col.addLayout(mp)
        col.addStretch(1)
        return scroll

    def _build_main(self) -> QWidget:
        self.main_stack = QStackedWidget()

        # empty state
        empty = QWidget()
        el = QVBoxLayout(empty)
        el.setContentsMargins(16, 16, 16, 16)
        drop = QFrame()
        drop.setObjectName("dropBig")
        drop.setCursor(Qt.CursorShape.PointingHandCursor)
        drop.mousePressEvent = lambda e: self._pick_input()
        dl = QVBoxLayout(drop)
        dl.setSpacing(10)
        dl.addStretch(1)
        # as the Tk window's empty table did: bolder while there is nothing to read
        self.kit_picture = KitPicture(alpha=46, max_width=400)
        dl.addWidget(self.kit_picture)
        dl.addSpacing(8)
        big = _label("Drop drums here")
        big.setFont(font(20, 600))
        big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dl.addWidget(big)
        sub = _label("A drum stem gives the best result. A whole song works too —\n"
                     "turn on “extract drums first”. Drop a folder to convert every "
                     "file in it.", "muted")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dl.addWidget(sub)
        fmt = _label("WAV · FLAC · MP3 · OGG", "mono")
        fmt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dl.addSpacing(6)
        dl.addWidget(fmt)
        dl.addStretch(1)
        el.addWidget(drop)
        self.main_stack.addWidget(empty)

        # working state
        work = QWidget()
        wl = QVBoxLayout(work)
        wl.setContentsMargins(16, 16, 16, 16)
        wl.setSpacing(12)
        wl.addWidget(self._build_convert())
        self.timeline_card = self._build_timeline()
        wl.addWidget(self.timeline_card)
        wl.addWidget(self._build_results(), 1)
        self.main_stack.addWidget(work)
        return self.main_stack

    def _card(self) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        return f

    def _build_convert(self) -> QWidget:
        card = self._card()
        lay = QHBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(14)
        self.btn_run = QPushButton("Convert")
        self.btn_run.setObjectName("primary")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self._start)
        lay.addWidget(self.btn_run)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("big")
        self.btn_stop.clicked.connect(self._stop)
        lay.addWidget(self.btn_stop)

        info = QVBoxLayout()
        info.setSpacing(7)
        top = QHBoxLayout()
        self.status = _label("")
        self.status.setFont(font(13, 550))
        self.status.setMinimumWidth(10)
        self.status_right = _label("", "mono")
        self.status_right.setFont(font(12, mono=True))
        top.addWidget(self.status, 1)
        top.addWidget(self.status_right)
        info.addLayout(top)
        self.bar = ProgressBar()
        info.addWidget(self.bar)
        chips = QHBoxLayout()
        chips.setSpacing(16)
        self.chips: dict[str, tuple[QWidget, Dot, QLabel]] = {}
        for marker, label in STAGE_CHIPS:
            w = QWidget()
            cl = QHBoxLayout(w)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(5)
            d = Dot("muted", 5, 2.5)
            t = _label(label, "hint")
            cl.addWidget(d)
            cl.addWidget(t)
            chips.addWidget(w)
            self.chips[marker] = (w, d, t)
        chips.addStretch(1)
        info.addLayout(chips)
        lay.addLayout(info, 1)
        return card

    def _build_timeline(self) -> QWidget:
        card = self._card()
        col = QVBoxLayout(card)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        head = QFrame()
        head.setObjectName("cardHead")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(14, 10, 14, 10)
        hl.setSpacing(12)
        self.btn_play = PlayButton()
        self.btn_play.clicked.connect(self._toggle_play)
        hl.addWidget(self.btn_play)
        self.time_label = _label("0:00.0 / 0:00.0")
        self.time_label.setFont(font(12, mono=True))
        self.time_label.setFixedWidth(120)
        hl.addWidget(self.time_label)
        self.listen_seg = Segmented([("orig", "Original"), ("midi", "MIDI"),
                                     ("ab", "A / B")])
        self.listen_seg.setToolTip("A / B: the recording in the left ear, the MIDI "
                                   "rendered in the right — timing or wrong drums "
                                   "stand out on headphones")
        self.listen_seg.changed.connect(self._set_listen)
        self.listen_seg.set_value("orig")
        hl.addWidget(self.listen_seg)
        self.listen_note = _label("", "hint")
        hl.addWidget(self.listen_note)
        hl.addStretch(1)
        self.stat_labels: dict[str, tuple[QWidget, QLabel]] = {}
        for key, caption in (("tempo", "Tempo"), ("notes", "Notes"),
                             ("fused", "From stems"), ("ride", "Ride")):
            box = QFrame()
            box.setObjectName("stat")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(14, 0, 0, 0)
            bl.setSpacing(1)
            cap = _label(caption.upper(), "colhead")
            cap.setAlignment(Qt.AlignmentFlag.AlignRight)
            val = _label("—")
            val.setFont(font(13, 600, mono=True))
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            bl.addWidget(cap)
            bl.addWidget(val)
            hl.addWidget(box)
            self.stat_labels[key] = (box, val)
        col.addWidget(head)
        self.timeline = Timeline()
        self.timeline.seek.connect(self._seek)
        wrap = QVBoxLayout()
        wrap.setContentsMargins(0, 6, 0, 10)
        wrap.addWidget(self.timeline)
        col.addLayout(wrap)
        return card

    def _build_results(self) -> QWidget:
        card = self._card()
        col = QVBoxLayout(card)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        head = QFrame()
        head.setObjectName("cardHead")
        head.setFixedHeight(42)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(8, 0, 10, 0)
        hl.setSpacing(4)
        self.tab_group = QButtonGroup(self)
        self.tabs: dict[str, QPushButton] = {}
        for key, label in (("result", "Result"), ("log", "Log")):
            b = QPushButton(label)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self._set_tab(k))
            self.tab_group.addButton(b)
            hl.addWidget(b)
            self.tabs[key] = b
        hl.addStretch(1)
        self.table_hint = _label("", "muted")
        hl.addWidget(self.table_hint)
        hl.addSpacing(8)
        self.btn_reset = QPushButton("Reset to General MIDI")
        self.btn_reset.clicked.connect(self._reset_overrides)
        hl.addWidget(self.btn_reset)
        self.btn_open = QPushButton("Show file")
        self.btn_open.clicked.connect(self._reveal)
        hl.addWidget(self.btn_open)
        col.addWidget(head)

        self.result_stack = QStackedWidget()
        table = QWidget()
        tl = QVBoxLayout(table)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        heads = QFrame()
        heads.setObjectName("colheads")
        heads.setFixedHeight(30)
        hh = QHBoxLayout(heads)
        hh.setContentsMargins(18, 0, 18, 0)
        hh.setSpacing(16)
        for i, (title, width) in enumerate(COLS):
            lab = _label(title.upper(), "colhead")
            if width:
                lab.setFixedWidth(width)
            if title == "Hits":
                lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            hh.addWidget(lab, 13 if i == 0 else 20 if width is None else 0)
        tl.addWidget(heads)
        # faint once there are rows to read over it
        scroll = KitScroll(alpha=24, max_width=420)
        self.table_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows = QWidget()
        self.rows_layout = QVBoxLayout(rows)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(0)
        self.rows: dict[int, ResultRow] = {}
        for pitch, name in KIT:
            self._add_row(pitch, name)
        self.rows_layout.addStretch(1)
        scroll.setWidget(rows)
        tl.addWidget(scroll, 1)
        self.result_stack.addWidget(table)

        logw = QWidget()
        ll = QVBoxLayout(logw)
        ll.setContentsMargins(10, 10, 10, 10)
        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        ll.addWidget(self.log)
        self.result_stack.addWidget(logw)
        col.addWidget(self.result_stack, 1)
        self._set_tab("result")
        return card

    def _add_row(self, pitch: int, name: str) -> ResultRow:
        row = ResultRow(pitch, name, self.c.notes.get(pitch, pitch),
                        self.c.channels.get(pitch, DRUM_CHANNEL))
        row.edited.connect(lambda r=row: self._row_edited(r))
        self.rows_layout.insertWidget(len(self.rows), row)
        self.rows[pitch] = row
        return row

    # ---------------------------------------------------------------- state sync
    def _set(self, key: str, value, sync: bool = True) -> None:
        setattr(self.c, key, value)
        if sync:
            self._sync()

    def _toggle_more(self) -> None:
        self._more = not self._more
        self._sync()

    def _sync(self) -> None:
        """Pushes Choices and the phase into every widget that shows them."""
        c = self.c
        has_input = bool(c.input) and Path(c.input).exists()
        if self.phase == "empty" and has_input:
            self.phase = "ready"
        self.main_stack.setCurrentIndex(0 if self.phase == "empty" else 1)
        self.file_card.setVisible(has_input)
        self.drop_small.setVisible(not has_input)
        if self.out_edit.text() != c.output:
            self.out_edit.setText(c.output)
        self.out_edit.setCursorPosition(len(c.output))

        # a hidden separator stays visible while it is the one selected
        chosen_more = any(s.more and s.id == c.separator for s in SEPARATORS)
        gpu = bool(self.plan and self.plan[0])
        for s in SEPARATORS:
            card = self.sep_cards[s.id]
            card.setVisible(not s.more or self._more or s.id == c.separator)
            card.set_selected(s.id == c.separator)
            card.est.setText("≈ " + human_duration(
                estimate_seconds(s.id, self.audio_seconds, gpu))
                if self.audio_seconds else "")
        hidden = sum(1 for s in SEPARATORS if s.more) - (1 if chosen_more else 0)
        self.more_btn.setText("Hide other separators" if self._more
                              else f"Show {hidden} more separator{'s' * (hidden != 1)}")
        self.more_btn.setVisible(self._more or hidden > 0)

        for key, row in self.opt_rows.items():
            row.toggle.blockSignals(True)
            avail = (fuse_available(c.separator) if key == "fuse"
                     else rescue_available(c.separator) if key == "rescue" else True)
            row.set_available(avail)
            # an unavailable option shows off, but the preference is kept for later
            row.toggle.setChecked(bool(getattr(c, key)) and avail)
            row.toggle.blockSignals(False)
        for r, v in ((self.opt_song, c.song), (self.opt_trip, c.triplets)):
            r.toggle.blockSignals(True)
            r.toggle.setChecked(v)
            r.toggle.blockSignals(False)
        self.opt_trip.set_available(c.quant in TRIPLETS)
        self.quant_seg.set_value(c.quant)
        if self.tempo_edit.text() != c.tempo:
            self.tempo_edit.setText(c.tempo)
        if self.thr_edit.text() != c.thresholds:
            self.thr_edit.setText(c.thresholds)
        self._refresh_run()

    def _refresh_run(self) -> None:
        running = self.phase == "converting"
        done = self.phase == "done"
        self.btn_run.setEnabled(not running and self.phase != "empty")
        self.btn_stop.setEnabled(running)
        single = self._single_file()
        self.btn_open.setEnabled(bool(self.out_path and self.out_path.exists()) and not running)

        # status line
        right = ""
        if running:
            pct = round(self.progress)
            text = self.status_override or self.stage or "Starting…"
            elapsed = time.monotonic() - self.started
            right = f"{pct}%"
            if self.progress > 12:
                left = elapsed / self.progress * (100 - self.progress)
                right += f" · {human_duration(left)} left"
        elif done:
            total = sum(h for h, _, _ in self.results.values())
            text = (f"Done · {total:,} notes written" if total and single
                    else "Done" if single else "Done · folder converted")
            right = f"took {clock(self.took, tenths=False)}"
        elif self.status_override:
            text = self.status_override
        else:
            text = "Ready"
            if self.audio_seconds:
                gpu = bool(self.plan and self.plan[0])
                est = estimate_seconds(self.c.separator, self.audio_seconds, gpu)
                text += (f" · {self.audio_seconds / 60:.1f} min of audio → roughly "
                         f"{human_duration(est)}")
        self.status.setText(text)
        self.status_right.setText(right)
        self.bar.value = self.progress if (running or done) else 0
        self.bar.update()

        # stage chips
        order = [m for m, _ in STAGE_CHIPS]
        cur = order.index(self.stage_marker) if self.stage_marker in order else -1
        for m, (w, d, t) in self.chips.items():
            w.setVisible(m != "[0/4]" or self.c.song)
            i = order.index(m)
            key = ("text" if done or (running and i < cur)
                   else "accent" if running and i == cur else "muted")
            d.key = key
            d.update()
            t.setStyleSheet(f"color: {Tokens.t[key]}; font-size: 11px;")

        # stats
        for key, (box, val) in self.stat_labels.items():
            v = self.live.get(key)
            val.setText(v or "—")
            box.setVisible(key in ("tempo", "notes") or bool(v))

        # timeline and listening
        self.timeline_card.setVisible(single)
        self.timeline.progress = (self.progress / 100) if running else None
        for mode, b in self.listen_seg.buttons.items():
            b.setEnabled(mode == "orig" or (done and bool(self.out_path)
                                            and self.out_path.is_file()))
        self.timeline.mode = self.listen
        self.timeline.update()

        # results
        notes, chans = overrides(self.c)
        n = len(notes) + len(chans)
        differ = (f"{_count(len(notes), 'note')}, {_count(len(chans), 'channel')} "
                  f"differ from General MIDI")
        if done and n and self._overrides_changed_since_run():
            self.table_hint.setText("Changed — convert again to apply")
            self.map_hint.setText("Changed since the last run — convert again to apply")
        elif n:
            self.table_hint.setText(differ)
            self.map_hint.setText(differ)
        else:
            self.table_hint.setText("Click a note or channel to change it")
            self.map_hint.setText("General MIDI notes, all on channel 10")
        self.btn_reset.setVisible(bool(n))
        self.btn_map_reset.setEnabled(bool(n))

    def _single_file(self) -> bool:
        return bool(self.c.input) and Path(self.c.input).is_file()

    def _overrides_changed_since_run(self) -> bool:
        return getattr(self, "_run_overrides", None) not in (None, overrides(self.c))

    # ---------------------------------------------------------------- theming
    def _set_theme(self, name: str) -> None:
        self.theme = name
        self._apply_theme()
        self._save_settings()

    def _apply_theme(self) -> None:
        Tokens.name = self.theme
        Tokens.t = THEMES[self.theme]
        app = QApplication.instance()
        app.setStyleSheet(stylesheet())
        # Qt 6.8+ also takes the native title bar dark or light with it
        try:
            hints = app.styleHints()
            hints.setColorScheme(Qt.ColorScheme.Dark if self.theme == "dark"
                                 else Qt.ColorScheme.Light)
        except Exception:
            pass
        self.theme_seg.set_value(self.theme)
        for w in self.findChildren(QWidget):
            w.update()
        self._refresh_run()

    # ---------------------------------------------------------------- input
    def _pick_input(self) -> None:
        start = self._last_dir()
        p, _ = QFileDialog.getOpenFileName(
            self, "Choose audio", start,
            "Audio (*.wav *.mp3 *.flac *.aiff *.aif *.m4a *.ogg *.opus);;All files (*)")
        if p:
            self._set_input(Path(p))

    def _pick_folder(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Choose a folder of audio files",
                                             self._last_dir())
        if p:
            self._set_input(Path(p))

    def _pick_output(self) -> None:
        if self.c.input and Path(self.c.input).is_dir():
            p = QFileDialog.getExistingDirectory(self, "Save MIDI files to",
                                                 self.c.output or self._last_dir())
        else:
            p, _ = QFileDialog.getSaveFileName(self, "Save MIDI",
                                               self.c.output or self._last_dir(),
                                               "MIDI (*.mid)")
        if p:
            self.c.output = p
            self._sync()

    def _last_dir(self) -> str:
        return getattr(self, "_last", "") or str(Path.home())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self._set_input(Path(url.toLocalFile()))
                break

    def _set_input(self, path: Path) -> None:
        if self.phase == "converting":
            return
        self._stop_playback()
        self.c.input = str(path)
        self.c.output = str(path / "midi" if path.is_dir() else path.with_suffix(".mid"))
        self._last = str(path.parent)
        self.phase = "ready"
        self.out_path = None
        self.results = {}
        self.live = {}
        self.progress = 0
        self.status_override = ""
        self.renders = {}
        self.timeline.hits = []
        self.timeline.env = []
        self.timeline.position = 0.0
        for r in self.rows.values():
            r.set_result(None, None, None)
        self.listen = "orig"
        self.listen_seg.set_value("orig")
        self._describe_input(path)
        self._sync()
        if path.is_file():
            threading.Thread(target=self._envelope_worker, args=(path,),
                             daemon=True).start()

    def _describe_input(self, path: Path) -> None:
        self.audio_seconds = 0.0
        self.file_name.setText(path.name)
        self.file_name.setToolTip(str(path))
        try:
            import soundfile as sf
            if path.is_dir():
                files = audio_files(path)
                if not files:
                    self.file_meta.setText("no audio files in this folder")
                    return
                self.audio_seconds = sum(sf.info(str(f)).duration for f in files)
                self.file_meta.setText(f"{len(files)} files · "
                                       f"{clock(self.audio_seconds, tenths=False)}")
            else:
                info = sf.info(str(path))
                self.audio_seconds = info.duration
                ch = {1: "mono", 2: "stereo"}.get(info.channels, f"{info.channels} ch")
                self.file_meta.setText(f"{clock(info.duration, tenths=False)} · "
                                       f"{info.samplerate / 1000:g} kHz · {ch}")
        except Exception:
            self.file_meta.setText(path.suffix.lstrip(".").upper() or "folder")
        self.timeline.duration = self.audio_seconds if path.is_file() else 0.0
        self._update_time()

    def _envelope_worker(self, path: Path) -> None:
        try:
            env = envelope(path)
        except Exception:
            env = []
        self.bridge.env.emit(str(path), env)

    def _got_envelope(self, path: str, env) -> None:
        if path == self.c.input:
            self.timeline.env = env
            self.timeline.update()

    def _got_devices(self, plan) -> None:
        self.plan = plan
        sep, _ = plan
        text, ok = device_line(plan)
        self.dev_line.setText(text)
        self.dev_line.show()
        self.dev_line.setProperty("role", "ok" if ok else "muted")
        _repolish(self.dev_line)
        self.dev_text.setText(sep or "CPU")
        self.dev_dot.key = "ok" if ok else "faint"
        self.dev_dot.update()
        self._sync()

    # ---------------------------------------------------------------- overrides
    def _row_edited(self, row) -> None:
        """An edit in either editor, the sidebar map or the result table."""
        self.c.notes[row.pitch] = row.note.currentIndex()
        self.c.channels[row.pitch] = row.chan.currentIndex()
        self._show_overrides()
        self._refresh_run()

    def _show_overrides(self) -> None:
        """Puts the note and channel map from Choices into both editors of it."""
        for rows in (self.rows, self.map_rows):
            for pitch, row in rows.items():
                for box, value in ((row.note, self.c.notes.get(pitch, pitch)),
                                   (row.chan, self.c.channels.get(pitch, DRUM_CHANNEL))):
                    if box.currentIndex() != value:
                        box.blockSignals(True)
                        box.setCurrentIndex(value)
                        box.blockSignals(False)

    def _reset_overrides(self) -> None:
        for pitch, _ in KIT:
            self.c.notes[pitch] = pitch
            self.c.channels[pitch] = DRUM_CHANNEL
        self._show_overrides()
        self._refresh_run()

    def _spread_channels(self) -> None:
        """Kick on channel 1 (0 to the CLI), snare on 2, and so on in kit order."""
        for i, (pitch, _) in enumerate(KIT):
            self.c.channels[pitch] = i % 16
        self._show_overrides()
        self._refresh_run()

    def _set_tab(self, key: str) -> None:
        self.tabs[key].setChecked(True)
        self.result_stack.setCurrentIndex(0 if key == "result" else 1)

    # ---------------------------------------------------------------- running
    def _start(self) -> None:
        src = Path(self.c.input)
        if not self.c.input or not src.exists():
            QMessageBox.warning(self, "No file", "Choose an existing audio file or folder.")
            return
        if not self.c.output:
            self.c.output = str(src / "midi" if src.is_dir() else src.with_suffix(".mid"))
        try:
            tempo = tempo_value(self.c.tempo)
        except ValueError:
            QMessageBox.warning(
                self, "Invalid tempo",
                "Enter a tempo between 20 and 400 BPM, or leave the field empty "
                "to detect it automatically.")
            return
        if quant_value(self.c) and tempo is None:
            if QMessageBox.question(
                    self, "Quantizing without a set tempo",
                    "The quantization grid is derived from the tempo, which will be "
                    "detected automatically. Tempo detectors often land on double or "
                    "half the real value, and then quantizing ruins the groove.\n\n"
                    "If you know the project tempo, type it in.\n\nContinue?",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            ) != QMessageBox.StandardButton.Ok:
                return

        self._stop_playback()
        self.listen = "orig"
        self.listen_seg.set_value("orig")
        self.renders = {}
        self.out_path = Path(self.c.output)
        self.results = {}
        self.live = {}
        self.timeline.hits = []
        for r in self.rows.values():
            r.set_result(None, None, None)
        self.log.clear()
        self.in_reading_table = False
        self.progress = 0
        self.stage_span = (0, 100)
        self.stage = ""
        self.stage_marker = ""
        self.status_override = ""
        self._stopping = False
        self.phase = "converting"
        self.started = time.monotonic()
        self._run_overrides = overrides(self.c)
        # The run's own copy of the settings. The summary and the MIDI are read back
        # with it, not with whatever the pickers say when the run ends: a note changed
        # during a two-minute conversion otherwise told the read-back that the drum
        # had moved, and every one of its hits vanished from the lane and the count.
        self.run_c = copy.deepcopy(self.c)
        self._save_settings()

        cmd = build_command(self.c)
        self._emit("$ " + " ".join(cmd))
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

        def worker():
            try:
                flags = NO_WINDOW
                if os.name == "nt":
                    flags |= subprocess.CREATE_NEW_PROCESS_GROUP
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", env=env,
                    cwd=str(ROOT), creationflags=flags)
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.bridge.line.emit(line)
                code = self.proc.wait()
            except Exception as exc:  # noqa: BLE001
                self.bridge.line.emit(f"ERROR: {exc}\n")
                code = -1
            self.bridge.done.emit(code)

        threading.Thread(target=worker, daemon=True).start()
        self._sync()
        self.tick.start()

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self._stopping = True
            self.proc.terminate()
            self.status_override = "Stopping…"
            self._refresh_run()

    def _emit(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip("\n"))

    def _handle(self, line: str) -> None:
        # tqdm progress from the separators would flood the log
        if "it/s" in line or "s/it" in line:
            return

        # Separation is 90% of a run, so its own percentage is mapped into the slice of
        # the bar that stage owns. Without this the bar sits still for minutes.
        match = _SEPARATING.search(line)
        if match:
            inner = int(match.group(1))
            lo, hi = self.stage_span
            self.progress = lo + (hi - lo) * inner / 100
            self.status_override = "Separating stems…"
            self._refresh_run()
            return

        self._emit(line)
        self._update_live(line)
        for marker, pct in STAGES.items():
            if marker in line:
                self.progress = pct
                self.stage_span = STAGE_SPANS.get(marker, (pct, pct))
                self.stage_marker = marker
                self.stage = line.split("]", 1)[-1].strip().rstrip(".").strip() + "…"
                self.status_override = ""
        stripped = line.rstrip("\n")
        if stripped.startswith("instrument"):
            self.in_reading_table = True
            self.results = {}
        elif self.in_reading_table:
            if set(stripped.strip()) <= {"-"} and stripped.strip():
                pass
            elif (row := summary_row(stripped)) is not None:
                name, hits, lo, hi = row
                pitch = row_pitch(name, self.run_c)
                if pitch is not None and pitch >= 0:
                    self.results[pitch] = (hits, lo, hi)
                    target = self.rows.get(pitch) or self._add_row(pitch, name)
                    target.set_result(hits, lo, hi)
            elif not stripped.strip():
                self.in_reading_table = False
        self._refresh_run()

    def _update_live(self, line: str) -> None:
        """Pulls the run's findings out of the log and into the stats."""
        for key, pattern in _LIVE.items():
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1)
            if key == "tempo":
                self.live["tempo"] = f"{float(value):.0f} BPM"
            elif key == "notes":
                self.live["notes"] = f"{int(value):,}"
            elif key == "fused":
                self.live["fused"] = f"+{value}"
            elif key == "ride":
                self.live["ride"] = value

    def _finish(self, code: int) -> None:
        self.proc = None
        self.took = time.monotonic() - self.started
        ok = code == 0 and self.out_path is not None and self.out_path.exists()
        self.status_override = ""
        if ok:
            self.phase = "done"
            self.progress = 100
            if self.results:
                self.live["notes"] = f"{sum(h for h, _, _ in self.results.values()):,}"
            if self.out_path.is_file():
                try:
                    self.timeline.hits = read_hits(self.out_path, self.run_c)
                except Exception as exc:  # noqa: BLE001
                    self._emit(f"(timeline: could not read the MIDI back: {exc})")
        else:
            self.phase = "ready"
            self.progress = 0
            if self._stopping:
                self.status_override = "Stopped"
            else:
                self.status_override = ("Failed — see the Log tab" if code
                                        else "Finished, but no MIDI was written")
                self._set_tab("log")
        if not self._is_playing():
            self.tick.stop()
        self._sync()

    def _reveal(self) -> None:
        """Shows the result in the system file manager."""
        if not self.out_path or not self.out_path.exists():
            return
        path = self.out_path
        try:
            if sys.platform == "win32":
                if path.is_dir():
                    subprocess.Popen(["explorer", str(path)])
                else:
                    subprocess.Popen(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                # -R reveals the file inside its folder, matching Explorer's behaviour
                subprocess.Popen(["open", "-R", str(path)] if path.is_file()
                                 else ["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path if path.is_dir()
                                                 else path.parent)])
        except (OSError, FileNotFoundError):
            QMessageBox.information(self, "File", str(path))

    # ---------------------------------------------------------------- listening
    def _ensure_player(self) -> bool:
        if self.player is not None:
            return True
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        except ImportError:
            self.listen_note.setText("playback needs QtMultimedia")
            return False
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.mediaStatusChanged.connect(self._media_status)
        self.player.playbackStateChanged.connect(lambda _: self._play_state())
        self.player.errorOccurred.connect(
            lambda _e, msg: self.listen_note.setText(msg[:60]))
        return True

    def _source_for(self, mode: str) -> Path | None:
        if mode == "orig":
            return Path(self.c.input) if self._single_file() else None
        return self.renders.get(mode)

    def _set_listen(self, mode: str) -> None:
        was_playing = self._is_playing()
        self.listen = mode
        self.listen_note.setText("")
        self.timeline.mode = mode
        self.timeline.update()
        if mode != "orig" and mode not in self.renders:
            self._render(mode, then_play=was_playing)
            if self.player is not None:
                self.player.pause()
            return
        self._load_source(play=was_playing)

    def _load_source(self, play: bool) -> None:
        src = self._source_for(self.listen)
        if src is None or not self._ensure_player():
            return
        url = QUrl.fromLocalFile(str(src))
        if self.player.source() != url:
            self._loading = True
            self._want_play = play
            self.player.setSource(url)
        elif self._loading:
            # still loading: remember the request; the load applies it
            self._want_play = self._want_play or play
        elif play:
            self.player.setPosition(int(self.timeline.position * 1000))
            self.player.play()

    def _media_status(self, status) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.MediaStatus.LoadedMedia and self._loading:
            # Where the playhead is *now*, not where it was when loading began: a seek
            # made while a render was loading used to be undone by the stale position.
            self.player.setPosition(int(self.timeline.position * 1000))
            self._loading = False
            if self._want_play:
                self.player.play()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.timeline.position = 0.0
            self._update_time()

    def _render(self, mode: str, then_play: bool) -> None:
        """Renders the MIDI (or the A/B pair) to audio once, into a temp cache."""
        midi = self.out_path
        if not (midi and midi.is_file()):
            return
        why = render_missing()
        if why:
            # Asked on every click rather than once, so installing the renderer while
            # the window is open is enough; no restart.
            self.listen = "orig"
            self.listen_seg.set_value("orig")
            self.timeline.mode = "orig"
            self.timeline.update()
            self.listen_note.setText("MIDI playback needs FluidSynth — "
                                     "python setup_env.py --with-render")
            self.listen_note.setToolTip(why)
            if why != getattr(self, "_logged_missing", None):
                self._logged_missing = why
                self._emit(f"cannot play the MIDI: {why}")
            return
        key = hashlib.sha1(f"{midi}|{midi.stat().st_mtime}|{self.c.input}"
                           .encode()).hexdigest()[:10]
        cache = Path(tempfile.gettempdir()) / "drum2midi_listen"
        cache.mkdir(exist_ok=True)
        wav = cache / f"{midi.stem[:40]}_{key}_{mode}.wav"
        self.listen_note.setText("rendering…")
        self._want_play = then_play

        def worker():
            if wav.exists():
                self.bridge.rendered.emit(mode, str(wav), "")
                return
            cmd = [PY, str(ROOT / "render_midi.py"), str(midi), "-o", str(wav)]
            if mode == "ab":
                cmd += ["--against", self.c.input]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=str(ROOT), creationflags=NO_WINDOW)
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            self.bridge.rendered.emit(mode, str(wav) if wav.exists() else "", out)

        threading.Thread(target=worker, daemon=True).start()

    def _got_render(self, mode: str, path: str, output: str) -> None:
        if output:
            self._emit(output)
        if not path:
            self.listen_note.setText("render failed — see Log")
            self.listen_note.setToolTip(output[-400:])
            return
        self.renders[mode] = Path(path)
        if self.listen == mode:
            self.listen_note.setText("left: original · right: MIDI" if mode == "ab" else "")
            self._load_source(play=self._want_play)

    def _toggle_play(self) -> None:
        if self._is_playing():
            self.player.pause()
            return
        if self.listen != "orig" and self.listen not in self.renders:
            self._render(self.listen, then_play=True)
            return
        self._load_source(play=True)

    def _is_playing(self) -> bool:
        if self.player is None:
            return False
        from PySide6.QtMultimedia import QMediaPlayer
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def _play_state(self) -> None:
        playing = self._is_playing()
        self.btn_play.playing = playing
        self.btn_play.update()
        if playing:
            self.tick.start()
        elif self.phase != "converting":
            self.tick.stop()

    def _stop_playback(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
            self._loading = False

    def _seek(self, t: float) -> None:
        self.timeline.position = t
        if (self.player is not None and self.player.source().isValid()
                and not self._loading):
            self.player.setPosition(int(t * 1000))
        self._update_time()
        self.timeline.update()

    def _tick(self) -> None:
        if self._is_playing():
            self.timeline.position = self.player.position() / 1000
            # FluidSynth writes ~7 s of silence after the last note has died away, so
            # the MIDI and A/B renders outlast the recording; without this the playhead
            # ran off the end of the timeline through that tail
            if 0 < self.timeline.duration <= self.timeline.position:
                self.player.pause()
                self.player.setPosition(0)
                self.timeline.position = 0.0
            self._update_time()
            self.timeline.update()
        if self.phase == "converting":
            self._refresh_run()

    def _update_time(self) -> None:
        self.time_label.setText(f"{clock(self.timeline.position)} / "
                                f"{clock(self.timeline.duration)}")

    # ---------------------------------------------------------------- settings
    def _save_settings(self) -> None:
        c = self.c
        try:
            SETTINGS.write_text(json.dumps({
                "sep": c.separator, "song": c.song, "learned": c.learned,
                "fuse": c.fuse, "rescue": c.rescue, "hats": c.hats, "split": c.split,
                "quant": c.quant, "triplets": c.triplets, "thresholds": c.thresholds,
                "tempo": c.tempo, "theme": self.theme,
                "overrides": {str(p): [str(c.notes.get(p, p)),
                                       str(c.channels.get(p, DRUM_CHANNEL))]
                              for p, _ in KIT},
                "last_dir": getattr(self, "_last", ""),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_settings(self) -> None:
        try:
            d = json.loads(SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        c = self.c
        if d.get("sep") in {s.id for s in SEPARATORS}:
            c.separator = d["sep"]
        for key, attr in (("song", "song"), ("learned", "learned"), ("fuse", "fuse"),
                          ("rescue", "rescue"), ("hats", "hats"), ("split", "split"),
                          ("triplets", "triplets")):
            if isinstance(d.get(key), bool):
                setattr(c, attr, d[key])
        quant = str(d.get("quant", ""))
        # older settings stored the triplet grids as their own values
        inverse = {v: k for k, v in TRIPLETS.items()}
        if quant in inverse:
            c.quant, c.triplets = inverse[quant], True
        elif quant in {v for v, _ in QUANT_STEPS}:
            c.quant = quant
        c.tempo = str(d.get("tempo", ""))
        c.thresholds = str(d.get("thresholds", ""))
        for key, pair in (d.get("overrides") or {}).items():
            try:
                pitch, note, chan = int(key), int(pair[0]), int(pair[1])
            except (ValueError, TypeError, IndexError):
                continue
            if pitch in self.rows and 0 <= note <= 127 and 0 <= chan <= 15:
                c.notes[pitch], c.channels[pitch] = note, chan
        self._show_overrides()
        if d.get("theme") in THEMES:
            self.theme = d["theme"]
        self._last = d.get("last_dir", "") or ""

    def closeEvent(self, e):
        if self.proc and self.proc.poll() is None:
            if QMessageBox.question(
                    self, "Conversion running",
                    "A conversion is still running. Stop it and close?"
            ) != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.proc.terminate()
        self._stop_playback()
        self._save_settings()
        super().closeEvent(e)


class ProgressBar(QWidget):
    def __init__(self):
        super().__init__()
        self.value = 0.0
        self.setFixedHeight(6)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        r = QRectF(0, 0, self.width(), 6)
        path = QPainterPath()
        path.addRoundedRect(r, 3, 3)
        p.setClipPath(path)
        p.fillRect(r, Tokens.c("sunken"))
        p.setBrush(Tokens.c("accent"))
        p.drawRoundedRect(QRectF(0, 0, self.width() * self.value / 100, 6), 3, 3)


# -------------------------------------------------------------------- startup
def _set_taskbar_identity() -> None:
    """Declares an Application User Model ID only when asked.

    Declaring one redirects the taskbar to look for a shortcut carrying that ID and to
    take the icon from there, instead of from the window. That indirection is useful
    for grouping, but it defeats the per-window icon this program sets directly -- so
    it is off unless DRUM2MIDI_APPID is set.
    """
    if os.name != "nt":
        return
    app_id = os.environ.get("DRUM2MIDI_APPID")
    if not app_id:
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def make_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    Tokens.ui = _first_installed(UI_FAMILIES, app.font().family())
    Tokens.mono = _first_installed(MONO_FAMILIES, "monospace")
    app.setFont(font(13))
    for icon in (ROOT / "docs" / "drum2midi.ico", ROOT / "docs" / "logo_256.png"):
        if icon.exists():
            app.setWindowIcon(QIcon(str(icon)))
            break
    return app


def main() -> None:
    _set_taskbar_identity()
    app = make_app()
    win = Window()
    win.resize(1320, 860)
    win.setMinimumSize(1040, 700)
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        win._set_input(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
