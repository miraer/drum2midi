"""Native window for drum2midi.

Launch without a console:  drum2midi.bat  (or pythonw drum2midi_gui.pyw)

Conversion runs in a separate process; its output is read line by line on a background
thread and handed to the UI through a queue, so the window stays responsive and the run
can be cancelled.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "gui_settings.json"
PY = sys.executable
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

sys.path.insert(0, str(ROOT))
try:
    import theme
except ImportError:  # the GUI must still open without the theme module
    theme = None

STAGES = {"[0/4]": 10, "[1/4]": 30, "[2/4]": 60, "[3/4]": 85, "[4/4]": 95}

# How much of the bar each stage owns, so a stage that reports its own progress can
# fill its slice smoothly instead of jumping.
STAGE_SPANS = {"[0/4]": (10, 28), "[1/4]": (30, 58), "[2/4]": (60, 84),
               "[3/4]": (85, 94), "[4/4]": (95, 99)}

# the pipeline forwards the separator's percentage as "      separating: 42%"
_SEPARATING = re.compile(r"separating:\s*(\d{1,3})%")

# Findings the pipeline already prints; surfaced live instead of left in the log.
_LIVE = {
    "tempo": re.compile(r"Writing MIDI at ([\d.]+) BPM"),
    "notes": re.compile(r"(\d+) hits detected"),
    "fused": re.compile(r"stem fusion: \+(\d+)"),
    "ride": re.compile(r"ride/crash: (\d+) hits reassigned"),
}

# the early per-class breakdown, printed as soon as ADTOF returns
_FOUND = re.compile(r"found ([A-Za-z0-9 \-]+): (\d+)$")

# drum2midi already prints General MIDI names, so no translation table is needed
RU = {}


_NOTE_LETTERS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Cubase, Logic, Ableton and Reaper all call MIDI 60 "C3"; this follows them, as the
# rest of the project does.
_MIDDLE_C_OCTAVE = 3


def note_name(pitch: int) -> str:
    """36 -> "C1". The number is what gets written; the name is for the human."""
    if not 0 <= pitch <= 127:
        return "—"
    return f"{_NOTE_LETTERS[pitch % 12]}{pitch // 12 - (5 - _MIDDLE_C_OCTAVE)}"


def note_name_from_text(text: str) -> str:
    try:
        return note_name(int(text.strip()))
    except (ValueError, TypeError):
        return "—"


def _accelerator() -> str | None:
    """Name of the GPU the separator will use, or None when it will run on the CPU.

    Imported lazily and defensively: the GUI must still open when torch is missing
    or a driver is broken.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from devices import SEPARATOR, describe_device, pick_devices
        dev = pick_devices("auto")[SEPARATOR]
        return None if dev == "cpu" else describe_device(dev)
    except Exception:
        return None


SEPARATORS = [
    ("uvr", "MDX23C \u2014 6 stems with ride/crash, best quality"),
    ("larsnet", "LarsNet \u2014 5 stems, no ride. Fast: about 10 seconds per minute "
                "of audio, but weaker on ghost notes and cymbals"),
    ("none", "No separation \u2014 fastest, but velocity is a flat 100"),
    ("drumsep", "DrumSep \u2014 4 stems, not recommended"),
    ("hybrid", "Hybrid \u2014 not recommended"),
]

QUANTIZE = [("", "no quantization"), ("4", "1/4"), ("8", "1/8"),
            ("16", "1/16"), ("32", "1/32"), ("12", "1/8 triplets"), ("24", "1/16 triplets")]

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
# given the right icon and colour. Both kick numbers appear because 35 is the internal
# class and 36 is what gets written.
LABEL_TO_PITCH = {
    "kick": 36, "snare": 38, "snare rim": 40,
    "floor tom": 43, "floor tom hi": 43, "low tom": 45, "mid tom": 47,
    "hi-mid tom": 48, "high tom": 50,
    "hi-hat closed": 42, "hi-hat pedal": 44, "hi-hat open": 46,
    "crash": 49, "ride": 51,
    # coarse class names used by the early preview, before articulation is decided
    "toms": 47, "hi-hat": 42, "cymbals": 49,
}


class ResultTable(tk.Canvas):
    """The per-drum summary, drawn by hand so the kit can sit behind the rows.

    A ttk.Treeview was used first, but its rows paint an opaque background, so any
    picture behind them is only visible in the gap below the last row. Drawing the
    table onto a canvas is a little more code and gives the rows a real backdrop, an
    empty state, and control over spacing.
    """

    HEADINGS = ("Instrument", "Hits", "Velocity min", "Velocity max")
    ROW_H = 23
    HEAD_H = 26

    def __init__(self, master: tk.Widget, app: "App"):
        super().__init__(master, highlightthickness=0, borderwidth=0, height=150)
        self.app = app
        self.rows: list[tuple] = []
        self.scrollbar = None
        self._bg_photo = None
        self._bg_dy = 0
        self._redraw_job = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<MouseWheel>", self._wheel)

    # -- contents --------------------------------------------------------------
    def clear(self) -> None:
        self.rows = []
        self.yview_moveto(0)
        self.redraw()

    def add(self, label: str, hits, lo, hi) -> None:
        self.rows.append((label, str(hits), str(lo), str(hi)))
        self.redraw()

    # -- scrolling -------------------------------------------------------------
    def scroll(self, *args) -> None:
        self.yview(*args)
        self._place_background()

    def _wheel(self, event) -> None:
        self.yview_scroll(-1 if event.delta > 0 else 1, "units")
        self._place_background()

    def _place_background(self) -> None:
        """Keeps the kit still while the rows scroll past it."""
        if not self.find_withtag("bg"):
            return
        self.coords("bg", self.canvasx(self.winfo_width() / 2),
                    self.canvasy(self.winfo_height() / 2) + self._bg_dy)

    # -- drawing ---------------------------------------------------------------
    def set_background(self, photo) -> None:
        self._bg_photo = photo
        self.redraw()

    def redraw(self) -> None:
        p = getattr(self.app, "_palette", None)
        if p is None:
            return
        self.delete("all")
        self.configure(background=p.base)
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())

        if self._bg_photo is not None:
            # when the table is empty the picture and its caption are centred as one
            # block, so the caption does not fall off the bottom
            self._bg_dy = -13 if not self.rows else 0
            self.create_image(w / 2, h / 2 + self._bg_dy, image=self._bg_photo,
                              tags="bg")

        if not self.rows:
            self.create_text(w / 2, h / 2 + self._bg_dy + self._bg_height() / 2 + 13,
                             text="every drum found will appear here, in its own colour",
                             fill=p.dim, font=("Segoe UI", 9))
            self.configure(scrollregion=(0, 0, w, h))
            self._show_scrollbar(False)
            return

        cols = (0.04, 0.50, 0.68, 0.87)
        for i, (title, fx) in enumerate(zip(self.HEADINGS, cols)):
            self.create_text(fx * w, self.HEAD_H / 2, text=title, fill=p.dim,
                             anchor="w" if i == 0 else "center",
                             font=("Segoe UI", 9))
        self.create_line(0, self.HEAD_H, w, self.HEAD_H, fill=p.line)

        for r, (label, hits, lo, hi) in enumerate(self.rows):
            y = self.HEAD_H + self.ROW_H * r + self.ROW_H / 2
            colour = self.app._row_colour(label)
            icon = self.app._icon_for(label)
            x = cols[0] * w
            if icon:
                self.create_image(x, y, image=icon, anchor="w")
                x += icon.width() + 6
            self.create_text(x, y, text=label, fill=colour, anchor="w",
                             font=("Segoe UI", 9))
            for fx, value in zip(cols[1:], (hits, lo, hi)):
                self.create_text(fx * w, y, text=value, fill=colour, anchor="center",
                                 font=("Segoe UI", 9))

        content = self.HEAD_H + self.ROW_H * len(self.rows) + 4
        self.configure(scrollregion=(0, 0, w, max(content, h)))
        self._place_background()
        self._show_scrollbar(content > h)

    def _show_scrollbar(self, needed: bool) -> None:
        """A permanent scrollbar on a table of eight rows is just clutter."""
        if self.scrollbar is None:
            return
        if needed:
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()

    def _bg_height(self) -> int:
        return self._bg_photo.height() if self._bg_photo is not None else 0


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(6, weight=1)

        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.out_path: Path | None = None
        # which slice of the progress bar the current stage owns
        self._stage_span: tuple[int, int] = (0, 100)
        # true while the results table holds the early preview, before velocities
        self._early_rows = False
        self.in_reading_table = False

        self.v_input = tk.StringVar()
        self.v_output = tk.StringVar()
        self.v_sep = tk.StringVar(value="uvr")
        self.v_song = tk.BooleanVar(value=False)
        self.v_learned = tk.BooleanVar(value=True)
        self.v_fuse = tk.BooleanVar(value=True)
        self.v_rescue = tk.BooleanVar(value=True)
        self.v_hats = tk.BooleanVar(value=True)
        self.v_split = tk.BooleanVar(value=False)
        self.v_quant = tk.StringVar(value="")
        self.v_tempo = tk.StringVar(value="")
        self.v_thresholds = tk.StringVar(value="")
        self.v_status = tk.StringVar(value="Ready")
        self.v_estimate = tk.StringVar(value="")
        self.v_live_tempo = tk.StringVar(value="—")
        self.v_live_notes = tk.StringVar(value="—")
        self.v_live_fused = tk.StringVar(value="—")
        self.v_live_ride = tk.StringVar(value="—")
        self.v_adv_note = tk.StringVar(value="")
        self.v_theme = tk.StringVar(value="light")
        self.adv_rows: dict = {}

        self._build()
        self._load_settings()
        self.after(100, self._pump)

    def _load_icons(self) -> None:
        """Renders one small icon per drum. Optional: the GUI works without Pillow."""
        self._icons: dict = {}
        self._icon_colours: dict = {}
        try:
            sys.path.insert(0, str(ROOT))
            from PIL import ImageTk

            import drum_icons
        except Exception:
            return
        for pitch in (36, 38, 40, 43, 45, 47, 48, 50, 42, 44, 46, 49, 51):
            try:
                img = drum_icons.draw_icon(pitch, 18)
                # PhotoImage objects must be kept alive by us, not by the widget
                self._icons[pitch] = ImageTk.PhotoImage(img)
                self._icon_colours[pitch] = drum_icons.colour(pitch)
            except Exception:
                continue

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        self._load_icons()
        self._logo = None
        logo_path = ROOT / "docs" / "logo_64.png"
        if logo_path.exists():
            try:
                # Tk 8.6 reads PNG natively. The reference must be kept on the instance
                # or the image is garbage collected and the label goes blank.
                self._logo = tk.PhotoImage(file=str(logo_path))
            except tk.TclError:
                self._logo = None
        if self._logo is not None:
            head = ttk.Frame(self)
            head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            ttk.Label(head, image=self._logo).grid(row=0, column=0, rowspan=2,
                                                   padx=(0, 10))
            ttk.Label(head, text="drum2midi",
                      font=("Segoe UI Semibold", 16)).grid(row=0, column=1, sticky="sw")
            ttk.Label(head, text="drum recordings to General MIDI — measured, not guessed",
                      style="Dim.TLabel").grid(row=1, column=1, sticky="nw")
            head.columnconfigure(2, weight=1)
            ttk.Button(head, text=self._theme_label(), width=9,
                       command=self._flip_theme).grid(row=0, column=3, rowspan=2,
                                                      sticky="e")
            self.btn_theme = head.grid_slaves(row=0, column=3)[0]

        files = ttk.LabelFrame(self, text="Files", padding=10)
        files.grid(row=1, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Drums:").grid(row=0, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.v_input).grid(row=0, column=1, sticky="ew", padx=6)
        btns = ttk.Frame(files)
        btns.grid(row=0, column=2)
        ttk.Button(btns, text="File…", width=7,
                   command=self._pick_input).grid(row=0, column=0)
        ttk.Button(btns, text="Folder…", width=8,
                   command=self._pick_folder).grid(row=0, column=1, padx=(4, 0))

        ttk.Checkbutton(files, text="this is a whole song — extract drums first",
                        variable=self.v_song).grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(files, textvariable=self.v_estimate, style="Warn.TLabel"
                  ).grid(row=3, column=1, sticky="w", padx=6, pady=(4, 8))

        ttk.Label(files, text="Save to:").grid(row=2, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.v_output).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(files, text="Browse…", command=self._pick_output).grid(row=2, column=2)

        sep = ttk.LabelFrame(self, text="Separator", padding=10)
        sep.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for i, (val, label) in enumerate(SEPARATORS):
            ttk.Radiobutton(sep, text=label, value=val, variable=self.v_sep,
                            command=self._sync).grid(row=i, column=0, sticky="w")
        gpu = _accelerator()
        ttk.Label(sep,
                  text=(f"Separation runs on {gpu}; transcription stays on the CPU, "
                        f"where recurrent layers are faster"
                        if gpu else
                        "No supported GPU found — everything runs on the CPU"),
                  style="Ok.TLabel" if gpu else "Dim.TLabel"
                  ).grid(row=len(SEPARATORS), column=0, sticky="w", pady=(6, 0))

        # Options and the channel map are both "settings you set before converting", so
        # they live together in one notebook rather than having one of them exiled to
        # the results pane at the bottom, which is where output belongs.
        opt_nb = ttk.Notebook(self)
        opt_nb.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        opts = ttk.Frame(opt_nb, padding=10)
        adv = ttk.Frame(opt_nb, padding=10)
        opt_nb.add(opts, text="Options")
        opt_nb.add(adv, text="Channels & notes")
        opts.columnconfigure(3, weight=1)
        # A notebook sizes itself to its tallest tab, and the channel map is three times
        # the height of the options. Left alone that donates 340px of empty space to the
        # options tab and takes it from the results pane below, which is the part worth
        # the room. So it follows whichever tab is actually showing.
        self._opt_nb = opt_nb
        opt_nb.bind("<<NotebookTabChanged>>", self._fit_settings_tab, add="+")
        self.cb_fuse = ttk.Checkbutton(
            opts, text="fuse onsets found in stems", variable=self.v_fuse)
        self.cb_fuse.grid(row=0, column=0, sticky="w", columnspan=2)
        ttk.Checkbutton(opts, text="learned models (velocity, pedal hi-hat)",
                        variable=self.v_learned).grid(row=1, column=0, sticky="w", columnspan=2)
        self.cb_rescue = ttk.Checkbutton(
            opts, text="rescue toms from the stem (LarsNet only)", variable=self.v_rescue)
        self.cb_rescue.grid(row=2, column=0, sticky="w", columnspan=2)
        ttk.Checkbutton(opts, text="split hi-hat open/closed",
                        variable=self.v_hats).grid(row=3, column=0, sticky="w", columnspan=2)
        ttk.Checkbutton(opts, text="one MIDI track per drum",
                        variable=self.v_split).grid(row=4, column=0, sticky="w", columnspan=2)

        ttk.Label(opts, text="Quantize:").grid(row=0, column=2, sticky="e", padx=(16, 4))
        box = ttk.Combobox(opts, state="readonly", width=16,
                           values=[label for _, label in QUANTIZE])
        box.current(0)
        box.grid(row=0, column=3, sticky="w")
        box.bind("<<ComboboxSelected>>",
                 lambda e: self.v_quant.set(QUANTIZE[box.current()][0]))

        ttk.Label(opts, text="Thresholds:").grid(row=1, column=2, sticky="e", padx=(16, 4))
        ttk.Entry(opts, textvariable=self.v_thresholds, width=22).grid(row=1, column=3, sticky="w")
        ttk.Label(opts, text="empty = tuned defaults", style="Dim.TLabel"
                  ).grid(row=2, column=3, sticky="w")

        ttk.Label(opts, text="Tempo:").grid(row=3, column=2, sticky="e", padx=(16, 4))
        tempo_row = ttk.Frame(opts)
        tempo_row.grid(row=3, column=3, sticky="w")
        ttk.Entry(tempo_row, textvariable=self.v_tempo, width=7).grid(row=0, column=0)
        ttk.Label(tempo_row, text="BPM — empty = detect automatically",
                  style="Dim.TLabel").grid(row=0, column=1, padx=(6, 0))

        self._build_advanced(adv)

        run = ttk.Frame(self)
        run.grid(row=4, column=0, sticky="ew", pady=(10, 2))
        run.columnconfigure(2, weight=1)
        self.btn_run = ttk.Button(run, text="Convert", command=self._start)
        self.btn_run.grid(row=0, column=0)
        self.btn_stop = ttk.Button(run, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=6)
        self.bar = ttk.Progressbar(run, mode="determinate", maximum=100)
        self.bar.grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Label(run, textvariable=self.v_status, width=18,
                  anchor="w").grid(row=0, column=3)

        # A live summary of what the run has found so far. The pipeline already
        # reports all of it, but buried in the log where nobody watches it.
        live = ttk.Frame(self)
        live.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        # kick indigo, tom green, crash purple, ride teal — the same colours the icons
        # and the logo use. Named styles rather than literals, so a theme switch
        # repaints them.
        self._live_colours = ("#3f51b5", "#2e7d32", "#8e24aa", "#00897b")
        for i, (label, var) in enumerate((
                ("Tempo", self.v_live_tempo),
                ("Notes", self.v_live_notes),
                ("From stems", self.v_live_fused),
                ("Ride", self.v_live_ride))):
            ttk.Label(live, text=f"{label}:", style="Dim.TLabel").grid(
                row=0, column=i * 2, padx=(0 if i == 0 else 18, 5))
            ttk.Label(live, textvariable=var, style=f"Live{i}.TLabel",
                      font=("Segoe UI Semibold", 10)).grid(row=0, column=i * 2 + 1)

        nb = ttk.Notebook(self)
        nb.grid(row=6, column=0, sticky="nsew")
        res = ttk.Frame(nb, padding=6)
        logf = ttk.Frame(nb, padding=6)
        nb.add(res, text="Result")
        nb.add(logf, text="Log")
        for f in (res, logf):
            f.columnconfigure(0, weight=1)
            f.rowconfigure(0, weight=1)

        self.table = ResultTable(res, self)
        self.table.grid(row=0, column=0, sticky="nsew")
        sbr = ttk.Scrollbar(res, orient="vertical", command=self.table.scroll)
        sbr.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=sbr.set)
        self.table.scrollbar = sbr
        # The panel's height depends on the window, the screen and the font scaling,
        # so the picture is fitted to it rather than drawn at a fixed size.
        self._kit_size = 0
        self._kit_job = None
        self.table.bind("<Configure>", self._kit_fit, add="+")

        self.log = tk.Text(logf, height=10, wrap="none", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set, state="disabled")

        after = ttk.Frame(self)
        after.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        self.btn_open = ttk.Button(after, text="Show file", command=self._reveal,
                                   state="disabled")
        self.btn_open.grid(row=0, column=0)
        self.btn_ab = ttk.Button(after, text="Compare by ear (A/B)", command=self._render_ab,
                                 state="disabled")
        self.btn_ab.grid(row=0, column=1, padx=6)
        self._restyle()
        self._sync()

    # ------------------------------------------------------- theming
    def _theme_label(self) -> str:
        return "Dark" if self.v_theme.get() == "light" else "Light"

    def _flip_theme(self) -> None:
        self.v_theme.set("dark" if self.v_theme.get() == "light" else "light")
        self._restyle()
        self._save_settings()

    def _restyle(self) -> None:
        """Apply the theme and repaint the parts ttk styles do not reach.

        Treeview tags, tk.Text and the instrument colours are widget options rather
        than style options, so they have to be set again by hand on every switch.
        """
        if theme is None:
            return
        p = theme.apply(self.winfo_toplevel(), self.v_theme.get())
        self._palette = p
        style = ttk.Style()
        for i, colour in enumerate(getattr(self, "_live_colours", ())):
            style.configure(f"Live{i}.TLabel", foreground=theme.readable(colour, p))
        if hasattr(self, "log"):
            state = str(self.log.cget("state"))
            self.log.configure(state="normal")
            self.log.configure(**theme.text_options(p))
            self.log.configure(state=state)
        if hasattr(self, "btn_theme"):
            self.btn_theme.configure(text=self._theme_label())
        if hasattr(self, "btn_run"):
            self.btn_run.configure(style="Accent.TButton")
        self._load_kit()

    def _row_colour(self, label: str) -> str:
        """The colour a result row is written in: the drum's own, fit for the theme."""
        pitch = LABEL_TO_PITCH.get(label.strip().lower())
        base = self._icon_colours.get(pitch) if pitch else None
        if base is None:
            return self._palette.ink
        return theme.readable(base, self._palette) if theme else base

    def _kit_fit(self, event) -> None:
        """Picks a kit size that fits the panel, and redraws when it changes.

        Debounced: a window drag fires <Configure> on every pixel, and each redraw is
        a full render plus a PNG round trip.
        """
        usable = max(0, event.height - 44)
        width = int(min(event.width * 0.46, usable / 0.76))
        width = max(120, min(340, width))
        if abs(width - self._kit_size) < 10:
            return
        self._kit_size = width
        if self._kit_job is not None:
            self.after_cancel(self._kit_job)
        self._kit_job = self.after(120, self._load_kit)

    def _load_kit(self) -> None:
        """Redraws the kit behind the results table, sized for the current panel.

        Optional in the same way the icons are: without Pillow the table simply has a
        plain background.
        """
        if not hasattr(self, "table"):
            return
        self._kit_job = None
        try:
            import base64
            import io

            import drum_kit_art
            bg = tuple(int(self._palette.base[i:i + 2], 16) for i in (1, 3, 5))
            # bolder while the table is empty, faint once it has rows to read
            alpha = 46 if not self.table.rows else 22
            img = drum_kit_art.render(self._kit_size or 258,
                                      dark=self.v_theme.get() == "dark", alpha=alpha)
            buf = io.BytesIO()
            drum_kit_art._on(bg, img).convert("RGB").save(buf, format="PNG")
            self._kit = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
            self.table.set_background(self._kit)
        except Exception:
            self._kit = None

    # ------------------------------------------------------- advanced tab
    def _fit_settings_tab(self, event=None) -> None:
        """Size the settings notebook to the tab on show, not to the tallest one."""
        nb = getattr(self, "_opt_nb", None)
        if not nb or not nb.tabs():
            return
        try:
            current = nb.nametowidget(nb.select())
        except Exception:
            return
        current.update_idletasks()
        nb.configure(height=max(current.winfo_reqheight(), 1))

    def _build_advanced(self, frame: ttk.Frame) -> None:
        """Override table: a custom note and channel for every drum.

        Defaults follow General MIDI on channel 9. Only rows that actually differ from
        the default are passed to the CLI.
        """
        ttk.Label(frame, wraplength=700, style="Dim.TLabel",
                  text="By default everything goes to channel 9, the General MIDI drum "
                       "channel. Here you can spread instruments across channels or remap "
                       "notes for a sampler that does not follow GM."
                  ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        head = ttk.Frame(frame)
        head.grid(row=1, column=0, columnspan=4, sticky="w")
        for col, (text, width) in enumerate((("Instrument", 20), ("Note", 8),
                                             ("", 10), ("Channel", 8))):
            ttk.Label(head, text=text, width=width, font=("", 9, "bold")
                      ).grid(row=0, column=col, padx=(0, 10), sticky="w")

        body = ttk.Frame(frame)
        body.grid(row=2, column=0, columnspan=4, sticky="nw")
        self.adv_rows = {}
        self._adv_names = {}
        for i, (pitch, label) in enumerate(KIT):
            note_var = tk.StringVar(value=str(pitch))
            chan_var = tk.StringVar(value=str(DRUM_CHANNEL))
            name_var = tk.StringVar(value=note_name(pitch))

            icon = self._icons.get(pitch)
            cell = ttk.Frame(body)
            cell.grid(row=i, column=0, padx=(0, 10), sticky="w", pady=1)
            if icon is not None:
                ttk.Label(cell, image=icon).grid(row=0, column=0, padx=(0, 6))
            ttk.Label(cell, text=label, width=16).grid(row=0, column=1, sticky="w")

            ttk.Spinbox(body, from_=0, to=127, width=6, textvariable=note_var).grid(
                row=i, column=1, padx=(0, 10), pady=1)
            # the letter name, updated live: a number alone is hard to place on a
            # keyboard, and every DAW labels the octave differently
            ttk.Label(body, textvariable=name_var, width=10,
                      style="Accent.TLabel").grid(row=i, column=2, padx=(0, 10),
                                                  sticky="w")
            ttk.Spinbox(body, from_=0, to=15, width=6, textvariable=chan_var).grid(
                row=i, column=3, padx=(0, 10), pady=1)

            note_var.trace_add(
                "write",
                lambda *_, v=note_var, n=name_var: n.set(note_name_from_text(v.get())))
            self.adv_rows[pitch] = (note_var, chan_var)
            self._adv_names[pitch] = name_var

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Button(buttons, text="Reset to General MIDI",
                   command=self._reset_advanced).grid(row=0, column=0)
        ttk.Button(buttons, text="Spread across channels 0–9",
                   command=self._spread_channels).grid(row=0, column=1, padx=6)
        ttk.Label(buttons, textvariable=self.v_adv_note, style="Dim.TLabel"
                  ).grid(row=0, column=2, padx=(10, 0))
        self._refresh_adv_note()

    def _reset_advanced(self) -> None:
        for pitch, (note_var, chan_var) in self.adv_rows.items():
            note_var.set(str(pitch))
            chan_var.set(str(DRUM_CHANNEL))
        self._refresh_adv_note()

    def _spread_channels(self) -> None:
        for i, (pitch, (_, chan_var)) in enumerate(self.adv_rows.items()):
            chan_var.set(str(i % 16))
        self._refresh_adv_note()

    def _refresh_adv_note(self) -> None:
        notes, chans = self._overrides()
        if not notes and not chans:
            self.v_adv_note.set("no overrides")
        else:
            self.v_adv_note.set(f"changed: {len(notes)} notes, {len(chans)} channels")

    def _overrides(self) -> tuple[dict, dict]:
        """Only rows that differ from the default; malformed input is ignored."""
        notes, chans = {}, {}
        for pitch, (note_var, chan_var) in getattr(self, "adv_rows", {}).items():
            try:
                n = int(note_var.get())
                if n != pitch and 0 <= n <= 127:
                    notes[pitch] = n
            except ValueError:
                pass
            try:
                c = int(chan_var.get())
                if c != DRUM_CHANNEL and 0 <= c <= 15:
                    chans[pitch] = c
            except ValueError:
                pass
        return notes, chans

    # ------------------------------------------------------------- behaviour
    def _sync(self) -> None:
        sep = self.v_sep.get()
        uvr = sep == "uvr"
        self.cb_fuse.state(["!disabled"] if uvr else ["disabled"])
        self.v_fuse.set(bool(uvr))
        # rescue_toms only helps when the tom stem is dirty, which is LarsNet's case;
        # on MDX23C's clean stems it adds false positives (toms 0.605 -> 0.398)
        rescue_ok = sep in ("larsnet", "drumsep", "hybrid")
        self.cb_rescue.state(["!disabled"] if rescue_ok else ["disabled"])
        self.v_rescue.set(bool(rescue_ok))
        self._update_estimate()

    def _update_estimate(self) -> None:
        """Shows the expected runtime; with MDX23C it is far longer than the audio."""
        path = self.v_input.get()
        if not path or not Path(path).exists():
            self.v_estimate.set("")
            return
        try:
            import soundfile as sf
            p = Path(path)
            if p.is_dir():
                exts = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".opus"}
                files = [f for f in p.iterdir()
                         if f.is_file() and f.suffix.lower() in exts]
                if not files:
                    self.v_estimate.set("no audio files in this folder")
                    return
                dur = sum(sf.info(str(f)).duration for f in files)
                prefix = f"{len(files)} files, "
            else:
                dur = sf.info(path).duration
                prefix = ""
        except Exception:
            self.v_estimate.set("")
            return
        # measured on this machine: seconds of processing per second of audio.
        # MDX23C is the only stage that moves to the GPU, so only its rate changes.
        gpu = _accelerator() is not None
        rate = {"uvr": 1.9 if gpu else 15.0, "larsnet": 0.6, "none": 0.35,
                "drumsep": 1.2, "hybrid": 2.9 if gpu else 16.0}.get(self.v_sep.get(), 1.0)
        secs = dur * rate
        human = (f"{secs/3600:.1f} h" if secs >= 5400
                 else f"{secs/60:.0f} min" if secs >= 90 else f"{secs:.0f} s")
        self.v_estimate.set(f"{prefix}{dur/60:.1f} min of audio  →  roughly {human}")

    def _pick_input(self) -> None:
        p = filedialog.askopenfilename(
            title="Choose audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.aiff *.aif *.m4a *.ogg"),
                       ("All files", "*.*")])
        if p:
            self.v_input.set(p)
            if not self.v_output.get():
                self.v_output.set(str(Path(p).with_suffix(".mid")))
            self._update_estimate()

    def _pick_folder(self) -> None:
        p = filedialog.askdirectory(title="Choose a folder of audio files")
        if p:
            self.v_input.set(p)
            self.v_output.set(str(Path(p) / "midi"))
            self._update_estimate()

    def _pick_output(self) -> None:
        p = filedialog.asksaveasfilename(title="Save MIDI", defaultextension=".mid",
                                         filetypes=[("MIDI", "*.mid")])
        if p:
            self.v_output.set(p)

    def _command(self) -> list[str]:
        cmd = [PY, str(ROOT / "drum2midi.py"), self.v_input.get(),
               "-o", self.v_output.get(), "--device", "auto"]
        sep = self.v_sep.get()
        if sep == "none":
            cmd.append("--no-separate")
        else:
            cmd += ["--separator", sep]
            cmd += ["--fuse-stem-onsets", "on" if self.v_fuse.get() else "off"]
        if self.v_song.get():
            cmd.append("--from-song")
        if not self.v_learned.get():
            cmd.append("--no-learned")
        if not self.v_rescue.get():
            cmd.append("--no-rescue-toms")
        if not self.v_hats.get():
            cmd.append("--no-split-hats")
        if self.v_split.get():
            cmd.append("--split-tracks")
        notes, chans = self._overrides()
        if notes:
            cmd += ["--notes", ",".join(f"{k}={v}" for k, v in sorted(notes.items()))]
        if chans:
            cmd += ["--channels", ",".join(f"{k}={v}" for k, v in sorted(chans.items()))]
        if self.v_quant.get():
            cmd += ["--quantize", self.v_quant.get()]
        tempo = self._tempo_value()
        if tempo is not None:
            cmd += ["--tempo", f"{tempo:g}"]
        if self.v_thresholds.get().strip():
            cmd += ["--thresholds", self.v_thresholds.get().strip()]
        return cmd

    def _tempo_value(self) -> float | None:
        """Returns the tempo to force, None to let the pipeline estimate it.
        Raises ValueError if the field holds something that is not a usable BPM."""
        raw = self.v_tempo.get().strip().replace(",", ".")
        if not raw:
            return None
        value = float(raw)
        if not 20.0 <= value <= 400.0:
            raise ValueError("tempo outside the 20–400 range")
        return value

    def _start(self) -> None:
        src = Path(self.v_input.get())
        if not self.v_input.get() or not src.exists():
            messagebox.showwarning("No file", "Choose an existing audio file or folder.")
            return
        if not self.v_output.get():
            self.v_output.set(str(src / "midi" if src.is_dir()
                                  else src.with_suffix(".mid")))

        try:
            self._tempo_value()
        except ValueError:
            messagebox.showwarning(
                "Invalid tempo",
                "Enter a tempo between 20 and 400 BPM, or leave the field empty "
                "to detect it automatically.")
            return

        if self.v_quant.get() and self._tempo_value() is None:
            if not messagebox.askokcancel(
                    "Quantizing without a set tempo",
                    "The quantization grid is derived from the tempo, which will be "
                    "detected automatically. Tempo detectors often land on double or "
                    "half the real value, and then quantizing ruins the groove.\n\n"
                    "If you know the project tempo, type it in.\n\n"
                    "Continue?"):
                return

        self.out_path = Path(self.v_output.get())
        self.table.clear()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.in_reading_table = False
        self.bar["value"] = 0
        self._stage_span = (0, 100)
        for var in (self.v_live_tempo, self.v_live_notes,
                    self.v_live_fused, self.v_live_ride):
            var.set("—")
        self.v_status.set("Working…")
        self.btn_run.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self.btn_open.state(["disabled"])
        self.btn_ab.state(["disabled"])
        self._save_settings()

        self._refresh_adv_note()

        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        cmd = self._command()
        self._emit("$ " + " ".join(cmd) + "\n")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", env=env,
                    cwd=str(ROOT), creationflags=NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.q.put(("line", line))
                code = self.proc.wait()
            except Exception as exc:  # noqa: BLE001
                self.q.put(("line", f"ERROR: {exc}\n"))
                code = -1
            self.q.put(("done", code))

        threading.Thread(target=worker, daemon=True).start()

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.v_status.set("Stopping…")

    def _emit(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self._handle(payload)
                else:
                    self._finish(payload)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def _handle(self, line: str) -> None:
        # tqdm progress from the separators would flood the log
        if "it/s" in line or "s/it" in line:
            return

        # Separation is 90% of a run, so its own percentage is mapped into the slice of
        # the bar that stage owns. Without this the bar sits still for minutes.
        match = _SEPARATING.search(line)
        if match:
            inner = int(match.group(1))
            lo, hi = self._stage_span
            self.bar["value"] = lo + (hi - lo) * inner / 100
            self.v_status.set(f"Separating… {inner}%")
            return

        self._emit(line)
        self._update_live(line)
        for marker, pct in STAGES.items():
            if marker in line:
                self.bar["value"] = pct
                self._stage_span = STAGE_SPANS.get(marker, (pct, pct))
                self.v_status.set(line.split("]", 1)[-1].strip()[:40] or "Working…")
        stripped = line.rstrip("\n")
        if stripped.startswith("instrument"):
            self.in_reading_table = True
            # the final table replaces the early preview
            self.table.clear()
            self._early_rows = False
            return
        if self.in_reading_table:
            if set(stripped.strip()) <= {"-"} and stripped.strip():
                return
            parts = stripped.rsplit(None, 3)
            if len(parts) == 4 and parts[1].isdigit():
                name, hits, lo, hi = parts
                label = name.strip()
                self.table.add(label, hits, lo, hi)
            elif not stripped.strip():
                self.in_reading_table = False

    def _update_live(self, line: str) -> None:
        """Pulls the run's findings out of the log and into the live strip."""
        early = _FOUND.search(line.rstrip())
        if early:
            # fill the results table straight away, then let the final summary
            # overwrite it with velocities once separation is done
            label = early.group(1).strip()
            if not self._early_rows:
                self.table.clear()
            self._early_rows = True
            self.table.add(label, early.group(2), "…", "…")
            return

        for key, pattern in _LIVE.items():
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1)
            if key == "tempo":
                detected = not self.v_tempo.get().strip()
                self.v_live_tempo.set(f"{float(value):.0f} BPM"
                                      + (" (detected)" if detected else " (set)"))
            elif key == "notes":
                self.v_live_notes.set(value)
            elif key == "fused":
                self.v_live_fused.set(f"+{value}")
            elif key == "ride":
                self.v_live_ride.set(value)

    def _icon_for(self, label: str):
        """The kit icon for a row of the results table, or "" when there is none."""
        pitch = LABEL_TO_PITCH.get(label.lower())
        return self._icons.get(pitch, "")

    def _finish(self, code: int) -> None:
        self.proc = None
        self.btn_run.state(["!disabled"])
        self.btn_stop.state(["disabled"])
        ok = code == 0 and self.out_path and self.out_path.exists()
        self.bar["value"] = 100 if ok else 0
        self.v_status.set("Done" if ok else ("Cancelled" if code else "Error"))
        if ok:
            self.btn_open.state(["!disabled"])
            # A/B needs a single MIDI paired with a single source file
            if not self.out_path.is_dir():
                self.btn_ab.state(["!disabled"])
        elif code not in (0, -1):
            messagebox.showerror("Failed",
                                 "Conversion ended with an error. See the Log tab.")

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
            messagebox.showinfo("File", str(path))

    def _open_externally(self, path: Path) -> None:
        """Hands a file to whatever the desktop uses to play it."""
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except (OSError, FileNotFoundError):
            pass

    def _render_ab(self) -> None:
        # A/B compares one MIDI against one source file; meaningless for a batch
        if not (self.out_path and self.out_path.exists()) or self.out_path.is_dir():
            return
        wav = self.out_path.with_name(self.out_path.stem + "_AB.wav")
        self.v_status.set("Rendering A/B…")
        self.btn_ab.state(["disabled"])

        def worker():
            cmd = [PY, str(ROOT / "render_midi.py"), str(self.out_path),
                   "--against", self.v_input.get(), "-o", str(wav)]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=str(ROOT), creationflags=NO_WINDOW)
            self.q.put(("line", (r.stdout or "") + (r.stderr or "")))
            if wav.exists():
                self._open_externally(wav)
            self.q.put(("done", 0))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------- settings
    def _save_settings(self) -> None:
        try:
            SETTINGS.write_text(json.dumps({
                "sep": self.v_sep.get(), "song": self.v_song.get(),
                "learned": self.v_learned.get(), "fuse": self.v_fuse.get(),
                "rescue": self.v_rescue.get(), "hats": self.v_hats.get(),
                "split": self.v_split.get(),
                "quant": self.v_quant.get(), "thresholds": self.v_thresholds.get(),
                "tempo": self.v_tempo.get(),
                "theme": self.v_theme.get(),
                "overrides": {str(k): [v[0].get(), v[1].get()]
                              for k, v in self.adv_rows.items()},
                "last_dir": str(Path(self.v_input.get()).parent) if self.v_input.get() else "",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_settings(self) -> None:
        try:
            d = json.loads(SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.v_sep.set(d.get("sep", "uvr"))
        self.v_song.set(d.get("song", False))
        self.v_learned.set(d.get("learned", True))
        self.v_rescue.set(d.get("rescue", True))
        self.v_hats.set(d.get("hats", True))
        self.v_split.set(d.get("split", False))
        self.v_quant.set(d.get("quant", ""))
        self.v_tempo.set(d.get("tempo", ""))
        self.v_thresholds.set(d.get("thresholds", ""))
        for key, pair in (d.get("overrides") or {}).items():
            row = self.adv_rows.get(int(key))
            if row and isinstance(pair, list) and len(pair) == 2:
                row[0].set(str(pair[0]))
                row[1].set(str(pair[1]))
        if d.get("theme") in ("light", "dark"):
            self.v_theme.set(d["theme"])
            self._restyle()
        self._refresh_adv_note()
        self._sync()


def _apply_icon(root: tk.Tk) -> None:
    """Sets the window and taskbar icon, by whichever route works.

    Three mechanisms, because on Windows none of them covers every case:
    iconbitmap() with an .ico is the Tk route, iconphoto() with a PNG is what other
    platforms honour, and WM_SETICON puts the handle on the window directly, which is
    what the taskbar button actually reads. A missing icon must never stop the window
    from opening.
    """
    ico = ROOT / "docs" / "drum2midi.ico"
    png = ROOT / "docs" / "logo_64.png"
    if ico.exists():
        for call in (lambda: root.iconbitmap(default=str(ico)),
                     lambda: root.iconbitmap(str(ico))):
            try:
                call()
            except tk.TclError:
                continue
    if png.exists():
        try:
            root._icon_image = tk.PhotoImage(file=str(png))
            root.iconphoto(True, root._icon_image)
        except tk.TclError:
            pass
    if os.name == "nt" and ico.exists():
        _set_win_icon(root, ico)


def _set_win_icon(root: tk.Tk, ico: Path) -> None:
    """Loads the .ico through the Win32 API and attaches it to the window.

    Tk's iconbitmap only sets the small icon on some builds, which leaves the taskbar
    showing the interpreter's. LoadImage with LR_LOADFROMFILE reads the file directly,
    and both icon slots are set: ICON_SMALL for the title bar, ICON_BIG for the taskbar
    and Alt-Tab.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()

        IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x0010, 0x0040
        WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1

        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        handles = {}
        for slot, size in ((ICON_SMALL, 16), (ICON_BIG, 32)):
            handle = user32.LoadImageW(None, str(ico), IMAGE_ICON, size, size,
                                       LR_LOADFROMFILE)
            if not handle:
                handle = user32.LoadImageW(None, str(ico), IMAGE_ICON, 0, 0,
                                           LR_LOADFROMFILE | LR_DEFAULTSIZE)
            if handle:
                user32.SendMessageW(hwnd, WM_SETICON, slot, handle)
                handles[slot] = handle

        # The window class still carries the interpreter's icon, and that is what the
        # shell falls back to when the per-window icon is not honoured. Replacing it
        # affects only this process's Tk windows.
        GCLP_HICON, GCLP_HICONSM = -14, -34
        user32.SetClassLongPtrW.restype = ctypes.c_void_p
        user32.SetClassLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int,
                                            ctypes.c_void_p]
        for index, slot in ((GCLP_HICON, ICON_BIG), (GCLP_HICONSM, ICON_SMALL)):
            if slot in handles:
                user32.SetClassLongPtrW(hwnd, index, handles[slot])
    except Exception:
        pass


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


def main() -> None:
    _set_taskbar_identity()
    root = tk.Tk()
    root.title("drum2midi — drums to MIDI")
    root.minsize(760, 900)
    _apply_icon(root)
    App(root)
    # Tk finishes creating the real top-level window during the first idle cycle, and
    # on some builds that discards an icon set beforehand. Re-applying once the window
    # exists is cheap and makes the outcome independent of that timing.
    root.after(300, lambda: _apply_icon(root))
    root.mainloop()


if __name__ == "__main__":
    main()
