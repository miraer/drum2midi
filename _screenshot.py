"""Renders the GUI to PNGs for the README, with a realistic example filled in.

Builds the window offscreen, populates it with plausible values and grabs it, so the
picture does not depend on whatever state a manually launched instance is in.

    python _screenshot.py [light|dark] [empty|ready|converting|done]
                          [--input drums.wav] [--midi drums.mid] [--out file.png]

Without --input it uses input/demo.wav when present. A finished run is simulated by
feeding the window the pipeline's own summary table; with --midi the timeline is
filled from that file.
"""

import argparse
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("words", nargs="*", default=[])
ap.add_argument("--input", type=Path, default=ROOT / "input" / "demo.wav")
ap.add_argument("--midi", type=Path, default=None)
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()
mode = "light" if "light" in args.words else "dark"
phase = next((w for w in args.words if w in ("empty", "ready", "converting", "done")),
             "done")

spec = importlib.util.spec_from_loader(
    "gui", SourceFileLoader("gui", str(ROOT / "drum2midi_gui.pyw")))
gui = importlib.util.module_from_spec(spec)
# dataclasses look their module up by name while the class is being built
sys.modules["gui"] = gui
spec.loader.exec_module(gui)

# never let a screenshot overwrite the user's saved settings
gui.SETTINGS = Path(gui.tempfile.gettempdir()) / "drum2midi_screenshot_settings.json"

app = gui.make_app(sys.argv[:1])
win = gui.Window()
win.theme = mode
win._apply_theme()
win.resize(1320, 860)
# the window asks on a thread; ask here so the device chip is not caught mid-check
win._got_devices(gui._device_plan())

if phase != "empty" and args.input.exists():
    win._set_input(args.input)
    # a neutral path, not wherever the example happens to live on this machine
    win.c.output = str(Path("C:/Music/Session") / args.input.with_suffix(".mid").name)
    # the envelope is computed on a thread; do it inline for a deterministic picture
    win.timeline.env = gui.envelope(args.input)
    win.timeline.position = min(72.4, win.timeline.duration * 0.27)
    win._update_time()

if phase == "converting":
    win.phase = "converting"
    win.started = gui.time.monotonic() - 150
    for line in ("[1/4] Transcribing drums.wav with ADTOF ...\n",
                 "      1004 hits detected\n",
                 "[2/4] Separating 272.0s (6 stems) ...\n"):
        win._handle(line)
    win._handle("      separating: 58%\n")
elif phase == "done":
    for line in ("[4/4] Writing MIDI at 101.0 BPM, unquantized\n",
                 "      1004 hits detected\n",
                 "      stem fusion: +75 onsets the mix pass missed\n",
                 "      ride/crash: 286 hits reassigned to ride\n",
                 "instrument        hits  vel min  vel max\n",
                 "----------------------------------------\n",
                 "Kick               270       15      127\n",
                 "Snare              318       15      127\n",
                 "Hi-hat closed      111       22      116\n",
                 "Hi-hat pedal         6       22       42\n",
                 "Hi-hat open         18       41      118\n",
                 "Floor tom hi        21       15      125\n",
                 "Mid tom              8      116      127\n",
                 "High tom            28       15      122\n",
                 "Crash               88       49      127\n",
                 "Ride               286       31      112\n",
                 "\n"):
        win._handle(line)
    win.phase = "done"
    # what _finish does: the stat is the notes written, after fusion
    win.live["notes"] = f"{sum(h for h, _, _ in win.results.values()):,}"
    win.progress = 100
    win.took = 521
    if args.midi and args.midi.exists():
        win.out_path = args.midi
        win.timeline.hits = gui.read_hits(args.midi, win.c)
    win.listen = "ab"
    win.listen_seg.set_value("ab")
win._sync()

win.show()
for _ in range(5):
    app.processEvents()
out = args.out or ROOT / "docs" / ("gui.png" if mode == "light" else f"gui_{mode}.png")
if phase == "empty" and args.out is None:
    out = out.with_name(out.stem + "_empty.png")
out.parent.mkdir(exist_ok=True)
win.grab().save(str(out))
print("saved:", out)
