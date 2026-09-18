"""Renders the GUI to a PNG for the README, with a realistic example filled in.

Builds the window headlessly, populates it with plausible values, forces it on screen
and grabs it. Avoids depending on whatever state a manually launched instance is in.
"""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent

import tkinter as tk

spec = importlib.util.spec_from_file_location("gui", ROOT / "drum2midi_gui.pyw")
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)

root = tk.Tk()
root.title("drum2midi — drums to MIDI")

app = gui.App(root)
mode = "dark" if "dark" in sys.argv[1:] else "light"
app.v_theme.set(mode)
app._restyle()

# a plausible example rather than an empty form
demo = ROOT / "input" / "demo.wav"
if demo.exists():
    app.v_input.set(str(demo))
    app.v_output.set(str(demo.with_suffix(".mid")))
    app._update_estimate()
app.v_sep.set("uvr")
app._sync()

# a finished run, so the Result tab is not blank
for line in ("instrument        hits  vel min  vel max\n",
             "----------------------------------------\n",
             "Kick                63      106      127\n",
             "Snare               40       34      127\n",
             "Hi-hat closed       38       32      119\n",
             "Hi-hat open          6       33       94\n",
             "Mid tom              3      102      127\n",
             "Crash               10       66      127\n",
             "Ride                 7       62      121\n"):
    if "empty" not in sys.argv[1:]:
        app._handle(line)
app.v_live_tempo.set("110 BPM (detected)")
app.v_live_notes.set("121")
app.v_live_fused.set("+8")
app.v_live_ride.set("7")
app.bar["value"] = 100
app.v_status.set("Done")
app.btn_open.state(["!disabled"])
app.btn_ab.state(["!disabled"])

for _w in app.winfo_children():
    if _w.winfo_class() == 'TNotebook':
        _w.select(2 if "notes" in sys.argv[1:] else 0)
        break
root.geometry("820x960+60+40")
root.update_idletasks()
root.deiconify()
root.lift()
root.attributes("-topmost", True)
root.update()
time.sleep(1.5)
root.update()

x, y = root.winfo_rootx(), root.winfo_rooty()
w, h = root.winfo_width(), root.winfo_height()
print(f"window: {x},{y} {w}x{h}")

out = ROOT / "docs" / ("gui.png" if mode == "light" else f"gui_{mode}.png")
for tag in ("notes", "empty"):
    if tag in sys.argv[1:]:
        out = out.with_name(out.stem + f"_{tag}.png")
out.parent.mkdir(exist_ok=True)

ps = f'''
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap({w}, {h})
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen({x}, {y}, 0, 0, (New-Object System.Drawing.Size({w}, {h})))
$bmp.Save("{out.as_posix()}", [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
'''
subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
root.attributes("-topmost", False)
root.destroy()

print("saved:", out, out.stat().st_size // 1024, "KB" if out.exists() else "MISSING")
