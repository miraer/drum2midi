"""Renders the theme's widgets to PNGs, one per mode, for eyeballing and the README.

Kept separate from _screenshot.py, which shoots the real window: this one is a swatch
of every widget the theme restyles, so a regression in one part is obvious.

    python _theme_shots.py [light|dark ...]
"""

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import theme  # noqa: E402


def grab(root: tk.Tk, out: Path) -> None:
    root.update_idletasks()
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    time.sleep(1.2)
    root.update()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    out.parent.mkdir(exist_ok=True)
    ps = (
        "Add-Type -AssemblyName System.Drawing; "
        f"$b = New-Object System.Drawing.Bitmap({w},{h}); "
        "$g = [System.Drawing.Graphics]::FromImage($b); "
        f"$g.CopyFromScreen({x},{y},0,0,(New-Object System.Drawing.Size({w},{h}))); "
        f"$b.Save('{out}', [System.Drawing.Imaging.ImageFormat]::Png); "
        "$g.Dispose(); $b.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    root.attributes("-topmost", False)


def shoot(mode: str) -> None:
    root = tk.Tk()
    root.title(f"drum2midi — {mode}")
    p = theme.apply(root, mode)

    f = ttk.Frame(root, padding=16)
    f.pack(fill="both", expand=True)
    ttk.Label(f, text="drum2midi", font=("Segoe UI", 15, "bold")).pack(anchor="w")
    ttk.Label(f, text="drum recordings to General MIDI — measured, not guessed",
              style="Dim.TLabel").pack(anchor="w")

    box = ttk.Labelframe(f, text="Separator")
    box.pack(fill="x", pady=(10, 8))
    v = tk.StringVar(value="uvr")
    ttk.Radiobutton(box, text="MDX23C — 6 stems with ride/crash, best quality",
                    value="uvr", variable=v).pack(anchor="w")
    ttk.Radiobutton(box, text="LarsNet — 5 stems, no ride. Fast",
                    value="larsnet", variable=v).pack(anchor="w")
    for label, on in (("fuse onsets found in stems", 1),
                      ("learned models (velocity, pedal hi-hat)", 1),
                      ("rescue toms from the stem (LarsNet only)", 0)):
        var = tk.IntVar(value=on)
        cb = ttk.Checkbutton(box, text=label, variable=var)
        cb.pack(anchor="w")
        cb.var = var
        if label.startswith("rescue"):
            cb.state(["disabled"])

    row = ttk.Frame(f)
    row.pack(fill="x", pady=4)
    ttk.Label(row, text="Drums:").pack(side="left", padx=(0, 6))
    e = ttk.Entry(row)
    e.insert(0, r"input\demo.wav")
    e.pack(side="left", fill="x", expand=True)
    ttk.Button(row, text="File…").pack(side="left", padx=6)

    row2 = ttk.Frame(f)
    row2.pack(fill="x", pady=4)
    ttk.Label(row2, text="Note:").pack(side="left", padx=(0, 6))
    sp = ttk.Spinbox(row2, from_=0, to=127, width=6)
    sp.set(36)
    sp.pack(side="left")
    ttk.Label(row2, text="C1", style="Accent.TLabel").pack(side="left", padx=6)
    ttk.Label(row2, text="Quantize:").pack(side="left", padx=(14, 6))
    cb = ttk.Combobox(row2, values=["no quantization", "1/8", "1/16"],
                      width=16, state="readonly")
    cb.current(0)
    cb.pack(side="left")

    ttk.Progressbar(f, value=64).pack(fill="x", pady=(12, 6))
    live = ttk.Frame(f)
    live.pack(fill="x")
    for lab, val, st in (("Tempo:", "110 BPM (detected)", "Accent.TLabel"),
                         ("Notes:", "121", "Ok.TLabel"),
                         ("From stems:", "+8", "Ok.TLabel"),
                         ("Ride:", "7", "Warn.TLabel")):
        ttk.Label(live, text=lab, style="Dim.TLabel").pack(side="left", padx=(0, 4))
        ttk.Label(live, text=val, style=st).pack(side="left", padx=(0, 14))

    btns = ttk.Frame(f)
    btns.pack(fill="x", pady=10)
    ttk.Button(btns, text="Convert", style="Accent.TButton").pack(side="left")
    ttk.Button(btns, text="Stop", state="disabled").pack(side="left", padx=6)
    ttk.Button(btns, text="Show file").pack(side="left")

    nb = ttk.Notebook(f)
    nb.pack(fill="both", expand=True)
    for name in ("Result", "Log", "Channels & notes"):
        t = ttk.Frame(nb, padding=8)
        if name == "Result":
            txt = tk.Text(t, height=6, font=("Consolas", 9),
                          **theme.text_options(p))
            txt.insert("1.0", "instrument        hits  vel min  vel max\n"
                              "----------------------------------------\n"
                              "Kick                63      106      127\n"
                              "Snare               40       34      127\n"
                              "Hi-hat closed       38       32      119\n"
                              "Ride                 7       62      121")
            txt.pack(fill="both", expand=True)
        else:
            ttk.Label(t, text=f"{name} goes here", style="Dim.TLabel").pack(anchor="w")
        nb.add(t, text=name)
    nb.select(0)

    root.geometry("620x620+70+50")
    grab(root, ROOT / "docs" / f"theme_{mode}.png")
    out = ROOT / "docs" / f"theme_{mode}.png"
    print(f"saved: {out}", f"{out.stat().st_size // 1024} KB" if out.exists() else "MISSING")
    root.destroy()


for m in (sys.argv[1:] or ["light", "dark"]):
    shoot(m)
