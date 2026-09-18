"""Reports what a running drum2midi window actually has attached as its icon.

Run this while the GUI is open. It finds the window by title, asks it for both icon
slots, and compares them against the handles Windows hands out for our .ico and for
pythonw.exe -- which answers "is the taskbar showing our icon or the interpreter's?"
without needing to see the screen.

    python check_running_icon.py
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

user32 = ctypes.windll.user32
WM_GETICON = 0x007F
ICON_SMALL, ICON_BIG, ICON_SMALL2 = 0, 1, 2
GCLP_HICON, GCLP_HICONSM = -14, -34

titles = ["drum2midi — drums to MIDI", "drum2midi - drums to MIDI"]
hwnd = 0
for title in titles:
    hwnd = user32.FindWindowW(None, title)
    if hwnd:
        break

if not hwnd:
    # fall back to scanning every visible top-level window for one named drum2midi
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum(h, _):
        length = user32.GetWindowTextLengthW(h)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(h, buf, length + 1)
            if "drum2midi" in buf.value.lower():
                found.append((h, buf.value))
        return True

    user32.EnumWindows(enum, 0)
    if found:
        hwnd, title = found[0]
        print(f"found by scan: {title!r}")

if not hwnd:
    print("No drum2midi window is open. Start it, then run this again.")
    raise SystemExit(1)

print(f"window handle: {hwnd}")
for name, slot in (("ICON_SMALL", ICON_SMALL), ("ICON_BIG", ICON_BIG),
                   ("ICON_SMALL2", ICON_SMALL2)):
    print(f"  WM_GETICON {name:<12} = {user32.SendMessageW(hwnd, WM_GETICON, slot, 0)}")

user32.GetClassLongPtrW.restype = ctypes.c_void_p
for name, index in (("class HICON", GCLP_HICON), ("class HICONSM", GCLP_HICONSM)):
    print(f"  {name:<24} = {user32.GetClassLongPtrW(hwnd, index)}")

pid = wintypes.DWORD()
user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
print(f"  owning process id        = {pid.value}")

import subprocess
out = subprocess.run(["powershell", "-NoProfile", "-Command",
                      f"(Get-Process -Id {pid.value}).Path"],
                     capture_output=True, text=True).stdout.strip()
print(f"  owning executable        = {out}")
