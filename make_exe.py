"""Builds drum2midi.exe: a copy of pythonw.exe carrying our icon.

Everything else was tried first and did not move the taskbar icon: WM_SETICON on the
real top-level window (verified attached), an explicit AppUserModel ID in the process
(verified by readback), the same ID on Start-menu and desktop shortcuts, a correctly
formatted .ico with BMP entries at small sizes, and a cleared icon cache.

The reason is simpler than any of those. A taskbar button for a windowed process
ultimately falls back to the icon resource of the executable, and every Python GUI here
is pythonw.exe. So the fix is to give the program its own executable: copy the
interpreter and replace its icon resources in place, using the Win32 resource update
API. No compiler and no extra packages are involved -- the copy is still pythonw, it
just looks like us.

    python make_exe.py
    python make_exe.py --check
"""

from __future__ import annotations

import argparse
import ctypes
import shutil
import struct
import sys
from ctypes import wintypes
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RT_ICON, RT_GROUP_ICON = 3, 14
LANG_NEUTRAL = 0


def read_ico(path: Path):
    """Splits an .ico into its directory entries and image payloads."""
    data = path.read_bytes()
    count = struct.unpack_from("<H", data, 4)[0]
    entries, images = [], []
    for i in range(count):
        off = 6 + i * 16
        width, height, colours, _, planes, bits, size, start = struct.unpack_from(
            "<BBBBHHII", data, off)
        entries.append((width, height, colours, planes, bits, size))
        images.append(data[start:start + size])
    return entries, images


def build_group(entries, first_id: int) -> bytes:
    """The RT_GROUP_ICON resource: a directory pointing at RT_ICON resource ids."""
    out = struct.pack("<HHH", 0, 1, len(entries))
    for i, (width, height, colours, planes, bits, size) in enumerate(entries):
        out += struct.pack("<BBBBHHIH", width, height, colours, 0,
                           planes, bits, size, first_id + i)
    return out


def apply_icon(exe: Path, ico: Path) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.BeginUpdateResourceW.restype = wintypes.HANDLE
    kernel32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    kernel32.UpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR,
                                         wintypes.LPCWSTR, wintypes.WORD,
                                         wintypes.LPVOID, wintypes.DWORD]
    kernel32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]

    entries, images = read_ico(ico)
    handle = kernel32.BeginUpdateResourceW(str(exe), False)
    if not handle:
        raise OSError(f"BeginUpdateResource failed: {ctypes.get_last_error()}")

    first_id = 1
    for i, payload in enumerate(images):
        ok = kernel32.UpdateResourceW(
            handle, wintypes.LPCWSTR(RT_ICON),
            ctypes.cast(first_id + i, wintypes.LPCWSTR), LANG_NEUTRAL,
            payload, len(payload))
        if not ok:
            raise OSError(f"UpdateResource(icon {i}) failed: {ctypes.get_last_error()}")

    group = build_group(entries, first_id)
    ok = kernel32.UpdateResourceW(
        handle, wintypes.LPCWSTR(RT_GROUP_ICON),
        ctypes.cast(1, wintypes.LPCWSTR), LANG_NEUTRAL, group, len(group))
    if not ok:
        raise OSError(f"UpdateResource(group) failed: {ctypes.get_last_error()}")

    if not kernel32.EndUpdateResourceW(handle, False):
        raise OSError(f"EndUpdateResource failed: {ctypes.get_last_error()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    import os
    if os.name != "nt":
        print("Windows only.")
        return 1

    src = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    # The interpreter locates its virtual environment by looking for pyvenv.cfg one
    # level above its own directory, so the copy has to live in Scripts too. Placed in
    # the project root it starts up and then fails with "failed to locate pyvenv.cfg".
    dest = ROOT / ".venv" / "Scripts" / "drum2midi.exe"
    ico = ROOT / "docs" / "drum2midi.ico"
    for path in (src, ico):
        if not path.exists():
            print(f"missing: {path}")
            return 1

    if args.check:
        print(f"source : {src}")
        print(f"target : {dest}  ({'exists' if dest.exists() else 'not built'})")
        entries, _ = read_ico(ico)
        print(f"icon   : {len(entries)} entries")
        return 0

    shutil.copy2(src, dest)
    apply_icon(dest, ico)
    print(f"built {dest}  ({dest.stat().st_size // 1024} KB)")
    print("\nIt is pythonw.exe with our icon resources, so it runs the GUI identically.")
    print("Next:  python make_shortcut.py --both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
