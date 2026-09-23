"""Creates a desktop/Start-menu shortcut with the right icon and identity.

Setting the window icon is not enough on Windows. The taskbar button is matched to an
Application User Model ID, and when a script is launched through pythonw.exe the shell
uses the interpreter's identity and therefore its icon. Declaring the ID inside the
process fixes grouping, but the *pinned* icon comes from the shortcut that launched it.

So the shortcut has to carry the same ID: System.AppUserModel.ID, property key
{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3},5. That is what makes Windows treat the window
and the shortcut as the same application.

    python make_shortcut.py                 # onto the desktop
    python make_shortcut.py --start-menu

On Linux it writes a freedesktop entry instead, so drum2midi appears in the
application menu with its icon, launched through drum2midi.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
APP_ID = "drum2midi.gui"


def build_with_powershell(target_dir: Path, pythonw: Path, script: Path,
                          icon: Path) -> Path:
    """WScript.Shell writes the .lnk; the AppUserModel ID is added afterwards."""
    import subprocess

    lnk = target_dir / "drum2midi.lnk"
    ps = f'''
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("{lnk}")
$s.TargetPath = "{pythonw}"
$s.Arguments = '"{script}"'
$s.WorkingDirectory = "{ROOT}"
$s.IconLocation = "{icon},0"
$s.Description = "drum recordings to General MIDI"
$s.Save()
Write-Output "created"
'''
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True)
    if "created" not in (res.stdout or ""):
        print((res.stderr or res.stdout).strip()[:300])
        raise RuntimeError("could not create the shortcut")
    return lnk


def set_app_id(lnk: Path, app_id: str, attempts: int = 6) -> bool:
    """Writes System.AppUserModel.ID into the shortcut's property store.

    Retried because the COM object that just wrote the .lnk can still hold the file
    for a moment after the helper process exits.
    """
    import time

    try:
        from win32com.propsys import propsys, pscon
        from win32com.shell import shellcon
    except ImportError:
        return False
    # GPS_READWRITE lives in shellcon in some pywin32 builds and propsys in others
    flags = (getattr(pscon, "GPS_READWRITE", None)
             or getattr(shellcon, "GPS_READWRITE", None) or 0x00000002)
    last = None
    for attempt in range(attempts):
        try:
            store = propsys.SHGetPropertyStoreFromParsingName(
                str(lnk), None, flags, propsys.IID_IPropertyStore)
            store.SetValue(pscon.PKEY_AppUserModel_ID,
                           propsys.PROPVARIANTType(app_id))
            store.Commit()
            return True
        except Exception as exc:
            last = exc
            time.sleep(0.5 * (attempt + 1))
    print(f"  (could not write the ID: {last})")
    return False


def desktop_entry(root: Path) -> str:
    """The freedesktop .desktop entry for a checkout at `root` (Linux menus)."""
    def quoted(path: Path) -> str:
        # Two layers, per the spec: an Exec argument in double quotes escapes " ` $
        # and backslash with a backslash, and then the value as a whole is a string,
        # whose own escaping doubles every backslash -- so a literal backslash ends up
        # as four.
        text = path.as_posix()
        for ch in ("\\", '"', "`", "$"):
            text = text.replace(ch, "\\" + ch)
        return '"' + text.replace("\\", "\\\\") + '"'
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=drum2midi",
        "Comment=Drum recordings to General MIDI",
        f"Exec={quoted(root / 'drum2midi.sh')} %f",
        f"Icon={(root / 'docs' / 'logo_256.png').as_posix()}",
        "Terminal=false",
        "Categories=AudioVideo;Audio;Music;",
        "MimeType=audio/x-wav;audio/wav;audio/flac;audio/mpeg;audio/ogg;",
        "",
    ])


def install_linux_entry() -> int:
    """Writes the entry to ~/.local/share/applications, where menus look."""
    apps = Path.home() / ".local" / "share" / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    target = apps / "drum2midi.desktop"
    target.write_text(desktop_entry(ROOT), encoding="utf-8")
    target.chmod(0o755)
    print(f"menu entry -> {target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-menu", action="store_true",
                    help="place it in the Start menu instead of the desktop")
    ap.add_argument("--both", action="store_true",
                    help="Start menu and desktop; the Start menu copy is the one "
                         "Windows matches against the AppUserModel ID")
    ap.add_argument("--clear-icon-cache", action="store_true",
                    help="delete the shell's cached icons and restart Explorer")
    args = ap.parse_args()

    import os
    if sys.platform.startswith("linux"):
        return install_linux_entry()
    if os.name != "nt":
        print("On macOS, start the window with ./drum2midi.sh; there is no menu entry "
              "to make.")
        return 1

    pythonw = ROOT / ".venv" / "Scripts" / "drum2midi.exe"
    if not pythonw.exists():
        pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        print("note: drum2midi.exe not built, falling back to pythonw.exe\n"
              "      (the taskbar will show Python's icon; run make_exe.py first)")
    script = ROOT / "drum2midi_gui.pyw"
    icon = ROOT / "docs" / "drum2midi.ico"
    for path in (pythonw, script, icon):
        if not path.exists():
            print(f"missing: {path}")
            return 1

    desktop = Path(os.environ["USERPROFILE"]) / "Desktop"
    start_menu = (Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" /
                  "Start Menu" / "Programs")

    # Windows resolves an AppUserModel ID to an icon by looking for a Start menu
    # shortcut carrying that ID. A desktop shortcut is never consulted, which is why
    # the taskbar keeps showing the interpreter's icon without this one.
    targets = [start_menu]
    if args.both or not args.start_menu:
        targets.append(desktop)

    made = []
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        lnk = build_with_powershell(target, pythonw, script, icon)
        ok = set_app_id(lnk, APP_ID)
        made.append((lnk, ok))
        print(f"created {lnk}" + ("" if ok else "   (AppUserModel.ID not written)"))

    if args.clear_icon_cache:
        clear_icon_cache()

    print(f"\nAppUserModel.ID: {APP_ID}, declared by both the window and the shortcut.")
    print("Launch from the Start menu entry. A pinned button keeps its old icon until\n"
          "it is unpinned and pinned again -- Windows caches those per identity.")
    return 0


def clear_icon_cache() -> None:
    """Deletes the shell icon caches and restarts Explorer.

    Restarting Explorer alone does not help: the caches are files under
    %LOCALAPPDATA%, and the shell reloads them on start.
    """
    import os
    import subprocess

    local = Path(os.environ["LOCALAPPDATA"])
    patterns = [local / "IconCache.db",
                *(local / "Microsoft" / "Windows" / "Explorer").glob("iconcache*.db")]
    print("\nclearing the icon cache:")
    subprocess.run(["taskkill", "/F", "/IM", "explorer.exe"],
                   capture_output=True, text=True)
    removed = 0
    for path in patterns:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    print(f"  removed {removed} cache file(s)")
    subprocess.Popen(["explorer.exe"])
    print("  Explorer restarted")


if __name__ == "__main__":
    raise SystemExit(main())
