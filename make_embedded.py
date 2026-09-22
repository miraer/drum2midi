"""Builds a small, non-Store Python that owns its own taskbar icon.

The Microsoft Store interpreter lives under WindowsApps, and Windows binds the taskbar
button to that process no matter what icons the application sets on its window, its
window class or its launcher. The embeddable distribution is an ordinary directory of
files, so its pythonw.exe can carry our icon and become the process that owns the
window.

No packages are reinstalled: the existing virtual environment's site-packages is added
to the embedded interpreter's path, so torch, ADTOF and the rest are shared. tkinter is
not shipped with the embeddable build, so its pieces are copied from the Store install.

    python make_embedded.py
    python make_embedded.py --check
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
VERSION = "3.13.14"
URL = f"https://www.python.org/ftp/python/{VERSION}/python-{VERSION}-embed-amd64.zip"

# The window is PySide6 since the redesign, and Qt comes from the virtual environment's
# site-packages, which `._pth` already points at -- so nothing Qt needs is copied here.
# tkinter used to be, because the embeddable package omits it and the old window required
# it. The only tkinter left in the project is a `--show` preview in drum_kit_art.py, a
# developer tool that is not launched through this runtime, so the copy has gone.
#
# What stays is the DLL sweep below. Its original justification was tcl86t.dll pulling in
# the Visual C++ runtime, and that reason has gone with tkinter -- PySide6 ships its own
# msvcp140 and vcruntime140. It is kept deliberately rather than removed on the same
# reasoning that retired it: the embeddable package ships only part of the standard
# library's DLLs, the files are small, and nothing here has measured which of them torch
# or ADTOF reach for. Removing it should follow a build that is launched, not an argument.
#
# Measured on a clean rebuild: tcl/, Lib/tkinter and _tkinter.pyd are gone, and Qt --
# QtWidgets and QtMultimedia -- imports under runtime/drum2midi.exe. The sweep still
# carries tcl86t.dll across, because it copies the host's DLLs directory wholesale, so
# one 1.5 MB file remains for a toolkit nothing loads. Said plainly rather than described
# as "tkinter removed", which would be the tidier sentence and not the true one.


def base_install() -> Path:
    return Path(sys.base_prefix)


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return dest
    print(f"  fetching {URL}")
    urllib.request.urlretrieve(URL, dest)
    print(f"  {dest.stat().st_size // 1024} KB")
    return dest


def copy_support_dlls(base: Path) -> int:
    copied = 0
    # The embeddable package ships only part of the standard library's DLLs. These are
    # small and copying them avoids chasing dependencies one at a time.
    for dll in sorted((base / "DLLs").glob("*.dll")):
        target = RUNTIME / dll.name
        if not target.exists():
            shutil.copy2(dll, target)
            copied += 1
    for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        src = base / name
        target = RUNTIME / name
        if src.exists() and not target.exists():
            shutil.copy2(src, target)
            copied += 1
    return copied


def write_path_file(site_packages: Path) -> None:
    """The ._pth file decides what the embedded interpreter can import."""
    pth = next(RUNTIME.glob("python*._pth"), None)
    if pth is None:
        return
    lines = [f"python{VERSION.replace('.', '')[:3]}.zip", ".", "Lib",
             "DLLs", str(site_packages), "", "import site"]
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sitecustomize(venv: Path) -> None:
    """Registers the virtual environment's native libraries, and runs its .pth files.

    Two separate problems, both invisible until something fails at import time.

    DLLs: packages like torch+xpu ship their native libraries outside site-packages --
    here in .venv/Library/bin -- and locate them relative to sys.prefix. The embedded
    interpreter has a different prefix, so they have to be declared explicitly, before
    anything imports torch.

    .pth files: a directory listed in ._pth is put straight onto sys.path, and the .pth
    files inside it are never executed. Editable installs are nothing but a .pth file,
    so `pip install -e` packages silently vanish -- adtof_pytorch is installed that way
    and its ModuleNotFoundError is what sent us looking. site.addsitedir processes them
    properly.
    """
    body = f'''"""Added by make_embedded.py: share the virtual environment with the runtime."""
import os
import site
from pathlib import Path

_VENV = Path(r"{venv}")

for _name in ("Library/bin", "Scripts", "Library/lib"):
    _path = _VENV / _name
    if _path.is_dir():
        try:
            os.add_dll_directory(str(_path))
        except OSError:
            pass
        os.environ["PATH"] = str(_path) + os.pathsep + os.environ.get("PATH", "")

# Runs the .pth files in site-packages, which a bare ._pth entry does not. Without
# this, every `pip install -e` package is missing at runtime.
_site = _VENV / "Lib" / "site-packages"
if _site.is_dir():
    site.addsitedir(str(_site))
'''
    (RUNTIME / "sitecustomize.py").write_text(body, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    site_packages = ROOT / ".venv" / "Lib" / "site-packages"
    exe = RUNTIME / "pythonw.exe"

    if args.check:
        print(f"runtime dir   : {RUNTIME}  ({'present' if RUNTIME.exists() else 'absent'})")
        print(f"pythonw       : {'present' if exe.exists() else 'absent'}")
        print(f"site-packages : {site_packages}")
        print(f"base install  : {base_install()}")
        return 0

    print("[1/4] embeddable distribution")
    archive = download(ROOT / "work" / f"python-{VERSION}-embed.zip")
    RUNTIME.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RUNTIME)
    print(f"  extracted to {RUNTIME}")

    print("[2/4] support DLLs from the existing install")
    n = copy_support_dlls(base_install())
    print(f"  copied {n} item(s)")

    print("[3/4] path configuration")
    write_path_file(site_packages)
    write_sitecustomize(ROOT / ".venv")
    print(f"  site-packages shared from {site_packages}")
    print(f"  native DLLs from {ROOT / '.venv' / 'Library' / 'bin'}")

    print("[4/4] icon")
    sys.path.insert(0, str(ROOT))
    from make_exe import apply_icon
    ico = ROOT / "docs" / "drum2midi.ico"
    target = RUNTIME / "drum2midi.exe"
    shutil.copy2(exe, target)
    apply_icon(target, ico)
    print(f"  {target.name} carries our icon")

    print("\nTest it with:")
    print(f"    {target} drum2midi_gui.pyw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
