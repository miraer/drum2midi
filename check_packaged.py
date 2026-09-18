"""Is the interpreter running as a packaged (MSIX/Store) app?

That single fact explains a taskbar icon that refuses to change. A packaged process has
an identity assigned by its AppX manifest, and Windows uses that identity -- and its
icon -- for the taskbar button. SetCurrentProcessExplicitAppUserModelID is ignored for
such processes, which is why replacing window icons, class icons and executable
resources all had no effect.

    python check_packaged.py
"""

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

APPMODEL_ERROR_NO_PACKAGE = 15700


def package_family(pid: int | None = None) -> str | None:
    handle = kernel32.GetCurrentProcess() if pid is None else \
        kernel32.OpenProcess(0x1000, False, pid)
    length = ctypes.c_uint32(0)
    rc = kernel32.GetPackageFamilyName(handle, ctypes.byref(length), None)
    if rc == APPMODEL_ERROR_NO_PACKAGE:
        return None
    buf = ctypes.create_unicode_buffer(length.value)
    rc = kernel32.GetPackageFamilyName(handle, ctypes.byref(length), buf)
    return buf.value if rc == 0 else None


def app_model_id(pid: int | None = None) -> str | None:
    handle = kernel32.GetCurrentProcess() if pid is None else \
        kernel32.OpenProcess(0x1000, False, pid)
    length = ctypes.c_uint32(0)
    rc = kernel32.GetApplicationUserModelId(handle, ctypes.byref(length), None)
    if rc == APPMODEL_ERROR_NO_PACKAGE:
        return None
    buf = ctypes.create_unicode_buffer(length.value)
    rc = kernel32.GetApplicationUserModelId(handle, ctypes.byref(length), buf)
    return buf.value if rc == 0 else None


print(f"interpreter : {sys.executable}")
print(f"base prefix : {sys.base_prefix}")

family = package_family()
aumid = app_model_id()
if family or aumid:
    print("\nThis interpreter IS a packaged app:")
    print(f"  package family : {family}")
    print(f"  application id : {aumid}")
    print("\nWindows assigns the taskbar icon from that package identity, and ignores\n"
          "SetCurrentProcessExplicitAppUserModelID. No amount of window, class or\n"
          "executable icon work will change the taskbar button.")
    print("\nFix: use a non-packaged Python (python.org installer) for the venv.")
else:
    print("\nNot a packaged app -- the taskbar icon is controllable from the process.")
