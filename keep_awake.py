"""Keeps Windows awake while a long run is in progress.

Modern Standby puts the machine to sleep after a few idle minutes even while a batch
job is running, because a CPU-bound background process does not count as activity. An
overnight benchmark then makes a few minutes of progress per hour.

SetThreadExecutionState is the right tool: the request lives only as long as this
process, so it cannot leave the machine permanently awake the way a power-plan edit
would. Used as a context manager, or standalone with --pid to guard a running job.

    with keep_awake():
        long_running_thing()

    python keep_awake.py --pid 12345
"""

from __future__ import annotations

import ctypes
import sys
import time
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def _set(flags: int) -> bool:
    if not sys.platform.startswith("win"):
        return False
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))


@contextmanager
def keep_awake(reason: str = ""):
    """Blocks idle sleep for the duration of the block. A no-op off Windows."""
    # AWAYMODE keeps the machine working with the screen off; it is refused on some
    # systems, so fall back to a plain system request rather than giving up.
    ok = _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
    if not ok:
        ok = _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield ok
    finally:
        if ok:
            _set(ES_CONTINUOUS)


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]) or "--pid" not in sys.argv:
        print(__doc__)
        return 0

    pid = int(sys.argv[sys.argv.index("--pid") + 1])
    import subprocess

    def alive() -> bool:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out

    if not alive():
        print(f"pid {pid} is not running")
        return 1

    with keep_awake() as ok:
        print(f"holding the machine awake for pid {pid} "
              f"({'granted' if ok else 'REFUSED by the OS'})", flush=True)
        while alive():
            time.sleep(20)
    print(f"pid {pid} exited; sleep allowed again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
