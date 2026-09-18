"""Counts the performance cores of a hybrid CPU, and sets torch's thread count from it.

PyTorch defaults to one thread per physical core. That is right on a uniform CPU and
wrong on Intel's hybrid laptop parts: a Core Ultra 7 165H reports 16 cores, but only 6
of them are performance cores -- the rest are efficient cores that run the same work
several times slower. Parallel ops split the work evenly and then join, so a share
landing on an E-core makes the whole layer wait for it.

Measured on this machine with the real ADTOF model, on an otherwise busy system:

    threads    1      2      6      8     12     16     22
    ms      5673   2099   2089   4097   4945   5565   7941

The default (16) was 2.66x slower than 6, and using every logical CPU was worse than
using one. So the count is derived from the hardware instead of accepting the default.

Windows exposes an EfficiencyClass per core through GetLogicalProcessorInformationEx;
the highest class is the performance one. Everywhere else this falls back to torch's
own default, which is already correct on uniform CPUs.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from functools import lru_cache

RELATION_PROCESSOR_CORE = 0


@lru_cache(maxsize=1)
def performance_cores() -> int | None:
    """Number of performance cores, or None when that cannot be determined."""
    if not sys.platform.startswith("win"):
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        length = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE, None, ctypes.byref(length))
        buf = (ctypes.c_byte * length.value)()
        if not kernel32.GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE, buf, ctypes.byref(length)):
            return None
    except Exception:
        return None

    # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: DWORD Relationship, DWORD Size, then a
    # PROCESSOR_RELATIONSHIP whose second byte is EfficiencyClass. Walking the variable
    # length records by Size avoids declaring the whole nested structure.
    classes = []
    offset = 0
    total = length.value
    while offset + 8 <= total:
        size = int.from_bytes(bytes(buf[offset + 4:offset + 8]), "little")
        if size <= 0 or offset + size > total:
            break
        classes.append(buf[offset + 9] & 0xFF)   # EfficiencyClass
        offset += size

    if not classes:
        return None
    top = max(classes)
    count = sum(1 for c in classes if c == top)
    # A uniform CPU reports one class for everything; there is nothing to correct.
    return count if count != len(classes) else None


def best_thread_count() -> int:
    """Threads to use for CPU inference."""
    override = os.environ.get("DRUM2MIDI_THREADS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    cores = performance_cores()
    if cores:
        return cores
    try:
        import torch
        return torch.get_num_threads()
    except Exception:
        return os.cpu_count() or 4


def apply(verbose: bool = False) -> int:
    """Applies the thread count to torch. Returns what was set."""
    import torch

    want = best_thread_count()
    before = torch.get_num_threads()
    if want != before:
        torch.set_num_threads(want)
    if verbose and want != before:
        print(f"      CPU threads: {want} (performance cores) instead of {before}")
    return want


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        raise SystemExit(0)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"logical CPUs       : {os.cpu_count()}")
    print(f"performance cores  : {performance_cores()}")
    try:
        import torch
        print(f"torch default      : {torch.get_num_threads()}")
    except Exception as exc:
        print(f"torch default      : unavailable ({exc})")
    print(f"will use           : {best_thread_count()}")
