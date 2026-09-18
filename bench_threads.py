"""How many CPU threads should the transcriber actually use?

`torch.get_num_threads()` defaults to the physical core count, which is the right
answer on a uniform CPU. Modern Intel laptop chips are not uniform: a Core Ultra 7 165H
has 6 performance cores (12 threads), 8 efficient cores and 2 low-power efficient cores.
Work is split evenly across threads and then joined, so the slowest thread sets the
pace -- handing a share of a matrix to an E-core can make the whole layer wait for it.

This times the real ADTOF model at several thread counts and reports the best.

    python bench_threads.py
    python bench_threads.py --audio input/demo.wav --repeats 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

# Run each thread count in its own process: torch caches its thread pool, so changing
# the count after the first parallel op silently does nothing.
CHILD = r"""
import sys, time, torch
torch.set_num_threads({threads})
sys.path.insert(0, r"{root}")
from drum2midi import _adtof_model
from adtof_pytorch import load_audio_for_model
model = _adtof_model("cpu")
x = load_audio_for_model(r"{audio}")
with torch.no_grad():
    model(x)
    best = float("inf")
    for _ in range({repeats}):
        t0 = time.perf_counter()
        model(x)
        best = min(best, time.perf_counter() - t0)
print(f"RESULT {{best:.4f}} {{torch.get_num_threads()}}")
"""


def run(threads: int, audio: Path, repeats: int) -> float:
    code = CHILD.format(threads=threads, root=ROOT, audio=audio, repeats=repeats)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    for line in res.stdout.splitlines():
        if line.startswith("RESULT"):
            return float(line.split()[1])
    tail = (res.stderr or "").strip().splitlines()
    print(f"  {threads} threads: failed ({tail[-1][:90] if tail else 'no output'})")
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default=str(ROOT / "input" / "demo.wav"))
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"not found: {audio}")
        return 1

    import os
    logical = os.cpu_count() or 8
    counts = sorted({1, 2, 4, 6, 8, 12, 16, logical})
    counts = [c for c in counts if c <= logical]

    import torch
    print(f"torch default: {torch.get_num_threads()} threads, {logical} logical CPUs")
    print(f"audio: {audio.name}\n")

    times = {}
    for c in counts:
        times[c] = run(c, audio, args.repeats)
        if times[c] == times[c]:
            print(f"  {c:>3} threads: {times[c]*1000:>8.1f} ms")

    ok = {c: t for c, t in times.items() if t == t}
    if not ok:
        return 1
    best = min(ok, key=ok.get)
    default = torch.get_num_threads()
    print(f"\nfastest: {best} threads ({ok[best]*1000:.1f} ms)")
    if default in ok:
        gain = ok[default] / ok[best]
        print(f"default ({default}): {ok[default]*1000:.1f} ms "
              f"-> {gain:.2f}x slower than best" if gain > 1.02
              else f"default ({default}) is already within 2% of the best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
