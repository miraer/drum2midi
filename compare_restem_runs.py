"""Compares two ReStem trigger_events.json files event for event.

Used to work out which offline quality mode produced a stored run, when the export
itself does not record the mode. Re-transcribe the same track in a known mode and
compare: an exact match identifies the mode, a mismatch rules it out.

Also useful on its own, because it quantifies what the quality setting actually buys --
if Better and Best differ by three events on a 400-event track, that is worth knowing
before re-running 23 tracks.

    python compare_restem_runs.py stored.json new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sr = data.get("sr", 44100)
    hop = data.get("hop", 441)
    out = {}
    for stem, events in (data.get("stems") or {}).items():
        times = []
        for e in events:
            if "sample" in e:
                times.append(e["sample"] / sr)
            elif "frame" in e:
                times.append(e["frame"] * hop / sr)
        out[stem] = sorted(times)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="the stored run")
    ap.add_argument("b", type=Path, help="the new run")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="seconds within which two events count as the same")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    stems = sorted(set(A) | set(B))

    print(f"\n{'stem':<10}{'stored':>8}{'new':>8}{'matched':>9}{'only A':>8}{'only B':>8}")
    print("-" * 52)
    total_a = total_b = total_m = 0
    for stem in stems:
        a, b = A.get(stem, []), B.get(stem, [])
        used = [False] * len(b)
        matched = 0
        for t in a:
            for j, u in enumerate(b):
                if not used[j] and abs(u - t) <= args.tol:
                    used[j] = True
                    matched += 1
                    break
        total_a += len(a)
        total_b += len(b)
        total_m += matched
        print(f"{stem:<10}{len(a):>8}{len(b):>8}{matched:>9}"
              f"{len(a) - matched:>8}{len(b) - matched:>8}")

    print("-" * 52)
    print(f"{'total':<10}{total_a:>8}{total_b:>8}{total_m:>9}"
          f"{total_a - total_m:>8}{total_b - total_m:>8}")

    if total_a == total_b == total_m:
        print("\nIDENTICAL — the stored run was made in this mode")
        return 0
    share = total_m / max(total_a, 1)
    print(f"\nDIFFERENT — {share:.1%} of the stored events reappear")
    print("The stored run was not made in this mode, or the mode is not deterministic.")
    print("Run the other modes before concluding which one it was.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
