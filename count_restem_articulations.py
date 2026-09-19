"""Count hi-hat articulations in ReStem's exported MIDI, per file.

The letter to ReStem asserts that across 23 recordings their export contained no open
hi-hat at all. The MDB comparison says otherwise -- it scores their open hi-hats at
37 correct of 251 -- and both statements cannot be true. One of them is in a letter about
to be sent to the people who wrote the software, so it is worth counting rather than
arguing.

Note numbers only. ReStem's own default mapping is 42 closed, 44 pedal, 46 open, which
its user's guide documents.

    python count_restem_articulations.py
    python count_restem_articulations.py --dir bench/restem_midi
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
HAT = {42: "closed", 44: "pedal", 46: "open"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None,
                    help="directory of ReStem .mid exports")
    args = ap.parse_args()

    import pretty_midi

    roots = [args.dir] if args.dir else [
        ROOT / "bench" / "restem", ROOT / "bench" / "restem_midi",
        ROOT / "restem_out", ROOT / "bench" / "restem_out"]
    files: list[Path] = []
    for r in roots:
        if r and r.exists():
            files += sorted(r.rglob("*.mid"))
    if not files:
        print("no ReStem MIDI found in: " + ", ".join(str(r) for r in roots))
        return 1

    total = Counter()
    per_file_open = []
    print(f"{'file':<38}{'closed':>8}{'pedal':>7}{'open':>6}")
    print("-" * 59)
    for f in files:
        c = Counter()
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception as exc:
            print(f"  {f.name:<36} unreadable: {exc}")
            continue
        for inst in pm.instruments:
            for n in inst.notes:
                if n.pitch in HAT:
                    c[HAT[n.pitch]] += 1
        total.update(c)
        if c["open"]:
            per_file_open.append((f.name, c["open"]))
        print(f"{f.name[:37]:<38}{c['closed']:>8}{c['pedal']:>7}{c['open']:>6}")

    print("-" * 59)
    print(f"{'TOTAL over ' + str(len(files)) + ' files':<38}"
          f"{total['closed']:>8}{total['pedal']:>7}{total['open']:>6}")
    if per_file_open:
        print(f"\nfiles containing at least one open hi-hat: {len(per_file_open)}")
        for name, n in per_file_open:
            print(f"  {name}  {n}")
    else:
        print("\nNo note 46 in any file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
