"""Which drums are in ReStem's own MIDI export, and which are only in ours?

Comparing the two across 23 tracks, their export holds 3628 notes against our 9168 from
the same trigger JSON -- but 3623 of their 3628 are present in ours. That is not the
shape of a filter dropping quiet hits; it is the shape of a subset.

ReStem's guide says MIDI OUT is per stem and starts switched off in the standalone
application, so an export contains only the stems that were enabled. If the missing notes
group cleanly by drum, that is the explanation, and our conversion is faithful for
everything present. If they scatter across all drums, it is a filter and the two are not
interchangeable.

Read-only. Prints per-pitch counts so the answer is visible rather than asserted.

    python restem_export_pitches.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DEFAULT_THEIRS = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "ReStem 2" / "midi-drag"

NAME = {35: "kick", 36: "kick", 38: "snare", 40: "snare", 41: "floor tom",
        42: "closed hat", 43: "floor tom", 44: "pedal hat", 45: "low tom",
        46: "open hat", 47: "mid tom", 48: "high tom", 49: "crash", 50: "high tom",
        51: "ride", 52: "china", 53: "ride bell", 55: "splash", 57: "crash",
        59: "ride", 60: "other"}


def pitches(path: Path) -> Counter:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    return Counter(n.pitch for inst in pm.instruments for n in inst.notes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theirs", type=Path, default=DEFAULT_THEIRS)
    ap.add_argument("--ours", type=Path, default=ROOT / "restem_midi")
    ap.add_argument("--on", default="17.09")
    args = ap.parse_args()

    theirs: dict[str, Path] = {}
    for f in args.theirs.rglob("MusicDelta_*.mid"):
        day = dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%d.%m")
        if day != args.on:
            continue
        stem = f.name.replace("_midi.mid", "")
        prev = theirs.get(stem)
        if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
            theirs[stem] = f

    ours_total, theirs_total = Counter(), Counter()
    per_track = []
    for stem, path in sorted(theirs.items()):
        o = args.ours / f"{stem}.mid"
        if not o.exists():
            continue
        po, pt = pitches(o), pitches(path)
        ours_total += po
        theirs_total += pt
        per_track.append((stem, set(po), set(pt)))

    print(f"{'pitch':>6}  {'drum':<12}{'ours':>8}{'theirs':>9}   present in their export")
    print("-" * 66)
    for p in sorted(set(ours_total) | set(theirs_total)):
        mark = "yes" if theirs_total[p] else "NO"
        print(f"{p:>6}  {NAME.get(p, '?'):<12}{ours_total[p]:>8}{theirs_total[p]:>9}   {mark}")
    print("-" * 66)
    print(f"{'':>6}  {'TOTAL':<12}{sum(ours_total.values()):>8}"
          f"{sum(theirs_total.values()):>9}")

    missing = {p for p in ours_total if not theirs_total[p]}
    extra = {p for p in theirs_total if not ours_total[p]}
    print()
    if missing:
        names = ", ".join(f"{p} ({NAME.get(p, '?')})" for p in sorted(missing))
        print(f"Absent from every export: {names}")
    if extra:
        names = ", ".join(f"{p} ({NAME.get(p, '?')})" for p in sorted(extra))
        print(f"In their export but never in ours: {names}")

    # does the same set of drums go missing on every track, or does it vary
    sets = {frozenset(t) for _, _, t in per_track}
    print(f"\ndistinct pitch sets across their {len(per_track)} exports: {len(sets)}")
    if len(sets) > 1:
        print("  the set varies by track, so it is not one fixed configuration")
        for stem, o, t in per_track:
            gone = sorted(o - t)
            if gone:
                print(f"  {stem[:28]:<30} missing {', '.join(str(x) for x in gone)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
