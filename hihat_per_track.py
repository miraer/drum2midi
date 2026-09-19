"""Which tracks does ReStem actually lose the hi-hat on?

The README says three of 23. Counting note numbers in its exports shows five files with
no hi-hat note at all, which is not the same claim: a track whose reference contains no
hi-hat either is one ReStem got right.

So this compares, per track, what the reference holds against what each system emitted.
Written after a neighbouring claim in the same paragraph turned out to be a
generalisation from one song, so the rule for this section is now that any per-track
assertion has a script behind it.

    python hihat_per_track.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
ANN = ROOT / "mdbdrums" / "MDB Drums" / "annotations" / "class"
RESTEM = ROOT / "restem_midi"
HAT_NOTES = {42, 44, 46}
# MDB's class labels for the hi-hat family
HAT_LABELS = {"HH", "CHH", "OHH", "PHH"}


def reference_hats(path: Path) -> tuple[int, int]:
    total = hats = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        total += 1
        if parts[1].upper() in HAT_LABELS:
            hats += 1
    return hats, total


def midi_hats(path: Path) -> int:
    import pretty_midi
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return -1
    return sum(1 for inst in pm.instruments for n in inst.notes if n.pitch in HAT_NOTES)


def main() -> int:
    if not ANN.exists():
        print(f"no annotations at {ANN}")
        return 1

    rows = []
    for ann in sorted(ANN.glob("*_class.txt")):
        stem = ann.name.replace("_class.txt", "")
        ref_hats, ref_total = reference_hats(ann)
        mid = RESTEM / f"{stem}_Drum.mid"
        got = midi_hats(mid) if mid.exists() else None
        rows.append((stem, ref_hats, ref_total, got))

    print(f"{'track':<28}{'ref hats':>9}{'ref all':>9}{'ReStem':>8}   verdict")
    print("-" * 74)
    lost = silent_ok = 0
    for stem, ref_hats, ref_total, got in rows:
        if got is None:
            verdict = "no export"
        elif got == 0 and ref_hats == 0:
            verdict = "correctly silent"
            silent_ok += 1
        elif got == 0 and ref_hats > 0:
            verdict = "LOST the hi-hat"
            lost += 1
        else:
            verdict = ""
        shown = "-" if got is None else str(got)
        print(f"{stem:<28}{ref_hats:>9}{ref_total:>9}{shown:>8}   {verdict}")

    print("-" * 74)
    print(f"tracks where the reference has hi-hat and ReStem emitted none: {lost}")
    print(f"tracks where neither has any (not a failure):                  {silent_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
