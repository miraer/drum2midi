"""Scores ADT_STR on MDB Drums with our code, so the comparison is honest.

ADT_STR reports 0.79 SUM on MDB, but in an 8-class drum-only setting; this pipeline
reports 0.882 over 5 classes. Those two numbers say nothing about each other. The only
way to compare is to score both with the same mapping, the same annotations and the same
tolerance -- which is what benchmark_mdb.py already does for us, reused verbatim here.

One asymmetry is worth stating plainly: ADT_STR reads the drum recording directly, while
our 0.882 also comes from the drum recording, so neither side gets a separation
advantage. What differs is that ADT_STR emits 26 classes which are folded down to 5,
and folding can only help it -- a tom confused for another tom still counts as a tom.

    python score_adt_str.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

MDB = ROOT / "mdbdrums" / "MDB Drums"
ANN = MDB / "annotations" / "class"
OURS = ROOT / "bench" / "ours_midi"
THEIRS = ROOT / "bench" / "adtstr_midi"
WINDOW = 0.05

from benchmark_mdb import CLASSES, ORDER, PRETTY, read_annotation  # noqa: E402

# ADT_STR predicts 26 classes and uses General MIDI numbers our 5-class map ignores:
# Chinese cymbal, splash, ride bell, second crash. Folding all of them into "cymbals"
# can only help it, so the comparison stays fair rather than convenient.
WIDE = dict(CLASSES)
WIDE["CY"] = (49, 51, 52, 53, 55, 57, 59)
WIDE["TT"] = (41, 43, 45, 47, 48, 50)
WIDE["HH"] = (42, 44, 46, 26, 22)


def read_midi(path: Path, classes=None) -> dict:
    classes = classes or CLASSES
    notes = [n for inst in pretty_midi.PrettyMIDI(str(path)).instruments
             for n in inst.notes]
    return {k: np.array(sorted(n.start for n in notes if n.pitch in pitches))
            for k, pitches in classes.items()}


def score(pairs: list) -> dict:
    """Micro-averaged precision, recall and F1 per class, over all tracks."""
    acc = {k: {"tp": 0, "ref": 0, "est": 0} for k in CLASSES}
    for ref, est in pairs:
        for k in CLASSES:
            r, e = ref[k], est[k]
            tp = 0
            if len(r) and len(e):
                _, p, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
                tp = int(round(p * len(e)))
            acc[k]["tp"] += tp
            acc[k]["ref"] += len(r)
            acc[k]["est"] += len(e)
    return acc


def prf(c: dict) -> tuple:
    p = c["tp"] / max(c["est"], 1)
    r = c["tp"] / max(c["ref"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    if not THEIRS.exists():
        print(f"no ADT_STR output in {THEIRS}; run  python run_adt_str.py")
        return 1

    theirs_files = {p.parent.name: p for p in THEIRS.rglob("*.mid")}
    if not theirs_files:
        print("ADT_STR produced no MIDI")
        return 1

    rows = []
    for stem, mid in sorted(theirs_files.items()):
        ann = ANN / f"{stem.replace('_Drum', '')}_class.txt"
        ours = OURS / f"{stem}.mid"
        if not ann.exists():
            continue
        ref = read_annotation(ann)
        rows.append((stem, ref, read_midi(mid, WIDE),
                     read_midi(ours) if ours.exists() else None))

    if not rows:
        print("no annotated tracks matched")
        return 1

    have_ours = all(r[3] is not None for r in rows)
    print(f"{len(rows)} tracks scored with mir_eval at {WINDOW*1000:.0f} ms"
          + ("" if have_ours else "   (our MIDI cache incomplete, showing ADT_STR only)"))

    theirs = score([(r[1], r[2]) for r in rows])
    ours = score([(r[1], r[3]) for r in rows]) if have_ours else None

    head = f"{'class':<10}{'reference':>10}{'ADT_STR F1':>12}"
    if ours:
        head += f"{'ours F1':>10}{'delta':>9}"
    print("\n" + head)
    print("-" * len(head))

    micro_t = {"tp": 0, "ref": 0, "est": 0}
    micro_o = {"tp": 0, "ref": 0, "est": 0}
    for k in ORDER:
        for key in micro_t:
            micro_t[key] += theirs[k][key]
            if ours:
                micro_o[key] += ours[k][key]
        line = f"{PRETTY[k]:<10}{theirs[k]['ref']:>10}{prf(theirs[k])[2]:>12.3f}"
        if ours:
            d = prf(ours[k])[2] - prf(theirs[k])[2]
            line += f"{prf(ours[k])[2]:>10.3f}{d:>+9.3f}"
        print(line)

    print("-" * len(head))
    line = f"{'MICRO':<10}{micro_t['ref']:>10}{prf(micro_t)[2]:>12.3f}"
    if ours:
        line += f"{prf(micro_o)[2]:>10.3f}{prf(micro_o)[2] - prf(micro_t)[2]:>+9.3f}"
    print(line)

    tp, tr, tf = prf(micro_t)
    print(f"\nADT_STR precision {tp:.3f}, recall {tr:.3f}")
    if ours:
        op, orr, _ = prf(micro_o)
        print(f"ours    precision {op:.3f}, recall {orr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
