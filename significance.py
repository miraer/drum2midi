"""Do the headline numbers survive resampling? Bootstrap over MDB's 23 tracks.

A MICRO F1 computed over 7994 onsets reads as though it were precise to three decimals.
It is not. Those onsets come from 23 recordings, and the recording is the unit that
varies: one awkward track moves the total far more than one awkward hit. So the interval
has to come from resampling tracks, not onsets.

This matters most where the class is small. MDB carries 90 tom onsets in total -- 1.13%
of the set -- so any per-class tom figure, ours or ReStem's, rests on 90 examples.

Reuses compare_with_restem.py's scoring so the point estimates match the README exactly,
and only adds the interval around them.

    python significance.py
    python significance.py --rounds 10000 --ours bench/note36
    python significance.py --own-ci          # interval around our own per-class F1
    python significance.py --theirs restem_midi_better   # their weaker mode

The default --ours is bench/ceiling, the export the README's tables are built from.
bench/note36 predates TOM_CEILING and still scores toms at 0.605; pointing this script
at it reproduces every row of the README except toms, which is the one row the ceiling
moves. The directory measured is printed in the header so no log is ambiguous about it.

The default --theirs is restem_midi, which holds ReStem's **Best (Offline) + Bleed
Reduction** output -- its best mode on this material, and what the README publishes
against. restem_midi_better holds the Better (Offline) renders the first comparison used
and scores 0.820 instead of 0.834. Both are kept because the difference between them is
itself a published result, and because a script whose default quietly reproduces a
superseded number is a defect this project has already had once.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_with_restem import (  # noqa: E402
    AUDIO, CLS, FAMILIES, FAMILY_PITCHES, PRETTY, SUB, load_midi, match, prf, read_ann)


def collect_tracks():
    """The MDB recordings that have both a class and a subclass annotation."""
    tracks = []
    for wav in sorted(AUDIO.glob("*.wav")):
        base = wav.stem.replace("_Drum", "")
        c, s = CLS / f"{base}_class.txt", SUB / f"{base}_subclass.txt"
        if c.exists() and s.exists():
            tracks.append((wav, read_ann(c), read_ann(s)))
    return tracks


def per_track(midi_dir: Path, tracks) -> dict:
    """Counts tp/ref/est per family for every track, so tracks can be resampled."""
    out = {}
    for wav, cls_ann, _sub in tracks:
        mid = midi_dir / f"{wav.stem}.mid"
        if not mid.exists():
            continue
        notes = load_midi(mid)
        row = {}
        for f in FAMILIES:
            ref_t = [t for t, l in cls_ann if l == f]
            est_t = [t for t, p, _ in notes if p in FAMILY_PITCHES[f]]
            found, _ = match(ref_t, est_t)
            row[f] = {"tp": int(found.sum()), "ref": len(ref_t), "est": len(est_t)}
        out[wav.stem] = row
    return out


def f1(rows, picks, family=None) -> float:
    tp = ref = est = 0
    for name in picks:
        row = rows[name]
        for f in ([family] if family else FAMILIES):
            tp += row[f]["tp"]
            ref += row[f]["ref"]
            est += row[f]["est"]
    return prf(tp, ref, est)[2]


def interval(values, lo=0.025, hi=0.975):
    v = sorted(values)
    return v[int(lo * len(v))], v[int(hi * len(v))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--ours", type=Path, default=ROOT / "bench" / "ceiling")
    ap.add_argument("--theirs", type=Path, default=ROOT / "restem_midi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--own-ci", action="store_true",
                    help="also bootstrap our own per-class F1, not just the difference")
    args = ap.parse_args()

    tracks = collect_tracks()
    if not tracks:
        print("MDB Drums not found; see the Install section of README.md")
        return 1

    ours = per_track(args.ours, tracks)
    theirs = per_track(args.theirs, tracks)
    shared = sorted(set(ours) & set(theirs))
    # Which export produced these numbers is part of the result, not a detail.
    print(f"ours:   {args.ours}")
    print(f"theirs: {args.theirs}")
    print(f"{len(shared)} tracks scored by both systems")
    if len(shared) < 3:
        print("not enough overlap to resample")
        return 1

    onsets = {f: sum(ours[t][f]["ref"] for t in shared) for f in FAMILIES}
    total = sum(onsets.values())
    print("\nhow much evidence each class actually has")
    for f in FAMILIES:
        print(f"  {PRETTY.get(f, f):<10}{onsets[f]:>6} onsets  {100*onsets[f]/total:5.2f}%")

    rng = random.Random(args.seed)
    draws = [[shared[rng.randrange(len(shared))] for _ in shared]
             for _ in range(args.rounds)]

    print(f"\nbootstrap over {len(shared)} tracks, {args.rounds} resamples")
    print(f"{'class':<10}{'ours':>8}{'ReStem':>8}{'diff':>9}"
          f"{'95% CI on the difference':>28}   verdict")
    print("-" * 78)

    for family in [None] + list(FAMILIES):
        label = "MICRO" if family is None else PRETTY.get(family, family)
        a = f1(ours, shared, family)
        b = f1(theirs, shared, family)
        diffs = [f1(ours, d, family) - f1(theirs, d, family) for d in draws]
        lo, hi = interval(diffs)
        sig = "significant" if lo > 0 or hi < 0 else "NOT significant"
        print(f"{label:<10}{a:>8.3f}{b:>8.3f}{a - b:>+9.3f}"
              f"      [{lo:+.3f}, {hi:+.3f}]      {sig}")

    print("\nA difference whose interval contains zero is one this test set cannot\n"
          "resolve. That is a statement about the 23 recordings, not about either\n"
          "system being equal.")

    if args.own_ci:
        # The README compares MDB's tom half-width against ENST's. That comparison
        # needs the interval around our own F1, which the table above does not carry.
        print(f"\ninterval around our own F1, same {len(shared)} tracks resampled")
        print(f"{'class':<10}{'F1':>8}{'95% CI':>22}{'half-width':>13}")
        print("-" * 53)
        for family in [None] + list(FAMILIES):
            label = "MICRO" if family is None else PRETTY.get(family, family)
            point = f1(ours, shared, family)
            lo, hi = interval([f1(ours, d, family) for d in draws])
            print(f"{label:<10}{point:>8.3f}      [{lo:.3f}, {hi:.3f}]{(hi - lo) / 2:>13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
