"""Where does ReStem's kick go when Bleed Reduction removes it?

The first reading of this failure was that the option nulls the kick channel: on
drummer 1's ENST recordings, Best + Bleed Reduction emits 16 kick events against
1238 annotated onsets. That reading was wrong in a way that matters to anyone
trying to fix it.

ReStem exports one MIDI pitch per stem, and 60 is its "other" stem. Counting note-60
events shows them arriving in roughly the quantity the missing kick should have, and
matching them against the annotation shows them arriving at the same *times*: 90% of
drummer 1's note-60 events land within the 50 ms window of an annotated kick onset.
Shifting those events in time collapses the agreement to 21-29%, which is what rules
out "there are simply many events, so some coincide".

So the kick is detected and then attributed to the wrong stem. This script reports
that, paired where both modes exist for the same recording, because the paired form
is the only one that isolates the option from the material.

    python restem_kick_misroute.py
    python restem_kick_misroute.py --shift 0.25    # the null control
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pretty_midi

from benchmark_enst import IGNORED, LABEL_TO_CLASS

ROOT = Path(__file__).resolve().parent
ENST = ROOT / "enst" / "enst_drums_public"
WINDOW = 0.05
KICK_PITCHES = (35, 36)
OTHER_PITCH = 60  # ReStem's "other" stem, per its own stem-to-pitch mapping

ARMS = {
    "Bleed Reduction on": ROOT / "bench" / "enst_bestplus_renders",
    "off": ROOT / "restem_export",
}


def annotated_kicks(name: str) -> tuple[np.ndarray | None, int | None]:
    for d in (1, 2, 3):
        path = ENST / f"drummer_{d}" / "annotation" / f"{name}.txt"
        if not path.exists():
            continue
        times = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in IGNORED:
                if LABEL_TO_CLASS.get(parts[1]) == "KD":
                    times.append(float(parts[0]))
        return np.array(sorted(times)), d
    return None, None


def matched(ref: np.ndarray, est: np.ndarray) -> int:
    if not len(ref) or not len(est):
        return 0
    used = np.zeros(len(est), dtype=bool)
    tp = 0
    for t in ref:
        near = np.where((~used) & (np.abs(est - t) <= WINDOW))[0]
        if len(near):
            used[near[np.argmin(np.abs(est[near] - t))]] = True
            tp += 1
    return tp


def pitches_of(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    kick = np.array(sorted(n.start for n in notes if n.pitch in KICK_PITCHES))
    other = np.array(sorted(n.start for n in notes if n.pitch == OTHER_PITCH))
    return kick, other


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shift", type=float, default=0.0,
                    help="displace the note-60 events by this many seconds; the "
                         "agreement should collapse if it is real (default: 0)")
    args = ap.parse_args()

    missing = [n for n, p in ARMS.items() if not p.is_dir()]
    if missing:
        print(f"no renders for: {', '.join(missing)}", file=sys.stderr)
        return 1

    both = sorted(set(p.name for p in ARMS["Bleed Reduction on"].iterdir() if p.is_dir())
                  & set(p.name for p in ARMS["off"].iterdir() if p.is_dir()))
    if not both:
        print("no recording exists in both modes -- nothing can be paired", file=sys.stderr)
        return 1

    if args.shift:
        print(f"NULL CONTROL: note-60 events displaced by {args.shift:+.2f}s\n")
    print(f"{len(both)} recording(s) rendered in both modes, so the option is the only "
          f"thing that differs\n")
    print(f"{'recording':<40}{'drm':>4}{'ref':>5}"
          f"{'on: kick':>10}{'other':>7}{'off: kick':>11}{'other':>7}")
    print("-" * 84)

    tot = {arm: {"ref": 0, "kick": 0, "hit": 0} for arm in ARMS}
    for name in both:
        ref, drummer = annotated_kicks(name)
        if ref is None or not len(ref):
            continue
        row = f"{name[:39]:<40}{drummer:>4}{len(ref):>5}"
        for arm, base in ARMS.items():
            mid = base / name / f"{name}_midi.mid"
            if not mid.exists():
                continue
            kick, other = pitches_of(mid)
            hit = matched(ref, other + args.shift)
            tot[arm]["ref"] += len(ref)
            tot[arm]["kick"] += len(kick)
            tot[arm]["hit"] += hit
            width = (10, 7) if arm == "Bleed Reduction on" else (11, 7)
            row += f"{len(kick):>{width[0]}}{len(other):>{width[1]}}"
        print(row)

    print("-" * 84)
    for arm in ARMS:
        t = tot[arm]
        r = max(t["ref"], 1)
        print(f"{arm:<20} kick on 35/36: {t['kick']}/{t['ref']} ({t['kick'] / r:.0%})"
              f"   annotated kicks landing on note 60: {t['hit']} ({t['hit'] / r:.0%})")

    on, off = tot["Bleed Reduction on"], tot["off"]
    print()
    moved = on["kick"] < 0.5 * on["ref"] <= off["kick"] and on["hit"] > off["hit"]
    if args.shift:
        # Under the null control the verdict must not be reachable. An earlier version
        # printed it anyway, because displaced events still agree with the annotation
        # more often in one arm than the other -- which is a statement about how many
        # events each arm emits, not about where they fall.
        print(f"Null control: with the events displaced by {args.shift:+.2f}s, "
              f"agreement is {on['hit'] / max(on['ref'], 1):.0%} with the option on "
              f"and {off['hit'] / max(off['ref'], 1):.0%} with it off.")
        print("Compare against the undisplaced run; this arm of the test is meant to")
        print("fail, and no conclusion is drawn from it.")
    elif moved:
        print("The kick leaves pitch 35/36 and appears on pitch 60 when the option is on,")
        print("on the same audio. It is being attributed to the wrong stem, not lost.")
    else:
        print("The paired contrast does not show the kick moving to pitch 60 here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
