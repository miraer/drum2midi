"""Does ReStem's Bleed Reduction remove kick events on MDB, as it does on ENST?

On ENST the option nulls the kick channel of one drummer entirely while leaving
another drummer untouched, so the effect is an interaction rather than a property
of the setting. MDB is the corpus our published comparison rests on, and there the
two modes differ by 0.048 in MICRO F-measure -- a number small enough to hide a
collapsed class. This counts kick events in both modes per track, so a collapse
would be visible as such instead of being absorbed into an aggregate.

Both render sets already exist on disk; nothing is re-rendered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from benchmark_mdb import ANN, CLASSES, WINDOW, read_annotation, read_estimate

ROOT = Path(__file__).resolve().parent
ARMS = {
    "Best+ (Bleed Reduction on)": ROOT / "restem_midi",
    "Better (off)": ROOT / "restem_midi_better",
}


def matched(ref: np.ndarray, est: np.ndarray) -> int:
    """Greedy one-to-one match inside the MIREX tolerance, as benchmark_mdb does."""
    used = np.zeros(len(est), dtype=bool)
    tp = 0
    for t in ref:
        near = np.where((~used) & (np.abs(est - t) <= WINDOW))[0]
        if len(near):
            used[near[np.argmin(np.abs(est[near] - t))]] = True
            tp += 1
    return tp


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("drum_class", nargs="?", default="KD", choices=sorted(CLASSES),
                     help="which class to count (default: KD)")
    args = ap_.parse_args()
    cls = args.drum_class

    missing = [name for name, path in ARMS.items() if not path.is_dir()]
    if missing:
        print(f"no renders for: {', '.join(missing)}", file=sys.stderr)
        return 1

    tracks = sorted(p.stem for p in ARMS["Better (off)"].glob("*.mid"))
    if not tracks:
        print("no MIDI in the reference arm", file=sys.stderr)
        return 1

    print(f"ReStem on MDB, {cls} events, both modes ({len(tracks)} tracks)\n")
    print(f"{'track':<34}{'ref':>6}{'Best+':>8}{'Better':>8}{'Bt+ hit':>9}{'Btr hit':>9}")
    print("-" * 74)

    tot = {"ref": 0}
    per_arm = {a: {"est": 0, "tp": 0} for a in ARMS}
    collapsed: list[str] = []

    for t in tracks:
        # renders are named <track>_Drum.mid; annotations are <track>_class.txt
        ann = ANN / f"{t[:-5] if t.endswith('_Drum') else t}_class.txt"
        if not ann.exists():
            print(f"no annotation for {t}", file=sys.stderr)
            continue
        ref = read_annotation(ann)[cls]
        if not len(ref):
            continue
        tot["ref"] += len(ref)
        row = {}
        for arm, base in ARMS.items():
            mid = base / f"{t}.mid"
            est = read_estimate(mid)[cls] if mid.exists() else np.array([])
            tp = matched(ref, est)
            per_arm[arm]["est"] += len(est)
            per_arm[arm]["tp"] += tp
            row[arm] = (len(est), tp)
        a, b = row["Best+ (Bleed Reduction on)"], row["Better (off)"]
        # A collapse is the ENST signature: the class all but vanishes in one arm
        # while the other keeps it. Flag it per track; do not let a mean hide it.
        if b[1] >= 5 and a[1] <= 0.2 * b[1]:
            collapsed.append(t)
        mark = "  <-- collapse" if t in collapsed else ""
        name = t if len(t) <= 33 else t[:30] + "..."
        print(f"{name:<34}{len(ref):>6}{a[0]:>8}{b[0]:>8}"
              f"{a[1] / max(len(ref), 1):>8.0%}{b[1] / max(len(ref), 1):>8.0%}{mark}")

    print("-" * 74)
    r = tot["ref"]
    if not r:
        # An empty table must not be allowed to read as a reassuring negative:
        # the first version of this script printed "no track collapses" from
        # zero matched tracks, which is the failure it exists to catch.
        print("no track matched an annotation -- nothing was measured",
              file=sys.stderr)
        return 1
    ap, bp = per_arm["Best+ (Bleed Reduction on)"], per_arm["Better (off)"]
    print(f"{'total':<34}{r:>6}{ap['est']:>8}{bp['est']:>8}"
          f"{ap['tp'] / max(r, 1):>8.0%}{bp['tp'] / max(r, 1):>8.0%}")

    print()
    if collapsed:
        print(f"{len(collapsed)} track(s) collapse in Best+: {', '.join(collapsed)}")
        print("This is the ENST signature and it reproduces on MDB.")
    else:
        print(f"No track collapses. The ENST {cls} failure does not reproduce on MDB,")
        print("so the published MDB comparison is not measuring that bug.")
        d = (bp["tp"] - ap["tp"]) / max(r, 1)
        print(f"Recall difference across the corpus: {d:+.1%} "
              f"(Better minus Best+), on {r} annotated onsets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
