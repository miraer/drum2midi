"""Where do the false toms come from -- the transcriber, or the separator?

The second machine measured that false tom onsets land near an annotated snare far more
often than chance: 2.48x on MDB, 2.36x on ENST, while true tom onsets show nothing of the
kind. That is the tom channel answering "tom" to a snare.

Their runs were all `--no-separate`, so no separator existed in them and the confusion
can only be ADTOF's. But the shipped pipeline does separate, and the published numbers
come from separated stems. Two possibilities that the ENST work cannot distinguish:

  the separator is irrelevant here    ADTOF confuses tom with snare either way, and
                                      separation neither causes nor fixes it
  the separator matters               it either leaks snare into what ADTOF sees, making
                                      things worse, or removes it, making things better

This measures the same thing on the exported MIDI of both configurations, so the answer
is about our shipped path rather than about a research setting.

Method: for every estimated tom onset with no reference tom within the MIREX window,
ask how often it falls within `--window` of a reference snare. The null is a circular
shift of the estimated times, which preserves their rhythm and destroys their alignment,
so "near a snare" is measured against how often anything would be near a snare.

    python tom_bleed.py
    python tom_bleed.py --ours bench/ceiling --shifts 2000
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=ROOT / "bench" / "ceiling",
                    help="transcriptions to examine")
    ap.add_argument("--window", type=float, default=0.05,
                    help="seconds; the MIREX tolerance")
    ap.add_argument("--shifts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import pretty_midi

    from compare_with_restem import AUDIO, CLS, FAMILY_PITCHES, read_ann

    if not AUDIO.exists():
        print(f"MDB Drums not found at {AUDIO}")
        return 1
    print(f"scoring {args.ours}, window {args.window * 1000:.0f} ms")

    rng = random.Random(args.seed)
    near_true, near_false, n_true, n_false = 0, 0, 0, 0
    chance_hits = np.zeros(args.shifts)
    chance_n = 0

    for wav in sorted(AUDIO.glob("*.wav")):
        base = wav.stem.replace("_Drum", "")
        ann = CLS / f"{base}_class.txt"
        mid = args.ours / f"{wav.stem}.mid"
        if not (ann.exists() and mid.exists()):
            continue
        cls = read_ann(ann)
        ref_tom = np.array(sorted(t for t, lab in cls if lab == "TT"))
        ref_snare = np.array(sorted(t for t, lab in cls if lab == "SD"))
        if not ref_snare.size:
            continue

        notes = [n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                 for n in inst.notes]
        est_tom = np.array(sorted(n.start for n in notes
                                  if n.pitch in FAMILY_PITCHES["TT"]))
        if not est_tom.size:
            continue
        span = max(float(est_tom.max()), float(ref_snare.max())) + 1.0

        def near(times, targets):
            if not len(times) or not len(targets):
                return np.zeros(len(times), dtype=bool)
            idx = np.searchsorted(targets, times)
            out = np.zeros(len(times), dtype=bool)
            for k, t in enumerate(times):
                j = idx[k]
                for cand in (j - 1, j):
                    if 0 <= cand < len(targets) and abs(targets[cand] - t) <= args.window:
                        out[k] = True
                        break
            return out

        # An estimate is "false" when no reference tom is within the window of it.
        is_true = near(est_tom, ref_tom) if ref_tom.size else np.zeros(len(est_tom), bool)
        false_tom = est_tom[~is_true]
        true_tom = est_tom[is_true]

        n_false += len(false_tom)
        n_true += len(true_tom)
        near_false += int(near(false_tom, ref_snare).sum())
        near_true += int(near(true_tom, ref_snare).sum())

        if len(false_tom):
            chance_n += len(false_tom)
            for s in range(args.shifts):
                shifted = np.sort((false_tom + rng.uniform(0, span)) % span)
                chance_hits[s] += near(shifted, ref_snare).sum()

    if not n_false:
        print("no false tom onsets to examine")
        return 1

    obs = near_false / n_false
    chance = float(chance_hits.mean()) / max(chance_n, 1)
    lo, hi = np.percentile(chance_hits / max(chance_n, 1), [2.5, 97.5])
    print(f"\n{n_false} false tom onsets, {n_true} true ones")
    print(f"  false toms near a snare   {obs:.1%}")
    print(f"  chance, {args.shifts} circular shifts   {chance:.1%}  "
          f"[{lo:.1%}, {hi:.1%}]")
    print(f"  lift                      {obs / max(chance, 1e-9):.2f}x")
    if n_true:
        print(f"  true toms near a snare    {near_true / n_true:.1%}  "
              f"(the control: real toms should not show this)")
    verdict = "outside" if obs > hi or obs < lo else "inside"
    print(f"\nObserved is {verdict} the null interval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
