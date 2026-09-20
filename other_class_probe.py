"""Would an "other" class buy anything, or only look tidy?

ReStem has an Other stem and a neutral note for hits it cannot place. The obvious
question is whether we should do the same with weak percussive onsets we cannot
classify. There are two versions of that question and only one of them is measurable
here.

**As a class to be scored, MDB cannot answer it.** The corpus carries 70 `OT` onsets in
total, 0.88% of 7994, spread over 3 of the 23 recordings. That is a smaller and more
concentrated base than the tom row, which this project already refuses to draw
conclusions from. Any F-measure for an "other" class computed here would be noise with a
number attached.

**As a way of removing false positives, it is measurable, and that is what this does.**
Auxiliary percussion does not vanish when the model has no class for it -- it gets
answered with whatever class is nearest, and those answers are false positives in a
scored class. The Beatles excerpt is the known case: 32 tambourine hits, no hi-hat
anywhere in its reference, and hi-hat onsets emitted regardless. If a useful share of our
false positives sit on top of `OT` annotations, then diverting them to a neutral note
would raise precision. If they do not, an "other" class is a musical nicety with no
effect on any number we publish -- which is a fine thing to want, but a different claim.

The oracle here is deliberate and is not a proposal. It asks what a *perfect* diverter
would buy, using the reference to decide, so the answer is a ceiling. Nothing that has to
make the decision from audio can do better, and this project has watched four ideas die
in the gap between a ceiling and a mechanism.

    python other_class_probe.py
    python other_class_probe.py --ours bench/ceiling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TOL = 0.05  # the MIREX window the rest of this project scores with


def main() -> int:
    import numpy as np

    from compare_with_restem import (CLS, FAMILIES, FAMILY_PITCHES, PRETTY,
                                     load_midi, match, prf, read_ann)
    from significance import collect_tracks

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=ROOT / "bench" / "ceiling")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    totals = {f: {"tp": 0, "fp": 0, "fp_on_other": 0, "ref": 0} for f in FAMILIES}
    other_onsets = 0
    other_tracks = []
    scored = 0

    # The same recordings significance.py scores -- it requires both a class and a
    # subclass annotation, and iterating the audio directory instead picked up one more
    # and reported 8645 emitted notes against its 8504. A number is unusable until you
    # can say what produced it, and two scripts disagreeing about the corpus is the
    # cheapest way to produce one that cannot be checked.
    for wav, ann, _sub in collect_tracks():
        base = wav.stem.replace("_Drum", "")
        ann_path = CLS / f"{base}_class.txt"
        mid = args.ours / f"{wav.stem}.mid"
        if not mid.exists():
            continue
        scored += 1
        ann = read_ann(ann_path)
        others = sorted(t for t, lab in ann if lab not in FAMILIES)
        if others:
            other_onsets += len(others)
            other_tracks.append((wav.stem, len(others)))
        notes = load_midi(mid)
        oth = np.array(others)

        for f in FAMILIES:
            ref_t = sorted(t for t, lab in ann if lab == f)
            est_t = sorted(t for t, p, _ in notes if p in FAMILY_PITCHES[f])
            totals[f]["ref"] += len(ref_t)
            if not est_t:
                continue
            found, _ = match(ref_t, est_t)
            tp = int(found.sum())
            totals[f]["tp"] += tp
            # False positives defined the way precision defines them: everything emitted
            # that is not a true positive. Deriving them from match()'s index output
            # instead made the emitted total 8645 against significance.py's 8504, which
            # is the sort of disagreement that makes both numbers unusable.
            totals[f]["fp"] += len(est_t) - tp
            # Which emitted notes are *candidates* for diversion: those with no reference
            # onset of their own class nearby, but an out-of-class onset that is.
            if oth.size:
                ref_arr = np.array(ref_t) if ref_t else np.empty(0)
                near = 0
                for t in est_t:
                    if ref_arr.size and np.min(np.abs(ref_arr - t)) <= args.tol:
                        continue
                    if np.min(np.abs(oth - t)) <= args.tol:
                        near += 1
                totals[f]["fp_on_other"] += min(near, len(est_t) - tp)

    print(f"{scored} recordings scored from {args.ours}\n")
    print(f"MDB carries {other_onsets} onsets outside the five classes, "
          f"{other_onsets / 7994:.2%} of the corpus, in {len(other_tracks)} recording(s):")
    for name, n in other_tracks:
        print(f"    {name.replace('MusicDelta_', '').replace('_Drum', ''):<16}{n}")

    print(f"\nour false positives, and how many sit on one of those onsets "
          f"(+/-{args.tol * 1000:.0f} ms)")
    print(f"{'class':<10}{'emitted':>9}{'false':>8}{'on other':>10}{'share':>8}")
    print("-" * 45)
    tp_all = fp_all = on_all = 0
    for f in FAMILIES:
        t = totals[f]
        emitted = t["tp"] + t["fp"]
        share = t["fp_on_other"] / t["fp"] if t["fp"] else 0.0
        print(f"{PRETTY.get(f, f):<10}{emitted:>9}{t['fp']:>8}{t['fp_on_other']:>10}"
              f"{share:>8.1%}")
        tp_all += t["tp"]
        fp_all += t["fp"]
        on_all += t["fp_on_other"]
    print("-" * 45)
    print(f"{'total':<10}{tp_all + fp_all:>9}{fp_all:>8}{on_all:>10}"
          f"{(on_all / fp_all if fp_all else 0):>8.1%}")

    ref_all = sum(totals[f]["ref"] for f in FAMILIES)
    before = prf(tp_all, ref_all, tp_all + fp_all)
    after = prf(tp_all, ref_all, tp_all + fp_all - on_all)
    print(f"\nMICRO now                     P {before[0]:.3f}  R {before[1]:.3f}  "
          f"F1 {before[2]:.3f}")
    print(f"with every one of them diverted  P {after[0]:.3f}  R {after[1]:.3f}  "
          f"F1 {after[2]:.3f}   ({after[2] - before[2]:+.3f})")

    print(f"""
Read the second line as a ceiling and not a result. It uses the reference annotation to
decide which notes to move, so nothing that has to decide from audio can beat it, and
the gap between {after[2] - before[2]:+.3f} and what a real detector would achieve is
entirely downside.

Note also what the ceiling is made of: {on_all} notes out of {fp_all} false positives.
The rest of our false positives are not auxiliary percussion at all -- they are the model
answering the wrong drum, or answering where there is nothing. An "other" class does not
touch those, and they are the overwhelming majority.

There remains a reason to emit an unclassified note that has nothing to do with F-measure:
a drummer opening the MIDI can see that something was struck and decide for themselves,
where today the hit is either silently dropped or silently mislabelled. That is a product
argument and it should be made as one, without borrowing evidence from a scoring table
that, on this corpus, cannot support it either way.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
