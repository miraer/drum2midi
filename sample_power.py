"""Before spending 13 hours rendering, ask what interval those recordings would buy.

The second machine has proposed a pre-declared 60-recording sample of ENST for a second
ReStem comparison, because 210 costs 30-45 hours on a trial that lapses on 26 September.
The question nobody had asked is whether 60 recordings can resolve the effect at all. A
run that ends in an interval straddling zero has cost the window and settled nothing, and
this project's own rule -- no difference is real without an interval -- has an obvious
corollary: know the interval you are buying before you buy it.

This measures the spread that is actually there, rather than assuming one. It scores our
cached ENST transcriptions per recording, then resamples subsets of each size to see how
wide the interval around a per-class F1 comes out. It does it two ways:

  over recordings   what significance.py does, and what a 60-track run would report
  over drummers     a cluster bootstrap: draw 3 drummers WITH REPLACEMENT from the 3 that
                    exist, pool their recordings, and sample within that pool. This is the
                    question "what if the three players had been three other players", and
                    with only three clusters it is necessarily coarse -- it can produce a
                    resample of one drummer repeated three times, which is the point. The
                    gap between the columns is what treating recordings as independent
                    costs when the player is what actually varies.

It cannot simulate ReStem, which has never been run on ENST. What it gives is the width
of the interval a sample of each size supports, which is the part that decides whether the
render is worth starting.

    python sample_power.py
    python sample_power.py --sizes 23,60,120,210 --class HH
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_enst import DEFAULT_KINDS  # noqa: E402

KINDS = set(DEFAULT_KINDS)

# The MDB comparison's measured half-widths, for scale. From significance.py on the 23
# recordings: MICRO +0.062 [+0.032, +0.097], hi-hat +0.139 [+0.062, +0.249].
MDB_HALFWIDTH = {"MICRO": 0.0325, "HH": 0.0935}
MDB_EFFECT = {"MICRO": 0.062, "HH": 0.139}


def collect(data: Path, midi_dir: Path):
    """Per-recording tp/ref/est for the cached ENST transcriptions, with the drummer."""
    import mir_eval
    import numpy as np
    import pretty_midi
    from benchmark_enst import IGNORED, LABEL_TO_CLASS
    from benchmark_mdb import CLASSES, ORDER, WINDOW

    rows = []
    for d in (1, 2, 3):
        adir = data / f"drummer_{d}" / "annotation"
        if not adir.exists():
            continue
        for ann in sorted(adir.glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in KINDS:
                continue
            wav = data / f"drummer_{d}" / "audio" / "wet_mix" / f"{ann.stem}.wav"
            if not wav.exists():
                continue
            mid = midi_dir / f"{ann.stem}_d{d}.mid"
            ref = {k: [] for k in CLASSES}
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 2 or p[1] in IGNORED:
                    continue
                cls = LABEL_TO_CLASS.get(p[1])
                if cls:
                    ref[cls].append(float(p[0]))
            notes = ([n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                      for n in inst.notes] if mid.exists() else [])
            row = {}
            for k in ORDER:
                r = np.array(sorted(ref[k]))
                e = np.array(sorted(n.start for n in notes if n.pitch in CLASSES[k]))
                if r.size and e.size:
                    _f, prec, _rec = mir_eval.onset.f_measure(r, e, window=WINDOW)
                    tp = int(round(prec * len(e)))
                else:
                    tp = 0
                row[k] = (tp, int(r.size), int(e.size))
            rows.append({"name": ann.stem, "drummer": d, "counts": row})
    return rows


def f1(picked, cls) -> float:
    tp = ref = est = 0
    for r in picked:
        if cls == "MICRO":
            for v in r["counts"].values():
                tp, ref, est = tp + v[0], ref + v[1], est + v[2]
        else:
            v = r["counts"][cls]
            tp, ref, est = tp + v[0], ref + v[1], est + v[2]
    if not est or not ref or not tp:
        return 0.0
    p, rc = tp / est, tp / ref
    return 2 * p * rc / (p + rc)


def halfwidth(samples) -> float:
    s = sorted(samples)
    lo, hi = s[int(0.025 * len(s))], s[int(0.975 * len(s))]
    return (hi - lo) / 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=ROOT / "enst" / "enst_drums_public")
    ap.add_argument("--midi", type=Path, default=ROOT / "bench" / "enst")
    ap.add_argument("--sizes", default="23,60,105,210")
    ap.add_argument("--classes", default="MICRO,HH,SD,TT")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"no ENST at {args.data}")
        return 1
    rows = collect(args.data, args.midi)
    if not rows:
        print("nothing scored; is bench/enst populated?")
        return 1

    by_drummer: dict[int, list] = {}
    for r in rows:
        by_drummer.setdefault(r["drummer"], []).append(r)
    print(f"{len(rows)} ENST recordings scored from cached MIDI, "
          f"{len(by_drummer)} drummers "
          f"({', '.join(f'd{k}:{len(v)}' for k, v in sorted(by_drummer.items()))})")

    sizes = [int(s) for s in args.sizes.split(",") if int(s) <= len(rows)]
    classes = [c.strip() for c in args.classes.split(",")]
    rng = random.Random(args.seed)
    drummers = sorted(by_drummer)

    for cls in classes:
        whole = f1(rows, cls)
        print(f"\n{cls}: our F1 on all {len(rows)} ENST recordings is {whole:.3f}")
        print(f"{'sample':>8}{'half-width':>13}{'clustered':>12}{'inflation':>11}"
              f"{'resolves':>28}")
        print("-" * 72)
        clus_by_size: dict[int, float] = {}
        for n in sizes:
            flat, clus = [], []
            for _ in range(args.rounds):
                flat.append(f1([rng.choice(rows) for _ in range(n)], cls))
                # cluster bootstrap: the drummers are the units, drawn with replacement
                pool = []
                for _ in range(len(drummers)):
                    pool.extend(by_drummer[rng.choice(drummers)])
                clus.append(f1([rng.choice(pool) for _ in range(n)], cls))
            hw, hwc = halfwidth(flat), halfwidth(clus)
            clus_by_size[n] = hwc
            eff = MDB_EFFECT.get(cls)
            if eff is None:
                verdict = "no MDB effect to compare"
            elif hwc < eff:
                verdict = f"yes, even clustered, at {eff:+.3f}"
            elif hw < eff:
                verdict = f"only if drummers are not the unit"
            else:
                verdict = f"NO -- straddles zero at {eff:+.3f}"
            infl = hwc / hw if hw else float("nan")
            print(f"{n:>8}{hw:>13.3f}{hwc:>12.3f}{infl:>10.2f}x  {verdict:<30}")
        if cls in MDB_HALFWIDTH:
            print(f"{'MDB 23':>8}{MDB_HALFWIDTH[cls]:>13.3f}{'—':>12}{'—':>11}"
                  f"  measured, paired against ReStem")
        if len(sizes) > 1:
            first = sizes[0]
            plateau = clus_by_size.get(sizes[-1])
            shrink = clus_by_size.get(first, 0) / plateau if plateau else 0
            eff = MDB_EFFECT.get(cls)
            if plateau and eff and plateau > eff:
                print(f"  Clustered, the width flattens near {plateau:.3f} and never "
                      f"reaches {eff:+.3f}.")
                print(f"  Rendering more recordings does not fix this: there are three "
                      f"drummers and")
                print(f"  that is the ceiling. {cls} is not a question ENST can answer.")
            elif plateau and eff:
                print(f"  Clustered width flattens near {plateau:.3f}, inside {eff:+.3f}, "
                      f"so {cls} is answerable")
                print(f"  and {sizes[1] if len(sizes) > 1 else first} recordings already "
                      f"buys most of what {sizes[-1]} would.")

    print("""
Two things this does not say. The half-widths above are around OUR F1 on ENST, not around
a difference against ReStem, and a paired difference between two systems on the same
recordings is usually tighter than either system's own interval -- so these are pessimistic
for the comparison and should be read as an upper bound on the width, not a prediction.

And the clustered column is the one to argue about. Resampling drummers first is the
harsher assumption: it asks what happens if the player, not the recording, is what varies.
The truth is somewhere between the two columns, and where it sits is a property of ENST
that no amount of bootstrapping the recordings can reveal. Three drummers is three
drummers.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
