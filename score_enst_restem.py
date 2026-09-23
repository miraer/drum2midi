"""Scores ReStem against us on ENST, from whatever the batch has finished rendering.

ReStem's batch writes one folder per recording with its own `*_midi.mid` inside, which
is better than the per-file path's cache copy: it is the product's own export rather
than our conversion of its trigger JSON, and the two were checked against each other on
two tracks and agreed note for note including velocity.

Two things this is careful about.

**The declared sample is a separate quantity.** `bench/enst60.txt` was fixed before any
render, on the machine that was not going to do the rendering. Everything else is an
extension. They are reported separately and never pooled into one number, because a
sample chosen in advance and a sample that happens to have finished are different kinds
of evidence and only one of them can carry a claim.

**Drummers, not recordings, are the unit that varies.** ENST is 210 recordings from 3
players in 3 rooms. Resampling recordings treats 210 independent units where there may
be 3, so both intervals are reported: over recordings, and over drummers drawn with
replacement. sample_power.py measured the gap at up to 3.4x, and the hi-hat row -- the
one carrying the MDB headline -- was answerable at 60 recordings under either.

    python score_enst_restem.py
    python score_enst_restem.py --ours bench/enst --rounds 4000
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

from benchmark_enst import DEFAULT_KINDS  # noqa: E402

KINDS = set(DEFAULT_KINDS)

# ReStem writes one MIDI note per stem voice; the mapping is the one restem_to_midi.py
# uses and which its own export validated.
RESTEM_TO_CLASS = {
    36: "KD", 35: "KD",
    38: "SD", 40: "SD", 37: "SD",
    42: "HH", 44: "HH", 46: "HH",
    41: "TT", 43: "TT", 45: "TT", 47: "TT", 48: "TT", 50: "TT",
    49: "CY", 51: "CY", 52: "CY", 53: "CY", 55: "CY", 57: "CY", 59: "CY",
}


def enst_reference(data: Path):
    """Per-recording reference onsets in the five classes, from ENST's annotations."""
    from benchmark_enst import IGNORED, LABEL_TO_CLASS
    out = {}
    for d in (1, 2, 3):
        adir = data / f"drummer_{d}" / "annotation"
        if not adir.exists():
            continue
        for ann in sorted(adir.glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in KINDS:
                continue
            ref = {}
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 2 or p[1] in IGNORED:
                    continue
                cls = LABEL_TO_CLASS.get(p[1])
                if cls:
                    ref.setdefault(cls, []).append(float(p[0]))
            out[ann.stem] = {"ref": ref, "drummer": d}
    return out


def notes_by_class(mid: Path, mapping=None):
    import pretty_midi
    out = {}
    pm = pretty_midi.PrettyMIDI(str(mid))
    for inst in pm.instruments:
        for n in inst.notes:
            cls = (mapping or {}).get(n.pitch) if mapping else None
            if mapping is None:
                from benchmark_mdb import CLASSES
                cls = next((k for k, v in CLASSES.items() if n.pitch in v), None)
            if cls:
                out.setdefault(cls, []).append(n.start)
    return out


def counts(ref: dict, est: dict, order, window: float):
    import mir_eval
    import numpy as np
    row = {}
    for k in order:
        r = np.array(sorted(ref.get(k, [])))
        e = np.array(sorted(est.get(k, [])))
        if r.size and e.size:
            _f, prec, _rec = mir_eval.onset.f_measure(r, e, window=window)
            tp = int(round(prec * len(e)))
        else:
            tp = 0
        row[k] = {"tp": tp, "ref": int(r.size), "est": int(e.size)}
    return row


def f1(rows, picks, order, cls=None) -> float:
    tp = ref = est = 0
    for name in picks:
        for k in ([cls] if cls else order):
            v = rows[name][k]
            tp, ref, est = tp + v["tp"], ref + v["ref"], est + v["est"]
    if not tp or not ref or not est:
        return 0.0
    p, r = tp / est, tp / ref
    return 2 * p * r / (p + r)


def interval(vals):
    v = sorted(vals)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v))]


def report(title, names, ours, theirs, order, pretty, drummers, rounds, seed):
    if not names:
        print(f"\n{title}: nothing rendered yet")
        return
    by_drummer = {}
    for n in names:
        by_drummer.setdefault(drummers[n], []).append(n)
    print(f"\n{title}: {len(names)} recordings, "
          f"{len(by_drummer)} drummer(s), "
          f"{sum(ours[n][k]['ref'] for n in names for k in order)} onsets")
    print(f"{'class':<9}{'ours':>7}{'ReStem':>8}{'diff':>8}"
          f"{'95% over recordings':>24}{'over drummers':>22}")
    print("-" * 78)
    rng = random.Random(seed)
    dr = sorted(by_drummer)
    for cls in [None] + list(order):
        o, t = f1(ours, names, order, cls), f1(theirs, names, order, cls)
        flat, clus = [], []
        for _ in range(rounds):
            s = [rng.choice(names) for _ in names]
            flat.append(f1(ours, s, order, cls) - f1(theirs, s, order, cls))
            pool = []
            for _ in range(len(dr)):
                pool.extend(by_drummer[rng.choice(dr)])
            s2 = [rng.choice(pool) for _ in names]
            clus.append(f1(ours, s2, order, cls) - f1(theirs, s2, order, cls))
        lo, hi = interval(flat)
        clo, chi = interval(clus)
        label = "MICRO" if cls is None else pretty.get(cls, cls)
        mark = " *" if lo > 0 or hi < 0 else "  "
        cmark = " *" if clo > 0 or chi < 0 else "  "
        print(f"{label:<9}{o:>7.3f}{t:>8.3f}{o - t:>+8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{mark}"
              f"{f'[{clo:+.3f}, {chi:+.3f}]':>20}{cmark}")


def main() -> int:
    from benchmark_mdb import ORDER, PRETTY, WINDOW

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "enst" / "enst_drums_public")
    ap.add_argument("--ours", type=Path, default=ROOT / "bench" / "enst")
    ap.add_argument("--theirs", type=Path, default=ROOT / "restem_export")
    ap.add_argument("--declared", type=Path, default=ROOT / "bench" / "enst60.txt")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    refs = enst_reference(args.data)
    declared = set(args.declared.read_text(encoding="utf-8").split()) \
        if args.declared.exists() else set()

    ours, theirs, drummers = {}, {}, {}
    for folder in sorted(args.theirs.iterdir()):
        if not folder.is_dir() or folder.name.startswith("MusicDelta"):
            continue
        name = folder.name
        if name not in refs:
            continue
        their_mid = folder / f"{name}_midi.mid"
        our_mid = args.ours / f"{name}_d{refs[name]['drummer']}.mid"
        if not their_mid.exists() or not our_mid.exists():
            continue
        ref = refs[name]["ref"]
        theirs[name] = counts(ref, notes_by_class(their_mid, RESTEM_TO_CLASS),
                              ORDER, WINDOW)
        ours[name] = counts(ref, notes_by_class(our_mid), ORDER, WINDOW)
        drummers[name] = refs[name]["drummer"]

    names = sorted(ours)
    print(f"{len(names)} recordings scored by both, from {args.theirs.name}")
    print(f"ours: {args.ours}  (no separation -- the shipped pipeline adds it and is "
          f"being measured separately)")

    inside = [n for n in names if n in declared]
    outside = [n for n in names if n not in declared]
    report(f"declared sample ({len(inside)} of {len(declared)} rendered)",
           inside, ours, theirs, ORDER, PRETTY, drummers, args.rounds, args.seed)
    report(f"extension beyond the declared sample", outside, ours, theirs, ORDER,
           PRETTY, drummers, args.rounds, args.seed)

    print("""
The two blocks are never added together. The declared sample was fixed before any render
and can carry a claim; the extension is whatever else finished and can only illustrate
one. Pooling them would quietly convert the second into the first.

`over drummers` resamples the three players with replacement rather than the recordings.
It is the harsher and more honest column for anything that depends on who is playing,
and on ENST that is most things. Where the two columns disagree about whether an
interval clears zero, the drummer column is the one to believe.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
