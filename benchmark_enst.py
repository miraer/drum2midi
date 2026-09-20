"""Benchmark drum2midi on ENST-Drums, the acoustic set that actually has toms.

MDB Drums holds 90 tom onsets and five of its 23 tracks carry 86 of them. This is the
state that justified buying ENST, quoted as it stood then: bootstrapping over tracks
gave tom F1 0.605 with a 95% CI of [0.341, 0.747], and a known-large effect -- adaptive
against fixed tom thresholding, which nearly doubles tom F1 -- came back as
+0.282 [-0.074, +0.469], an interval containing zero around an effect we know is real.
A benchmark that cannot see a doubling cannot see anything a training run would produce.
This script asks whether ENST can. (MDB's tom figure is now 0.589 [0.337, 0.735] under
TOM_CEILING; the half-width barely moved, which is the point.)

ENST's musical recordings carry 2617 tom onsets across 210 recordings: 29x the onsets
over 9x the recordings, and the recording is the unit the interval is computed over.

**Licence: ENST-Drums is CC BY-NC-ND 4.0.** Evaluation only. Nothing may be trained on
it, no weights derived from it may be published, and no derived annotations may be
redistributed. E-GMD (CC BY 4.0) is the set to train on; this one is the ruler.

Two traps this script is built to avoid, both of which have caught this project before:

  tom stems     ENST ships isolated close-miked tom_1/ tom_2/. Scoring toms against a
                tom stem makes the class trivially separable and the number meaningless
                -- the 0.968-against-0.623 failure in a new costume. Scoring is on the
                drum mix, and --mix says which one in the output.

  isolated hits 108 of ENST's recordings are 'hits': five strokes of one drum in silence.
                They are not music and they are excluded by default. --include-hits turns
                them on and prints what they do to the number, which is the cheapest
                available demonstration of why balanced material must not be scored.

Separation is deliberately off. Re-verified under TOM_CEILING, not inherited: on MDB,
bench/ceiling and --no-separate produce bit-identical tom numbers -- 90 ref, 134 est,
66 matched, precision 0.493, recall 0.733, F1 0.589 in both -- while every other class
moves (snare 0.844 against 0.805, hi-hat 0.892 against 0.863, MICRO 0.882 against 0.860).
The separator provably does not touch this one class, so the tom figure here is
comparable with MDB's despite being computed without a GPU. The pre-ceiling check that
first established this read 125 est, 65 matched, F1 0.605; the counts moved, the
identity did not.

    python benchmark_enst.py                      # musical recordings, wet mix
    python benchmark_enst.py --limit 20           # quick pass
    python benchmark_enst.py --rescore            # re-score cached MIDI, no inference
    python benchmark_enst.py --include-hits       # show the balanced-material effect
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _find_repo(explicit) -> Path:
    """Works dropped into the project root, or run from anywhere with --repo."""
    here = Path(__file__).resolve().parent
    for cand in (explicit, os.environ.get("DRUM2MIDI_ROOT"), here):
        if cand and (Path(cand) / "drum2midi.py").exists():
            return Path(cand).resolve()
    raise SystemExit("cannot find drum2midi.py; pass --repo or set $DRUM2MIDI_ROOT")


# ENST's vocabulary is finer than the five classes this project transcribes. Every
# judgement call is commented so a reader can disagree with one decision rather than
# with the whole mapping.
LABEL_TO_CLASS = {
    "bd": "KD",                                    # bass drum
    "sd": "SD",                                    # snare
    "sd-": "SD",   # soft or ghost snare. Still a snare onset; MDB annotates ghosts too,
                   # so counting them keeps the two corpora commensurable.
    "rs": "SD",    # rim shot -- struck on the snare, and our taxonomy has no rim class.
    "cs": "SD",    # cross stick. Judgement call: acoustically unlike a struck snare, but
                   # it is a snare-voice event and ADTOF can only answer with 38/40.
    "chh": "HH", "ohh": "HH",                      # closed and open hi-hat
    "lt": "TT", "mt": "TT",                        # low and medium tom
    "lmt": "TT", "lft": "TT",                      # low-medium tom, low floor tom
    "ltr": "TT", "mtr": "TT",                      # tom rim shots -- still tom voices
    "rc1": "CY", "rc2": "CY", "rc3": "CY", "rc4": "CY",       # ride
    "cr1": "CY", "cr2": "CY", "cr5": "CY",                    # crash
    "c1": "CY", "c2": "CY", "c3": "CY", "c4": "CY",           # unnumbered cymbals
    "ch1": "CY", "ch5": "CY",                                 # chinese
    "spl2": "CY",                                             # splash
}

# Counted and reported, never scored: our pipeline cannot emit these, so putting them in
# the reference would charge it for a class it has no way to answer.
IGNORED = {
    "cb": "cowbell -- auxiliary percussion, outside the five classes",
    "sticks": "stick clicks, usually a count-in rather than a drum",
    "sweep": "brush sweep -- a sustained gesture, not an onset",
}


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mix", default="wet_mix", choices=["wet_mix", "dry_mix"])
    ap.add_argument("--kinds", default="phrase,solo,minus-one,MIDI-minus-one")
    ap.add_argument("--only", default=None,
                    help="substring the recording name must contain, e.g. mallets")
    ap.add_argument("--separate", action="store_true",
                    help="run the separator too, which is the path the tool takes in "
                         "real use; the default is off so tom figures stay comparable "
                         "with MDB")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--include-hits", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--tag", default="enst")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-h", "--help", action="store_true")
    args, passthrough = ap.parse_known_args()
    if args.help:
        print(__doc__)
        return 0

    root = _find_repo(args.repo)
    sys.path.insert(0, str(root))

    import mir_eval
    import numpy as np
    import pretty_midi

    from benchmark_mdb import CLASSES, ORDER, PRETTY, WINDOW

    data = root / "enst" / "enst_drums_public"
    if not data.exists():
        print(f"ENST not found at {data}; run: python fetch_enst.py")
        return 1

    kinds = set(args.kinds.split(","))
    if args.include_hits:
        kinds.add("hits")

    items = []
    for d in (1, 2, 3):
        for ann in sorted((data / f"drummer_{d}" / "annotation").glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in kinds:
                continue
            wav = data / f"drummer_{d}" / "audio" / args.mix / f"{ann.stem}.wav"
            if args.only and args.only not in ann.stem:
                continue
            if wav.exists():
                items.append((f"d{d}/{ann.stem}", wav, ann, parts[1]))
    items = items[: args.limit]
    if not items:
        print(f"no recordings matched kinds={sorted(kinds)}")
        return 1

    outdir = root / "bench" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"ENST-Drums (CC BY-NC-ND 4.0, evaluation only)")
    print(f"{len(items)} recordings, mix={args.mix}, kinds={','.join(sorted(kinds))}")
    if args.separate:
        print("separation ON: the path the tool takes in real use, not comparable with "
              "the MDB tom figures\n")
    else:
        print(f"separation off: the separator provably does not move toms on MDB\n")

    if not args.rescore:
        t0 = time.time()
        sep = [] if args.separate else ["--no-separate"]
        for i, (name, wav, _ann, _k) in enumerate(items, 1):
            mid = outdir / f"{wav.stem}_d{wav.parents[2].name[-1]}.mid"
            res = subprocess.run(
                [sys.executable, str(root / "drum2midi.py"), str(wav), "-o", str(mid),
                 "--device", args.device] + sep + passthrough,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if res.returncode != 0:
                tail = (res.stderr or res.stdout).strip().splitlines()
                print(f"  [{i:3d}/{len(items)}] FAIL {name}: "
                      f"{tail[-1][:100] if tail else '?'}")
            elif i % 10 == 0 or i == len(items):
                print(f"  [{i:3d}/{len(items)}] ok   {name}", flush=True)
        print(f"  inference took {time.time() - t0:.0f}s\n")

    label_counts = collections.Counter()
    ignored_counts = collections.Counter()
    per_rec = {}
    silent = []

    for name, wav, ann, kind in items:
        mid = outdir / f"{wav.stem}_d{wav.parents[2].name[-1]}.mid"
        ref = {k: [] for k in CLASSES}
        for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split()
            if len(p) < 2:
                continue
            label = p[1]
            if label in IGNORED:
                ignored_counts[label] += 1
                continue
            cls = LABEL_TO_CLASS.get(label)
            if cls is None:
                ignored_counts[f"UNMAPPED:{label}"] += 1
                continue
            label_counts[label] += 1
            ref[cls].append(float(p[0]))

        # A recording the pipeline could not transcribe at all is a result, not a gap.
        # Dropping it would quietly remove the cases where the model fails hardest and
        # inflate every number below, so it is scored with an empty estimate instead.
        if mid.exists():
            notes = [n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                     for n in inst.notes]
        else:
            notes = []
            if any(ref[k] for k in ORDER):
                silent.append(name)

        row = {}
        for k in ORDER:
            r = np.array(sorted(ref[k]))
            e = np.array(sorted(n.start for n in notes if n.pitch in CLASSES[k]))
            if r.size and e.size:
                f, prec, _rec = mir_eval.onset.f_measure(r, e, window=WINDOW)
                tp = int(round(prec * len(e)))
            else:
                tp = 0
            row[k] = {"tp": tp, "ref": int(r.size), "est": int(e.size)}
        per_rec[name] = {"counts": row, "kind": kind}

    if not per_rec:
        print("no transcriptions to score")
        return 1

    def prf(rows, picks, cls):
        tp = sum(rows[n]["counts"][cls]["tp"] for n in picks)
        ref = sum(rows[n]["counts"][cls]["ref"] for n in picks)
        est = sum(rows[n]["counts"][cls]["est"] for n in picks)
        if ref == 0 or est == 0:
            return float("nan"), tp, ref, est
        p, r = tp / est, tp / ref
        return 2 * p * r / max(p + r, 1e-9), tp, ref, est

    names = sorted(per_rec)
    print(f"{'drum':<10}{'ref':>7}{'est':>7}{'match':>7}{'prec':>8}{'rec':>8}{'F1':>8}")
    print("-" * 55)
    micro = [0, 0, 0]
    for k in ORDER:
        f, tp, ref, est = prf(per_rec, names, k)
        if ref == 0:
            continue
        micro[0] += tp; micro[1] += ref; micro[2] += est
        print(f"{PRETTY[k]:<10}{ref:>7}{est:>7}{tp:>7}"
              f"{tp / max(est, 1):>8.3f}{tp / max(ref, 1):>8.3f}{f:>8.3f}")
    p, r = micro[0] / max(micro[2], 1), micro[0] / max(micro[1], 1)
    print("-" * 55)
    print(f"{'MICRO':<10}{micro[1]:>7}{micro[2]:>7}{micro[0]:>7}"
          f"{p:>8.3f}{r:>8.3f}{2 * p * r / max(p + r, 1e-9):>8.3f}")

    print(f"\nreference onsets by ENST label")
    for label, n in label_counts.most_common():
        print(f"  {label:<8}{LABEL_TO_CLASS[label]:<4}{n:>7}")
    if ignored_counts:
        print(f"\ncounted but not scored")
        for label, n in ignored_counts.most_common():
            print(f"  {label:<8}{n:>7}  {IGNORED.get(label, 'not in the mapping')}")

    if silent:
        print(f"\n{len(silent)} recording(s) the pipeline transcribed to nothing at all, "
              f"scored as\nzero estimates rather than dropped:")
        for name in silent[:10]:
            print(f"  {name}")
        if len(silent) > 10:
            print(f"  ... and {len(silent) - 10} more")

    rng = random.Random(args.seed)
    draws = [[names[rng.randrange(len(names))] for _ in names]
             for _ in range(args.rounds)]

    def ci(cls):
        vals = [v for v in (prf(per_rec, d, cls)[0] for d in draws) if v == v]
        v = sorted(vals)
        return v[int(0.025 * len(v))], v[int(0.975 * len(v))]

    print(f"\nbootstrap over {len(names)} recordings, {args.rounds} resamples")
    print(f"{'drum':<10}{'F1':>8}{'95% CI':>22}{'half-width':>12}")
    print("-" * 52)
    for k in ORDER:
        f, _, ref, _ = prf(per_rec, names, k)
        if ref == 0:
            continue
        lo, hi = ci(k)
        print(f"{PRETTY[k]:<10}{f:>8.3f}   [{lo:.3f}, {hi:.3f}]{(hi - lo) / 2:>12.3f}")

    tf, _, _, _ = prf(per_rec, names, "TT")
    lo, hi = ci("TT")
    half = (hi - lo) / 2
    print(f"\n{'=' * 62}")
    print(f"TOM HALF-WIDTH: {half:.3f}      (MDB: 0.199)")
    print(f"tom F1 on ENST {tf:.3f}   against 0.589 on MDB")
    verdict = ("resolves +0.10, the instrument works" if half < 0.10 else
               "resolves +0.10 only marginally" if half < 0.15 else
               "CANNOT resolve +0.10 -- another unusable benchmark")
    print(f"verdict: {verdict}")
    print(f"{'=' * 62}")

    by_kind = collections.Counter()
    for n in names:
        by_kind[per_rec[n]["kind"]] += 1
    print(f"\ntom F1 by recording kind, since they are not the same material")
    for kind in sorted(by_kind):
        picks = [n for n in names if per_rec[n]["kind"] == kind]
        f, _tp, ref, _est = prf(per_rec, picks, "TT")
        score = f"{f:.3f}" if ref else "n/a"
        print(f"  {kind:<18}{by_kind[kind]:>4} recs{ref:>7} toms   F1 {score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
