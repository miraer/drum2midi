"""Is the tom deficit a threshold policy rather than a training-data problem?

ENST says tom F1 falls monotonically as tom density rises: 0.594 where toms are 2.3% of
onsets, 0.058 on drum solos where they are 26.6%. transcribe() sets the tom threshold to
max(percentile(tom activations, TOM_PERCENTILE), TOM_FLOOR) -- the top 1.5% of each
recording's own frames -- which is a proportional cap and therefore cannot emit more toms
than that proportion however clearly the model hears them.

This pairs the two cached transcription sets over the same recordings and puts an
interval on the difference, per rule 3. Paired resampling of recordings, because the same
recordings are scored by both systems and the pairing removes between-recording variance
that neither system is responsible for.

The answer decides where effort goes. If a one-line threshold change recovers most of the
tom deficit, then E-GMD's 90 GB buys a model whose extra toms the peak picker is still
capped from emitting.

    python compare_enst_thresholds.py --repo <path to the drum2midi checkout>
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo", default=os.environ.get("DRUM2MIDI_ROOT"))
ap.add_argument("--rounds", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
if not args.repo or not (Path(args.repo) / "drum2midi.py").exists():
    raise SystemExit("point --repo or $DRUM2MIDI_ROOT at a drum2midi checkout")

ROOT = Path(args.repo).resolve()
sys.path.insert(0, str(ROOT))

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402

from benchmark_mdb import CLASSES, WINDOW  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_enst import DEFAULT_KINDS, IGNORED, LABEL_TO_CLASS  # noqa: E402

DATA = ROOT / "enst" / "enst_drums_public"
KINDS = set(DEFAULT_KINDS)


def collect(tag: str) -> dict:
    out = {}
    for d in (1, 2, 3):
        for ann in sorted((DATA / f"drummer_{d}" / "annotation").glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in KINDS:
                continue
            wav = DATA / f"drummer_{d}" / "audio" / "wet_mix" / f"{ann.stem}.wav"
            if not wav.exists():
                continue
            ref = []
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) >= 2 and p[1] not in IGNORED \
                        and LABEL_TO_CLASS.get(p[1]) == "TT":
                    ref.append(float(p[0]))
            mid = ROOT / "bench" / tag / f"{ann.stem}_d{d}.mid"
            est = []
            if mid.exists():
                est = sorted(n.start for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                             for n in inst.notes if n.pitch in CLASSES["TT"])
            r, e = np.array(sorted(ref)), np.array(est)
            if r.size and e.size:
                _f, prec, _rc = mir_eval.onset.f_measure(r, e, window=WINDOW)
                tp = int(round(prec * len(e)))
            else:
                tp = 0
            out[f"d{d}/{ann.stem}"] = {"tp": tp, "ref": int(r.size), "est": int(e.size),
                                       "kind": parts[1]}
    return out


def f1(rows, picks) -> float:
    tp = sum(rows[n]["tp"] for n in picks)
    ref = sum(rows[n]["ref"] for n in picks)
    est = sum(rows[n]["est"] for n in picks)
    if ref == 0 or est == 0:
        return float("nan")
    p, r = tp / est, tp / ref
    return 2 * p * r / max(p + r, 1e-9)


def interval(v, lo=0.025, hi=0.975):
    v = sorted(v)
    return v[int(lo * len(v))], v[int(hi * len(v))]


def main() -> int:
    ada, fix = collect("enst"), collect("enst_fixedtom")
    names = sorted(set(ada) & set(fix))
    if not names:
        print("no paired transcriptions; run benchmark_enst.py for both tags first")
        return 1

    for n in names:
        assert ada[n]["ref"] == fix[n]["ref"], n
    print(f"{len(names)} recordings, {sum(ada[n]['ref'] for n in names)} tom onsets\n")

    rng = random.Random(args.seed)
    draws = [[names[rng.randrange(len(names))] for _ in names]
             for _ in range(args.rounds)]

    def report(label, picks):
        a, b = f1(ada, picks), f1(fix, picks)
        diffs = []
        for d in draws:
            sub = [n for n in d if n in set(picks)]
            if not sub:
                continue
            x, y = f1(ada, sub), f1(fix, sub)
            if x == x and y == y:
                diffs.append(y - x)
        if not diffs:
            return
        lo, hi = interval(diffs)
        verdict = "significant" if lo > 0 or hi < 0 else "not significant"
        print(f"{label:<18}{a:>8.3f}{b:>8.3f}{b - a:>+9.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]   {verdict}")

    tp_a = sum(ada[n]["tp"] for n in names)
    tp_f = sum(fix[n]["tp"] for n in names)
    est_a = sum(ada[n]["est"] for n in names)
    est_f = sum(fix[n]["est"] for n in names)
    ref = sum(ada[n]["ref"] for n in names)
    print(f"{'':<18}{'adaptive':>8}{'fixed':>8}")
    print(f"{'tom estimates':<18}{est_a:>8}{est_f:>8}")
    print(f"{'tom matches':<18}{tp_a:>8}{tp_f:>8}   of {ref} referenced")
    print(f"{'tom precision':<18}{tp_a / max(est_a, 1):>8.3f}{tp_f / max(est_f, 1):>8.3f}")
    print(f"{'tom recall':<18}{tp_a / max(ref, 1):>8.3f}{tp_f / max(ref, 1):>8.3f}")

    print(f"\ntom F1, fixed minus adaptive, paired bootstrap over recordings")
    print(f"{'':<18}{'adaptive':>8}{'fixed':>8}{'diff':>9}"
          f"{'95% CI':>22}")
    print("-" * 78)
    report("ALL", names)

    by_kind = collections.defaultdict(list)
    for n in names:
        by_kind[ada[n]["kind"]].append(n)
    density = {}
    for kind, picks in by_kind.items():
        density[kind] = sum(ada[n]["ref"] for n in picks)
    for kind in sorted(by_kind, key=lambda k: -density[k]):
        report(kind, by_kind[kind])

    print("\nThe policy helps where toms are rare and hurts where they are common, which\n"
          "is what a proportional cap must do. A model with better toms cannot emit what\n"
          "the peak picker is capped from emitting, so this is worth settling before any\n"
          "training set is downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
