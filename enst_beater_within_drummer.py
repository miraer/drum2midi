"""A second beater control with the opposite confound: hold the drummer, vary the style.

Task B asked whether ENST pairs one drummer playing one piece with two beaters. **It does
not**, and `enst_beater_pairs.py` shows why: ENST assigns one beater per drummer per
style. Drummer 1 plays rock with rods, afro with mallets, shuffle-blues and waltz with
brushes; drummers 2 and 3 play those same styles with sticks. Every beater comparison the
corpus permits is therefore cross-drummer by construction, and no subsetting fixes it.

But the confound can be attacked from the other side. The published control holds **style**
constant and lets the drummer vary. Drummer 1 alone played all four beaters -- 34 stick
recordings, 17 brushes, 9 rods, 7 mallets -- so a within-drummer comparison holds the
**player** constant and lets style vary instead.

Neither is clean. They are confounded in orthogonal directions, which is the useful part:
an effect that survives both is not explained by either confound, and one that appears in
only one of them was probably that confound all along.

Scoring is `benchmark_enst.py`'s, from the same cached transcriptions, so the numbers sit
beside the published ones rather than beneath them.

    python enst_beater_within_drummer.py --repo <path> --tag enst
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
# Written on a machine where this script lived outside the checkout, so it took --repo.
# It lives in the checkout now, so its own directory is the default and the flag stays
# for anyone running it from elsewhere.
_here = Path(__file__).resolve().parent
_default = os.environ.get("DRUM2MIDI_ROOT") or (
    str(_here) if (_here / "drum2midi.py").exists() else None)
ap.add_argument("--repo", default=_default)
ap.add_argument("--tag", default="enst")
ap.add_argument("--rounds", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
if not args.repo or not (Path(args.repo) / "drum2midi.py").exists():
    raise SystemExit("point --repo or $DRUM2MIDI_ROOT at a drum2midi checkout")

ROOT = Path(args.repo).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402

from benchmark_mdb import CLASSES, ORDER, WINDOW  # noqa: E402
from benchmark_enst import IGNORED, LABEL_TO_CLASS  # noqa: E402

BEATERS = ("sticks", "rods", "brushes", "mallets")
DATA = ROOT / "enst" / "enst_drums_public"
OUT = ROOT / "bench" / args.tag


def beater_of(stem: str):
    return next((b for b in BEATERS if stem.endswith(b)), None)


def score_one(ann: Path, mid: Path | None) -> tuple:
    """tp / ref / est summed over the five classes, as benchmark_enst does.

    `mid` may be None for a recording that produced no transcription at all, which is
    scored as zero estimates rather than dropped.
    """
    ref = {k: [] for k in CLASSES}
    for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) < 2 or p[1] in IGNORED:
            continue
        cls = LABEL_TO_CLASS.get(p[1])
        if cls:
            ref[cls].append(float(p[0]))
    notes = []
    if mid is not None and mid.exists():
        notes = [n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                 for n in inst.notes]
    tp = rf = es = 0
    for k in ORDER:
        r = np.array(sorted(ref[k]))
        e = np.array(sorted(n.start for n in notes if n.pitch in CLASSES[k]))
        if r.size and e.size:
            _f, prec, _rc = mir_eval.onset.f_measure(r, e, window=WINDOW)
            tp += int(round(prec * len(e)))
        rf += int(r.size)
        es += int(e.size)
    return tp, rf, es


def micro(rows, picks) -> float:
    tp = sum(rows[n][0] for n in picks)
    rf = sum(rows[n][1] for n in picks)
    es = sum(rows[n][2] for n in picks)
    if rf == 0 or es == 0:
        return float("nan")
    p, r = tp / es, tp / rf
    return 2 * p * r / max(p + r, 1e-9)


def interval(v, lo=0.025, hi=0.975):
    v = sorted(v)
    return v[int(lo * len(v))], v[int(hi * len(v))]


def main() -> int:
    if not OUT.exists():
        print(f"no cached transcriptions at {OUT}; run benchmark_enst.py first")
        return 1

    rows, meta = {}, {}
    silent = []
    for d in (1, 2, 3):
        for ann in sorted((DATA / f"drummer_{d}" / "annotation").glob("*.txt")):
            b = beater_of(ann.stem)
            if not b:
                continue
            mid = OUT / f"{ann.stem}_d{d}.mid"
            # A recording the pipeline could not transcribe at all is a result, not a
            # gap. Skipping it silently removes the cases where the model fails hardest
            # and flatters exactly the beater under test: dropping the one mallet
            # recording that produced nothing moves pooled mallet MICRO 0.409 -> 0.415.
            if not mid.exists():
                if not (DATA / f"drummer_{d}" / "audio" / "wet_mix"
                        / f"{ann.stem}.wav").exists():
                    continue
                silent.append(f"d{d}/{ann.stem}")
            rows[ann.stem] = score_one(ann, mid if mid.exists() else None)
            meta[ann.stem] = (d, b, ann.stem.split("_")[2])
    if not rows:
        print("nothing scored")
        return 1
    if silent:
        print(f"{len(silent)} recording(s) transcribed to nothing, scored as zero "
              f"estimates rather than dropped:")
        for s in silent:
            print(f"  {s}")
    print(f"scored {len(rows)} recordings from bench/{args.tag}\n")

    print("recordings by drummer and beater")
    print(f"{'drummer':<10}" + "".join(f"{b:>10}" for b in BEATERS))
    print("-" * 50)
    grid = collections.defaultdict(list)
    for s, (d, b, _st) in meta.items():
        grid[(d, b)].append(s)
    for d in (1, 2, 3):
        print(f"{d:<10}" + "".join(f"{len(grid[(d, b)]):>10}" for b in BEATERS))

    print(f"\nMICRO F1 by drummer and beater")
    print(f"{'drummer':<10}" + "".join(f"{b:>10}" for b in BEATERS))
    print("-" * 50)
    for d in (1, 2, 3):
        line = f"{d:<10}"
        for b in BEATERS:
            picks = grid[(d, b)]
            line += f"{micro(rows, picks):>10.3f}" if picks else f"{'-':>10}"
        print(line)

    rng = random.Random(args.seed)
    print(f"\nwithin drummer 1 only -- player held constant, style varies")
    print(f"{'comparison':<26}{'soft':>8}{'sticks':>9}{'diff':>9}{'95% CI':>20}")
    print("-" * 74)
    sticks = grid[(1, "sticks")]
    for b in ("rods", "brushes", "mallets"):
        soft = grid[(1, b)]
        if not soft or not sticks:
            continue
        a, c = micro(rows, soft), micro(rows, sticks)
        diffs = []
        for _ in range(args.rounds):
            sa = [soft[rng.randrange(len(soft))] for _ in soft]
            sc = [sticks[rng.randrange(len(sticks))] for _ in sticks]
            x, y = micro(rows, sa), micro(rows, sc)
            if x == x and y == y:
                diffs.append(x - y)
        lo, hi = interval(diffs)
        flag = "" if lo <= 0 <= hi else "  significant"
        print(f"{b + ' vs sticks':<26}{a:>8.3f}{c:>9.3f}{a - c:>+9.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{flag}")

    print(f"\nfor comparison, pooled across all three drummers")
    print(f"{'beater':<12}{'recs':>6}{'MICRO F1':>10}")
    print("-" * 30)
    for b in BEATERS:
        picks = [s for s, (_d, bb, _st) in meta.items() if bb == b]
        if picks:
            print(f"{b:<12}{len(picks):>6}{micro(rows, picks):>10.3f}")

    print("\nThe published control holds style constant and varies the drummer. This one\n"
          "holds the drummer constant and varies style. An effect present in both is not\n"
          "explained by either confound; one present in only the first was probably the\n"
          "drummer, and in only the second probably the repertoire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
