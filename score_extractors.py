"""Scores whatever the extractor comparison has already produced, without re-running it.

compare_extractors.py transcribes and scores in one pass, so a run that is interrupted
-- or, as happened here, a run that was accidentally started twice and deadlocked over
the GPU -- leaves a folder full of usable MIDI and prints nothing.

This reads that folder and reports what is there, per extractor, saying plainly how many
tracks each figure rests on. Extractors with different track counts are not compared
directly; only the tracks all of them finished are used for the head-to-head.

    python score_extractors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_extractors import ANN, CACHE, EXTRACTORS, prf, read_midi, score  # noqa: E402
from benchmark_mdb import ORDER, PRETTY, read_annotation  # noqa: E402


def collect(name: str) -> dict:
    """Maps track stem -> (reference, estimate) for every MIDI this extractor made."""
    out = {}
    folder = CACHE / name
    if not folder.exists():
        return out
    for mid in sorted(folder.glob("*.mid")):
        # "MusicDelta_Rock_MIX" -> the annotation is filed under "..._Drum"
        stem = mid.stem.replace("_MIX", "")
        ann = next((p for p in ANN.glob(f"{stem}*.txt")), None)
        if ann is None:
            continue
        try:
            out[stem] = (read_annotation(ann), read_midi(mid))
        except Exception as exc:
            print(f"  ! {mid.name}: {exc}")
    return out


def table(title: str, acc: dict, tracks: int) -> None:
    print(f"\n{title}  ({tracks} tracks)")
    print(f"{'drum':<10}{'ref':>7}{'est':>7}{'match':>7}"
          f"{'prec':>8}{'rec':>8}{'F1':>8}")
    print("-" * 55)
    tot = {"tp": 0, "ref": 0, "est": 0}
    for k in ORDER:
        c = acc[k]
        p, r, f = prf(c)
        print(f"{PRETTY[k]:<10}{c['ref']:>7}{c['est']:>7}{c['tp']:>7}"
              f"{p:>8.3f}{r:>8.3f}{f:>8.3f}")
        for key in tot:
            tot[key] += c[key]
    p, r, f = prf(tot)
    print("-" * 55)
    print(f"{'MICRO':<10}{tot['ref']:>7}{tot['est']:>7}{tot['tp']:>7}"
          f"{p:>8.3f}{r:>8.3f}{f:>8.3f}")
    return f


def bootstrap(got: dict, shared: set, rounds: int = 4000, seed: int = 0) -> None:
    """Asks whether the gaps between extractors survive resampling the tracks.

    A MICRO F1 computed over 7996 onsets looks precise, but those onsets come from 23
    recordings, and a recording is the unit that actually varies. Resampling tracks with
    replacement gives an interval that reflects that, and it is the difference between
    "this extractor is better" and "we cannot tell on 23 tracks".
    """
    import random
    names = [n for n in got if got[n]]
    if len(names) < 2 or not shared:
        return
    tracks = sorted(shared)
    rng = random.Random(seed)

    def micro(name: str, picks) -> float:
        acc = score([got[name][t] for t in picks])
        tot = {"tp": 0, "ref": 0, "est": 0}
        for k in acc:
            for key in tot:
                tot[key] += acc[k][key]
        return prf(tot)[2]

    draws = [[tracks[rng.randrange(len(tracks))] for _ in tracks]
             for _ in range(rounds)]
    curves = {n: [micro(n, d) for d in draws] for n in names}

    print("\n" + "=" * 55)
    print(f"bootstrap over {len(tracks)} tracks, {rounds} resamples")
    print(f"{'extractor':<14}{'MICRO F1':>10}{'95% CI':>20}")
    for n in names:
        c = sorted(curves[n])
        lo, hi = c[int(0.025 * rounds)], c[int(0.975 * rounds)]
        print(f"{n:<14}{micro(n, tracks):>10.4f}   [{lo:.4f}, {hi:.4f}]")

    print("\npairwise differences (positive means the first one is ahead)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diffs = sorted(curves[a][k] - curves[b][k] for k in range(rounds))
            lo, hi = diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds)]
            wins = sum(d > 0 for d in diffs) / rounds
            verdict = "significant" if lo > 0 or hi < 0 else "NOT significant"
            print(f"  {a} - {b}: {micro(a, tracks) - micro(b, tracks):+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]  wins {wins:.0%}  {verdict}")


def main() -> int:
    got = {name: collect(name) for name in EXTRACTORS}
    for name, d in got.items():
        print(f"{name:<14}{len(d):>3} tracks scored")

    shared = set.intersection(*(set(d) for d in got.values() if d)) if any(got.values()) else set()
    print(f"\ncommon to all: {len(shared)} tracks")

    micro = {}
    for name, d in got.items():
        if not d:
            continue
        micro[name] = table(f"=== {name} (all it finished)", score(d.values()), len(d))

    if len(shared) >= 2:
        print("\n" + "=" * 55)
        print("head to head, on the tracks every extractor finished")
        fair = {}
        for name, d in got.items():
            if not d:
                continue
            fair[name] = table(f"=== {name}", score([d[s] for s in sorted(shared)]),
                               len(shared))
        best = max(fair, key=fair.get)
        print(f"\nbest on the shared set: {best} (MICRO F1 {fair[best]:.4f})")
        for name, f in sorted(fair.items(), key=lambda kv: -kv[1]):
            if name != best:
                print(f"  {name}: {f:.4f}  ({fair[best] - f:+.4f} behind)")
        bootstrap(got, shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
