"""Do "open" hi-hats ring longer than "closed" ones? Seven acoustic tracks, with an interval.

The 1.51x figure was withdrawn because the recording behind it sits outside the public
test set and cannot be checked by anyone but the machine that produced it. Disco gave
0.66x, the other way, but one track is one track -- which is the error this project has
retracted four times in a day.

So this counts the corpus. Every MDB track ReStem has rendered here is measured the same
way, per-track ratios are reported beside the pooled one, and the interval comes from
resampling tracks rather than hits, because the recording is the unit that varies:
one track with an unusual kit moves the total far more than one unusual cymbal strike.

Decay follows check_hihat_decay.py exactly -- time to fall 20 dB below the hit's own peak,
measured on ReStem's hi-hat stem at the times our transcription marks -- so the numbers
are comparable with the ones already published. The high-pass is on by default: the trial
watermark sits at 603 Hz and a hi-hat near 12 kHz, so removing it costs nothing.

A result near 1.0 is a real answer. It would mean the distinction is not measurable this
way, which is a different and more useful statement than "withdrawn for lack of
provenance".

    python hihat_decay_corpus.py --stems bench/restem_stems --midi bench/note36
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SR = 22050
LABEL = {42: "closed", 44: "pedal", 46: "open"}


def highpass(y, cutoff: float):
    """Windowed-sinc FIR, matching check_hihat_decay.py rather than adding scipy."""
    import numpy as np
    n = 257
    fc = cutoff / (SR / 2)
    k = np.arange(n) - (n - 1) / 2
    lp = np.sinc(fc * k) * fc * np.hanning(n)
    lp /= lp.sum()
    hp = -lp
    hp[(n - 1) // 2] += 1.0
    return np.convolve(y, hp, mode="same")


def decays(stem: Path, mid: Path, cutoff: float) -> dict:
    import librosa
    import numpy as np
    import pretty_midi

    y, _ = librosa.load(str(stem), sr=SR, mono=True)
    if cutoff > 0:
        y = highpass(y, cutoff)

    def decay_at(at: float, floor_db: float = -20.0, limit: float = 0.6):
        a = int(at * SR)
        b = min(a + int(limit * SR), len(y))
        if b - a < int(0.05 * SR):
            return None
        seg = np.abs(y[a:b])
        if seg.max() < 1e-4:
            return None
        win = int(0.005 * SR)
        env = np.convolve(seg, np.ones(win) / win, mode="same")
        target = env.max() * (10 ** (floor_db / 20))
        below = np.where(env < target)[0]
        below = below[below > np.argmax(env)]
        return float(below[0] / SR) if len(below) else limit

    out = {}
    capped = {}
    for inst in pretty_midi.PrettyMIDI(str(mid)).instruments:
        for n in inst.notes:
            name = LABEL.get(n.pitch)
            if not name:
                continue
            d = decay_at(n.start)
            if d is not None:
                out.setdefault(name, []).append(d)
                # A hit whose envelope never falls 20 dB inside the window has not been
                # measured, it has been truncated -- usually because the next hit lands
                # first. Counted, because a track where both medians sit on the cap
                # yields a ratio of exactly 1.00 that means nothing at all.
                if abs(d - 0.6) < 1e-9:
                    capped[name] = capped.get(name, 0) + 1
    return out, capped


def ratio(groups: dict) -> float | None:
    o, c = groups.get("open"), groups.get("closed")
    if not o or not c:
        return None
    return statistics.median(o) / statistics.median(c)


def pairwise(o, c) -> float:
    return sum(1 for x in o for z in c if x > z) / (len(o) * len(c))


def interval(v, lo=0.025, hi=0.975):
    v = sorted(v)
    return v[int(lo * len(v))], v[int(hi * len(v))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", type=Path, required=True,
                    help="folder of per-track capture folders each holding hh.wav")
    ap.add_argument("--midi", type=Path, required=True)
    ap.add_argument("--highpass", type=float, default=1000.0)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-hits", type=int, default=5,
                    help="a group with fewer hits than this is not a per-track estimate")
    args = ap.parse_args()

    per_track, capped_by = {}, {}
    for folder in sorted(args.stems.iterdir()):
        hh = folder / "hh.wav"
        if not hh.exists():
            continue
        mid = args.midi / f"{folder.name}.mid"
        if not mid.exists():
            print(f"  no transcription for {folder.name}; skipped")
            continue
        per_track[folder.name], capped_by[folder.name] = decays(hh, mid, args.highpass)

    if not per_track:
        print(f"no usable captures under {args.stems}")
        return 1

    print(f"high-pass {args.highpass:.0f} Hz, decay = time to fall 20 dB below peak\n")
    print(f"{'track':<14}{'closed':>8}{'open':>7}{'med c':>9}{'med o':>9}"
          f"{'ratio':>8}{'pairwise':>10}{'capped':>9}")
    print("-" * 75)

    usable, saturated = [], []
    for name, g in sorted(per_track.items()):
        short = name.replace("MusicDelta_", "").replace("_Drum", "")
        c, o = g.get("closed", []), g.get("open", [])
        r = ratio(g)
        thin = len(c) < args.min_hits or len(o) < args.min_hits
        ncap = sum(capped_by[name].values())
        share = ncap / max(len(c) + len(o), 1)
        # Both medians at (or within a hair of) the window edge means the ratio is an
        # artefact of the window, not a property of the hits. Tested against 0.9x the
        # limit rather than exact equality: medians of even-length samples average two
        # values and never land exactly on the cap.
        edge = 0.6 * 0.9
        pinned = bool(c and o and statistics.median(c) >= edge
                      and statistics.median(o) >= edge)
        if pinned:
            saturated.append(short)
        rs = "-" if r is None else (f"{r:.2f}x" + ("*" if thin else "")
                                    + ("!" if pinned else ""))
        pw = "-" if not (c and o) else f"{pairwise(o, c):.0%}"
        print(f"{short:<14}{len(c):>8}{len(o):>7}"
              f"{(statistics.median(c) * 1000 if c else 0):>8.0f}m"
              f"{(statistics.median(o) * 1000 if o else 0):>8.0f}m{rs:>8}{pw:>10}"
              f"{share:>9.0%}")
        if r is not None and not pinned:
            usable.append((name, g, thin))

    print(f"\n* fewer than {args.min_hits} hits in one group; the per-track ratio is "
          f"not an estimate")
    print(f"! both medians pinned to the 0.6 s window: the next hit arrives before the "
          f"decay completes,\n  so the ratio is 1.00 by construction. Excluded from the "
          f"pooled figure.")
    if saturated:
        print(f"  excluded on that ground: {', '.join(saturated)}")

    if len(usable) < 3:
        print(f"\nonly {len(usable)} track(s) with both articulations; no interval")
        return 0

    def pooled(names) -> float | None:
        o, c = [], []
        for n in names:
            g = dict(per_track[n])
            o += g.get("open", [])
            c += g.get("closed", [])
        if not o or not c:
            return None
        return statistics.median(o) / statistics.median(c)

    names = [n for n, _g, _t in usable]
    point = pooled(names)
    rng = random.Random(args.seed)
    draws = []
    for _ in range(args.rounds):
        pick = [names[rng.randrange(len(names))] for _ in names]
        v = pooled(pick)
        if v is not None:
            draws.append(v)
    lo, hi = interval(draws)

    allo = [x for n in names for x in per_track[n].get("open", [])]
    allc = [x for n in names for x in per_track[n].get("closed", [])]

    print(f"\n{'=' * 66}")
    print(f"pooled over {len(names)} tracks: {len(allc)} closed, {len(allo)} open hits")
    print(f"  open / closed median decay   {point:.2f}x")
    print(f"  95% CI, resampling tracks    [{lo:.2f}, {hi:.2f}]")
    print(f"  a random open hit outlasts a random closed one  {pairwise(allo, allc):.0%}")
    print(f"{'=' * 66}")

    if lo <= 1.0 <= hi:
        print("\nThe interval contains 1.0: on this corpus the distinction is not\n"
              "measurable this way. That is an answer, not a missing result.")
    elif hi < 1.0:
        print("\nThe interval is entirely below 1.0: hits we label 'open' ring SHORTER\n"
              "than those we label 'closed', which inverts the physical expectation and\n"
              "means the labels are not tracking what they claim to.")
    else:
        print("\nThe interval is entirely above 1.0: open hits do ring longer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
