"""Does the pipeline fail on mallets as a class, or was that one odd recording?

The ENST run reported exactly one recording of 210 that transcribed to nothing at all:
`048_phrase_afro_simple_slow_mallets`. Mallets have a soft, slow attack and the whole
pipeline is built on onset detection, so a class failure is the obvious hypothesis -- and
the obvious hypothesis is what this project keeps having to withdraw.

ENST encodes the beater in every filename: sticks, rods, brushes, mallets. So the
question is answerable by counting rather than by arguing, using the transcriptions the
benchmark already cached. No inference, no GPU.

A class failure and one bad file look identical if you only look at the file that failed.
They look nothing alike once the other six mallet recordings are scored.

    python enst_mallets.py
    python enst_mallets.py --tag enst --rounds 4000
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

BEATERS = ("sticks", "rods", "brushes", "mallets")


def beater_of(stem: str) -> str | None:
    for b in BEATERS:
        if b in stem:
            return b
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="enst", help="cached transcriptions under bench/")
    ap.add_argument("--mix", default="wet_mix", choices=["wet_mix", "dry_mix"])
    ap.add_argument("--kinds", default="phrase,solo,minus-one,MIDI-minus-one")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mir_eval
    import numpy as np
    import pretty_midi

    from benchmark_mdb import CLASSES, ORDER, WINDOW
    from benchmark_enst import IGNORED, LABEL_TO_CLASS

    data = ROOT / "enst" / "enst_drums_public"
    outdir = ROOT / "bench" / args.tag
    if not data.exists():
        print(f"ENST not found at {data}; run: python fetch_enst.py")
        return 1
    if not outdir.exists():
        print(f"no cached transcriptions at {outdir}; run: python benchmark_enst.py")
        return 1
    print(f"scoring cached transcriptions in {outdir}")

    kinds = set(args.kinds.split(","))
    rows = {}
    for d in (1, 2, 3):
        for ann in sorted((data / f"drummer_{d}" / "annotation").glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in kinds:
                continue
            wav = data / f"drummer_{d}" / "audio" / args.mix / f"{ann.stem}.wav"
            if not wav.exists():
                continue
            beater = beater_of(ann.stem)
            if beater is None:
                continue

            ref = {k: [] for k in CLASSES}
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 2 or p[1] in IGNORED:
                    continue
                cls = LABEL_TO_CLASS.get(p[1])
                if cls is not None:
                    ref[cls].append(float(p[0]))

            mid = outdir / f"{ann.stem}_d{d}.mid"
            if mid.exists():
                notes = [n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                         for n in inst.notes]
            else:
                notes = []          # scored as zero estimates, never dropped

            counts = {}
            for k in ORDER:
                r = np.array(sorted(ref[k]))
                e = np.array(sorted(n.start for n in notes if n.pitch in CLASSES[k]))
                if r.size and e.size:
                    _f, prec, _rec = mir_eval.onset.f_measure(r, e, window=WINDOW)
                    tp = int(round(prec * len(e)))
                else:
                    tp = 0
                counts[k] = {"tp": tp, "ref": int(r.size), "est": int(e.size)}
            rows[f"d{d}/{ann.stem}"] = {"beater": beater, "counts": counts,
                                        "silent": not notes}

    if not rows:
        print("no recordings matched")
        return 1

    def micro(picks) -> float:
        tp = sum(rows[n]["counts"][k]["tp"] for n in picks for k in ORDER)
        ref = sum(rows[n]["counts"][k]["ref"] for n in picks for k in ORDER)
        est = sum(rows[n]["counts"][k]["est"] for n in picks for k in ORDER)
        if not ref or not est:
            return float("nan")
        p, r = tp / est, tp / ref
        return 2 * p * r / max(p + r, 1e-9)

    print(f"\n{len(rows)} recordings carry a beater in the filename\n")
    print(f"{'beater':<10}{'recs':>6}{'silent':>8}{'ref':>8}{'est':>8}{'MICRO F1':>10}")
    print("-" * 50)
    by_beater = {}
    for b in BEATERS:
        picks = [n for n in rows if rows[n]["beater"] == b]
        if not picks:
            continue
        by_beater[b] = picks
        ref = sum(rows[n]["counts"][k]["ref"] for n in picks for k in ORDER)
        est = sum(rows[n]["counts"][k]["est"] for n in picks for k in ORDER)
        mute = sum(1 for n in picks if rows[n]["silent"])
        print(f"{b:<10}{len(picks):>6}{mute:>8}{ref:>8}{est:>8}{micro(picks):>10.3f}")

    if "mallets" not in by_beater:
        print("\nno mallet recordings found")
        return 1

    mal = by_beater["mallets"]
    rest = [n for n in rows if rows[n]["beater"] != "mallets"]
    diff = micro(mal) - micro(rest)

    # Resample recordings, because the recording is the unit that varies. Seven of them
    # is a small sample and the interval is the only honest way to say so.
    rng = random.Random(args.seed)
    draws = []
    for _ in range(args.rounds):
        a = [mal[rng.randrange(len(mal))] for _ in mal]
        b = [rest[rng.randrange(len(rest))] for _ in rest]
        draws.append(micro(a) - micro(b))
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]

    print(f"\nmallets against every other beater, {args.rounds} resamples over recordings")
    print(f"  mallets  {micro(mal):.3f}   ({len(mal)} recordings)")
    print(f"  others   {micro(rest):.3f}   ({len(rest)} recordings)")
    print(f"  difference {diff:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  {'significant' if lo > 0 or hi < 0 else 'NOT significant'}")

    print("\nevery mallet recording, so the failing one can be seen in context")
    print(f"{'recording':<48}{'ref':>6}{'est':>6}{'MICRO':>8}")
    print("-" * 68)
    for n in sorted(mal):
        ref = sum(rows[n]["counts"][k]["ref"] for k in ORDER)
        est = sum(rows[n]["counts"][k]["est"] for k in ORDER)
        mark = "  <- silent" if rows[n]["silent"] else ""
        print(f"{n[:47]:<48}{ref:>6}{est:>6}{micro([n]):>8.3f}{mark}")

    # Seven of the eight mallet recordings are one drummer playing afro material, so the
    # deficit above is beater and material confounded. ENST also has that same material
    # played with sticks, which separates them.
    def style_of(name: str) -> str:
        parts = name.split("/")[-1].split("_")
        return parts[2] if len(parts) > 2 else "?"

    styles = {style_of(n) for n in mal}
    print("\ncontrol: the same material played with sticks, which separates beater "
          "from repertoire")
    print(f"{'style':<10}{'mallets':>20}{'sticks':>20}{'difference':>14}")
    print("-" * 64)
    MIN_RECS = 3        # resampling 1 of 1 returns a point interval that always
                        # "excludes zero"; refusing is the only honest output
    for st in sorted(styles):
        m = [n for n in mal if style_of(n) == st]
        s = [n for n in rows if rows[n]["beater"] == "sticks" and style_of(n) == st]
        if not s:
            print(f"{st:<10}{micro(m):>13.3f} ({len(m):>2}){'  no sticks':>20}")
            continue
        if len(m) < MIN_RECS or len(s) < MIN_RECS:
            print(f"{st:<10}{micro(m):>13.3f} ({len(m):>2}){micro(s):>13.3f} "
                  f"({len(s):>2}){micro(m) - micro(s):>+10.3f}"
                  f"   too few to resample")
            continue
        d = micro(m) - micro(s)
        draws = []
        for _ in range(args.rounds):
            a = [m[rng.randrange(len(m))] for _ in m]
            b = [s[rng.randrange(len(s))] for _ in s]
            draws.append(micro(a) - micro(b))
        draws.sort()
        clo, chi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
        verdict = "significant" if clo > 0 or chi < 0 else "not significant"
        print(f"{st:<10}{micro(m):>13.3f} ({len(m):>2}){micro(s):>13.3f} ({len(s):>2})"
              f"{d:>+10.3f}  [{clo:+.3f}, {chi:+.3f}] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
