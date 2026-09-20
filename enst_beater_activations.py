"""At a known mallet hit, is the model nearly seeing it, or seeing nothing?

`enst_mallets.py` established that soft beaters fail as a class: MICRO 0.409 for mallets
against 0.841 for everything else, and only 41% of annotated onsets get any estimate at
all. The obvious remedy is to lower the thresholds. That is also what looked obvious for
the deaf passages in Limitation 8, where the activations turned out to be noise rather
than a near miss, so no threshold could have recovered them.

This asks the same question for beaters, and it is answerable exactly because the
annotation says where the hits are. For every annotated onset, take the model's
activation for that onset's own class in a short window around it:

  near miss   activation sits below the threshold but well above the floor. Lowering the
              threshold trades false positives for these, which is a tuning decision.
  noise       activation is at the level of the gaps between hits. Nothing recovers it,
              and lowering thresholds would only add false notes elsewhere.

The control matters more than the mallet number: the same afro material played with
sticks, through the same model, so the comparison is beater against beater rather than
recording against recording.

    python enst_beater_activations.py
    python enst_beater_activations.py --limit 4 --device cpu
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

# ADTOF emits five channels in this order; ORDER in benchmark_mdb uses the same classes
# under different names, so the map is explicit rather than positional-by-luck.
ACT_INDEX = {"KD": 0, "SD": 1, "TT": 2, "HH": 3, "CY": 4}
FPS = 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", default="wet_mix", choices=["wet_mix", "dry_mix"])
    ap.add_argument("--soft", default="mallets",
                    choices=["mallets", "brushes", "rods"],
                    help="the soft beater to compare against sticks")
    ap.add_argument("--style", default="afro",
                    help="material played with both beaters, so the comparison is fair")
    ap.add_argument("--limit", type=int, default=None,
                    help="recordings per beater")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", type=float, default=0.05,
                    help="seconds either side of the annotated onset")
    args = ap.parse_args()

    import numpy as np

    import drum2midi
    from benchmark_enst import IGNORED, LABEL_TO_CLASS
    from blind_spots import activations

    thr = list(drum2midi.DEFAULT_THRESHOLDS)
    data = ROOT / "enst" / "enst_drums_public"
    if not data.exists():
        print(f"ENST not found at {data}; run: python fetch_enst.py")
        return 1

    picked = {args.soft: [], "sticks": []}
    for d in (1, 2, 3):
        for ann in sorted((data / f"drummer_{d}" / "annotation").glob("*.txt")):
            stem = ann.stem
            if args.style not in stem:
                continue
            for beater in picked:
                if beater in stem:
                    wav = data / f"drummer_{d}" / "audio" / args.mix / f"{stem}.wav"
                    if wav.exists():
                        picked[beater].append((f"d{d}/{stem}", wav, ann))
    for b in picked:
        picked[b] = picked[b][: args.limit]

    if not picked[args.soft] or not picked["sticks"]:
        print(f"need both beaters on '{args.style}' material; found "
              f"{len(picked[args.soft])} {args.soft} and {len(picked['sticks'])} stick")
        return 1

    print(f"thresholds {thr}")
    print(f"material '{args.style}', {args.mix}, no separation, device={args.device}")
    print(f"{len(picked[args.soft])} {args.soft} and {len(picked['sticks'])} stick "
          f"recordings\n")

    summary = {}
    for beater, items in picked.items():
        at_onset, floors = [], []
        for name, wav, ann in items:
            act = activations(wav, args.device)
            n = act.shape[0]
            hits = []
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 2 or p[1] in IGNORED:
                    continue
                cls = LABEL_TO_CLASS.get(p[1])
                if cls is None:
                    continue
                t = float(p[0])
                lo = max(0, int((t - args.window) * FPS))
                hi = min(n, int((t + args.window) * FPS) + 1)
                if hi <= lo:
                    continue
                col = ACT_INDEX[cls]
                peak = float(act[lo:hi, col].max())
                hits.append((peak, thr[col]))
            if not hits:
                continue
            at_onset.extend(hits)
            # The floor is what the same channel does away from any annotated hit --
            # the level a threshold would have to clear just to ignore silence.
            mask = np.ones(n, dtype=bool)
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 2:
                    continue
                t = float(p[0])
                mask[max(0, int((t - 0.1) * FPS)):min(n, int((t + 0.1) * FPS))] = False
            if mask.any():
                floors.append(float(np.percentile(act[mask].max(axis=1), 95)))
            print(f"  {name[:52]:<54}{len(hits):>5} onsets", flush=True)

        peaks = np.array([h[0] for h in at_onset])
        thrs = np.array([h[1] for h in at_onset])
        ratio = peaks / thrs
        summary[beater] = {
            "n": len(peaks),
            "median": float(np.median(ratio)),
            "over": float((ratio >= 1).mean()),
            "near": float(((ratio >= 0.5) & (ratio < 1)).mean()),
            "noise": float((ratio < 0.5).mean()),
            "floor": float(np.mean(floors)) if floors else float("nan"),
        }

    print(f"\nactivation at annotated onsets, as a fraction of that class's threshold")
    print(f"{'beater':<10}{'onsets':>8}{'median':>9}{'>=thr':>9}"
          f"{'near miss':>11}{'noise':>9}{'gap 95th':>10}")
    print("-" * 66)
    for b in ("sticks", args.soft):
        s = summary[b]
        print(f"{b:<10}{s['n']:>8}{s['median']:>9.2f}{s['over']:>8.0%}"
              f"{s['near']:>11.0%}{s['noise']:>9.0%}{s['floor']:>10.3f}")

    m, k = summary[args.soft], summary["sticks"]
    print(f"\n'near miss' is half a threshold up to the threshold; 'noise' is below "
          f"half.\n'gap 95th' is what the channel does between hits: a threshold below "
          f"it fires\non silence.")
    # An earlier version compared the two bands to each other and announced "mostly
    # noise" on a 13-against-11 margin, which read as a finding and was a coin toss.
    # The median against the stick control is the comparison that carries weight, and
    # the recoverable share is stated as a bound rather than a verdict.
    ratio = m["median"] / max(k["median"], 1e-9)
    print(f"\n{args.soft} sit at {m['median']:.2f} of their threshold against "
          f"{k['median']:.2f} for sticks, a factor of {1 / max(ratio, 1e-9):.1f}.")
    print(f"Sticks clear their threshold on {k['over']:.0%} of onsets, "
          f"{args.soft} on {m['over']:.0%}.")
    print(f"At most {m['near']:.0%} of {args.soft} onsets are recoverable by a lower "
          f"threshold;\n{m['noise']:.0%} sit below half a threshold and nothing reaches "
          f"those.")
    if m["median"] < 1.0:
        print(f"A median below 1.0 means the typical {args.soft} onset does not clear "
              f"its threshold\nat all, which is a different condition from a class that "
              f"clears it and scores badly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
