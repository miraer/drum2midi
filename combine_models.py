"""If one model wins the snare and another wins the cymbals, what would routing buy?

The survey found a split rather than a winner: Vogl's CRNN beats us on snare (+0.111 and
+0.081, both intervals clear of zero) while we beat it on cymbals (+0.143 and +0.231,
likewise). The obvious question is whether taking each class from whichever model is
better would beat either alone.

That question can be answered without running anything. MICRO F1 is computed from summed
counts across classes, so if every note of a class comes from one system, the routed
precision and recall are exact arithmetic on numbers already measured. Nothing here is
simulated or approximated.

WHAT THIS SCRIPT CANNOT DO, and both are reasons not to call its output a result:

  1. No confidence interval. It reads corpus totals, and the bootstrap resamples tracks,
     because the recording is the unit that varies. Getting an interval needs per-track
     counts for every system -- which for Vogl means its MIDI, which lives on the other
     machine. Until then the difference is a point estimate and the project's own rule
     says a difference without an interval is not real.

  2. The routing is chosen by looking at the scores it is then judged by. That is
     selection on the test set, so the headline figure is an ORACLE: an upper bound on
     what routing could buy, not an estimate of what it would buy on new material. The
     honest test is to fix the routing on one corpus and score it on another.

What it does do is bound the idea cheaply, and check whether the bound survives the
decisions that are only noise -- if flipping a class whose interval crosses zero moves
the total, the routing was reading tea leaves.

    python combine_models.py
    python combine_models.py --baseline ours --top 12
    python combine_models.py --counts fresh.csv     # system,drum,ref,est,match
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DRUMS = ["kick", "snare", "hi-hat", "toms", "cymbals"]

# Per-class onset counts on the 23 MDB Drums tracks: one scorer, 50 ms MIREX tolerance.
# Kept in the source rather than a data file because this repository's rule is that every
# number comes from a script in it, and a script that reads a file nobody else has is not
# reproducible. `ref` is the reference annotation, identical for every system.
#
#   ours          ADTOF with snare 0.14 and the adaptive tom policy, no separation --
#                 the survey basis, MICRO 0.860. Regenerate:
#                     python benchmark_mdb.py --rescore --tag nosep_ceiling
#   ours+sep      what this project actually ships: the same model with separation in
#                 front, MICRO 0.882, the README headline. Regenerate:
#                     python benchmark_mdb.py --rescore --tag ceiling
#   vogl-*        Vogl's DAFx'18 CRNNs, run by the second machine on 20 September 2026
#                 (score_vogl_CRNN_8.log, score_vogl_CRNN_18.log) and transcribed here.
#                 NOT produced on this machine. the 0.10 variant is the threshold that scored best
#                 on this corpus -- hindsight, in its favour, excluded by default. @0.15
#                 is parameter-free and is the defensible one.
COUNTS: dict[str, dict[str, tuple[int, int, int]]] = {
    "ours": {
        "kick": (1539, 1561, 1489), "snare": (2654, 2506, 2077),
        "hi-hat": (2639, 2489, 2213), "toms": (90, 134, 66),
        "cymbals": (1002, 1142, 933)},
    "ours+sep": {
        "kick": (1539, 1570, 1493), "snare": (2654, 2808, 2304),
        "hi-hat": (2639, 2848, 2448), "toms": (90, 134, 66),
        "cymbals": (1002, 1144, 932)},
    "vogl-8/0.10": {
        "kick": (1539, 1586, 1522), "snare": (2654, 2783, 2491),
        "hi-hat": (2639, 3154, 2477), "toms": (90, 254, 73),
        "cymbals": (1002, 1420, 881)},
    "vogl-8/0.15": {
        "kick": (1539, 1552, 1504), "snare": (2654, 2359, 2197),
        "hi-hat": (2639, 2758, 2337), "toms": (90, 140, 64),
        "cymbals": (1002, 685, 558)},
    "vogl-18/0.15": {
        "kick": (1539, 1588, 1515), "snare": (2654, 2466, 2269),
        "hi-hat": (2639, 2937, 2310), "toms": (90, 95, 52),
        "cymbals": (1002, 676, 536)},
}

# Classes where the survey's paired bootstrap against ours crossed zero, so the routing
# decision on them is a coin flip rather than evidence. Used for the robustness check.
COIN_FLIP = {"hi-hat"}


def f1(est: int, ref: int, match: int) -> float:
    if not est or not ref or not match:
        return 0.0
    p, r = match / est, match / ref
    return 2 * p * r / (p + r)


def load(path: Path) -> dict[str, dict[str, tuple[int, int, int]]]:
    rows = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    out: dict[str, dict[str, tuple[int, int, int]]] = defaultdict(dict)
    for row in csv.DictReader(rows):
        out[row["system"]][row["drum"]] = (
            int(row["ref"]), int(row["est"]), int(row["match"]))
    return dict(out)


def micro(counts: dict[str, dict], routing: dict[str, str]) -> tuple[float, int, int, int]:
    ref = est = match = 0
    for drum, system in routing.items():
        r, e, m = counts[system][drum]
        ref, est, match = ref + r, est + e, match + m
    return f1(est, ref, match), ref, est, match


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=Path, default=None,
                    help="CSV of system,drum,ref,est,match; overrides the built-in table")
    ap.add_argument("--baseline", default="ours")
    ap.add_argument("--exclude", action="append",
                    default=["vogl-8/0.10"],
                    help="system to leave out, repeatable; the hindsight-tuned "
                         "vogl-8/0.10 is excluded unless you clear this")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    if args.counts is not None:
        if not args.counts.exists():
            print(f"no counts file: {args.counts}")
            return 1
        counts = load(args.counts)
        source = args.counts.name
    else:
        counts = COUNTS
        source = "the built-in survey table"
    systems = [s for s in counts if s not in args.exclude]
    if args.baseline not in systems:
        print(f"baseline {args.baseline!r} not in {sorted(counts)}")
        return 1
    missing = [(s, d) for s in systems for d in DRUMS if d not in counts[s]]
    if missing:
        print(f"incomplete counts: {missing}")
        return 1

    print(f"{len(systems)} systems, {len(DRUMS)} classes, corpus totals from "
          f"{source}\n")

    print(f"{'per-class F1':<16}" + "".join(f"{d:>10}" for d in DRUMS) + f"{'MICRO':>9}")
    print("-" * 81)
    single: dict[str, float] = {}
    for s in systems:
        whole = {d: s for d in DRUMS}
        m, *_ = micro(counts, whole)
        single[s] = m
        cells = "".join(f"{f1(counts[s][d][1], counts[s][d][0], counts[s][d][2]):>10.3f}"
                        for d in DRUMS)
        print(f"{s:<16}{cells}{m:>9.3f}")
    best_single = max(single, key=single.get)
    print("-" * 81)

    # every way of assigning each class to a system
    scored = []
    for combo in itertools.product(systems, repeat=len(DRUMS)):
        routing = dict(zip(DRUMS, combo))
        m, ref, est, match = micro(counts, routing)
        scored.append((m, routing, est, match))
    scored.sort(key=lambda t: -t[0])

    base = single[args.baseline]
    print(f"\nall {len(scored)} routings, best {args.top} "
          f"(delta against {args.baseline} at {base:.3f})\n")
    print(f"{'MICRO':>7}{'delta':>9}   " + "  ".join(f"{d:<12}" for d in DRUMS))
    print("-" * 81)
    for m, routing, _, _ in scored[:args.top]:
        cells = "  ".join(f"{routing[d]:<12}" for d in DRUMS)
        print(f"{m:>7.3f}{m - base:>+9.3f}   {cells}")

    oracle_m, oracle, _, _ = scored[0]
    worst_m = scored[-1][0]
    print("-" * 81)
    print(f"{'worst':>7}{worst_m - base:>+9.3f}   "
          + "  ".join(f"{scored[-1][1][d]:<12}" for d in DRUMS))

    # does the answer depend on decisions that were noise?
    flips = [d for d in DRUMS if d in COIN_FLIP]
    if flips:
        print(f"\nrobustness: the survey's interval crossed zero on {', '.join(flips)}, "
              f"so that\nchoice is noise. Forcing it the other way:")
        for d in flips:
            for alt in systems:
                if alt == oracle[d]:
                    continue
                forced = dict(oracle, **{d: alt})
                m, *_ = micro(counts, forced)
                print(f"  {d} -> {alt:<14} MICRO {m:.3f}  "
                      f"({m - oracle_m:+.3f} against the oracle)")

    print(f"\nbest single system: {best_single} at {single[best_single]:.3f}")
    print(f"oracle routing:     {oracle_m:.3f}  "
          f"({oracle_m - single[best_single]:+.3f} against it, "
          f"{oracle_m - base:+.3f} against {args.baseline})")
    print(f"spread across routings: {worst_m:.3f} to {oracle_m:.3f}")

    # where does the gain actually come from -- one class at a time, everything else
    # left at the baseline, so the classes cannot borrow credit from each other
    print(f"\none class at a time, everything else {args.baseline}:")
    print(f"{'class':<10}{'share':>8}{'from':>15}{'F1':>8}{'was':>8}{'MICRO':>9}{'delta':>9}")
    print("-" * 67)
    ref_total = sum(counts[args.baseline][d][0] for d in DRUMS)
    for d in DRUMS:
        share = counts[args.baseline][d][0] / ref_total
        alts = [(f1(counts[s][d][1], counts[s][d][0], counts[s][d][2]), s) for s in systems]
        best_f1, best_s = max(alts)
        was = f1(*[counts[args.baseline][d][i] for i in (1, 0, 2)])
        m, *_ = micro(counts, dict({k: args.baseline for k in DRUMS}, **{d: best_s}))
        print(f"{d:<10}{share:>7.1%}{best_s:>15}{best_f1:>8.3f}{was:>8.3f}"
              f"{m:>9.3f}{m - base:>+9.3f}")
    print("-" * 67)
    print("A class can only move MICRO in proportion to its share of the onsets, which is")
    print("why the cymbal advantage that started this question is worth less than it looks")
    print("and the snare is worth more.")

    print("""
Read the oracle as a ceiling, not a score. The class-to-model assignment was chosen by
reading the same corpus it is scored on, so it cannot be worse than either model by
construction and it flatters itself by exactly as much as the per-class noise allows.
It has no confidence interval here at all, because corpus totals cannot be resampled
over tracks. Two things would turn it into a finding:

  per-track counts for every system, so significance.py can bootstrap the routed MIDI
  the same way it bootstraps everything else; and a routing fixed on one corpus and
  scored on a different one, which is the only version of this that predicts anything.

Until both exist this is a reason to run an experiment, not a number to publish.

One more thing the arithmetic is blind to. Scoring treats the classes independently, so
the totals above are exact -- but a MIDI file is not scored, it is opened. Take the snare
from one model and the toms from another and nothing stops both from putting a note on
the same beat; our own tom work measured 53% of false toms landing within 50 ms of a real
snare. MICRO cannot see that, a drummer can. Any routed system needs looking at as well
as scoring.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
