"""Does Bleed Reduction help ReStem on ENST, or only on MDB?

The published comparison uses Bleed Reduction on MDB, where it is worth +0.014 overall,
and the letter to ReStem called that their strongest configuration. That was an MDB
figure stated as though it held everywhere, which is exactly the error this project keeps
catching in itself: a number measured on one population restated for another.

It also matters in the other direction. `restem_kick_misroute.py` shows the option filing
one drummer's kick under the "other" stem, so if Bleed Reduction were a clear win overall
that finding would sit awkwardly beside a recommendation to use it.

No new renders are needed. 41 of the declared 60 ENST recordings already exist in both
arms, across all three drummers, which is enough for a paired contrast: same audio, same
annotations, one checkbox between them.

Drummer 1 is excluded by default, and the reason is not squeamishness. That kit meets the
defect in `restem_kick_misroute.py`: with the option on its kick very nearly disappears,
and it carries 34% of the corpus's annotated kick onsets. Including it would measure how
much of the corpus meets a defect already described elsewhere, not what Bleed Reduction
does to separation, and would report the option as catastrophic for a reason that has
nothing to do with what it is for. Scoped to drummers 2 and 3 the question is answerable
either way. Pass `--drummers 1,2,3` to see the confounded version.

    python restem_bleed_enst.py
    python restem_bleed_enst.py --rounds 4000
    python restem_bleed_enst.py --drummers 1,2,3      # the confounded comparison

Both intervals are reported, over recordings and over drummers, because a per-kit effect
is precisely what is suspected here and resampling recordings alone would hide it.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from score_enst_restem import (RESTEM_TO_CLASS, counts, enst_reference, f1,
                               interval, notes_by_class)

ROOT = Path(__file__).resolve().parent


def arms(data: Path, on_dir: Path, off_dir: Path, order, window, keep):
    """Score both arms on the recordings that exist in each."""
    refs = enst_reference(data)
    on, off, drummers = {}, {}, {}
    for folder in sorted(on_dir.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if name not in refs or refs[name]["drummer"] not in keep:
            continue
        on_mid = folder / f"{name}_midi.mid"
        off_mid = off_dir / name / f"{name}_midi.mid"
        if not on_mid.exists() or not off_mid.exists():
            continue
        ref = refs[name]["ref"]
        on[name] = counts(ref, notes_by_class(on_mid, RESTEM_TO_CLASS), order, window)
        off[name] = counts(ref, notes_by_class(off_mid, RESTEM_TO_CLASS), order, window)
        drummers[name] = refs[name]["drummer"]
    return on, off, drummers


def report(names, on, off, drummers, order, pretty, rounds, seed):
    by_drummer = {}
    for n in names:
        by_drummer.setdefault(drummers[n], []).append(n)
    onsets = sum(on[n][k]["ref"] for n in names for k in order)
    dr = sorted(by_drummer)
    # Fewer than three clusters cannot be resampled into anything: with two kits the
    # draw has four outcomes resting on three distinct values, so the bracket would
    # carry the typography of evidence without any of the content. An earlier version
    # printed it anyway and then explained in prose that it meant nothing, which let a
    # reader keep whichever half suited them.
    clustered = len(dr) >= 3
    print(f"\n{len(names)} recordings in both arms, "
          f"{len(dr)} drummer(s), {onsets} onsets")
    print("positive means Bleed Reduction helps them")
    if not clustered:
        print(f"no cluster-level interval: {len(dr)} kits cannot be resampled")
    print()
    head = f"{'class':<9}{'bleed on':>10}{'bleed off':>11}{'diff':>8}{'95% over recordings':>24}"
    print(head + (f"{'over drummers':>22}" if clustered else ""))
    print("-" * (len(head) + (22 if clustered else 0)))

    rng = random.Random(seed)
    micro_half = None
    for cls in [None] + list(order):
        a, b = f1(on, names, order, cls), f1(off, names, order, cls)
        flat, clus = [], []
        for _ in range(rounds):
            s = [rng.choice(names) for _ in names]
            flat.append(f1(on, s, order, cls) - f1(off, s, order, cls))
            if clustered:
                pool = []
                for _ in range(len(dr)):
                    pool.extend(by_drummer[rng.choice(dr)])
                s2 = [rng.choice(pool) for _ in names]
                clus.append(f1(on, s2, order, cls) - f1(off, s2, order, cls))
        lo, hi = interval(flat)
        if cls is None:
            micro_half = max(abs(lo), abs(hi))
        label = "MICRO" if cls is None else pretty.get(cls, cls)
        # A star is withheld when a bound rounds to zero at the precision printed: a
        # bracket that reads [+0.000, ...] beside a mark claiming it excludes zero
        # asks the reader to believe the mark over the number.
        visible = min(abs(lo), abs(hi)) >= 0.0005
        mark = " *" if (lo > 0 or hi < 0) and visible else "  "
        line = (f"{label:<9}{a:>10.3f}{b:>11.3f}{a - b:>+8.3f}"
                f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{mark}")
        if clustered:
            clo, chi = interval(clus)
            line += (f"{f'[{clo:+.3f}, {chi:+.3f}]':>20}"
                     + (" *" if clo > 0 or chi < 0 else "  "))
        print(line)
        if cls is not None and micro_half and 0 < abs(a - b) < micro_half:
            print(f"{'':<9}{'':>10}{'':>11}{'':>8}"
                  f"{'smaller than the MICRO interval above it':>44}")

    print("\nper drummer, MICRO only -- the aggregate hides a per-kit failure and this "
          "audit exists because one was found:")
    print(f"{'drummer':<9}{'recs':>6}{'bleed on':>10}{'bleed off':>11}{'diff':>8}")
    print("-" * 44)
    for d in dr:
        ns = by_drummer[d]
        a, b = f1(on, ns, order, None), f1(off, ns, order, None)
        print(f"{d:<9}{len(ns):>6}{a:>10.3f}{b:>11.3f}{a - b:>+8.3f}")


def main() -> int:
    from benchmark_mdb import ORDER, PRETTY, WINDOW

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "enst" / "enst_drums_public")
    ap.add_argument("--on", type=Path, default=ROOT / "bench" / "enst_bestplus_renders")
    ap.add_argument("--off", type=Path, default=ROOT / "restem_export")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drummers", default="2,3",
                    help="which kits to include; drummer 1 is held out because the "
                         "kick defect confounds the mode there")
    args = ap.parse_args()

    keep = {int(d) for d in args.drummers.split(",") if d.strip()}
    on, off, drummers = arms(args.data, args.on, args.off, ORDER, WINDOW, keep)
    names = sorted(on)
    if not names:
        print("no recording exists in both arms; nothing to compare")
        return 1
    if keep != {2, 3}:
        print(f"including drummers {sorted(keep)} -- note that drummer 1 meets the kick "
              "defect and its presence measures that rather than the mode")
    report(names, on, off, drummers, ORDER, PRETTY, args.rounds, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
