"""Draws the ENST sample for the second ReStem comparison, before anything is rendered.

The rule was agreed in correspondence and is implemented here rather than described: a
stratified random sample of 60 musical recordings, seed 0, balanced across the 3 drummers
and the 4 recording kinds, excluding anything under 15 seconds because an open/closed
hi-hat ratio from a 10-second exercise is not a measurement.

Two properties matter more than the sampling itself.

It is drawn on the machine that will NOT do the rendering, and committed before the first
render starts. A sample chosen by the party watching the renders complete is not a
pre-declared sample, however honestly it is chosen.

And it is deterministic. Anyone can re-run this and get the same 60 names, so the claim
"these were fixed in advance" is checkable rather than asserted. If the file and this
script ever disagree, the script wins and the run is void.

The outcome measure is fixed here too, for the same reason. sample_power.py showed that a
sample of this size resolves hi-hat -- half-width 0.068 even under a cluster bootstrap over
the 3 drummers, against the +0.139 measured on MDB -- and that MICRO is unanswerable on
ENST at any sample size, its clustered width flattening near 0.083 against an effect of
+0.062. So this run reports hi-hat. A MICRO figure computed from it afterwards would be a
number the design cannot support.

    python preregister_enst_sample.py
    python preregister_enst_sample.py --out enst_sample_60.txt
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
KINDS = ["phrase", "solo", "minus-one", "MIDI-minus-one"]
MIN_SECONDS = 15.0


def inventory(data: Path):
    import soundfile as sf
    out = []
    for d in (1, 2, 3):
        adir = data / f"drummer_{d}" / "annotation"
        if not adir.exists():
            continue
        for ann in sorted(adir.glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in KINDS:
                continue
            wav = data / f"drummer_{d}" / "audio" / "wet_mix" / f"{ann.stem}.wav"
            if not wav.exists():
                continue
            out.append({"name": ann.stem, "drummer": d, "kind": parts[1],
                        "seconds": float(sf.info(str(wav)).duration)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "enst" / "enst_drums_public")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"no ENST at {args.data}")
        return 1

    all_recs = inventory(args.data)
    eligible = [r for r in all_recs if r["seconds"] >= MIN_SECONDS]
    print(f"{len(all_recs)} musical recordings; {len(eligible)} at or over "
          f"{MIN_SECONDS:.0f}s, {len(all_recs) - len(eligible)} excluded as too short")

    cells: dict[tuple[int, str], list] = {}
    for r in eligible:
        cells.setdefault((r["drummer"], r["kind"]), []).append(r)

    # Proportional allocation, largest-remainder, so the cells sum to exactly n without
    # the rounding quietly favouring whichever cell is iterated first.
    quotas = {k: args.n * len(v) / len(eligible) for k, v in cells.items()}
    take = {k: min(int(q), len(cells[k])) for k, q in quotas.items()}
    while sum(take.values()) < args.n:
        room = [k for k in cells if take[k] < len(cells[k])]
        if not room:
            break
        k = max(room, key=lambda k: (quotas[k] - take[k], -take[k], str(k)))
        take[k] += 1

    rng = random.Random(args.seed)
    picked = []
    for k in sorted(cells, key=lambda k: (k[0], KINDS.index(k[1]))):
        pool = sorted(cells[k], key=lambda r: r["name"])
        picked.extend(rng.sample(pool, take[k]))
    picked.sort(key=lambda r: (r["drummer"], KINDS.index(r["kind"]), r["name"]))

    print(f"\n{'drummer':>8}" + "".join(f"{k:>16}" for k in KINDS) + f"{'total':>8}")
    print("-" * 80)
    for d in (1, 2, 3):
        row = [sum(1 for r in picked if r["drummer"] == d and r["kind"] == k)
               for k in KINDS]
        avail = [len(cells.get((d, k), [])) for k in KINDS]
        print(f"{d:>8}" + "".join(f"{a:>11} of {b:<2}" for a, b in zip(row, avail))
              + f"{sum(row):>8}")
    print("-" * 80)
    mins = sum(r["seconds"] for r in picked) / 60
    print(f"{'total':>8}" + "".join(
        f"{sum(1 for r in picked if r['kind'] == k):>16}" for k in KINDS)
        + f"{len(picked):>8}")
    print(f"\n{len(picked)} recordings, {mins:.1f} minutes of audio, "
          f"median {sorted(r['seconds'] for r in picked)[len(picked)//2]:.1f}s")

    # The bracket: rendered first and last, compared byte for byte, as the only available
    # check that the quality setting did not move during the run. Pick the median-length
    # recording so it is representative and cheap rather than the longest.
    bracket = sorted(picked, key=lambda r: r["seconds"])[len(picked) // 2]
    print(f"bracket track, rendered first and last: {bracket['name']} "
          f"({bracket['seconds']:.1f}s)")

    body = "\n".join(f"{r['drummer']}\t{r['kind']}\t{r['seconds']:.2f}\t{r['name']}"
                     for r in picked)
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    print(f"sha256 of the selection: {digest}")

    if args.out:
        header = (
            f"# ENST sample for the second ReStem comparison. Drawn before any render.\n"
            f"# rule: stratified by drummer and kind, seed {args.seed}, "
            f"n={args.n}, minimum {MIN_SECONDS:.0f}s\n"
            f"# drawn by preregister_enst_sample.py on the machine that is NOT rendering\n"
            f"# outcome measure, fixed here: hi-hat F1, paired bootstrap over recordings\n"
            f"#   and over drummers as clusters. MICRO is NOT an outcome of this run:\n"
            f"#   sample_power.py shows its clustered width flattens near 0.083 against\n"
            f"#   an effect of +0.062, so ENST cannot answer it at any sample size.\n"
            f"# bracket: {bracket['name']}, rendered first and last, compared byte for byte\n"
            f"# sha256(selection body) = {digest}\n"
            f"# drummer\tkind\tseconds\tname\n")
        args.out.write_text(header + body + "\n", encoding="utf-8")
        print(f"written to {args.out}")

    print("""
Three outcomes, fixed before the first render and not to be renegotiated after it:

  the hi-hat advantage holds with an interval clear of zero -- the headline is a claim
  about the two systems rather than about MDB, and it says so;

  it collapses -- the headline is corpus-specific, and that goes at the top of the README
  rather than into a footnote;

  the interval is wide and contains zero -- reported as "we ran a second corpus and it
  could not resolve it", which is a result and gets published like one.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
