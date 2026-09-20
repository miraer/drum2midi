"""How many kick onsets does ReStem emit on ENST, against how many are annotated?

Written because the first seven renders showed one to four kick notes where the reference
had eighteen to two hundred and thirty-four, and a claim that large needs more than seven
recordings behind it. This reports every ENST recording rendered so far, so the evidence
grows with the batch instead of waiting for someone to re-run an analysis.

The comparison is deliberately crude -- counts, not F-measure. A detector that finds the
right number of kicks in the wrong places is a different problem from one that finds
none, and at this stage the question is which of those is happening.

What was ruled out before treating the gap as real, recorded here so nobody has to trust
a chat log:

  our note mapping      the trigger editor reads NOTE=Auto and its tom selectors are
                        D2/B1/A1/F1, exactly the 41/45/47/50 the export carries
  a muted stem          all seven mute and solo toggles are off, and a stem with MIDI
                        OUT disabled emits zero rather than one
  missing from audio    40-120 Hz energy on these files is stronger than on MDB, by a
                        ratio of 9.6-12.3 against 4.0-5.0
  our side failing too  our own kick over all 210 ENST recordings is 0.902, against
                        0.960 on MDB -- the material does not defeat kick detection

    python enst_kick_audit.py
    python enst_kick_audit.py --theirs restem_export
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

KINDS = {"phrase", "solo", "minus-one", "MIDI-minus-one"}
KICK_PITCHES = {35, 36}


def main() -> int:
    import pretty_midi

    from benchmark_enst import IGNORED, LABEL_TO_CLASS

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "enst" / "enst_drums_public")
    ap.add_argument("--theirs", type=Path, default=ROOT / "restem_export")
    args = ap.parse_args()

    ref = {}
    for d in (1, 2, 3):
        adir = args.data / f"drummer_{d}" / "annotation"
        if not adir.exists():
            continue
        for ann in sorted(adir.glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in KINDS:
                continue
            n = 0
            total = 0
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 2 or p[1] in IGNORED:
                    continue
                cls = LABEL_TO_CLASS.get(p[1])
                if cls:
                    total += 1
                if cls == "KD":
                    n += 1
            ref[ann.stem] = (n, total, d)

    rows = []
    for folder in sorted(args.theirs.iterdir()):
        if not folder.is_dir() or folder.name.startswith("MusicDelta"):
            continue
        mid = folder / f"{folder.name}_midi.mid"
        base = folder.name.replace("DRY_", "")
        if not mid.exists() or base not in ref:
            continue
        pm = pretty_midi.PrettyMIDI(str(mid))
        notes = [n.pitch for inst in pm.instruments for n in inst.notes]
        kick = sum(1 for p in notes if p in KICK_PITCHES)
        other = sum(1 for p in notes if p == 60)
        rows.append((folder.name, ref[base][2], ref[base][0], kick, other, len(notes),
                     ref[base][1]))

    if not rows:
        print(f"nothing rendered yet under {args.theirs}")
        return 0

    print(f"{len(rows)} ENST recording(s) rendered so far\n")
    print(f"{'recording':<38}{'drm':>4}{'ref kick':>9}{'theirs':>8}"
          f"{'note 60':>9}{'emitted':>9}{'ref all':>9}")
    print("-" * 86)
    ref_k = their_k = 0
    for name, d, rk, tk, o, tot, rall in rows:
        print(f"{name[:37]:<38}{d:>4}{rk:>9}{tk:>8}{o:>9}{tot:>9}{rall:>9}")
        ref_k += rk
        their_k += tk
    print("-" * 86)
    print(f"{'total':<38}{'':>4}{ref_k:>9}{their_k:>8}")
    share = their_k / ref_k if ref_k else 0
    print(f"\nReStem emits {their_k} kick notes against {ref_k} annotated onsets: "
          f"{share:.1%} of them.")

    drummers = sorted({d for _, d, _, _, _, _, _ in rows})
    print(f"\nDrummers covered: {', '.join(str(d) for d in drummers)} "
          f"of 3.")
    if len(drummers) < 3:
        print("  That is the limit on this so far. ENST is three players with three kits,")
        print("  and a kick that one kit produces unusually would look exactly like this.")
        print("  The declared sample is stratified across all three, so the batch reaches")
        print("  them on its own -- until it does, this is a statement about "
              f"drummer {drummers[0]}, not about ENST.")
    else:
        print("  All three, so this is no longer a statement about one kit.")

    print("\nCounts only. A recall figure would need matching in time, which is the next")
    print("step if this survives the dry-mix control and a second machine's renders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
