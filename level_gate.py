"""Would a level gate like ReStem's change anything about what we emit?

ReStem gates each stem at a level -- Thresh, default −60.0 dBFS. We have no such control:
emission is decided entirely by the peak-picker's confidence threshold on ADTOF's output,
and `velocities()` clamps a quiet hit to the lowest velocity rather than dropping it. So
the two products differ architecturally and not only in the value of a number.

That difference is either interesting or irrelevant, and which one is measurable: take
every note we emit and ask how loud the audio actually is at that instant. If essentially
none of them fall below −60 dBFS, their gate would never have fired on our output either,
the architectural difference costs nothing on this material, and the comparison is
unaffected.

Reported per class, because a gate that never touches kicks may still touch ghost snares.

    python level_gate.py
    python level_gate.py --ours bench/ceiling --gate -60
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=ROOT / "bench" / "ceiling")
    ap.add_argument("--gate", type=float, default=-60.0,
                    help="dBFS, the level ReStem's Thresh defaults to")
    ap.add_argument("--window", type=float, default=0.02,
                    help="seconds after the onset to take the peak from")
    args = ap.parse_args()

    import numpy as np
    import pretty_midi
    import soundfile as sf

    from compare_with_restem import AUDIO, FAMILY_PITCHES, PRETTY

    if not AUDIO.exists():
        print(f"MDB Drums not found at {AUDIO}")
        return 1
    print(f"notes from {args.ours}, levels from the drum recording itself")
    print(f"gate at {args.gate:+.1f} dBFS "
          f"({10 ** (args.gate / 20):.6f} of full scale)\n")

    per_class: dict[str, list[float]] = {}
    for wav in sorted(AUDIO.glob("*.wav")):
        mid = args.ours / f"{wav.stem}.mid"
        if not mid.exists():
            continue
        audio, sr = sf.read(str(wav), always_2d=True, dtype="float32")
        sig = np.abs(audio.mean(axis=1))
        notes = [n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                 for n in inst.notes]
        for n in notes:
            a = int(n.start * sr)
            b = min(len(sig), a + int(args.window * sr))
            if b <= a:
                continue
            peak = float(sig[a:b].max())
            db = 20.0 * np.log10(max(peak, 1e-12))
            fam = next((f for f, ps in FAMILY_PITCHES.items() if n.pitch in ps), None)
            if fam:
                per_class.setdefault(fam, []).append(db)

    if not per_class:
        print("nothing scored")
        return 1

    print(f"{'class':<10}{'notes':>7}{'quietest':>11}{'median':>9}"
          f"{'below gate':>12}")
    print("-" * 49)
    total = below = 0
    for fam, vals in per_class.items():
        v = np.array(vals)
        n_below = int((v < args.gate).sum())
        total += len(v)
        below += n_below
        print(f"{PRETTY.get(fam, fam):<10}{len(v):>7}{v.min():>11.1f}"
              f"{float(np.median(v)):>9.1f}{n_below:>12}")
    print("-" * 49)
    print(f"{'all':<10}{total:>7}{'':>11}{'':>9}{below:>12}"
          f"   ({below / max(total, 1):.3%})")

    print()
    if below == 0:
        print(f"No note we emit is quieter than {args.gate:+.0f} dBFS, so a gate at that")
        print("level would not have removed a single one. The architectural difference")
        print("does not touch this comparison.")
    else:
        print(f"{below} note(s) would be removed by a gate at {args.gate:+.0f} dBFS.")
        print("Whether that helps or hurts depends on how many of them are correct,")
        print("which this script does not decide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
