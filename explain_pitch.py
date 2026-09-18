"""What does an unknown pitch in another transcriber's file correspond to?

Products disagree about General MIDI numbering, and some use notes outside the drum map
entirely. Rather than guess, this looks at what *our* transcription says was happening at
the same moments: for every note of the chosen pitch in file B, find the nearest note in
file A and report which instrument it was.

    python explain_pitch.py ours.mid theirs.mid 60
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pretty_midi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 4 or {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0 if len(sys.argv) > 1 else 1)

NAMES = {35: "kick", 36: "kick", 38: "snare", 40: "snare", 37: "side stick",
         42: "hi-hat closed", 44: "hi-hat pedal", 46: "hi-hat open",
         41: "low tom", 43: "low tom", 45: "mid tom", 47: "mid tom",
         48: "high tom", 50: "high tom", 49: "crash", 57: "crash",
         51: "ride", 59: "ride", 53: "ride bell", 55: "splash", 60: "pitch 60"}

a_path, b_path, pitch = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
window = float(sys.argv[4]) / 1000 if len(sys.argv) > 4 else 0.05


def notes(path: Path):
    m = pretty_midi.PrettyMIDI(str(path))
    return sorted(((n.start, n.pitch, n.velocity)
                   for inst in m.instruments for n in inst.notes))


a, b = notes(a_path), notes(b_path)
targets = [t for t, p, _ in b if p == pitch]
print(f"{b_path.name}: {len(targets)} notes at pitch {pitch}")
print(f"matching against {a_path.name} within {window*1000:.0f} ms\n")

hits = Counter()
vels = []
for t in targets:
    near = [(abs(s - t), p, v) for s, p, v in a if abs(s - t) <= window]
    if not near:
        hits["(nothing within the window)"] += 1
        continue
    near.sort()
    _, p, v = near[0]
    hits[NAMES.get(p, f"note {p}")] += 1
    vels.append(v)

print(f"{'what we detected there':<28}{'count':>7}{'share':>9}")
print("-" * 44)
for name, n in hits.most_common():
    print(f"{name:<28}{n:>7}{n/len(targets):>9.1%}")

their_vels = [v for _, p, v in b if p == pitch]
if their_vels:
    print(f"\ntheir velocity at pitch {pitch}: "
          f"min {min(their_vels)}, max {max(their_vels)}, "
          f"mean {sum(their_vels)/len(their_vels):.0f}")
