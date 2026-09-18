"""Pitch histogram of one or more MIDI files, side by side.

Two transcribers can agree completely about what was played and still share no notes,
because General MIDI offers several numbers for the same drum (35/36 for kick, 38/40 for
snare) and products pick differently. Comparing before checking this produces nonsense.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pretty_midi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 2 or {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    print("    python midi_pitches.py a.mid [b.mid ...]")
    raise SystemExit(0 if len(sys.argv) > 1 else 1)

GM = {35: "Acoustic Bass Drum", 36: "Bass Drum 1", 37: "Side Stick",
      38: "Acoustic Snare", 39: "Hand Clap", 40: "Electric Snare",
      41: "Low Floor Tom", 42: "Closed Hi-hat", 43: "High Floor Tom",
      44: "Pedal Hi-hat", 45: "Low Tom", 46: "Open Hi-hat", 47: "Low-Mid Tom",
      48: "Hi-Mid Tom", 49: "Crash Cymbal 1", 50: "High Tom",
      51: "Ride Cymbal 1", 52: "Chinese Cymbal", 53: "Ride Bell",
      54: "Tambourine", 55: "Splash Cymbal", 56: "Cowbell",
      57: "Crash Cymbal 2", 59: "Ride Cymbal 2", 60: "Hi Bongo"}

files = [Path(p) for p in sys.argv[1:]]
counts = []
for path in files:
    m = pretty_midi.PrettyMIDI(str(path))
    counts.append(Counter(n.pitch for inst in m.instruments for n in inst.notes))

for i, path in enumerate(files):
    print(f"[{i}] {path.name}  ({sum(counts[i].values())} notes)")
print()

header = f"{'pitch':>6}  {'General MIDI name':<22}" + "".join(f"{f'[{i}]':>8}" for i in range(len(files)))
print(header)
print("-" * len(header))
for pitch in sorted(set().union(*counts)):
    row = f"{pitch:>6}  {GM.get(pitch, '?'):<22}"
    for c in counts:
        row += f"{c.get(pitch, 0):>8}"
    print(row)
