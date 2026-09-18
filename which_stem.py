"""Which stem does a given MIDI pitch come from?

ReStem exports one stem per drum plus an "other" stem, and writes notes at pitch 60,
which General MIDI calls Hi Bongo. Rather than guessing what that pitch means, this
measures it: for every note of the chosen pitch, compare the energy in each stem at that
instant against the energy at random instants. The stem that is loudest exactly when the
note fires is the one the note describes.

    python which_stem.py export_dir 60
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 3 or {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0 if len(sys.argv) > 1 else 1)

export = Path(sys.argv[1])
pitch = int(sys.argv[2])
SR = 22050
WINDOW = 0.04

mid = next(export.glob("*_midi.mid"), None)
if mid is None:
    print(f"no *_midi.mid in {export}")
    raise SystemExit(1)

times = sorted(n.start for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
               for n in inst.notes if n.pitch == pitch)
if not times:
    print(f"no notes at pitch {pitch}")
    raise SystemExit(1)

print(f"{len(times)} notes at pitch {pitch}\n")

import librosa

rng = np.random.default_rng(0)
rows = []
for wav in sorted(export.glob("*.wav")):
    name = wav.stem.split("_")[-1]
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    dur = len(y) / SR

    def energy(at: float) -> float:
        a = int(max(at, 0) * SR)
        b = min(a + int(WINDOW * SR), len(y))
        return float(np.sqrt(np.mean(y[a:b] ** 2))) if b > a else 0.0

    hit = np.array([energy(t) for t in times if t < dur])
    # the same number of random instants, as a baseline for "how loud is this stem
    # in general" -- a busy stem would otherwise look like a match for everything
    base = np.array([energy(t) for t in rng.uniform(0, max(dur - WINDOW, 0.1), len(hit))])
    ratio = hit.mean() / base.mean() if base.mean() > 0 else float("inf")
    rows.append((name, hit.mean(), base.mean(), ratio))

rows.sort(key=lambda r: -r[3])
print(f"{'stem':<10}{'at the notes':>14}{'at random':>12}{'ratio':>9}")
print("-" * 46)
for name, h, b, r in rows:
    print(f"{name:<10}{h:>14.5f}{b:>12.5f}{r:>9.2f}")

best = rows[0]
print(f"\npitch {pitch} tracks the '{best[0]}' stem "
      f"({best[3]:.1f}x louder at the notes than at random moments)")
