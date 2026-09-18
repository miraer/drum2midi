"""Do the hi-hats we call "open" actually ring longer than the ones we call "closed"?

Two transcribers disagreed about this song: ours reports 18 open hi-hats and 6 pedal
ones, ReStem reports none at all (every one of its 188 CC4 values says "closed").
Only one of those can be right, and it is decidable without listening: an open hi-hat
is two cymbals free to vibrate, so it decays slowly, while a closed one is clamped and
stops almost immediately.

This measures decay time on the hi-hat stem at the moments each transcription marks,
and compares the groups.

    python check_hihat_decay.py ours.mid path/to/hh.wav
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

mid_path, stem_path = Path(sys.argv[1]), Path(sys.argv[2])
SR = 22050

import librosa

y, _ = librosa.load(str(stem_path), sr=SR, mono=True)
notes = sorted(((n.start, n.pitch) for inst in pretty_midi.PrettyMIDI(str(mid_path)).instruments
                for n in inst.notes))

LABEL = {42: "closed", 44: "pedal", 46: "open"}


def decay_time(at: float, floor_db: float = -20.0, limit: float = 0.6) -> float | None:
    """Seconds until the hit falls `floor_db` below its own peak."""
    a = int(at * SR)
    b = min(a + int(limit * SR), len(y))
    if b - a < int(0.05 * SR):
        return None
    seg = np.abs(y[a:b])
    if seg.max() < 1e-4:
        return None
    # envelope, smoothed enough to ignore single-sample spikes
    win = int(0.005 * SR)
    env = np.convolve(seg, np.ones(win) / win, mode="same")
    peak = env.max()
    target = peak * (10 ** (floor_db / 20))
    below = np.where(env < target)[0]
    below = below[below > np.argmax(env)]
    return float(below[0] / SR) if len(below) else limit


groups: dict[str, list[float]] = {}
for start, pitch in notes:
    name = LABEL.get(pitch)
    if not name:
        continue
    d = decay_time(start)
    if d is not None:
        groups.setdefault(name, []).append(d)

print(f"hi-hat stem: {stem_path.name}")
print(f"decay measured as time to fall 20 dB below the hit's own peak\n")
print(f"{'articulation':<12}{'hits':>6}{'median':>10}{'mean':>9}{'p90':>9}")
print("-" * 46)
for name in ("closed", "pedal", "open"):
    v = groups.get(name)
    if not v:
        print(f"{name:<12}{0:>6}{'-':>10}{'-':>9}{'-':>9}")
        continue
    print(f"{name:<12}{len(v):>6}{np.median(v)*1000:>9.0f}m"
          f"{np.mean(v)*1000:>8.0f}m{np.percentile(v,90)*1000:>8.0f}m")

if groups.get("open") and groups.get("closed"):
    o, c = np.array(groups["open"]), np.array(groups["closed"])
    ratio = np.median(o) / np.median(c)
    print(f"\nopen hits ring {ratio:.2f}x as long as closed ones")
    # Mann-Whitney style check without scipy: how often does a random open hit
    # outlast a random closed one? 0.5 means the labels carry no information.
    wins = sum(1 for x in o for z in c if x > z) / (len(o) * len(c))
    print(f"a random 'open' hit outlasts a random 'closed' one {wins:.0%} of the time")
    print("(50% would mean the labels are meaningless)")
