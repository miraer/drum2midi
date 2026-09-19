"""Do the hi-hats we call "open" actually ring longer than the ones we call "closed"?

An open hi-hat is two cymbals free to vibrate, so it decays slowly; a closed one is
clamped and stops almost immediately. That should be measurable on a hi-hat stem without
anyone listening.

It was measured once, on a single recording outside the public test set, and gave 1.51 --
open ringing half again as long as closed. That figure is **withdrawn**. Re-measured on
MusicDelta_Disco, MDB's most hi-hat-dense track at 756 hits, the same code gives **0.70**,
with open hits ringing *shorter*. One private song against one public one is not a
disagreement worth resolving by argument, so more tracks are being measured; until then
neither number should be quoted.

The mistake is worth naming because this project made it four times in one day: a figure
taken from one recording and never counted across the corpus.

    python check_hihat_decay.py ours.mid path/to/hh.wav
    python check_hihat_decay.py ours.mid path/to/hh.wav --highpass 1000

`--highpass` exists because of a specific hazard. ReStem's free trial mixes a 603 Hz
watermark tone into every stem it writes, including offline renders, and the envelope
here is broadband. A hi-hat's energy is around 12 kHz, so discarding everything below
1 kHz costs nothing and removes the watermark entirely. On Disco it moves the ratio from
0.70 to 0.66, so the watermark is not what produced the disagreement.

Note also that the default 22.05 kHz sample rate puts Nyquist at 11 kHz, just below where
hi-hat energy actually sits. That is fine for a decay envelope, which only needs relative
amplitude over time, but it is not the tool to reach for if you want spectral detail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

args = [a for a in sys.argv[1:]]
if {"-h", "--help"} & set(args) or len(args) < 2:
    print(__doc__)
    raise SystemExit(0 if args else 1)

highpass = 0.0
if "--highpass" in args:
    i = args.index("--highpass")
    highpass = float(args[i + 1])
    del args[i:i + 2]

mid_path, stem_path = Path(args[0]), Path(args[1])
SR = 22050

import librosa

y, _ = librosa.load(str(stem_path), sr=SR, mono=True)
if highpass > 0:
    # one-pole-per-stage Butterworth would need scipy; a windowed-sinc FIR is enough
    # here and keeps the dependency list as it is
    n = 257
    fc = highpass / (SR / 2)
    k = np.arange(n) - (n - 1) / 2
    lp = np.sinc(fc * k) * fc * np.hanning(n)
    lp /= lp.sum()
    hp = -lp
    hp[(n - 1) // 2] += 1.0
    y = np.convolve(y, hp, mode="same")
    print(f"high-passed above {highpass:.0f} Hz before measuring")
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
