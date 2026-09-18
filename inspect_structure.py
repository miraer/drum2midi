"""MIDI structure: tracks, channels, message types.

    python inspect_structure.py a.mid [b.mid ...]
"""
import sys
from collections import Counter

import mido

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 2 or {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0 if len(sys.argv) > 1 else 1)

for path in sys.argv[1:]:
    m = mido.MidiFile(path)
    fmt = {0: "0 - everything in one track",
           1: "1 - parallel tracks, shared timeline",
           2: "2 - independent sequences"}.get(m.type, str(m.type))
    print(f"=== {path}")
    print(f"    format {fmt}")
    print(f"    tracks {len(m.tracks)}, division {m.ticks_per_beat}")
    for i, tr in enumerate(m.tracks):
        kinds = Counter(msg.type for msg in tr)
        chans = sorted({msg.channel for msg in tr if hasattr(msg, "channel")})
        meta = sum(1 for msg in tr if msg.is_meta)
        chan_txt = ", ".join(str(c) for c in chans) if chans else "none (meta only)"
        print(f"    track {i}: {len(tr):>5} events | channels: {chan_txt}")
        print(f"                meta events {meta}; types: "
              + ", ".join(f"{k}={v}" for k, v in kinds.most_common(5)))
    print()
