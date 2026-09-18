"""What a MIDI file actually contains: header, tempo event, delta times.

    python inspect_midi.py [file.mid]      # defaults to out/final.mid
"""
import sys

import mido

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

path = sys.argv[1] if len(sys.argv) > 1 else "out/final.mid"
m = mido.MidiFile(path)

print(f"file: {path}")
print("HEADER (MThd chunk):")
print(f"   format   = {m.type}")
print(f"   tracks  = {len(m.tracks)}")
print(f"   division = {m.ticks_per_beat} ticks per quarter   <- this only, no tempo here")
print()

for i, tr in enumerate(m.tracks):
    print(f"TRACK {i} ({tr.name or 'unnamed'}), events: {len(tr)}")
    acc = 0
    shown = 0
    for msg in tr:
        acc += msg.time
        interesting = msg.type in ("set_tempo", "time_signature", "note_on")
        if interesting and shown < 8:
            if msg.type == "set_tempo":
                extra = f"{round(60_000_000 / msg.tempo, 1)} BPM"
            elif msg.type == "note_on":
                extra = f"note {msg.note}, vel {msg.velocity}"
            else:
                extra = f"{msg.numerator}/{msg.denominator}"
            print(f"   delta {msg.time:>5} ticks -> absolute {acc:>6} | "
                  f"{msg.type:<14} {extra}")
            shown += 1
    print()

print("Summary:")
print("  * PPQ lives in the file header")
print("  * tempo lives INSIDE a track as a set_tempo meta event at its own position,")
print("    so there can be several of them - that is a tempo map")
print("  * every event stores a DELTA from the previous one, not an absolute position;")
print("    the absolute position comes from accumulating them")
