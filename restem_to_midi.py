"""Converts ReStem's trigger_events.json into MIDI so it can be scored by the same code.

TrigNet format (trignet-events-v1):
  sr, hop  44100 and 441, i.e. the same 100 frames per second ADTOF uses
  stems    kick, snare, hh, toms, ride, crash; each event carries
             sample  hit position in samples
             prob    detector confidence
             vel     loudness 0..1
             cc4     hi-hat pedal position (hh only)
             tomClass, fundHz  tom class and fundamental pitch (toms only)

    python restem_to_midi.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "restem_events"
DST = ROOT / "restem_midi"

# the notes ReStem actually writes to MIDI (verified against its export)
BASE_NOTE = {"kick": 36, "snare": 38, "hh": 42, "toms": 47, "ride": 51,
             "crash": 49, "other": 60}
# tomClass 0..3 -> Floor, Low, Mid, High
TOM_NOTE = {0: 41, 1: 45, 2: 47, 3: 50}
# cc4 thresholds recovered from ReStem's own MIDI export:
# 374 JSON events against 374 MIDI notes, the boundaries came out clean:
#   closed  cc4 < 0.46      pedal  0.46..0.76      open  cc4 >= 0.76
PEDAL_CC4 = 0.46
OPEN_CC4 = 0.76


def convert(path: Path, out: Path, note_len: float = 0.06,
            open_cc4: float = OPEN_CC4, pedal_cc4: float = PEDAL_CC4) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sr = float(data.get("sr", 44100))
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=960)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="ReStem")
    stats = {}

    for stem, events in data.get("stems", {}).items():
        stats[stem] = len(events)
        for e in events:
            t = float(e["sample"]) / sr
            vel = int(max(1, min(127, round(float(e.get("vel", 0.8)) * 127))))
            if stem == "toms":
                pitch = TOM_NOTE.get(int(e.get("tomClass", 2)), 47)
            elif stem == "hh":
                cc4 = float(e.get("cc4", 0.0))
                pitch = 46 if cc4 >= open_cc4 else (44 if cc4 >= pedal_cc4 else 42)
            else:
                pitch = BASE_NOTE.get(stem, 38)
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                               start=t, end=t + note_len))
    inst.notes.sort(key=lambda n: n.start)
    midi.instruments.append(inst)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    midi.write(str(out))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--open-cc4", type=float, default=OPEN_CC4)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(src.glob("*.json"))
    if not files:
        print(f"no json in {src}")
        return 1

    total = {}
    for f in files:
        stats = convert(f, dst / f"{f.stem}.mid", open_cc4=args.open_cc4)
        for k, v in stats.items():
            total[k] = total.get(k, 0) + v
        print(f"  {f.stem:<34} " + " ".join(f"{k}={v}" for k, v in sorted(stats.items())))

    print(f"\n{len(files)} tracks total -> {dst}")
    print("events per stem:", dict(sorted(total.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
