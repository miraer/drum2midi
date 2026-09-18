"""Which General MIDI notes do real drum modules actually use?

General MIDI allows several numbers for the same drum -- 35 and 36 both mean bass drum,
38 and 40 both mean snare -- so a transcriber has to choose, and a DAW or sampler has to
recognise the choice. Opinions differ; this counts what real hardware emits.

The Groove MIDI Dataset is recorded from a Roland TD-11 module, so its note numbers are
what that instrument sends and what drum plugins are built to receive.

    python gm_note_survey.py
    python gm_note_survey.py --limit 200
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GMD = ROOT / "gmd" / "groove"

GM = {35: "Acoustic Bass Drum", 36: "Bass Drum 1", 37: "Side Stick",
      38: "Acoustic Snare", 39: "Hand Clap", 40: "Electric Snare",
      41: "Low Floor Tom", 42: "Closed Hi-hat", 43: "High Floor Tom",
      44: "Pedal Hi-hat", 45: "Low Tom", 46: "Open Hi-hat", 47: "Low-Mid Tom",
      48: "Hi-Mid Tom", 49: "Crash Cymbal 1", 50: "High Tom",
      51: "Ride Cymbal 1", 52: "Chinese Cymbal", 53: "Ride Bell",
      55: "Splash Cymbal", 57: "Crash Cymbal 2", 58: "Vibraslap",
      59: "Ride Cymbal 2", 60: "Hi Bongo"}

FAMILY = {
    "kick": (35, 36),
    "snare": (38, 40),
    "hi-hat": (42, 44, 46),
    "toms": (41, 43, 45, 47, 48, 50),
    "crash": (49, 52, 55, 57),
    "ride": (51, 53, 59),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dir", default=str(GMD))
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"not found: {root}\nSee the Install section of README.md.")
        return 1

    files = sorted(root.rglob("*.mid"))[: args.limit]
    if not files:
        print(f"no MIDI files under {root}")
        return 1

    counts = Counter()
    used_in = Counter()
    for path in files:
        try:
            m = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            continue
        here = Counter(n.pitch for inst in m.instruments for n in inst.notes)
        counts.update(here)
        used_in.update(here.keys())

    total = sum(counts.values())
    print(f"files: {len(files)}   notes: {total}\n")
    print(f"{'pitch':>6}  {'General MIDI name':<22}{'notes':>9}{'share':>8}{'files':>8}")
    print("-" * 55)
    for pitch, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{pitch:>6}  {GM.get(pitch,'?'):<22}{n:>9}{n/total:>8.1%}"
              f"{used_in[pitch]:>8}")

    print(f"\nwithin each family, which number wins:")
    for name, pitches in FAMILY.items():
        present = [(p, counts.get(p, 0)) for p in pitches if counts.get(p, 0)]
        if not present:
            continue
        present.sort(key=lambda kv: -kv[1])
        fam_total = sum(n for _, n in present)
        detail = ", ".join(f"{p} ({n/fam_total:.0%})" for p, n in present)
        print(f"  {name:<8} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
