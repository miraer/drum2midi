"""Prints the note map: MIDI number, note name, drum, and when it is emitted.

Note names are ambiguous by convention. The same MIDI number 60 is called C3 in Cubase,
Logic, Ableton and Reaper (middle C = C3), C4 in scientific pitch notation and Sibelius,
and C5 in FL Studio. Only the number is unambiguous, so all three are shown and the
number is what the pipeline actually writes.

The table is built from drum2midi's own constants, so it cannot drift out of date.

    python note_names.py
    python note_names.py --octave cubase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from drum2midi import (ADTOF_CYM, ADTOF_HAT, ADTOF_KICK, ADTOF_SNARE,  # noqa: E402
                       OUTPUT_NOTES, TOM_LADDER)

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Which octave number the DAW gives to MIDI note 60.
CONVENTIONS = {"cubase": 3, "scientific": 4, "fl": 5}

GM_NAME = {
    36: "Bass Drum 1", 38: "Acoustic Snare", 40: "Electric Snare",
    42: "Closed Hi-hat", 44: "Pedal Hi-hat", 46: "Open Hi-hat",
    43: "High Floor Tom", 47: "Low-Mid Tom", 50: "High Tom",
    49: "Crash Cymbal 1", 51: "Ride Cymbal 1",
}

# What the pipeline emits, and under what conditions.
ROWS = [
    (OUTPUT_NOTES.get(ADTOF_KICK, ADTOF_KICK), "Kick", "always"),
    (ADTOF_SNARE, "Snare", "always"),
    (ADTOF_HAT, "Hi-hat closed", "default hi-hat"),
    (46, "Hi-hat open", "--split-hats (default on) when the stem rings out"),
    (44, "Hi-hat pedal", "--pedal, learned or heuristic"),
    (TOM_LADDER[3][0], "Tom low", "toms, when 2 or 3 pitches are resolved"),
    (TOM_LADDER[1][0], "Tom mid", "toms, always the single-tom choice"),
    (TOM_LADDER[3][2], "Tom high", "toms, when 3 pitches are resolved"),
    (ADTOF_CYM, "Crash", "default cymbal"),
    (51, "Ride", "--separator uvr, when the ride stem rings louder"),
]


def note_name(number: int, middle_c_octave: int) -> str:
    octave = number // 12 - (5 - middle_c_octave)
    return f"{NAMES[number % 12]}{octave}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--octave", choices=sorted(CONVENTIONS), default=None,
                    help="show only one naming convention")
    args = ap.parse_args()

    shown = [args.octave] if args.octave else list(CONVENTIONS)
    heads = {"cubase": "Cubase/Logic", "scientific": "scientific", "fl": "FL Studio"}

    header = f"{'MIDI':>5}  " + "".join(f"{heads[c]:>14}" for c in shown)
    header += f"  {'drum':<16}{'General MIDI name':<20}when"
    print(f"All notes go to MIDI channel 10 (index 9), the percussion channel.\n")
    print(header)
    print("-" * len(header))
    for number, drum, when in ROWS:
        line = f"{number:>5}  "
        for c in shown:
            line += f"{note_name(number, CONVENTIONS[c]):>14}"
        line += f"  {drum:<16}{GM_NAME.get(number, '?'):<20}{when}"
        print(line)

    print("\nThe number is what is written to the file; the letter names differ only in\n"
          "how each DAW labels the same octave. Remap with --notes, e.g. --notes 36=35.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
