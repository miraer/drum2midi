"""Which General MIDI drum notes does a sampler actually produce sound for?

The worry behind "every DAW uses different notes" is that a transcription will be silent
or wrong in someone else's instrument. That is testable: write one note per pitch, render
through a General MIDI soundfont, and measure the energy each one produced.

A pitch that renders loudly is mapped in that instrument. A pitch that renders silence is
not, and would be lost on import.

    python probe_soundfont.py
    python probe_soundfont.py --low 35 --high 59
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GM = {35: "Acoustic Bass Drum", 36: "Bass Drum 1", 37: "Side Stick",
      38: "Acoustic Snare", 39: "Hand Clap", 40: "Electric Snare",
      41: "Low Floor Tom", 42: "Closed Hi-hat", 43: "High Floor Tom",
      44: "Pedal Hi-hat", 45: "Low Tom", 46: "Open Hi-hat", 47: "Low-Mid Tom",
      48: "Hi-Mid Tom", 49: "Crash Cymbal 1", 50: "High Tom",
      51: "Ride Cymbal 1", 52: "Chinese Cymbal", 53: "Ride Bell",
      54: "Tambourine", 55: "Splash Cymbal", 56: "Cowbell",
      57: "Crash Cymbal 2", 58: "Vibraslap", 59: "Ride Cymbal 2",
      60: "Hi Bongo", 61: "Low Bongo"}


def find_tools() -> tuple[Path | None, Path | None]:
    fs = next((p for p in (ROOT / "tools").rglob("fluidsynth.exe")), None) \
        if (ROOT / "tools").exists() else None
    sound = ROOT / "tools" / "MuseScore_General.sf3"
    return fs, (sound if sound.exists() else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=int, default=35)
    ap.add_argument("--high", type=int, default=61)
    args = ap.parse_args()

    fs, sound = find_tools()
    if not fs or not sound:
        print("FluidSynth or the soundfont is missing; run "
              "setup_env.py --with-render")
        return 1

    pitches = list(range(args.low, args.high + 1))
    spacing = 1.0
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=960)
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="probe")
    for i, p in enumerate(pitches):
        kit.notes.append(pretty_midi.Note(velocity=110, pitch=p,
                                          start=i * spacing, end=i * spacing + 0.4))
    midi.instruments.append(kit)

    tmp = Path(tempfile.mkdtemp(prefix="sfprobe_"))
    mid_path, wav_path = tmp / "probe.mid", tmp / "probe.wav"
    midi.write(str(mid_path))

    res = subprocess.run([str(fs), "-ni", "-F", str(wav_path), "-r", "44100",
                          str(sound), str(mid_path)],
                         capture_output=True, text=True)
    if not wav_path.exists():
        print(f"render failed: {(res.stderr or res.stdout)[-200:]}")
        return 1

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"soundfont: {sound.name}\n")
    print(f"{'MIDI':>5}  {'General MIDI name':<22}{'peak':>9}{'':>4}")
    print("-" * 44)
    silent = []
    for i, p in enumerate(pitches):
        a, b = int(i * spacing * sr), int((i * spacing + 0.9) * sr)
        seg = audio[a:min(b, len(audio))]
        peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
        mark = "" if peak > 0.01 else "  <- silent"
        if peak <= 0.01:
            silent.append(p)
        print(f"{p:>5}  {GM.get(p,'?'):<22}{peak:>9.4f}{mark}")

    print(f"\n{len(pitches)-len(silent)} of {len(pitches)} pitches produce sound.")
    if silent:
        print(f"silent: {silent}")
    else:
        print("Every General MIDI percussion note in this range is mapped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
