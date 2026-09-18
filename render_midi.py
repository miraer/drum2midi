"""Render a drum MIDI back to audio so the result can be judged by ear, not only by F1.

Renders with FluidSynth + the MuseScore General soundfont, and can place the original
recording and the rendered MIDI in opposite stereo channels. Any timing or
classification error then stands out immediately on headphones.

    python render_midi.py out.mid
    python render_midi.py out.mid --against input/drums.wav -o ab.wav
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SR = 44100
ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
SOUNDFONT = TOOLS / "MuseScore_General.sf3"


def find_fluidsynth() -> Path:
    """The bundled build first, then whatever is installed on PATH.

    Windows has no package manager by default, so setup_env.py downloads a build into
    tools/. On Linux and macOS FluidSynth comes from apt/brew and is already on PATH,
    which is checked here so those platforms need no bundled copy.
    """
    hits = list(TOOLS.glob("fluidsynth/**/bin/fluidsynth.exe"))
    if hits:
        return hits[0]
    hits = [p for p in TOOLS.glob("fluidsynth/**/fluidsynth*") if p.is_file()]
    if hits:
        return hits[0]
    on_path = shutil.which("fluidsynth")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        "FluidSynth not found. On Windows run setup_env.py --with-render, which "
        "downloads a build into tools/. On macOS: brew install fluid-synth. "
        "On Debian/Ubuntu: apt install fluidsynth.")


def render(midi: Path, out_wav: Path, gain: float = 0.8) -> Path:
    if not SOUNDFONT.exists():
        raise FileNotFoundError(f"Soundfont missing: {SOUNDFONT}")
    exe = find_fluidsynth()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [str(exe), "-ni", "-g", str(gain), "-F", str(out_wav), "-r", str(SR),
         str(SOUNDFONT), str(midi)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0 or not out_wav.exists():
        raise RuntimeError((res.stderr or res.stdout or "fluidsynth failed")[-300:])
    return out_wav


def _mono(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), always_2d=True)
    y = x.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def normalise(x: np.ndarray, target: float = 0.7) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (target / peak) if peak > 1e-9 else x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--against", type=Path, default=None,
                    help="Original audio; produces a stereo A/B (original left, MIDI right)")
    ap.add_argument("--gain", type=float, default=0.8)
    args = ap.parse_args()

    midi = args.midi.resolve()
    if not midi.exists():
        print(f"not found: {midi}")
        return 1
    out = (args.out or midi.with_suffix(".wav")).resolve()

    if args.against is None:
        render(midi, out, args.gain)
        print(f"rendered -> {out}")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="render_"))
    try:
        rendered = _mono(render(midi, tmp / "midi.wav", args.gain))
        original = _mono(args.against.resolve())
        n = max(len(rendered), len(original))
        left = np.zeros(n); right = np.zeros(n)
        left[:len(original)] = normalise(original)
        right[:len(rendered)] = normalise(rendered)
        sf.write(str(out), np.stack([left, right], axis=1), SR)
        print(f"A/B written -> {out}")
        print("  left  = original recording")
        print("  right = transcribed MIDI")
        print(f"  {n / SR:.1f}s; listen on headphones - drift or wrong drums will be obvious")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
