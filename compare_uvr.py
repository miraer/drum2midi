"""Measure stem quality of LarsNet vs the UVR MDX23C 6-stem drum separator.

Contrast = median stem level at known hits / median level at random times.
Run separate_uvr.py (or the audio-separator CLI) first to produce the UVR stems.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SR = 44100
ROOT = Path(__file__).resolve().parent
UVR_SUFFIX = "_MDX23C-DrumSep-aufr33-jarredou.flac"

# Ground truth of input/synth.wav, produced by evaluate.py
BEAT, LEAD, BARS = 0.5, 0.25, 8
TOMS = [15.25 + i * 0.165 for i in range(6)]
HATS = [LEAD + b * 2.0 + i * 0.5 * BEAT for b in range(BARS - 1) for i in range(8)]
CRASH = [LEAD, LEAD + 4 * 2.0]

rng = np.random.default_rng(3)


def mono(path: Path) -> np.ndarray:
    x, _ = sf.read(str(path))
    return x.mean(axis=1) if x.ndim > 1 else x


def peak(sig: np.ndarray, t: float, w: float = 0.08) -> float:
    a, b = max(0, int(t * SR)), min(len(sig), int((t + w) * SR))
    return float(np.max(np.abs(sig[a:b]))) if b > a else 0.0


def contrast(sig: np.ndarray, times, lo=0.0, hi=14.0):
    hit = float(np.median([peak(sig, t) for t in times]))
    floor = float(np.median([peak(sig, float(t)) for t in rng.uniform(lo, hi, 300)]))
    return hit, floor, hit / max(floor, 1e-9)


def row(drum: str, source: str, path: Path, times) -> None:
    if not path.exists():
        print(f"{drum:<9}{source:<10}{'-- missing --':>30}")
        return
    hit, floor, c = contrast(mono(path), times)
    print(f"{drum:<9}{source:<10}{hit:>10.4f}{floor:>10.4f}{c:>11.1f}x")


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    lars = ROOT / "cmp" / "lars_synth"
    uvr = ROOT / "uvr" / "synth"
    if not lars.exists():
        print(f"LarsNet stems missing; run:\n"
              f"  python drum2midi.py input/synth.wav -o out/x.mid --keep-stems cmp/lars_synth")
        return 1

    print(f"{'drum':<9}{'source':<10}{'level':>10}{'floor':>10}{'contrast':>12}")
    print("-" * 51)
    row("toms", "LarsNet", lars / "toms.wav", TOMS)
    row("toms", "MDX23C", uvr / f"synth_(toms){UVR_SUFFIX}", TOMS)
    print()
    row("hi-hat", "LarsNet", lars / "hihat.wav", HATS)
    row("hi-hat", "MDX23C", uvr / f"synth_(hh){UVR_SUFFIX}", HATS)
    print()
    print("Cymbals: LarsNet emits ONE stem, MDX23C splits ride from crash.")
    print("The synthetic fixture contains crashes only, so a good ride stem should stay quiet.")
    row("crash", "LarsNet", lars / "cymbals.wav", CRASH)
    row("crash", "MDX23C", uvr / f"synth_(crash){UVR_SUFFIX}", CRASH)
    row("ride", "MDX23C", uvr / f"synth_(ride){UVR_SUFFIX}", CRASH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
