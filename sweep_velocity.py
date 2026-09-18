"""Tune the velocity mapping against GMD's real module velocities.

Velocities are read at the ground-truth hit times, so this isolates the loudness
mapping from any onset-detection error. Reuses the stems cached by analyze_pedal.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GMD = ROOT / "gmd" / "groove"
WORK = ROOT / "bench" / "pedal"
SR = 44100
sys.path.insert(0, str(ROOT))
from drum2midi import _mono, peak_at  # noqa: E402

FAMILY = {
    **{p: "kick" for p in (35, 36)},
    **{p: "snare" for p in (38, 40, 37)},
    **{p: "tom" for p in (48, 50, 45, 47, 43, 58)},
    **{p: "hat" for p in (42, 22, 46, 26, 44)},
    **{p: "cym" for p in (49, 55, 57, 52, 51, 59, 53)},
}
STEM = {"kick": "kick.wav", "snare": "snare.wav", "hat": "hihat.wav",
        "tom": "toms.wav", "cym": "cymbals.wav"}
FAMILIES = ["kick", "snare", "hat", "tom", "cym"]


def velocities(peaks: np.ndarray, dyn_range: float, vmin: int, vmax: int) -> np.ndarray:
    loudest = peaks.max()
    if loudest <= 1e-6:
        return np.full(len(peaks), 100.0)
    db = 20.0 * np.log10(np.maximum(peaks, 1e-9) / loudest)
    norm = np.clip((db + dyn_range) / dyn_range, 0.0, 1.0)
    return np.clip(np.round(vmin + (vmax - vmin) * norm), 1, 127)


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    import csv
    if not (GMD / "info.csv").exists():
        print(f"Groove MIDI Dataset not found in {GMD}\n"
              f"See the Install section of README.md.")
        return 1
    rows = {r["id"].replace("/", "_"): r
            for r in csv.DictReader(open(GMD / "info.csv", encoding="utf-8"))}

    samples = {f: [] for f in FAMILIES}   # (peaks array, gt velocity array) per track
    tracks = [d for d in WORK.iterdir() if d.is_dir() and (d / "kick.wav").exists()]
    if not tracks:
        print(f"No cached stems in {WORK}. Run: python analyze_pedal.py --limit 8")
        return 1

    for d in tracks:
        meta = rows.get(d.name)
        if not meta:
            continue
        pm = pretty_midi.PrettyMIDI(str(GMD / meta["midi_filename"]))
        notes = [(float(n.start), FAMILY.get(n.pitch), int(n.velocity))
                 for inst in pm.instruments for n in inst.notes if FAMILY.get(n.pitch)]
        for fam in FAMILIES:
            path = d / STEM[fam]
            hits = [(t, v) for t, f, v in notes if f == fam]
            if not path.exists() or len(hits) < 8:
                continue
            sig = _mono(np.asarray(sf.read(str(path))[0]).T)
            peaks = np.array([peak_at(sig, t) for t, _ in hits])
            gt = np.array([v for _, v in hits], dtype=float)
            if peaks.max() <= 1e-6 or gt.std() == 0:
                continue
            samples[fam].append((peaks, gt))

    print(f"Tracks with cached stems: {len(tracks)}\n")
    grid = [12.0, 18.0, 24.0, 30.0, 36.0, 48.0]
    print(f"{'drum':<8}{'hits':>7}" + "".join(f"{g:>8.0f}dB" for g in grid) + f"{'best':>9}")
    print("-" * (15 + 10 * len(grid) + 9))
    for fam in FAMILIES:
        if not samples[fam]:
            continue
        n = sum(len(p) for p, _ in samples[fam])
        scores = []
        for dyn in grid:
            rs = []
            for peaks, gt in samples[fam]:
                est = velocities(peaks, dyn, 15, 127)
                if est.std() > 0:
                    rs.append(float(np.corrcoef(gt, est)[0, 1]))
            scores.append(float(np.mean(rs)) if rs else float("nan"))
        best = grid[int(np.nanargmax(scores))]
        print(f"{fam:<8}{n:>7}" + "".join(f"{s:>10.3f}" for s in scores) + f"{best:>8.0f}dB")

    print("\nPer-track Pearson r, averaged. Current default is 36 dB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
