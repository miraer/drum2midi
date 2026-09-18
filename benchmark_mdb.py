"""Benchmark drum2midi on MDB Drums - 23 real tracks, 7994 hand-annotated onsets.

This replaces the synthetic fixture with real drum recordings. Scoring uses
mir_eval with the 50 ms window from the MIREX drum transcription task, so the
numbers are comparable to published ADT results.

    python benchmark_mdb.py                      # full run, default separator
    python benchmark_mdb.py --limit 5            # quick pass over 5 tracks
    python benchmark_mdb.py --no-separate        # ADTOF only, much faster
    python benchmark_mdb.py --separator hybrid   # any drum2midi flag is forwarded
    python benchmark_mdb.py --rescore            # re-score cached MIDI, no inference
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mdbdrums" / "MDB Drums"
AUDIO = DATA / "audio" / "drum_only"
ANN = DATA / "annotations" / "class"
WINDOW = 0.05  # MIREX drum transcription tolerance

# MDB Drums label -> the MIDI pitches our pipeline can emit for it
CLASSES = {
    "KD": (35, 36),
    "SD": (38, 40),
    "HH": (42, 44, 46),
    "TT": (41, 43, 45, 47, 48, 50),
    "CY": (49, 51),
}
ORDER = ["KD", "SD", "HH", "TT", "CY"]
PRETTY = {"KD": "kick", "SD": "snare", "HH": "hi-hat", "TT": "toms", "CY": "cymbals"}


def read_annotation(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {k: [] for k in CLASSES}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in out:
            out[parts[1]].append(float(parts[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def read_estimate(path: Path) -> dict[str, np.ndarray]:
    notes = pretty_midi.PrettyMIDI(str(path)).instruments[0].notes
    return {k: np.array(sorted(n.start for n in notes if n.pitch in pitches))
            for k, pitches in CLASSES.items()}


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--tag", default="default", help="Subfolder for cached MIDI")
    ap.add_argument("-h", "--help", action="store_true")
    args, passthrough = ap.parse_known_args()
    if args.help:
        print(__doc__)
        return 0

    if not AUDIO.exists():
        print(f"MDB Drums not found at {DATA}")
        return 1

    tracks = sorted(AUDIO.glob("*.wav"))[: args.limit]
    outdir = ROOT / "bench" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.rescore:
        print(f"Running pipeline on {len(tracks)} tracks "
              f"({' '.join(passthrough) if passthrough else 'defaults'}) ...")
        t0 = time.time()
        for i, wav in enumerate(tracks, 1):
            mid = outdir / f"{wav.stem}.mid"
            res = subprocess.run(
                [sys.executable, str(ROOT / "drum2midi.py"), str(wav), "-o", str(mid),
                 "--device", "auto"] + passthrough,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            flag = "ok " if res.returncode == 0 else "FAIL"
            print(f"  [{i:2d}/{len(tracks)}] {flag} {wav.stem}")
            if res.returncode != 0:
                print("        " + res.stderr.strip().splitlines()[-1][:120])
        print(f"  inference took {time.time() - t0:.0f}s\n")

    totals = {k: {"tp": 0, "ref": 0, "est": 0} for k in CLASSES}
    per_track = []

    for wav in tracks:
        mid = outdir / f"{wav.stem}.mid"
        ann = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
        if not mid.exists() or not ann.exists():
            continue
        ref, est = read_annotation(ann), read_estimate(mid)
        track_f = []
        for k in ORDER:
            r, e = ref[k], est[k]
            if r.size == 0 and e.size == 0:
                continue
            if r.size and e.size:
                f, p, rc = mir_eval.onset.f_measure(r, e, window=WINDOW)
                tp = int(round(p * len(e)))
            else:
                f = tp = 0
            totals[k]["tp"] += tp
            totals[k]["ref"] += len(r)
            totals[k]["est"] += len(e)
            if r.size:
                track_f.append(f)
        per_track.append((wav.stem, float(np.mean(track_f)) if track_f else 0.0))

    print(f"{'drum':<10}{'ref':>7}{'est':>7}{'match':>7}{'prec':>8}{'rec':>8}{'F1':>8}")
    print("-" * 55)
    micro = {"tp": 0, "ref": 0, "est": 0}
    for k in ORDER:
        t = totals[k]
        if t["ref"] == 0:
            continue
        p = t["tp"] / max(t["est"], 1)
        r = t["tp"] / max(t["ref"], 1)
        f = 2 * p * r / max(p + r, 1e-9)
        for key in micro:
            micro[key] += t[key]
        print(f"{PRETTY[k]:<10}{t['ref']:>7}{t['est']:>7}{t['tp']:>7}{p:>8.3f}{r:>8.3f}{f:>8.3f}")
    p = micro["tp"] / max(micro["est"], 1)
    r = micro["tp"] / max(micro["ref"], 1)
    print("-" * 55)
    print(f"{'MICRO':<10}{micro['ref']:>7}{micro['est']:>7}{micro['tp']:>7}"
          f"{p:>8.3f}{r:>8.3f}{2 * p * r / max(p + r, 1e-9):>8.3f}")

    per_track.sort(key=lambda x: x[1])
    print(f"\nworst 3 tracks: " + ", ".join(f"{n.replace('MusicDelta_', '')} {v:.2f}"
                                           for n, v in per_track[:3]))
    print(f"best 3 tracks:  " + ", ".join(f"{n.replace('MusicDelta_', '')} {v:.2f}"
                                          for n, v in per_track[-3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
