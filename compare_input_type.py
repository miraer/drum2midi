"""How much accuracy is lost when a whole song is fed in instead of a drum stem.

MDB Drums ships the same 23 recordings twice: drum_only (isolated drums) and full_mix
(the whole band). The annotations are shared, so the comparison is clean - only the
input changes.

A full mix goes through an extra htdemucs stage that extracts the drums, and that
stage's errors stack on top of everything else.

    python compare_input_type.py --separator larsnet
    python compare_input_type.py --separator larsnet --rescore
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
MDB = ROOT / "mdbdrums" / "MDB Drums"
DRUM = MDB / "audio" / "drum_only"
MIX = MDB / "audio" / "full_mix"
ANN = MDB / "annotations" / "class"
WINDOW = 0.05

CLASSES = {
    "KD": (35, 36), "SD": (38, 40), "TT": (41, 43, 45, 47, 48, 50),
    "HH": (42, 44, 46), "CY": (49, 51),
}
PRETTY = {"KD": "kick", "SD": "snare", "TT": "toms", "HH": "hi-hat", "CY": "cymbals"}


def read_ann(path: Path) -> dict:
    out = {k: [] for k in CLASSES}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] in out:
            out[p[1]].append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def read_midi(path: Path) -> dict:
    notes = [n for inst in pretty_midi.PrettyMIDI(str(path)).instruments for n in inst.notes]
    return {k: np.array(sorted(n.start for n in notes if n.pitch in p))
            for k, p in CLASSES.items()}


def score(ref, est) -> tuple:
    if not len(ref) or not len(est):
        return 0, len(ref), len(est)
    _, p, _ = mir_eval.onset.f_measure(ref, est, window=WINDOW)
    return int(round(p * len(est))), len(ref), len(est)


def prf(c):
    p = c["tp"] / max(c["est"], 1)
    r = c["tp"] / max(c["ref"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--separator", default="larsnet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()

    tracks = []
    for stem_wav in sorted(DRUM.glob("*.wav")):
        base = stem_wav.stem.replace("_Drum", "")
        mix_wav = MIX / f"{base}_MIX.wav"
        ann = ANN / f"{base}_class.txt"
        if mix_wav.exists() and ann.exists():
            tracks.append((base, stem_wav, mix_wav, ann))
    tracks = tracks[: args.limit]
    print(f"tracks: {len(tracks)}, separator: {args.separator}")

    variants = {
        "stem": (ROOT / "bench" / f"inp_stem_{args.separator}", []),
        "song": (ROOT / "bench" / f"inp_song_{args.separator}", ["--from-song"]),
    }

    if not args.rescore:
        for label, (outdir, extra) in variants.items():
            outdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            for i, (base, stem_wav, mix_wav, _) in enumerate(tracks, 1):
                src = mix_wav if extra else stem_wav
                out = outdir / f"{base}.mid"
                if out.exists():
                    continue
                subprocess.run(
                    [sys.executable, str(ROOT / "drum2midi.py"), str(src), "-o", str(out),
                     "--device", "auto", "--separator", args.separator] + extra,
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
                el = time.time() - t0
                print(f"  [{label}] {i:2d}/{len(tracks)} {base:<28} "
                      f"{el/60:.1f} min", flush=True)

    totals = {lab: {k: {"tp": 0, "ref": 0, "est": 0} for k in CLASSES} for lab in variants}
    for base, _, _, ann_path in tracks:
        ref = read_ann(ann_path)
        for label, (outdir, _) in variants.items():
            mid = outdir / f"{base}.mid"
            if not mid.exists():
                continue
            est = read_midi(mid)
            for k in CLASSES:
                tp, nr, ne = score(ref[k], est[k])
                totals[label][k]["tp"] += tp
                totals[label][k]["ref"] += nr
                totals[label][k]["est"] += ne

    labels = list(variants)
    print(f"\n{'instrument':<11}{'reference':>8}" + "".join(f"{l + ' F1':>12}" for l in labels)
          + f"{'delta':>9}")
    print("-" * 56)
    micro = {l: {"tp": 0, "ref": 0, "est": 0} for l in labels}
    for k in CLASSES:
        row = f"{PRETTY[k]:<11}{totals[labels[0]][k]['ref']:>8}"
        f1s = []
        for l in labels:
            c = totals[l][k]
            for key in micro[l]:
                micro[l][key] += c[key]
            f1s.append(prf(c)[2])
            row += f"{f1s[-1]:>12.3f}"
        row += f"{f1s[1] - f1s[0]:>+9.3f}"
        print(row)
    print("-" * 56)
    f1s = [prf(micro[l])[2] for l in labels]
    print(f"{'TOTAL':<11}{micro[labels[0]]['ref']:>8}"
          + "".join(f"{f:>12.3f}" for f in f1s) + f"{f1s[1] - f1s[0]:>+9.3f}")
    for l in labels:
        p, r, _ = prf(micro[l])
        print(f"   {l:<8} precision {p:.3f}  recall {r:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
