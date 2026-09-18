"""Build a per-hit feature dataset from GMD, for the learned velocity and pedal models.

Loads LarsNet once and streams every track through it, so this is far faster than
shelling out to drum2midi per file. Stems are consumed in memory and discarded;
only the extracted features are kept.

    python build_dataset.py --limit 90
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GMD = ROOT / "gmd" / "groove"
LARSNET_DIR = ROOT / "larsnet"
OUT = ROOT / "bench" / "gmd_features.npz"
SR = 44100

sys.path.insert(0, str(ROOT))
from features import FEATURE_NAMES, hit_features, peak_in  # noqa: E402

FAMILY = {
    **{p: "kick" for p in (35, 36)},
    **{p: "snare" for p in (38, 40, 37)},
    **{p: "toms" for p in (48, 50, 45, 47, 43, 58)},
    **{p: "hihat" for p in (42, 22, 46, 26, 44)},
    **{p: "cymbals" for p in (49, 55, 57, 52, 51, 59, 53)},
}
ARTIC = {**{p: "closed" for p in (42, 22)}, **{p: "open" for p in (46, 26)}, 44: "pedal",
         **{p: "crash" for p in (49, 55, 57, 52)},
         **{p: "ride" for p in (51, 59, 53)}}
FAMILIES = ["kick", "snare", "toms", "hihat", "cymbals"]


@contextlib.contextmanager
def _in_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=90)
    ap.add_argument("--lo", type=float, default=5.0)
    ap.add_argument("--hi", type=float, default=60.0)
    args = ap.parse_args()

    import torch

    rows = [r for r in csv.DictReader(open(GMD / "info.csv", encoding="utf-8"))
            if r.get("audio_filename") and (GMD / r["audio_filename"]).exists()
            and args.lo <= float(r["duration"]) <= args.hi]
    # keep the official split balance so training never sees a test track
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "test", "validation")}
    take = {"train": int(args.limit * 0.6), "test": int(args.limit * 0.25),
            "validation": int(args.limit * 0.15)}
    chosen = []
    for s, k in take.items():
        chosen += by_split.get(s, [])[:k]
    print(f"tracks: {len(chosen)} "
          f"({', '.join(f'{s}={min(len(by_split.get(s, [])), k)}' for s, k in take.items())})")

    sys.path.insert(0, str(LARSNET_DIR))
    with _in_dir(LARSNET_DIR):
        from larsnet import LarsNet
        net = LarsNet(wiener_filter=True, wiener_exponent=1.0, device="cpu",
                      config="config.yaml")

        X, y, fam, art, trk, spl = [], [], [], [], [], []
        t0 = time.time()
        for i, r in enumerate(chosen, 1):
            wav = GMD / r["audio_filename"]
            audio, sr = sf.read(str(wav), always_2d=True)
            audio = audio.T.astype(np.float32)
            if sr != SR:
                continue
            if audio.shape[0] == 1:
                audio = np.vstack([audio, audio])
            with torch.no_grad():
                stems = net.separate_wiener(torch.from_numpy(audio))
            stems = {k: v.cpu().numpy() for k, v in stems.items()}
            mono = {k: v.mean(axis=0) if v.ndim > 1 else v for k, v in stems.items()}
            mix = audio.mean(axis=0)
            mix_max = float(np.max(np.abs(mix))) or 1.0

            pm = pretty_midi.PrettyMIDI(str(GMD / r["midi_filename"]))
            notes = [(float(nt.start), FAMILY.get(nt.pitch), ARTIC.get(nt.pitch),
                      int(nt.velocity))
                     for inst in pm.instruments for nt in inst.notes if FAMILY.get(nt.pitch)]

            track_max = {}
            for f in FAMILIES:
                sig = mono.get(f)
                peaks = [peak_in(sig, t) for t, ff, _, _ in notes if ff == f] if sig is not None else []
                track_max[f] = max(peaks) if peaks else 0.0

            n_hits = 0
            for t, f, a, v in notes:
                sig = mono.get(f)
                if sig is None or track_max.get(f, 0.0) <= 0:
                    continue
                X.append(hit_features(sig, t, track_max[f], peak_in(mix, t), mix_max))
                y.append(v); fam.append(f); art.append(a or ""); trk.append(r["id"])
                spl.append(r["split"])
                n_hits += 1
            print(f"  [{i:3d}/{len(chosen)}] {r['id']:<32} {n_hits:5d} hits "
                  f"({time.time() - t0:.0f}s)")

    sys.path.remove(str(LARSNET_DIR))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=np.array(X, np.float32), y=np.array(y, np.int16),
                        family=np.array(fam), artic=np.array(art),
                        track=np.array(trk), split=np.array(spl),
                        names=np.array(FEATURE_NAMES))
    print(f"\nsaved {len(X)} hits x {len(FEATURE_NAMES)} features -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
