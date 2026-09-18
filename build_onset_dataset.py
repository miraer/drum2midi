"""Training data: separated stems from GMD plus onset labels from the MIDI.

The model must see exactly what it will get at inference time, so training runs on the
separator's output - with all of its bleed and artefacts - rather than on clean audio.

The separator is chosen with --separator:
  larsnet  fast (about 6x real time), but dirtier stems
  uvr      MDX23C, much cleaner (tom contrast 341x vs LarsNet's 138x), but about
           10x slower than real time

Saves log-mel spectrograms (float16) and per-frame onset targets.

    python build_onset_dataset.py --limit 20                  # pilot
    python build_onset_dataset.py --separator uvr             # clean stems
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import shutil
import subprocess
import sys
import tempfile
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
UVR_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
SR = 44100
FPS = 100
HOP = SR // FPS          # 441
N_MELS = 64

STEMS = ["kick", "snare", "toms", "hihat", "cymbals"]
# Roland TD-11 notes in GMD -> stem index
PITCH_TO_STEM = {
    **{p: 0 for p in (35, 36)},
    **{p: 1 for p in (38, 40, 37)},
    **{p: 2 for p in (48, 50, 45, 47, 43, 58)},
    **{p: 3 for p in (42, 22, 46, 26, 44)},
    **{p: 4 for p in (49, 55, 57, 52, 51, 59, 53)},
}
# MDX23C output stem names; ride and crash are merged into cymbals,
# because GMD's annotation does not separate them at the stem level
UVR_MAP = {"kick": "kick", "snare": "snare", "toms": "toms",
           "hh": "hihat", "crash": "cymbals", "ride": "cymbals"}


@contextlib.contextmanager
def _in_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


_MEL = None


def log_mel(sig: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram, (T, N_MELS), float32."""
    global _MEL
    import librosa
    if _MEL is None:
        _MEL = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=N_MELS,
                                   fmin=20, fmax=20000)
    S = np.abs(librosa.stft(sig, n_fft=2048, hop_length=HOP, center=True))
    return np.log10(1.0 + (_MEL @ S)).T.astype(np.float32)


def onset_targets(notes, n_frames: int) -> np.ndarray:
    """(5, T) float32. A 1.0 peak on the hit frame and 0.5 on its neighbours -
    standard target smoothing, otherwise the net is punished for one-frame misses."""
    y = np.zeros((len(STEMS), n_frames), dtype=np.float32)
    for t, pitch in notes:
        s = PITCH_TO_STEM.get(pitch)
        if s is None:
            continue
        f = int(round(t * FPS))
        if not 0 <= f < n_frames:
            continue
        y[s, f] = 1.0
        for d in (-1, 1):
            if 0 <= f + d < n_frames:
                y[s, f + d] = max(y[s, f + d], 0.5)
    return y


def separate_uvr(src: Path, sep, tmp_root: Path) -> dict:
    """MDX23C stems; ride and crash are summed into a shared cymbals stem.

    Separator writes into the directory given at construction time, so the temp
    folder is set once outside and its files are removed after each track."""
    for old in tmp_root.glob("*"):
        try:
            old.unlink()
        except OSError:
            pass
    produced = sep.separate(str(src))
    out = {}
    for f in produced:
        p = Path(f) if Path(f).is_absolute() else tmp_root / f
        if not p.exists():
            continue
        for raw, name in UVR_MAP.items():
            if f"({raw})" in p.name:
                y, _ = sf.read(str(p), always_2d=True)
                y = y.T.astype(np.float32).mean(axis=0)
                if name in out:
                    n = min(len(out[name]), len(y))
                    out[name] = out[name][:n] + y[:n]
                else:
                    out[name] = y
    for f in tmp_root.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--max-dur", type=float, default=90.0)
    ap.add_argument("--separator", default="larsnet", choices=["larsnet", "uvr"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        ROOT / "bench" / ("onset_data" if args.separator == "larsnet"
                          else f"onset_data_{args.separator}")

    import torch

    rows = [r for r in csv.DictReader(open(GMD / "info.csv", encoding="utf-8"))
            if r.get("audio_filename") and (GMD / r["audio_filename"]).exists()
            and args.min_dur <= float(r["duration"]) <= args.max_dur]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[: args.limit]
    total_h = sum(float(r["duration"]) for r in rows) / 3600
    print(f"tracks: {len(rows)}  ({total_h:.2f} h of audio), separator: {args.separator}")
    print(f"output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    t0 = time.time()

    uvr_sep = None
    uvr_tmp = None
    if args.separator == "uvr":
        from audio_separator.separator import Separator
        # Separation is almost all of this script's runtime, so moving it to the GPU
        # turns an overnight job into an afternoon one.
        from xpu_separate import patch_audio_separator
        if patch_audio_separator():
            print("  separator: Intel GPU (XPU)")
        uvr_tmp = Path(tempfile.mkdtemp(prefix="uvrds_"))
        uvr_sep = Separator(output_dir=str(uvr_tmp),
                            model_file_dir=str(ROOT / "uvr" / "models"), log_level=40)
        uvr_sep.load_model(model_filename=UVR_MODEL)

    sys.path.insert(0, str(LARSNET_DIR))
    with _in_dir(LARSNET_DIR):
        net = None
        if args.separator == "larsnet":
            from larsnet import LarsNet
            net = LarsNet(wiener_filter=True, wiener_exponent=1.0, device="cpu",
                          config="config.yaml")

        for i, r in enumerate(rows, 1):
            key = r["id"].replace("/", "_")
            dest = out_dir / f"{key}.npz"
            if dest.exists():
                skipped += 1
                continue
            try:
                audio, sr = sf.read(str(GMD / r["audio_filename"]), always_2d=True)
                if sr != SR:
                    continue
                audio = audio.T.astype(np.float32)
                if audio.shape[0] == 1:
                    audio = np.vstack([audio, audio])

                if net is not None:
                    with torch.no_grad():
                        sep = net.separate_wiener(torch.from_numpy(audio))
                    mono = {}
                    for name in STEMS:
                        s = sep[name].cpu().numpy()
                        mono[name] = s.mean(axis=0) if s.ndim > 1 else s
                else:
                    mono = separate_uvr(GMD / r["audio_filename"], uvr_sep, uvr_tmp)
                    if len(mono) < len(STEMS):
                        print(f"  ! {r['id']}: got {len(mono)} stems")
                        continue

                mels = [log_mel(mono[name]) for name in STEMS]
                n_frames = min(m.shape[0] for m in mels)
                X = np.stack([m[:n_frames] for m in mels]).astype(np.float16)

                pm = pretty_midi.PrettyMIDI(str(GMD / r["midi_filename"]))
                notes = [(float(n.start), n.pitch)
                         for inst in pm.instruments for n in inst.notes]
                y = onset_targets(notes, n_frames)

                np.savez_compressed(dest, X=X, y=y, split=r["split"], style=r["style"])
                done += 1
            except Exception as exc:
                print(f"  ! {r['id']}: {str(exc)[:70]}")
                continue

            if i % 10 == 0 or i == len(rows):
                el = time.time() - t0
                rate = el / max(done, 1)
                left = rate * (len(rows) - i) / 60
                print(f"  [{i:4d}/{len(rows)}] done {done}, skipped {skipped}, "
                      f"{rate:.1f} s/track, eta ~{left:.0f} min", flush=True)

    sys.path.remove(str(LARSNET_DIR))
    if uvr_tmp is not None:
        shutil.rmtree(uvr_tmp, ignore_errors=True)
    size = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1024 / 1024
    print(f"\nsaved {len(list(out_dir.glob('*.npz')))} tracks, "
          f"{size:.0f} MB -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
