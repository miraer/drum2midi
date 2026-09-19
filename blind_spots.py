"""How much of a recording falls into the transcriber's blind spot?

A passage in a real song produced ADTOF activations of 0.008-0.08 where the raw audio
had clear onsets, strength up to 12.55. That is noise rather than a sub-threshold
signal, so no threshold recovers it. Before building a fallback, the question is how
often it happens: if it is one passage in one song, a fallback risks more false notes
than it saves.

Method. Take the onset envelope of the audio and pick its peaks, which is a
transcriber-independent view of where hits are. For each peak, look at the ADTOF
activations in a short window around it. A peak is "blind" when every class stays below
half its threshold -- not merely below threshold, which would just mean a near miss.

The output is the share of audible onsets the model does not see, and where they are.

    python blind_spots.py song.wav
    python blind_spots.py song.wav --midi ours.mid --csv bench/blind.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import drum2midi  # noqa: E402

FPS = 100
LABELS = ["kick", "snare", "tom", "hi-hat", "cymbal"]


def onset_peaks(path: Path, delta: float = 0.6):
    """Where the audio itself says a hit happened, independent of any model."""
    audio, sr = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    y = librosa.resample(mono, orig_sr=sr, target_sr=22050)
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=256)
    times = librosa.times_like(env, sr=22050, hop_length=256)
    idx = librosa.util.peak_pick(env, pre_max=6, post_max=6, pre_avg=12, post_avg=12,
                                 delta=delta, wait=8)
    return times[idx], env[idx], len(mono) / sr


def activations(path: Path, device: str = "cpu") -> np.ndarray:
    import torch
    from adtof_pytorch import load_audio_for_model

    model = drum2midi._adtof_model(device)
    x = load_audio_for_model(str(path)).to(device)
    with torch.no_grad():
        act = model(x).cpu().numpy()[0]
    if act.ndim == 2 and act.shape[0] < act.shape[1]:
        act = act.T
    return act


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--delta", type=float, default=0.6,
                    help="peak-picking sensitivity on the raw envelope")
    ap.add_argument("--window", type=float, default=0.06,
                    help="seconds either side of a peak to look for activation")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    thr = np.array(drum2midi.DEFAULT_THRESHOLDS, dtype=float)
    print(f"thresholds {list(thr)}")

    times, strengths, duration = onset_peaks(args.audio, args.delta)
    print(f"{args.audio.name}: {duration:.1f}s, {len(times)} audible onsets "
          f"({len(times) / duration:.2f}/s)")

    act = activations(args.audio, args.device)
    print(f"activations {act.shape[0] / FPS:.1f}s at {FPS} fps\n")

    half = int(round(args.window * FPS))
    rows = []
    for t, s in zip(times, strengths):
        centre = int(round(t * FPS))
        lo, hi = max(0, centre - half), min(act.shape[0], centre + half + 1)
        if lo >= hi:
            continue
        peak = act[lo:hi].max(axis=0)
        ratio = float((peak / thr).max())        # 1.0 means it just reached threshold
        rows.append((float(t), float(s), ratio, LABELS[int(np.argmax(peak / thr))]))

    if not rows:
        print("no onsets found")
        return 1

    ratios = np.array([r[2] for r in rows])
    blind = ratios < 0.5
    near = (ratios >= 0.5) & (ratios < 1.0)
    seen = ratios >= 1.0

    print(f"{'category':<34}{'onsets':>8}{'share':>8}")
    print("-" * 50)
    print(f"{'seen (reached threshold)':<34}{int(seen.sum()):>8}"
          f"{seen.mean():>8.1%}")
    print(f"{'near miss (half to full)':<34}{int(near.sum()):>8}{near.mean():>8.1%}")
    print(f"{'blind (under half threshold)':<34}{int(blind.sum()):>8}{blind.mean():>8.1%}")

    if blind.any():
        # group blind onsets into passages, so one bad bar is not reported as 30 events
        bt = [r[0] for r, b in zip(rows, blind) if b]
        runs, start, prev = [], bt[0], bt[0]
        for t in bt[1:]:
            if t - prev > 2.0:
                runs.append((start, prev))
                start = t
            prev = t
        runs.append((start, prev))
        runs = [(a, b) for a, b in runs if b - a > 0.5]
        covered = sum(b - a for a, b in runs)
        print(f"\n{len(runs)} blind passages longer than 0.5s, "
              f"{covered:.1f}s total = {covered / duration:.1%} of the recording")
        for a, b in sorted(runs, key=lambda r: r[0] - r[1])[:10]:
            n = sum(1 for t in bt if a <= t <= b)
            print(f"  {a:7.2f}s - {b:7.2f}s  ({b - a:5.1f}s, {n:>3} onsets)")

        strong = [r for r, x in zip(rows, blind) if x and r[1] > np.median(strengths)]
        print(f"\n{len(strong)} of the blind onsets are louder than the median onset, "
              f"so they are not faint hits")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as fh:
            fh.write("time,strength,ratio_to_threshold,best_class\n")
            for t, s, r, c in rows:
                fh.write(f"{t:.3f},{s:.3f},{r:.4f},{c}\n")
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
