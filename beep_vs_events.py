"""Do ReStem's trigger events land on the trial beep?

check_trial_beep.py established that a 603 Hz tone is baked into every stem, identically,
which is the trial watermark the user's guide warns about. That alone does not damage the
comparison. What would damage it is the trigger engine firing on it, because ReStem's
extra notes are most of why our published MICRO F1 is higher than theirs.

This is the direct test: find when the beep sounds, then ask whether ReStem emitted an
onset at those moments more often than chance would give.

The null model matters. Onsets are dense, so some will coincide with any set of times by
accident. The comparison is therefore against the same number of randomly placed instants,
repeated, which gives a distribution rather than a single number to argue with.

    python beep_vs_events.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEMS = Path(os.environ.get("APPDATA", "")) / "ReStem 2" / "cache" / "stems"
BEEP_HZ = 603.0
TOL = 0.05  # the same 50 ms the benchmark uses


def beep_times(wav: Path, hz: float = BEEP_HZ) -> tuple[np.ndarray, float]:
    """Times where a narrow band around `hz` spikes well above its own median."""
    import soundfile as sf

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    win, hop = 2048, 256
    view = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    mag = np.abs(np.fft.rfft(view * np.hanning(win), axis=1))
    freqs = np.fft.rfftfreq(win, 1 / sr)

    band = (freqs > hz - 30) & (freqs < hz + 30)
    energy = mag[:, band].max(axis=1)
    rest = np.median(mag, axis=1) + 1e-9
    ratio = energy / rest

    thr = np.percentile(ratio, 99.0)
    hot = ratio > thr
    # collapse runs into one time each
    times, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            times.append((i + np.argmax(ratio[i:j + 1])) * hop / sr)
            i = j + 1
        else:
            i += 1
    return np.array(times), len(x) / sr


def load_events(path: Path) -> tuple[dict[str, np.ndarray], float]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    sr = data.get("sr", 44100)
    hop = data.get("hop", 441)
    out, last = {}, 0.0
    for stem, events in (data.get("stems") or {}).items():
        t = []
        for e in events:
            if "sample" in e:
                t.append(e["sample"] / sr)
            elif "frame" in e:
                t.append(e["frame"] * hop / sr)
        out[stem] = np.sort(np.array(t))
        if len(t):
            last = max(last, max(t))
    return out, last


def hits(targets: np.ndarray, onsets: np.ndarray, tol: float) -> int:
    if len(targets) == 0 or len(onsets) == 0:
        return 0
    idx = np.searchsorted(onsets, targets)
    n = 0
    for t, i in zip(targets, idx):
        for j in (i - 1, i):
            if 0 <= j < len(onsets) and abs(onsets[j] - t) <= tol:
                n += 1
                break
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", type=Path, default=STEMS)
    ap.add_argument("--beep-source", default="",
                    help="stem to read the beep from. An empty stem is ideal: whatever "
                         "tone is in it can only be the watermark")
    ap.add_argument("--trials", type=int, default=2000)
    args = ap.parse_args()

    ev_path = args.stems / "trigger_events.json"
    if not ev_path.exists():
        print(f"no trigger_events.json under {args.stems}")
        return 1

    events, last = load_events(ev_path)
    if args.beep_source:
        wav = args.stems / args.beep_source
    else:
        # prefer a stem the trigger engine found nothing in: any tone there is the
        # watermark and nothing else, which removes the guesswork
        empty = [s for s, o in events.items() if len(o) == 0]
        wav = next((args.stems / f"{s}.wav" for s in empty
                    if (args.stems / f"{s}.wav").exists()), None)
        if wav is None:
            wav = next((args.stems / n for n in ("other.wav", "crash.wav", "ride.wav")
                        if (args.stems / n).exists()), None)
    if wav is None or not wav.exists():
        print("no stem audio to read the beep from")
        return 1

    beeps, duration = beep_times(wav)
    print(f"beep source : {wav.name}, {duration:.1f}s")
    print(f"beeps found : {len(beeps)}")
    if len(beeps) > 1:
        gaps = np.diff(beeps)
        print(f"spacing     : median {np.median(gaps):.2f}s, "
              f"sd {gaps.std():.3f}s  (a watermark is regular; music is not)")
    print(f"tolerance   : +/-{TOL*1000:.0f} ms\n")

    rng = np.random.default_rng(0)
    print(f"  {'stem':<8}{'onsets':>8}{'on beeps':>10}{'chance':>9}{'p':>8}")
    for stem, onsets in events.items():
        if len(onsets) == 0:
            continue
        real = hits(beeps, onsets, TOL)
        null = np.empty(args.trials, dtype=int)
        for k in range(args.trials):
            fake = np.sort(rng.uniform(0, duration, size=len(beeps)))
            null[k] = hits(fake, onsets, TOL)
        p = float((null >= real).mean())
        mark = "  <-- above chance" if p < 0.05 and real > 0 else ""
        print(f"  {stem:<8}{len(onsets):>8}{real:>10}{null.mean():>9.1f}{p:>8.3f}{mark}")

    print("\nIf no stem fires on the beep above chance, the watermark is mixed in after\n"
          "the trigger engine and the published comparison is unaffected by it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
