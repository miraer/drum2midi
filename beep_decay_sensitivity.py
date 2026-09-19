"""How much does ReStem's trial watermark distort a decay measurement?

check_hihat_decay.py measures a broadband envelope, and the trial mixes a 603 Hz tone
into every stem it writes. A tone sitting inside a hit's decay window holds the envelope
up and makes the hit look like it rang longer than it did. That measurement is the only
number in our ReStem comparison taken from their audio rather than their MIDI, so it is
the only place the watermark could have touched our published figures.

This measures decay at moments where the watermark is sounding and at moments where it is
not, with the filter off and on, and reports the difference. If filtering barely moves the
numbers, the watermark never mattered and the 1.51x ratio stands as measured.

    python beep_decay_sensitivity.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEMS = Path(os.environ.get("APPDATA", "")) / "ReStem 2" / "cache" / "stems"
SR = 22050
BEEP_HZ = 603.0


def highpass(y: np.ndarray, fc_hz: float, n: int = 257) -> np.ndarray:
    fc = fc_hz / (SR / 2)
    k = np.arange(n) - (n - 1) / 2
    lp = np.sinc(fc * k) * fc * np.hanning(n)
    lp /= lp.sum()
    hp = -lp
    hp[(n - 1) // 2] += 1.0
    return np.convolve(y, hp, mode="same")


def decay_time(y: np.ndarray, at: float, floor_db: float = -20.0,
               limit: float = 0.6) -> float | None:
    a = int(at * SR)
    b = min(a + int(limit * SR), len(y))
    if b - a < int(0.05 * SR):
        return None
    seg = np.abs(y[a:b])
    if seg.max() < 1e-6:
        return None
    win = int(0.005 * SR)
    env = np.convolve(seg, np.ones(win) / win, mode="same")
    target = env.max() * (10 ** (floor_db / 20))
    below = np.where(env < target)[0]
    below = below[below > np.argmax(env)]
    return float(below[0] / SR) if len(below) else limit


def beep_times(y: np.ndarray) -> np.ndarray:
    win, hop = 2048, 256
    view = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
    mag = np.abs(np.fft.rfft(view * np.hanning(win), axis=1))
    freqs = np.fft.rfftfreq(win, 1 / SR)
    band = (freqs > BEEP_HZ - 30) & (freqs < BEEP_HZ + 30)
    ratio = mag[:, band].max(axis=1) / (np.median(mag, axis=1) + 1e-9)
    hot = ratio > np.percentile(ratio, 99.0)
    out, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            out.append((i + int(np.argmax(ratio[i:j + 1]))) * hop / SR)
            i = j + 1
        else:
            i += 1
    return np.array(out)


def summarise(label: str, raw: np.ndarray, filt: np.ndarray) -> None:
    if not len(raw):
        print(f"  {label:<26} no usable windows")
        return
    change = (np.median(raw) - np.median(filt)) / max(np.median(filt), 1e-6)
    print(f"  {label:<26}{len(raw):>5}{np.median(raw)*1000:>10.0f} ms"
          f"{np.median(filt)*1000:>11.0f} ms{change:>+9.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", type=Path, default=STEMS)
    ap.add_argument("--cut", type=float, default=1000.0)
    args = ap.parse_args()

    wav = args.stems / "hh.wav"
    if not wav.exists():
        print(f"no hi-hat stem at {wav}")
        return 1

    import librosa
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    yf = highpass(y, args.cut)

    src = args.stems / "ride.wav"          # empty stem: its tone is watermark only
    if src.exists():
        ref, _ = librosa.load(str(src), sr=SR, mono=True)
    else:
        ref = y
    beeps = beep_times(ref)
    dur = len(y) / SR
    print(f"stem {wav.name}, {dur:.1f}s, watermark instants found: {len(beeps)}")
    print(f"high-pass at {args.cut:.0f} Hz; a hi-hat lives near 12 kHz, the tone at "
          f"{BEEP_HZ:.0f} Hz\n")

    rng = np.random.default_rng(0)
    far = []
    while len(far) < 200:
        t = rng.uniform(0, max(dur - 0.6, 0.1))
        if len(beeps) == 0 or np.min(np.abs(beeps - t)) > 0.6:
            far.append(t)

    print(f"  {'window':<26}{'n':>5}{'unfiltered':>13}{'filtered':>14}{'change':>9}")
    print("  " + "-" * 67)
    for label, times in (("on the watermark", beeps), ("away from it", np.array(far))):
        raw = np.array([d for t in times if (d := decay_time(y, float(t))) is not None])
        filt = np.array([d for t in times if (d := decay_time(yf, float(t))) is not None])
        n = min(len(raw), len(filt))
        summarise(label, raw[:n], filt[:n])

    print("\nIf 'on the watermark' changes much more than 'away from it', the tone was\n"
          "propping up the envelope and any unfiltered decay figure is contaminated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
