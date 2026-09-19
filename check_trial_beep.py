"""Did the trial beep contaminate the ReStem comparison?

ReStem's user's guide, page 14, carries a warning: the free trial mixes a short beep into
the separated audio every few seconds, and it is baked into the stems, including ones
rendered offline. Our entire published comparison was run on that trial.

If TrigNet reads the stems after the beep is mixed in, it may be triggering on it, which
would inflate ReStem's false-positive count -- and ReStem's extra notes are most of why we
score higher. That would make our headline result an artefact of their trial rather than a
property of their product, which is a far worse error than any we have retracted so far.

This looks for a periodic component two ways, on whatever ReStem left in its cache:

  audio   a strong, narrow spectral peak recurring at a near-constant interval
  events  onsets in trigger_events.json whose spacing clusters around one value

A negative result here is as useful as a positive one: it would mean the beep is added on
playback rather than before the trigger engine, and the comparison stands.

    python check_trial_beep.py
    python check_trial_beep.py --stems "C:/path/to/stems"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_STEMS = Path(os.environ.get("APPDATA", "")) / "ReStem 2" / "cache" / "stems"


def periodic_onsets(times: list[float], tol: float = 0.05) -> tuple[float, int, float]:
    """Largest cluster of near-equal gaps: (gap, count, share of all gaps)."""
    if len(times) < 4:
        return (0.0, 0, 0.0)
    gaps = np.diff(np.sort(np.asarray(times, dtype=float)))
    gaps = gaps[gaps > 0.2]  # a beep every few seconds, not drum spacing
    if len(gaps) < 3:
        return (0.0, 0, 0.0)
    rounded = Counter(np.round(gaps / tol) * tol)
    gap, count = rounded.most_common(1)[0]
    return (float(gap), int(count), count / len(gaps))


def scan_events(path: Path) -> None:
    print(f"\n=== {path.name} — onset spacing per stem ===")
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    sr = data.get("sr", 44100)
    hop = data.get("hop", 441)
    for stem, events in (data.get("stems") or {}).items():
        times = []
        for e in events:
            if "sample" in e:
                times.append(e["sample"] / sr)
            elif "frame" in e:
                times.append(e["frame"] * hop / sr)
        gap, count, share = periodic_onsets(times)
        flag = ""
        if count >= 4 and share > 0.5 and gap > 0.5:
            flag = "  <-- REGULAR, look at this"
        print(f"  {stem:<8} {len(times):>5} onsets   dominant gap "
              f"{gap:>5.2f}s x{count:<4} ({share:4.0%}){flag}")


def scan_audio(path: Path, top_n: int = 3) -> None:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    win = 2048
    hop = 512
    frames = 1 + (len(x) - win) // hop
    if frames < 8:
        print(f"  {path.name:<14} too short")
        return

    mag = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win),
        axis=1))
    freqs = np.fft.rfftfreq(win, 1 / sr)

    # A beep is a narrow tone: one bin far above the median of its own frame.
    med = np.median(mag, axis=1, keepdims=True) + 1e-9
    peakiness = mag.max(axis=1) / med[:, 0]
    bin_of = mag.argmax(axis=1)

    loud = peakiness > np.percentile(peakiness, 98)
    if loud.sum() < 4:
        print(f"  {path.name:<14} no narrow tonal frames stand out")
        return

    times = np.flatnonzero(loud) * hop / sr
    gap, count, share = periodic_onsets(list(times))
    hz = float(np.median(freqs[bin_of[loud]]))
    flag = "  <-- REGULAR" if count >= 4 and share > 0.5 and gap > 0.5 else ""
    print(f"  {path.name:<14} {loud.sum():>4} tonal frames, median {hz:>7.0f} Hz, "
          f"dominant gap {gap:>5.2f}s ({share:4.0%}){flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", type=Path, default=DEFAULT_STEMS)
    args = ap.parse_args()

    if not args.stems.exists():
        print(f"no stems at {args.stems}")
        return 1

    print(f"stems: {args.stems}")
    events = args.stems / "trigger_events.json"
    if events.exists():
        scan_events(events)

    print("\n=== audio — narrow tonal frames and their spacing ===")
    for wav in sorted(args.stems.glob("*.wav")):
        try:
            scan_audio(wav)
        except Exception as exc:
            print(f"  {wav.name:<14} {exc}")

    print("\nA beep every few seconds would show as the same dominant gap across every\n"
          "stem, at the same frequency. Drum onsets do not line up that way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
