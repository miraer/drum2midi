"""Per-hit feature extraction, shared by dataset building and inference.

Both paths must compute features identically or the learned models silently degrade,
so this lives in one place and is imported by build_dataset.py and drum2midi.py.
"""

from __future__ import annotations

import numpy as np

SR = 44100
N_MEL = 16

FEATURE_NAMES = (
    ["peak", "attack_rms", "tail_rms", "ratio", "centroid", "crest",
     "peak_rel_db", "attack_rel_db", "mix_peak_rel_db"]
    + [f"mel{i}" for i in range(N_MEL)]
)

_MEL = None


def _mel_filters(n_fft: int = 2048) -> np.ndarray:
    global _MEL
    if _MEL is None:
        import librosa
        _MEL = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=N_MEL, fmin=40, fmax=16000)
    return _MEL


def peak_in(sig: np.ndarray, t: float, pre: float = 0.005, post: float = 0.035) -> float:
    a = max(0, int((t - pre) * SR))
    b = min(len(sig), int((t + post) * SR))
    return float(np.max(np.abs(sig[a:b]))) if b > a else 0.0


def hit_features(sig: np.ndarray, t: float, track_max: float,
                 mix_peak: float, mix_max: float) -> np.ndarray:
    n = len(sig)
    peak = peak_in(sig, t)
    att = sig[int(t * SR):min(int((t + 0.05) * SR), n)]
    tail = sig[min(int((t + 0.08) * SR), n):min(int((t + 0.25) * SR), n)]
    att_rms = float(np.sqrt(np.mean(att ** 2))) if att.size else 0.0
    tail_rms = float(np.sqrt(np.mean(tail ** 2))) if tail.size else 0.0

    seg = sig[int(t * SR):min(n, int((t + 0.06) * SR))]
    if seg.size < 2048:
        seg = np.pad(seg, (0, 2048 - seg.size))
    spec = np.abs(np.fft.rfft(seg[:2048] * np.hanning(2048)))
    mel = np.log10(_mel_filters() @ spec + 1e-8)
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    band = freqs >= 200
    total = spec[band].sum()
    centroid = float((spec[band] * freqs[band]).sum() / total) if total > 0 else 0.0

    def rel_db(x, ref):
        return 20.0 * np.log10(max(x, 1e-9) / max(ref, 1e-9))

    return np.array(
        [peak, att_rms, tail_rms, tail_rms / (att_rms + 1e-9), centroid,
         peak / (att_rms + 1e-9), rel_db(peak, track_max), rel_db(att_rms, track_max),
         rel_db(mix_peak, mix_max)] + list(mel), dtype=np.float32)


def features_for(sig: np.ndarray, times, mix: np.ndarray) -> np.ndarray:
    """Feature matrix for every hit of one drum, normalised against that drum's
    loudest hit in this track - the same convention the models were trained on."""
    if not len(times):
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    peaks = [peak_in(sig, t) for t in times]
    track_max = max(peaks) if peaks else 0.0
    mix_max = float(np.max(np.abs(mix))) if mix.size else 1.0
    if track_max <= 0:
        track_max = 1e-9
    return np.vstack([hit_features(sig, t, track_max, peak_in(mix, t), mix_max)
                      for t in times])

if __name__ == "__main__":
    # a library, not a command; printing the docstring beats exiting silently
    print(__doc__)
    print("This module is imported by other scripts; it has no command line.")
