"""A stem trigger: hit detection in the amplitude domain.

ADTOF looks for notes in a mixture and is trained on mixtures. On an isolated stem the
instrument is already known, so the task reduces to finding transients - that is a
trigger's job, not a transcriber's. This is how Drumagog and ReStem's TrigNet work.

The key idea against ghost notes: the threshold is relative to the local level rather
than absolute, and the detector works on the derivative of the LOG envelope. A quiet hit
produces the same relative jump as a loud one, so it stops disappearing under the
dynamic range of the performance.
"""

from __future__ import annotations

import numpy as np

SR = 44100


def peak_envelope(sig: np.ndarray, hop: int = 64) -> np.ndarray:
    """Frame-wise peak envelope. hop=64 -> ~1.45 ms, well inside the 50 ms tolerance."""
    n = len(sig) // hop
    if n < 2:
        return np.zeros(max(n, 1), dtype=np.float32)
    trimmed = np.abs(sig[: n * hop]).reshape(n, hop)
    return trimmed.max(axis=1).astype(np.float32)


def onset_function(sig: np.ndarray, hop: int = 64, floor_db: float = -70.0) -> np.ndarray:
    """Positive derivative of the log envelope.

    On a log scale a velocity-20 hit and a velocity-120 hit produce a comparable
    jump, so the function does not depend on how loudly the passage is played.
    """
    env = peak_envelope(sig, hop)
    ref = float(np.max(env)) if env.size else 0.0
    if ref <= 1e-9:
        return np.zeros_like(env)
    log_env = 20.0 * np.log10(np.maximum(env, 1e-9) / ref)
    log_env = np.maximum(log_env, floor_db)
    diff = np.diff(log_env, prepend=log_env[0])
    return np.maximum(diff, 0.0)


def _sliding_stats(x: np.ndarray, win: int) -> tuple:
    """Rolling mean and standard deviation via cumulative sums."""
    win = max(win, 3)
    pad = win // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    c1 = np.cumsum(np.insert(padded, 0, 0.0))
    c2 = np.cumsum(np.insert(padded ** 2, 0, 0.0))
    n = len(x)
    s1 = c1[win:win + n] - c1[:n]
    s2 = c2[win:win + n] - c2[:n]
    mean = s1 / win
    var = np.maximum(s2 / win - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def trigger(sig: np.ndarray, sensitivity: float = 3.0, holdoff: float = 0.020,
            hop: int = 64, window: float = 0.5, min_jump_db: float = 1.5,
            noise_floor_db: float = -55.0) -> np.ndarray:
    """Hit times in seconds.

    sensitivity     how many standard deviations above the local mean a peak must be
    holdoff         how long to ignore new hits after one fires
    min_jump_db     minimum envelope jump, rejects noise
    noise_floor_db  level below which a hit counts as stem silence
    """
    if sig.size < hop * 4:
        return np.array([])
    fn = onset_function(sig, hop)
    if not fn.any():
        return np.array([])

    win = max(int(window * SR / hop), 8)
    mean, std = _sliding_stats(fn, win)
    thr = mean + sensitivity * np.maximum(std, 1e-6)

    env = peak_envelope(sig, hop)
    ref = float(np.max(env)) or 1.0
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9) / ref)

    candidate = (fn > thr) & (fn > min_jump_db) & (env_db > noise_floor_db)
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return np.array([])

    # keep the strongest of adjacent candidates, then respect the holdoff
    hold = max(int(holdoff * SR / hop), 1)
    kept = []
    last = -10 ** 9
    for i in idx:
        if i - last < hold:
            if kept and fn[i] > fn[kept[-1]]:
                kept[-1] = i
                last = i
            continue
        kept.append(int(i))
        last = i
    return np.array(kept, dtype=float) * hop / SR

if __name__ == "__main__":
    # a library, not a command; printing the docstring beats exiting silently
    print(__doc__)
    print("This module is imported by other scripts; it has no command line.")
