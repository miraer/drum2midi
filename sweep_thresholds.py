"""Grid-search ADTOF peak-picking thresholds against MDB Drums ground truth.

Model inference is the expensive part and thresholds only affect peak picking, so
activations are computed once per track, cached, and then swept for free.

    python sweep_thresholds.py            # full grid over all 23 tracks
    python sweep_thresholds.py --limit 6  # quicker
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import mir_eval
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mdbdrums" / "MDB Drums"
AUDIO = DATA / "audio" / "drum_only"
ANN = DATA / "annotations" / "class"
CACHE = ROOT / "bench" / "activations"
WINDOW = 0.05

# ADTOF emits its 5 classes in this order; MDB uses these labels for the same drums
LABELS = ["KD", "SD", "TT", "HH", "CY"]
PRETTY = {"KD": "kick", "SD": "snare", "TT": "toms", "HH": "hi-hat", "CY": "cymbals"}
DEFAULTS = [0.22, 0.24, 0.32, 0.22, 0.30]

GRID = {
    "KD": [0.10, 0.14, 0.18, 0.22],
    "SD": [0.08, 0.11, 0.14, 0.18, 0.24],
    "TT": [0.32, 0.45, 0.60, 0.75, 0.88],
    "HH": [0.10, 0.14, 0.18, 0.22],
    "CY": [0.16, 0.22, 0.28, 0.34],
}


def activations_for(wav: Path) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{wav.stem}.npy"
    if cached.exists():
        return np.load(cached)

    import torch
    from adtof_pytorch import (calculate_n_bins, create_frame_rnn_model,
                               get_default_weights_path, load_audio_for_model,
                               load_pytorch_weights)

    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    weights = get_default_weights_path()
    if weights and Path(weights).exists():
        model = load_pytorch_weights(model, weights, strict=False)
    with torch.no_grad():
        act = model(load_audio_for_model(str(wav))).cpu().numpy()[0]
    np.save(cached, act)
    return act


def read_annotation(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {k: [] for k in LABELS}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] in out:
            out[p[1]].append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def pick(act_col: np.ndarray, threshold: float, fps: int = 100) -> np.ndarray:
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    proc = NotePeakPickingProcessor(threshold=threshold, pre_avg=0.1, post_avg=0.01,
                                    pre_max=0.02, post_max=0.01, combine=0.02, fps=fps)
    return np.array([t for t, _ in proc.process(act_col)])


def score(counts: dict) -> float:
    p = counts["tp"] / max(counts["est"], 1)
    r = counts["tp"] / max(counts["ref"], 1)
    return 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tracks = sorted(AUDIO.glob("*.wav"))[: args.limit]
    print(f"Computing activations for {len(tracks)} tracks (cached after first run) ...")
    data = []
    for i, wav in enumerate(tracks, 1):
        ann = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
        if not ann.exists():
            continue
        data.append((activations_for(wav), read_annotation(ann)))
        print(f"  [{i:2d}/{len(tracks)}] {wav.stem}")

    # Each class is scored independently, so the grid factorises: sweep one drum at a time.
    print("\nSweeping each class independently ...\n")
    best = {}
    print(f"{'drum':<10}{'threshold':>10}{'F1':>8}   {'(default)':>10}{'F1':>8}")
    print("-" * 50)
    for ci, label in enumerate(LABELS):
        rows = []
        for thr in GRID[label]:
            c = {"tp": 0, "ref": 0, "est": 0}
            for act, ref in data:
                r = ref[label]
                e = pick(act[:, ci], thr)
                c["ref"] += len(r)
                c["est"] += len(e)
                if len(r) and len(e):
                    _, p, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
                    c["tp"] += int(round(p * len(e)))
            rows.append((score(c), thr))
        rows.sort(reverse=True)
        best[label] = rows[0][1]

        cd = {"tp": 0, "ref": 0, "est": 0}
        for act, ref in data:
            r = ref[label]
            e = pick(act[:, ci], DEFAULTS[ci])
            cd["ref"] += len(r); cd["est"] += len(e)
            if len(r) and len(e):
                _, p, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
                cd["tp"] += int(round(p * len(e)))
        print(f"{PRETTY[label]:<10}{rows[0][1]:>10.2f}{rows[0][0]:>8.3f}   "
              f"{DEFAULTS[ci]:>10.2f}{score(cd):>8.3f}")

    tuned = [best[l] for l in LABELS]
    print("\nTuned thresholds (kick,snare,tom,hat,cymbal):")
    print("  " + ",".join(f"{v:.2f}" for v in tuned))

    for name, thr in (("default", DEFAULTS), ("tuned", tuned)):
        micro = {"tp": 0, "ref": 0, "est": 0}
        for ci, label in enumerate(LABELS):
            for act, ref in data:
                r = ref[label]
                e = pick(act[:, ci], thr[ci])
                micro["ref"] += len(r); micro["est"] += len(e)
                if len(r) and len(e):
                    _, p, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
                    micro["tp"] += int(round(p * len(e)))
        p = micro["tp"] / max(micro["est"], 1)
        r = micro["tp"] / max(micro["ref"], 1)
        print(f"  MICRO {name:<8} P={p:.3f} R={r:.3f} F1={score(micro):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
