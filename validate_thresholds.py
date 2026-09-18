"""Items 3 and 4: honest threshold validation, and per-track adaptive thresholds.

The thresholds currently shipped were tuned on all 23 MDB tracks and then reported on
those same 23 tracks, which is optimistically biased. This runs 2-fold cross-validation
over the tracks so every number comes from material the tuning never saw, and compares
three policies:

  default   ADTOF stock thresholds
  global    one tuned value per class, fitted on the training fold
  adaptive  per-track threshold at a percentile of that track's activation
            distribution, with the percentile fitted on the training fold

Uses the activation cache written by sweep_thresholds.py, so it is fast.

    python validate_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mir_eval
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
MDB = ROOT / "mdbdrums" / "MDB Drums"
AUDIO = MDB / "audio" / "drum_only"
ANN = MDB / "annotations" / "class"
CACHE = ROOT / "bench" / "activations"
WINDOW = 0.05

LABELS = ["KD", "SD", "TT", "HH", "CY"]
PRETTY = {"KD": "kick", "SD": "snare", "TT": "toms", "HH": "hi-hat", "CY": "cymbals"}
STOCK = [0.22, 0.24, 0.32, 0.22, 0.30]
GLOBAL_GRID = {"KD": [0.10, 0.14, 0.18, 0.22, 0.26], "SD": [0.08, 0.11, 0.14, 0.18, 0.24],
               "TT": [0.32, 0.45, 0.60, 0.75, 0.88], "HH": [0.10, 0.14, 0.18, 0.22, 0.26],
               "CY": [0.16, 0.22, 0.28, 0.34, 0.40]}
# percentile of a track's own activation values, plus an absolute floor
PCT_GRID = [88.0, 92.0, 95.0, 96.5, 98.0, 99.0]
FLOOR = {"KD": 0.08, "SD": 0.06, "TT": 0.25, "HH": 0.08, "CY": 0.12}


def pick(col: np.ndarray, thr: float) -> np.ndarray:
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def read_ann(path: Path) -> dict:
    out = {k: [] for k in LABELS}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] in out:
            out[p[1]].append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def tally(ref, est, acc):
    acc["ref"] += len(ref)
    acc["est"] += len(est)
    if len(ref) and len(est):
        _, p, _ = mir_eval.onset.f_measure(ref, est, window=WINDOW)
        acc["tp"] += int(round(p * len(est)))


def f1(acc) -> float:
    p = acc["tp"] / max(acc["est"], 1)
    r = acc["tp"] / max(acc["ref"], 1)
    return 2 * p * r / max(p + r, 1e-9)


def evaluate(data, ci, label, mode, param) -> dict:
    acc = {"tp": 0, "ref": 0, "est": 0}
    for act, ref in data:
        col = act[:, ci]
        if mode == "adaptive":
            thr = max(float(np.percentile(col, param)), FLOOR[label])
        else:
            thr = param
        tally(ref[label], pick(col, thr), acc)
    return acc


def main() -> int:
    tracks = sorted(AUDIO.glob("*.wav"))
    data = []
    for wav in tracks:
        cached = CACHE / f"{wav.stem}.npy"
        ann = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
        if cached.exists() and ann.exists():
            data.append((np.load(cached), read_ann(ann)))
    if len(data) < 6:
        print(f"Need the activation cache first: python sweep_thresholds.py")
        return 1
    print(f"{len(data)} tracks, 2-fold cross-validation\n")

    folds = [(list(range(0, len(data), 2)), list(range(1, len(data), 2)))]
    folds.append((folds[0][1], folds[0][0]))

    results = {m: {l: {"tp": 0, "ref": 0, "est": 0} for l in LABELS}
               for m in ("default", "global", "adaptive", "hybrid")}
    chosen = {"global": {l: [] for l in LABELS}, "adaptive": {l: [] for l in LABELS}}

    for train_idx, test_idx in folds:
        train = [data[i] for i in train_idx]
        test = [data[i] for i in test_idx]
        for ci, label in enumerate(LABELS):
            best_g = max(GLOBAL_GRID[label],
                         key=lambda t: f1(evaluate(train, ci, label, "global", t)))
            best_a = max(PCT_GRID,
                         key=lambda q: f1(evaluate(train, ci, label, "adaptive", q)))
            chosen["global"][label].append(best_g)
            chosen["adaptive"][label].append(best_a)
            # hybrid: adapt per track only for toms, where a fixed value cannot cope
            # with how much their density varies; keep stock elsewhere, except snare
            # where the tuned value is consistent across folds.
            hyb = ("adaptive", best_a) if label == "TT" else \
                  ("global", best_g if label == "SD" else STOCK[ci])
            for mode, param in (("default", STOCK[ci]), ("global", best_g),
                                ("adaptive", best_a), ("hybrid", hyb[1])):
                real_mode = hyb[0] if mode == "hybrid" else mode
                acc = evaluate(test, ci, label, real_mode, param)
                for k in results[mode][label]:
                    results[mode][label][k] += acc[k]

    modes = ("default", "global", "adaptive", "hybrid")
    print(f"{'drum':<10}" + "".join(f"{m:>12}" for m in modes))
    print("-" * 58)
    micro = {m: {"tp": 0, "ref": 0, "est": 0} for m in results}
    for label in LABELS:
        row = []
        for m in modes:
            c = results[m][label]
            for k in micro[m]:
                micro[m][k] += c[k]
            row.append(f"{f1(c):>12.3f}")
        print(f"{PRETTY[label]:<10}" + "".join(row))
    print("-" * 58)
    print(f"{'MICRO':<10}" + "".join(f"{f1(micro[m]):>12.3f}" for m in modes))

    print("\nheld-out results only - each fold is scored with thresholds fitted on the other fold")
    print("\nvalues picked per fold:")
    for label in LABELS:
        print(f"  {PRETTY[label]:<9} global {chosen['global'][label]}   "
              f"adaptive pct {chosen['adaptive'][label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
