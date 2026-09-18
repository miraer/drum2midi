"""The peak picker's combine window - are flams and double hits being lost.

The picker merges peaks closer than 20 ms. That guards against double triggers, but it
also merges flams. Measured on MDB: what happens to F1 as the window narrows.

    python sweep_combine.py
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
STOCK = [0.22, 0.14, 0.32, 0.22, 0.30]
TOM_PCT, TOM_FLOOR = 98.5, 0.25


def pick(col, thr, combine):
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=combine, fps=100)
    return np.array([t for t, _ in p.process(col)])


def read_ann(path: Path):
    out = {k: [] for k in LABELS}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] in out:
            out[p[1]].append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def main() -> int:
    data = []
    for wav in sorted(AUDIO.glob("*.wav")):
        c = CACHE / f"{wav.stem}.npy"
        a = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
        if c.exists() and a.exists():
            data.append((np.load(c), read_ann(a)))
    print(f"tracks: {len(data)}\n")

    print(f"{'window':>7}" + "".join(f"{l:>9}" for l in LABELS)
          + f"{'MICRO':>9}{'notes':>8}")
    print("-" * (7 + 9 * 6 + 8))
    for combine in (0.020, 0.015, 0.012, 0.010, 0.007):
        micro = {"tp": 0, "ref": 0, "est": 0}
        cells = []
        for ci, label in enumerate(LABELS):
            acc = {"tp": 0, "ref": 0, "est": 0}
            for act, ref in data:
                thr = STOCK[ci]
                if label == "TT":
                    thr = max(float(np.percentile(act[:, ci], TOM_PCT)), TOM_FLOOR)
                est = pick(act[:, ci], thr, combine)
                r = ref[label]
                acc["ref"] += len(r); acc["est"] += len(est)
                if len(r) and len(est):
                    _, p, _ = mir_eval.onset.f_measure(r, est, window=WINDOW)
                    acc["tp"] += int(round(p * len(est)))
            for k in micro:
                micro[k] += acc[k]
            p = acc["tp"] / max(acc["est"], 1); rr = acc["tp"] / max(acc["ref"], 1)
            cells.append(f"{2*p*rr/max(p+rr,1e-9):>9.3f}")
        p = micro["tp"] / max(micro["est"], 1); rr = micro["tp"] / max(micro["ref"], 1)
        mark = "  <- current" if abs(combine - 0.02) < 1e-9 else ""
        print(f"{combine*1000:>5.0f}ms" + "".join(cells)
              + f"{2*p*rr/max(p+rr,1e-9):>9.3f}{micro['est']:>8}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
