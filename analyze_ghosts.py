"""Where exactly are notes lost - a diagnosis by hit strength.

58% of snare hits in GMD are quieter than velocity 50. If our recall falls off a cliff
on quiet hits, then ghost notes are the pipeline's main weakness, and this shows whether
a threshold can fix it and at what cost in precision.

ADTOF activations are cached, so the threshold sweep is almost free.

    python analyze_ghosts.py --limit 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GMD = ROOT / "gmd" / "groove"
CACHE = ROOT / "bench" / "gmd_activations"
WINDOW = 0.05

# ADTOF class index -> the Roland notes it is supposed to catch
CLASSES = [
    (0, "kick", (35, 36)),
    (1, "snare", (38, 40, 37)),
    (2, "toms", (48, 50, 45, 47, 43, 58)),
    (3, "hihat", (42, 22, 46, 26, 44)),
    (4, "cymbals", (49, 55, 57, 52, 51, 59, 53)),
]
STOCK = [0.22, 0.14, 0.32, 0.22, 0.30]
BINS = [(1, 20), (20, 35), (35, 50), (50, 70), (70, 95), (95, 128)]


def activations_for(model, wav: Path, key: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{key}.npy"
    if cached.exists():
        return np.load(cached)
    from adtof_pytorch import load_audio_for_model
    import torch
    with torch.no_grad():
        act = model(load_audio_for_model(str(wav))).cpu().numpy()[0]
    np.save(cached, act)
    return act


def pick(col: np.ndarray, thr: float) -> np.ndarray:
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def match(ref_times: np.ndarray, est: np.ndarray) -> np.ndarray:
    """For each reference note: was it found (greedy nearest match)."""
    found = np.zeros(len(ref_times), dtype=bool)
    used = set()
    for i, t in enumerate(ref_times):
        best, dist = None, WINDOW
        for j, e in enumerate(est):
            if j in used:
                continue
            d = abs(e - t)
            if d <= dist:
                best, dist = j, d
        if best is not None:
            used.add(best)
            found[i] = True
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from drum2midi import _adtof_model
    model = _adtof_model("cpu")

    rows = [r for r in csv.DictReader(open(GMD / "info.csv", encoding="utf-8"))
            if r.get("audio_filename") and (GMD / r["audio_filename"]).exists()
            and 5 <= float(r["duration"]) <= 60]
    rows.sort(key=lambda r: (r["split"] != "test", r["id"]))
    rows = rows[: args.limit]

    data = []
    for i, r in enumerate(rows, 1):
        key = r["id"].replace("/", "_")
        act = activations_for(model, GMD / r["audio_filename"], key)
        pm = pretty_midi.PrettyMIDI(str(GMD / r["midi_filename"]))
        notes = [(float(n.start), n.pitch, int(n.velocity))
                 for inst in pm.instruments for n in inst.notes]
        data.append((act, notes))
        print(f"  [{i:2d}/{len(rows)}] {r['id']}")

    print(f"\n{'='*72}\nRECALL BY HIT STRENGTH (default thresholds)\n{'='*72}")
    print(f"{'instrument':<10}" + "".join(f"{f'{lo}-{hi-1}':>11}" for lo, hi in BINS))
    print("-" * (10 + 11 * len(BINS)))

    stats = {}
    for ci, name, pitches in CLASSES:
        per_bin = [[0, 0] for _ in BINS]   # [found, total]
        for act, notes in data:
            ref = sorted((t, v) for t, p, v in notes if p in pitches)
            if not ref:
                continue
            times = np.array([t for t, _ in ref])
            vels = np.array([v for _, v in ref])
            found = match(times, pick(act[:, ci], STOCK[ci]))
            for bi, (lo, hi) in enumerate(BINS):
                m = (vels >= lo) & (vels < hi)
                per_bin[bi][0] += int(found[m].sum())
                per_bin[bi][1] += int(m.sum())
        stats[name] = per_bin
        cells = []
        for got, tot in per_bin:
            cells.append(f"{got/tot:>10.2f} " if tot >= 20 else f"{'—':>11}")
        print(f"{name:<10}" + "".join(cells))

    print("\nin brackets: how many reference hits fell into each bin:")
    for name, per_bin in stats.items():
        print(f"  {name:<9}" + "".join(f"{tot:>8}" for _, tot in per_bin))

    print(f"\n{'='*72}\nCAN A THRESHOLD FIX IT (snare)\n{'='*72}")
    ci, pitches = 1, (38, 40, 37)
    print(f"{'threshold':>7}{'recall quiet':>14}{'recall loud':>16}"
          f"{'precision':>12}{'F1':>8}")
    print("-" * 57)
    for thr in (0.14, 0.10, 0.07, 0.05, 0.03):
        quiet = [0, 0]; loud = [0, 0]; tp = est_n = ref_n = 0
        for act, notes in data:
            ref = sorted((t, v) for t, p, v in notes if p in pitches)
            if not ref:
                continue
            times = np.array([t for t, _ in ref])
            vels = np.array([v for _, v in ref])
            est = pick(act[:, ci], thr)
            found = match(times, est)
            q = vels < 50
            quiet[0] += int(found[q].sum()); quiet[1] += int(q.sum())
            loud[0] += int(found[~q].sum()); loud[1] += int((~q).sum())
            tp += int(found.sum()); est_n += len(est); ref_n += len(times)
        p = tp / max(est_n, 1); r = tp / max(ref_n, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        mark = "  <- current" if abs(thr - 0.14) < 1e-9 else ""
        print(f"{thr:>7.2f}{quiet[0]/max(quiet[1],1):>14.3f}"
              f"{loud[0]/max(loud[1],1):>16.3f}{p:>12.3f}{f:>8.3f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
