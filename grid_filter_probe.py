"""Can the rhythmic grid tell a real onset from a false one at a low threshold?

The idea under test, and it is a good one: drop the peak-picking threshold so weak hits
survive, then keep only the candidates that land on the beat grid, on the grounds that a
real drum hit is played in time and a spurious activation is not.

There is one obvious way for it to fail, and it has to be measured rather than argued
about. **False positives in drum transcription are usually other drums.** A hi-hat
leaking into the tom channel is played exactly as much in time as the tom is, so a grid
test would keep it. If that is what our false positives are, the grid cannot separate
them from anything and the idea is dead however elegant it looks.

This measures the ceiling before anyone builds it. MDB ships hand-annotated beats, so the
grid here is perfect -- better than any estimator could produce from audio. If the
separation is not visible with a perfect grid it will not appear with an estimated one.

Reported per class: how far candidates sit from the nearest grid position, true and false
separately, and what the best achievable F1 is when a grid tolerance is swept alongside a
lowered threshold.

    python grid_filter_probe.py
    python grid_filter_probe.py --class HH --low 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_mdb import ANN, CLASSES, WINDOW, read_annotation  # noqa: E402

BEATS = ROOT / "mdbdrums" / "MDB Drums" / "annotations" / "beats"
AUDIO = ROOT / "mdbdrums" / "MDB Drums" / "audio" / "drum_only"
FPS = 100
COL = {"KD": 0, "SD": 1, "TT": 2, "HH": 3, "CY": 4}


def grid_for(track: str, per_beat: int) -> np.ndarray | None:
    """Beat times subdivided, so a hit on an off-beat sixteenth still counts as in time."""
    path = BEATS / f"{track}_MIX.beats"
    if not path.exists():
        return None
    beats = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts:
            beats.append(float(parts[0]))
    beats = np.array(sorted(beats))
    if len(beats) < 2:
        return None
    out = []
    for a, b in zip(beats[:-1], beats[1:]):
        for k in range(per_beat):
            out.append(a + (b - a) * k / per_beat)
    out.append(beats[-1])
    return np.array(out)


def phase_error(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Distance to the nearest grid point, as a fraction of the grid spacing.

    Normalised because a 40 ms error means something different at 80 bpm and at 200.
    0 is exactly on the grid, 0.5 is as far off as it is possible to be.
    """
    if not len(times) or len(grid) < 2:
        return np.array([])
    spacing = float(np.median(np.diff(grid)))
    idx = np.searchsorted(grid, times)
    idx = np.clip(idx, 1, len(grid) - 1)
    left, right = grid[idx - 1], grid[idx]
    nearest = np.where(times - left < right - times, left, right)
    return np.abs(times - nearest) / spacing


def split_true_false(cand: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(cand):
        return cand, cand
    if not len(ref):
        return np.array([]), cand
    d = np.abs(cand[:, None] - ref[None, :]).min(axis=1)
    return cand[d <= WINDOW], cand[d > WINDOW]


def split_false(bad: np.ndarray, all_onsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Was a false candidate a real hit wearing the wrong label, or nothing at all?

    This decides whether the grid idea can apply before any grid is drawn. A candidate
    that coincides with *some* annotated onset is a misclassification: a drum was struck,
    the model heard it, and put it in the wrong class. Those are played in time and a
    rhythmic test keeps them, so if the errors are mostly of this kind the problem is
    classification and the grid cannot touch it.

    A candidate with no annotated onset anywhere near it is a phantom -- the model
    responded to decay, bleed or noise. Those have no reason to be rhythmic, and they are
    the only errors a grid filter could remove.
    """
    if not len(bad):
        return bad, bad
    if not len(all_onsets):
        return np.array([]), bad
    d = np.abs(bad[:, None] - all_onsets[None, :]).min(axis=1)
    return bad[d <= WINDOW], bad[d > WINDOW]


def f1(ref: np.ndarray, est: np.ndarray) -> float:
    if not len(ref) or not len(est):
        return 0.0
    used = np.zeros(len(est), dtype=bool)
    tp = 0
    for t in ref:
        near = np.where((~used) & (np.abs(est - t) <= WINDOW))[0]
        if len(near):
            used[near[np.argmin(np.abs(est[near] - t))]] = True
            tp += 1
    p, r = tp / len(est), tp / len(ref)
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=float, default=0.05,
                    help="the lowered threshold candidates are taken at (default: 0.05)")
    ap.add_argument("--per-beat", type=int, default=4,
                    help="grid subdivisions per beat (default: 4, sixteenths)")
    ap.add_argument("--limit", type=int, default=23)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import drum2midi
    from adtof_pytorch import LABELS_5, PeakPicker, load_audio_for_model
    import torch

    model = drum2midi._adtof_model(args.device)
    tracks = sorted(AUDIO.glob("*.wav"))[: args.limit]

    store: dict[str, list] = {c: [] for c in COL}
    for wav in tracks:
        base = wav.stem.replace("_Drum", "")
        ann = ANN / f"{base}_class.txt"
        grid = grid_for(base, args.per_beat)
        if not ann.exists() or grid is None:
            continue
        ref_all = read_annotation(ann)
        x = load_audio_for_model(str(wav)).to(args.device)
        with torch.no_grad():
            act = model(x).cpu().numpy()

        for cls, col in COL.items():
            ref = ref_all[cls]
            if len(ref) < 10:
                continue
            thr = list(drum2midi.DEFAULT_THRESHOLDS)
            thr[col] = args.low
            cand = np.array(sorted(PeakPicker(thresholds=thr, fps=FPS).pick(
                act, labels=LABELS_5, label_offset=0)[0].get(LABELS_5[col], [])))
            good, bad = split_true_false(cand, ref)
            everything = np.array(sorted(np.concatenate(
                [v for v in ref_all.values() if len(v)]))) if any(
                len(v) for v in ref_all.values()) else np.array([])
            wrong_label, phantom = split_false(bad, everything)
            store[cls].append((ref, cand, good, bad, grid, wrong_label, phantom))

    print(f"threshold lowered to {args.low}, grid = {args.per_beat} per beat, "
          f"MDB, {len(tracks)} tracks\n")
    print("First: what ARE the false candidates? A hit in the wrong class, or nothing?\n")
    print(f"{'class':<9}{'true':>8}{'false':>8}{'wrong label':>14}{'phantom':>10}"
          f"{'phantom share':>15}")
    print("-" * 66)
    phantom_share = {}
    for cls in ("KD", "SD", "HH", "TT", "CY"):
        rows = store[cls]
        if not rows:
            continue
        ng = sum(len(r[2]) for r in rows)
        nb = sum(len(r[3]) for r in rows)
        nw = sum(len(r[5]) for r in rows)
        np_ = sum(len(r[6]) for r in rows)
        if not nb:
            continue
        phantom_share[cls] = np_ / nb
        print(f"{cls:<9}{ng:>8}{nb:>8}{nw:>14}{np_:>10}{np_ / nb:>14.0%}")

    print("\nA 'wrong label' candidate coincides with a real onset of some other class:")
    print("the drum was struck and put in the wrong box. Those are played in time, so a")
    print("rhythmic test keeps them and the grid cannot help. A 'phantom' has no")
    print("annotated onset anywhere near it, and is the only kind a grid could remove.")

    print(f"\n\nSecond: where do they sit relative to the grid?\n")
    print(f"{'class':<9}{'true':>16}{'wrong label':>16}{'phantom':>14}"
          f"{'phantom - true':>17}")
    print("-" * 72)
    verdict = []
    for cls in ("KD", "SD", "HH", "TT", "CY"):
        rows = store[cls]
        if not rows:
            continue

        def pool(i):
            parts = [phase_error(r[i], r[4]) for r in rows if len(r[i])]
            return np.concatenate(parts) if parts else np.array([])

        pg, pw, pp = pool(2), pool(5), pool(6)
        if not len(pg) or not len(pp):
            continue
        sep = float(np.median(pp) - np.median(pg))
        verdict.append((cls, sep, phantom_share.get(cls, 0.0)))
        w = f"{np.median(pw):>16.3f}" if len(pw) else f"{'--':>16}"
        print(f"{cls:<9}{np.median(pg):>16.3f}{w}{np.median(pp):>14.3f}{sep:>+17.3f}")

    print("\nPhase error is the distance to the nearest grid position as a fraction of "
          "the\ngrid spacing: 0 is exactly in time, 0.5 is the furthest a hit can be.")

    if verdict:
        best = max(s for _, s, _ in verdict)
        mean_phantom = float(np.mean([p for _, _, p in verdict]))
        print()
        if mean_phantom < 0.4:
            print(f"Most false candidates are real hits in the wrong class "
                  f"({1 - mean_phantom:.0%} on average).")
            print("So the dominant problem at a low threshold is classification, not")
            print("detection, and a rhythmic filter cannot address it: a misclassified")
            print("hit is exactly as much in time as a correct one.")
        elif best < 0.02:
            print(f"Phantoms are {mean_phantom:.0%} of the false candidates, so there is")
            print("something for a grid to remove -- but they sit as close to the grid as")
            print("real hits do, so it cannot tell them apart. Worth knowing why before")
            print("building anything: a phantom next to a real hit inherits its timing.")
        else:
            print(f"Phantoms are {mean_phantom:.0%} of the false candidates and sit "
                  f"measurably\nfurther off the grid. The idea has room. It needs "
                  f"testing against the\nshipped policy on held-out material, with the "
                  f"grid estimated from audio\nrather than annotated, since a real "
                  f"system has no annotated beats.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
