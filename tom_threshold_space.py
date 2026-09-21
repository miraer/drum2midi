"""The adaptive tom threshold is computed in one space and applied in another.

`NotePeakPickingProcessor` subtracts a 100 ms trailing mean before thresholding:

    proc = max(0, act - moving_average(act))
    is_peak = (proc >= local_max) & (proc >= threshold)

So the threshold it enforces lives in `proc`. The four fixed thresholds were tuned
there. The tom policy is the one class that computes its threshold instead, and it takes
the percentile of the **raw** activation:

    thr[TOM] = clamp(percentile(activations[:, TOM], 98.5), 0.25, 0.45)

Raw activation reaches 0.9 on a struck tom; `proc` on the same frame is much smaller,
because a tom in a fill is preceded by 100 ms of tom. So a number drawn from one
distribution is used as a cut on the other, and it is far too high by construction rather
than by tuning. On 239 missed ENST tom onsets the raw activation clears the threshold 49%
of the time and `proc` clears it **0%** of the time.

The fix has no free parameter: take the percentile of the same signal the threshold is
compared against. That is what makes it testable on both corpora at once without fitting
anything, which no other remedy tried today could claim.

    python tom_threshold_space.py
    python tom_threshold_space.py --corpus enst --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_mdb import ANN as MDB_ANN, WINDOW, read_annotation  # noqa: E402
from benchmark_enst import IGNORED, LABEL_TO_CLASS  # noqa: E402

MDB_AUDIO = ROOT / "mdbdrums" / "MDB Drums" / "audio" / "drum_only"
ENST = ROOT / "enst" / "enst_drums_public"
FPS = 100
TOM = 2


def f1(ref: np.ndarray, est: np.ndarray) -> tuple[float, int, int]:
    if not len(ref) or not len(est):
        return 0.0, 0, len(est)
    used = np.zeros(len(est), dtype=bool)
    tp = 0
    for t in ref:
        near = np.where((~used) & (np.abs(est - t) <= WINDOW))[0]
        if len(near):
            used[near[np.argmin(np.abs(est[near] - t))]] = True
            tp += 1
    p, r = tp / len(est), tp / len(ref)
    return (2 * p * r / (p + r) if p + r else 0.0), tp, len(est)


def enst_toms(name: str) -> np.ndarray | None:
    for d in (1, 2, 3):
        p = ENST / f"drummer_{d}" / "annotation" / f"{name}.txt"
        if p.exists():
            return np.array(sorted(
                float(l.split()[0]) for l in p.read_text(encoding="utf-8",
                                                         errors="replace").splitlines()
                if len(l.split()) >= 2 and l.split()[1] not in IGNORED
                and LABEL_TO_CLASS.get(l.split()[1]) == "TT"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=("mdb", "enst", "both"), default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import drum2midi
    from adtof_pytorch import LABELS_5, PeakPicker, load_audio_for_model
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    import torch

    model = drum2midi._adtof_model(args.device)
    # the processor's own averaging, so this cannot drift from what the picker does
    probe = NotePeakPickingProcessor(threshold=0.0, fps=FPS)

    def proc_of(col: np.ndarray) -> np.ndarray:
        pre = int(round(probe.pre_avg * FPS))
        post = int(round(probe.post_avg * FPS))
        return np.maximum(0.0, col - probe._moving_average(col, pre, post))

    jobs = []
    if args.corpus in ("mdb", "both"):
        for wav in sorted(MDB_AUDIO.glob("*.wav"))[: args.limit]:
            ann = MDB_ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
            if ann.exists():
                jobs.append(("MDB", wav, read_annotation(ann)["TT"]))
    if args.corpus in ("enst", "both"):
        names = [n.strip() for n in
                 (ROOT / "bench" / "enst60.txt").read_text().splitlines() if n.strip()]
        for name in names[: args.limit]:
            wav = ROOT / "restem_in" / f"{name}.wav"
            ref = enst_toms(name)
            if wav.exists() and ref is not None and len(ref):
                jobs.append(("ENST", wav, ref))

    totals: dict[tuple[str, str], list[int]] = {}
    thr_seen: dict[tuple[str, str], list[float]] = {}
    for corpus, wav, ref in jobs:
        x = load_audio_for_model(str(wav)).to(args.device)
        with torch.no_grad():
            act = model(x).cpu().numpy()[0]
        col = act[:, TOM]
        pr = proc_of(col)

        variants = {
            "shipped (percentile of raw)": float(np.percentile(col, drum2midi.TOM_PERCENTILE)),
            "percentile of proc": float(np.percentile(pr, drum2midi.TOM_PERCENTILE)),
        }
        for name, raw_thr in variants.items():
            thr = min(max(raw_thr, drum2midi.TOM_FLOOR), drum2midi.TOM_CEILING)
            t = list(drum2midi.DEFAULT_THRESHOLDS)
            t[TOM] = thr
            est = np.array(sorted(PeakPicker(thresholds=t, fps=FPS).pick(
                act[None, ...], labels=LABELS_5, label_offset=0)[0].get(LABELS_5[TOM], [])))
            _, tp, n_est = f1(ref, est)
            key = (corpus, name)
            totals.setdefault(key, [0, 0, 0])
            totals[key][0] += tp
            totals[key][1] += len(ref)
            totals[key][2] += n_est
            thr_seen.setdefault(key, []).append(thr)
        # unclamped, to show what the clamp is doing in the corrected space
        thr = float(np.percentile(pr, drum2midi.TOM_PERCENTILE))
        t = list(drum2midi.DEFAULT_THRESHOLDS)
        t[TOM] = thr
        est = np.array(sorted(PeakPicker(thresholds=t, fps=FPS).pick(
            act[None, ...], labels=LABELS_5, label_offset=0)[0].get(LABELS_5[TOM], [])))
        _, tp, n_est = f1(ref, est)
        key = (corpus, "percentile of proc, no clamp")
        totals.setdefault(key, [0, 0, 0])
        totals[key][0] += tp
        totals[key][1] += len(ref)
        totals[key][2] += n_est
        thr_seen.setdefault(key, []).append(thr)

    order = ["shipped (percentile of raw)", "percentile of proc",
             "percentile of proc, no clamp"]
    for corpus in ("MDB", "ENST"):
        rows = [(n, totals[(corpus, n)]) for n in order if (corpus, n) in totals]
        if not rows:
            continue
        print(f"\n{corpus}: {rows[0][1][1]} annotated toms")
        print(f"{'policy':<32}{'median thr':>12}{'emitted':>9}{'matched':>9}"
              f"{'P':>8}{'R':>8}{'F1':>8}")
        print("-" * 86)
        for name, (tp, ref_n, est_n) in rows:
            p = tp / max(est_n, 1)
            r = tp / max(ref_n, 1)
            print(f"{name:<32}{np.median(thr_seen[(corpus, name)]):>12.3f}{est_n:>9}"
                  f"{tp:>9}{p:>8.3f}{r:>8.3f}"
                  f"{(2 * p * r / (p + r) if p + r else 0):>8.3f}")

    print("\nNo threshold was fitted. The corrected policy uses the same percentile, the")
    print("same clamp and the same picker -- only the signal the percentile is taken from")
    print("changes, from the raw activation to the one the threshold is compared against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
