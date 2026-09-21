"""The trailing window is a parameter too, and nobody swept it.

Every tom remedy tried so far moved the threshold. The measured mechanism is not the
threshold alone: at a rejected tom onset the 100 ms trailing mean eats **43%** of the
model's response against 27% at one we keep, because a tom in a fill is preceded by
100 ms of tom. The subtraction is what makes a dense passage invisible, and its window
is `pre_avg` in `NotePeakPickingProcessor` -- a constant, not a trained weight.

So before concluding that toms need a different model, sweep it. The rule is the same as
for every other policy here: choose on one corpus, report on the other, and publish the
held-out number next to the fitted one so the gap is visible.

The sweep moves the tom channel only. The other four classes keep the shipped window,
because their thresholds were tuned against it and changing the signal underneath them
would be measuring something else.

    python tom_window_sweep.py
    python tom_window_sweep.py --rounds 4000
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
WINDOWS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]


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


def counts(ref: np.ndarray, est: np.ndarray) -> tuple[int, int, int]:
    if not len(ref) or not len(est):
        return 0, len(ref), len(est)
    used = np.zeros(len(est), dtype=bool)
    tp = 0
    for t in ref:
        near = np.where((~used) & (np.abs(est - t) <= WINDOW))[0]
        if len(near):
            used[near[np.argmin(np.abs(est[near] - t))]] = True
            tp += 1
    return tp, len(ref), len(est)


def f1_from(rows: np.ndarray) -> float:
    tp, ref, est = rows[:, 0].sum(), rows[:, 1].sum(), rows[:, 2].sum()
    p, r = tp / max(est, 1), tp / max(ref, 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import drum2midi
    from adtof_pytorch import load_audio_for_model
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    import torch

    model = drum2midi._adtof_model(args.device)

    def pick_toms(act: np.ndarray, pre_avg: float) -> np.ndarray:
        """Exactly the shipped path for the tom channel, with one window changed."""
        proc = NotePeakPickingProcessor(threshold=0.0, fps=FPS, pre_avg=pre_avg)
        col = act[:, TOM]
        pa = int(round(pre_avg * FPS))
        po = int(round(proc.post_avg * FPS))
        sub = np.maximum(0.0, col - proc._moving_average(col, pa, po))
        wmax = proc._local_maxima(sub, int(round(proc.pre_max * FPS)),
                                  int(round(proc.post_max * FPS)))
        thr = min(max(float(np.percentile(col, drum2midi.TOM_PERCENTILE)),
                      drum2midi.TOM_FLOOR), drum2midi.TOM_CEILING)
        idx = np.where((sub >= wmax) & (sub >= thr))[0]
        if not len(idx):
            return np.array([])
        comb = max(1, int(round(proc.combine * FPS)))
        kept, group = [], [int(idx[0])]
        for i in idx[1:]:
            if i - group[-1] <= comb:
                group.append(int(i))
            else:
                kept.append(max(group, key=lambda j: sub[j]))
                group = [int(i)]
        kept.append(max(group, key=lambda j: sub[j]))
        return np.array([k / FPS for k in kept])

    jobs = []
    for wav in sorted(MDB_AUDIO.glob("*.wav")):
        ann = MDB_ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
        if ann.exists():
            jobs.append(("MDB", wav, read_annotation(ann)["TT"]))
    for name in [n.strip() for n in
                 (ROOT / "bench" / "enst60.txt").read_text().splitlines() if n.strip()]:
        wav = ROOT / "restem_in" / f"{name}.wav"
        ref = enst_toms(name)
        if wav.exists() and ref is not None:
            jobs.append(("ENST", wav, ref))

    per: dict[tuple[str, float], list] = {}
    for corpus, wav, ref in jobs:
        x = load_audio_for_model(str(wav)).to(args.device)
        with torch.no_grad():
            act = model(x).cpu().numpy()[0]
        for w in WINDOWS:
            per.setdefault((corpus, w), []).append(counts(ref, pick_toms(act, w)))

    tables = {c: {w: np.array(per[(c, w)], dtype=float) for w in WINDOWS}
              for c in ("MDB", "ENST")}

    print(f"tom F1 against the trailing-mean window, all else shipped\n")
    print(f"{'pre_avg':>9}{'MDB':>10}{'ENST':>10}")
    print("-" * 30)
    for w in WINDOWS:
        tag = "  <- shipped" if abs(w - 0.10) < 1e-9 else ""
        print(f"{w:>9.2f}{f1_from(tables['MDB'][w]):>10.3f}"
              f"{f1_from(tables['ENST'][w]):>10.3f}{tag}")

    best = {c: max(WINDOWS, key=lambda w: f1_from(tables[c][w])) for c in tables}
    print(f"\n{'':<8}{'shipped':>10}{'own best':>10}{'at own w':>10}"
          f"{'HELD OUT':>10}{'w from other':>14}")
    print("-" * 62)
    rng = np.random.default_rng(0)
    for c, other in (("MDB", "ENST"), ("ENST", "MDB")):
        w_other = best[other]
        base, held = tables[c][0.10], tables[c][w_other]
        diff = f1_from(held) - f1_from(base)
        idx = np.arange(len(base))
        draws = []
        for _ in range(args.rounds):
            s = rng.choice(idx, len(idx), replace=True)
            draws.append(f1_from(held[s]) - f1_from(base[s]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        star = " *" if lo * hi > 0 else "  "
        print(f"{c:<8}{f1_from(base):>10.3f}{f1_from(tables[c][best[c]]):>10.3f}"
              f"{best[c]:>10.2f}{f1_from(held):>10.3f}{w_other:>14.2f}"
              f"   {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")

    print("\n'HELD OUT' is this corpus scored at the window chosen on the other one, and")
    print("it is the only column that can carry a claim. The interval is a paired")
    print("bootstrap over recordings against the shipped window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
