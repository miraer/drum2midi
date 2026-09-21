"""Why do toms fail on soft beaters -- the threshold, or the model underneath it?

On ENST our toms lose to ReStem by 0.128, and the worst cases are brushes and mallets:
one brush solo carries a hundred annotated tom onsets and scores 0.056 against their
0.779. Two explanations were on the table and they call for opposite remedies.

  the threshold   the adaptive policy picks a percentile of each file's own tom
                  activations, and on soft beaters that distribution is not the one it
                  was designed against, so the threshold lands in the wrong place
  the model       ADTOF's tom activations are weak on brushes and mallets whatever
                  threshold is applied, which is a training-data statement

The oracle separates them. Sweeping the tom threshold over the whole usable range and
keeping the best F1 each file can reach gives the ceiling of any threshold policy,
including ones nobody has written. If the oracle is close to what we already score, no
policy can help and the limit is the model. If the oracle is far above, the policy is
the problem and is worth rewriting.

A control matters as much as the measurement: brush recordings are quieter, so *every*
activation could be depressed rather than the tom one specifically. The snare column
answers that -- brushes are played on the snare constantly, so if snare survives while
toms collapse, the effect is not simply level.

    python tom_failure_probe.py
    python tom_failure_probe.py --limit 8 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_enst import IGNORED, LABEL_TO_CLASS  # noqa: E402

ENST = ROOT / "enst" / "enst_drums_public"
WINDOW = 0.05
FPS = 100
TOM_COL = 2
SNARE_COL = 1
SWEEP = np.round(np.arange(0.02, 0.81, 0.01), 3)


def annotated(name: str) -> tuple[dict[str, np.ndarray], int] | tuple[None, None]:
    for d in (1, 2, 3):
        path = ENST / f"drummer_{d}" / "annotation" / f"{name}.txt"
        if not path.exists():
            continue
        out: dict[str, list[float]] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in IGNORED:
                cls = LABEL_TO_CLASS.get(parts[1])
                if cls:
                    out.setdefault(cls, []).append(float(parts[0]))
        return {k: np.array(sorted(v)) for k, v in out.items()}, d
    return None, None


def f1_at(ref: np.ndarray, est: np.ndarray) -> float:
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


def beater(name: str) -> str:
    for b in ("brushes", "mallets", "rods", "sticks"):
        if b in name:
            return b
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10,
                    help="recordings per beater (default: 10)")
    ap.add_argument("--min-toms", type=int, default=10,
                    help="skip recordings with fewer annotated toms (default: 10)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import drum2midi
    from adtof_pytorch import LABELS_5, PeakPicker, load_audio_for_model
    import torch

    tom_key = LABELS_5[TOM_COL]

    picked: dict[str, list[tuple[str, Path]]] = {}
    for d in (1, 2, 3):
        adir = ENST / f"drummer_{d}" / "annotation"
        if not adir.is_dir():
            continue
        for ann in sorted(adir.glob("*.txt")):
            wav = ROOT / "restem_in" / f"{ann.stem}.wav"
            if not wav.exists():
                continue
            picked.setdefault(beater(ann.stem), []).append((ann.stem, wav))

    model = drum2midi._adtof_model(args.device)
    rows = []
    for b in ("sticks", "rods", "brushes", "mallets"):
        taken = 0
        for name, wav in picked.get(b, []):
            if taken >= args.limit:
                break
            ref, _ = annotated(name)
            if ref is None or len(ref.get("TT", [])) < args.min_toms:
                continue
            x = load_audio_for_model(str(wav)).to(args.device)
            with torch.no_grad():
                act = model(x).cpu().numpy()[0]
            taken += 1

            toms, snare = ref["TT"], ref.get("SD", np.array([]))
            # activation the model actually produces where a tom is annotated
            idx = np.clip((toms * FPS).astype(int), 0, len(act) - 1)
            at_tom = float(np.median(act[idx, TOM_COL]))
            sidx = np.clip((snare * FPS).astype(int), 0, len(act) - 1)
            at_snare = float(np.median(act[sidx, SNARE_COL])) if len(snare) else float("nan")

            best, best_thr = 0.0, None
            for thr in SWEEP:
                t = list(drum2midi.DEFAULT_THRESHOLDS)
                t[TOM_COL] = float(thr)
                est = PeakPicker(thresholds=t, fps=FPS).pick(
                    act[None, ...], labels=LABELS_5, label_offset=0)[0]
                score = f1_at(toms, np.array(sorted(est.get(tom_key, []))))
                if score > best:
                    best, best_thr = score, float(thr)

            shipped = list(drum2midi.DEFAULT_THRESHOLDS)
            pol = min(max(float(np.percentile(act[:, TOM_COL], drum2midi.TOM_PERCENTILE)),
                          drum2midi.TOM_FLOOR), drum2midi.TOM_CEILING)
            shipped[TOM_COL] = pol
            est = PeakPicker(thresholds=shipped, fps=FPS).pick(
                act[None, ...], labels=LABELS_5, label_offset=0)[0]
            ours = f1_at(toms, np.array(sorted(est.get(tom_key, []))))

            rows.append((b, name, len(toms), at_tom, at_snare, ours, best, best_thr, pol))
            print(f"  {b:<8} {name[:36]:<37} toms {len(toms):>4}  "
                  f"act {at_tom:.3f}  ours {ours:.3f}  oracle {best:.3f} @ {best_thr}")

    if not rows:
        print("no recordings matched", file=sys.stderr)
        return 1

    print(f"\n{'beater':<10}{'recs':>5}{'toms':>7}{'act@tom':>9}{'act@snare':>11}"
          f"{'ours':>8}{'oracle':>8}{'headroom':>10}{'below floor':>13}")
    print("-" * 83)
    for b in ("sticks", "rods", "brushes", "mallets"):
        sel = [r for r in rows if r[0] == b]
        if not sel:
            continue
        n = len(sel)
        under = sum(1 for r in sel if r[7] is not None and r[7] < drum2midi.TOM_FLOOR)
        have = sum(1 for r in sel if r[7] is not None)
        print(f"{b:<10}{n:>5}{sum(r[2] for r in sel):>7}"
              f"{np.median([r[3] for r in sel]):>9.3f}"
              f"{np.nanmedian([r[4] for r in sel]):>11.3f}"
              f"{np.mean([r[5] for r in sel]):>8.3f}"
              f"{np.mean([r[6] for r in sel]):>8.3f}"
              f"{np.mean([r[6] - r[5] for r in sel]):>+10.3f}"
              f"{under:>8}/{have:<4}")

    # The headline the table above is really making. TOM_FLOOR exists to stop the
    # adaptive threshold drifting too low on tom-sparse material; if the best threshold
    # a recording could have is below that floor, no percentile policy bounded by it can
    # ever reach the optimum, whatever the activations look like.
    have = [r for r in rows if r[7] is not None]
    under = [r for r in have if r[7] < drum2midi.TOM_FLOOR]
    print()
    print(f"oracle threshold below TOM_FLOOR ({drum2midi.TOM_FLOOR}): "
          f"{len(under)} of {len(have)} recordings")
    if have:
        print(f"median oracle threshold: {np.median([r[7] for r in have]):.3f}, "
              f"against a policy bounded to "
              f"[{drum2midi.TOM_FLOOR}, {drum2midi.TOM_CEILING}]")
    if len(under) > 0.6 * len(have):
        print("The floor, not the activations, is what the policy cannot get past here.")
        print("Soft beaters depress every activation -- see act@snare -- but the snare")
        print(f"threshold is {drum2midi.DEFAULT_THRESHOLDS[SNARE_COL]} and can follow them "
              f"down, while toms cannot go below {drum2midi.TOM_FLOOR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
