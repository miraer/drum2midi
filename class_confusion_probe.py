"""When a hit lands in the wrong class, is the right class the runner-up or silent?

92% of our false positives at a low threshold are real drum hits wearing the wrong
label. No threshold policy touches that, so the question is what would.

There are two kinds of confusion and they call for completely different work:

  recoverable   the correct class is also responding at that instant, just lower than
                the one that won. A decision that compared the five columns at each
                onset instead of thresholding them independently would get it right,
                and that is a small change to the picker.

  deaf          the correct class is not responding at all. Nothing in the output can
                recover it; the model does not hear that drum on that material, which
                is a training-data statement and should be published as one.

The split is what decides whether there is an engineering fix or a finding. This also
reports which drum becomes which, because a confusion concentrated in one pair is worth
different work from one spread across the kit.

    python class_confusion_probe.py
    python class_confusion_probe.py --margin 0.05
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_mdb import ANN, WINDOW, read_annotation  # noqa: E402

AUDIO = ROOT / "mdbdrums" / "MDB Drums" / "audio" / "drum_only"
FPS = 100
ORDER = ["KD", "SD", "TT", "HH", "CY"]
COL = {c: i for i, c in enumerate(ORDER)}
PRETTY = {"KD": "kick", "SD": "snare", "TT": "toms", "HH": "hi-hat", "CY": "cymbals"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=23)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="how much higher the correct class must be to count as "
                         "recoverable (default: 0, simple argmax)")
    args = ap.parse_args()

    import drum2midi
    from adtof_pytorch import LABELS_5, PeakPicker, load_audio_for_model
    import torch

    model = drum2midi._adtof_model(args.device)

    confusion: Counter = Counter()
    recoverable: Counter = Counter()
    deaf: Counter = Counter()
    act_when_wrong: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for wav in sorted(AUDIO.glob("*.wav"))[: args.limit]:
        base = wav.stem.replace("_Drum", "")
        ann = ANN / f"{base}_class.txt"
        if not ann.exists():
            continue
        ref = read_annotation(ann)
        x = load_audio_for_model(str(wav)).to(args.device)
        with torch.no_grad():
            act = model(x).cpu().numpy()[0]

        picks = PeakPicker(thresholds=list(drum2midi.DEFAULT_THRESHOLDS),
                           fps=FPS).pick(act[None, ...], labels=LABELS_5,
                                         label_offset=0)[0]
        for pred in ORDER:
            est = np.array(sorted(picks.get(LABELS_5[COL[pred]], [])))
            if not len(est):
                continue
            own = ref[pred]
            for t in est:
                if len(own) and np.abs(own - t).min() <= WINDOW:
                    continue                      # correct, not a confusion
                # which drum, if any, was actually struck here
                truth = None
                for cls in ORDER:
                    r = ref[cls]
                    if len(r) and np.abs(r - t).min() <= WINDOW:
                        truth = cls
                        break
                if truth is None:
                    continue                      # phantom, measured elsewhere
                confusion[(pred, truth)] += 1
                frame = int(np.clip(round(t * FPS), 0, len(act) - 1))
                a_pred = float(act[frame, COL[pred]])
                a_true = float(act[frame, COL[truth]])
                act_when_wrong.setdefault((pred, truth), []).append((a_pred, a_true))
                if a_true > a_pred + args.margin:
                    recoverable[(pred, truth)] += 1
                else:
                    deaf[(pred, truth)] += 1

    total = sum(confusion.values())
    if not total:
        print("no confusions found", file=sys.stderr)
        return 1

    print(f"{total} confusions at the shipped thresholds, MDB\n")
    print("which drum becomes which -- rows are what we emitted, columns what was played")
    print(f"\n{'emitted':<10}" + "".join(f"{PRETTY[c]:>10}" for c in ORDER) + f"{'total':>9}")
    print("-" * 70)
    for pred in ORDER:
        row = f"{PRETTY[pred]:<10}"
        tot = 0
        for truth in ORDER:
            n = confusion.get((pred, truth), 0)
            tot += n
            row += f"{n if n else '.':>10}"
        print(row + f"{tot:>9}")

    rec = sum(recoverable.values())
    print(f"\n\nis the correct class the runner-up, or silent?\n")
    print(f"{'confusion':<24}{'count':>7}{'correct louder':>16}{'share':>8}"
          f"{'act emitted':>13}{'act correct':>13}")
    print("-" * 82)
    for (pred, truth), n in confusion.most_common(8):
        r = recoverable.get((pred, truth), 0)
        pairs = act_when_wrong[(pred, truth)]
        print(f"{PRETTY[pred] + ' <- ' + PRETTY[truth]:<24}{n:>7}{r:>16}{r / n:>8.0%}"
              f"{np.median([p[0] for p in pairs]):>13.3f}"
              f"{np.median([p[1] for p in pairs]):>13.3f}")

    print("-" * 82)
    print(f"{'all':<24}{total:>7}{rec:>16}{rec / total:>8.0%}")

    print()
    if rec / total > 0.5:
        print(f"{rec / total:.0%} of confusions have the correct class responding more")
        print("strongly than the one that won. Those are reachable without retraining:")
        print("the picker thresholds each class independently, so a column can win on")
        print("its own threshold while a louder column sits beside it unconsulted.")
        print("Comparing the five at each onset is a change to the decision, not the")
        print("model -- and it is testable corpus to corpus like any other policy.")
    else:
        print(f"Only {rec / total:.0%} of confusions have the correct class louder, so")
        print("most of them cannot be recovered from this output at all: the model is")
        print("not responding to the drum that was played. That is a training-data")
        print("statement and no decision rule repairs it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
